from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Literal, NamedTuple
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import delete, exists, select
from sqlalchemy.dialects.postgresql import insert

from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.graph_semantics import check_relations
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentStatus,
    Entity,
    EntityMention,
    Relation,
)
from knowflow.app.providers import chat_client, structured_request_options, task_route


class EntityFact(BaseModel):
    name: str = Field(min_length=1, max_length=500)
    evidence: str = Field(min_length=1, max_length=1200)


class RelationQualifier(BaseModel):
    kind: Literal["environment", "version", "condition", "negation", "modality", "time"]
    text: str = Field(min_length=2, max_length=500)


class RelationFact(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    target: str = Field(min_length=1, max_length=500)
    predicate: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=1200)
    evidence: str = Field(min_length=1, max_length=1200)
    confidence: float = Field(default=1.0, ge=0, le=1)
    qualifiers: list[RelationQualifier] = Field(default_factory=list, max_length=12)


class ChunkGraphFacts(BaseModel):
    entities: list[EntityFact] = Field(default_factory=list, max_length=100)
    relations: list[RelationFact] = Field(default_factory=list, max_length=100)
    rejected_relations: int = 0


GRAPH_PROMPT_VERSION = "qualified-relations-v4-predicate-roles"
SCOPE_SIGNAL = re.compile(
    r"\b(?:may|might|only|unless|if|when|must\s+not|in\s+(?:production|staging)|version\s+\d+)\b"
    r"|仅|只有|不得|禁止|允许|可能|如果|测试环境|生产环境",
    re.I,
)


def relation_scope_key(qualifiers):
    if not qualifiers:
        return ""
    return hashlib.sha256(
        json.dumps(sorted((q.kind, q.text) for q in qualifiers), ensure_ascii=False).encode()
    ).hexdigest()


GRAPH_BATCH_CHAR_LIMIT = 4000
GRAPH_BATCH_CHUNK_LIMIT = 60


class GraphSourceChunk(NamedTuple):
    chunk_id: UUID
    content: str
    section: str | None = None
    page_number: int | None = None


def normalize_entity_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.strip(" \t\r\n.,;:!?，。；：！？()[]{}<>\"'`")


def _supported_quote(source: str, quote: str) -> bool:
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    return bool(quote.strip()) and normalize(quote) in normalize(source)


def validate_chunk_facts(facts: ChunkGraphFacts, content: str) -> ChunkGraphFacts:
    """Discard any extracted fact whose names and supporting quote are not in source text."""
    entities: dict[str, EntityFact] = {}
    for fact in facts.entities:
        normalized = normalize_entity_name(fact.name)
        supported = _supported_quote(content, fact.name) and _supported_quote(
            content, fact.evidence
        )
        if normalized and supported:
            entities.setdefault(normalized, fact)

    accepted: dict[tuple[str, str, str, str], RelationFact] = {}
    for relation in facts.relations:
        source = entities.get(normalize_entity_name(relation.source))
        target = entities.get(normalize_entity_name(relation.target))
        if (
            source is not None
            and target is not None
            and source.name != target.name
            and _supported_quote(content, relation.evidence)
            and all(_supported_quote(content, q.text) for q in relation.qualifiers)
            and (not SCOPE_SIGNAL.search(relation.evidence) or relation.qualifiers)
        ):
            key = (
                normalize_entity_name(relation.source),
                normalize_entity_name(relation.target),
                relation.predicate.strip().casefold(),
                relation_scope_key(relation.qualifiers),
            )
            accepted.setdefault(key, relation)
    return ChunkGraphFacts(
        entities=list(entities.values()),
        relations=list(accepted.values()),
        rejected_relations=facts.rejected_relations + len(facts.relations) - len(accepted),
    )


def parse_chunk_graph_response(raw: str) -> ChunkGraphFacts:
    """Parse a JSON object from a model response and report schema errors safely."""
    decoder = json.JSONDecoder()
    payload = None
    for offset, character in enumerate(raw):
        if character != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(raw[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        raise RuntimeError("Graph extraction response did not contain a JSON object")

    try:
        parsed = ChunkGraphFacts.model_validate(payload)
        parsed.rejected_relations = 0
        return parsed
    except ValidationError as exc:
        issues = []
        for error in exc.errors(include_input=False, include_url=False)[:5]:
            path = ".".join(str(part) for part in error["loc"]) or "response"
            issues.append(f"{path}: {error['msg']}")
        raise RuntimeError(
            "Graph extraction response did not match the required JSON fields: " + "; ".join(issues)
        ) from exc


def batch_chunks(
    chunks: Sequence[GraphSourceChunk],
    *,
    max_chars: int = GRAPH_BATCH_CHAR_LIMIT,
    max_chunks: int = GRAPH_BATCH_CHUNK_LIMIT,
) -> list[list[GraphSourceChunk]]:
    """Pack nearby chunks, keeping each heading and PDF page as a separate scope."""
    if max_chars < 1 or max_chunks < 1:
        raise ValueError("Graph extraction batch limits must be positive")

    batches: list[list[GraphSourceChunk]] = []
    current: list[GraphSourceChunk] = []
    current_chars = 0
    current_scope: tuple[str | None, int | None] | None = None
    for chunk in chunks:
        content_length = len(chunk.content)
        scope = (chunk.section, chunk.page_number)
        if current and (
            scope != current_scope
            or len(current) >= max_chunks
            or current_chars + content_length > max_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current_scope = scope
        current.append(chunk)
        current_chars += content_length
    if current:
        batches.append(current)
    return batches


def _batch_source_content(batch: Sequence[GraphSourceChunk]) -> str:
    return "\n\n".join(
        f"[source chunk {index}; section={chunk.section or 'none'}; "
        f"page={chunk.page_number or 'none'}]\n{chunk.content}"
        for index, chunk in enumerate(batch, start=1)
    )


def distribute_batch_facts(
    batches: Sequence[Sequence[GraphSourceChunk]], facts_by_batch: Sequence[ChunkGraphFacts]
) -> list[ChunkGraphFacts]:
    """Attach extracted facts back to chunks that contain their exact evidence."""
    if len(batches) != len(facts_by_batch):
        raise ValueError("Each graph extraction batch must have one result")

    chunk_contents = [chunk for batch in batches for chunk in batch]
    entities_by_chunk: dict[UUID, dict[str, EntityFact]] = {
        chunk.chunk_id: {} for chunk in chunk_contents
    }
    relations_by_chunk: dict[UUID, list[RelationFact]] = {
        chunk.chunk_id: [] for chunk in chunk_contents
    }

    for batch, facts in zip(batches, facts_by_batch, strict=True):
        for fact in facts.entities:
            for chunk in batch:
                if _supported_quote(chunk.content, fact.name):
                    evidence = (
                        fact.evidence
                        if _supported_quote(chunk.content, fact.evidence)
                        else fact.name
                    )
                    entities_by_chunk[chunk.chunk_id].setdefault(
                        normalize_entity_name(fact.name),
                        EntityFact(name=fact.name, evidence=evidence),
                    )

        for fact in facts.relations:
            for chunk in batch:
                if not all(
                    _supported_quote(chunk.content, value)
                    for value in (fact.source, fact.target, fact.evidence)
                ):
                    continue
                # A relation is retained only when both endpoints and its quote are
                # present in the same original passage. The names themselves are
                # exact, auditable evidence for these two endpoint mentions.
                for name in (fact.source, fact.target):
                    entities_by_chunk[chunk.chunk_id].setdefault(
                        normalize_entity_name(name), EntityFact(name=name, evidence=name)
                    )
                relations_by_chunk[chunk.chunk_id].append(fact)
                break

    return [
        validate_chunk_facts(
            ChunkGraphFacts(
                entities=list(entities_by_chunk[chunk.chunk_id].values()),
                relations=relations_by_chunk[chunk.chunk_id],
            ),
            chunk.content,
        )
        for chunk in chunk_contents
    ]


async def extract_chunk_graph(content: str) -> ChunkGraphFacts:
    settings = get_settings()
    if not settings.llm_model:
        raise RuntimeError("Set LLM_MODEL and chat API credentials before graph indexing")
    client = chat_client()
    route = task_route("extract", settings, content)
    prompt = (
        "Extract only explicit entities and relationships supported by this source passage. "
        "Return only one JSON object, with no Markdown or extra text, using this exact shape: "
        '{"entities":[{"name":"...","evidence":"..."}],"relations":'
        '[{"source":"...","target":"...","predicate":"...",'
        '"description":"...","evidence":"...","confidence":0.0,'
        '"qualifiers":[{"kind":"environment","text":"exact source quote"}]}]}. '
        "Every entity must have name and evidence. Every relation must have source, target, "
        "predicate, description, evidence, and confidence from 0 to 1. Every evidence value "
        "must be an exact quote copied from the passage. Keep both top-level values as arrays; "
        "return empty arrays when no facts are supported. Extract at most 80 entities and 80 "
        "relations, prioritizing distinct facts with clear evidence. Do not infer unstated "
        "relationships. Preserve negation, may/must modality, environment, version and time. "
        "Copy each applicable qualifier into qualifiers using kind environment|version|condition|"
        "negation|modality|time and exact quoted text. Include every explicit scope and condition "
        "of the fact. Do not turn 'may write in staging' into an unconditional 'writes to'. "
        "Keep qualifiers within the relation's source chunk. Distinct scopes of the same endpoints "
        "must be separate facts. Omit a relation when its applicable scope cannot be determined."
        " The source-predicate-target triple itself must make a supported assertion: "
        "the target must be the actual object of the predicate, not just a nearby topic. "
        "Do not turn a timing or numeric property into an edge toward an unrelated object."
        " Preserve the complete predicate phrase needed to connect these endpoints. "
        "For example, from 'A stores messages in B', A --stores messages in--> B is valid, "
        "but A --stores--> B wrongly makes the storage location the stored object. "
        "From 'A waits two seconds before retrying B', do not emit A --waits two seconds--> B. "
        "A description cannot repair an incomplete or wrong predicate. "
        "Do not infer a prohibition from an inapplicable permission: 'this test permission "
        "does not apply to production' alone neither permits nor forbids the operation there."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"<source_passage>\n{content}\n</source_passage>"},
    ]
    for attempt in range(2):
        try:
            async with asyncio.timeout(settings.llm_timeout_seconds):
                response = await client.chat.completions.create(
                    model=route.model or "",
                    temperature=0,
                    messages=messages,
                    response_format={"type": "json_object"},
                    **structured_request_options(
                        route.model, settings.llm_base_url, settings, thinking=route.thinking
                    ),
                )
        except Exception as exc:
            # Provider exceptions can contain request data; avoid storing them verbatim.
            raise RuntimeError(
                f"Graph extraction provider request failed ({type(exc).__name__})"
            ) from exc
        raw = response.choices[0].message.content if response.choices else None
        try:
            if not raw:
                raise RuntimeError("Graph extraction provider returned an empty response")
            facts = parse_chunk_graph_response(raw)
            return validate_chunk_facts(facts, content)
        except RuntimeError as exc:
            if attempt:
                raise
            # Retry this batch once, keeping strict validation and source checks.
            assistant = {"role": "assistant", "content": raw or ""}
            reasoning = (
                getattr(response.choices[0].message, "reasoning_content", None)
                if response.choices
                else None
            )
            if reasoning:
                assistant["reasoning_content"] = reasoning
            messages = [
                *messages,
                assistant,
                {
                    "role": "user",
                    "content": (
                        "The previous response failed validation: "
                        f"{exc}. Return the complete corrected JSON object using exactly "
                        "the schema from the system message. Include every required field; "
                        "keep entities and relations as separate arrays. Preserve all "
                        "source-supported facts, copy evidence exactly, and add no new facts."
                    ),
                },
            ]
    raise RuntimeError("Graph extraction did not return a valid result")


async def prune_orphan_entities(session, knowledge_base_id: UUID) -> None:
    has_mention = exists().where(EntityMention.entity_id == Entity.id)
    await session.execute(
        delete(Entity).where(
            Entity.knowledge_base_id == knowledge_base_id,
            ~has_mention,
        )
    )


async def index_document_graph(document_id: UUID) -> None:
    """Extract and transactionally replace a document's evidence-backed graph index."""
    async with SessionLocal() as session:
        document = await session.get(Document, document_id)
        if document is None:
            return
        if document.status != DocumentStatus.PROCESSED:
            document.graph_status = "failed"
            document.graph_error = "Document must be processed before graph indexing"
            await session.commit()
            return

        document.graph_status = "processing"
        document.graph_error = None
        await session.commit()
        knowledge_base_id = document.knowledge_base_id
        result = await session.execute(
            select(Chunk.id, Chunk.content, Chunk.section, Chunk.page_number)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.ordinal)
        )
        chunks = [GraphSourceChunk(*row) for row in result.all()]

        try:
            semaphore = asyncio.Semaphore(4)

            async def extract_limited(content: str) -> ChunkGraphFacts:
                async with semaphore:
                    return await extract_chunk_graph(content)

            batches = batch_chunks(chunks)
            extracted_batches = await asyncio.gather(
                *(extract_limited(_batch_source_content(batch)) for batch in batches)
            )
            extracted = distribute_batch_facts(batches, extracted_batches)
            previous_reviews = list(
                await session.scalars(
                    select(Relation.semantic_review).where(Relation.document_id == document_id)
                )
            )
            semantic_reviews, semantic_summary = await check_relations(
                chunks, extracted, previous_reviews, get_settings()
            )
        except Exception as exc:
            await session.rollback()
            failed_document = await session.get(Document, document_id)
            if failed_document is not None:
                failed_document.graph_status = "failed"
                failed_document.graph_error = str(exc)[:1000]
                await session.commit()
            return

        try:
            document = await session.get(
                Document, document_id, populate_existing=True, with_for_update=True
            )
            if document is None:
                return
            current_chunks = list(
                (
                    await session.execute(
                        select(Chunk.id, Chunk.content)
                        .where(Chunk.document_id == document_id)
                        .order_by(Chunk.ordinal)
                    )
                ).all()
            )
            if document.status != DocumentStatus.PROCESSED or current_chunks != [
                (c.chunk_id, c.content) for c in chunks
            ]:
                # Reprocessing may replace the index while LLM checks are in flight.
                # Do not publish facts from a stale chunk snapshot.
                await session.rollback()
                return
            await session.execute(delete(Relation).where(Relation.document_id == document_id))
            await session.execute(
                delete(EntityMention).where(EntityMention.document_id == document_id)
            )
            await prune_orphan_entities(session, knowledge_base_id)

            entity_ids: dict[str, UUID] = {}
            entity_labels: dict[str, str] = {}
            for facts in extracted:
                for fact in facts.entities:
                    normalized = normalize_entity_name(fact.name)
                    if normalized:
                        entity_labels.setdefault(normalized, fact.name)

            for normalized, display_name in entity_labels.items():
                statement = (
                    insert(Entity)
                    .values(
                        knowledge_base_id=knowledge_base_id,
                        normalized_name=normalized,
                        display_name=display_name,
                    )
                    .on_conflict_do_nothing(
                        index_elements=[Entity.knowledge_base_id, Entity.normalized_name]
                    )
                    .returning(Entity.id)
                )
                entity_id = await session.scalar(statement)
                if entity_id is None:
                    entity_id = await session.scalar(
                        select(Entity.id).where(
                            Entity.knowledge_base_id == knowledge_base_id,
                            Entity.normalized_name == normalized,
                        )
                    )
                if entity_id is not None:
                    entity_ids[normalized] = entity_id

            mentions: dict[tuple[UUID, UUID], EntityMention] = {}
            relations: list[Relation] = []
            for chunk, facts in zip(chunks, extracted, strict=True):
                chunk_id = chunk.chunk_id
                local_ids: dict[str, UUID] = {}
                for fact in facts.entities:
                    normalized = normalize_entity_name(fact.name)
                    entity_id = entity_ids.get(normalized)
                    if entity_id is None:
                        continue
                    local_ids[normalized] = entity_id
                    mentions.setdefault(
                        (chunk_id, entity_id),
                        EntityMention(
                            knowledge_base_id=knowledge_base_id,
                            document_id=document_id,
                            chunk_id=chunk_id,
                            entity_id=entity_id,
                            surface_name=fact.name,
                            evidence=fact.evidence,
                        ),
                    )
                for fact_ordinal, fact in enumerate(facts.relations):
                    source_id = local_ids.get(normalize_entity_name(fact.source))
                    target_id = local_ids.get(normalize_entity_name(fact.target))
                    if source_id is None or target_id is None or source_id == target_id:
                        continue
                    relations.append(
                        Relation(
                            knowledge_base_id=knowledge_base_id,
                            document_id=document_id,
                            source_chunk_id=chunk_id,
                            source_entity_id=source_id,
                            target_entity_id=target_id,
                            predicate=fact.predicate.strip()[:200],
                            description=fact.description.strip(),
                            evidence=fact.evidence,
                            confidence=fact.confidence,
                            qualifiers={
                                "status": "reported",
                                "items": [q.model_dump() for q in fact.qualifiers],
                                "prompt_version": GRAPH_PROMPT_VERSION,
                            },
                            scope_key=relation_scope_key(fact.qualifiers),
                            semantic_status=semantic_reviews[(chunk_id, fact_ordinal)]["verdict"],
                            semantic_review=semantic_reviews[(chunk_id, fact_ordinal)],
                        )
                    )

            session.add_all(mentions.values())
            session.add_all(relations)
            document = await session.get(Document, document_id)
            if document is not None:
                document.graph_status = "processed"
                document.graph_error = None
                document.parsing_report = {
                    **(document.parsing_report or {}),
                    "graph_index": {
                        "prompt_version": GRAPH_PROMPT_VERSION,
                        "extraction_routes": [
                            task_route(
                                "extract", get_settings(), _batch_source_content(batch)
                            ).public()
                            for batch in batches
                        ],
                        "source_chunks": len(chunks),
                        "entities": len(entity_ids),
                        "relations": len(relations),
                        "semantic_review": semantic_summary,
                        "rejected_batch_relations": sum(
                            f.rejected_relations for f in extracted_batches
                        ),
                        "note": "限定引文存在性已检查；仍不保证完整抽取或关系语义正确。",
                    },
                }
            await session.commit()
        except Exception as exc:
            await session.rollback()
            failed_document = await session.get(Document, document_id)
            if failed_document is not None:
                failed_document.graph_status = "failed"
                failed_document.graph_error = f"Could not save graph index ({type(exc).__name__})"
                await session.commit()
