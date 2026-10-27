from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import reduce
from itertools import zip_longest
from typing import Literal
from uuid import UUID

from sqlalchemy import case, exists, func, literal_column, or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from knowflow.app.config import Settings, get_settings
from knowflow.app.document_links import REFERENCE, reference_label
from knowflow.app.graph_index import normalize_entity_name
from knowflow.app.ingestion import searchable_text
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentStatus,
    Entity,
    EntityMention,
    Relation,
)
from knowflow.app.providers import embed_texts

RetrievalStrategy = Literal["vector", "keyword", "hybrid", "hybrid_rerank", "graph", "graph_hybrid"]
RRF_K = 60
MAX_QUERY_PARTS = 4
# Strategies whose recall path embeds the question at least once.
VECTOR_STRATEGIES = {"vector", "hybrid", "hybrid_rerank", "graph_hybrid"}
EXCLUSION_CLAUSE = re.compile(
    r"^(?:不是(?:在)?问|不要(?:查|检索|搜索|回答)|排除|不包括|"
    r"not\s+asking(?:\s+about)?\b|do\s+not\s+(?:search|retrieve|include)\b|exclude\b)",
    re.I,
)


@dataclass(slots=True)
class RetrievalHit:
    chunk: Chunk
    filename: str
    cosine_distance: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    fused_score: float | None = None
    graph_rank: int | None = None
    rerank_rank: int | None = None
    rerank_score: float | None = None
    rerank_latency_ms: float | None = None
    rerank_trace: dict | None = None
    table_rank: int | None = None
    english_keyword_rank: int | None = None
    reference_paths: list[dict] = field(default_factory=list)
    retrieval_stage: str = "base"
    distance_gate_fallback: bool = False


def append_graph_candidates(
    base: list[RetrievalHit], graph: list[RetrievalHit], max_additions: int
) -> list[RetrievalHit]:
    """Preserve text top-k/order, then append a bounded, deduplicated graph pool.

    This protects candidate availability, not relevance acceptance or final correctness.
    Graph scores are not added to text scores, so a third channel cannot evict a
    necessary text source before the shared relevance and context gates see it.
    """
    result = list(base)
    by_id = {hit.chunk.id: hit for hit in result}
    added = 0
    for hit in graph:
        current = by_id.get(hit.chunk.id)
        if current is not None:
            current.graph_rank = hit.graph_rank
        elif added < max_additions:
            hit.retrieval_stage = "graph_expansion"
            result.append(hit)
            by_id[hit.chunk.id] = hit
            added += 1
    return result


def graph_candidate_diagnostics(hits: list[RetrievalHit], limit: int) -> dict:
    return {
        "policy": "text-seeded-graph-union-v3-empty-recovery",
        "base_count": sum(hit.retrieval_stage == "base" for hit in hits),
        "text_expansion_count": sum(hit.retrieval_stage == "text_expansion" for hit in hits),
        "added_count": sum(hit.retrieval_stage == "graph_expansion" for hit in hits),
        "shared_count": sum(
            hit.retrieval_stage == "base" and hit.graph_rank is not None for hit in hits
        ),
        "max_additions": limit,
        "candidate_count_before_references": len(hits),
        "distance_fallback_count": sum(hit.distance_gate_fallback for hit in hits),
        "note": (
            "保留文本前 k 条及顺序，另外补充最多 k 条图谱候选；"
            "图谱也从前 3 条文本来源中有已审核关系支持的实体起步，不推断实体别名。"
            "扩展候选仍需相关性检查和上下文预算限制，不保证被采用。"
            "基础 Recall@k 不含新增候选；扩展收益不是同候选数量的对照。"
        ),
    }


def _query_variants(query: str) -> list[str]:
    """Split explicit multi-question input so each clause gets its own retrieval pass."""
    parts = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"[。？！?!；;\n]+", query)
        if part.strip()
    ]
    if len(parts) < 2:
        return [query]
    unique_parts: list[str] = []
    seen = {query.casefold()}
    for part in parts:
        # A standalone exclusion is a constraint on the question, not a positive
        # subquery. Fusing it independently rewards the explicitly excluded topic.
        if EXCLUSION_CLAUSE.search(part):
            continue
        normalized = part.casefold()
        if normalized not in seen:
            seen.add(normalized)
            unique_parts.append(part)
        if len(unique_parts) >= MAX_QUERY_PARTS:
            break
    return unique_parts or [query]


def _retrieval_queries(query, strategy, search_queries):
    queries = _query_variants(query)
    if search_queries and strategy in {"hybrid", "hybrid_rerank", "graph_hybrid"}:
        queries = list(dict.fromkeys([query, *search_queries, *queries]))[:MAX_QUERY_PARTS]
    return queries


async def _cached_vector_options(queries, settings, query_vectors):
    """Reuse exact inputs inside one per-document retrieval call, never across users/rounds."""
    if query_vectors is None:
        return {}
    missing = list(dict.fromkeys(q for q in queries if q not in query_vectors))
    if missing:
        values = await embed_texts(missing, settings)
        # Do not populate a partial cache or silently associate the wrong query.
        if len(values) != len(missing):
            raise ValueError("Precomputed vector count must match the query count")
        query_vectors.update(zip(missing, values, strict=True))
    return {"vectors": [query_vectors[q] for q in queries]}


def _keyword_terms(query: str) -> list[str]:
    """Drop punctuation-only tokens so operators do not become broad FTS terms."""
    terms = dict.fromkeys(searchable_text(query).split())
    return [term for term in terms if len(term) > 1 and any(char.isalnum() for char in term)][:64]


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]], k: int = RRF_K
) -> list[RetrievalHit]:
    """Combine ranked results without comparing unrelated raw relevance scores."""
    if k <= 0:
        raise ValueError("RRF k must be positive")

    combined: dict[UUID, RetrievalHit] = {}
    for ranked_list in ranked_lists:
        for rank, hit in enumerate(ranked_list, start=1):
            current = combined.get(hit.chunk.id)
            if current is None:
                current = RetrievalHit(chunk=hit.chunk, filename=hit.filename)
                combined[hit.chunk.id] = current
            current.fused_score = (current.fused_score or 0.0) + 1 / (k + rank)
            current.distance_gate_fallback |= hit.distance_gate_fallback
            if hit.vector_rank is not None:
                if current.vector_rank is None or rank < current.vector_rank:
                    current.cosine_distance = hit.cosine_distance
                    current.vector_rank = rank
            if hit.keyword_rank is not None:
                if current.keyword_rank is None or rank < current.keyword_rank:
                    current.keyword_rank = rank
            if hit.graph_rank is not None:
                if current.graph_rank is None or rank < current.graph_rank:
                    current.graph_rank = rank
            if hit.table_rank is not None:
                if current.table_rank is None or rank < current.table_rank:
                    current.table_rank = rank
            if hit.english_keyword_rank is not None:
                if current.english_keyword_rank is None or rank < current.english_keyword_rank:
                    current.english_keyword_rank = rank
            if hit.rerank_score is not None:
                current.rerank_score = hit.rerank_score
                current.rerank_rank = rank

    return sorted(
        combined.values(),
        key=lambda hit: (-float(hit.fused_score or 0), str(hit.chunk.id)),
    )


async def _vector_search(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    limit: int,
    settings: Settings,
    *,
    vector: list[float] | None = None,
) -> list[RetrievalHit]:
    return (
        await _vector_search_many(
            session,
            knowledge_base_id,
            [query],
            limit,
            settings,
            **({"vectors": [vector]} if vector is not None else {}),
        )
    )[0]


async def _vector_search_many(
    session: AsyncSession,
    knowledge_base_id: UUID,
    queries: list[str],
    limit: int,
    settings: Settings,
    *,
    vectors: list[list[float]] | None = None,
) -> list[list[RetrievalHit]]:
    if vectors is None:
        vectors = await embed_texts(queries, settings)
    elif len(vectors) != len(queries):
        raise ValueError("Precomputed vector count must match the query count")
    ranked_results: list[list[RetrievalHit]] = []
    for vector in vectors:
        distance = Chunk.embedding.cosine_distance(vector).label("cosine_distance")
        result = await session.execute(
            select(Chunk, Document.filename, distance)
            .join(Document, Chunk.document_id == Document.id)
            .where(
                Chunk.knowledge_base_id == knowledge_base_id,
                Chunk.embedding_model == settings.embedding_model,
                Document.status == DocumentStatus.PROCESSED,
            )
            .order_by(distance, Chunk.id)
            .limit(limit)
        )
        ranked_results.append(
            [
                RetrievalHit(chunk, filename, cosine_distance=float(score), vector_rank=rank)
                for rank, (chunk, filename, score) in enumerate(result.all(), start=1)
                if float(score) <= settings.rag_max_cosine_distance
            ]
        )
    return ranked_results


async def _keyword_search(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    limit: int,
    *,
    minimum_matches: int | None = None,
    prefer_more_matches: bool = False,
) -> list[RetrievalHit]:
    terms = _keyword_terms(query)
    if not terms:
        return []
    # OR the segmented terms, but require two matches for longer queries. This
    # keeps one common name such as "pgvector" from making every language link a hit.
    term_queries = [func.plainto_tsquery("simple", term) for term in terms]
    ts_query = reduce(lambda left, right: left.op("||")(right), term_queries)
    matched_terms = reduce(
        lambda left, right: left + right,
        [
            case((Chunk.search_vector.op("@@")(term_query), 1), else_=0)
            for term_query in term_queries
        ],
    )
    if minimum_matches is None:
        minimum_matches = 2 if len(term_queries) >= 3 else 1
    rank = func.ts_rank_cd(Chunk.search_vector, ts_query).label("keyword_score")
    # A scattered one-term mention in a long overview is weaker evidence than a
    # passage that carries several of the question's own words, so recall inside a
    # single file can be ordered by how many of those words it actually contains.
    ordering = (
        (matched_terms.desc(), rank.desc(), Chunk.id)
        if prefer_more_matches
        else (rank.desc(), Chunk.id)
    )
    result = await session.execute(
        select(Chunk, Document.filename, rank)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Document.status == DocumentStatus.PROCESSED,
            Chunk.search_vector.op("@@")(ts_query),
            matched_terms >= minimum_matches,
        )
        .order_by(*ordering)
        .limit(limit)
    )
    return [
        RetrievalHit(chunk, filename, keyword_rank=position)
        for position, (chunk, filename, _) in enumerate(result.all(), start=1)
    ]


async def _english_keyword_search(session, knowledge_base_id, query, limit):
    # Separate stemming channel: never replace protected base top-k with its scores.
    terms = list(dict.fromkeys(re.findall(r"\b[a-zA-Z]{2,}\b", query.lower())))[:64]
    if len(terms) < 2:
        return []
    dictionary = literal_column("'english'::regconfig")
    vector = func.to_tsvector(dictionary, Chunk.search_text)
    term_queries = [func.plainto_tsquery(dictionary, term) for term in terms]
    ts_query = reduce(lambda a, b: a.op("||")(b), term_queries)
    # Stop words produce empty tsqueries and cannot count as matches.
    matches = reduce(
        lambda a, b: a + b,
        [case((vector.op("@@")(q), 1), else_=0) for q in term_queries],
    )
    rank = func.ts_rank_cd(vector, ts_query)
    result = await session.execute(
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Document.status == DocumentStatus.PROCESSED,
            vector.op("@@")(ts_query),
            matches >= 2,
        )
        .order_by(rank.desc(), Chunk.id)
        .limit(limit)
    )
    return [
        RetrievalHit(chunk, filename, english_keyword_rank=i)
        for i, (chunk, filename) in enumerate(result.all(), 1)
    ]


async def _append_english_candidates(session, kb_id, queries, base, limit, settings):
    if not settings.english_keyword_retrieval_enabled:
        return base
    rankings = [await _english_keyword_search(session, kb_id, q, limit * 3) for q in queries]
    by_id = {hit.chunk.id: hit for hit in base}
    result = list(base)
    added = 0
    for hit in reciprocal_rank_fusion(rankings):
        existing = by_id.get(hit.chunk.id)
        if existing is not None:
            existing.english_keyword_rank = hit.english_keyword_rank
        elif added < limit:
            hit.retrieval_stage = "text_expansion"
            result.append(hit)
            by_id[hit.chunk.id] = hit
            added += 1
    return result


def mentioned_table_keys(query: str, spans: list[dict]) -> set[str]:
    """Literal cell labels are retrieval hints, never inferred entity identities."""
    keys = set()
    for span in spans:
        key = str(span.get("row_key", "")).strip()
        if not 2 <= len(key) <= 200:
            continue
        pattern = re.escape(key)
        if key.isascii():
            pattern = r"(?<![a-zA-Z0-9_])" + pattern + r"(?![a-zA-Z0-9_])"
        if re.search(pattern, query, flags=re.IGNORECASE):
            keys.add(key.casefold())
    return keys


async def _table_row_search(
    session: AsyncSession, knowledge_base_id: UUID, query: str, limit: int
) -> list[RetrievalHit]:
    spans = func.jsonb_array_elements(Chunk.source_spans).table_valued("value").alias("row_span")
    key = spans.c.value.op("->>")("row_key")
    normalized = re.sub(r"\s+", "", query).lower()
    candidate = (
        select(1)
        .select_from(spans)
        .where(
            func.length(key).between(2, 200),
            func.strpos(normalized, func.regexp_replace(func.lower(key), r"\s+", "", "g")) > 0,
        )
        .correlate(Chunk)
        .exists()
    )
    table_mentions = [m for m in REFERENCE.finditer(query) if reference_label(m)[0] == "table"]
    legacy_candidate = candidate
    requested = {reference_label(m)[1] for m in table_mentions}
    if requested:
        # Constrain only this extra ranking channel, not vector/keyword retrieval.
        # Literal page/table metadata does not infer cross-document identity.
        candidate = (
            select(1)
            .select_from(spans)
            .where(spans.c.value.op("->>")("table_label").in_(requested))
            .correlate(Chunk)
            .exists()
        )
    statement = (
        select(Chunk, Document.filename)
        .join(Document, Document.id == Chunk.document_id)
        .where(
            Chunk.knowledge_base_id == knowledge_base_id,
            Document.status == DocumentStatus.PROCESSED,
        )
        .order_by(Chunk.id)
        .limit(max(limit * 50, 500))
    )
    rows = (await session.execute(statement.where(candidate))).all()
    if requested and not rows:
        # Preserve the older row-key channel for indexes without table labels.
        # This remains a retrieval hint, not a claim of matching the requested table.
        requested = set()
        rows = (await session.execute(statement.where(legacy_candidate))).all()
    cell_query = query
    for mention in reversed(table_mentions):
        cell_query = cell_query[: mention.start()] + " " + cell_query[mention.end() :]
    terms = _keyword_terms(cell_query)

    def score(chunk):
        if not requested:
            return len(mentioned_table_keys(query, chunk.source_spans)), 0, 0, 0
        return table_row_match_score(
            cell_query,
            [span for span in chunk.source_spans if span.get("table_label") in requested],
            terms,
        )

    matched = [(score(chunk), chunk, filename) for chunk, filename in rows]
    matched = sorted(
        (row for row in matched if requested or row[0][0]),
        key=lambda row: (*(-value for value in row[0]), row[2], row[1].ordinal),
    )
    return [
        RetrievalHit(chunk, filename, table_rank=rank)
        for rank, (_, chunk, filename) in enumerate(matched[:limit], 1)
    ]


def table_row_match_score(query: str, spans: list[dict], terms: list[str]) -> tuple:
    """Prefer conditions co-occurring in a row over pooled hits from unrelated rows.

    Literal text matches remain retrieval hints, not verified predicate matching.
    Aggregate coverage is a tie-breaker for questions requesting multiple rows.
    """
    rows = [set(span.get("cell_keys", [])) for span in spans if span.get("cell_keys")]

    def matches(cells):
        exact = len(mentioned_table_keys(query, [{"row_key": value} for value in cells]))
        overlap = sum(any(term.casefold() in cell.casefold() for cell in cells) for term in terms)
        return exact, overlap

    return (
        *max((matches(cells) for cells in rows), default=(0, 0)),
        *matches(set().union(*rows) if rows else set()),
    )


async def _graph_search(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    limit: int,
    *,
    seed_chunk_ids: list[UUID] | None = None,
) -> list[RetrievalHit]:
    normalized_query = normalize_entity_name(query)
    # Entity-only candidates must also be backed by an eligible relation in the
    # same passage, otherwise rejected edges can bypass the gate via mentions.
    approved_mention = exists().where(
        Relation.source_chunk_id == EntityMention.chunk_id,
        Relation.knowledge_base_id == knowledge_base_id,
        Relation.semantic_status == "supported",
        or_(
            Relation.source_entity_id == EntityMention.entity_id,
            Relation.target_entity_id == EntityMention.entity_id,
        ),
    )
    if len(normalized_query) < 2:
        return []
    terms = _keyword_terms(query)
    if not terms:
        return []
    term_queries = [func.plainto_tsquery("simple", term) for term in terms]
    ts_query = reduce(lambda left, right: left.op("||")(right), term_queries)
    rank = func.ts_rank_cd(Chunk.search_vector, ts_query)
    literal_seed = func.strpos(normalized_query, Entity.normalized_name) > 0
    # Semantic text retrieval has already supplied these source identities. Do
    # not require their graph neighbours to repeat the query's literal language;
    # cross-language and paraphrased questions may share no surface words.
    lexical_filter = true() if seed_chunk_ids else Chunk.search_vector.op("@@")(ts_query)
    seed_match = literal_seed
    if seed_chunk_ids:
        # A retrieved passage is a source of explicit entity mentions, not an
        # assertion that a short name in the question equals a longer graph name.
        # The eligible-relation gate and KB/document scope still apply below.
        seed_match = or_(literal_seed, EntityMention.chunk_id.in_(seed_chunk_ids[:3]))

    seed_result = await session.execute(
        select(
            Entity.id,
            Entity.normalized_name,
            func.length(Entity.normalized_name).label("entity_name_length"),
            literal_seed.label("literal_seed"),
        )
        .join(EntityMention, EntityMention.entity_id == Entity.id)
        .join(Document, Document.id == EntityMention.document_id)
        .where(
            Entity.knowledge_base_id == knowledge_base_id,
            EntityMention.knowledge_base_id == knowledge_base_id,
            approved_mention,
            Document.status == DocumentStatus.PROCESSED,
            func.length(Entity.normalized_name) >= 2,
            seed_match,
        )
        .distinct()
        .order_by(
            literal_seed.desc(), func.length(Entity.normalized_name).desc(), Entity.normalized_name
        )
        .limit(20)
    )
    seed_rows = seed_result.all()
    seeds = {row[0] for row in seed_rows}
    if not seeds:
        return []

    # Rank direct entity passages by query-term coverage instead of chunk order.
    mention_result = await session.execute(
        select(Chunk, Document.filename, EntityMention.entity_id, rank)
        .join(EntityMention, EntityMention.chunk_id == Chunk.id)
        .join(Entity, Entity.id == EntityMention.entity_id)
        .join(Document, Document.id == EntityMention.document_id)
        .where(
            EntityMention.knowledge_base_id == knowledge_base_id,
            approved_mention,
            EntityMention.entity_id.in_(seeds),
            Document.status == DocumentStatus.PROCESSED,
            lexical_filter,
        )
        .order_by(rank.desc(), Chunk.id)
        .limit(max(limit * 50, 500))
    )
    mentions_by_chunk: dict[UUID, tuple[Chunk, str, set[UUID], float]] = {}
    for chunk, filename, entity_id, score in mention_result.all():
        previous = mentions_by_chunk.get(chunk.id)
        if previous is None:
            mentions_by_chunk[chunk.id] = (chunk, filename, {entity_id}, float(score))
        else:
            previous[2].add(entity_id)
            mentions_by_chunk[chunk.id] = (
                previous[0],
                previous[1],
                previous[2],
                max(previous[3], float(score)),
            )

    seed_count = len(seeds)
    candidates: dict[UUID, tuple[int, float, float, Chunk, str]] = {}
    for chunk_id, (chunk, filename, matched_entities, score) in mentions_by_chunk.items():
        entity_coverage = len(matched_entities) / seed_count
        candidates[chunk_id] = (0, score + entity_coverage, 1.0, chunk, filename)

    frontier = seeds
    visited = set(seeds)
    for hop in range(1, 3):
        if not frontier:
            break
        edge_result = await session.execute(
            select(Relation, Chunk, Document.filename, rank)
            .join(Chunk, Chunk.id == Relation.source_chunk_id)
            .join(Document, Document.id == Relation.document_id)
            .where(
                Relation.knowledge_base_id == knowledge_base_id,
                Relation.semantic_status == "supported",
                Document.status == DocumentStatus.PROCESSED,
                or_(
                    Relation.source_entity_id.in_(frontier),
                    Relation.target_entity_id.in_(frontier),
                ),
                lexical_filter,
            )
            .order_by(rank.desc(), Relation.confidence.desc(), Relation.id)
            .limit(max(limit * 50, 500))
        )
        next_frontier: set[UUID] = set()
        for relation, chunk, filename, score in edge_result.all():
            previous = candidates.get(chunk.id)
            candidate = (hop, float(score), relation.confidence, chunk, filename)
            if previous is None or (hop, -float(score), -relation.confidence) < (
                previous[0],
                -previous[1],
                -previous[2],
            ):
                candidates[chunk.id] = candidate
            for endpoint in (relation.source_entity_id, relation.target_entity_id):
                if endpoint not in visited:
                    next_frontier.add(endpoint)
        visited.update(next_frontier)
        frontier = next_frontier

    ordered = sorted(
        candidates.values(),
        key=lambda row: (row[0], -row[1], -row[2], str(row[3].id)),
    )[:limit]
    return [
        RetrievalHit(chunk, filename, graph_rank=rank)
        for rank, (_, _, _, chunk, filename) in enumerate(ordered, start=1)
    ]


async def _nearest_after_gate(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    queries: list[str],
    limit: int,
    settings: Settings,
    *,
    vector: list[float] | None = None,
    query_vectors: dict[str, list[float]] | None = None,
) -> list[RetrievalHit]:
    """Offer the nearest sources when the distance gate left the pool empty.

    Distance is not calibrated relevance: a sweep or negation question can sit
    far from every chunk while still being answerable. The recovery only widens
    candidate availability and marks every survivor; the mandatory relevance
    and answer gates still decide whether it survives.
    """
    fallback_queries = list(dict.fromkeys([query, *queries]))[:MAX_QUERY_PARTS]
    from knowflow.app.source_scope import ScopedReadSession

    # A selected file is already a bounded search domain, so keep its k budget.
    recovery_limit = limit if isinstance(session, ScopedReadSession) else 1
    # Reuse the caller's vector only when the recovery asks for the same queries.
    options = await _cached_vector_options(fallback_queries, settings, query_vectors)
    reusable = vector is not None and fallback_queries == list(queries) and len(queries) == 1
    if not options and reusable:
        options = {"vectors": [vector]}
    nearest = await _vector_search_many(
        session,
        knowledge_base_id,
        fallback_queries,
        recovery_limit,
        settings.model_copy(update={"rag_max_cosine_distance": 2}),
        **options,
    )
    for ranking in nearest:
        for hit in ranking:
            hit.distance_gate_fallback = True
    return reciprocal_rank_fusion(nearest)[:limit]


async def _hybrid_text_candidates(
    session,
    knowledge_base_id,
    query,
    queries,
    limit,
    settings,
    *,
    query_vectors=None,
):
    vector_hits = await _vector_search_many(
        session,
        knowledge_base_id,
        queries,
        limit * 3,
        settings,
        **await _cached_vector_options(queries, settings, query_vectors),
    )
    keyword_hits = [
        await _keyword_search(session, knowledge_base_id, item, limit * 3) for item in queries
    ]
    table_hits = (
        await _table_row_search(session, knowledge_base_id, query, limit * 3)
        if settings.table_row_retrieval_enabled
        else []
    )
    base = reciprocal_rank_fusion([*vector_hits, *keyword_hits, table_hits])[:limit]
    if base or not settings.retrieval_assessment_enabled:
        return await _append_english_candidates(
            session, knowledge_base_id, queries, base, limit, settings
        )
    base = await _nearest_after_gate(
        session,
        knowledge_base_id,
        query,
        queries,
        limit,
        settings,
        query_vectors=query_vectors,
    )
    return await _append_english_candidates(
        session, knowledge_base_id, queries, base, limit, settings
    )


async def retrieve(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    strategy: RetrievalStrategy,
    top_k: int | None = None,
    settings: Settings | None = None,
    *,
    search_queries: list[str] | None = None,
    document_ids: tuple[UUID, ...] | None = None,
    query_vector: list[float] | None = None,
    query_vectors: dict[str, list[float]] | None = None,
) -> list[RetrievalHit]:
    """Retrieve authorized evidence; graph_hybrid returns text k plus up to k graph additions."""
    settings = settings or get_settings()
    if document_ids is not None:
        from knowflow.app.source_scope import ScopedReadSession

        if not document_ids:
            return []
        session = ScopedReadSession(session, knowledge_base_id, document_ids)
    limit = top_k or settings.rag_top_k
    queries = _retrieval_queries(query, strategy, search_queries)
    if strategy == "vector":
        if len(queries) > 1:
            searches = await _vector_search_many(
                session,
                knowledge_base_id,
                queries,
                limit * 3,
                settings,
                **await _cached_vector_options(queries, settings, query_vectors),
            )
            hits = reciprocal_rank_fusion(searches)[:limit]
        else:
            options = await _cached_vector_options([query], settings, query_vectors)
            hits = await _vector_search(
                session,
                knowledge_base_id,
                query,
                limit,
                settings,
                vector=options["vectors"][0] if options else query_vector,
            )
        if hits or not settings.retrieval_assessment_enabled:
            return hits
        # Same empty recovery as the hybrid channel: a sweep or negation question
        # can sit far from every chunk yet still be answerable. Survivors are
        # flagged and still pass through the mandatory gates.
        return await _nearest_after_gate(
            session,
            knowledge_base_id,
            query,
            queries,
            limit,
            settings,
            vector=query_vector,
            query_vectors=query_vectors,
        )
    if strategy == "keyword":
        if len(queries) > 1:
            searches = [
                await _keyword_search(session, knowledge_base_id, item, limit * 3)
                for item in queries
            ]
            return reciprocal_rank_fusion(searches)[:limit]
        return await _keyword_search(session, knowledge_base_id, query, limit)
    if strategy == "hybrid":
        return await _hybrid_text_candidates(
            session,
            knowledge_base_id,
            query,
            queries,
            limit,
            settings,
            query_vectors=query_vectors,
        )
    if strategy == "hybrid_rerank":
        from knowflow.app.reranker import rerank_hits

        vector_hits = await _vector_search_many(
            session,
            knowledge_base_id,
            queries,
            limit * 3,
            settings,
            **await _cached_vector_options(queries, settings, query_vectors),
        )
        keyword_hits = [
            await _keyword_search(session, knowledge_base_id, item, limit * 3) for item in queries
        ]
        table_hits = (
            await _table_row_search(session, knowledge_base_id, query, limit * 3)
            if settings.table_row_retrieval_enabled
            else []
        )
        candidates = reciprocal_rank_fusion([*vector_hits, *keyword_hits, table_hits])[: limit * 3]
        base = await rerank_hits(query, candidates, limit, settings)
        return await _append_english_candidates(
            session, knowledge_base_id, queries, base, limit, settings
        )
    if strategy == "graph":
        if len(queries) > 1:
            searches = [
                await _graph_search(session, knowledge_base_id, item, limit * 3) for item in queries
            ]
            return reciprocal_rank_fusion(searches)[:limit]
        return await _graph_search(session, knowledge_base_id, query, limit)
    if strategy == "graph_hybrid":
        base = await _hybrid_text_candidates(
            session,
            knowledge_base_id,
            query,
            queries,
            limit,
            settings,
            query_vectors=query_vectors,
        )
        graph_lists = [
            await _graph_search(
                session,
                knowledge_base_id,
                item,
                limit * 3,
                seed_chunk_ids=[hit.chunk.id for hit in base[:3]],
            )
            for item in queries
        ]
        graph = reciprocal_rank_fusion(graph_lists)
        return append_graph_candidates(base, graph, max_additions=limit)
    raise ValueError(f"Unsupported retrieval strategy: {strategy}")


async def retrieve_per_document(
    session: AsyncSession,
    knowledge_base_id: UUID,
    query: str,
    strategy,
    top_k: int | None,
    settings: Settings | None = None,
    *,
    document_ids: tuple[UUID, ...] | list[UUID],
    search_queries: list[str] | None = None,
) -> list[RetrievalHit]:
    """Give every explicitly named document a chance to contribute evidence.

    A plain global top-k lets the most keyword-dense file fill the window, so a
    comparison answer can claim the other file was never mentioned.
    """
    ids = list(dict.fromkeys(document_ids))
    if not ids:
        return []
    settings = settings or get_settings()
    total = top_k or settings.rag_top_k
    per_document = max(1, (total + len(ids) - 1) // len(ids))
    # Embed actual recall variants in one batch. The local map is discarded when
    # this call ends, and shares vectors only, never source results or SQL scope.
    query_vectors = {} if strategy in VECTOR_STRATEGIES else None
    if query_vectors is not None:
        queries = _retrieval_queries(query, strategy, search_queries)
        if strategy == "vector" and len(queries) == 1:
            queries = [query]
        await _cached_vector_options(queries, settings, query_vectors)
    groups: list[list[RetrievalHit]] = []
    for document_id in ids:
        hits = await retrieve(
            session,
            knowledge_base_id,
            query,
            strategy,
            per_document,
            settings,
            search_queries=search_queries,
            document_ids=(document_id,),
            query_vector=query_vectors.get(query) if query_vectors is not None else None,
            query_vectors=query_vectors,
        )
        if strategy == "vector":
            # A nearest-neighbour sample often misses the file's own wording. The
            # lexical channel costs no model call and finds the question's terms
            # inside this one file, so a sweep stops reporting "not mentioned".
            from knowflow.app.source_scope import ScopedReadSession

            lexical = await _keyword_search(
                ScopedReadSession(session, knowledge_base_id, (document_id,)),
                knowledge_base_id,
                query,
                # A single-file scope gets a small lexical quota; the rest of its
                # window stays with the vector ranking.
                min(per_document, 2),
                # Inside one file a single distinctive term is already located
                # evidence; demanding two would hide that file's only mention, so
                # passages carrying more of the question's words rank first.
                minimum_matches=1,
                prefer_more_matches=True,
            )
            for index, hit in enumerate(lexical, 1):
                hit.keyword_rank = index
            matched = [hit for hit in hits if not hit.distance_gate_fallback]
            sampled = [hit for hit in hits if hit.distance_gate_fallback]
            # The file's own wording ranks first, then real vector matches, and a
            # neighbour sample only fills the leftover quota: a tie must never drop
            # evidence for a sample, and the order must not depend on chunk ids.
            ordered: list[RetrievalHit] = []
            for hit in [*lexical, *matched, *sampled]:
                if hit.chunk.id not in {item.chunk.id for item in ordered}:
                    ordered.append(hit)
            hits = ordered[:per_document]
        groups.append(hits)
    # A channel can return base k PLUS expanded candidates. Concatenating those
    # lists then applying the global limit lets the first file consume every
    # slot. Interleave files first, preserving rank within each source, then cap.
    merged: list[RetrievalHit] = []
    seen: set[UUID] = set()
    for row in zip_longest(*groups):
        for hit in row:
            if hit is None:
                continue
            if hit.chunk.id in seen:
                continue
            seen.add(hit.chunk.id)
            merged.append(hit)
    return merged[: max(total, per_document * len(ids))]
