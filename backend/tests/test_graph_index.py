from types import SimpleNamespace
from uuid import UUID

import pytest
from knowflow.app import graph_index
from knowflow.app.graph_index import (
    ChunkGraphFacts,
    EntityFact,
    GraphSourceChunk,
    RelationFact,
    RelationQualifier,
    batch_chunks,
    distribute_batch_facts,
    extract_chunk_graph,
    normalize_entity_name,
    parse_chunk_graph_response,
    validate_chunk_facts,
)


def test_scoped_relations_preserve_distinct_conditions_and_reject_invented_or_missing_scope():
    content = "In staging Alpha may write to Redis. In production Alpha must not write to Redis."
    entities = [EntityFact(name=n, evidence=n) for n in ("Alpha", "Redis")]
    allowed = RelationFact(
        source="Alpha",
        target="Redis",
        predicate="writes_to",
        description="May write in staging",
        evidence="In staging Alpha may write to Redis.",
        qualifiers=[
            RelationQualifier(kind="environment", text="In staging"),
            RelationQualifier(kind="modality", text="may"),
        ],
    )
    denied = RelationFact(
        source="Alpha",
        target="Redis",
        predicate="writes_to",
        description="Must not write in production",
        evidence="In production Alpha must not write to Redis.",
        qualifiers=[
            RelationQualifier(kind="environment", text="In production"),
            RelationQualifier(kind="negation", text="must not"),
        ],
    )
    result = validate_chunk_facts(
        ChunkGraphFacts(entities=entities, relations=[allowed, denied]), content
    )
    assert len(result.relations) == 2
    forged = allowed.model_copy(
        update={"qualifiers": [RelationQualifier(kind="environment", text="outside")]}
    )
    missing = allowed.model_copy(update={"qualifiers": []})
    rejected = validate_chunk_facts(
        ChunkGraphFacts(entities=entities, relations=[forged, missing]), content
    )
    assert not rejected.relations
    assert rejected.rejected_relations == 2


def test_entity_normalization_merges_case_and_compatibility_forms() -> None:
    assert normalize_entity_name("  ＰｏｓｔｇｒｅＳＱＬ  ") == "postgresql"
    assert normalize_entity_name("数据库  连接") == "数据库 连接"


def test_graph_facts_require_exact_source_names_and_evidence() -> None:
    source = "PostgreSQL stores the runbook and supports vector search."
    facts = ChunkGraphFacts(
        entities=[
            EntityFact(name="PostgreSQL", evidence="PostgreSQL stores the runbook"),
            EntityFact(name="vector search", evidence="supports vector search"),
            EntityFact(name="private cloud", evidence="private cloud"),
        ],
        relations=[
            RelationFact(
                source="PostgreSQL",
                target="vector search",
                predicate="supports",
                description="PostgreSQL supports vector search.",
                evidence="supports vector search",
                confidence=0.9,
            ),
            RelationFact(
                source="PostgreSQL",
                target="private cloud",
                predicate="runs_on",
                description="PostgreSQL runs on a private cloud.",
                evidence="private cloud",
            ),
        ],
    )

    validated = validate_chunk_facts(facts, source)

    assert [entity.name for entity in validated.entities] == ["PostgreSQL", "vector search"]
    assert len(validated.relations) == 1
    assert validated.relations[0].target == "vector search"


def test_parse_chunk_graph_response_accepts_fenced_json() -> None:
    raw = 'Extracted facts:\n```json\n{"entities":[],"relations":[]}\n```'

    facts = parse_chunk_graph_response(raw)

    assert facts == ChunkGraphFacts(entities=[], relations=[])


def test_parse_chunk_graph_response_reports_schema_path_without_echoing_content() -> None:
    raw = '{"entities":[{"name":"","evidence":"private source text"}],"relations":[]}'

    try:
        parse_chunk_graph_response(raw)
    except RuntimeError as exc:
        assert "entities.0.name" in str(exc)
        assert "private source text" not in str(exc)
    else:
        raise AssertionError("invalid graph response should fail schema validation")


def test_batch_chunks_obeys_character_and_chunk_limits() -> None:
    chunks = [GraphSourceChunk(UUID(int=index + 1), "abc") for index in range(5)]

    batches = batch_chunks(chunks, max_chars=7, max_chunks=2)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert all(sum(len(chunk.content) for chunk in batch) <= 7 for batch in batches)


def test_batch_chunks_never_crosses_heading_or_pdf_page_boundaries() -> None:
    chunks = [
        GraphSourceChunk(UUID(int=1), "part one", section="Setup"),
        GraphSourceChunk(UUID(int=2), "part two", section="Setup"),
        GraphSourceChunk(UUID(int=3), "next section", section="Search"),
        GraphSourceChunk(UUID(int=4), "page two", page_number=2),
    ]

    batches = batch_chunks(chunks, max_chars=100, max_chunks=10)

    assert [[chunk.chunk_id for chunk in batch] for batch in batches] == [
        [UUID(int=1), UUID(int=2)],
        [UUID(int=3)],
        [UUID(int=4)],
    ]


def test_distribute_batch_facts_keeps_only_chunk_grounded_relations() -> None:
    first_id, second_id = UUID(int=1), UUID(int=2)
    batch = [
        GraphSourceChunk(first_id, "PostgreSQL supports vector search."),
        GraphSourceChunk(second_id, "Vector embeddings power retrieval."),
    ]
    facts = ChunkGraphFacts(
        entities=[
            EntityFact(name="PostgreSQL", evidence="PostgreSQL"),
            EntityFact(name="vector search", evidence="vector search"),
            EntityFact(name="vector embeddings", evidence="Vector embeddings"),
            EntityFact(name="retrieval", evidence="retrieval"),
        ],
        relations=[
            RelationFact(
                source="PostgreSQL",
                target="vector search",
                predicate="supports",
                description="PostgreSQL supports vector search.",
                evidence="supports vector search",
                confidence=0.9,
            ),
            RelationFact(
                source="PostgreSQL",
                target="retrieval",
                predicate="powers",
                description="PostgreSQL powers retrieval.",
                evidence="supports vector search and powers retrieval",
                confidence=0.8,
            ),
        ],
    )

    distributed = distribute_batch_facts([batch], [facts])

    assert len(distributed) == 2
    assert len(distributed[0].relations) == 1
    assert distributed[0].relations[0].target == "vector search"
    assert distributed[1].relations == []


def test_dense_chunk_deduplicates_endpoint_mentions_and_repeated_relations() -> None:
    names = [f"Entity-{index:03}" for index in range(60)]
    content = "; ".join(f"{names[index]} links to {names[index + 1]}" for index in range(59))
    relations = [
        RelationFact(
            source=names[index],
            target=names[index + 1],
            predicate="links_to",
            description=f"{names[index]} links to {names[index + 1]}",
            evidence=f"{names[index]} links to {names[index + 1]}",
        )
        for index in range(59)
    ]
    facts = ChunkGraphFacts(
        entities=[EntityFact(name=name, evidence=name) for name in names],
        relations=[*relations, relations[0]],
    )

    result = distribute_batch_facts([[GraphSourceChunk(UUID(int=1), content)]], [facts])[0]

    assert len(result.entities) == 60
    assert len(result.relations) == 59


@pytest.mark.asyncio
async def test_extract_chunk_graph_requests_json_mode(monkeypatch) -> None:
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content='{"entities":[],"relations":[]}')
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(
        graph_index,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="mimo-test",
            llm_timeout_seconds=60,
            llm_base_url=None,
            structured_mimo_thinking="provider_default",
        ),
    )
    monkeypatch.setattr(graph_index, "chat_client", lambda: fake_client)

    facts = await extract_chunk_graph("No graph facts in this sentence.")

    assert facts == ChunkGraphFacts(entities=[], relations=[])
    assert calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extract_chunk_graph_retries_malformed_schema_once(monkeypatch) -> None:
    calls = []
    responses = [
        '{"entities":[],"relations":[{"source":"PostgreSQL"}]}',
        '{"entities":[],"relations":[]}',
    ]

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=responses.pop(0)))]
            )

    monkeypatch.setattr(
        graph_index,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="test",
            llm_timeout_seconds=60,
            llm_base_url=None,
            structured_mimo_thinking="provider_default",
        ),
    )
    monkeypatch.setattr(
        graph_index,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    facts = await extract_chunk_graph("PostgreSQL is mentioned here.")

    assert facts == ChunkGraphFacts()
    assert len(calls) == 2
    assert "relations.0.target" in calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_extract_chunk_graph_does_not_silently_drop_persistently_invalid_rows(monkeypatch):
    calls = []

    class FakeCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"relations":[{}]}'))]
            )

    monkeypatch.setattr(
        graph_index,
        "get_settings",
        lambda: SimpleNamespace(
            llm_model="test",
            llm_timeout_seconds=60,
            llm_base_url=None,
            structured_mimo_thinking="provider_default",
        ),
    )
    monkeypatch.setattr(
        graph_index,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())),
    )

    with pytest.raises(RuntimeError, match="required JSON fields"):
        await extract_chunk_graph("A source passage.")

    assert len(calls) == 2
