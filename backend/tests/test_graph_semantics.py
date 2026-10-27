import asyncio
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import answer_review, graph_semantics
from knowflow.app.config import Settings
from knowflow.app.graph_index import ChunkGraphFacts, GraphSourceChunk, RelationFact
from knowflow.app.providers import structured_request_options


def sample():
    chunk = GraphSourceChunk(uuid4(), "A uses B. A waits two seconds.")
    facts = ChunkGraphFacts(
        relations=[
            RelationFact(
                source="A",
                target="B",
                predicate="uses",
                description="A uses B.",
                evidence="A uses B.",
            )
        ]
    )
    return chunk, facts


def test_structured_provider_switch_does_not_leak_to_other_models_or_hosts():
    settings = Settings(_env_file=None)
    assert structured_request_options(
        "mimo-v2.6-pro", "https://api.xiaomimimo.com/v1", settings
    ) == {"extra_body": {"thinking": {"type": "disabled"}}}
    assert (
        structured_request_options("other-model", "https://api.xiaomimimo.com/v1", settings) == {}
    )
    assert structured_request_options("mimo-v2.6-pro", "https://example.com/v1", settings) == {}
    settings.structured_mimo_thinking = "provider_default"
    assert (
        structured_request_options("mimo-v2.6-pro", "https://api.xiaomimimo.com/v1", settings) == {}
    )


@pytest.mark.asyncio
async def test_gate_retains_all_candidates_and_caches_only_matching_completed_reviews(
    monkeypatch,
):
    settings = Settings(_env_file=None)
    chunk, facts = sample()
    facts.relations.extend(
        [
            facts.relations[0].model_copy(update={"predicate": "waits two seconds"}),
            facts.relations[0].model_copy(update={"predicate": "unknown"}),
        ]
    )
    calls = []

    async def judge(graph, _settings, *, max_edges):
        assert max_edges is None
        calls.append(len(graph["edges"]))
        return {
            "units": [
                {
                    "id": e["id"],
                    "verdict": "unsupported"
                    if e["predicate"] == "waits two seconds"
                    else "supported",
                }
                for e in graph["edges"]
                if e["predicate"] != "unknown"
            ]
        }

    monkeypatch.setattr(graph_semantics, "review_graph_edges", judge)
    results, summary = await graph_semantics.check_relations([chunk], [facts], [], settings)
    assert summary["counts"] == {"supported": 1, "unsupported": 1, "contradicted": 0, "unknown": 1}
    assert summary["eligible_relations"] == 1
    results2, summary2 = await graph_semantics.check_relations(
        [chunk], [facts], list(results.values()), settings
    )
    assert calls == [3, 1]  # Unknown work is retried, completed support/rejection is cached.
    assert summary2["cache_hits"] == 2
    assert results2[(chunk.chunk_id, 1)]["verdict"] == "unsupported"
    settings.llm_model = "different-reviewer"
    await graph_semantics.check_relations([chunk], [facts], list(results.values()), settings)
    assert calls[-1] == 3


@pytest.mark.asyncio
async def test_reviewer_failure_excludes_every_candidate_without_discarding_it(monkeypatch):
    chunk, facts = sample()

    async def outage(*_, **__):
        raise TimeoutError("provider error must not be persisted")

    monkeypatch.setattr(graph_semantics, "review_graph_edges", outage)
    results, summary = await graph_semantics.check_relations(
        [chunk], [facts], [], Settings(_env_file=None)
    )
    assert len(results) == 1 and summary["eligible_relations"] == 0
    assert results[(chunk.chunk_id, 0)]["verdict"] == "unknown"
    assert "provider error" not in str(results)


@pytest.mark.asyncio
async def test_auto_review_covers_more_than_the_offline_sample_limit(monkeypatch):
    edges = [
        {
            "id": str(i),
            "chunk_id": "source",
            "source_text": "A uses B.",
            "source": "A",
            "target": "B",
            "predicate": "uses",
            "description": "A uses B.",
        }
        for i in range(45)
    ]
    seen = []

    async def units_review(units, *_):
        seen.extend(u["id"] for u in units)
        assert len(units) <= 4
        return {"status": "complete", "units": [{**u, "verdict": "supported"} for u in units]}

    monkeypatch.setattr(answer_review, "review_units", units_review)
    result = await answer_review.review_graph_edges(
        {"edges": edges, "filtered_edge_count": 45}, Settings(_env_file=None), max_edges=None
    )
    assert set(seen) == {str(i) for i in range(45)}
    assert result["reviewed_edges"] == 45 and not result["sampled"]


def test_cache_is_invalidated_when_source_or_predicate_changes():
    chunk, facts = sample()
    original = graph_semantics.review_fingerprint(
        chunk, facts.relations[0], Settings(_env_file=None)
    )
    changed_source = chunk._replace(content="A must not use B.")
    assert (
        graph_semantics.review_fingerprint(
            changed_source, facts.relations[0], Settings(_env_file=None)
        )
        != original
    )
    assert (
        graph_semantics.review_fingerprint(
            chunk,
            facts.relations[0].model_copy(update={"predicate": "does not use"}),
            Settings(_env_file=None),
        )
        != original
    )


@pytest.mark.asyncio
async def test_review_has_an_absolute_deadline_even_when_transport_does_not_timeout(monkeypatch):
    async def hanging_request(**_):
        await asyncio.sleep(10)

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=hanging_request))
    )
    client.with_options = lambda **_: client
    monkeypatch.setattr(answer_review, "chat_client", lambda: client)
    settings = Settings(_env_file=None)
    settings.eval_judge_timeout_seconds = 0.02
    start = time.perf_counter()
    result = await answer_review.review_units([], [], [], settings, "test")
    assert result["status"] == "unavailable"
    assert result["error_type"] == "TimeoutError"
    assert time.perf_counter() - start < 1
