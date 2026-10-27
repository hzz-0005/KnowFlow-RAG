"""A round may reuse query vectors, never document scope or another round's cache."""

import json
import os
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import retrieval
from knowflow.app.config import Settings


def fingerprint(question):
    return [float(sum(question.encode("utf-8"))), float(len(question))]


def install_sources(monkeypatch, *, empty_at_gate=False):
    calls, scopes = [], []

    async def embed(queries, settings):
        calls.append((list(queries), settings.embedding_model))
        return [fingerprint(q) for q in queries]

    async def vectors(session, kb, queries, limit, settings, **kwargs):
        assert session.kb_id == kb
        assert len(session.document_ids) == 1
        scopes.append(session.document_ids[0])
        values = kwargs.get("vectors")
        if values is None:
            values = await retrieval.embed_texts(queries, settings)
        assert values == [fingerprint(q) for q in queries]
        if empty_at_gate and settings.rag_max_cosine_distance < 2:
            return [[] for _ in queries]
        doc = session.document_ids[0]
        return [
            [
                retrieval.RetrievalHit(
                    chunk=SimpleNamespace(id=UUID(int=doc.int + rank + 1), document_id=doc),
                    filename=str(doc),
                    vector_rank=rank + 1,
                    cosine_distance=0.1,
                )
                for rank in range(min(limit, 2))
            ]
            for _ in queries
        ]

    async def no_hits(*args, **kwargs):
        return []

    async def rerank(_query, hits, limit, _settings):
        return hits[:limit]

    monkeypatch.setattr(retrieval, "embed_texts", embed)
    monkeypatch.setattr(retrieval, "_vector_search_many", vectors)
    monkeypatch.setattr(retrieval, "_keyword_search", no_hits)
    monkeypatch.setattr(retrieval, "_graph_search", no_hits)
    monkeypatch.setattr("knowflow.app.reranker.rerank_hits", rerank)
    return calls, scopes


def signatures(rows):
    return [
        (
            h.chunk.id,
            h.chunk.document_id,
            h.vector_rank,
            h.fused_score,
            h.distance_gate_fallback,
            h.retrieval_stage,
        )
        for h in rows
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["vector", "hybrid", "hybrid_rerank", "graph_hybrid"])
@pytest.mark.parametrize("compound", [False, True])
@pytest.mark.parametrize("empty_at_gate", [False, True])
async def test_scoped_batch_reuses_each_actual_query_without_changing_hits(
    monkeypatch,
    strategy,
    compound,
    empty_at_gate,
):
    calls, scopes = install_sources(monkeypatch, empty_at_gate=empty_at_gate)
    settings = Settings(
        _env_file=None, table_row_retrieval_enabled=False, english_keyword_retrieval_enabled=False
    )
    kb, docs = uuid4(), [uuid4() for _ in range(4)]
    question = "生产策略保留多久？费用如何计算？" if compound else "每份策略的保留时间分别是多少？"
    rewrites = ["production retention period", "保留时间"] if strategy != "vector" else None

    # Same file scopes, limits and merge policy as the old per-document loop.
    shared = (await retrieval.embed_texts([question], settings))[0]
    groups = [
        await retrieval.retrieve(
            None,
            kb,
            question,
            strategy,
            2,
            settings,
            document_ids=(doc,),
            search_queries=rewrites,
            query_vector=shared,
        )
        for doc in docs
    ]
    expected = [h for row in zip_longest(*groups) for h in row if h is not None][:8]
    baseline_calls = len(calls)
    baseline_texts = sum(len(batch) for batch, _model in calls)
    calls.clear()
    scopes.clear()

    actual = await retrieval.retrieve_per_document(
        None,
        kb,
        question,
        strategy,
        8,
        settings,
        document_ids=docs,
        search_queries=rewrites,
    )
    assert signatures(actual) == signatures(expected)
    assert set(scopes) == set(docs)
    counts = Counter(q for batch, _model in calls for q in batch)
    assert counts and set(counts.values()) == {1}
    assert len(calls) <= 2  # A previously unused full question is lazy recovery only.
    if strategy != "vector" or compound:
        assert len(calls) < baseline_calls
    evidence = os.getenv("KNOWFLOW_TEST_EVIDENCE_DIR")
    if evidence:
        path = Path(evidence) / f"vector-reuse-{strategy}-{compound}-{empty_at_gate}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "measurement": "deterministic_software_control_not_provider_benchmark",
                    "documents": len(docs),
                    "strategy": strategy,
                    "compound": compound,
                    "empty_at_gate": empty_at_gate,
                    "baseline_embedding_calls": baseline_calls,
                    "reused_embedding_calls": len(calls),
                    "baseline_texts": baseline_texts,
                    "reused_texts": sum(len(batch) for batch, _model in calls),
                    "candidate_signatures_unchanged": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


@pytest.mark.asyncio
async def test_vector_reuse_does_not_cross_requests_or_model_settings(monkeypatch):
    calls, _ = install_sources(monkeypatch)
    docs = [uuid4(), uuid4()]
    for model in ("controlled-model-a", "controlled-model-b", "controlled-model-a"):
        await retrieval.retrieve_per_document(
            None,
            uuid4(),
            "保留多少天？",
            "hybrid",
            4,
            Settings(
                _env_file=None,
                embedding_model=model,
                table_row_retrieval_enabled=False,
                english_keyword_retrieval_enabled=False,
            ),
            document_ids=docs,
        )
    assert [model for _batch, model in calls] == [
        "controlled-model-a",
        "controlled-model-b",
        "controlled-model-a",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["keyword", "graph"])
async def test_non_vector_strategies_do_not_embed(monkeypatch, strategy):
    calls, _ = install_sources(monkeypatch)
    await retrieval.retrieve_per_document(
        None,
        uuid4(),
        "检查范围",
        strategy,
        4,
        Settings(_env_file=None),
        document_ids=[uuid4(), uuid4()],
    )
    assert calls == []


@pytest.mark.asyncio
async def test_partial_provider_batch_cannot_poison_the_round_cache(monkeypatch):
    async def incomplete(_queries, _settings):
        return [[1.0, 2.0]]

    monkeypatch.setattr(retrieval, "embed_texts", incomplete)
    cache = {}
    with pytest.raises(ValueError, match="count"):
        await retrieval._cached_vector_options(["first", "second"], Settings(_env_file=None), cache)
    assert cache == {}
