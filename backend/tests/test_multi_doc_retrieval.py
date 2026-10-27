"""Per-document quota retrieval for explicit multi-file comparisons."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import retrieval


def hit(document_id, label):
    return SimpleNamespace(
        chunk=SimpleNamespace(id=uuid4(), document_id=document_id), filename=label
    )


@pytest.mark.asyncio
async def test_each_named_document_contributes_evidence(monkeypatch):
    seen_scopes = []
    docs = [uuid4(), uuid4(), uuid4()]

    async def fake_retrieve(_session, _kb, _query, _strategy, top_k, _settings, **kwargs):
        scopes = kwargs.get("document_ids") or ()
        seen_scopes.extend(scopes)
        document_id = scopes[0]
        return [hit(document_id, f"doc-{index}") for index in range(top_k)]

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)

    merged = await retrieval.retrieve_per_document(
        None, uuid4(), "compare", "hybrid", 6, None, document_ids=tuple(docs)
    )

    assert set(seen_scopes) == set(docs)
    assert {row.chunk.document_id for row in merged} == set(docs)


@pytest.mark.asyncio
async def test_a_single_document_keeps_the_plain_call(monkeypatch):
    calls = []

    async def fake_retrieve(*args, **kwargs):
        calls.append(kwargs)
        return [hit(args[0] if False else uuid4(), "only")]

    monkeypatch.setattr(retrieval, "retrieve", fake_retrieve)
    document_id = uuid4()

    await retrieval.retrieve_per_document(
        None, uuid4(), "question", "keyword", 5, None, document_ids=(document_id,)
    )

    assert len(calls) == 1 and calls[0]["document_ids"] == (document_id,)
