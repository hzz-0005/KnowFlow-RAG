from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app.reranker import apply_rerank_results
from knowflow.app.retrieval import RetrievalHit


def _hit(content: str) -> RetrievalHit:
    return RetrievalHit(
        chunk=SimpleNamespace(id=uuid4(), content=content),
        filename=f"{content}.md",
        vector_rank=1,
        keyword_rank=2,
        fused_score=0.02,
    )


def test_reranker_reorders_candidates_without_losing_retrieval_provenance() -> None:
    hits = [_hit("first"), _hit("second"), _hit("third")]

    ranked = apply_rerank_results(
        hits,
        [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.7},
            {"index": 1, "relevance_score": 0.1},
        ],
        top_k=2,
    )

    assert [hit.chunk.content for hit in ranked] == ["third", "first"]
    assert ranked[0].rerank_rank == 1
    assert ranked[0].rerank_score == 0.95
    assert ranked[0].vector_rank == 1
    assert ranked[0].keyword_rank == 2
    assert ranked[0].fused_score == 0.02


def test_reranker_rejects_invalid_or_incomplete_provider_indexes() -> None:
    with pytest.raises(ValueError, match="invalid candidate index"):
        apply_rerank_results([_hit("first")], [{"index": 4, "relevance_score": 0.2}], 1)
    with pytest.raises(ValueError, match="incomplete"):
        apply_rerank_results(
            [_hit("first"), _hit("second")], [{"index": 0, "relevance_score": 0.2}], 2
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["valid", "timeout", "http_error", "list_body", "null_row", "bool_score", "incomplete"]
)
async def test_http_rerank_trace_and_fail_closed(monkeypatch, case):
    import json

    import httpx
    from fastapi import HTTPException
    from knowflow.app.config import Settings
    from knowflow.app.reranker import rerank_hits

    original = httpx.AsyncClient

    def handler(request):
        assert request.headers["authorization"] == "Bearer do-not-record"
        assert json.loads(request.content)["parameters"]["top_n"] == 2
        if case == "timeout":
            raise httpx.ReadTimeout("private upstream text", request=request)
        if case == "http_error":
            return httpx.Response(503, text="private upstream text")
        payload = {
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        }
        if case == "list_body":
            payload = []
        if case == "null_row":
            payload["output"]["results"] = [None]
        if case == "bool_score":
            payload["output"]["results"][0]["relevance_score"] = True
        if case == "incomplete":
            payload["output"]["results"].pop()
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: original(transport=httpx.MockTransport(handler), **kw)
    )
    settings = Settings(
        _env_file=None,
        rerank_api_key="do-not-record",
        rerank_base_url="https://example.invalid/rerank",
        rerank_model="fixture-model",
    )
    hits = [_hit("first"), _hit("second")]
    if case == "valid":
        ranked = await rerank_hits("question", hits, 1, settings)
        assert ranked == [hits[1]]
        trace = ranked[0].rerank_trace
        assert trace["input_order"] == [str(h.chunk.id) for h in hits]
        assert [r["before_rank"] for r in trace["ranking"]] == [2, 1]
        assert trace["candidate_count"] == 2 and trace["returned_count"] == 1
        assert trace["model_revision_pin"] == "unknown"
    else:
        with pytest.raises(HTTPException) as failure:
            await rerank_hits("question", hits, 1, settings)
        trace = failure.value.detail["rerank"]
        assert trace["status"] == "failed" and trace["fallback_used"] is False
        assert all(h.rerank_rank is None for h in hits)
        assert "do-not-record" not in json.dumps(failure.value.detail)
        assert "private upstream text" not in json.dumps(failure.value.detail)
        assert trace["reason"] == (
            case if case in {"timeout", "http_error"} else "invalid_response"
        )


def test_ties_are_stable_and_bad_late_row_does_not_mutate_any_hit():
    hits = [_hit("one"), _hit("two")]
    assert (
        apply_rerank_results(
            hits, [{"index": 1, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.5}], 2
        )
        == hits
    )
    clean = [_hit("one"), _hit("two")]
    with pytest.raises(ValueError):
        apply_rerank_results(
            clean,
            [{"index": 0, "relevance_score": 0.5}, {"index": 1, "relevance_score": float("nan")}],
            2,
        )
    assert all(h.rerank_rank is None for h in clean)


@pytest.mark.asyncio
async def test_online_unconfigured_rerank_stops_before_paid_stages(monkeypatch):
    from fastapi import HTTPException
    from knowflow.app import api
    from knowflow.app.config import Settings

    async def authorized(*args):
        pass

    async def forbidden(*args, **kwargs):
        raise AssertionError("Paid stage must not be reached")

    monkeypatch.setattr(api, "owned_knowledge_base", authorized)
    monkeypatch.setattr(
        api,
        "get_settings",
        lambda: Settings(
            _env_file=None, rerank_api_key=None, rerank_base_url=None, rerank_model=None
        ),
    )
    monkeypatch.setattr(api, "plan_query", forbidden)
    monkeypatch.setattr(api, "retrieve", forbidden)
    with pytest.raises(HTTPException) as exc:
        await api.run_query(
            uuid4(), SimpleNamespace(strategy="hybrid_rerank"), SimpleNamespace(id=uuid4()), None
        )
    assert exc.value.status_code == 503
    assert exc.value.detail["rerank"]["status"] == "not_started"
