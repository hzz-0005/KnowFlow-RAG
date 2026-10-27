from __future__ import annotations

import hashlib
import json
import math
from time import perf_counter

import httpx
from fastapi import HTTPException

from knowflow.app.config import Settings, get_settings
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.round_budget import rerank_request


def require_rerank_configuration(settings: Settings) -> None:
    """Fail before paid query planning/embedding when the chosen service is absent."""
    if not settings.rerank_api_key or not settings.rerank_base_url or not settings.rerank_model:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "重排服务尚未配置，本次未调用模型、未退回其他检索策略。",
                "rerank": {
                    "contract": "rerank-execution-v1",
                    "status": "not_started",
                    "reason": "not_configured",
                    "fallback_used": False,
                    "candidate_count": None,
                    "model_revision_pin": "unknown",
                },
            },
        )


def apply_rerank_results(
    hits: list[RetrievalHit], results: list[dict[str, object]], top_k: int
) -> list[RetrievalHit]:
    """Map provider result indexes to candidates and retain their source provenance."""
    indexed: list[tuple[float, int, RetrievalHit]] = []
    seen: set[int] = set()
    for row in results:
        if not isinstance(row, dict):
            raise ValueError("Rerank API returned an invalid result row")
        index, score = row.get("index"), row.get("relevance_score")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(hits):
            raise ValueError("Rerank API returned an invalid candidate index")
        if index in seen:
            raise ValueError("Rerank API returned a duplicate candidate index")
        if isinstance(score, bool):
            raise ValueError("Rerank API returned a boolean relevance score")
        try:
            score_value = float(score)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("Rerank API returned an invalid relevance score") from exc
        if not math.isfinite(score_value) or not 0 <= score_value <= 1:
            raise ValueError("Rerank API returned a relevance score outside [0, 1]")
        seen.add(index)
        indexed.append((score_value, index, hits[index]))

    if len(seen) != len(hits):
        raise ValueError("Rerank API returned an incomplete candidate ranking")
    indexed.sort(key=lambda item: (-item[0], item[1]))
    ranked: list[RetrievalHit] = []
    for rank, (score, _, hit) in enumerate(indexed, start=1):
        hit.rerank_rank = rank
        hit.rerank_score = score
        ranked.append(hit)
    return ranked[:top_k]


async def rerank_hits(
    query: str,
    hits: list[RetrievalHit],
    top_k: int,
    settings: Settings | None = None,
) -> list[RetrievalHit]:
    settings = settings or get_settings()
    if not hits:
        return []
    trace = {
        "contract": "rerank-execution-v1",
        "status": "started",
        "fallback_used": False,
        "model": settings.rerank_model,
        "model_revision_pin": "unknown",
        "candidate_count": len(hits),
        "requested_top_k": top_k,
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "candidate_pool_sha256": hashlib.sha256(
            json.dumps(
                [(str(h.chunk.id), h.chunk.content) for h in hits],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "input_order": [str(h.chunk.id) for h in hits],
        "note": "同池执行记录，不是质量提升或校准概率；模型修订尚未固定。",
    }

    def fail(reason, status_code, latency=0):
        trace.update(status="failed", reason=reason, latency_ms=latency)
        return HTTPException(
            status_code=status_code,
            detail={"message": "重排未完成，本次未退回其他策略，也未生成答案。", "rerank": trace},
        )

    if not settings.rerank_api_key or not settings.rerank_base_url or not settings.rerank_model:
        raise fail("not_configured", 503)

    headers = {"Authorization": f"Bearer {settings.rerank_api_key.get_secret_value()}"}
    payload = {
        "model": settings.rerank_model,
        "input": {"query": query, "documents": [hit.chunk.content for hit in hits]},
        "parameters": {"top_n": len(hits), "instruct": settings.rerank_instruction},
    }
    started = perf_counter()
    try:

        async def request():
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                response = await client.post(
                    settings.rerank_base_url, headers=headers, json=payload
                )
                response.raise_for_status()
                return response.json()

        response_payload = await rerank_request(payload, request)
        if not isinstance(response_payload, dict):
            raise ValueError("Rerank API response is not an object")
        output = response_payload.get("output", response_payload)
        results = output.get("results") if isinstance(output, dict) else None
        if not isinstance(results, list):
            raise ValueError("Rerank API response has no results array")
        ranked = apply_rerank_results(hits, results, len(hits))
        elapsed_ms = (perf_counter() - started) * 1000
        trace.update(
            status="completed",
            latency_ms=elapsed_ms,
            returned_count=min(top_k, len(hits)),
            ranking=[
                {
                    "chunk_id": str(h.chunk.id),
                    "before_rank": hits.index(h) + 1,
                    "after_rank": h.rerank_rank,
                    "score": h.rerank_score,
                }
                for h in ranked
            ],
        )
        for hit in ranked:
            hit.rerank_latency_ms = elapsed_ms
            hit.rerank_trace = trace
        return ranked[:top_k]
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        reason = (
            "timeout"
            if isinstance(exc, httpx.TimeoutException)
            else "http_error"
            if isinstance(exc, httpx.HTTPError)
            else "invalid_response"
        )
        raise fail(reason, 502, (perf_counter() - started) * 1000) from exc
