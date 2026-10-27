"""Automatic source-bound relation checks; uncertain facts stay out of traversal."""

import hashlib
import json
import time
from datetime import UTC, datetime

from knowflow.app.answer_review import (
    GRAPH_REVIEW_QUESTION,
    GRAPH_REVIEW_VERSION,
    REVIEW_PROMPT,
    review_graph_edges,
)
from knowflow.app.providers import structured_request_options, task_route

GATE_VERSION = "graph-semantic-gate-v1"
KNOWN = {"supported", "unsupported", "contradicted"}


def review_fingerprint(chunk, fact, settings):
    separate = bool(
        settings.eval_judge_model and settings.eval_judge_base_url and settings.eval_judge_api_key
    )
    payload = {
        "source": chunk.content,
        "source_chunk_id": str(chunk.chunk_id),
        "fact": fact.model_dump(exclude={"confidence"}),
        "model_route": task_route("review", settings).public(),
        "provider": settings.eval_judge_base_url if separate else settings.llm_base_url,
        "review_prompt": REVIEW_PROMPT,
        "prompt_version": GRAPH_REVIEW_VERSION,
        "graph_review_question": GRAPH_REVIEW_QUESTION,
        "gate_version": GATE_VERSION,
        "request_options": structured_request_options(
            task_route("review", settings).model,
            settings.eval_judge_base_url if separate else settings.llm_base_url,
            settings,
            thinking=task_route("review", settings).thinking,
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


async def check_relations(chunks, facts_by_chunk, previous, settings):
    started = time.perf_counter()
    separate = bool(
        settings.eval_judge_model and settings.eval_judge_base_url and settings.eval_judge_api_key
    )
    cached = {
        row["fingerprint"]: row
        for row in previous
        if row.get("fingerprint") and row.get("verdict") in KNOWN
    }
    results, pending = {}, []
    cache_hits = 0
    for chunk, facts in zip(chunks, facts_by_chunk, strict=True):
        for ordinal, fact in enumerate(facts.relations):
            key = (chunk.chunk_id, ordinal)
            fingerprint = review_fingerprint(chunk, fact, settings)
            if fingerprint in cached:
                results[key] = {**cached[fingerprint], "cache_hit": True}
                cache_hits += 1
                continue
            review_id = str(len(pending))
            pending.append(
                (
                    key,
                    fingerprint,
                    {
                        "id": review_id,
                        "chunk_id": str(chunk.chunk_id),
                        "source_text": chunk.content,
                        "source": fact.source,
                        "target": fact.target,
                        "predicate": fact.predicate,
                        "description": fact.description,
                        "qualifiers": {"items": [q.model_dump() for q in fact.qualifiers]},
                    },
                )
            )
    if pending:
        try:
            review = await review_graph_edges(
                {"edges": [p[2] for p in pending], "filtered_edge_count": len(pending)},
                settings,
                max_edges=None,
            )
        except Exception:
            # Keep candidates inspectable even when the reviewer is unavailable.
            review = {"units": []}
        units = {u["id"]: u for u in review.get("units", [])}
        now = datetime.now(UTC).isoformat()
        for key, fingerprint, edge in pending:
            unit = units.get(edge["id"], {})
            verdict = unit.get("verdict")
            results[key] = {
                "fingerprint": fingerprint,
                "verdict": verdict if verdict in KNOWN else "unknown",
                "reason": unit.get("reason") or "关系评审不可用或结果未知，未进入图谱检索。",
                "evidence": unit.get("evidence", []),
                "model": review.get("model"),
                "same_model_as_generator": review.get("same_model_as_generator"),
                "prompt_version": GRAPH_REVIEW_VERSION,
                "gate_version": GATE_VERSION,
                "reviewed_at": now,
                "cache_hit": False,
            }
    counts = {
        status: sum(r["verdict"] == status for r in results.values())
        for status in [*sorted(KNOWN), "unknown"]
    }
    return results, {
        "gate_version": GATE_VERSION,
        "reviewed_relations": len(results),
        "cache_hits": cache_hits,
        "new_reviews": len(pending),
        "counts": counts,
        "eligible_relations": counts["supported"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "review_request_options": structured_request_options(
            task_route("review", settings).model,
            settings.eval_judge_base_url if separate else settings.llm_base_url,
            settings,
            thinking=task_route("review", settings).thinking,
        ),
        "note": (
            "仅模型判支持且引文校验通过的关系参与图遍历；排除及未知候选保留。模型判定仍可能出错。"
        ),
    }
