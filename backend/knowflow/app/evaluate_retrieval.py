from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.evaluation import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from knowflow.app.reference_expansion import expand_references
from knowflow.app.retrieval import RetrievalStrategy, graph_candidate_diagnostics, retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if not (PROJECT_ROOT / "evaluation").is_dir():
    # Installed wheels in the API image live outside /app. The working directory
    # holds the evaluation corpus; Git is deliberately absent from the runtime.
    PROJECT_ROOT = Path.cwd()
DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "dataset.json"
DEFAULT_RESULTS = PROJECT_ROOT / "evaluation" / "results"
STRATEGIES: tuple[RetrievalStrategy, ...] = (
    "keyword",
    "vector",
    "hybrid",
    "hybrid_rerank",
    "graph",
    "graph_hybrid",
)


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _evidence_match(hit, evidence: dict[str, str]) -> bool:
    return hit.filename == evidence["filename"] and _normalize(evidence["contains"]) in _normalize(
        hit.chunk.content
    )


def _code_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


async def run_evaluation(
    knowledge_base_id: uuid.UUID,
    strategy: RetrievalStrategy,
    top_k: int,
    dataset_path: Path = DEFAULT_DATASET,
    expand_document_references: bool = False,
) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    settings = get_settings()
    cases: list[dict[str, Any]] = []

    async with SessionLocal() as session:
        for case in dataset["cases"]:
            started = perf_counter()
            hits = await retrieve(
                session,
                knowledge_base_id,
                case["question"],
                strategy,
                top_k,
                settings,
            )
            base_hits = [hit for hit in hits if hit.retrieval_stage == "base"]
            graph_expansion = (
                graph_candidate_diagnostics(hits, top_k) if strategy == "graph_hybrid" else None
            )
            hits, expansion = await expand_references(
                session,
                knowledge_base_id,
                case["question"],
                hits,
                enabled=expand_document_references and settings.reference_expansion_enabled,
            )
            latency_ms = (perf_counter() - started) * 1000
            gold = case["relevant_evidence"]
            matched_gold: set[str] = set()
            ranked_labels: list[str] = []
            retrieved = []
            for hit in base_hits[:top_k]:
                match = next(
                    (
                        evidence
                        for evidence in gold
                        if evidence["id"] not in matched_gold and _evidence_match(hit, evidence)
                    ),
                    None,
                )
                if match is None:
                    ranked_labels.append(f"unmatched:{hit.chunk.id}")
                else:
                    matched_gold.add(match["id"])
                    ranked_labels.append(match["id"])
                retrieved.append(
                    {
                        "chunk_id": str(hit.chunk.id),
                        "document_id": str(hit.chunk.document_id),
                        "filename": hit.filename,
                        "matched_evidence_id": match["id"] if match else None,
                        "vector_rank": hit.vector_rank,
                        "keyword_rank": hit.keyword_rank,
                        "graph_rank": hit.graph_rank,
                        "table_rank": hit.table_rank,
                        "rerank_rank": hit.rerank_rank,
                        "rerank_score": hit.rerank_score,
                    }
                )

            relevant_ids = {item["id"] for item in gold}
            case_result: dict[str, Any] = {
                "id": case["id"],
                "question_type": case["question_type"],
                "question": case["question"],
                "reference_answer": case["reference_answer"],
                "should_refuse": case["should_refuse"],
                "answer_evaluation": "not_run; retrieval-only experiment",
                "retrieval_latency_ms": round(latency_ms, 3),
                "reranking_latency_ms": (
                    round(hits[0].rerank_latency_ms, 3)
                    if hits and hits[0].rerank_latency_ms is not None
                    else None
                ),
                "retrieved": retrieved,
                "matched_evidence_ids": sorted(matched_gold),
                "reference_expansion": expansion,
                "graph_expansion": graph_expansion,
                "expanded_candidates": [
                    {
                        "chunk_id": str(hit.chunk.id),
                        "filename": hit.filename,
                        "page_number": hit.chunk.page_number,
                        "reference_paths": hit.reference_paths,
                    }
                    for hit in hits
                ],
            }
            if relevant_ids:
                expanded_gold = {
                    item["id"] for item in gold if any(_evidence_match(hit, item) for hit in hits)
                }
                case_result["metrics"] = {
                    f"recall@{top_k}": recall_at_k(ranked_labels, relevant_ids, top_k),
                    f"precision@{top_k}": precision_at_k(ranked_labels, relevant_ids, top_k),
                    "mrr": reciprocal_rank(ranked_labels, relevant_ids),
                    f"ndcg@{top_k}": ndcg_at_k(ranked_labels, relevant_ids, top_k),
                    "candidate_evidence_recall": len(expanded_gold) / len(relevant_ids),
                    "candidate_complete_evidence_set": float(expanded_gold == relevant_ids),
                }
            else:
                case_result["metrics"] = {"no_answer_returned_any_chunk": bool(hits)}
            cases.append(case_result)

    answerable = [case["metrics"] for case in cases if case["should_refuse"] is False]
    no_answer = [case["metrics"] for case in cases if case["should_refuse"] is True]

    def average(metric: str) -> float:
        return sum(row[metric] for row in answerable) / len(answerable) if answerable else 0.0

    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": _code_revision(),
        "dataset_version": dataset["version"],
        "dataset_source": dataset["source"],
        "knowledge_base_id": str(knowledge_base_id),
        "strategy": strategy,
        "top_k": top_k,
        "reference_expansion_enabled": expand_document_references
        and settings.reference_expansion_enabled,
        "candidate_metric_note": (
            "展开后指标针对相关性检查和生成预算之前的候选池，"
            "不代表进入模型的完整证据率或回答正确率。"
        ),
        "table_row_hints_enabled": settings.table_row_retrieval_enabled
        and strategy in {"hybrid", "hybrid_rerank", "graph_hybrid"},
        "models": {
            "embedding": settings.embedding_model
            if strategy in ("vector", "hybrid", "hybrid_rerank", "graph_hybrid")
            else None,
            "embedding_dimensions": settings.embedding_dimensions
            if strategy in ("vector", "hybrid", "hybrid_rerank", "graph_hybrid")
            else None,
            "reranker": settings.rerank_model if strategy == "hybrid_rerank" else None,
        },
        "metrics": {
            f"mean_recall@{top_k}": average(f"recall@{top_k}"),
            f"mean_precision@{top_k}": average(f"precision@{top_k}"),
            "mrr": average("mrr"),
            "mean_candidate_evidence_recall": average("candidate_evidence_recall"),
            "candidate_complete_evidence_set_rate": average("candidate_complete_evidence_set"),
            f"mean_ndcg@{top_k}": average(f"ndcg@{top_k}"),
            "mean_retrieval_latency_ms": (
                sum(case["retrieval_latency_ms"] for case in cases) / len(cases) if cases else 0.0
            ),
            "no_answer_empty_retrieval_rate": (
                sum(not row["no_answer_returned_any_chunk"] for row in no_answer) / len(no_answer)
                if no_answer
                else None
            ),
        },
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-only KnowFlow evaluation.")
    parser.add_argument("--knowledge-base-id", required=True, type=uuid.UUID)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--expand-references",
        action="store_true",
        help="Expand references after base top-k; report candidate coverage separately.",
    )
    args = parser.parse_args()
    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    report = asyncio.run(
        run_evaluation(
            args.knowledge_base_id, args.strategy, args.top_k, args.dataset, args.expand_references
        )
    )
    output_path = args.output or DEFAULT_RESULTS / f"{report['run_id']}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output_path), "metrics": report["metrics"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
