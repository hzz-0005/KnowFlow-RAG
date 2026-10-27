"""Run the real answer pipeline against fixed gold evidence and compare strategies."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from knowflow.app.answer_budget import budget_contract
from knowflow.app.answer_review import review_answer, review_graph_edges
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.evaluate_retrieval import PROJECT_ROOT, _code_revision
from knowflow.app.graph_inspection import inspect_graph
from knowflow.app.guard_budget import budget_contract as guard_budget_contract
from knowflow.app.models import Chunk, Document, DocumentBlock, KnowledgeBase, Relation, User
from knowflow.app.providers import ANSWER_PROMPT_VERSION, routing_contract
from knowflow.app.quality_metrics import (
    METRICS_VERSION,
    compare_variants,
    equivalent_question_consistency,
    evaluate_trace,
    regression_check,
    summarize,
)
from knowflow.app.query_plan import rewrite_contract
from knowflow.app.relevance_budget import budget_contract as relevance_budget_contract
from knowflow.app.schemas import QueryRequest

DATASETS = {
    "engineering": "evaluation/dataset.json",
    "tables": "evaluation/complex_documents/table_caption_queries.json",
    "crosspage": "evaluation/complex_documents/tableformer_crosspage_queries.json",
}
ACTIVE_RUNS = set()
ACTIVE_KBS = set()


def validate_dataset(data):
    if not isinstance(data, dict) or not isinstance(data.get("version"), str):
        raise ValueError("题集缺少版本标识。")
    cases = data.get("cases", [])
    if not 1 <= len(cases) <= 50 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("题集需含 1–50 个唯一问题。")
    for case in cases:
        QueryRequest(question=case["question"])
        if not isinstance(case.get("should_refuse"), bool):
            raise ValueError("每题需标注是否应拒答。")
        gold = case["relevant_evidence"]
        if not isinstance(gold, list) or len(gold) > 30:
            raise ValueError("标准证据数量无效。")
        if len({g["id"] for g in gold}) != len(gold):
            raise ValueError("同题标准证据编号不能重复。")
        if (
            not isinstance(case.get("reference_answer"), str)
            or len(case["reference_answer"]) > 8000
        ):
            raise ValueError("每题需要长度不超过 8000 字符的参考答案。")
        if not case["should_refuse"] and not gold:
            raise ValueError("可回答题需要标准证据。")
        for item in gold:
            if not all(
                isinstance(item.get(k), str) and item[k].strip()
                for k in ("id", "filename", "contains")
            ):
                raise ValueError("每条标准证据需要编号、文件名和原文。")
            if len(item["contains"]) > 5000 or len(item["filename"]) > 255:
                raise ValueError("标准引文或文件名过长。")
    return data


def save_report(report, directory):
    path = Path(directory) / f"{report['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Windows readers/scanners may briefly hold a destination without FILE_SHARE_DELETE.
    # Keep the existing JSON intact; retry only recognized Windows contention codes.
    for delay in (0.01, 0.05, 0.1, None):
        try:
            os.replace(temp, path)
            break
        except PermissionError as exc:
            if delay is None or getattr(exc, "winerror", None) not in {5, 32, 33}:
                raise
            time.sleep(delay)
    return path


async def source_snapshot(kb_id):
    async with SessionLocal() as session:
        kb = await session.get(KnowledgeBase, kb_id)
        if kb is None:
            raise ValueError("Knowledge base not found")
        owner_id = kb.owner_id
        docs = list(
            await session.scalars(
                select(Document).where(Document.knowledge_base_id == kb_id).order_by(Document.id)
            )
        )
        indexed = list(
            await session.scalars(
                select(Chunk)
                .join(Document)
                .where(Chunk.knowledge_base_id == kb_id, Document.status == "processed")
            )
        )
        blocks = list(
            await session.execute(
                select(DocumentBlock, Document.filename)
                .join(Document)
                .where(Document.knowledge_base_id == kb_id, Document.status == "processed")
            )
        )
        filenames = {doc.id: doc.filename for doc in docs}
        graph = await inspect_graph(session, kb_id, limit=500)
        graph_values = list(
            (
                await session.execute(
                    select(
                        Relation.id,
                        Relation.source_entity_id,
                        Relation.target_entity_id,
                        Relation.predicate,
                        Relation.evidence,
                        Relation.qualifiers,
                        Relation.semantic_status,
                        Relation.semantic_review,
                    )
                    .join(Document)
                    .where(Relation.knowledge_base_id == kb_id, Document.status == "processed")
                    .order_by(Relation.id)
                )
            ).all()
        )
        graph_sha = hashlib.sha256(
            json.dumps([tuple(map(str, r)) for r in graph_values], sort_keys=True).encode()
        ).hexdigest()
        corpus = [(d.filename, d.sha256, str(d.status), d.graph_status) for d in docs]
        corpus_sha = hashlib.sha256(json.dumps(corpus).encode()).hexdigest()
        index_sha = hashlib.sha256(
            json.dumps(sorted((str(c.id), c.content, c.embedding_model) for c in indexed)).encode()
        ).hexdigest()
        return {
            "owner_id": owner_id,
            "corpus_sha256": corpus_sha,
            "index_sha256": index_sha,
            "graph": graph,
            "graph_sha256": graph_sha,
            "documents": [
                {"filename": d.filename, "status": d.status, "graph_status": d.graph_status}
                for d in docs
            ],
            "indexed": [
                {"chunk_id": str(c.id), "filename": filenames[c.document_id], "content": c.content}
                for c in indexed
            ],
            "parsed": [{"filename": name, "content": block.text} for block, name in blocks],
        }


def initial_report(kb_id, dataset, strategies, top_k, judge, run_id=None):
    settings = get_settings()
    return {
        "run_id": str(run_id or uuid.uuid4()),
        "kind": "rag_quality",
        "metrics_version": METRICS_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "execution_environment": "cli",
        "heartbeat_at": datetime.now(UTC).isoformat(),
        "phase": "准备语料与索引快照",
        "code_revision": _code_revision(),
        "knowledge_base_id": str(kb_id),
        "dataset_version": dataset["version"],
        "dataset_source": dataset.get("source", ""),
        "dataset_sha256": hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest(),
        "strategy": strategies[-1],
        "strategies": strategies,
        "top_k": top_k,
        "judge_enabled": judge,
        "total_cases": len(dataset["cases"]) * len(strategies),
        "metrics": {},
        "variants": {},
        "cases": [],
        "models": {
            "embedding": settings.embedding_model,
            "generator": settings.llm_model,
            "complex_generator": settings.llm_complex_model,
            "simple_thinking": settings.llm_simple_thinking,
            "complex_thinking": settings.llm_complex_thinking,
            "judge_thinking": settings.eval_judge_thinking,
            "judge": settings.eval_judge_model or settings.llm_model,
        },
        "note": "固定标注回归，不代表线上总体准确率；模型评审仅为估算，未知项不计作通过。",
        "answer_prompt_version": ANSWER_PROMPT_VERSION,
        "answer_budget_contract": budget_contract(settings),
        "guard_budget_contract": guard_budget_contract(settings),
        "relevance_budget_contract": relevance_budget_contract(settings),
        "model_routing_contract": routing_contract(settings),
        "query_rewrite_contract": rewrite_contract(settings),
        "pipeline": {
            "english_keyword_retrieval_enabled": settings.english_keyword_retrieval_enabled,
            "query_rewrite_enabled": settings.query_rewrite_enabled,
            "query_rewrite_timeout_seconds": settings.query_rewrite_timeout_seconds,
            "answer_guard_enabled": settings.answer_guard_enabled,
            "answer_guard_thinking": settings.answer_guard_thinking,
            "answer_repair_enabled": settings.answer_repair_enabled,
            "answer_repair_timeout_seconds": settings.answer_repair_timeout_seconds,
            "answer_guard_timeout_seconds": settings.answer_guard_timeout_seconds,
        },
    }


async def execute_quality(
    kb_id, dataset, strategies, top_k, judge=False, report=None, baseline=None
):
    from knowflow.app.api import run_query

    settings = get_settings()
    report = report or initial_report(kb_id, dataset, strategies, top_k, judge)
    run_id = report["run_id"]
    ACTIVE_RUNS.add(run_id)
    ACTIVE_KBS.add(str(kb_id))

    async def heartbeat():
        while True:
            report["heartbeat_at"] = datetime.now(UTC).isoformat()
            save_report(report, settings.evaluation_results_dir)
            await asyncio.sleep(10)

    pulse = asyncio.create_task(heartbeat())
    try:
        snapshot = await source_snapshot(kb_id)
        report.update(
            corpus_sha256=snapshot["corpus_sha256"],
            index_sha256=snapshot["index_sha256"],
            graph_inspection=snapshot["graph"],
            graph_sha256=snapshot["graph_sha256"],
            documents=snapshot["documents"],
        )
        required_files = {g["filename"] for c in dataset["cases"] for g in c["relevant_evidence"]}
        ready = {d["filename"] for d in snapshot["documents"] if d["status"] == "processed"}
        if missing := sorted(required_files - ready):
            raise ValueError("当前题集所需文档尚未就绪：" + "、".join(missing))
        graph_requested = any(strategy in {"graph", "graph_hybrid"} for strategy in strategies)
        report["review_scope"] = {
            "answers": judge,
            "graph_relations": judge and graph_requested,
            "note": "普通文本策略只评审答案；仅选择图谱策略时额外评审关系。",
        }
        if judge and graph_requested and snapshot["graph"]["edges"]:
            report["phase"] = "评审图谱关系的原文支持"
            save_report(report, settings.evaluation_results_dir)
            report["graph_review"] = await review_graph_edges(snapshot["graph"], settings)
            save_report(report, settings.evaluation_results_dir)
        for strategy in strategies:
            cases = []
            report["variants"][strategy] = {"cases": cases, "metrics": {}, "denominators": {}}
            for case in dataset["cases"]:
                report["phase"] = f"{strategy} · {case['id']} · 执行真实问答"
                save_report(report, settings.evaluation_results_dir)
                started = time.perf_counter()
                row = {
                    "id": case["id"],
                    "question": case["question"],
                    "question_type": case.get("question_type"),
                    "equivalence_group": case.get("equivalence_group"),
                    "should_refuse": case["should_refuse"],
                    "metrics": {},
                }
                try:
                    if (
                        strategy.startswith("graph")
                        and not snapshot["graph"]["eligible_edge_count"]
                    ):
                        raise ValueError(
                            "当前没有可检索图谱关系，先构建图谱；不以空图结果证明无收益。"
                        )
                    audit = {"top_k": top_k}
                    async with SessionLocal() as session:
                        owner = await session.get(User, snapshot["owner_id"])
                        response = await run_query(
                            kb_id,
                            QueryRequest(
                                question=case["question"],
                                strategy=strategy,
                                top_k=top_k,
                                expand_references=True,
                            ),
                            owner,
                            session,
                            audit,
                        )
                    row["query_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                    payload = response.model_dump(mode="json")
                    row.update(
                        evaluate_trace(
                            case, audit, payload, snapshot["indexed"], snapshot["parsed"]
                        )
                    )
                    row["trace"] = audit
                    row["response"] = payload
                    if judge and not response.refused:
                        report["phase"] = f"{strategy} · {case['id']} · 评审答案依据与要点"
                        save_report(report, settings.evaluation_results_dir)
                        row["answer_review"] = await review_answer(
                            case, response.answer, audit["context"], settings
                        )
                        row["metrics"].update(
                            {
                                k: row["answer_review"].get(k)
                                for k in ("faithfulness_estimate", "fact_recall_estimate")
                            }
                        )
                    else:
                        row["answer_review"] = {
                            "status": "refused" if response.refused else "not_run"
                        }
                except Exception as exc:
                    # Avoid persisting provider error strings which may contain credentials.
                    row["error"] = (
                        str(exc)
                        if isinstance(exc, ValueError)
                        else "本题执行失败，请检查本地服务日志；未计作通过。"
                    )
                row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
                cases.append(row)
                metrics, denominators = summarize(cases)
                report["variants"][strategy].update(
                    metrics=metrics,
                    denominators=denominators,
                    equivalence_checks=equivalent_question_consistency(cases),
                )
                report.update(
                    completed_cases=sum(len(v["cases"]) for v in report["variants"].values()),
                )
                if strategy == report["strategy"]:
                    report.update(metrics=metrics, cases=cases)
                save_report(report, settings.evaluation_results_dir)
            latencies = sorted(r["query_latency_ms"] for r in cases if "query_latency_ms" in r)
            report["variants"][strategy]["performance"] = {
                "query_mean_ms": sum(latencies) / len(latencies) if latencies else None,
                "query_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1]
                if latencies
                else None,
                "successful_queries": len(latencies),
                "billing_cost": None,
                "note": "问答耗时不含离线答案评审；当前不具备供应商实际计费数据。",
            }
            print(f"完成 {strategy}: {len(cases)} 题", flush=True)
        fresh = await source_snapshot(kb_id)
        report["source_snapshot_unchanged"] = (
            fresh["corpus_sha256"] == snapshot["corpus_sha256"]
            and fresh["index_sha256"] == snapshot["index_sha256"]
            and fresh["graph_sha256"] == snapshot["graph_sha256"]
        )
        first = strategies[0]
        report["comparisons"] = {
            strategy: compare_variants(
                report["variants"][first]["cases"], report["variants"][strategy]["cases"]
            )
            for strategy in strategies[1:]
        }
        report["status"] = (
            "completed"
            if report["source_snapshot_unchanged"]
            and not any(c.get("error") for v in report["variants"].values() for c in v["cases"])
            else "incomplete"
        )
        if not report["source_snapshot_unchanged"]:
            report["comparison_warning"] = "评测期间索引改变，对照结果不可直接比较。"
        if baseline:
            report["baseline_check"] = regression_check(report, baseline)
    except Exception as exc:
        report.update(
            status="failed",
            error=str(exc) if isinstance(exc, ValueError) else "评测服务异常，未生成有效结论。",
            error_type=type(exc).__name__,
            error_errno=getattr(exc, "errno", None),
            error_winerror=getattr(exc, "winerror", None),
        )
    finally:
        pulse.cancel()
        with suppress(asyncio.CancelledError):
            await pulse
        report["finished_at"] = datetime.now(UTC).isoformat()
        try:
            save_report(report, settings.evaluation_results_dir)
        finally:
            ACTIVE_RUNS.discard(run_id)
            ACTIVE_KBS.discard(str(kb_id))
    return report


def reviews_complete(report):
    """Strict review completion includes graph checks and answered cases."""
    graph_review = report.get("graph_review")
    if graph_review:
        if graph_review.get("status") != "complete" or graph_review.get("sampled"):
            return False
        if graph_review.get("reviewed_edges", 0) < graph_review.get("total_edges", 0):
            return False
    return all(
        row.get("refused") or row.get("answer_review", {}).get("status") == "complete"
        for variant in report["variants"].values()
        for row in variant["cases"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--knowledge-base-id", type=uuid.UUID, required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / DATASETS["engineering"])
    parser.add_argument("--strategies", nargs="+", default=["hybrid", "graph_hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    strategies = list(dict.fromkeys(args.strategies))
    for strategy in strategies:
        QueryRequest(question="validate", strategy=strategy, top_k=args.top_k)
    dataset = validate_dataset(json.loads(args.dataset.read_text(encoding="utf-8")))
    report = asyncio.run(
        execute_quality(
            args.knowledge_base_id,
            dataset,
            strategies,
            args.top_k,
            args.judge,
            baseline=json.loads(args.baseline.read_text("utf-8")) if args.baseline else None,
        )
    )
    print(json.dumps({"run_id": report["run_id"], "status": report["status"]}, ensure_ascii=False))
    if report["status"] != "completed":
        raise SystemExit(1)
    if (
        report.get("baseline_check", {}).get("status", "no_regression_found")
        != "no_regression_found"
    ):
        raise SystemExit(1)
    if args.strict and args.judge and not reviews_complete(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
