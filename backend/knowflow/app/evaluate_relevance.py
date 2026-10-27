"""Evaluate retrieval relevance decisions against fixed, independent pair labels."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from knowflow.app.config import get_settings
from knowflow.app.relevance import ACCEPTED_VERDICTS, PROMPT_VERSION, assess_retrieval

ROOT = Path(__file__).resolve().parents[3]


def summarize_decisions(decisions: list[dict]) -> dict:
    tp = sum(row["expected_usable"] and row["accepted"] for row in decisions)
    fp = sum(not row["expected_usable"] and row["accepted"] for row in decisions)
    fn = sum(row["expected_usable"] and not row["accepted"] for row in decisions)
    tn = sum(not row["expected_usable"] and not row["accepted"] for row in decisions)
    return {
        "pairs": len(decisions),
        "true_accepts": tp,
        "false_accepts": fp,
        "false_rejects": fn,
        "true_rejects": tn,
        "accepted_precision": tp / (tp + fp) if tp + fp else None,
        "relevant_recall": tp / (tp + fn) if tp + fn else None,
        "decision_accuracy": (tp + tn) / len(decisions) if decisions else None,
        "unknown_count": sum(row["verdict"] == "unknown" for row in decisions),
    }


def compare_regressions(report: dict, baseline: dict) -> list[str]:
    if report["dataset_sha256"] != baseline["dataset_sha256"]:
        raise ValueError("Baseline uses a different dataset; compare fixed inputs only")
    before = {row["id"]: row for row in baseline["decisions"]}
    return [
        row["id"]
        for row in report["decisions"]
        if before.get(row["id"], {}).get("passed") and not row["passed"]
    ]


async def run(dataset_path: Path) -> dict:
    dataset_bytes = dataset_path.read_bytes()
    dataset = json.loads(dataset_bytes)
    settings = get_settings()
    if not settings.retrieval_assessment_enabled:
        raise ValueError("Enable RETRIEVAL_ASSESSMENT_ENABLED before evaluating")
    semaphore = asyncio.Semaphore(2)

    async def evaluate_case(case):
        inputs = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"knowflow:{case['id']}:{row['id']}")),
                "content": row["content"],
            }
            for row in case["passages"]
        ]
        async with semaphore:
            assessments, diagnostics = await assess_retrieval(case["question"], inputs, settings)
        rows = []
        for gold, source in zip(case["passages"], inputs, strict=True):
            assessment = assessments[source["id"]]
            accepted = assessment.verdict in ACCEPTED_VERDICTS
            rows.append(
                {
                    "id": f"{case['id']}/{gold['id']}",
                    "question": case["question"],
                    "expected_usable": gold["expected_usable"],
                    "accepted": accepted,
                    "passed": accepted == gold["expected_usable"],
                    "verdict": assessment.verdict,
                    "expected_verdict": gold.get("expected_verdict"),
                    "verdict_matches": assessment.verdict == gold["expected_verdict"]
                    if "expected_verdict" in gold
                    else None,
                    "reason": assessment.reason,
                    "supporting_quotes": assessment.supporting_quotes,
                    "assessment_status": diagnostics.status,
                }
            )
        print(
            f"完成：{case['id']}，{sum(row['passed'] for row in rows)}/{len(rows)} "
            "个采纳判断符合标注"
        )
        return rows

    groups = await asyncio.gather(*(evaluate_case(case) for case in dataset["cases"]))
    decisions = [row for group in groups for row in group]
    return {
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(UTC).isoformat(),
        "kind": "retrieval_relevance_regression",
        "dataset_version": dataset["version"],
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "dataset_source": dataset["source"],
        "assessor_model": settings.llm_model,
        "prompt_version": PROMPT_VERSION,
        "metrics": summarize_decisions(decisions),
        "probability_calibration": "not_run; no confidence probabilities are emitted",
        "decisions": decisions,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evaluation/relevance_cases.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--strict", action="store_true", help="Exit nonzero on any mislabeled acceptance"
    )
    args = parser.parse_args()
    report = asyncio.run(run(args.dataset))
    regressions = (
        compare_regressions(report, json.loads(args.baseline.read_text(encoding="utf-8")))
        if args.baseline
        else []
    )
    report["regressions"] = regressions
    output = args.output or ROOT / f"evaluation/results/relevance-{report['run_id']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["metrics"]
    print(
        f"片段对共 {metrics['pairs']} 个，误采纳 {metrics['false_accepts']} 个，"
        f"漏采纳 {metrics['false_rejects']} 个，未核验 {metrics['unknown_count']} 个。"
    )
    print(f"报告：{output}；此结果不是线上准确率或概率校准结果。")
    if regressions or (args.strict and not all(row["passed"] for row in report["decisions"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
