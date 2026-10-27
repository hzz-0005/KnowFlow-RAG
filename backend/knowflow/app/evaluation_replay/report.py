from __future__ import annotations

import csv
import html
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import VERSION
from .analyze import aggregate, comparisons, reliability
from .common import ReplayError, digest, json_write, number, read_json
from .loader import load_manifest, verify_inputs
from .normalize import normalize

DISCLAIMER = (
    "本报告不含任务成功率。unknown ≠ 0，unknown ≠ 失败；主指标在本版仅为接口，"
    "历史题因缺少断言级标注不可评分。"
)


def _jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def _csv(path, rows):
    columns = sorted({key for row in rows for key in row}) or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=True, allow_nan=False)
                        if value is None or isinstance(value, (dict, list, bool))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _text(value):
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ")


def _display(value):
    return "unknown" if value is None else f"{value:.5f}"


def _quality_gates(path, context, metrics):
    if path is None:
        return {"enabled": False, "passed": None, "results": []}
    path = path.resolve()
    if not path.is_relative_to(context["root"]):
        raise ReplayError("Gate file must be inside manifest root")
    gates = read_json(path)
    if not isinstance(gates, dict) or gates.get("schema") != "knowflow.quality_gates.v1":
        raise ReplayError("Unknown quality gates schema")
    if not isinstance(gates.get("rules"), list) or not gates["rules"]:
        raise ReplayError("Explicit gates require nonempty rules")
    context["inputs"].append({"path": str(path), "sha256": digest(path), "role": "quality_gates"})
    by_key = {(m["group_id"], m["metric_id"]): m for m in metrics}
    results = []
    for rule in gates.get("rules", []):
        if not isinstance(rule, dict) or not any(
            k in rule for k in ("min_mean", "min_coverage", "max_unknown")
        ):
            raise ReplayError("Gate needs an explicit numeric condition")
        for key in ("min_mean", "min_coverage"):
            if key in rule and (not number(rule[key]) or not 0 <= rule[key] <= 1):
                raise ReplayError("Gate means/coverage must be in [0,1]")
        if "max_unknown" in rule and (
            type(rule["max_unknown"]) is not int or rule["max_unknown"] < 0
        ):
            raise ReplayError("Gate max_unknown must be a nonnegative integer")
        row = by_key.get((rule["group_id"], rule["metric_id"]))
        failed = []
        if row is None:
            failed.append("metric_missing")
        for field, key in (("min_mean", "mean"), ("min_coverage", "metric_coverage")):
            if field in rule and (row is None or row[key] is None or row[key] < rule[field]):
                failed.append(field)
        if "max_unknown" in rule and (row is None or row["n_unknown"] > rule["max_unknown"]):
            failed.append("max_unknown")
        results.append({"rule": rule, "passed": not failed, "failed_conditions": failed})
    return {"enabled": True, "passed": all(r["passed"] for r in results), "results": results}


def _markdown(out, context, ledger, metrics, rates, deltas):
    lines = [
        "# M1 历史评测账本（实现稿，待外审）",
        "",
        DISCLAIMER,
        "",
        "退出码0仅表示离线核算完成，不表示质量通过。下表为final诊断视图，"
        "全部attempt另存；完成比例和严格执行覆盖均不代表答案正确或前端送达。",
        "",
        "## 执行与覆盖",
        "",
        "| 运行/策略 | 执行范围 | planned | attempts | "
        "any完成 | final完成 | strict覆盖 | 语义主指标 |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rates:

        def fraction(key, record=row):
            r = record[key]
            return (
                f"{r['numerator']}/{r['denominator']} ({_display(r['value'])}; "
                f"unknown={r['n_unknown']})"
            )

        lines.append(
            f"| {_text(row['group_id'])} | {_text(row['execution_scope'])} | "
            f"{row['n_planned']} | {row['n_attempts']} | "
            f"{fraction('plan_completion_any')} | {fraction('plan_completion_final')} | "
            f"{fraction('strict_coverage')} | {_display(row['verified_task_success']['value'])} |"
        )
    lines += [
        "",
        "## 分层诊断：不是任务成功率",
        "",
        "| 运行/策略 | 指标 | 均值 | eligible | observed | unknown | N/A | 适用性未知 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    visible = {
        "base_evidence_recall",
        "candidates_evidence_recall",
        "context_evidence_recall",
        "excerpt_match",
        "source_cited",
        "faithfulness_estimate",
        "refusal_flag_matches_label",
        "verified_task_success",
    }
    for row in metrics:
        if row["metric_id"] not in visible:
            continue
        lines.append(
            f"| {_text(row['group_id'])} | {_text(row['metric_id'])} | "
            f"{_display(row['mean'])} | {row['n_eligible']} | {row['n_observed']} | "
            f"{row['n_unknown']} | {row['n_not_applicable']} | "
            f"{row['n_applicability_unknown']} |"
        )
    lines += [
        "",
        "## 题型分层",
        "",
        "按原始 question_type 分层，不猜测缺失题型；逐层分母与值见 metrics_by_question_type.csv。",
        "| 运行/策略 | 题型 | 计划槽位 |",
        "|---|---|---:|",
    ]
    counts = Counter((p["group_id"], p["question_type"]) for p in ledger["plans"])
    lines += [f"| {_text(group)} | {_text(kind)} | {n} |" for (group, kind), n in counts.items()]
    lines += [
        "",
        "小样本/相关单位/缺审核项保留EVIDENCE_INSUFFICIENT；"
        "不能从描述均值推出新文档总体能力。完整指标、原因与敏感度见metrics_long.csv。",
        "",
        "## 仅聚合分区",
        "",
        "以下输入仅按声明归档，不进入逐题表、配对差分或重算区间。",
    ]
    lines += [
        f"- `{_text(a['path'])}`"
        for a in context["artifacts"]
        if a["input_granularity"] == "aggregate_only"
    ]
    lines += [
        "",
        "## 尚未适配的逐题/组件输入",
        "",
        "这些输入保留原粒度，仅归档来源；本版不计入问答执行数。",
    ]
    lines += [
        f"- `{_text(a['path'])}`"
        for a in context["artifacts"]
        if a["input_granularity"] == "per_sample"
    ]
    lines += [
        "",
        "## 独立性与来源",
        "",
        "计划数、尝试数和唯一题目身份分开保存在inventory.json，"
        "唯一题目身份也不自动等于独立人工标注数。公开gold不证明未见盲测。",
        "CSV为UTF-8无BOM；null为字面null，ID按字符串读取。详见README-consume.md。",
    ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    delta_lines = [
        "# 比较与不可比较说明",
        "",
        "严格集为空是合法结果，不放宽版本/降级证据来出数。观测交集仅为条件诊断，排除不等于失败。",
        "",
    ]
    for d in deltas:
        delta_lines.append(
            f"- {_text(d['comparison_id'])}/{_text(d['metric_id'])}: "
            f"{d['status']}; paired={d['n_paired']}/{d['n_shared']}; "
            f"delta={_display(d['delta'])}; "
            f"incompatibilities={_text(d['global_incompatibilities'])}"
        )
    if not deltas:
        delta_lines.append("没有在manifest声明比较，不自动挑选一组给出提升。")
    delta_lines += ["", "逐题配对、排除理由、题目增删和区间方法见comparisons.json。"]
    (out / "delta.md").write_text("\n".join(delta_lines) + "\n", encoding="utf-8")


def replay(manifest: Path, out: Path, *, require_comparable=False, gates=None, argv=None):
    started = datetime.now(UTC).isoformat()
    code_files = sorted(Path(__file__).parent.glob("*.py"))
    code_files.append(Path(__file__).parent.parent / "evaluate_unified.py")
    implementation = [{"path": str(p), "sha256": digest(p)} for p in code_files]
    context = load_manifest(manifest)
    out = out.resolve()
    if out.exists():
        raise ReplayError("Output directory already exists; replay never overwrites a prior audit")
    for source in context["inputs"]:
        path = Path(source["path"])
        path = path if path.is_absolute() else context["root"] / path
        if path.resolve().is_relative_to(out):
            raise ReplayError("Output would contain/overwrite an input")
    ledger = normalize(context)
    metrics = aggregate(context, ledger)
    strata = aggregate(context, ledger, stratum="question_type")
    rates = reliability(ledger)
    deltas = comparisons(context, ledger)
    gate_result = _quality_gates(gates, context, metrics)
    code = (
        4
        if gate_result["enabled"] and not gate_result["passed"]
        else 3
        if require_comparable and (not deltas or any(d["status"] != "comparable" for d in deltas))
        else 0
    )
    verify_inputs(context)
    out.mkdir(parents=True)
    for name, rows in (
        ("plans", ledger["plans"]),
        ("attempts", ledger["attempts"]),
        ("samples", ledger["samples"]),
        ("attempt_metrics", ledger["attempt_metrics"]),
        ("strict_rows", ledger["strict_rows"]),
        ("exclusions", ledger["exclusions"]),
    ):
        _jsonl(out / f"{name}.jsonl", rows)
    _csv(out / "samples.csv", ledger["samples"])
    _csv(out / "metrics_long.csv", metrics)
    _csv(out / "metrics_by_question_type.csv", strata)
    _csv(out / "reliability.csv", rates)
    json_write(out / "metrics.json", metrics)
    json_write(out / "metrics_by_question_type.json", strata)
    json_write(out / "reliability.json", rates)
    json_write(out / "comparisons.json", deltas)
    json_write(out / "quality_gates.json", gate_result)
    unique = sorted({p["sample_key"] for p in ledger["plans"]})
    json_write(
        out / "inventory.json",
        {
            "schema": VERSION,
            "inputs": context["inputs"],
            "groups": ledger["groups"],
            "datasets": [d["spec"] for d in context["datasets"].values()],
            "aggregate_only": [
                a for a in context["artifacts"] if a["input_granularity"] == "aggregate_only"
            ],
            "per_sample_unadapted": [
                a for a in context["artifacts"] if a["input_granularity"] == "per_sample"
            ],
            "n_planned_slots": len(ledger["plans"]),
            "n_actual_attempts": len(ledger["attempts"]),
            "n_unique_question_identities": len(unique),
            "sample_keys": unique,
            "note": "Inventory count is not a claim of independent annotation or quality.",
        },
    )
    _markdown(out, context, ledger, metrics, rates, deltas)
    (out / "README-consume.md").write_text(
        "# 消费约定\n\nJSON/JSONL显式UTF-8，键名ASCII；CSV为UTF-8无BOM，null是字面null。\n"
        "PowerShell: Import-Csv -Encoding utf8；"
        "Get-Content -Raw -Encoding utf8 | ConvertFrom-Json。\n"
        "Python: python -X utf8，open(...,encoding='utf-8',newline='')，csv.DictReader。\n"
        "不要自动把ID转换为数字，不用split(',')读取字段。\n"
        "summary不含任务成功率；audit记录退出状态及全部输入/输出指纹。\n",
        encoding="utf-8",
    )
    stdout = f"M1 offline replay finished; exit_code={code}; quality acceptance not implied.\n"
    stderr = (
        "Requested comparison unavailable.\n"
        if code == 3
        else "Explicit quality gates did not pass.\n"
        if code == 4
        else ""
    )
    (out / "command.stdout.log").write_text(stdout, encoding="utf-8")
    (out / "command.stderr.log").write_text(stderr, encoding="utf-8")
    verify_inputs(context)
    if any(digest(Path(item["path"])) != item["sha256"] for item in implementation):
        raise ReplayError("Implementation changed while replaying; no valid audit produced")
    audit = {
        "schema": VERSION,
        "started_at": started,
        "completed_at": datetime.now(UTC).isoformat(),
        "argv": argv or [],
        "cwd": str(Path.cwd()),
        "manifest_root": str(context["root"]),
        "exit_code": code,
        "meaning": "Replay completion, not quality acceptance",
        "input_hashes_unchanged": True,
        "inputs": context["inputs"],
        "implementation": implementation,
        "outputs": [{"path": p.name, "sha256": digest(p)} for p in sorted(out.iterdir())],
        "provider_database_modules_loaded": [
            m
            for m in ("knowflow.app.db", "knowflow.app.providers", "openai", "httpx")
            if m in sys.modules
        ],
        "stdout": "command.stdout.log",
        "stderr": "command.stderr.log",
    }
    json_write(out / "audit.json", audit)
    (out / "audit.sha256").write_text(
        digest(out / "audit.json") + "  audit.json\n", encoding="ascii"
    )
    return {"exit_code": code, "stdout": stdout, "stderr": stderr, "out": str(out)}
