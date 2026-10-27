"""M1 accounting counterexamples; no service or provider is required."""

import builtins
import csv
import json
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from knowflow.app.evaluation_replay.analyze import aggregate, comparisons, reliability
from knowflow.app.evaluation_replay.annotations import (
    approved_evidence_complete,
    quantities_equal,
)
from knowflow.app.evaluation_replay.common import ReplayError, digest, read_json
from knowflow.app.evaluation_replay.intervals import clopper_pearson, interval, wilson
from knowflow.app.evaluation_replay.loader import load_manifest, verify_inputs
from knowflow.app.evaluation_replay.normalize import normalize
from knowflow.app.evaluation_replay.report import replay


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def case(sid="001", **updates):
    return {
        "id": sid,
        "question": '中文问题,"引号"\n换行 ' + sid,
        "should_refuse": False,
        "reference_answer": "example",
        "relevant_evidence": [{"id": "e", "contains": "x", "filename": "doc.pdf"}],
        **updates,
    }


def attempt(sid="001", aid="a1", status="completed", order=1, **updates):
    return {
        "sample_id": sid,
        "attempt_id": aid,
        "status": status,
        "order": order,
        "retry_of_attempt_id": None,
        "metrics": {"score": 0.9, "recall": 1.0},
        "degraded_events": [],
        "degradation_evidence_status": "known",
        "model_revision_pin": {"status": "pinned", "revision_id": "test-fixed-v1"},
        **updates,
    }


def bundle(tmp_path, *, cases=None, attempts=None, metrics=None, second=False, artifact=None):
    cases = cases if cases is not None else [case()]
    attempts = attempts if attempts is not None else [attempt()]
    metrics = metrics or [
        {"id": "score", "eligibility": "all", "interval_method": "none"},
        {"id": "recall", "eligibility": "evidence", "interval_method": "none"},
        {"id": "service_execution_completed", "eligibility": "all", "interval_method": "none"},
        {"id": "verified_task_success", "eligibility": "semantic", "interval_method": "none"},
    ]
    for m in metrics:
        m.setdefault("version", "v1")
    write(tmp_path / "dataset.json", {"version": "fixture-v1", "cases": cases})
    write(
        tmp_path / "contracts.json", {"schema": "knowflow.metric_contracts.v1", "metrics": metrics}
    )

    def source(name):
        return {"path": name, "sha256": digest(tmp_path / name)}

    runs = []
    for run_id in ["run1", "run2"] if second else ["run1"]:
        data = {
            "schema": "knowflow.replay_trace.v1",
            "run_id": run_id,
            "top_k": 5,
            "metrics_version": "fixture-v1",
            "corpus_sha256": "corpus",
            "index_sha256": "index",
            "attempts": [
                {
                    **deepcopy(a),
                    "attempt_id": run_id + "-" + a["attempt_id"],
                    "retry_of_attempt_id": (
                        run_id + "-" + a["retry_of_attempt_id"]
                        if a.get("retry_of_attempt_id")
                        else None
                    ),
                }
                for a in attempts
            ],
        }
        write(tmp_path / f"{run_id}.json", data)
        runs.append(
            {
                "id": run_id,
                **source(f"{run_id}.json"),
                "adapter": "ledger_v1",
                "input_granularity": "per_sample",
                "dataset_id": "fixture",
                "strategies": ["hybrid"],
                "comparison_fingerprint": "explicit-fixture-matched-budget",
            }
        )
    manifest = {
        "schema": "knowflow.m1.manifest.v1",
        "root": ".",
        "completion_views": ["any", "final"],
        "contracts": source("contracts.json"),
        "datasets": [{"id": "fixture", "family": "fixture", **source("dataset.json")}],
        "runs": runs,
        "artifacts": [],
        "comparisons": [],
    }
    if second:
        manifest["comparisons"] = [
            {
                "id": "paired",
                "before": "run1:hybrid",
                "after": "run2:hybrid",
                "profile": "strict_fixed_revision",
                "metrics": ["score"],
            }
        ]
    if artifact is not None:
        write(tmp_path / "aggregate.json", artifact)
        manifest["artifacts"] = [
            {
                **source("aggregate.json"),
                "adapter": "aggregate_json",
                "input_granularity": "aggregate_only",
            }
        ]
    write(tmp_path / "manifest.json", manifest)
    return tmp_path / "manifest.json"


def load(path):
    context = load_manifest(path)
    ledger = normalize(context)
    return context, ledger


def metric(context, ledger, mid):
    return next(r for r in aggregate(context, ledger) if r["metric_id"] == mid)


def test_r1_partial_coverage_is_not_cross_case_missing(tmp_path):
    rows = [
        attempt("A", "a", metrics={"score": 0.9}),
        attempt("B", "b", status="failed", metrics={}),
        attempt(
            "C",
            "c",
            metrics={"score": 0.6},
            metric_states={"score": {"state": "observed", "observation_coverage": 2 / 3}},
        ),
    ]
    context, ledger = load(bundle(tmp_path, cases=[case(i) for i in "ABC"], attempts=rows))
    score = metric(context, ledger, "score")
    assert (score["n_eligible"], score["n_observed"], score["n_unknown"]) == (3, 2, 1)
    assert score["mean"] == pytest.approx(0.75)
    assert score["metric_coverage"] == pytest.approx(2 / 3)
    assert metric(context, ledger, "service_execution_completed")["mean"] == pytest.approx(2 / 3)
    assert sum(score["unknown_by_primary_reason"].values()) == 1
    assert next(r for r in ledger["samples"] if r["sample_id"] == "B")["execution_failed"]


def test_failed_execution_retains_earlier_recall(tmp_path):
    context, ledger = load(
        bundle(tmp_path, attempts=[attempt(status="failed", metrics={"recall": 0})])
    )
    assert metric(context, ledger, "recall")["mean"] == 0
    assert metric(context, ledger, "score")["n_unknown"] == 1
    assert metric(context, ledger, "service_execution_completed")["mean"] == 0


def test_any_final_and_strict_attempt_views_are_distinct(tmp_path):
    rows = [
        attempt(aid="first"),
        attempt(aid="retry", order=2, status="failed", metrics={}, retry_of_attempt_id="first"),
    ]
    _, ledger = load(bundle(tmp_path, attempts=rows))
    r = reliability(ledger)[0]
    assert len(ledger["strict_rows"]) == 1  # earlier successful attempt stays strict
    assert r["plan_completion_any"]["value"] == 1
    assert r["plan_completion_final"]["value"] == 0
    assert r["strict_coverage"]["value"] == 1
    assert r["n_planned"] == 1 and r["n_attempts"] == 2
    assert r["verified_task_success"]["value"] is None
    assert r["verified_success_lower_bound"]["value"] is None


def test_r2_retry_lineage_unpinned_is_exclusion_not_failure(tmp_path):
    rows = [
        attempt("P1", "a", status="failed", metrics={}),
        attempt("P1", "b", order=2, retry_of_attempt_id="a"),
        attempt("P2", "c", model_revision_pin={"status": "unpinned", "reason": "alias"}),
    ]
    _, ledger = load(bundle(tmp_path, cases=[case("P1"), case("P2")], attempts=rows))
    r = reliability(ledger)[0]
    assert (r["n_planned"], r["n_attempts"]) == (2, 3)
    assert r["plan_completion_any"]["value"] == r["plan_completion_final"]["value"] == 1
    assert r["strict_coverage"]["value"] == 0.5
    assert r["last_attempt_status_counts"] == {"completed": 2}
    assert any(e["reason"] == "model_revision_unpinned" for e in ledger["exclusions"])
    assert next(a for a in ledger["attempts"] if a["attempt_id"].endswith("-b"))[
        "retry_of_attempt_id"
    ].endswith("-a")


@pytest.mark.parametrize("status", ["unknown", "skipped", "partial"])
def test_execution_states_retained(tmp_path, status):
    _, ledger = load(bundle(tmp_path, attempts=[attempt(status=status)]))
    assert ledger["plans"][0]["last_attempt_status"] == status
    assert len(ledger["attempts"]) == 1
    assert reliability(ledger)[0]["plan_completion_final"]["numerator"] == 0


def test_unknown_order_never_picks_best_attempt(tmp_path):
    _, ledger = load(
        bundle(
            tmp_path,
            attempts=[attempt(aid="a", order=1), attempt(aid="b", order=1, status="failed")],
        )
    )
    assert ledger["plans"][0]["last_attempt_status"] == "unknown"
    assert reliability(ledger)[0]["plan_completion_any"]["value"] == 1
    assert reliability(ledger)[0]["plan_completion_final"]["value"] is None


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "missing_parent",
        "cycle",
        "cross_plan",
        "bad_status",
        "stored_boolean",
        "fake_degradation",
    ],
)
def test_invalid_attempt_identity_is_rejected(tmp_path, failure):
    rows = [attempt("A", "a"), attempt("B", "b")]
    if failure == "duplicate":
        rows[1]["attempt_id"] = "a"
    elif failure == "missing_parent":
        rows[0]["retry_of_attempt_id"] = "absent"
    elif failure == "cycle":
        rows[0]["retry_of_attempt_id"] = "a"
    elif failure == "cross_plan":
        rows[1]["retry_of_attempt_id"] = "a"
    elif failure == "bad_status":
        rows[0]["status"] = "degraded"
    elif failure == "stored_boolean":
        rows[0]["execution_failed"] = False
    else:
        rows[0]["degraded_events"] = [
            {"stage": "model", "reason": "model_unpinned", "evidence": "x"}
        ]
    with pytest.raises(ReplayError):
        load(bundle(tmp_path, cases=[case("A"), case("B")], attempts=rows))


def test_no_result_is_unknown_and_empty_retrieval_is_observed_zero(tmp_path):
    context, ledger = load(
        bundle(
            tmp_path, cases=[case("A"), case("B")], attempts=[attempt("A", metrics={"recall": 0})]
        )
    )
    r = metric(context, ledger, "recall")
    assert (r["mean"], r["n_observed"], r["n_unknown"]) == (0, 1, 1)


def test_applicability_partition_and_missing_gold(tmp_path):
    cases = [
        case("refuse", should_refuse=True, relevant_evidence=[]),
        case("missing", relevant_evidence=[]),
        case("unlabeled", should_refuse=None, relevant_evidence=[]),
    ]
    context, ledger = load(bundle(tmp_path, cases=cases, attempts=[]))
    r = metric(context, ledger, "recall")
    assert (r["n_eligible"], r["n_not_applicable"], r["n_applicability_unknown"]) == (1, 1, 1)
    assert r["n_observed"] == 0 and r["mean"] is None and r["n_unknown"] == 1


def test_selected_protocol_is_not_missing_parent_case(tmp_path):
    path = bundle(tmp_path, cases=[case(i) for i in "ABC"], attempts=[])
    m = read_json(path)
    m["runs"][0]["selected_sample_ids"] = ["A", "B"]
    write(path, m)
    context, ledger = load(path)
    assert metric(context, ledger, "score")["n_planned"] == 2
    assert {p["sample_id"] for p in ledger["plans"]} == {"A", "B"}


def test_partial_label_does_not_create_missing_value(tmp_path):
    row = attempt(
        metrics={}, metric_states={"score": {"state": "observed", "observation_coverage": 2 / 3}}
    )
    with pytest.raises(ReplayError, match="legitimate"):
        load(bundle(tmp_path, attempts=[row]))


@pytest.mark.parametrize("value", [-0.01, 1.01, "0.7", True])
def test_invalid_metric_not_silently_coerced(tmp_path, value):
    with pytest.raises(ReplayError):
        load(bundle(tmp_path, attempts=[attempt(metrics={"score": value})]))


def test_binary_contract_accepts_legacy_bool(tmp_path):
    context, ledger = load(
        bundle(
            tmp_path,
            attempts=[attempt(metrics={"binary": False})],
            metrics=[{"id": "binary", "eligibility": "all", "value_type": "binary"}],
        )
    )
    assert metric(context, ledger, "binary")["mean"] == 0


def test_semantic_label_required_for_task_score(tmp_path):
    label = {
        "status": "reviewed",
        "reviewer": "fixture-human",
        "evidence": "annotation:1",
        "task_success": False,
    }
    context, ledger = load(bundle(tmp_path, attempts=[attempt(semantic_verification=label)]))
    assert metric(context, ledger, "verified_task_success")["mean"] == 0
    assert reliability(ledger)[0]["verified_success_lower_bound"]["value"] == 0


def test_aggregate_only_ids_never_become_sample_rows(tmp_path):
    path = bundle(
        tmp_path,
        artifact={
            "case_ids": ["fake-1", "fake-2"],
            "metrics": {"score": 0.5},
            "denominators": {"score": 2},
        },
    )
    context, ledger = load(path)
    assert len(ledger["plans"]) == 1
    assert context["artifacts"][0]["aggregate_fields"][0]["metrics"][0]["denominator"] == 2
    result = replay(path, tmp_path / "out")
    text = (Path(result["out"]) / "samples.jsonl").read_text(encoding="utf-8")
    assert "fake-1" not in text


def test_hash_mismatch_and_duplicate_json_keys_rejected(tmp_path):
    path = bundle(tmp_path)
    (tmp_path / "dataset.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="SHA256"):
        load_manifest(path)
    p = tmp_path / "duplicate.json"
    p.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ReplayError, match="Duplicate"):
        read_json(p)


def test_input_hash_rechecked_before_publication(tmp_path):
    context, _ = load(bundle(tmp_path))
    (tmp_path / "run1.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ReplayError, match="changed"):
        verify_inputs(context)


def test_comparison_empty_strict_subset_and_retry_ambiguity(tmp_path):
    path = bundle(
        tmp_path, second=True, attempts=[attempt(model_revision_pin={"status": "unknown"})]
    )
    context, ledger = load(path)
    r = comparisons(context, ledger)[0]
    assert r["n_paired"] == 0 and r["delta"] is None and r["exclusions"]


def test_version_budget_gold_and_sample_differences_are_visible(tmp_path):
    path = bundle(tmp_path, second=True)
    m = read_json(path)
    m["runs"][1]["comparison_fingerprint"] = "different-budget"
    write(path, m)
    context, ledger = load(path)
    assert (
        "comparison_fingerprint_different"
        in comparisons(context, ledger)[0]["global_incompatibilities"]
    )


def test_valid_paired_comparison_and_zero_empirical_range(tmp_path):
    c = [
        {
            "id": "score",
            "eligibility": "all",
            "interval_method": "paired_bootstrap",
            "interval_target": "fixed_set_paired_change",
            "independent_units": True,
        }
    ]
    context, ledger = load(bundle(tmp_path, second=True, metrics=c))
    result = comparisons(context, ledger)[0]
    assert result["n_paired"] == 1 and result["delta"] == 0
    assert result["interval"]["bootstrap_empirical_range"] == [0, 0]
    assert result["interval"]["ci"] is None
    assert result["interval"]["diagnostic"] == "NO_OBSERVED_DIFFERENCE"


def test_r3_methods_and_independent_units():
    assert wilson([1] * 10) == pytest.approx([0.7224672, 1], abs=1e-6)
    assert clopper_pearson([1] * 10) == pytest.approx([0.025**0.1, 1])
    assert clopper_pearson([0] * 10) == pytest.approx([0, 1 - 0.025**0.1])
    c = {"interval_method": "binomial_wilson", "independent_units": True}
    assert interval([1] * 10, c)["n_independent"] == 10
    assert interval([1] * 4, c, groups=["doc"] * 4)["ci"] is None
    assert interval([0.2, 0.3], c)["reason"] == "non_binary_values"
    assert interval([], c)["ci"] is None


def test_paired_and_cluster_bootstrap_contracts():
    c = {
        "interval_method": "paired_bootstrap",
        "independent_units": True,
        "interval_target": "fixed_set_paired_change",
        "seed": 42,
        "resamples": 1000,
    }
    result = interval([-0.5, 0.1, 0.3, 0.7], c, paired=True)
    assert result == interval([-0.5, 0.1, 0.3, 0.7], c, paired=True)
    assert result["n_paired"] == 4 and result["n_independent"] is None
    assert result["ci"][0] <= result["ci"][1]
    c["interval_method"] = "cluster_bootstrap"
    assert (
        interval([0, 0.5], c, paired=True, groups=["doc", "doc"])["reason"]
        == "insufficient_clusters"
    )
    c["bootstrap_algorithm"] = "BCa"
    assert interval([0, 0], c, paired=True)["ci"] is None


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        (("123.4", "亿元"), ("12340", "百万元"), True),
        (("123.4", "亿元"), ("1234", "百万元"), False),
        (("123.4", "亿元"), ("123.4", "万元"), False),
        (("-1.2", "元"), ("1.2", "元"), False),
        (("1", "%"), ("1", "ratio"), False),
    ],
)
def test_annotation_units_are_not_stripped(gold, pred, expected):
    assert (
        quantities_equal({"value": gold[0], "unit": gold[1]}, {"value": pred[0], "unit": pred[1]})
        is expected
    )


def test_explicit_tolerance_and_reviewed_evidence_groups():
    assert quantities_equal(
        {"value": 100, "unit": "元"}, {"value": 100.9, "unit": "元"}, relative_tolerance=0.01
    )
    assert not quantities_equal(
        {"value": 100, "unit": "元"}, {"value": 102, "unit": "元"}, relative_tolerance=0.01
    )
    assert not approved_evidence_complete([["A", "B"], ["C"]], ["A"])
    assert approved_evidence_complete([["A", "B"], ["C"]], ["C"])
    assert approved_evidence_complete([["A", "B"]], None) is None


def test_csv_roundtrip_and_output_integrity(tmp_path):
    path = bundle(tmp_path)
    replay(path, tmp_path / "out")
    with (tmp_path / "out/samples.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["sample_id"] == "001" and rows[0]["question"] == case()["question"]
    audit = read_json(tmp_path / "out/audit.json")
    for record in audit["outputs"]:
        assert digest(tmp_path / "out" / record["path"]) == record["sha256"]
    with pytest.raises(ReplayError, match="exists"):
        replay(path, tmp_path / "out")


def test_offline_replay_never_imports_runtime_or_uses_network(tmp_path, monkeypatch):
    path = bundle(tmp_path)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"knowflow.app.db", "knowflow.app.providers", "openai", "httpx", "sqlalchemy"}:
            raise AssertionError("Runtime/provider import forbidden")
        return original_import(name, *args, **kwargs)

    def denied(*args, **kwargs):
        raise AssertionError("Network forbidden")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(socket.socket, "connect", denied)
    replay(path, tmp_path / "out")


def test_cli_exit_states_and_no_provider_modules(tmp_path):
    path = bundle(
        tmp_path, second=True, attempts=[attempt(model_revision_pin={"status": "unknown"})]
    )
    argv = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "knowflow.app.evaluate_unified",
        "--manifest",
        str(path),
    ]
    normal = subprocess.run([*argv, "--out", str(tmp_path / "normal")], capture_output=True)
    assert normal.returncode == 0, normal.stderr
    audit = read_json(tmp_path / "normal/audit.json")
    assert audit["provider_database_modules_loaded"] == []
    assert Path(audit["manifest_root"]) == tmp_path
    assert "-X" in audit["argv"] and "utf8" in audit["argv"]
    strict = subprocess.run(
        [*argv, "--out", str(tmp_path / "strict"), "--require-comparable"], capture_output=True
    )
    assert strict.returncode == 3
    write(
        tmp_path / "gates.json",
        {
            "schema": "knowflow.quality_gates.v1",
            "rules": [{"group_id": "run1:hybrid", "metric_id": "score", "min_mean": 0.99}],
        },
    )
    gate = subprocess.run(
        [*argv, "--out", str(tmp_path / "gate"), "--gates", str(tmp_path / "gates.json")],
        capture_output=True,
    )
    assert gate.returncode == 4
    (tmp_path / "dataset.json").write_text("{}", encoding="utf-8")
    invalid = subprocess.run([*argv, "--out", str(tmp_path / "bad")], capture_output=True)
    assert invalid.returncode == 2 and not (tmp_path / "bad").exists()


def test_known_exclusion_is_not_unknown(tmp_path):
    context, ledger = load(
        bundle(
            tmp_path,
            attempts=[
                attempt(model_revision_pin={"status": "unpinned", "reason": "mutable alias"})
            ],
        )
    )
    rate = reliability(ledger)[0]["strict_coverage"]
    assert (rate["numerator"], rate["n_unknown"], rate["n_excluded"]) == (0, 0, 1)
    assert rate["value"] == 0
    assert metric(context, ledger, "verified_task_success")["mean"] is None


def test_service_metric_cannot_override_execution_status(tmp_path):
    with pytest.raises(ReplayError, match="exclusively from status"):
        load(
            bundle(
                tmp_path,
                attempts=[
                    attempt(
                        status="failed",
                        metric_states={
                            "service_execution_completed": {
                                "state": "unknown",
                                "primary_reason": "x",
                            }
                        },
                    )
                ],
            )
        )


def test_binary_contract_rejects_fraction(tmp_path):
    with pytest.raises(ReplayError, match="Nonbinary"):
        load(
            bundle(
                tmp_path,
                attempts=[attempt(metrics={"binary": 0.5})],
                metrics=[{"id": "binary", "eligibility": "all", "value_type": "binary"}],
            )
        )


def test_missing_gold_identity_does_not_match_itself(tmp_path):
    context, ledger = load(bundle(tmp_path, second=True))
    for row in ledger["samples"]:
        row["gold_sha256"] = None
    result = comparisons(context, ledger)[0]
    assert result["n_paired"] == 0
    assert "gold_identity_unknown" in result["exclusions"][0]["reasons"]


def test_partial_observation_is_value_but_not_full_interval(tmp_path):
    context, ledger = load(
        bundle(
            tmp_path,
            attempts=[
                attempt(
                    metrics={"binary": 1},
                    metric_states={"binary": {"state": "observed", "observation_coverage": 2 / 3}},
                )
            ],
            metrics=[
                {
                    "id": "binary",
                    "eligibility": "all",
                    "value_type": "binary",
                    "interval_method": "binomial_wilson",
                    "independent_units": True,
                }
            ],
        )
    )
    row = metric(context, ledger, "binary")
    assert row["n_observed"] == 1 and row["mean"] == 1
    assert row["interval"]["ci"] is None
    assert row["interval"]["method_effective"] == "none"


def test_document_cluster_blocks_binary_interval(tmp_path):
    context, ledger = load(
        bundle(
            tmp_path,
            cases=[case(cluster_id="doc")],
            attempts=[attempt(metrics={"binary": 1})],
            metrics=[
                {
                    "id": "binary",
                    "eligibility": "all",
                    "value_type": "binary",
                    "interval_method": "binomial_wilson",
                    "independent_units": True,
                }
            ],
        )
    )
    row = metric(context, ledger, "binary")
    assert row["interval"]["n_clusters"] == 1
    assert row["interval"]["ci"] is None
    c = {"interval_method": "cluster_bootstrap", "interval_target": "fixed_set_paired_change"}
    assert (
        interval([0, 1], c, paired=True, groups=["doc", None])["reason"]
        == "cluster_identity_missing"
    )


def test_contract_k_and_version_alias_are_not_silently_merged(tmp_path):
    context, ledger = load(
        bundle(
            tmp_path,
            metrics=[
                {"id": "recall", "eligibility": "evidence", "k": 10},
                {
                    "id": "excerpt_match",
                    "source_metric": "recall",
                    "eligibility": "evidence",
                    "source_versions": ["different-version"],
                },
            ],
        )
    )
    assert metric(context, ledger, "recall")["n_not_applicable"] == 1
    assert metric(context, ledger, "excerpt_match")["mean"] is None


@pytest.mark.parametrize("key", ["contracts", "dataset", "runs", "granularity", "escape"])
def test_loader_rejects_invalid_shapes_and_escape(tmp_path, key):
    path = bundle(tmp_path)
    m = read_json(path)
    if key in {"contracts", "dataset"}:
        ref = m["contracts"] if key == "contracts" else m["datasets"][0]
        write(tmp_path / ref["path"], [])
        ref["sha256"] = digest(tmp_path / ref["path"])
    elif key == "runs":
        m["runs"] = [None]
    elif key == "granularity":
        m["runs"][0]["input_granularity"] = "aggregate_only"
    else:
        m["contracts"]["path"] = "../contracts.json"
    write(path, m)
    with pytest.raises(ReplayError):
        load_manifest(path)


def test_strata_and_unadapted_inputs_are_distinct(tmp_path):
    path = bundle(tmp_path, artifact={"metrics": {"score": 0.5}})
    m = read_json(path)
    m["artifacts"][0].update(input_granularity="per_sample", adapter="inventory_only")
    write(path, m)
    replay(path, tmp_path / "out")
    inv = read_json(tmp_path / "out/inventory.json")
    assert inv["aggregate_only"] == [] and len(inv["per_sample_unadapted"]) == 1
    strata = read_json(tmp_path / "out/metrics_by_question_type.json")
    assert all(r["stratum_value"] == "unknown" for r in strata)
    assert "尚未适配的逐题/组件输入" in (tmp_path / "out/summary.md").read_text(encoding="utf-8")


def test_frozen_historical_stage_metrics_and_missing_semantic_labels():
    root = Path(__file__).resolve().parents[2]
    if (
        not (root / "evaluation/unified_v2/manifest.json").is_file()
        and (root / "SOURCE_MANIFEST.json").is_file()
    ):
        pytest.skip("Private historical runs not distributed; synthetic M1 accounting tests run")
    context = load_manifest(root / "evaluation/unified_v2/manifest.json")
    ledger = normalize(context)
    rows = {(r["group_id"], r["metric_id"]): r for r in aggregate(context, ledger)}
    stemming = "1bebcf0f-85e8-4ca9-a6e3-93585462a9f5:hybrid"
    for mid, expected in {
        "base_evidence_recall": 0,
        "candidates_evidence_recall": 1,
        "context_evidence_recall": 1,
        "excerpt_match": 0,
        "source_cited": 1,
    }.items():
        assert rows[stemming, mid]["mean"] == expected
        assert rows[stemming, mid]["n_observed"] == 2
    graph = rows["32ebb087-4a14-4aaa-b435-5b24a895a52b:graph_hybrid", "faithfulness_estimate"]
    assert (graph["n_planned"], graph["n_observed"], graph["n_unknown"]) == (4, 2, 2)
    assert sum(graph["unknown_by_primary_reason"].values()) == 2
    assert ledger["strict_rows"] == []
    assert all(r["verified_task_success"]["value"] is None for r in reliability(ledger))
    assert all(r["n_paired"] == 0 for r in comparisons(context, ledger))
    selected = [
        p
        for p in ledger["plans"]
        if p["run_id"]
        in {
            "b68098cf-37e0-4c3d-a886-f681de44afcc",
            "a3919141-b786-4b27-ad54-adb9dc144316",
            "1bebcf0f-85e8-4ca9-a6e3-93585462a9f5",
            "protocol:real-pdf-repair",
        }
    ]
    # The protocol name is manifest data; inspect all explicitly planned-only slots as well.
    planned_only = {
        r["spec"]["id"] for r in context["runs"] if r["spec"]["adapter"] == "planned_only"
    }
    selected += [p for p in ledger["plans"] if p["run_id"] in planned_only and p not in selected]
    assert len(selected) == 8 and len({p["sample_key"] for p in selected}) == 3
    assert sum(p["attempt_count"] for p in selected) == 7
    verify_inputs(context)
