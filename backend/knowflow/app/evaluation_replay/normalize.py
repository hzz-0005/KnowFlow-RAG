from __future__ import annotations

from .common import ReplayError, identity, number
from .observations import measurement

STATUSES = {"completed", "partial", "failed", "skipped", "unknown"}


def _historical_attempt(row, run, strategy, position):
    response = row.get("response")
    # The writer persists response only after run_query returned; a subsequent judge
    # error is not proof that the query failed. Keep the two phases distinct.
    if run["spec"]["adapter"] == "retrieval_run" and isinstance(row.get("retrieved"), list):
        status = "completed"
    elif isinstance(response, dict) and isinstance(response.get("refused"), bool):
        status = "completed"
    elif row.get("error"):
        status = "failed" if not row.get("metrics") else "unknown"
    else:
        status = "unknown"
    return {
        "attempt_id": f"{run['spec']['id']}:{strategy}:{position}",
        "sample_id": row["id"],
        "status": status,
        "retry_of_attempt_id": None,
        "order": position,
        "metrics": row.get("metrics", {}),
        "source_phase": "query_response" if response else "legacy_trace",
        "primary_error": "execution_error" if row.get("error") else None,
        "degraded_events": [],
        "degradation_evidence_status": "unknown",
        "model_revision_pin": {"status": "unknown", "reason": "historical_revision_unpinned"},
        "raw": row,
        "source_path": (
            f"cases[{position}]"
            if run["spec"]["adapter"] == "retrieval_run"
            else f"variants.{strategy}.cases[{position}]"
        ),
    }


def _validate_attempts(attempts, selected):
    by_id = {}
    for row in attempts:
        if not isinstance(row, dict):
            raise ReplayError("Attempt must be an object")
        if "execution_failed" in row:
            raise ReplayError("execution_failed is derived, not a stored truth source")
        aid = row.get("attempt_id")
        if not isinstance(aid, str) or not aid or aid in by_id:
            raise ReplayError("Duplicate/missing attempt_id")
        if row.get("sample_id") not in selected or row.get("status") not in STATUSES:
            raise ReplayError(f"Unknown selected sample/status: {aid}")
        if not isinstance(row.get("degraded_events", []), list):
            raise ReplayError("degraded_events must be a list")
        if row.get("degradation_evidence_status", "unknown") not in {"known", "unknown"}:
            raise ReplayError("Unknown degradation evidence status")
        for event in row.get("degraded_events", []):
            if not isinstance(event, dict) or not all(
                event.get(k) for k in ("stage", "reason", "evidence")
            ):
                raise ReplayError("Degradation event needs stage, reason, evidence")
            if event["reason"] == "model_unpinned":
                raise ReplayError("Model version constraint is not an actual degradation event")
        pin = row.get("model_revision_pin", {"status": "unknown"})
        if not isinstance(pin, dict) or pin.get("status") not in {"pinned", "unpinned", "unknown"}:
            raise ReplayError("Invalid model revision status")
        if pin.get("status") == "pinned" and not pin.get("revision_id"):
            raise ReplayError("Pinned version needs revision_id")
        if not isinstance(row.get("metrics", {}), dict):
            raise ReplayError("Attempt metrics must be an object")
        states = row.get("metric_states", {})
        if not isinstance(states, dict) or any(not isinstance(s, dict) for s in states.values()):
            raise ReplayError("Metric states must contain objects")
        for metric, value in row.get("metrics", {}).items():
            if value is not None and not number(value) and not isinstance(value, bool):
                raise ReplayError(f"Invalid metric value: {aid}/{metric}")
        by_id[aid] = row
    for row in attempts:
        seen, parent = {row["attempt_id"]}, row.get("retry_of_attempt_id")
        while parent:
            if parent not in by_id or parent in seen:
                raise ReplayError("Missing/cyclic retry lineage")
            ancestor = by_id[parent]
            if ancestor["sample_id"] != row["sample_id"]:
                raise ReplayError("Retry lineage crosses sample/plan identity")
            if (
                number(row.get("order"))
                and number(ancestor.get("order"))
                and ancestor["order"] >= row["order"]
            ):
                raise ReplayError("Retry parent must precede its child")
            seen.add(parent)
            parent = ancestor.get("retry_of_attempt_id")


def terminal(attempts):
    if not attempts:
        return None, "not_attempted"
    if len(attempts) == 1:
        return attempts[0], None
    orders = [row.get("order") for row in attempts]
    if any(not number(x) for x in orders) or len(set(orders)) != len(orders):
        return None, "attempt_order_unrecoverable"
    return max(attempts, key=lambda row: row["order"]), None


def strict_reasons(attempt):
    if attempt is None:
        return ["no_known_terminal_attempt"]
    reasons = []
    if attempt["status"] != "completed":
        reasons.append("execution_not_completed")
    if attempt.get("degradation_evidence_status", "unknown") != "known":
        reasons.append("degradation_unknown")
    if attempt.get("degraded_events"):
        reasons.append("actual_degradation")
    pin = attempt.get("model_revision_pin", {}).get("status", "unknown")
    if pin != "pinned":
        reasons.append("model_revision_unpinned" if pin == "unpinned" else "model_revision_unknown")
    return reasons


def strict_state(attempts):
    """Separate known exclusion from missing evidence; neither is an execution failure."""
    if any(not strict_reasons(a) for a in attempts):
        return "eligible"
    for a in attempts:
        known_exclusion = (
            a["status"] in {"failed", "partial", "skipped"}
            or bool(a.get("degraded_events"))
            or a.get("model_revision_pin", {}).get("status") == "unpinned"
        )
        if not known_exclusion:
            return "unknown"
    return "excluded" if attempts else "unknown"


def _group(run, strategy):
    spec, data, ds = run["spec"], run["data"], run["dataset"]
    return {
        "group_id": f"{spec['id']}:{strategy}",
        "run_id": spec["id"],
        "strategy": strategy,
        "dataset_id": spec["dataset_id"],
        "execution_scope": "retrieval_only" if spec["adapter"] == "retrieval_run" else "query",
        "dataset_family": ds["spec"].get("family", spec["dataset_id"]),
        "top_k": data.get("top_k"),
        "metrics_version": data.get("metrics_version") or "legacy",
        "candidate_policy": data.get("pipeline"),
        "models": data.get("models"),
        "answer_prompt_version": data.get("answer_prompt_version"),
        "corpus_sha256": data.get("corpus_sha256"),
        "index_sha256": data.get("index_sha256"),
        "comparison_fingerprint": spec.get("comparison_fingerprint"),
        "label_origin": ds["spec"].get("label_origin", "unknown"),
        "independence": ds["spec"].get("independence", "not_established"),
        "dataset_version": ds["data"].get("version"),
        "dataset_sha256": ds["spec"]["sha256"],
        "source_artifact": spec["path"],
        "source_sha256": spec["sha256"],
        "input_granularity": spec["input_granularity"],
        "recoverability": "per_sample" if spec["adapter"] != "planned_only" else "planned_only",
        "protocol_metadata": {
            k: spec.get(k)
            for k in (
                "selected_sample_ids",
                "selection_reason",
                "parent_dataset_version",
                "frozen_at",
            )
        },
        "label_provenance": {
            k: ds["spec"].get(k)
            for k in (
                "label_origin",
                "developer_authored",
                "seen_before_tuning",
                "blinded_scoring",
                "heldout_protocol",
                "subset_selection",
            )
        },
        "runtime_provenance": {
            k: data.get(k)
            for k in (
                "config_sha256",
                "expansion_budget",
                "created_at",
                "code_sha256",
                "prompt_versions",
            )
        },
    }


def normalize(context):
    plans, attempts_out, samples, attempt_metrics = [], [], [], []
    exclusions, groups, strict_rows, seen_attempts = [], [], [], set()
    for run in context["runs"]:
        spec, data, dataset = run["spec"], run["data"], run["dataset"]
        for strategy in run["strategies"]:
            group = _group(run, strategy)
            groups.append(group)
            adapter = spec["adapter"]
            if adapter in {"rag_quality", "retrieval_run"}:
                rows = (
                    data["cases"]
                    if adapter == "retrieval_run"
                    else data["variants"].get(strategy, {}).get("cases", [])
                )
                if len({r.get("id") for r in rows}) != len(rows):
                    raise ReplayError("Repeated legacy case without retry lineage")
                for row in rows:
                    case = dataset["cases"].get(row.get("id"))
                    if case is None or row.get("question") != case["question"]:
                        raise ReplayError("Historical case question/identity mismatch")
                attempts = [_historical_attempt(r, run, strategy, i) for i, r in enumerate(rows)]
            elif adapter == "ledger_v1":
                if len(run["strategies"]) > 1 and any(
                    "strategy" not in a for a in data.get("attempts", [])
                ):
                    raise ReplayError("Multi-strategy attempts must declare strategy")
                attempts = [
                    {**a, "source_path": f"attempts[{i}]"}
                    for i, a in enumerate(data.get("attempts", []))
                    if a.get("strategy", strategy) == strategy
                ]
            else:
                attempts = []
            _validate_attempts(attempts, run["selected"])
            for a in attempts:
                if a["attempt_id"] in seen_attempts:
                    raise ReplayError("attempt_id must be globally unique")
                seen_attempts.add(a["attempt_id"])
            for sid in run["selected"]:
                case = dataset["cases"][sid]
                selected = [a for a in attempts if a["sample_id"] == sid]
                chosen, reason = terminal(selected)
                pid = identity([group["group_id"], sid])
                ever = (
                    True
                    if any(a["status"] == "completed" for a in selected)
                    else False
                    if selected and all(a["status"] != "unknown" for a in selected)
                    else None
                )
                strict = [a for a in selected if not strict_reasons(a)]
                plan = {
                    "plan_id": pid,
                    "group_id": group["group_id"],
                    "sample_id": sid,
                    "sample_key": identity([group["dataset_family"], sid, case["question"]]),
                    "question_sha256": identity(case["question"]),
                    "gold_sha256": (
                        None
                        if adapter == "retrieval_run" and not data.get("dataset_sha256")
                        else identity(
                            {
                                k: case.get(k)
                                for k in (
                                    "relevant_evidence",
                                    "should_refuse",
                                    "reference_answer",
                                    "required_facts",
                                )
                            }
                        )
                    ),
                    "question": case["question"],
                    "cluster_id": case.get("cluster_id"),
                    "execution_scope": group["execution_scope"],
                    "question_type": case.get("question_type", "unknown"),
                    "run_id": spec["id"],
                    "strategy": strategy,
                    "dataset_id": spec["dataset_id"],
                    "dataset_version": group["dataset_version"],
                    "input_granularity": spec["input_granularity"],
                    "last_attempt_id": chosen["attempt_id"] if chosen else None,
                    "last_attempt_status": chosen["status"] if chosen else "unknown",
                    "attempt_count": len(selected),
                    "terminal_reason": reason,
                    "any_completed": ever,
                    "has_strict_attempt": bool(strict),
                    "strict_eligibility_state": strict_state(selected),
                    "strict_attempt_count": len(strict),
                    "retry_pattern": (
                        "completed_then_final_not_completed"
                        if ever and chosen and chosen["status"] != "completed"
                        else "failed_then_completed"
                        if chosen
                        and chosen["status"] == "completed"
                        and any(a["status"] == "failed" for a in selected)
                        else "other"
                    ),
                }
                plans.append(plan)
                for a in selected:
                    cleaned = {k: v for k, v in a.items() if k != "raw"}
                    record = {
                        **cleaned,
                        "plan_id": pid,
                        "group_id": group["group_id"],
                        "sample_key": plan["sample_key"],
                        "run_id": spec["id"],
                        "source_artifact": spec["path"],
                    }
                    attempts_out.append(record)
                    reasons = strict_reasons(a)
                    if not reasons:
                        strict_rows.append(record)
                    for why in reasons:
                        exclusions.append(
                            {
                                "plan_id": pid,
                                "sample_id": sid,
                                "run_id": spec["id"],
                                "attempt_id": a["attempt_id"],
                                "reason": why,
                            }
                        )
                    for c in context["contracts"]:
                        attempt_metrics.append(
                            _sample(plan, a, c, case, group, spec, "each_attempt")
                        )
                for c in context["contracts"]:
                    samples.append(_sample(plan, chosen, c, case, group, spec, "final"))
    return {
        "plans": plans,
        "attempts": attempts_out,
        "samples": samples,
        "attempt_metrics": attempt_metrics,
        "exclusions": exclusions,
        "groups": groups,
        "strict_rows": strict_rows,
    }


def _sample(plan, attempt, contract, case, group, spec, view):
    observed = measurement(case, attempt, contract, group["top_k"], group["metrics_version"])
    if contract.get("requires_scope") == "query" and group["execution_scope"] != "query":
        observed.update(state="not_applicable", value=None, primary_reason="retrieval_only_scope")
    return {
        **plan,
        **observed,
        "view": view,
        "attempt_id": attempt["attempt_id"] if attempt else None,
        "execution_failed": attempt["status"] == "failed" if attempt else None,
        "source_artifact": spec["path"],
        "source_row": attempt.get("source_path", "attempts") if attempt else None,
        "metric_version": f"{group['metrics_version']}:{contract.get('version', 'v1')}",
        "degraded": (
            bool(attempt.get("degraded_events"))
            if attempt and attempt.get("degradation_evidence_status") == "known"
            else None
        ),
    }
