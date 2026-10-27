"""Read declared measurements; never infer answer correctness from a retrieval score."""

from .common import ReplayError, number

STATES = {"observed", "unknown", "not_applicable", "applicability_unknown"}
UNKNOWN_REASONS = {
    "execution_missing_result",
    "execution_error",
    "execution_timeout",
    "retrieval_error",
    "retrieval_timeout",
    "generation_error",
    "generation_timeout",
    "review_timeout",
    "review_unavailable",
    "review_partial_missing_required",
    "review_not_observed",
    "annotation_missing_gold",
    "annotation_missing_semantic_review",
    "metric_not_recorded",
    "aggregate_only",
    "unrecoverable",
    "protocol_not_selected",
    "attempt_order_unrecoverable",
}


def measurement(case, attempt, contract, top_k=None, metrics_version=None):
    mid = contract["id"]
    result = {
        "metric_id": mid,
        "state": "unknown",
        "value": None,
        "primary_reason": "execution_missing_result",
        "secondary_flags": [],
        "observation_coverage": None,
        "source_field": None,
        "invalid_annotation": False,
    }
    if contract.get("source_versions") and metrics_version not in contract["source_versions"]:
        return {
            **result,
            "state": "not_applicable",
            "primary_reason": "metric_alias_version_not_compatible",
        }
    if contract.get("k") is not None and top_k != contract["k"]:
        return {**result, "state": "not_applicable", "primary_reason": "contract_k_not_selected"}
    gold = case.get("relevant_evidence")
    if contract["eligibility"] == "evidence":
        if case.get("should_refuse") is True and gold == []:
            return {**result, "state": "not_applicable", "primary_reason": "no_gold_refusal"}
        if not gold:
            return {
                **result,
                "state": "unknown"
                if case.get("should_refuse") is False
                else "applicability_unknown",
                "primary_reason": "annotation_missing_gold",
                "invalid_annotation": True,
            }
    if attempt is None:
        return result
    raw = attempt.get("raw", attempt)
    source = contract.get("source_metric", mid)
    explicit = attempt.get("metric_states", {}).get(mid, {})
    if mid == "service_execution_completed" and explicit:
        raise ReplayError("Service execution observation is derived exclusively from status")
    if explicit:
        if explicit.get("state") not in STATES:
            raise ReplayError("Unknown metric observation state")
        if explicit["state"] != "observed":
            reason = explicit.get("primary_reason")
            if not isinstance(reason, str) or not reason:
                raise ReplayError("Unobserved metric needs a single primary_reason")
            if explicit["state"] == "unknown" and reason not in UNKNOWN_REASONS:
                raise ReplayError(f"Unknown primary_reason enumeration: {reason}")
            flags = explicit.get("secondary_flags", [])
            if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
                raise ReplayError("secondary_flags must be a list of strings")
            return {
                **result,
                "state": explicit["state"],
                "primary_reason": reason,
                "secondary_flags": flags,
            }
    if mid == "service_execution_completed":
        value = (
            1.0
            if attempt["status"] == "completed"
            else 0.0
            if attempt["status"] in {"failed", "partial", "skipped"}
            else None
        )
        source = "status"
    elif mid == "verified_task_success":
        label = raw.get("semantic_verification", {})
        legitimate = (
            label.get("status") == "reviewed"
            and label.get("reviewer")
            and label.get("evidence")
            and isinstance(label.get("task_success"), bool)
        )
        value = float(label["task_success"]) if legitimate else None
        source = "semantic_verification.task_success"
        result["primary_reason"] = "annotation_missing_semantic_review"
    else:
        value = attempt.get("metrics", {}).get(source)
    if isinstance(value, bool):
        if contract.get("value_type") != "binary":
            raise ReplayError(f"Boolean metric needs an explicit binary contract: {mid}")
        value = float(value)
    if value is not None:
        if not number(value) or not 0 <= value <= 1:
            raise ReplayError(f"Invalid metric value in [0,1]: {mid}")
        if contract.get("value_type") == "binary" and value not in (0, 1):
            raise ReplayError(f"Nonbinary value for binary metric: {mid}")
        coverage = explicit.get("observation_coverage")
        if coverage is not None and (not number(coverage) or not 0 <= coverage <= 1):
            raise ReplayError("Invalid within-metric observation coverage")
        return {
            **result,
            "state": "observed",
            "value": value,
            "primary_reason": None,
            "observation_coverage": coverage,
            "source_field": source,
        }
    if explicit.get("state") == "observed":
        raise ReplayError("observed requires a legitimate stored value")
    if mid != "verified_task_success":
        review = raw.get("answer_review", {})
        result["primary_reason"] = (
            "review_partial_missing_required"
            if review.get("status") == "partial"
            else (
                attempt.get("primary_error")
                if attempt.get("primary_error") in UNKNOWN_REASONS
                else "execution_error"
            )
            if attempt["status"] == "failed"
            else "review_not_observed"
            if "estimate" in source
            else "metric_not_recorded"
        )
    return result
