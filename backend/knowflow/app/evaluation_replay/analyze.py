from collections import Counter, defaultdict

from .common import ReplayError, identity
from .intervals import interval


def mean(values):
    return sum(values) / len(values) if values else None


def aggregate(context, ledger, *, stratum=None):
    by_metric = defaultdict(list)
    for row in ledger["samples"]:
        by_metric[row["group_id"], row["metric_id"], row.get(stratum) if stratum else None].append(
            row
        )
    contracts = {c["id"]: c for c in context["contracts"]}
    groups = {g["group_id"]: g for g in ledger["groups"]}
    results = []
    for (group, mid, stratum_value), rows in by_metric.items():
        counts = Counter(r["state"] for r in rows)
        reasons = Counter(r["primary_reason"] for r in rows if r["state"] == "unknown")
        observed = [r["value"] for r in rows if r["state"] == "observed"]
        eligible = counts["observed"] + counts["unknown"]
        assert len(rows) == eligible + counts["not_applicable"] + counts["applicability_unknown"]
        assert counts["unknown"] == sum(reasons.values())
        contract = contracts[mid]
        estimate = mean(observed)
        if mid == "verified_task_success" and len(observed) != len(rows):
            estimate = None
        observed_rows = [r for r in rows if r["state"] == "observed"]
        cluster_ids = [r["cluster_id"] for r in observed_rows]
        info = interval(observed, contract, groups=cluster_ids if any(cluster_ids) else None)
        incomplete = len(observed) < eligible or any(
            r["observation_coverage"] is not None and r["observation_coverage"] < 1
            for r in observed_rows
        )
        if incomplete and info["ci"] is not None:
            info.update(
                ci=None,
                method_effective="none",
                inferential_valid=False,
                reason="incomplete_observation_selection",
            )
        known = [r for r in rows if r["state"] == "observed" and r["degraded"] is not None]
        results.append(
            {
                "group_id": group,
                "dataset_id": groups[group]["dataset_id"],
                "dataset_version": groups[group]["dataset_version"],
                "stratum_field": stratum,
                "stratum_value": stratum_value,
                "contract": contract,
                "top_k": groups[group]["top_k"],
                "candidate_policy": groups[group]["candidate_policy"],
                "label_provenance": groups[group]["label_provenance"],
                "independence": groups[group]["independence"],
                "metric_id": mid,
                "view": "final",
                "metric_version": rows[0]["metric_version"],
                "contract_sha256": identity(contract),
                "mean": estimate,
                "observed_mean": mean(observed),
                "n_planned": len(rows),
                "n_eligible": eligible,
                "n_observed": counts["observed"],
                "n_unknown": counts["unknown"],
                "n_not_applicable": counts["not_applicable"],
                "n_applicability_unknown": counts["applicability_unknown"],
                "n_invalid_annotation": sum(r["invalid_annotation"] for r in rows),
                "n_attempted": sum(r["attempt_count"] > 0 for r in rows),
                "n_execution_failed_final": sum(r["last_attempt_status"] == "failed" for r in rows),
                "metric_coverage": len(observed) / eligible if eligible else None,
                "unknown_by_primary_reason": dict(reasons),
                "mean_including_known_degraded": mean([r["value"] for r in known]),
                "n_including_known_degraded": len(known),
                "mean_excluding_known_degraded": mean(
                    [r["value"] for r in known if not r["degraded"]]
                ),
                "n_excluding_known_degraded": sum(not r["degraded"] for r in known),
                "n_degradation_unknown": sum(r["degraded"] is None for r in rows),
                "interval": info,
                "warning": "EVIDENCE_INSUFFICIENT" if not info["inferential_valid"] else None,
            }
        )
    return results


def reliability(ledger):
    results = []
    for group in ledger["groups"]:
        plans = [p for p in ledger["plans"] if p["group_id"] == group["group_id"]]
        attempts = [a for a in ledger["attempts"] if a["group_id"] == group["group_id"]]
        n = len(plans)
        states = Counter(p["last_attempt_status"] for p in plans)
        any_unknown = sum(p["any_completed"] is None for p in plans)
        any_completed = sum(p["any_completed"] is True for p in plans)
        strict = sum(p["has_strict_attempt"] for p in plans)
        semantic = [
            s
            for s in ledger["samples"]
            if s["group_id"] == group["group_id"]
            and s["metric_id"] == "verified_task_success"
            and s["state"] == "observed"
        ]
        success = sum(s["value"] == 1 for s in semantic)

        def ratio(numerator, unknown, *, semantic_review=False, enabled=True, denominator=n):
            return {
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator if denominator and enabled else None,
                "n_unknown": unknown,
                "includes_semantic_review": semantic_review,
            }

        results.append(
            {
                **group,
                "n_planned": n,
                "n_attempts": len(attempts),
                "n_attempted_plans": sum(p["attempt_count"] > 0 for p in plans),
                "last_attempt_status_counts": dict(states),
                "n_attempts_failed": sum(a["status"] == "failed" for a in attempts),
                "n_degraded_attempts": sum(bool(a.get("degraded_events")) for a in attempts),
                "n_degradation_unknown": sum(
                    a.get("degradation_evidence_status") != "known" for a in attempts
                ),
                "strict_attempt_count": sum(p["strict_attempt_count"] for p in plans),
                "plan_completion_any": ratio(any_completed, any_unknown, enabled=n > any_unknown),
                "plan_completion_final": ratio(
                    states["completed"], states["unknown"], enabled=n > states["unknown"]
                ),
                "strict_coverage": {
                    **ratio(strict, sum(p["strict_eligibility_state"] == "unknown" for p in plans)),
                    "n_excluded": sum(p["strict_eligibility_state"] == "excluded" for p in plans),
                },
                "verified_task_success": ratio(
                    success,
                    n - len(semantic),
                    semantic_review=True,
                    enabled=n > 0 and len(semantic) == n,
                ),
                "verified_success_lower_bound": ratio(
                    success, n - len(semantic), semantic_review=True, enabled=bool(semantic)
                ),
                "retry_patterns": dict(Counter(p["retry_pattern"] for p in plans)),
                "note": "Execution/strict coverage is not answer correctness; exclusions "
                "are not execution failures. Unknown is retained.",
            }
        )
    return results


def comparisons(context, ledger):
    groups = {g["group_id"]: g for g in ledger["groups"]}
    contracts = {c["id"]: c for c in context["contracts"]}
    plans = {p["plan_id"]: p for p in ledger["plans"]}
    results = []
    strict_ids = {a["attempt_id"] for a in ledger["strict_rows"]}
    for spec in context["manifest"].get("comparisons", []):
        before, after = spec["before"], spec["after"]
        if before not in groups or after not in groups:
            raise ReplayError("Comparison refers to unknown group")
        profile = spec.get("profile", "strict_fixed_revision")
        if profile not in {"strict_fixed_revision", "observed_diagnostic"}:
            raise ReplayError("Unknown comparison profile")
        left, right = groups[before], groups[after]
        mismatch = []
        for key in (
            "top_k",
            "metrics_version",
            "corpus_sha256",
            "index_sha256",
            "comparison_fingerprint",
        ):
            if left.get(key) is None or right.get(key) is None:
                mismatch.append(f"{key}_unknown")
            elif left[key] != right[key]:
                mismatch.append(f"{key}_different")
        for mid in spec.get("metrics", contracts):
            if mid not in contracts:
                raise ReplayError(f"Comparison metric not declared: {mid}")
            lhs = {
                r["sample_key"]: r
                for r in ledger["samples"]
                if r["group_id"] == before and r["metric_id"] == mid
            }
            rhs = {
                r["sample_key"]: r
                for r in ledger["samples"]
                if r["group_id"] == after and r["metric_id"] == mid
            }
            shared = sorted(lhs.keys() & rhs.keys())
            exclusions, pairs = [], []
            for key in shared:
                a, b = lhs[key], rhs[key]
                reasons = list(mismatch)
                if not a["gold_sha256"] or not b["gold_sha256"]:
                    reasons.append("gold_identity_unknown")
                elif a["gold_sha256"] != b["gold_sha256"]:
                    reasons.append("gold_different")
                if a["cluster_id"] != b["cluster_id"]:
                    reasons.append("cluster_identity_different")
                if any(plans[r["plan_id"]]["attempt_count"] != 1 for r in (a, b)):
                    reasons.append("pairing_ambiguous_multiple_or_missing_attempts")
                if a["state"] != "observed" or b["state"] != "observed":
                    reasons.append("observation_unknown_or_inapplicable")
                if profile == "strict_fixed_revision" and any(
                    r["attempt_id"] not in strict_ids for r in (a, b)
                ):
                    reasons.append("strict_protocol_not_met")
                if reasons:
                    exclusions.append(
                        {
                            "sample_id": a["sample_id"],
                            "sample_key": key,
                            "before_attempt_id": a["attempt_id"],
                            "after_attempt_id": b["attempt_id"],
                            "reasons": reasons,
                        }
                    )
                else:
                    pairs.append(
                        {
                            "sample_id": a["sample_id"],
                            "sample_key": key,
                            "before": a["value"],
                            "after": b["value"],
                            "delta": b["value"] - a["value"],
                            "cluster_id": a["cluster_id"],
                        }
                    )
            differences = [r["delta"] for r in pairs]
            result = {
                "comparison_id": spec["id"],
                "metric_id": mid,
                "profile": profile,
                "before": before,
                "after": after,
                "n_shared": len(shared),
                "n_paired": len(pairs),
                "n_before_planned": len(lhs),
                "n_after_planned": len(rhs),
                "removed_sample_keys": sorted(lhs.keys() - rhs.keys()),
                "added_sample_keys": sorted(rhs.keys() - lhs.keys()),
                "status": "comparable" if pairs else "not_comparable",
                "delta": mean(differences),
                "pairs": pairs,
                "exclusions": exclusions,
                "global_incompatibilities": mismatch,
                "interval": interval(
                    differences,
                    contracts[mid],
                    paired=True,
                    groups=(
                        [p["cluster_id"] for p in pairs]
                        if any(p["cluster_id"] for p in pairs)
                        else None
                    ),
                ),
                "note": "Conditional paired intersection, not full planned performance. "
                "Observed diagnostics do not fix unpinned models or selection bias.",
            }
            results.append(result)
    return results
