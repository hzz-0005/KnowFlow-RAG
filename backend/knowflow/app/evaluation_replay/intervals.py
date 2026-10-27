"""Declared interval methods, without numerical/network dependencies."""

from __future__ import annotations

import math
import random
from statistics import NormalDist

from .common import ReplayError, number


def wilson(values, confidence=0.95):
    if not values or any(v not in (0, 1) for v in values):
        raise ReplayError("Wilson requires nonempty binary observations")
    n, p = len(values), sum(values) / len(values)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def _cdf(k, n, p):
    if p == 0:
        return 1.0
    if p == 1:
        return float(k >= n)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(i + 1)
        - math.lgamma(n - i + 1)
        + i * math.log(p)
        + (n - i) * math.log1p(-p)
        for i in range(k + 1)
    ]
    largest = max(terms)
    return min(1.0, math.exp(largest) * sum(math.exp(t - largest) for t in terms))


def clopper_pearson(values, confidence=0.95):
    if not values or any(v not in (0, 1) for v in values):
        raise ReplayError("Clopper-Pearson requires binary observations")
    n, k, alpha = len(values), sum(values), 1 - confidence

    def inverse_cdf(count, target):
        low, high = 0.0, 1.0
        for _ in range(70):
            mid = (low + high) / 2
            if _cdf(count, n, mid) > target:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    return [
        0.0 if k == 0 else inverse_cdf(k - 1, 1 - alpha / 2),
        1.0 if k == n else inverse_cdf(k, alpha / 2),
    ]


def interval(values, contract, *, paired=False, groups=None):
    method = contract.get("interval_method", "none")
    result = {
        "interval_method": method,
        "method_effective": "none",
        "ci": None,
        "inferential_valid": False,
        "n_paired": len(values) if paired else None,
        "n_independent": None,
        "n_clusters": len(set(groups)) if groups else None,
        "reason": "method_none",
        "bootstrap_empirical_range": None,
        "degenerate_distribution": False,
        "interval_target": contract.get("interval_target"),
        "interval_confidence": contract.get(
            "interval_confidence", contract.get("confidence", 0.95)
        ),
        "interval_sidedness": contract.get("interval_sidedness", "two_sided"),
    }
    if method == "none":
        return result
    if not values or any(not number(v) for v in values):
        return {**result, "reason": "missing_per_sample_observations"}
    if groups is not None and (
        len(groups) != len(values) or any(not isinstance(g, str) or not g for g in groups)
    ):
        return {**result, "reason": "cluster_identity_missing"}
    confidence = result["interval_confidence"]
    if not number(confidence) or not 0 < confidence < 1:
        raise ReplayError("Invalid confidence level")
    if result["interval_sidedness"] != "two_sided":
        return {**result, "reason": "sidedness_not_implemented_no_substitution"}
    if method in {"binomial_wilson", "clopper_pearson"}:
        if paired or contract.get("independent_units") is not True or groups:
            return {**result, "reason": "independent_binary_assumption_not_met"}
        if any(v not in (0, 1) for v in values):
            return {**result, "reason": "non_binary_values"}
        ci = (wilson if method == "binomial_wilson" else clopper_pearson)(values, confidence)
        return {
            **result,
            "ci": ci,
            "method_effective": method,
            "inferential_valid": True,
            "n_independent": len(values),
            "reason": None,
        }
    if method not in {"paired_bootstrap", "cluster_bootstrap"}:
        raise ReplayError(f"Unknown interval method: {method}")
    if not paired or contract.get("interval_target") != "fixed_set_paired_change":
        return {**result, "reason": "paired_target_required"}
    if contract.get("bootstrap_algorithm", "percentile") != "percentile":
        return {**result, "reason": "algorithm_not_implemented_no_substitution"}
    if method == "cluster_bootstrap":
        if not groups or len(groups) != len(values) or len(set(groups)) < 2:
            return {**result, "reason": "insufficient_clusters"}
        buckets = [
            [v for v, g in zip(values, groups, strict=True) if g == name]
            for name in sorted(set(groups))
        ]
    elif groups or contract.get("independent_units") is not True:
        return {**result, "reason": "resampling_unit_not_independent"}
    else:
        buckets = [[v] for v in values]
    if len(set(values)) == 1:
        return {
            **result,
            "reason": "degenerate_distribution",
            "degenerate_distribution": True,
            "bootstrap_empirical_range": [values[0], values[0]],
            "diagnostic": "NO_OBSERVED_DIFFERENCE" if values[0] == 0 else "CONSTANT_CHANGE",
        }
    repetitions = contract.get("resamples", 2000)
    if not isinstance(repetitions, int) or not 100 <= repetitions <= 100000:
        raise ReplayError("resamples must be 100..100000")
    rng, means = random.Random(contract.get("seed", 20261023)), []
    for _ in range(repetitions):
        sample = [value for _ in buckets for value in rng.choice(buckets)]
        means.append(sum(sample) / len(sample))
    means.sort()
    alpha = (1 - confidence) / 2
    lo, hi = (
        means[int(alpha * repetitions)],
        means[min(repetitions - 1, int((1 - alpha) * repetitions))],
    )
    if lo == hi:
        return {
            **result,
            "reason": "degenerate_resampling_distribution",
            "degenerate_distribution": True,
            "bootstrap_empirical_range": [lo, hi],
        }
    return {
        **result,
        "ci": [lo, hi],
        "method_effective": method,
        "inferential_valid": True,
        "reason": None,
    }
