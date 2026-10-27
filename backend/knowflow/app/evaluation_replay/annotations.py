"""Deterministic annotation checks, never a replacement for semantic adjudication."""

from decimal import Decimal, InvalidOperation

from .common import ReplayError

UNITS = {
    "元": ("CNY", "1"),
    "万元": ("CNY", "10000"),
    "亿元": ("CNY", "100000000"),
    "百万元": ("CNY", "1000000"),
    "%": ("ratio", ".01"),
    "ratio": ("ratio", "1"),
}


def quantities_equal(gold, prediction, *, relative_tolerance=None):
    """Compare explicitly supplied values/units only, not answer text or evidence."""
    if gold.get("unit") not in UNITS or prediction.get("unit") not in UNITS:
        return None
    gd, gf = UNITS[gold["unit"]]
    pd, pf = UNITS[prediction["unit"]]
    if gd != pd:
        return False
    try:
        left = Decimal(str(gold["value"])) * Decimal(gf)
        right = Decimal(str(prediction["value"])) * Decimal(pf)
        tolerance = (
            Decimal(str(relative_tolerance)) if relative_tolerance is not None else Decimal(0)
        )
    except (InvalidOperation, KeyError) as exc:
        raise ReplayError("Invalid numeric annotation") from exc
    if not all(v.is_finite() for v in (left, right, tolerance)) or tolerance < 0:
        raise ReplayError("Nonfinite/negative annotation tolerance")
    return abs(left - right) <= abs(left) * tolerance


def approved_evidence_complete(alternatives, approved_source_ids):
    """AND within a reviewed evidence group; OR across reviewed alternatives."""
    if approved_source_ids is None:
        return None
    return any(bool(group) and set(group) <= set(approved_source_ids) for group in alternatives)
