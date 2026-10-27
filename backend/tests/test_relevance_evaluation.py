import pytest
from knowflow.app.evaluate_relevance import compare_regressions, summarize_decisions


def test_acceptance_precision_recall_and_unknown_are_separate():
    result = summarize_decisions(
        [
            {"expected_usable": True, "accepted": True, "verdict": "partial"},
            {"expected_usable": True, "accepted": False, "verdict": "unknown"},
            {"expected_usable": False, "accepted": True, "verdict": "direct"},
            {"expected_usable": False, "accepted": False, "verdict": "background"},
        ]
    )
    assert result["accepted_precision"] == 0.5
    assert result["relevant_recall"] == 0.5
    assert result["unknown_count"] == 1
    assert result["false_accepts"] == result["false_rejects"] == 1


def test_regression_comparison_rejects_changed_corpus_and_detects_lost_cases():
    baseline = {"dataset_sha256": "fixed", "decisions": [{"id": "a", "passed": True}]}
    report = {"dataset_sha256": "fixed", "decisions": [{"id": "a", "passed": False}]}
    assert compare_regressions(report, baseline) == ["a"]
    with pytest.raises(ValueError, match="different dataset"):
        compare_regressions({**report, "dataset_sha256": "changed"}, baseline)
