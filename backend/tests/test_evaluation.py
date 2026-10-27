import json
from math import log2
from pathlib import Path

import pytest
from knowflow.app.evaluation import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_retrieval_metrics_measure_known_relevant_results() -> None:
    retrieved = ["chunk-a", "chunk-x", "chunk-b"]
    relevant = {"chunk-a", "chunk-b"}

    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert precision_at_k(retrieved, relevant, 2) == pytest.approx(0.5)
    assert reciprocal_rank(retrieved, relevant) == 1.0
    assert ndcg_at_k(retrieved, relevant, 3) == pytest.approx((1 + 1 / log2(4)) / (1 + 1 / log2(3)))
    assert mean_reciprocal_rank(
        [(retrieved, relevant), (["chunk-x", "chunk-b"], relevant)]
    ) == pytest.approx(0.75)


def test_metrics_validate_k_and_handle_empty_labels() -> None:
    with pytest.raises(ValueError, match="positive"):
        recall_at_k([], [], 0)
    assert recall_at_k(["chunk-a"], [], 1) == 0.0
    assert precision_at_k([], ["chunk-a"], 3) == 0.0
    assert reciprocal_rank(["chunk-a"], ["chunk-b"]) == 0.0
    assert mean_reciprocal_rank([]) == 0.0


def test_fixed_evaluation_set_has_supported_evidence_for_every_label() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / "evaluation" / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    cases = dataset["cases"]

    assert {case["question_type"] for case in cases} == {
        "single_fact",
        "exact_keyword",
        "semantic_paraphrase",
        "cross_document",
        "multi_hop",
        "no_answer",
        "distractor",
        "cross_lingual",
        "version_scope",
        "negation_scope",
    }
    for case in cases:
        for evidence in case["relevant_evidence"]:
            source = (project_root / "evaluation" / "corpus" / evidence["filename"]).read_text(
                encoding="utf-8"
            )
            assert evidence["contains"] in source
        assert case["should_refuse"] is bool(not case["relevant_evidence"])
