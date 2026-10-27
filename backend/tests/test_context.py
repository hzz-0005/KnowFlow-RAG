import pytest
from knowflow.app.context import build_evidence_context, estimate_tokens


def test_context_deduplicates_evidence_and_obeys_approximate_budget() -> None:
    evidence = [
        {"id": "chunk-1", "content": "A source passage. " * 100},
        {"id": "chunk-2", "content": "  A   source passage. " * 100},
        {"id": "chunk-3", "content": "A different source passage."},
    ]

    result = build_evidence_context(evidence, token_budget=40)

    assert len(result) == 1
    assert len(result[0]["content"]) < len(evidence[0]["content"])
    assert estimate_tokens(result[0]["content"]) + estimate_tokens("chunk-1") + 8 <= 40


def test_context_requires_a_positive_budget() -> None:
    with pytest.raises(ValueError, match="positive"):
        build_evidence_context([], token_budget=0)
