import json

import pytest
from knowflow.app.answer_review import validate_review
from knowflow.app.evaluate_quality import reviews_complete
from knowflow.app.quality_metrics import (
    citation_excerpt_present,
    compare_variants,
    equivalent_question_consistency,
    evaluate_trace,
    regression_check,
)


def row(key, text, name="manual.md"):
    return {"chunk_id": key, "filename": name, "content": text}


def test_graph_additions_do_not_inflate_base_recall_at_k_or_hide_candidate_budget():
    relevant = row("graph-only", "A connects to B")
    policy = {"policy": "text-preserving-graph-union-v1", "added_count": 1, "max_additions": 1}
    result = evaluate_trace(
        {
            "should_refuse": False,
            "relevant_evidence": [
                {"id": "a", "filename": "manual.md", "contains": relevant["content"]}
            ],
        },
        {
            "top_k": 1,
            "base": [row("noise", "irrelevant")],
            "candidates": [row("noise", "irrelevant"), relevant],
            "context": [relevant],
            "graph_expansion": policy,
        },
        {"refused": False, "answer": "A connects to B", "citations": []},
        [relevant],
        [relevant],
    )
    assert result["metrics"]["recall@1"] == 0
    assert result["metrics"]["precision@1"] == 0
    assert result["metrics"]["candidates_evidence_recall"] == 1
    assert result["evidence_checks"][0]["ranks"]["candidates"] == [2]
    assert result["graph_expansion"] == policy


def test_rank_and_multi_fact_coverage_are_independent_and_context_losses_are_located():
    case = {
        "should_refuse": False,
        "relevant_evidence": [
            {"id": "a", "filename": "manual.md", "contains": "A uses B"},
            {"id": "b", "filename": "manual.md", "contains": "B uses C"},
        ],
    }
    good = row("right", "A uses B. B uses C.")
    audit = {
        "top_k": 3,
        "base": [row("wrong", "unrelated"), good],
        "candidates": [good],
        "context": [row("right", "A uses B.")],
    }
    result = evaluate_trace(
        case,
        audit,
        {
            "refused": False,
            "answer": "A uses B.",
            "citations": [{"chunk_id": "right", "filename": "manual.md", "quote": "A uses B."}],
        },
        [good],
        [good],
    )
    assert result["metrics"]["recall@3"] == 1
    assert result["metrics"]["mrr"] == 0.5
    assert result["metrics"]["context_evidence_recall"] == 0.5
    assert result["metrics"]["context_complete_evidence"] == 0
    assert result["evidence_checks"][1]["stage"] == "assessment_or_context_filter"
    assert result["evidence_checks"][0]["ranks"]["base"] == [2]


def test_missing_original_quote_is_not_asserted_as_a_definite_parser_bug():
    case = {
        "should_refuse": False,
        "relevant_evidence": [
            {"id": "a", "filename": "manual.md", "contains": "different punctuation"}
        ],
    }
    result = evaluate_trace(case, {"top_k": 3}, {"refused": True, "answer": "未知"}, [], [])
    assert result["evidence_checks"][0]["stage"] == "parse_or_annotation_mismatch"


def test_short_valid_excerpt_credits_source_without_claiming_full_quote_coverage():
    full = row("policy", "For version 2025, the timeout threshold is 30 ms.")
    result = evaluate_trace(
        {
            "should_refuse": False,
            "relevant_evidence": [
                {"id": "timeout", "filename": "manual.md", "contains": full["content"]}
            ],
        },
        {"top_k": 3, "base": [full], "candidates": [full], "context": [full]},
        {
            "refused": False,
            "answer": "30 ms [1].",
            "citations": [
                {
                    "chunk_id": "policy",
                    "filename": "manual.md",
                    "quote": "timeout threshold is 30 ms",
                }
            ],
        },
        [full],
        [full],
    )
    assert result["metrics"]["citation_sources_complete_evidence"] == 1
    assert result["metrics"]["citations_complete_evidence"] == 0
    assert result["evidence_checks"][0]["stage"] == "citation_excerpt_shorter_than_gold"
    assert result["evidence_checks"][0]["ranks"]["citation_sources"] == [1]


@pytest.mark.parametrize(
    ("quote", "valid"),
    [
        ("A uses the ledger.\n…\nB retains it for 17 days.", True),
        ("A uses the ledger.\n…\nB retains it for 99 days.", False),
        ("A uses the ledger.\n…\n", False),
        ("A uses the ledger. ... B retains it for 17 days.", False),
    ],
)
def test_composed_api_quotes_require_every_nonempty_fragment_in_own_context(quote, valid):
    assert (
        citation_excerpt_present(
            quote, "A uses the ledger. Some unrelated context. B retains it for 17 days."
        )
        is valid
    )


@pytest.mark.parametrize(
    "change", ["forged_quote", "wrong_chunk", "wrong_file", "truncated_context"]
)
def test_citation_source_credit_requires_own_valid_quote_and_gold_in_actual_context(change):
    full = row("policy", "Timeout 30 ms. Other context.")
    context = dict(full)
    citation = {"chunk_id": "policy", "filename": "manual.md", "quote": "Other context."}
    if change == "forged_quote":
        citation["quote"] = "Timeout 99 ms."
    elif change == "wrong_chunk":
        citation["chunk_id"] = "different"
    elif change == "wrong_file":
        citation["filename"] = "other.md"
    else:
        context["content"] = "Other context."
    result = evaluate_trace(
        {
            "should_refuse": False,
            "relevant_evidence": [
                {"id": "timeout", "filename": "manual.md", "contains": "Timeout 30 ms."}
            ],
        },
        {"top_k": 3, "base": [full], "candidates": [full], "context": [context]},
        {"refused": False, "answer": "30 ms [1].", "citations": [citation]},
        [full],
        [full],
    )
    assert result["metrics"]["citation_sources_evidence_recall"] == 0


def test_refusal_during_assessor_outage_is_not_counted_as_correct_refusal():
    case = {"should_refuse": True, "relevant_evidence": []}
    result = evaluate_trace(
        case,
        {"top_k": 3},
        {"refused": True, "answer": "未知", "retrieval_diagnostics": {"status": "unavailable"}},
        [],
        [],
    )
    assert result["metrics"]["refusal_decision_correct"] is None


def test_two_languages_both_missing_evidence_are_consistent_but_not_successful():
    rows = [
        {
            "id": lang,
            "equivalence_group": "same",
            "evidence_checks": [],
            "metrics": {"context_complete_evidence": 0},
        }
        for lang in ["zh", "en"]
    ]
    result = equivalent_question_consistency(rows)[0]
    assert result["consistent"] is True
    assert result["all_complete"] is False


def test_history_comparison_catches_new_unknowns_and_does_not_compare_changed_gold():
    before = {
        "dataset_sha256": "same",
        "corpus_sha256": "same",
        "variants": {"hybrid": {"cases": [{"id": "a", "metrics": {"context_evidence_recall": 1}}]}},
    }
    after = {**before, "variants": {"hybrid": {"cases": [{"id": "a", "metrics": {}}]}}}
    assert regression_check(after, before)["status"] == "incomplete"
    after["variants"]["hybrid"]["cases"][0]["metrics"]["context_evidence_recall"] = 0
    assert regression_check(after, before)["status"] == "regressed"
    assert (
        regression_check({**after, "dataset_sha256": "changed"}, before)["status"]
        == "not_comparable"
    )


def test_ablation_keeps_graph_harm_visible():
    before = [{"id": "a", "metrics": {"context_evidence_recall": 1}}]
    after = [{"id": "a", "metrics": {"context_evidence_recall": 0.5}}]
    assert compare_variants(before, after)[0]["status"] == "regressed"


def test_history_catches_ranking_loss_even_when_evidence_recall_is_unchanged():
    before = {
        "dataset_sha256": "same",
        "corpus_sha256": "same",
        "top_k": 3,
        "variants": {
            "hybrid": {
                "cases": [
                    {
                        "id": "a",
                        "metrics": {
                            "base_evidence_recall": 1,
                            "mrr": 1,
                            "ndcg@3": 1,
                        },
                    }
                ]
            }
        },
    }
    after = {
        **before,
        "variants": {
            "hybrid": {
                "cases": [
                    {
                        "id": "a",
                        "metrics": {
                            "base_evidence_recall": 1,
                            "mrr": 0.5,
                            "ndcg@3": 0.63,
                        },
                    }
                ]
            }
        },
    }
    result = regression_check(after, before)
    assert {r["metric"] for r in result["regressions"]} == {"mrr", "ndcg@3"}
    assert regression_check({**after, "top_k": 5}, before)["status"] == "not_comparable"


def test_forged_review_quotes_or_missing_answer_units_become_unknown():
    units = [{"id": "u1", "text": "A uses B."}, {"id": "u2", "text": "It costs 50."}]
    raw = {
        "units": [
            {
                "id": "u1",
                "verdict": "supported",
                "evidence": [{"chunk_id": "c", "quote": "invented source quote"}],
            }
        ],
        "facts": [],
    }
    result = validate_review(json.dumps(raw), units, [], [row("c", "A uses B.")])
    assert result["faithfulness_estimate"] is None
    assert result["status"] == "partial"
    assert all(r["verdict"] == "unknown" for r in result["units"])


def test_answer_fact_credit_requires_a_quote_that_exists_in_the_actual_answer():
    result = validate_review(
        json.dumps(
            {
                "units": [],
                "facts": [{"id": "f", "verdict": "satisfied", "answer_quote": "B uses C."}],
            }
        ),
        [{"id": "u1", "text": "A uses B."}],
        [{"id": "f", "text": "B uses C."}],
        [],
    )
    assert result["fact_recall_estimate"] is None


def test_graph_relation_review_cannot_borrow_a_different_source_chunk_to_claim_support():
    result = validate_review(
        json.dumps(
            {
                "units": [
                    {
                        "id": "edge",
                        "verdict": "supported",
                        "evidence": [{"chunk_id": "elsewhere", "quote": "Alpha uses Redis."}],
                    }
                ]
            }
        ),
        [{"id": "edge", "text": "Alpha uses Redis.", "source_chunk_id": "own"}],
        [],
        [row("elsewhere", "Alpha uses Redis."), row("own", "Alpha must not use Redis.")],
    )
    assert result["units"][0]["verdict"] == "unknown"
    assert result["faithfulness_estimate"] is None


def test_strict_review_completion_does_not_ignore_unknown_graph_relations():
    report = {
        "variants": {"hybrid": {"cases": [{"answer_review": {"status": "complete"}}]}},
        "graph_review": {"status": "partial", "relation_support_estimate": None},
    }
    assert reviews_complete(report) is False
    report["graph_review"]["status"] = "complete"
    assert reviews_complete(report) is True


@pytest.mark.parametrize(
    "coverage",
    [
        {"sampled": True, "reviewed_edges": 40, "total_edges": 340},
        {"reviewed_edges": 40, "total_edges": 340},
        {"sampled": True},
    ],
)
def test_strict_review_completion_rejects_partial_coverage_even_with_legacy_complete_status(
    coverage,
):
    report = {
        "variants": {"hybrid": {"cases": [{"refused": True}]}},
        "graph_review": {"status": "complete", **coverage},
    }
    assert reviews_complete(report) is False


def test_strict_review_completion_accepts_a_fully_reviewed_graph():
    report = {
        "variants": {"hybrid": {"cases": [{"refused": True}]}},
        "graph_review": {
            "status": "complete",
            "sampled": False,
            "reviewed_edges": 340,
            "total_edges": 340,
        },
    }
    assert reviews_complete(report) is True


@pytest.mark.asyncio
async def test_graph_audit_budget_reports_sample_scope_separately_from_all_edges(monkeypatch):
    from knowflow.app import answer_review
    from knowflow.app.config import Settings

    async def reviewer(units, *_):
        return {
            "status": "complete",
            "units": [
                {**u, "verdict": "supported", "reason": "fixture", "evidence": []} for u in units
            ],
        }

    monkeypatch.setattr(answer_review, "review_units", reviewer)
    edges = [
        {
            "id": str(i),
            "source": "A",
            "target": "B",
            "predicate": "uses",
            "description": "A uses B.",
            "chunk_id": "source",
            "source_text": "A uses B.",
        }
        for i in range(3)
    ]
    graph = {"edges": edges, "filtered_edge_count": 3}
    settings = Settings(_env_file=None)
    sampled = await answer_review.review_graph_edges(graph, settings, max_edges=2)
    assert sampled["status"] == "sampled" and sampled["sampled"] is True
    assert (sampled["reviewed_edges"], sampled["total_edges"]) == (2, 3)
    assert sampled["relation_support_estimate"] == 1
    assert sampled["support_estimate_scope"] == "reviewed_edges_only"
    complete = await answer_review.review_graph_edges(graph, settings, max_edges=None)
    assert complete["status"] == "complete" and complete["sampled"] is False
