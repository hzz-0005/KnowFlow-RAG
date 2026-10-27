from uuid import UUID

import pytest
from knowflow.app import answer_guard
from knowflow.app.config import Settings
from knowflow.app.guard_budget import budget_contract
from knowflow.app.quality_metrics import regression_check

ID = str(UUID(int=701))
EVIDENCE = [{"id": ID, "content": "Evidence states a duration of 23 days."}]


def supported(unit):
    return {
        **unit,
        "verdict": "supported",
        "reason": "依据存在",
        "evidence": [{"chunk_id": ID, "quote": "duration of 23 days"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intro",
    ["Here are the details:", "下面分项说明：", "Résumé de la réponse :", "Version overview:"],
)
async def test_only_plain_nonfactual_omission_retains_facts_and_does_not_dispatch_repair(
    monkeypatch, intro
):
    calls = []

    async def review(units, *args, **kwargs):
        calls.append("review")
        return {
            "status": "complete",
            "units": [{**units[0], "verdict": "not_factual", "evidence": []}, supported(units[1])],
        }

    async def forbidden(*args, **kwargs):
        pytest.fail("Presentation-only filtering must not dispatch generation or rereview")

    monkeypatch.setattr(answer_guard, "review_units", review)
    monkeypatch.setattr(answer_guard, "repair_draft", forbidden)
    answer, info = await answer_guard.guard_answer(
        "What duration?", intro + "\n\nIt lasts 23 days.", EVIDENCE, Settings(_env_file=None)
    )
    assert "It lasts 23 days." in answer and ID in answer and intro not in answer
    assert calls == ["review"] and len(info["stages"]) == 1 and "repair" not in info
    assert info["status"] == "filtered" and info["omission_counts"] == {
        "nonfactual_prose": 1,
        "requires_attention": 0,
    }
    assert info["repair_trigger"]["reason"] == "nonfactual_prose_only"
    assert info["reviewed_blocks"][0]["filter_reason"] == "nonfactual_prose"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "unknown", "unsupported", "contradicted", "forged_quote"]
)
async def test_uncertain_or_unsupported_block_still_dispatches_one_repair(monkeypatch, problem):
    dispatched = []

    async def review(units, *args, **kwargs):
        rows = [{**units[0], "verdict": "not_factual", "evidence": []}, supported(units[1])]
        bad = supported(units[2])
        if problem == "duplicate":
            rows += [bad, bad]
        elif problem != "missing":
            if problem == "forged_quote":
                bad["evidence"][0]["quote"] = "invented quotation"
            else:
                bad["verdict"] = problem
            rows.append(bad)
        return {"status": "complete", "units": rows}

    async def repair(*args, **kwargs):
        dispatched.append(True)
        raise TimeoutError()

    monkeypatch.setattr(answer_guard, "review_units", review)
    monkeypatch.setattr(answer_guard, "repair_draft", repair)
    answer, info = await answer_guard.guard_answer(
        "Q", "Introduction.\n\nSupported fact.\n\nDubious fact.", EVIDENCE, Settings(_env_file=None)
    )
    assert dispatched == [True] and "Supported fact." in answer and "Dubious fact." not in answer
    assert info["omission_counts"]["requires_attention"] == 1
    assert info["repair_trigger"]["decision"] == "required"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "structured",
    [
        "| A | B |\n|---|---|\n|23|45|",
        "```mermaid\ngraph LR\nA-->B\n```",
        "- A policy",
        "> A statement",
        "    print(23)",
        "# A heading",
        "A | B",
        "Refer to `x`",
    ],
)
async def test_structured_block_never_uses_plain_prose_skip_even_if_judge_labels_it_nonfactual(
    monkeypatch, structured
):
    dispatched = []

    async def review(units, *args, **kwargs):
        return {
            "status": "complete",
            "units": [
                {**u, "verdict": "not_factual", "evidence": []}
                if u["block_id"] == "b1"
                else supported(u)
                for u in units
            ],
        }

    async def repair(*args, **kwargs):
        dispatched.append(True)
        raise TimeoutError()

    monkeypatch.setattr(answer_guard, "review_units", review)
    monkeypatch.setattr(answer_guard, "repair_draft", repair)
    answer, info = await answer_guard.guard_answer(
        "Q", structured + "\n\nA supported fact.", EVIDENCE, Settings(_env_file=None)
    )
    assert answer and dispatched == [True]
    assert info["reviewed_blocks"][0]["kind"] == "structured"
    assert info["omission_counts"]["requires_attention"] == 1


@pytest.mark.asyncio
async def test_all_nonfactual_answer_without_returnable_fact_does_not_suppress_source_first_repair(
    monkeypatch,
):
    dispatched = []

    async def review(units, *args, **kwargs):
        return {
            "status": "complete",
            "units": [{**u, "verdict": "not_factual", "evidence": []} for u in units],
        }

    async def repair(*args, **kwargs):
        dispatched.append(True)
        raise TimeoutError()

    monkeypatch.setattr(answer_guard, "review_units", review)
    monkeypatch.setattr(answer_guard, "repair_draft", repair)
    answer, info = await answer_guard.guard_answer(
        "Q", "Here is an answer:", EVIDENCE, Settings(_env_file=None)
    )
    assert not answer and dispatched == [True] and info["status"] == "rejected"


@pytest.mark.parametrize(
    "change",
    [
        {"review_status": "partial"},
        {"unknown_units": 1},
        {"omission_counts": None},
        {"omission_counts": {"nonfactual_prose": 1, "requires_attention": 1}},
        {"kept_blocks": 0},
    ],
)
def test_incomplete_or_legacy_trace_cannot_suppress_repair(change):
    validation = {
        "status": "filtered",
        "review_status": "complete",
        "unknown_units": 0,
        "total_blocks": 2,
        "kept_blocks": 1,
        "omission_counts": {"nonfactual_prose": 1, "requires_attention": 0},
        **change,
    }
    assert (
        answer_guard.repair_trigger("Fact", validation, Settings(_env_file=None))["decision"]
        == "required"
    )


def test_rule_version_is_part_of_existing_comparability_contract():
    contract = budget_contract(Settings(_env_file=None))
    old = {k: v for k, v in contract.items() if k != "repair_trigger_version"}
    assert (
        regression_check({"guard_budget_contract": contract}, {"guard_budget_contract": old})[
            "status"
        ]
        == "not_comparable"
    )
