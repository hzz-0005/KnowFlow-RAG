import asyncio
import json
from types import SimpleNamespace

import pytest
from knowflow.app import answer_guard, answer_review
from knowflow.app.config import Settings


@pytest.mark.asyncio
async def test_guard_normalizes_each_source_once_but_checks_every_unit(monkeypatch):
    source = "SOURCE prefix: evidence and more content"
    original = answer_guard.normalize
    seen = []

    def normalized(text):
        seen.append(text)
        return original(text)

    async def reviewer(units, *_, **_budget_kwargs):
        return {
            "status": "complete",
            "units": [
                {**u, "verdict": "supported", "evidence": [{"chunk_id": "c", "quote": "evidence"}]}
                for u in units
            ],
        }

    monkeypatch.setattr(answer_guard, "normalize", normalized)
    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    draft = "\n\n".join(f"Fact {i}." for i in range(8))
    answer, info = await answer_guard.guard_answer(
        "Q",
        draft,
        [{"id": "c", "content": source}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    assert answer == draft and seen.count(source) == 1
    assert info["total_units"] == info["units_with_valid_review"] == 8
    assert info["unknown_units"] == 0
    assert [s["stage"] for s in info["stages"]] == ["initial_review"]


def test_review_validation_reuses_normalized_sources_without_altering_verdicts(monkeypatch):
    original = answer_review.normalize
    source, seen = "SOURCE prefix: evidence and more content", []

    def normalized(text):
        seen.append(text)
        return original(text)

    monkeypatch.setattr(answer_review, "normalize", normalized)
    units = [{"id": str(i), "text": f"Fact {i}"} for i in range(10)]
    raw = {
        "units": [
            {
                "id": u["id"],
                "verdict": "supported",
                "evidence": [{"chunk_id": "c", "quote": "evidence"}],
            }
            for u in units
        ]
    }
    result = answer_review.validate_review(
        json.dumps(raw), units, [], [{"chunk_id": "c", "content": source}]
    )
    assert seen.count(source) == 1 and len(result["units"]) == 10
    assert result["status"] == "complete"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["adopted", "not_adopted", "generation_error", "review_error"])
async def test_stage_accounting_preserves_one_repair_and_initial_result(monkeypatch, outcome):
    async def review(question, draft, evidence, settings):
        if draft == "original":
            return "kept", {"status": "filtered", "latency_ms": 1}
        if outcome == "review_error":
            raise ValueError("bad review")
        return "new", {
            "status": "complete" if outcome == "adopted" else "rejected",
            "latency_ms": 1,
        }

    async def repair(*_, **_budget_kwargs):
        if outcome == "generation_error":
            raise TimeoutError("timeout")
        return "candidate"

    monkeypatch.setattr(answer_guard, "_review_draft", review)
    monkeypatch.setattr(answer_guard, "repair_draft", repair)
    answer, info = await answer_guard.guard_answer("Q", "original", [], Settings(_env_file=None))
    assert answer == ("new" if outcome == "adopted" else "kept")
    assert [s["stage"] for s in info["stages"]] == (
        ["initial_review", "repair_generation"]
        if outcome == "generation_error"
        else ["initial_review", "repair_generation", "repair_review"]
    )
    assert sum(s["latency_ms"] for s in info["stages"]) <= info["latency_ms"] + 1
    assert info["timing_contract"] == "answer-guard-stages-v1"
    if outcome.endswith("error"):
        assert info["stages"][-1]["status"] == "unavailable"
    json.dumps(info)  # Adopted repair and initial review must not form cycles.


@pytest.mark.asyncio
async def test_cancelled_review_propagates_instead_of_becoming_a_pass(monkeypatch):
    async def reviewer(*_, **_budget_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    with pytest.raises(asyncio.CancelledError):
        await answer_guard.guard_answer("Q", "Fact.", [], Settings(_env_file=None))


@pytest.mark.asyncio
async def test_provider_and_local_validation_timings_are_separate(monkeypatch):
    async def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "units": [
                                    {
                                        "id": "u",
                                        "verdict": "supported",
                                        "evidence": [{"chunk_id": "c", "quote": "Fact"}],
                                    }
                                ]
                            }
                        )
                    ),
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **kwargs: client
    monkeypatch.setattr(answer_review, "chat_client", lambda: client)
    result = await answer_review.review_units(
        [{"id": "u", "text": "Fact"}],
        [],
        [{"chunk_id": "c", "content": "Fact"}],
        Settings(_env_file=None, eval_judge_model=None, eval_judge_api_key=None),
        "Q",
    )
    assert result["status"] == "complete"
    assert result["timing"]["request_ms"] >= 0 and result["timing"]["validation_ms"] >= 0


@pytest.mark.asyncio
async def test_duplicate_review_rows_remain_unknown_and_reject_the_atomic_block(monkeypatch):
    async def reviewer(units, *_, **_budget_kwargs):
        def supported(unit):
            return {
                **unit,
                "verdict": "supported",
                "evidence": [{"chunk_id": "c", "quote": "Fact"}],
            }

        return {
            "status": "complete",
            "units": [supported(units[0]), supported(units[1]), supported(units[1])],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    answer, info = await answer_guard.guard_answer(
        "Q",
        "Fact one. Fact two.",
        [{"id": "c", "content": "Fact"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    assert not answer and info["status"] == "rejected"
    assert info["units_with_valid_review"] == 1 and info["unknown_units"] == 1
