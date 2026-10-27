import pytest
from knowflow.app import answer_guard
from knowflow.app.config import Settings


def test_tables_and_fenced_diagrams_remain_whole_review_blocks():

    text = "An answer.\n\n| A | B |\n|---|---|\n|1|2|\n\n```mermaid\ngraph LR\n\nA-->B\n```"

    blocks = answer_guard.answer_blocks(text)

    assert len(blocks) == 3

    assert blocks[1]["text"].count("\n") == 2

    assert blocks[2]["text"].endswith("```") and "A-->B" in blocks[2]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_verdict", ["unsupported", "contradicted", "unknown"])
async def test_guard_removes_unproven_whole_block_and_keeps_source_supported_block(
    monkeypatch, bad_verdict
):

    async def reviewer(units, *_, **_budget_kwargs):

        return {
            "status": "complete",
            "units": [
                {
                    **units[0],
                    "verdict": "supported",
                    "evidence": [{"chunk_id": "a", "quote": "If B, A."}],
                },
                {**units[1], "verdict": bad_verdict, "evidence": []},
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)

    answer, info = await answer_guard.guard_answer(
        "Under what condition?",
        "If B, A.\n\nOnly B can cause A.",
        [{"id": "a", "content": "If B, A."}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )

    assert answer == "If B, A."

    assert info["status"] == "filtered" and info["kept_blocks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "units",
    [
        [],
        [
            {
                "id": "b1u1",
                "verdict": "supported",
                "evidence": [{"chunk_id": "a", "quote": "invented"}],
            }
        ],
    ],
)
async def test_guard_never_treats_missing_or_forged_support_as_a_pass(monkeypatch, units):

    async def reviewer(*_, **_budget_kwargs):

        return {"status": "complete", "units": units}

    monkeypatch.setattr(answer_guard, "review_units", reviewer)

    answer, info = await answer_guard.guard_answer(
        "What?",
        "A.",
        [{"id": "a", "content": "B."}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )

    assert not answer and info["status"] == "rejected"


@pytest.mark.asyncio
async def test_review_timeout_is_unknown_and_does_not_return_the_draft(monkeypatch):

    import asyncio

    async def reviewer(*_, **_budget_kwargs):

        await asyncio.sleep(1)

    monkeypatch.setattr(answer_guard, "review_units", reviewer)

    answer, info = await answer_guard.guard_answer(
        "What?",
        "A claim.",
        [{"id": "a", "content": "A claim."}],
        Settings(_env_file=None, answer_repair_enabled=False, answer_guard_timeout_seconds=0.01),
    )

    assert not answer and info["status"] == "unavailable"


@pytest.mark.asyncio
async def test_a_supported_sentence_cannot_hide_an_unsupported_sentence_in_same_block(monkeypatch):

    async def reviewer(units, *_, **_budget_kwargs):

        assert len(units) == 2 and units[0]["block_id"] == units[1]["block_id"]

        return {
            "status": "complete",
            "units": [
                {
                    **units[0],
                    "verdict": "supported",
                    "evidence": [{"chunk_id": "a", "quote": "If B, A."}],
                },
                {**units[1], "verdict": "unsupported", "evidence": [], "reason": "必要性未证明"},
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)

    answer, info = await answer_guard.guard_answer(
        "When?",
        "If B, A. Only B can cause A.",
        [{"id": "a", "content": "If B, A."}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )

    assert not answer and info["kept_blocks"] == 0 and info["total_units"] == 2


@pytest.mark.asyncio
async def test_table_is_rejected_whole_if_one_row_is_unsupported(monkeypatch):

    async def reviewer(units, *_, **_budget_kwargs):

        return {
            "status": "complete",
            "units": [
                {
                    **u,
                    "verdict": "unsupported" if "invented" in u["text"] else "supported",
                    "evidence": [{"chunk_id": "a", "quote": "A B"}],
                }
                for u in units
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)

    answer, info = await answer_guard.guard_answer(
        "Compare",
        "| A | B |\n|---|---|\n|invented|3|",
        [{"id": "a", "content": "A B"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )

    assert not answer and info["total_blocks"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("second_status", ["complete", "filtered", "rejected", "unavailable"])
async def test_only_a_fully_rechecked_repair_is_adopted_and_attempted_once(
    monkeypatch, second_status
):

    calls = []

    async def review(question, draft, evidence, settings):

        calls.append(("review", draft))

        if draft == "original":
            return "verified portion", {"status": "filtered", "latency_ms": 1}

        return "rewritten answer", {"status": second_status, "latency_ms": 1}

    async def repair(question, draft, evidence, validation, settings, **_budget_kwargs):

        calls.append(("repair", draft))

        assert evidence == [{"id": "a", "content": "source"}]

        return "candidate"

    monkeypatch.setattr(answer_guard, "_review_draft", review)

    monkeypatch.setattr(answer_guard, "repair_draft", repair)

    answer, info = await answer_guard.guard_answer(
        "Question", "original", [{"id": "a", "content": "source"}], Settings(_env_file=None)
    )

    assert calls == [("review", "original"), ("repair", "original"), ("review", "candidate")]

    assert answer == ("rewritten answer" if second_status == "complete" else "verified portion")

    assert info["repair"]["status"] == ("adopted" if second_status == "complete" else "not_adopted")

    if second_status == "complete":
        assert info["initial_review"]["status"] == "filtered"


@pytest.mark.asyncio
async def test_repair_failure_cannot_replace_the_validated_portion(monkeypatch):

    async def review(*_):

        return "verified", {"status": "filtered", "latency_ms": 1}

    async def repair(*_, **_budget_kwargs):

        raise TimeoutError()

    monkeypatch.setattr(answer_guard, "_review_draft", review)

    monkeypatch.setattr(answer_guard, "repair_draft", repair)

    answer, info = await answer_guard.guard_answer(
        "Question", "draft", [], Settings(_env_file=None)
    )

    assert answer == "verified" and info["repair"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_online_thinking_switch_does_not_mutate_offline_judge_settings(monkeypatch):

    settings = Settings(
        _env_file=None,
        eval_judge_thinking="enabled",
        answer_guard_thinking="disabled",
        answer_repair_enabled=False,
    )

    async def review(units, facts, context, supplied, question, **_budget_kwargs):

        assert supplied.eval_judge_thinking == "disabled"

        return {"status": "unavailable", "units": []}

    monkeypatch.setattr(answer_guard, "review_units", review)

    _, info = await answer_guard.guard_answer("Q", "A.", [], settings)

    assert settings.eval_judge_thinking == "enabled" and info["thinking"] == "disabled"


@pytest.mark.asyncio
async def test_source_first_repair_does_not_receive_rejected_draft_or_gold(monkeypatch):

    import json
    from types import SimpleNamespace

    requests = []

    async def create(**kwargs):

        requests.append(kwargs)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Fresh."))
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    client.with_options = lambda **kwargs: client

    monkeypatch.setattr(answer_guard, "chat_client", lambda: client)

    result = await answer_guard.repair_draft(
        "Question",
        "REJECTED_DRAFT",
        [{"id": "a", "content": "Actual source"}],
        {
            "reviewed_units": [
                {
                    "text": "REJECTED_UNIT",
                    "verdict": "unsupported",
                    "reason": "Necessity not established",
                }
            ]
        },
        Settings(_env_file=None),
    )

    payload = json.loads(requests[0]["messages"][1]["content"])

    assert result == "Fresh." and payload["evidence"][0]["content"] == "Actual source"

    assert "draft" not in payload and "required_facts" not in payload

    assert "REJECTED_DRAFT" not in str(payload) and "REJECTED_UNIT" not in str(payload)

    assert payload["issues"][0]["reason"] == "Necessity not established"

@pytest.mark.asyncio
async def test_long_answers_review_without_thinking_to_keep_the_verdict_budget(monkeypatch):
    seen = {}

    async def reviewer(units, facts, context, supplied, question, **kwargs):
        seen["thinking"] = supplied.eval_judge_thinking
        seen["units"] = len(units)
        return {
            "status": "complete",
            "units": [
                {**unit, "verdict": "supported", "evidence": [{"chunk_id": "a", "quote": "A"}]}
                for unit in units
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    draft = "\n".join(f"- Point {index} [a]" for index in range(1, 26))
    answer, info = await answer_guard.guard_answer(
        "Q",
        draft,
        [{"id": "a", "content": "A"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )

    assert seen["thinking"] == "disabled"
    assert seen["units"] > 20
    assert info["thinking_downgraded"] == "unit_count_gt_20"
    assert answer


@pytest.mark.asyncio
async def test_long_answer_is_reviewed_in_batches_instead_of_refused_whole(monkeypatch):
    batch_sizes = []

    async def reviewer(units, *_, **_budget_kwargs):
        batch_sizes.append(len(units))
        return {
            "status": "complete",
            "units": [
                {
                    **unit,
                    "verdict": "supported",
                    "evidence": [{"chunk_id": "a", "quote": "A."}],
                }
                for unit in units
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    sentences = [f"Fact {index} [a]." for index in range(120)]
    draft = "\n\n".join(sentences)
    answer, info = await answer_guard.guard_answer(
        "Q?",
        draft,
        [{"id": "a", "content": "A."}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    # A cross-document listing is legitimately long; it must not be rejected wholesale.
    assert info["status"] != "budget_exceeded"
    assert len(batch_sizes) > 1
    assert max(batch_sizes) <= answer_guard.GUARD_UNIT_BATCH
    assert sum(batch_sizes) == len(sentences)
    assert info["unit_batches"] == len(batch_sizes)
    assert len(answer.split("\n\n")) == len(sentences)


@pytest.mark.asyncio
async def test_partially_reviewed_answer_only_keeps_the_reviewed_supported_blocks(monkeypatch):
    calls = []

    async def reviewer(units, *_, **_budget_kwargs):
        calls.append(len(units))
        if len(calls) > 1:
            return {"status": "unavailable", "units": [], "error_type": "APIError"}
        return {
            "status": "complete",
            "units": [
                {
                    **unit,
                    "verdict": "supported",
                    "evidence": [{"chunk_id": "a", "quote": "A."}],
                }
                for unit in units
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    sentences = [f"Fact {index} [a]." for index in range(90)]
    answer, info = await answer_guard.guard_answer(
        "Q?",
        "\n\n".join(sentences),
        [{"id": "a", "content": "A."}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    # Unreviewed sentences stay unknown and their blocks are dropped, not assumed good.
    assert info["review_status"] == "partial"
    assert info["batches_completed"] == 1
    assert len(answer.split("\n\n")) < len(sentences)
