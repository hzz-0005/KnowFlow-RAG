import asyncio
import hashlib
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from knowflow.app import answer_guard, answer_review
from knowflow.app.config import Settings
from knowflow.app.guard_budget import budget_contract
from knowflow.app.quality_metrics import regression_check

ID = str(UUID(int=601))
CONTEXT = [{"chunk_id": ID, "content": "Only test copies are retained for 23 days."}]
UNITS = [{"id": "u1", "text": "Test copies are retained for 23 days."}]


def settings(**kwargs):
    return Settings(
        _env_file=None,
        llm_model="qwen3.8-flash",
        llm_complex_model="qwen3.8-max",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        **kwargs,
    )


def reply(raw, finish="stop"):
    return SimpleNamespace(
        model="reported-fixture",
        usage={"prompt_tokens": 700, "completion_tokens": 100},
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=raw))],
    )


def audit_raw(units=UNITS):
    return json.dumps(
        {
            "units": [
                {
                    "id": u["id"],
                    "verdict": "supported",
                    "reason": "来源支持",
                    "evidence": [{"chunk_id": ID, "quote": "retained for 23 days"}],
                }
                for u in units
            ],
            "facts": [],
        }
    )


def client(create):
    instance = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    instance.with_options = lambda **kwargs: instance
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized", ["question", "facts", "context", "units"])
async def test_review_full_input_exhaustion_never_initializes_client_or_trims_targets(
    monkeypatch, oversized
):
    monkeypatch.setattr(
        answer_review, "chat_client", lambda: pytest.fail("Unexpected SDK initialization")
    )
    question, facts, context, units = "Q", [], CONTEXT, UNITS
    if oversized == "question":
        question = "Full original condition. " * 3000
    if oversized == "facts":
        facts = [{"id": "f", "text": "Required fact " * 4000}]
    if oversized == "context":
        context = [{"chunk_id": ID, "content": "Source " * 10000}]
    if oversized == "units":
        units = [{"id": str(i), "text": "Full unit"} for i in range(61)]
    trace = {}
    result = await answer_review.review_units(
        units,
        facts,
        context,
        settings(answer_review_total_token_budget=1024),
        question,
        budget_trace=trace,
    )
    assert result["status"] == "budget_exceeded" and result["faithfulness_estimate"] is None
    assert not trace["request_attempted"] and trace["provider_usage"] is None
    assert trace["unit_count"] == len(units) and trace["fact_count"] == len(facts)
    assert trace["source_count"] == len(context)


@pytest.mark.asyncio
async def test_review_sdk_receives_full_question_facts_and_context_with_same_trace(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return reply(audit_raw())

    monkeypatch.setattr(answer_review, "chat_client", lambda: client(create))
    question = "Only test; never transfer the value to production."
    facts = [{"id": "f", "text": "Retain the original required scope."}]
    trace = {}
    await answer_review.review_units(
        UNITS, facts, CONTEXT, settings(), question, budget_trace=trace
    )
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload == {
        "question": question,
        "units": UNITS,
        "required_facts": facts,
        "actual_context": CONTEXT,
    }
    assert calls[0]["max_completion_tokens"] == 8192
    assert (
        trace["prompt_sha256"]
        == hashlib.sha256(
            json.dumps(calls[0]["messages"], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )
    assert trace["provider_usage"]["prompt_tokens"] == 700 and trace["finish_reason"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "content_filter", None, "tool_calls"])
async def test_nonfinal_review_json_is_never_treated_as_supported(monkeypatch, finish):
    async def create(**kwargs):
        return reply(audit_raw(), finish)

    monkeypatch.setattr(answer_review, "chat_client", lambda: client(create))
    result = await answer_review.review_units(UNITS, [], CONTEXT, settings(), "Q")
    assert result["status"] == "unavailable" and result["faithfulness_estimate"] is None
    assert result["context_budget"]["response_received"]
    assert result["context_budget"]["reason_code"] != "validated_response"


@pytest.mark.asyncio
async def test_separate_judge_output_parameter_uses_judge_endpoint_and_closes_client(monkeypatch):
    calls, closed = [], []

    async def create(**kwargs):
        calls.append(kwargs)
        return reply(audit_raw())

    instance = client(create)

    async def close():
        closed.append(True)

    instance.close = close
    monkeypatch.setattr(answer_review, "_client", lambda *args: instance)
    monkeypatch.setattr(answer_review, "chat_client", lambda: pytest.fail("Wrong provider"))
    result = await answer_review.review_units(
        UNITS,
        [],
        CONTEXT,
        settings(
            eval_judge_model="deepseek-fixture",
            eval_judge_base_url="https://api.deepseek.com/v1",
            eval_judge_api_key="fixture-not-a-key",
        ),
        "Q",
    )
    assert "max_tokens" in calls[0] and "max_completion_tokens" not in calls[0]
    assert result["context_budget"]["output_limit_scope"] == "reasoning_and_answer"
    assert closed == [True]


@pytest.mark.asyncio
async def test_outer_guard_timeout_keeps_attempt_trace_and_unknown_units(monkeypatch):
    async def create(**kwargs):
        await asyncio.sleep(1)

    monkeypatch.setattr(answer_review, "chat_client", lambda: client(create))
    answer, info = await answer_guard.guard_answer(
        "Q",
        "A factual statement.",
        [{"id": ID, "content": CONTEXT[0]["content"]}],
        settings(answer_guard_timeout_seconds=0.01, answer_repair_enabled=False),
    )
    assert not answer and info["status"] == "unavailable"
    trace = info["stages"][0]["context_budget"]
    assert trace["request_attempted"] and not trace["response_received"]
    assert trace["reason_code"] == "outer_guard_timeout" and trace["provider_usage"] is None
    assert info["unknown_units"] == info["total_units"]


@pytest.mark.asyncio
async def test_repair_budget_blocks_before_sdk_and_does_not_receive_rejected_draft(monkeypatch):
    monkeypatch.setattr(answer_guard, "chat_client", lambda: pytest.fail("Unexpected SDK"))
    trace = {}
    with pytest.raises(ValueError, match="exceeds budget"):
        await answer_guard.repair_draft(
            "Q " * 5000,
            "REJECTED SECRET DRAFT",
            [],
            {},
            settings(answer_repair_total_token_budget=1024),
            budget_trace=trace,
        )
    assert not trace["request_attempted"] and trace["reason_code"] == "full_input_exceeds_budget"
    assert "REJECTED SECRET DRAFT" not in str(answer_guard.repair_messages("Q", [], {}))


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", None, "content_filter"])
async def test_nonfinal_repair_never_returns_partial_candidate(monkeypatch, finish):
    async def create(**kwargs):
        return reply("A plausible but truncated answer.", finish)

    monkeypatch.setattr(answer_guard, "chat_client", lambda: client(create))
    trace = {}
    with pytest.raises(ValueError):
        await answer_guard.repair_draft("Q", "old", [], {}, settings(), budget_trace=trace)
    assert trace["status"] == "unavailable" and trace["response_received"]


@pytest.mark.asyncio
async def test_complete_repair_chain_keeps_three_independent_budgets_and_rechecks_candidate(
    monkeypatch,
):
    calls = []

    async def review_create(**kwargs):
        calls.append(("review", kwargs))
        units = json.loads(kwargs["messages"][1]["content"])["units"]
        data = json.loads(audit_raw(units))
        if sum(kind == "review" for kind, _ in calls) == 1:
            data["units"][-1].update(verdict="unsupported", evidence=[])
        return reply(json.dumps(data))

    async def repair_create(**kwargs):
        calls.append(("repair", kwargs))
        return reply("Test copies are retained for 23 days.")

    monkeypatch.setattr(answer_review, "chat_client", lambda: client(review_create))
    monkeypatch.setattr(answer_guard, "chat_client", lambda: client(repair_create))
    answer, info = await answer_guard.guard_answer(
        "What is the period?",
        "First assertion.\n\nIncorrect assertion.",
        [{"id": ID, "content": CONTEXT[0]["content"]}],
        settings(),
    )
    assert [kind for kind, _ in calls] == ["review", "repair", "review"]
    assert answer and info["repair"]["status"] == "adopted"
    traces = [row["context_budget"] for row in info["stages"]]
    assert len({id(trace) for trace in traces}) == 3
    assert all(trace["status"] == "completed" for trace in traces)
    assert info["initial_review"]["status"] == "filtered"
    assert "required_facts" not in calls[1][1]["messages"][1]["content"]
    json.dumps(info)


def test_guard_budget_contract_mismatch_or_missing_history_is_not_comparable():
    contract = budget_contract(settings())
    assert regression_check({"guard_budget_contract": contract}, {})["status"] == "not_comparable"
    assert (
        regression_check(
            {"guard_budget_contract": contract},
            {"guard_budget_contract": {**contract, "packing": "different"}},
        )["status"]
        == "not_comparable"
    )
