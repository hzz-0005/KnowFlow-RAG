"""Deterministic transport/budget controls; not semantic model evaluations."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest
from knowflow.app import conversation_context as context
from knowflow.app import query_plan
from knowflow.app.config import Settings
from knowflow.app.understanding_budget import contract, measure


def prior():
    return SimpleNamespace(
        id="previous",
        question="Only staging 007, excluding production 12.50 in 2025.",
        resolved_question=None,
        status="completed",
        resolution={},
        response={"answer": "NEVER_COPY_ASSISTANT", "citations": []},
    )


def provider(monkeypatch, module, *, raw, finish="stop", usage=True, error=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=raw))],
            usage=SimpleNamespace(prompt_tokens=902, completion_tokens=73, total_tokens=975)
            if usage
            else None,
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **kw: client if kw == {"max_retries": 0} else pytest.fail("retry")
    monkeypatch.setattr(module, "chat_client", lambda: client)
    return calls


def setup(stage):
    module = context if stage == "context_resolution" else query_plan
    question = "What about 2025, excluding production?"
    raw = json.dumps(
        {
            "status": "resolved",
            "question": prior().question + " " + question,
            "reference_turn_ids": ["previous"],
        }
        if module is context
        else {"search_queries": ["2025 policy, excluding production"]}
    )
    return module, question, raw


async def run(stage, question, settings):
    return (
        await context.resolve_question(question, [prior()], settings)
        if stage == "context_resolution"
        else await query_plan.plan_query(question, settings)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["context_resolution", "query_rewrite"])
@pytest.mark.parametrize("difference", [0, -1])
async def test_exact_full_request_boundary_and_one_token_over(monkeypatch, stage, difference):
    module, question, raw = setup(stage)
    calls = provider(monkeypatch, module, raw=raw)
    settings = Settings(_env_file=None)
    baseline = await run(stage, question, settings)
    assert len(calls) == 1
    messages = calls[0]["messages"]
    required = measure(messages, settings, stage)["estimated_total_tokens"]
    setattr(settings, stage + "_total_token_budget", required + difference)
    calls.clear()
    result = await run(stage, question, settings)
    trace = result["request_budget"]
    assert trace["estimated_total_tokens"] == required
    assert trace["input_tokens_estimate"] > 0 and trace["safety_tokens"] == 512
    assert (
        trace["prompt_sha256"]
        == hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )
    if difference == 0:
        assert len(calls) == 1 and calls[0]["messages"] == messages
        assert calls[0][trace["output_limit_parameter"]] == 2048
        assert trace["request_attempted"] and trace["response_received"]
        assert trace["provider_usage"] == baseline["provider_usage"]
    else:
        assert not calls and trace["status"] == "blocked"
        assert not result["request_attempted"] and not result["response_received"]
        assert trace["provider_usage"] is None and trace["reason_code"] == "request_budget_exceeded"
        if stage == "context_resolution":
            assert (
                result["status"] == "clarify" and result["pending_intent"]["requires_restatement"]
            )
            assert not result["reference_turn_ids"] and "intent_context" not in result
        else:
            assert result["status"] == "unavailable" and result["phase"] == "budget"
            assert (
                result["search_queries"] == []
                and result["fallback_used"]
                and result["original_only"]
            )


@pytest.mark.asyncio
async def test_payload_limit_is_not_a_full_request_budget(monkeypatch):
    calls = provider(monkeypatch, context, raw="UNUSED")
    settings = Settings(
        _env_file=None,
        context_resolution_total_token_budget=1024,
        context_resolution_output_token_budget=64,
        context_resolution_safety_tokens=0,
    )
    # Padding in the immutable system prompt is counted too, without changing payload.
    monkeypatch.setattr(context, "PROMPT", context.PROMPT + " Extra system constraint." * 700)
    result = await context.resolve_question("How long?", [prior()], settings)
    assert result["context_tokens"] < 3000 and not calls
    assert result["request_budget"]["input_tokens_estimate"] > 1024
    assert result["reason"] == "request_budget_exceeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["context_resolution", "query_rewrite"])
async def test_raw_numbers_negation_scope_and_prior_answer_exclusion(monkeypatch, stage):
    module, question, raw = setup(stage)
    calls = provider(monkeypatch, module, raw=raw)
    result = await run(stage, question, Settings(_env_file=None))
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload.get("current_question", payload.get("question")) == question
    assert "NEVER_COPY_ASSISTANT" not in json.dumps(payload)
    if stage == "context_resolution":
        assert payload["previous_intents"][0]["user_input"] == prior().question
        assert result["intent_context"]["referenced_user_inputs"][0]["question"] == prior().question
    assert result["request_budget"]["prompt_sha256"] == result["request_messages_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["context_resolution", "query_rewrite"])
@pytest.mark.parametrize("finish", ["stop", "length", None])
async def test_output_limit_and_usage_observations_survive_validation(monkeypatch, stage, finish):
    module, question, raw = setup(stage)
    calls = provider(monkeypatch, module, raw=raw, finish=finish)
    result = await run(stage, question, Settings(_env_file=None))
    trace = result["request_budget"]
    assert len(calls) == 1 and trace["finish_reason"] == finish
    assert trace["provider_usage"] == {
        "prompt_tokens": 902,
        "completion_tokens": 73,
        "total_tokens": 975,
    }
    if finish != "stop":
        assert result["status"] == ("clarify" if stage == "context_resolution" else "unavailable")
        assert not result.get("search_queries") and "intent_context" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["context_resolution", "query_rewrite"])
async def test_usage_absent_is_null_and_request_errors_do_not_leak(monkeypatch, stage):
    module, question, raw = setup(stage)
    provider(monkeypatch, module, raw=raw, usage=False)
    result = await run(stage, question, Settings(_env_file=None))
    assert result["request_budget"]["provider_usage"] is None
    provider(monkeypatch, module, raw=raw, error=RuntimeError("sk-NOT-FOR-TRACE"))
    result = await run(stage, question, Settings(_env_file=None))
    assert result["request_budget"]["request_attempted"]
    assert not result["request_budget"]["response_received"]
    assert "sk-NOT-FOR-TRACE" not in json.dumps(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["context_resolution", "query_rewrite"])
async def test_external_cancellation_is_not_a_fallback_result(monkeypatch, stage):
    module, question, raw = setup(stage)
    provider(monkeypatch, module, raw=raw, error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await run(stage, question, Settings(_env_file=None))


@pytest.mark.asyncio
async def test_cold_and_disabled_paths_do_not_initialize_client_even_with_infeasible_budget(
    monkeypatch,
):
    def forbidden():
        pytest.fail("Should not initialize SDK")

    monkeypatch.setattr(context, "chat_client", forbidden)
    monkeypatch.setattr(query_plan, "chat_client", forbidden)
    settings = Settings(
        _env_file=None, query_rewrite_enabled=False, context_resolution_total_token_budget=1024
    )
    cold = await context.resolve_question("Explain CedarHub backup.", [], settings)
    disabled = await query_plan.plan_query("Explain CedarHub backup.", settings)
    assert cold["status"] == "standalone" and not cold["request_attempted"]
    assert disabled["status"] == "disabled" and not disabled["request_attempted"]
    assert "request_budget" not in cold and "request_budget" not in disabled


@pytest.mark.parametrize(
    ("model", "url", "field", "scope"),
    [
        (
            "qwen3.8-flash",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "max_completion_tokens",
            "reasoning_and_answer",
        ),
        ("deepseek-v4-flash", "https://api.deepseek.com/v1", "max_tokens", "reasoning_and_answer"),
        (
            "mimo-v2.6-pro",
            "https://api.xiaomimimo.com/v1",
            "max_completion_tokens",
            "reasoning_and_answer",
        ),
        ("qwen3.8-flash", "https://example.invalid/v1", "max_tokens", "unverified"),
    ],
)
def test_endpoint_semantics_preserved_and_contract_records_each_budget(model, url, field, scope):
    settings = Settings(_env_file=None, llm_model=model, llm_base_url=url)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]
    for stage in ("context_resolution", "query_rewrite"):
        trace = measure(messages, settings, stage)
        assert (trace["output_limit_parameter"], trace["output_limit_scope"]) == (field, scope)
        assert contract(settings, stage)["output_token_budget"] == 2048
    assert query_plan.rewrite_contract(settings)["request_budget"] == contract(
        settings, "query_rewrite"
    )
