"""Cold lexical cues use bounded classification, with no fabricated antecedents."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from knowflow.app import conversation_context as context
from knowflow.app.config import get_settings
from knowflow.app.round_budget import CURRENT, RoundBudget


def provider(monkeypatch, raw=None, *, finish="stop", error=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish,
                    message=SimpleNamespace(content=json.dumps(raw)),
                )
            ]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **kw: fake if kw == {"max_retries": 0} else pytest.fail("retry")
    monkeypatch.setattr(context, "chat_client", lambda: fake)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "之前发布的 Kestrel 手册中，校准周期是多少？",
        "Which checks does Nova continue to run after a timeout?",
        "在 Boreal 设备手册中，它的维护期是多少？",
        "What does the it field in the Aurora schema mean?",
    ],
)
async def test_self_contained_cold_lexical_cue_is_not_automatically_rejected(monkeypatch, question):
    calls = provider(
        monkeypatch,
        {
            "status": "standalone",
            "question": question,
            "reference_turn_ids": [],
        },
    )
    result = await context.resolve_question(question, [], get_settings())
    assert result["status"] == "standalone" and result["question"] == question
    assert "intent_context" not in result and not result["reference_turn_ids"]
    assert len(calls) == 1
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["no_prior_intents"] is True and not payload["previous_intents"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw,finish,reason",
    [
        (
            {"status": "resolved", "question": "Object", "reference_turn_ids": ["invented"]},
            "stop",
            "invalid_reference_turn",
        ),
        (
            {"status": "resolved", "question": "Object", "reference_turn_ids": []},
            "stop",
            "missing_reference_turn",
        ),
        (
            {"status": "standalone", "question": "Invented object", "reference_turn_ids": []},
            "stop",
            "standalone_input_changed",
        ),
        (
            {"status": "standalone", "question": "它的周期？", "reference_turn_ids": []},
            "length",
            "nonfinal_model_output",
        ),
    ],
)
async def test_no_history_does_not_allow_invented_references_or_rewritten_independent_input(
    monkeypatch, raw, finish, reason
):
    calls = provider(monkeypatch, raw, finish=finish)
    result = await context.resolve_question("它的周期？", [], get_settings())
    assert result["status"] == "clarify" and result["reason"] == reason
    assert not result["reference_turn_ids"] and len(calls) == 1
    assert result["pending_intent"]["question"] == "它的周期？"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("PRIVATE-CONTENT"), TimeoutError()])
async def test_cold_classifier_errors_keep_pending_and_no_private_exception(monkeypatch, error):
    calls = provider(monkeypatch, error=error)
    result = await context.resolve_question("它的周期？", [], get_settings())
    assert result["status"] == "clarify" and len(calls) == 1
    assert result["pending_intent"]["question"] == "它的周期？"
    assert "PRIVATE-CONTENT" not in json.dumps(result)


@pytest.mark.asyncio
async def test_cold_classifier_obeys_full_request_and_round_admission(monkeypatch):
    calls = provider(monkeypatch)
    settings = get_settings().model_copy(update={"context_resolution_total_token_budget": 1})
    result = await context.resolve_question("它的周期？", [], settings)
    assert result["reason"] == "request_budget_exceeded" and not calls
    settings = get_settings().model_copy(update={"query_max_provider_calls": 1})
    budget = RoundBudget(settings)
    budget.admit("prior", "chat", "fixture", 1, 1).update(status="completed")
    token = CURRENT.set(budget)
    try:
        result = await context.resolve_question("它的周期？", [], settings)
        assert result["reason"] == "round_budget_exhausted" and not calls
    finally:
        CURRENT.reset(token)


@pytest.mark.asyncio
async def test_cold_cancellation_is_not_converted_into_success(monkeypatch):
    provider(monkeypatch, error=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await context.resolve_question("它的周期？", [], get_settings())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "这段代码是做什么的？请引用。",
        "这张表格的数据来源是什么？",
        "该文档的结论适用于哪些场景？",
        "上面的说法有依据吗？",
        "前者和其他方案的区别是什么？",
    ],
)
async def test_cold_demonstrative_reference_inputs_route_to_bounded_classifier(
    monkeypatch, question
):
    calls = provider(
        monkeypatch,
        {"status": "clarify", "question": "请说明具体指哪一段内容。", "reference_turn_ids": []},
    )
    result = await context.resolve_question(question, [], get_settings())
    assert result["status"] == "clarify" and result["routing_reason"] == "cold_reference"
    assert result["question"] == "请说明具体指哪一段内容。"
    assert result["pending_intent"] == {"question": question, "supplements": []}
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "TableFormer 论文里用的 PubTabNet 数据集有多少张表格？请引用原文。",
        "库存清单里维护负责人是谁？",
    ],
)
async def test_cue_free_cold_question_keeps_zero_call_fast_path(monkeypatch, question):
    calls = provider(monkeypatch)
    result = await context.resolve_question(question, [], get_settings())
    assert result["status"] == "standalone" and result["routing_reason"] == "cold_independent"
    assert result["request_attempted"] is False and not calls
    assert result["question"] == question


@pytest.mark.parametrize(
    "version",
    [
        "conversation-reference-v7-source-handles",
        "conversation-reference-v8-cold-intent",
        "conversation-reference-v9-cold-demonstrative",
    ],
)
def test_old_independent_topic_still_bounds_history(version):
    old = SimpleNamespace(
        id="old",
        status="completed",
        question="OLD-TOPIC",
        resolved_question=None,
        resolution={},
    )
    root = SimpleNamespace(
        id="new",
        status="completed",
        question="NEW-TOPIC",
        resolved_question=None,
        resolution={
            "version": version,
            "status": "standalone",
            "question": "NEW-TOPIC",
            "reference_turn_ids": [],
        },
    )
    turns, scope = context.active_topic_history([root, old])
    assert turns == [root] and scope["older_loaded_turns_omitted"] == 1
