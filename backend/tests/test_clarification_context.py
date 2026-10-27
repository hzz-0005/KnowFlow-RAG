"""Synthetic controls for clarification continuity, not a semantic quality baseline."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import conversation_context as context
from knowflow.app.config import get_settings
from knowflow.app.token_budget import get_token_counter


def turn(question, *, status="completed", resolution=None):
    return SimpleNamespace(
        id=uuid4(),
        question=question,
        resolved_question=None,
        status=status,
        resolution=resolution or {},
        response={"answer": "NEVER_USE_OLD_ANSWER", "citations": []},
    )


def provider(monkeypatch, outputs):
    requests = []

    async def create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        requests.append(payload)
        output = outputs.pop(0)
        if callable(output):
            output = output(payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps(output, ensure_ascii=False)),
                )
            ]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **_: fake
    monkeypatch.setattr(context, "chat_client", lambda: fake)
    return requests


@pytest.mark.asyncio
async def test_cold_clarification_bare_name_resumes_original_request(monkeypatch):
    provider(
        monkeypatch,
        [
            {
                "status": "clarify",
                "question": "请补充具体对象。",
                "reference_turn_ids": [],
            }
        ],
    )
    first = await context.resolve_question("它是否直接输出坐标？", [], get_settings())
    pending = turn("它是否直接输出坐标？", status="needs_clarification", resolution=first)
    requests = provider(
        monkeypatch,
        [
            {
                "status": "resolved",
                "question": "结构解码器是否直接输出坐标？",
                "reference_turn_ids": [str(pending.id)],
            }
        ],
    )
    result = await context.resolve_question("结构解码器", [pending], get_settings())
    assert result["status"] == "resolved" and result["mode"] == "pending_clarification"
    assert result["original_question"] == "结构解码器"
    assert requests[0]["pending_clarification"]["original_question"] == pending.question
    assert "NEVER_USE_OLD_ANSWER" not in json.dumps(requests)


@pytest.mark.asyncio
async def test_repeated_clarification_preserves_all_user_supplements_and_can_switch(monkeypatch):
    pending = turn(
        "它在2025年是否启用？",
        status="needs_clarification",
        resolution={
            "question": "哪个系统？",
            "pending_intent": {
                "question": "它在2025年是否启用？",
                "supplements": [],
            },
        },
    )
    requests = provider(
        monkeypatch,
        [
            {"status": "clarify", "question": "哪个版本的检索服务？", "reference_turn_ids": []},
            {"status": "standalone", "question": "Explain PDF OCR", "reference_turn_ids": []},
            {"status": "standalone", "question": "Describe indexing", "reference_turn_ids": []},
        ],
    )
    second = await context.resolve_question("检索服务", [pending], get_settings())
    assert second["pending_intent"]["supplements"] == ["检索服务"]
    latest = turn("检索服务", status="needs_clarification", resolution=second)
    switched = await context.resolve_question("Explain PDF OCR", [latest, pending], get_settings())
    assert switched["status"] == "standalone" and "pending_intent" not in switched
    assert requests[1]["pending_clarification"]["user_supplements"] == ["检索服务"]
    completed = turn("Explain PDF OCR")
    # A completed topic consumes the clarification; history-aware routing still classifies
    # the next independent topic, then passes its exact wording without old constraints.
    independent = await context.resolve_question(
        "Describe indexing", [completed, latest], get_settings()
    )
    assert independent["status"] == "standalone"
    assert len(requests) == 3


def test_resolution_cannot_borrow_numbers_from_unreferenced_turns():
    selected, other = turn("Report 2025 policy"), turn("Revenue 9000")
    payload = context.context_payload(
        "它在2025年呢？", [selected, other], get_token_counter(get_settings().chunk_tokenizer_path)
    )
    with pytest.raises(ValueError, match="numeric"):
        context.validate_resolution(
            json.dumps(
                {
                    "status": "resolved",
                    "question": "Report 2025 revenue 9000",
                    "reference_turn_ids": [str(selected.id)],
                }
            ),
            payload,
        )


@pytest.mark.parametrize(
    "output",
    [
        {"status": "standalone", "question": "Changed question", "reference_turn_ids": []},
        {"status": "standalone", "question": "Original", "reference_turn_ids": ["pending"]},
        {"status": "resolved", "question": "Original", "reference_turn_ids": ["other"]},
        [],
    ],
)
def test_invalid_switch_or_omitted_pending_reference_is_rejected(output):
    payload = {
        "current_question": "Original",
        "previous_intents": [
            {"turn_id": "other", "user_intent": "Other"},
        ],
        "pending_clarification": {
            "turn_id": "pending",
            "original_question": "Which entity?",
            "user_supplements": [],
            "requires_restatement": False,
        },
    }
    with pytest.raises(ValueError):
        context.validate_resolution(json.dumps(output), payload)


@pytest.mark.asyncio
async def test_oversized_pending_requires_restatement_then_allows_new_question(monkeypatch):
    pending = turn(
        "it?",
        status="needs_clarification",
        resolution={
            "pending_intent": {"question": "huge " * 9000, "supplements": []},
        },
    )
    requests = provider(
        monkeypatch,
        [
            {
                "status": "standalone",
                "question": "Describe OCR",
                "reference_turn_ids": [],
            }
        ],
    )
    result = await context.resolve_question("decoder", [pending], get_settings())
    assert result["reason"] == "context_budget_exceeded" and not requests
    reset = turn("decoder", status="needs_clarification", resolution=result)
    resolved = await context.resolve_question("Describe OCR", [reset], get_settings())
    assert resolved["status"] == "standalone"
    assert requests[0]["pending_clarification"]["requires_restatement"]
    assert resolved["context_tokens"] <= 3000


@pytest.mark.asyncio
async def test_provider_failure_keeps_pending_and_redacts_exception(monkeypatch):
    pending = turn("it?", status="needs_clarification")  # historical v1 needs no migration

    def broken():
        raise RuntimeError("SECRET_PROVIDER_DETAIL")

    monkeypatch.setattr(context, "chat_client", broken)
    result = await context.resolve_question("decoder", [pending], get_settings())
    assert result["status"] == "clarify" and result["error_type"] == "RuntimeError"
    assert result["pending_intent"] == {"question": "it?", "supplements": ["decoder"]}
    assert "SECRET_PROVIDER_DETAIL" not in json.dumps(result)


def pending_payload():
    return {
        "current_question": "改查 OceanDesign 的归档周期，排除 MapleSync。",
        "previous_intents": [
            {
                "turn_id": "prior",
                "user_input": "MapleSync 测试归档保留多久？",
                "user_intent": "MapleSync staging retention?",
            }
        ],
        "pending_clarification": {
            "turn_id": "pending",
            "original_question": "另一套备份呢？",
            "user_supplements": [],
            "requires_restatement": False,
        },
    }


def test_current_new_literal_object_supersedes_the_pending_question():
    payload = pending_payload()
    result = {
        "status": "resolved",
        "question": payload["current_question"],
        "reference_turn_ids": ["prior"],
        "current_object_name": "OceanDesign",
        "request_kind": "fact_lookup",
    }

    validated = context.validate_resolution(json.dumps(result), payload)

    assert validated["pending_action"] == "follow_current_named_object"
    assert validated["superseded_pending_turn_id"] == "pending"


def test_a_resolution_without_any_literal_anchor_still_cannot_skip_pending():
    payload = pending_payload()
    payload["current_question"] = "那份归档保留多久？"
    result = {
        "status": "resolved",
        "question": payload["current_question"],
        "reference_turn_ids": ["prior"],
    }

    with pytest.raises(context.ResolutionError) as error:
        context.validate_resolution(json.dumps(result), payload)

    assert "pending_reference_omitted" in str(error.value)


def test_coverage_mode_is_validated_and_carried():
    payload = pending_payload()
    payload["current_question"] = "列出所有项目用到的技术栈"
    payload["previous_intents"] = []
    payload["pending_clarification"] = None
    raw = {
        "status": "standalone",
        "question": payload["current_question"],
        "reference_turn_ids": [],
        "coverage": "sweep",
    }

    validated = context.validate_resolution(json.dumps(raw), payload)
    assert validated["coverage"] == "sweep"

    raw["coverage"] = "everything"
    assert "coverage" not in context.validate_resolution(json.dumps(raw), payload)


def test_exclusion_and_overview_phrasings_map_to_the_right_kinds():
    from knowflow.app.conversation_context import aggregate_list_question

    # Both are decided by the model first; the local rule is only its fallback.
    assert aggregate_list_question("除了 AegisOps，其他项目用了什么工作流引擎？") is True
    assert aggregate_list_question("其余文件有没有提到灰度？") is True
    assert aggregate_list_question("AegisOps 的告警关联是怎么实现的？") is False
