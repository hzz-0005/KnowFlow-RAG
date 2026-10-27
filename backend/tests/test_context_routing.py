"""Fixed synthetic routing/transport contracts, not semantic model quality tests."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from knowflow.app import conversation_context as context
from knowflow.app.config import get_settings
from knowflow.app.token_budget import get_token_counter

DATASET = json.loads(
    (Path(__file__).resolve().parents[2] / "evaluation/context_routing_v1/cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", DATASET["cases"], ids=lambda row: row["id"])
async def test_frozen_routing_controls_do_not_depend_on_pronoun_keyword_lists(monkeypatch, case):
    history = []
    for row in case["history"]:
        repeat = row.get("question_repeat")
        question = repeat["text"] * repeat["count"] if repeat else row["question"]
        history.append(
            SimpleNamespace(
                id=row["id"],
                question=question,
                resolved_question=None,
                status="completed",
                resolution={},
                response={"answer": "OLD_ASSISTANT_NOT_EVIDENCE", "citations": []},
            )
        )
    expected = case["expected"]
    calls = []

    async def create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        calls.append(payload)
        assert payload["current_numeric_tokens"] == (
            ["2025", "2024"] if case["id"] == "en_date_correction" else []
        )
        assert "OLD_ASSISTANT_NOT_EVIDENCE" not in json.dumps(payload)
        # Deliberately concatenates raw intent rather than simulating semantic skill.
        question = (
            case["question"]
            if expected["status"] == "standalone"
            else (
                history[0].question + " " + case["question"]
                if expected["status"] == "resolved"
                else "Which object or option do you mean?"
            )
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": expected["status"],
                                "question": question,
                                "reference_turn_ids": expected["references"],
                            }
                        )
                    ),
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **options: client if options == {"max_retries": 0} else None
    monkeypatch.setattr(context, "chat_client", lambda: client)
    result = await context.resolve_question(case["question"], history, get_settings())
    assert result["status"] == expected["status"]
    if case["id"] == "recent_intent_budget_barrier":
        assert not calls and result["reason"] == "recent_intent_budget_exceeded"
        assert result["required_recent_intent_omitted"]["turn_id"] == "recent"
        assert result["pending_intent"]["requires_restatement"]
        assert not result["reference_turn_ids"]
    elif history or case["id"] == "cold_reference":
        assert len(calls) == 1 and result["request_attempted"]
        assert result["routing_reason"] == ("history_present" if history else "cold_reference")
    else:
        assert not calls and not result["request_attempted"]
    if expected["status"] == "resolved":
        assert result["intent_context"]["current_user_input"] == case["question"]
        assert result["intent_context"]["referenced_user_inputs"] == [
            {"turn_id": history[0].id, "question": history[0].question}
        ]
    if expected["status"] == "standalone":
        assert result["question"] == case["question"] and "intent_context" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["length", "content_filter", None])
async def test_nonfinal_resolution_cannot_start_retrieval_even_with_valid_json(monkeypatch, finish):
    history = [
        SimpleNamespace(
            id="prior",
            question="Only staging, never production.",
            resolved_question=None,
            status="completed",
            resolution={},
            response={},
        )
    ]
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish,
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "resolved",
                                "question": "staging retention",
                                "reference_turn_ids": ["prior"],
                            }
                        )
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=82, completion_tokens=7, total_tokens=89),
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **_: client
    monkeypatch.setattr(context, "chat_client", lambda: client)
    result = await context.resolve_question("How long?", history, get_settings())
    assert result["status"] == "clarify" and len(calls) == 1
    assert result["reason"] == "nonfinal_model_output"
    assert "intent_context" not in result and not result["reference_turn_ids"]
    assert result["provider_usage"]["completion_tokens"] == 7


def test_budget_barrier_is_reachable_with_valid_size_messages_and_bounded_inherited_intent():
    settings = get_settings()
    counter = get_token_counter(settings.chunk_tokenizer_path)
    inherited = [{"turn_id": f"origin-{i}", "question": "excluded " * 200} for i in range(10)]
    latest = SimpleNamespace(
        id="latest",
        question="recent " * 500,
        resolved_question="recent " * 500,
        status="completed",
        response={},
        resolution={"intent_context": {"referenced_user_inputs": inherited}},
    )
    older = SimpleNamespace(
        id="old",
        question="Other product",
        resolved_question=None,
        status="completed",
        resolution={},
        response={},
    )
    assert len(latest.question) <= 4000 and all(len(r["question"]) <= 4000 for r in inherited)
    raw_intent = {"current_user_input": latest.question, "referenced_user_inputs": inherited}
    assert counter.count(json.dumps(raw_intent, ensure_ascii=False)) <= 3000
    payload = context.context_payload("How long?", [latest, older], counter)
    assert payload["previous_intents"] == []
    assert payload["required_recent_intent_omitted"]["turn_id"] == "latest"


@pytest.mark.asyncio
async def test_history_classifier_failure_clarifies_instead_of_silently_dropping_context(
    monkeypatch,
):
    def broken():
        raise RuntimeError("PRIVATE-KEY-DETAIL")

    monkeypatch.setattr(context, "chat_client", broken)
    prior = SimpleNamespace(
        id="prior",
        question="Only staging, not production.",
        resolved_question=None,
        status="completed",
        resolution={},
        response={},
    )
    result = await context.resolve_question("How long?", [prior], get_settings())
    assert result["status"] == "clarify" and not result["request_attempted"]
    assert "PRIVATE-KEY-DETAIL" not in json.dumps(result)


def test_numeric_rejection_has_only_explicit_counts_not_untrusted_error_text():
    payload = {
        "current_question": "Use 2025, not 2024.",
        "previous_intents": [
            {"turn_id": "prior", "user_intent": "Ledger rules", "user_input": "Ledger rules"}
        ],
    }
    with pytest.raises(context.ResolutionError) as caught:
        context.validate_resolution(
            json.dumps(
                {
                    "status": "resolved",
                    "question": "Ledger rules 2025",
                    "reference_turn_ids": ["prior"],
                }
            ),
            payload,
        )
    assert caught.value.reason == "numeric_constraints_changed"
    assert caught.value.details == {
        "missing_current_numeric_count": 1,
        "unpermitted_numeric_count": 0,
    }


@pytest.mark.asyncio
async def test_external_cancellation_during_history_classification_propagates(monkeypatch):
    async def cancelled(**_):
        raise asyncio.CancelledError

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=cancelled)))
    client.with_options = lambda **_: client
    monkeypatch.setattr(context, "chat_client", lambda: client)
    prior = SimpleNamespace(
        id="prior",
        question="Only staging.",
        resolved_question=None,
        status="completed",
        resolution={},
        response={},
    )
    with pytest.raises(asyncio.CancelledError):
        await context.resolve_question("How long?", [prior], get_settings())
