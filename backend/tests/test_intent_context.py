"""Verify original constraints reach every answer gate even when a rewrite drops them."""

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import api, conversation_context
from knowflow.app.config import get_settings
from knowflow.app.intent_context import build_intent_context, task_question
from knowflow.app.models import Chunk
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import QueryRequest, RelevanceAssessment, RetrievalDiagnostics
from knowflow.app.token_budget import get_token_counter


@pytest.mark.parametrize(
    "original,current",
    [
        ("仅在测试环境，排除生产环境，查网关限额。", "它的版本改为2025，不是2024。"),
        (
            "Only staging, exclude production. Do not include archived records.",
            "For it use 2025, not 2024, and keep the same exclusions.",
        ),
        ("仅比较启用审计且未取消的任务，不查金额。", "它们按生效日期，不按发布日期。"),
    ],
)
def test_raw_conditions_are_preserved_separately_from_an_incomplete_hint(original, current):
    payload = {
        "current_question": current,
        "previous_intents": [
            {
                "turn_id": "selected",
                "user_intent": "gateway limits",
                "user_input": original,
            },
            {
                "turn_id": "unselected",
                "user_input": "UNRELATED_PRIVATE_INPUT",
                "user_intent": "UNRELATED_HINT",
            },
        ],
    }
    context = build_intent_context(payload, ["selected"])
    rendered = task_question("gateway limits", context)
    assert current in rendered and original in rendered
    assert "UNRELATED" not in rendered and "override prior" in rendered
    assert context["referenced_user_inputs"] == [{"turn_id": "selected", "question": original}]


def test_inherited_raw_inputs_survive_next_followup_without_old_assistant_facts():
    inherited = {"turn_id": "root", "question": "Only staging; do not include production."}
    turn = SimpleNamespace(
        id="recent",
        status="completed",
        question="Use version 2025",
        resolved_question="gateway limits 2025",
        resolution={
            "intent_context": {
                "referenced_user_inputs": [inherited, inherited],
            }
        },
        response={"answer": "OLD_ASSISTANT_FACT", "citations": []},
    )
    payload = conversation_context.context_payload(
        "它的限额呢？", [turn], get_token_counter(get_settings().chunk_tokenizer_path)
    )
    context = build_intent_context(payload, ["recent"])
    assert context["referenced_user_inputs"] == [
        inherited,
        {
            "turn_id": "recent",
            "question": "Use version 2025",
        },
    ]
    assert "OLD_ASSISTANT_FACT" not in json.dumps(payload)
    assert "Only staging" in task_question("gateway limits", context)


def test_numeric_values_in_record_identifiers_are_not_permitted_constraints():
    payload = {
        "current_question": "它的限额呢？",
        "previous_intents": [
            {
                "turn_id": "00000000-0000-0000-0000-000000010000",
                "user_intent": "gateway",
                "user_input": "gateway",
            }
        ],
    }
    with pytest.raises(ValueError, match="numeric"):
        conversation_context.validate_resolution(
            json.dumps(
                {
                    "status": "resolved",
                    "question": "gateway 10000",
                    "reference_turn_ids": [payload["previous_intents"][0]["turn_id"]],
                }
            ),
            payload,
        )


def test_client_cannot_supply_or_serialize_internal_intent_context():
    body = QueryRequest.model_validate(
        {"question": "public question", "_intent_context": {"current_user_input": "FORGED"}}
    )
    assert body._intent_context is None
    assert "_intent_context" not in QueryRequest.model_json_schema()["properties"]
    assert task_question(body.question, None) == body.question
    body._intent_context = {"current_user_input": "internal"}
    assert "_intent_context" not in body.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
async def test_raw_constraints_reach_answer_gates_not_retrieval(monkeypatch, empty):
    kb, doc, cid = uuid4(), uuid4(), uuid4()
    original = "仅测试环境，不要生产环境；生效年份改为2025，不是2024。"
    intent = build_intent_context(
        {
            "current_question": original,
            "previous_intents": [
                {
                    "turn_id": "root",
                    "user_input": "Compare gateway policies, exclude archived rows.",
                    "user_intent": "gateway policies",
                }
            ],
        },
        ["root"],
    )
    # Deliberately bad hint: the test verifies transport, not an LLM's ability to reason.
    hint = "production gateway policies"
    body = QueryRequest(question=hint, strategy="keyword", expand_references=False)
    body._intent_context = intent
    calls = []
    chunk = Chunk(
        id=cid,
        document_id=doc,
        knowledge_base_id=kb,
        ordinal=0,
        content="The staging gateway uses the 2025 policy.",
        source_spans=[],
        page_number=None,
        section="scope",
    )

    async def owned(*_):
        pass

    async def retrieve(_session, _kb, question, *_args, **_kwargs):
        assert question == hint
        return [] if empty else [RetrievalHit(chunk, "synthetic.txt", keyword_rank=1)]

    async def assess(question, rows, *_):
        calls.append(("relevance", question))
        return {
            r["id"]: RelevanceAssessment(
                verdict="direct", reason="synthetic control", supporting_quotes=[r["content"]]
            )
            for r in rows
        }, RetrievalDiagnostics(status="empty" if empty else "complete")

    async def generate(question, evidence, *_, **kwargs):
        calls.append(("generation", question))
        return f"Staging policy [{evidence[0]['id']}]"

    async def guard(question, answer, *_, **kwargs):
        assert kwargs["current_user_input"] == original
        calls.append(("guard", question))
        return answer, {"status": "synthetic_control"}

    class Session:
        async def scalars(self, *_):
            return []

    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "retrieve", retrieve)
    monkeypatch.setattr(api, "assess_retrieval", assess)
    monkeypatch.setattr(api, "generate_answer", generate)
    monkeypatch.setattr(api, "guard_answer", guard)
    result = await api.run_query(kb, body, SimpleNamespace(id=uuid4()), Session())
    assert [c[0] for c in calls] == (
        ["relevance"]
        if empty
        else [
            "relevance",
            "generation",
            "guard",
        ]
    )
    for _, question in calls:
        assert original in question and "exclude archived" in question
        assert question == task_question(hint, intent)
    assert result.query_plan["intent_context"] == intent
    assert result.refused is empty


@pytest.mark.asyncio
async def test_resolver_builds_context_from_raw_records_not_provider_fabrication(monkeypatch):
    prior = SimpleNamespace(
        id="prior",
        status="completed",
        question="Only staging, not production.",
        resolved_question="gateway policy",
        resolution={},
        response={"citations": []},
    )
    requests = []

    async def completion(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "resolved",
                                "question": "gateway limit",
                                "reference_turn_ids": ["prior"],
                                "intent_context": {"current_user_input": "FORGED"},
                            }
                        )
                    ),
                )
            ]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    fake.with_options = lambda **_: fake
    monkeypatch.setattr(conversation_context, "chat_client", lambda: fake)
    result = await conversation_context.resolve_question("它的限额呢？", [prior], get_settings())
    assert result["status"] == "resolved" and len(requests) == 1
    assert result["intent_context"]["current_user_input"] == "它的限额呢？"
    assert "Only staging" in json.dumps(result["intent_context"])
    assert "FORGED" not in json.dumps(result)


@pytest.mark.asyncio
async def test_numeric_correction_without_pronoun_enters_resolution_only_with_history(monkeypatch):
    prior = SimpleNamespace(
        id="prior",
        status="completed",
        question="Atlas policy 2024",
        resolved_question=None,
        resolution={},
        response={"citations": []},
    )
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "resolved",
                                "question": "Atlas policy 2025, excluding 2024",
                                "reference_turn_ids": ["prior"],
                            }
                        )
                    ),
                )
            ]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
    fake.with_options = lambda **_: fake
    monkeypatch.setattr(conversation_context, "chat_client", lambda: fake)
    correction = "改为2025，不是2024。"
    result = await conversation_context.resolve_question(correction, [prior], get_settings())
    assert result["status"] == "resolved" and len(calls) == 1
    assert result["intent_context"]["current_user_input"] == correction
    cold = await conversation_context.resolve_question("只查2025年的政策。", [], get_settings())
    assert cold["status"] == "standalone" and len(calls) == 1
