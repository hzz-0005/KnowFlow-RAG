"""Projection, persistence and failure controls; no semantic accuracy claims."""

import json
from copy import deepcopy
from uuid import UUID, uuid4

import pytest
from knowflow.app import conversation_api as api
from knowflow.app.conversation_context import validate_resolution
from knowflow.app.db import SessionLocal
from knowflow.app.models import Conversation, ConversationTurn, Document
from knowflow.app.partial_lookup import PartialLookupError, validate_partial
from knowflow.app.schemas import QueryResponse
from test_clarification_context import provider
from test_conversations import create, send, workspace

QUESTION = "它的训练数据规模和刚才报纸的发行年数分别是多少？请引用，不要外推。"


def fixture():
    payload = {
        "current_question": QUESTION,
        "previous_intents": [
            {
                "turn_id": "known",
                "user_input": "Kestrel 报纸发行了多少年？",
                "user_intent": "Kestrel 报纸发行了多少年？",
                "inherited_inputs": [],
            }
        ],
        "pending_clarification": {"turn_id": "pending", "requires_restatement": False},
    }
    plan = {
        "segments": [
            {"kind": "blocked", "text": "它的训练数据规模"},
            {"kind": "connector", "text": "和"},
            {"kind": "answerable", "text": "刚才报纸的发行年数"},
            {"kind": "shared", "text": "分别是多少？请引用，不要外推。"},
        ],
        "question": "Kestrel 报纸发行了多少年？请引用，不要外推。",
        "reference_turn_ids": ["known"],
        "depends_on": [],
    }
    return plan, payload


def test_partial_projection_preserves_shared_originals_without_blocked_task():
    plan, payload = fixture()
    result = validate_partial(plan, payload)
    assert "请引用，不要外推" in result["intent_context"]["current_user_input"]
    assert "训练" not in json.dumps(result["intent_context"], ensure_ascii=False)
    assert result["blocked_scopes"] == ["它的训练数据规模"]
    assert result["full_request_complete"] is False


@pytest.mark.parametrize("present", [True, False])
def test_no_independent_part_and_omitted_assessment_are_distinct(present):
    _, payload = fixture()
    data = {"status": "clarify", "question": "请补充模型名。", "reference_turn_ids": []}
    if present:
        data["partial_lookup"] = None
    result = validate_resolution(json.dumps(data), payload)
    assert result["partial_lookup_decision"] == ("none" if present else "omitted")
    assert "partial_lookup_rejection" not in result and "partial_lookup" not in result


def test_old_assistant_clarification_is_not_current_classifier_input():
    from knowflow.app.config import get_settings
    from knowflow.app.conversation_context import context_payload
    from knowflow.app.token_budget import get_token_counter
    from test_clarification_context import turn

    pending = turn(
        "某个对象的期限？",
        status="needs_clarification",
        resolution={
            "question": "OLD_ASSISTANT_CLAIM_NEVER_SUPPLY",
            "pending_intent": {"question": "某个对象的期限？", "supplements": []},
        },
    )
    payload = context_payload(
        "它和Archive规则分别是多少？",
        [pending],
        get_token_counter(get_settings().chunk_tokenizer_path),
    )
    assert "OLD_ASSISTANT" not in json.dumps(payload)
    assert payload["pending_clarification"]["original_question"] == pending.question


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("omit", "partial_input_not_exact_partition"),
        ("dependency", "partial_dependent_lookup"),
        ("foreign", "partial_invalid_reference"),
        ("pending", "partial_invalid_reference"),
        ("restate", "partial_restatement_required"),
        ("invent_number", "partial_numeric_changed"),
        ("no_blocked", "partial_missing_scope"),
        ("comparison", "partial_dependency_language"),
    ],
)
def test_reject_partial_does_not_discard_valid_clarifying_question(mutation, reason):
    plan, payload = fixture()
    if mutation == "omit":
        plan["segments"][0]["text"] = "它"
    if mutation == "dependency":
        plan["depends_on"] = ["blocked"]
    if mutation == "foreign":
        plan["reference_turn_ids"] = ["foreign"]
    if mutation == "pending":
        plan["reference_turn_ids"] = ["pending"]
    if mutation == "restate":
        payload["pending_clarification"]["requires_restatement"] = True
    if mutation == "invent_number":
        plan["question"] += "9999"
    if mutation == "no_blocked":
        plan["segments"][0]["kind"] = "shared"
    if mutation == "comparison":
        plan["segments"][-1]["text"] += "比较两者。"
        payload["current_question"] += "比较两者。"
    with pytest.raises(PartialLookupError, match=reason):
        validate_partial(plan, payload)
    validated = validate_resolution(
        json.dumps(
            {
                "status": "clarify",
                "question": "请明确模型。",
                "reference_turn_ids": [],
                "partial_lookup": plan,
            }
        ),
        payload,
    )
    assert validated["status"] == "clarify" and validated["question"] == "请明确模型。"
    assert validated["partial_lookup_rejection"] == reason and "partial_lookup" not in validated


def test_connector_label_cannot_drop_shared_constraints_from_retrieval():
    plan, payload = fixture()
    plan["segments"][-1]["kind"] = "connector"
    result = validate_partial(plan, payload)
    assert "请引用，不要外推" in result["compiled_query"]
    assert "请引用，不要外推" in result["intent_context"]["current_user_input"]


def test_without_history_binding_use_raw_projection_not_model_invented_name():
    plan, payload = fixture()
    plan["reference_turn_ids"] = []
    plan["question"] = "INVENTED_ENTITY 的发行年数是多少？"
    result = validate_partial(plan, payload)
    assert result["compiled_query"] == result["current_scope"]
    assert "INVENTED" not in result["compiled_query"]


def test_numeric_scope_excludes_blocked_numbers_and_keeps_shared_numbers():
    plan, payload = fixture()
    plan["segments"][0]["text"] += "2026"
    plan["segments"][-1]["text"] += "按2025版。"
    payload["current_question"] = "".join(r["text"] for r in plan["segments"])
    with pytest.raises(PartialLookupError, match="numeric_changed"):
        validate_partial(plan, payload)
    plan["question"] += "按2025版。"
    assert validate_partial(plan, payload)["question"].endswith("按2025版。")
    payload["previous_intents"][0]["user_input"] += "排除2026版。"
    plan["question"] += "2026"
    with pytest.raises(PartialLookupError, match="blocked_numbers"):
        validate_partial(plan, payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["answered", "refused", "error", "uncited"])
async def test_partial_dispatch_once_scoped_and_durable_with_remaining_intent(monkeypatch, outcome):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        known_id, pending_id = uuid4(), uuid4()
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            doc = Document(
                knowledge_base_id=UUID(kb),
                uploaded_by=conv.owner_id,
                filename="bulletin.txt",
                storage_key=str(uuid4()),
                media_type="text/plain",
                file_size=10,
                sha256="1" * 64,
                status="processed",
            )
            session.add(doc)
            await session.flush()
            did = doc.id
            session.add_all(
                [
                    ConversationTurn(
                        id=known_id,
                        conversation_id=conv.id,
                        sequence=1,
                        request_id=uuid4(),
                        request_sha256="0" * 64,
                        question="Kestrel 报纸发行了多少年？",
                        status="completed",
                        resolution={
                            "status": "standalone",
                            "question": "Kestrel 报纸发行了多少年？",
                            "version": "conversation-reference-v10-clarification-causes",
                            "reference_turn_ids": [],
                        },
                        response={"answer": "NEVER_REUSE_OLD_ANSWER", "citations": []},
                    ),
                    ConversationTurn(
                        id=pending_id,
                        conversation_id=conv.id,
                        sequence=2,
                        request_id=uuid4(),
                        request_sha256="0" * 64,
                        question="那篇论文的模型是哪个？",
                        status="needs_clarification",
                        resolution={
                            "status": "clarify",
                            "question": "请明确论文。",
                            "pending_intent": {
                                "question": "那篇论文的模型是哪个？",
                                "supplements": [],
                            },
                        },
                        response={"citations": []},
                    ),
                ]
            )
            conv.revision = 2
            await session.commit()
        plan, _ = fixture()
        plan["reference_turn_ids"] = [str(known_id)]
        provider(
            monkeypatch,
            [
                {
                    "status": "clarify",
                    "question": "请补充训练数据所属模型。",
                    "reference_turn_ids": [],
                    "partial_lookup": plan,
                }
            ],
        )
        calls = []

        async def query(kb_id, body, user, session):
            calls.append((body.question, deepcopy(body._intent_context)))
            if outcome == "error":
                raise RuntimeError("PRIVATE-provider-diagnostic")
            return QueryResponse(
                answer="A fresh supported fact [1]." if outcome != "refused" else "当前证据不足。",
                refused=outcome == "refused",
                strategy=body.strategy,
                citations=[
                    dict(
                        chunk_id=uuid4(),
                        document_id=did,
                        filename="bulletin.txt",
                        page_number=None,
                        section=None,
                        quote="A fresh supported fact.",
                    )
                ]
                if outcome == "answered"
                else [],
            )

        monkeypatch.setattr(api, "query_knowledge_base", query)
        key = str(uuid4())
        row = (await send(client, auth[0], cid, QUESTION, 2, key)).json()
        duplicate = (await send(client, auth[0], cid, QUESTION, 2, key)).json()
        assert row == duplicate and len(calls) == 1
        assert row["status"] == "needs_clarification" and row["resolved_question"] is None
        expected = (
            "answered" if outcome == "answered" else "failed" if outcome == "error" else "refused"
        )
        assert row["response"]["partial_answer"]["status"] == expected
        assert row["response"]["refused"] == (outcome != "answered")
        assert "PRIVATE" not in json.dumps(row) and "NEVER_REUSE" not in json.dumps(calls)
        assert "训练" not in json.dumps(calls, ensure_ascii=False)
        assert "请引用，不要外推" in calls[0][1]["current_user_input"]
        assert row["resolution"]["pending_intent"] == {
            "question": "那篇论文的模型是哪个？",
            "supplements": [QUESTION],
        }
        if outcome == "uncited":
            assert "A fresh supported fact" not in row["response"]["answer"]
        saved = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert saved["turns"][-1]["response"] == row["response"]
        async with SessionLocal() as session:
            from knowflow.app.task_memory import checkpoint

            turn = await session.get(ConversationTurn, UUID(row["id"]))
            conversation = await session.get(Conversation, UUID(cid))
            task = checkpoint(conversation, turn)
            assert task.status == "awaiting_clarification" and task.state["goal"] == QUESTION
            assert task.state["partial_answer"]["full_request_complete"] is False
        if outcome == "answered":
            provider(
                monkeypatch,
                [
                    {
                        "status": "resolved",
                        "question": "Kestrel 发行年数与 Orchid 模型训练数据规模？",
                        "reference_turn_ids": [row["id"], str(known_id)],
                    }
                ],
            )
            following = (await send(client, auth[0], cid, "Orchid 模型", 3)).json()
            assert following["status"] == "completed" and len(calls) == 2
            assert QUESTION in json.dumps(calls[-1][1], ensure_ascii=False)


def test_scoped_gap_names_the_files_that_said_nothing():
    from knowflow.app.api import unstated_in_scope
    from knowflow.app.source_scope import SourceScope

    scope = SourceScope(
        {
            "status": "filtered",
            "documents": [
                {"document_id": "a" * 8, "filename": "insightflow.md"},
                {"document_id": "b" * 8, "filename": "aegisops.md"},
            ],
        },
        (),
    )
    message = unstated_in_scope(scope, "它的量化成果呢？")
    assert message is not None
    assert "《insightflow.md》" in message and "《aegisops.md》" in message
    assert "量化成果" in message
    # An unknown or too-wide scope keeps the generic refusal.
    assert unstated_in_scope(SourceScope({"status": "unscoped", "documents": []}, ()), "x") is None
    wide = SourceScope(
        {"status": "filtered", "documents": [{"filename": f"doc{i}.md"} for i in range(6)]},
        (),
    )
    assert unstated_in_scope(wide, "x") is None
