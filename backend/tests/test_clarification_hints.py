"""Deterministic availability/state controls, not semantic success measurements."""

import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import conversation_context as context
from knowflow.app.clarification_hints import memory_invalidation
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.memory_models import UserMemory
from knowflow.app.models import Conversation, ConversationTurn, Document, KnowledgeBase, utc_now
from sqlalchemy import select
from test_clarification_context import provider, turn
from test_conversations import create, send, workspace
from test_memory import add_memory


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 7])
async def test_available_files_bounded_isolated_not_bindings_or_model_inputs(monkeypatch, count):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            other_owner = await session.scalar(
                select(KnowledgeBase.owner_id).where(KnowledgeBase.id == UUID(kb))
            )
            other = KnowledgeBase(owner_id=other_owner, name="Other KB")
            session.add(other)
            await session.flush()
            for i in range(count + 2):
                session.add(
                    Document(
                        knowledge_base_id=UUID(kb) if i <= count else other.id,
                        uploaded_by=conv.owner_id,
                        filename=f"FILE-{i:02d}.pdf",
                        storage_key=str(uuid4()),
                        media_type="application/pdf",
                        file_size=10,
                        sha256=f"{i:064x}",
                        status="processed" if i != count else "failed",
                    )
                )
            await session.commit()
        requests = provider(
            monkeypatch,
            [{"status": "clarify", "question": "请补充具体对象。", "reference_turn_ids": []}],
        )
        key = str(uuid4())
        result = (await send(client, auth[0], cid, "它是什么？", key=key)).json()
        assert result["status"] == "needs_clarification" and not calls
        assert not result["resolved_question"] and not result["response"]["citations"]
        hints = result["response"]["clarification"]
        assert len(hints["documents"]) == min(count, 5)
        assert hints["has_more"] == (count > 5)
        assert hints["kind"] == "available_documents_not_relevance_ranked"
        assert [d["filename"] for d in hints["documents"]] == [
            f"FILE-{i:02d}.pdf" for i in range(min(count, 5))
        ]
        assert "FILE-" not in json.dumps(requests)
        again = (await send(client, auth[0], cid, "它是什么？", key=key)).json()
        assert again == result and len(requests) == 1
        restored = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert restored["turns"][0]["response"]["clarification"] == hints
        assert (
            await client.get(f"/api/v1/conversations/{cid}", headers=auth[1])
        ).status_code == 404


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation,code",
    [
        ("disabled", "memory_disabled"),
        ("expired", "memory_expired"),
        ("revision", "memory_revised"),
        ("deleted", "memory_unavailable"),
    ],
)
async def test_old_pending_gets_actual_reason_without_reviving_content(monkeypatch, mutation, code):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        memory = (
            await add_memory(client, auth[0], kb, key="my scope", content="PRIVATE-OLD-OBJECT")
        ).json()
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            mid = UUID(memory["id"])
            m = await session.get(UserMemory, mid)
            if mutation == "deleted":
                await session.delete(m)
            elif mutation == "expired":
                m.expires_at = utc_now() - timedelta(seconds=1)
            elif mutation == "disabled":
                m.status = "disabled"
            else:
                m.revision += 1
                m.content = "PRIVATE-NEW-OBJECT"
            session.add_all(
                [
                    ConversationTurn(
                        conversation_id=UUID(cid),
                        sequence=1,
                        request_id=uuid4(),
                        request_sha256="0" * 64,
                        question="my scope",
                        resolved_question="PRIVATE-OLD-OBJECT",
                        status="completed",
                        resolution={
                            "reference_turn_ids": [f"memory:{mid}:0"],
                            "memory_context": {
                                "selected": [{"id": str(mid), "revision": 0}],
                                "referenced_ids": [str(mid)],
                            },
                        },
                        response={"citations": []},
                    ),
                    ConversationTurn(
                        conversation_id=UUID(cid),
                        sequence=2,
                        request_id=uuid4(),
                        request_sha256="0" * 64,
                        question="它呢？",
                        status="needs_clarification",
                        resolution={
                            "pending_intent": {
                                "question": "",
                                "supplements": [],
                                "requires_restatement": True,
                            }
                        },
                        response={"citations": []},
                    ),
                ]
            )
            conv = await session.get(Conversation, UUID(cid))
            conv.revision = 2
            await session.commit()
        requests = provider(
            monkeypatch,
            [
                {"status": "clarify", "question": "请明确对象。", "reference_turn_ids": []},
                {"status": "clarify", "question": "请明确对象。", "reference_turn_ids": []},
                {
                    "status": "standalone",
                    "question": "A complete new subject question",
                    "reference_turn_ids": [],
                },
            ],
        )
        for revision in [2, 3]:
            result = (await send(client, auth[0], cid, "它的周期呢？", revision)).json()
            assert result["status"] == "needs_clarification" and not calls
            assert result["resolution"]["restatement_reasons"] == [code]
            assert result["resolution"]["pending_intent"]["restatement_reasons"] == [code]
        assert "PRIVATE-" not in json.dumps(requests)
        assert all(r["pending_clarification"]["restatement_reasons"] == [code] for r in requests)
        new = (await send(client, auth[0], cid, "A complete new subject question", 4)).json()
        assert new["status"] == "completed" and calls[-1][1] == "A complete new subject question"
        assert new["response"]["clarification"] is None
        assert not new["resolution"].get("restatement_reasons")


@pytest.mark.asyncio
async def test_invalidation_reason_survives_unavailable_classifier(monkeypatch):
    pending = turn(
        "它呢？",
        status="needs_clarification",
        resolution={
            "pending_intent": {
                "question": "",
                "supplements": [],
                "requires_restatement": True,
                "restatement_reasons": ["memory_disabled", "PRIVATE-UNKNOWN-REASON"],
            }
        },
    )

    def unavailable():
        raise RuntimeError("PRIVATE-provider")

    monkeypatch.setattr(context, "chat_client", unavailable)
    result = await context.resolve_question("它呢？", [pending], get_settings())
    assert result["status"] == "clarify" and "已停用" in result["question"]
    assert result["restatement_reasons"] == ["memory_disabled"]
    assert "PRIVATE" not in json.dumps(result)


def test_memory_scope_rejection_does_not_disclose_another_owner():
    owner, kb = uuid4(), uuid4()
    memory = SimpleNamespace(
        owner_id=uuid4(), namespace="user", status="active", revision=0, expires_at=None
    )
    assert memory_invalidation(memory, {"revision": 0}, owner, kb) == "memory_scope_changed"
    memory.owner_id = owner
    memory.namespace = f"kb:{uuid4()}"
    assert memory_invalidation(memory, {"revision": 0}, owner, kb) == "memory_scope_changed"
