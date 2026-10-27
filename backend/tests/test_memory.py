"""Persistence/security/state controls, not semantic QA accuracy."""

from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import conversation_api as conversations
from knowflow.app.config import get_settings
from knowflow.app.conversation_context import context_payload
from knowflow.app.db import SessionLocal
from knowflow.app.memory_models import TaskAttempt, TaskMemory, UserMemory
from knowflow.app.memory_store import matches, recall
from knowflow.app.models import Conversation, ConversationTurn, KnowledgeBase, utc_now
from knowflow.app.schemas import QueryResponse
from knowflow.app.token_budget import get_token_counter
from sqlalchemy import select
from test_conversations import create, send, workspace


async def add_memory(client, auth, kb, **overrides):
    body = dict(kind="alias", key="my archive", content="Orchid staging archive", status="active")
    body.update(overrides)
    return await client.post(f"/api/v1/knowledge-bases/{kb}/memories", headers=auth, json=body)


@pytest.mark.asyncio
async def test_memory_lifecycle_namespace_revision_and_deletion(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        made = await add_memory(client, auth[0], kb)
        assert made.status_code == 201, made.text
        row = made.json()
        path = f"/api/v1/knowledge-bases/{kb}/memories/{row['id']}"
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb}/memories", headers=auth[1])
        ).status_code == 404
        assert (
            await client.delete(path + "?expected_revision=0", headers=auth[1])
        ).status_code == 404
        assert (await add_memory(client, auth[0], kb)).status_code == 409
        changed = {k: row[k] for k in ("kind", "key", "content", "status")}
        changed.update(expected_revision=0, content="Orchid production archive")
        updated = await client.put(path, headers=auth[0], json=changed)
        assert updated.status_code == 200 and updated.json()["revision"] == 1
        assert (await client.put(path, headers=auth[0], json=changed)).status_code == 409
        assert (
            await client.delete(path + "?expected_revision=0", headers=auth[0])
        ).status_code == 409
        assert (
            await client.delete(path + "?expected_revision=1", headers=auth[0])
        ).status_code == 200
        async with SessionLocal() as session:
            assert await session.get(UserMemory, UUID(row["id"])) is None
        assert not (
            await client.get(f"/api/v1/knowledge-bases/{kb}/memories", headers=auth[0])
        ).json()["items"]


@pytest.mark.asyncio
async def test_recall_expiry_candidates_disabled_and_cross_project(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        for key, status in [
            ("only active", "active"),
            ("only candidate", "candidate"),
            ("only disabled", "disabled"),
            ("only expired", "active"),
        ]:
            assert (
                await add_memory(client, auth[0], kb, key=key, status=status)
            ).status_code == 201
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            uid = conv.owner_id
            expired = await session.scalar(
                select(UserMemory).where(
                    UserMemory.key == "only expired", UserMemory.owner_id == uid
                )
            )
            expired.expires_at = utc_now() - timedelta(seconds=1)
            other = KnowledgeBase(owner_id=uid, name="other memory scope")
            session.add(other)
            await session.commit()
            other_id = other.id
        assert (
            await add_memory(client, auth[0], kb, scope="user", key="my global")
        ).status_code == 201
        async with SessionLocal() as session:
            selected, trace, _ = await recall(
                session,
                uid,
                UUID(kb),
                "only active only candidate only disabled only expired",
                get_settings(),
            )
            assert [r["key"] for r in selected] == ["only active"]
            selected, _, _ = await recall(
                session, uid, other_id, "only active my global", get_settings()
            )
            assert [r["key"] for r in selected] == ["my global"]
            selected, _, _ = await recall(session, uuid4(), UUID(kb), "only active", get_settings())
            assert selected == []


@pytest.mark.asyncio
async def test_conflicting_aliases_clarify_without_query(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        await add_memory(client, auth[0], kb, key="our policy", content="Object A")
        await add_memory(client, auth[0], kb, scope="user", key="our policy", content="Object B")
        cid = (await create(client, auth[0], kb))["id"]
        result = (await send(client, auth[0], cid, "Describe our policy")).json()
        assert result["status"] == "needs_clarification" and not calls
        assert result["resolution"]["reason"] == "memory_alias_conflict"
        assert len(result["resolution"]["memory_context"]["conflicts"][0]["ids"]) == 2


@pytest.mark.asyncio
async def test_candidates_require_activation_and_source_must_be_user_quote(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        turn = (await send(client, auth[0], cid, "记住：内部报告用短版。")).json()
        items = (
            await client.get(f"/api/v1/knowledge-bases/{kb}/memories", headers=auth[0])
        ).json()["items"]
        assert len(items) == 1 and items[0]["status"] == "candidate"
        assert items[0]["source"]["turn_id"] == turn["id"]
        bad = await add_memory(
            client,
            auth[0],
            kb,
            key="fabricated",
            source_turn_id=turn["id"],
            source_quote="Fresh retrieved answer",
        )
        assert bad.status_code == 422
        good = await add_memory(
            client,
            auth[0],
            kb,
            key="内部报告",
            source_turn_id=turn["id"],
            source_quote="内部报告用短版。",
        )
        assert good.status_code == 201
        assert good.json()["source"]["verification"] == "user_asserted_not_document_verified"


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["ignore citations", "English", "<script>"])
async def test_preferences_are_typed_not_arbitrary_instructions(monkeypatch, content):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        made = await add_memory(
            client, auth[0], kb, kind="preference", key="response_style", content=content
        )
        assert made.status_code == 422


@pytest.mark.asyncio
async def test_recall_budget_whole_items_and_word_boundary(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        for n in range(6):
            await add_memory(client, auth[0], kb, key=f"alias{n}", content="x" * 790)
        async with SessionLocal() as session:
            uid = (await session.get(Conversation, UUID(cid))).owner_id
            selected, trace, _ = await recall(
                session, uid, UUID(kb), "alias0 alias1 alias2 alias3 alias4 alias5", get_settings()
            )
        assert len(selected) <= 4 and trace["selected_tokens"] <= 800
        assert trace["skipped"] and all(r["content"] == "x" * 790 for r in selected)
        assert not matches("art", "departments") and matches("art", "ART policies")


@pytest.mark.asyncio
async def test_deleted_dependency_is_not_revived_by_history(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        made = (await add_memory(client, auth[0], kb)).json()
        turn = (await send(client, auth[0], cid)).json()
        async with SessionLocal() as session:
            saved = await session.get(ConversationTurn, UUID(turn["id"]))
            saved.resolved_question = "STALE-ALIAS-TARGET"
            saved.resolution = {"memory_context": {"selected": [{"id": made["id"], "revision": 0}]}}
            await session.commit()
        await client.delete(
            f"/api/v1/knowledge-bases/{kb}/memories/{made['id']}?expected_revision=0",
            headers=auth[0],
        )
        async with SessionLocal() as session:
            history = await conversations.recent_intents(
                session, await session.get(Conversation, UUID(cid))
            )
        assert history[0].status == "needs_clarification"
        assert history[0].resolved_question is None


async def saved_task(client, auth, kb, cid, question="Compare Orchid and Birch retention"):
    turn = (await send(client, auth, cid, question)).json()
    response = await client.post(
        f"/api/v1/knowledge-bases/{kb}/tasks", headers=auth, json={"source_turn_id": turn["id"]}
    )
    assert response.status_code == 201, response.text
    return response.json(), turn


@pytest.mark.asyncio
async def test_task_save_resume_across_conversations_idempotency_and_cancel(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        task, origin = await saved_task(client, auth[0], kb, cid)
        assert (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[1])).status_code == 404
        duplicate = await client.post(
            f"/api/v1/knowledge-bases/{kb}/tasks",
            headers=auth[0],
            json={"source_turn_id": origin["id"]},
        )
        assert duplicate.json()["id"] == task["id"]
        cid2 = (await create(client, auth[0], kb))["id"]
        key = str(uuid4())
        body = {
            "request_id": key,
            "expected_revision": 0,
            "task_id": task["id"],
            "task_revision": 0,
            "question": "Only production; exclude staging.",
            "strategy": "keyword",
        }
        result = await client.post(
            f"/api/v1/conversations/{cid2}/turns", headers=auth[0], json=body
        )
        assert result.status_code == 200, result.text
        again = await client.post(f"/api/v1/conversations/{cid2}/turns", headers=auth[0], json=body)
        assert again.json() == result.json() and len(calls) == 2
        detail = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()
        assert detail["status"] == "completed" and len(detail["attempts"]) == 1
        assert detail["attempts"][0]["result"]["semantic_quality"] == "unknown"
        assert detail["state"]["user_inputs"][-1]["question"] == body["question"]
        assert "Fresh retrieved answer" not in str(detail["state"])
        assert (
            await client.post(
                f"/api/v1/tasks/{task['id']}/cancel", headers=auth[0], json={"expected_revision": 0}
            )
        ).status_code == 409
        cancelled = await client.post(
            f"/api/v1/tasks/{task['id']}/cancel",
            headers=auth[0],
            json={"expected_revision": detail["revision"]},
        )
        assert cancelled.json()["status"] == "cancelled"
        body.update(
            request_id=str(uuid4()), expected_revision=1, task_revision=cancelled.json()["revision"]
        )
        assert (
            await client.post(f"/api/v1/conversations/{cid2}/turns", headers=auth[0], json=body)
        ).status_code == 409


@pytest.mark.asyncio
async def test_failed_task_retry_lineage_and_source_change(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        task, _ = await saved_task(client, auth[0], kb, cid)

        async def fail(*_):
            raise TimeoutError()

        monkeypatch.setattr(conversations, "query_knowledge_base", fail)
        body = {
            "request_id": str(uuid4()),
            "expected_revision": 1,
            "task_id": task["id"],
            "task_revision": 0,
            "question": "Continue with production only.",
            "strategy": "keyword",
        }
        response = await client.post(
            f"/api/v1/conversations/{cid}/turns", headers=auth[0], json=body
        )
        assert response.json()["status"] == "failed"
        detail = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()

        async def recover(_kb, request, *_):
            assert "Compare Orchid" in request.question
            return QueryResponse(
                answer="controlled", refused=False, strategy="keyword", citations=[]
            )

        monkeypatch.setattr(conversations, "query_knowledge_base", recover)
        async with SessionLocal() as session:
            row = await session.get(TaskMemory, UUID(task["id"]))
            row.state = {**row.state, "source_versions": {"removed-document": {"sha256": "old"}}}
            await session.commit()
        body.update(request_id=str(uuid4()), expected_revision=2, task_revision=detail["revision"])
        response = await client.post(
            f"/api/v1/conversations/{cid}/turns", headers=auth[0], json=body
        )
        assert response.json()["resolution"]["task_resume"]["source_versions_changed"] == [
            "removed-document"
        ]
        detail = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()
        assert len(detail["attempts"]) == 2
        assert detail["attempts"][1]["retry_of_attempt_id"] == detail["attempts"][0]["id"]


@pytest.mark.asyncio
async def test_task_fence_keeps_cancelled_state_and_records_late_attempt(monkeypatch):
    from knowflow.app.task_memory import finish

    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        task, origin = await saved_task(client, auth[0], kb, cid)
        async with SessionLocal() as session:
            row = await session.get(TaskMemory, UUID(task["id"]))
            row.status = "cancelled"
            row.active_turn_id = None
            session.add(TaskAttempt(task_id=row.id, turn_id=UUID(origin["id"])))
            await session.commit()
            turn = await session.get(ConversationTurn, UUID(origin["id"]))
            await finish(session, turn, "completed", {}, {"refused": False}, {})
            await session.commit()
            assert row.status == "cancelled"
            attempt = await session.scalar(select(TaskAttempt).where(TaskAttempt.task_id == row.id))
            assert not attempt.result["task_fence_accepted"]


def test_source_handles_are_ordered_not_assistant_prose():
    turn = SimpleNamespace(
        id="a",
        question="List sources",
        resolved_question=None,
        status="completed",
        resolution={},
        response={
            "answer": "UNTRUSTED-ASSISTANT",
            "citations": [
                {
                    "document_id": "doc",
                    "chunk_id": "chunk",
                    "filename": "file",
                    "quote": "Actual source excerpt",
                }
            ],
        },
    )
    data = context_payload(
        "The first source?", [turn], get_token_counter(get_settings().chunk_tokenizer_path)
    )
    handle = data["previous_intents"][0]["source_results"][0]
    assert handle["result_id"] == "a:source:1" and handle["ordinal"] == 1
    assert "UNTRUSTED-ASSISTANT" not in str(data)


def test_alias_dependencies_follow_inherited_raw_references_and_checkpoint():
    from knowflow.app.memory_store import dependencies
    from knowflow.app.task_memory import checkpoint

    mid = str(uuid4())
    resolution = {
        "memory_context": {
            "selected": [{"id": "unused-pref", "revision": 0}],
            "referenced_ids": [],
        },
        "intent_context": {
            "referenced_user_inputs": [
                {"turn_id": f"memory:{mid}:3", "question": "saved alias clue"}
            ]
        },
    }
    assert dependencies(resolution) == [{"id": mid, "revision": 3}]
    turn = SimpleNamespace(
        id=uuid4(),
        question="What about production?",
        status="completed",
        response={},
        resolution=resolution,
        source_versions={},
        resolved_question="Orchid production archive retention",
    )
    saved = checkpoint(SimpleNamespace(owner_id=uuid4(), knowledge_base_id=uuid4()), turn)
    assert saved.state["memory_dependencies"] == [{"id": mid, "revision": 3}]
    assert saved.state["retrieval_hint"] == turn.resolved_question


def test_saved_task_goal_and_current_updates_are_not_flattened_into_memory_conditions():
    import json

    from knowflow.app.intent_context import task_question

    intent = {
        "current_user_input": "Continue the saved task",
        "referenced_user_inputs": [
            {
                "turn_id": "memory:alias:1",
                "question": "User alias: Orchid staging, excluding production",
            },
            {"turn_id": "previous", "question": "What about production?"},
        ],
        "active_task": {
            "goal": {"user_input": "What about production?", "turn_id": "previous"},
            "user_updates": [{"user_input": "Use the corrected scope", "turn_id": "update"}],
        },
    }
    question = task_question("retrieval hint", intent)
    data = json.loads(question.split("User-intent data (JSON):\n", 1)[1])
    assert [r["turn_id"] for r in data["prior_user_inputs"]] == ["previous"]
    assert data["saved_memory_clues"][0]["turn_id"] == "memory:alias:1"
    assert data["active_task"]["goal"]["user_input"] == "What about production?"
    assert "NEVER override" in question and "latest ordered task user_updates" in question


@pytest.mark.asyncio
async def test_transitive_alias_deletion_invalidates_latest_followup_not_just_first_turn(
    monkeypatch,
):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        made = (await add_memory(client, auth[0], kb)).json()
        await send(client, auth[0], cid, "Describe my archive")
        second = (await send(client, auth[0], cid, "What about production?", 1)).json()
        async with SessionLocal() as session:
            row = await session.get(ConversationTurn, UUID(second["id"]))
            row.resolution = {
                "memory_context": {"selected": [], "referenced_ids": []},
                "intent_context": {
                    "referenced_user_inputs": [
                        {"turn_id": f"memory:{made['id']}:0", "question": "old alias target"}
                    ]
                },
            }
            await session.commit()
        await client.delete(
            f"/api/v1/knowledge-bases/{kb}/memories/{made['id']}?expected_revision=0",
            headers=auth[0],
        )
        async with SessionLocal() as session:
            history = await conversations.recent_intents(
                session, await session.get(Conversation, UUID(cid))
            )
        assert history[0].status == "needs_clarification"
        assert history[0].resolved_question is None


@pytest.mark.asyncio
async def test_task_claim_fences_parallel_conversations_and_recovers_expired_lease(monkeypatch):
    from knowflow.app.conversation_api import TurnRequest, claim_turn

    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        task, _ = await saved_task(client, auth[0], kb, cid)
        cid2 = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            uid = (await session.get(Conversation, UUID(cid))).owner_id
            body = TurnRequest(
                request_id=uuid4(),
                expected_revision=1,
                task_id=UUID(task["id"]),
                task_revision=0,
                question="Continue",
                strategy="keyword",
            )
            _, claimed, _ = await claim_turn(UUID(cid), body, uid, session)
            claimed_id = claimed.id
        # Same task, different conversation: stale revision rejected before another call.
        blocked = await client.post(
            f"/api/v1/conversations/{cid2}/turns",
            headers=auth[0],
            json={
                "request_id": str(uuid4()),
                "expected_revision": 0,
                "task_id": task["id"],
                "task_revision": 1,
                "question": "Continue",
                "strategy": "keyword",
            },
        )
        assert blocked.status_code == 409
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            conv.active_until = utc_now() - timedelta(seconds=1)
            await session.commit()
        resumed = await client.post(
            f"/api/v1/conversations/{cid2}/turns",
            headers=auth[0],
            json={
                "request_id": str(uuid4()),
                "expected_revision": 0,
                "task_id": task["id"],
                "task_revision": 1,
                "question": "Continue",
                "strategy": "keyword",
            },
        )
        assert resumed.status_code == 200, resumed.text
        detail = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()
        assert len(detail["attempts"]) == 2
        assert detail["attempts"][0]["status"] == "interrupted"
        assert detail["attempts"][1]["retry_of_attempt_id"] == detail["attempts"][0]["id"]
        async with SessionLocal() as session:
            old = await session.get(ConversationTurn, claimed_id)
            assert old.status == "pending"  # other conversation lease is recovered separately


@pytest.mark.asyncio
async def test_validated_plan_auto_checkpoint_and_explicit_save_dedup(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):

        async def planned(_kb, body, *_):
            return QueryResponse(
                answer="controlled",
                refused=False,
                strategy="keyword",
                citations=[],
                query_plan={
                    "task_plan": {
                        "version": "quoted-constraints-static-plan-v2-source-ids",
                        "steps": [
                            {"id": "s1", "kind": "lookup", "question": "lookup", "depends_on": []}
                        ],
                    }
                },
            )

        monkeypatch.setattr(conversations, "query_knowledge_base", planned)
        cid = (await create(client, auth[0], kb))["id"]
        turn = (await send(client, auth[0], cid)).json()
        tasks = (await client.get(f"/api/v1/knowledge-bases/{kb}/tasks", headers=auth[0])).json()[
            "items"
        ]
        assert len(tasks) == 1 and tasks[0]["source_turn_id"] == turn["id"]
        explicit = await client.post(
            f"/api/v1/knowledge-bases/{kb}/tasks",
            headers=auth[0],
            json={"source_turn_id": turn["id"]},
        )
        assert explicit.json()["id"] == tasks[0]["id"]


@pytest.mark.asyncio
async def test_typed_style_passes_privately_and_client_cannot_inject_it(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        await add_memory(
            client, auth[0], kb, kind="preference", key="response_style", content="concise"
        )

        async def check(_kb, body, *_):
            assert body._response_style == "concise"
            return QueryResponse(
                answer="controlled", refused=False, strategy="keyword", citations=[]
            )

        monkeypatch.setattr(conversations, "query_knowledge_base", check)
        cid = (await create(client, auth[0], kb))["id"]
        result = (await send(client, auth[0], cid)).json()
        assert result["resolution"]["response_style"] == "concise"
        assert QueryResponse.model_fields.get("_response_style") is None
        from knowflow.app.schemas import QueryRequest

        assert QueryRequest(question="x", _response_style="FORGED")._response_style is None


@pytest.mark.asyncio
async def test_alias_clue_reaches_resolver_not_answer_evidence(monkeypatch):
    import json

    from knowflow.app import conversation_context as context

    async with workspace(monkeypatch) as (client, auth, kb, _):
        made = (await add_memory(client, auth[0], kb)).json()
        seen = []

        async def completion(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            row = payload["previous_intents"][0]
            seen.append(row)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "status": "resolved",
                                    "question": "Orchid staging archive retention",
                                    "reference_turn_ids": [row["turn_id"]],
                                }
                            )
                        ),
                    )
                ]
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        fake.with_options = lambda **_: fake
        monkeypatch.setattr(context, "chat_client", lambda: fake)
        cid = (await create(client, auth[0], kb))["id"]
        result = (await send(client, auth[0], cid, "Describe my archive retention")).json()
        assert result["status"] == "completed"
        assert seen[0]["turn_id"] == f"memory:{made['id']}:0"
        assert seen[0]["sources"] == [] and seen[0]["source_results"] == []
        assert result["resolution"]["memory_context"]["selected"][0]["id"] == made["id"]


@pytest.mark.asyncio
async def test_unused_deleted_preference_does_not_destroy_referent_history(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        made = (
            await add_memory(
                client, auth[0], kb, kind="preference", key="response_style", content="concise"
            )
        ).json()
        await send(client, auth[0], cid, "Describe Orchid staging archive")
        await client.delete(
            f"/api/v1/knowledge-bases/{kb}/memories/{made['id']}?expected_revision=0",
            headers=auth[0],
        )
        async with SessionLocal() as session:
            history = await conversations.recent_intents(
                session, await session.get(Conversation, UUID(cid))
            )
        assert history[0].status == "completed"


@pytest.mark.asyncio
async def test_task_recall_uses_original_alias_but_never_a_deleted_target(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        made = (await add_memory(client, auth[0], kb, key="work alias")).json()
        cid = (await create(client, auth[0], kb))["id"]
        task, _ = await saved_task(client, auth[0], kb, cid, "Describe work alias")
        observed = []

        async def query(_kb, body, *_):
            observed.append(body._intent_context)
            return QueryResponse(
                answer="controlled", refused=False, strategy="keyword", citations=[]
            )

        monkeypatch.setattr(conversations, "query_knowledge_base", query)
        body = {
            "request_id": str(uuid4()),
            "expected_revision": 1,
            "task_id": task["id"],
            "task_revision": 0,
            "question": "Continue",
            "strategy": "keyword",
        }
        result = await client.post(f"/api/v1/conversations/{cid}/turns", headers=auth[0], json=body)
        assert result.status_code == 200, result.text
        assert any(
            r["turn_id"].startswith("memory:") for r in observed[-1]["referenced_user_inputs"]
        )
        await client.delete(
            f"/api/v1/knowledge-bases/{kb}/memories/{made['id']}?expected_revision=0",
            headers=auth[0],
        )
        detail = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()
        body.update(request_id=str(uuid4()), expected_revision=2, task_revision=detail["revision"])
        await client.post(f"/api/v1/conversations/{cid}/turns", headers=auth[0], json=body)
        assert not any(
            r["turn_id"].startswith("memory:") for r in observed[-1]["referenced_user_inputs"]
        )
        assert "Orchid staging archive" not in str(observed[-1])
