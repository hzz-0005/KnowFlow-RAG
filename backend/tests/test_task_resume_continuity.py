"""Actual keyword retrieval with scripted answers; no semantic success scoring."""

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import conversation_api, conversation_context
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.memory_models import TaskMemory, UserMemory
from knowflow.app.models import Chunk, Conversation, Document, utc_now
from knowflow.app.retrieval import retrieve
from knowflow.app.schemas import QueryResponse
from knowflow.app.task_memory import resume_context
from test_conversations import create, send, workspace
from test_memory import add_memory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first,second,correction,continuation",
    [
        ("Helios", "Nadir", "改查 Nadir inspection，排除 Helios calibration。", "继续该任务。"),
        (
            "Cirrus",
            "Iris",
            "Switch to Iris inspection; exclude Cirrus calibration.",
            "Continue the task.",
        ),
    ],
)
async def test_saved_correction_reaches_actual_keyword_retrieval_after_another_new_chat(
    monkeypatch, first, second, correction, continuation
):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        async with SessionLocal() as session:
            cid = (await create(client, auth[0], kb))["id"]
            owner = (await session.get(Conversation, UUID(cid))).owner_id
            for name, field in ((first, "calibration"), (second, "inspection")):
                doc = Document(
                    knowledge_base_id=UUID(kb),
                    uploaded_by=owner,
                    filename=f"{name}.txt",
                    storage_key=uuid4().hex,
                    media_type="text/plain",
                    file_size=20,
                    sha256=uuid4().hex,
                    status="processed",
                )
                session.add(doc)
                await session.flush()
                session.add(
                    Chunk(
                        document_id=doc.id,
                        knowledge_base_id=UUID(kb),
                        ordinal=0,
                        content=f"{name} {field}",
                        search_text=f"{name} {field}",
                        embedding=[0.0] * get_settings().embedding_dimensions,
                        embedding_model="synthetic-not-a-live-embedding",
                    )
                )
            await session.commit()
        observed = []

        async def query(kb_id, body, _user, session):
            hits = await retrieve(session, kb_id, body.question, "keyword", 5, get_settings())
            observed.append(
                {
                    "question": body.question,
                    "intent": body._intent_context,
                    "candidates": [hit.filename for hit in hits],
                }
            )
            return QueryResponse(
                answer="SCRIPTED-NOT-QUALITY", refused=False, strategy="keyword", citations=[]
            )

        monkeypatch.setattr(conversation_api, "query_knowledge_base", query)
        initial = (await send(client, auth[0], cid, first + " calibration")).json()
        saved = await client.post(
            f"/api/v1/knowledge-bases/{kb}/tasks",
            headers=auth[0],
            json={"source_turn_id": initial["id"]},
        )
        assert saved.status_code == 201, saved.text
        task = saved.json()
        for question in (correction, continuation):
            new_cid = (await create(client, auth[0], kb))["id"]
            response = await client.post(
                f"/api/v1/conversations/{new_cid}/turns",
                headers=auth[0],
                json={
                    "question": question,
                    "strategy": "keyword",
                    "request_id": str(uuid4()),
                    "expected_revision": 0,
                    "task_id": task["id"],
                    "task_revision": task["revision"],
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == "completed"
            task = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()
        if folder := os.environ.get("KNOWFLOW_TEST_EVIDENCE_DIR"):
            path = Path(folder) / (f"keyword-trajectory-{first}-" + uuid4().hex + ".json")
            path.write_text(
                json.dumps(
                    {"software_control_not_quality": True, "observed": observed, "task": task},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        assert correction in observed[-1]["question"]
        assert f"{second}.txt" in observed[-1]["candidates"]
        assert correction in json.dumps(observed[-1]["intent"], ensure_ascii=False)
        assert task["attempts"][1]["retry_of_attempt_id"] == task["attempts"][0]["id"]
        assert all(a["result"]["semantic_quality"] == "unknown" for a in task["attempts"])


@pytest.mark.asyncio
async def test_accumulated_task_updates_cannot_bypass_retrieval_input_bound(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        initial = (await send(client, auth[0], cid, "Summarize the maintenance policy.")).json()
        task = (
            await client.post(
                f"/api/v1/knowledge-bases/{kb}/tasks",
                headers=auth[0],
                json={"source_turn_id": initial["id"]},
            )
        ).json()
        async with SessionLocal() as session:
            row = await session.get(TaskMemory, UUID(task["id"]))
            row.state = {
                **row.state,
                "user_updates": [
                    {"turn_id": "update-a", "user_input": "exclude " * 500},
                    {"turn_id": "update-b", "user_input": "include " * 500},
                ],
            }
            await session.commit()
        result = await client.post(
            f"/api/v1/conversations/{cid}/turns",
            headers=auth[0],
            json={
                "question": "Continue",
                "strategy": "keyword",
                "request_id": str(uuid4()),
                "expected_revision": 1,
                "task_id": task["id"],
                "task_revision": task["revision"],
            },
        )
        assert result.status_code == 200
        assert result.json()["status"] == "needs_clarification"
        assert len(calls) == 1  # The original query only; no truncated task is dispatched.
        refreshed = (await client.get(f"/api/v1/tasks/{task['id']}", headers=auth[0])).json()
        assert refreshed["state"]["user_updates"] == row.state["user_updates"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["revision", "disabled", "expired"])
async def test_real_memory_dependency_then_mutation_does_not_revive_old_object(
    monkeypatch, mutation
):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        made = (
            await add_memory(client, auth[0], kb, key="my instrument", content="Helios calibration")
        ).json()
        payloads = []

        async def classify(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            payloads.append(payload)
            if payload.get("pending_clarification", {}).get("requires_restatement"):
                output = {
                    "status": "clarify",
                    "question": "请明确当前对象。",
                    "reference_turn_ids": [],
                }
            else:
                mid = f"memory:{made['id']}:0"
                assert any(r["turn_id"] == mid for r in payload["previous_intents"])
                output = {
                    "status": "resolved",
                    "question": "Helios calibration",
                    "reference_turn_ids": [mid],
                }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content=json.dumps(output)),
                    )
                ]
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=classify)))
        fake.with_options = lambda **_: fake
        monkeypatch.setattr(conversation_context, "chat_client", lambda: fake)
        cid = (await create(client, auth[0], kb))["id"]
        first = (await send(client, auth[0], cid, "my instrument details")).json()
        assert first["status"] == "completed" and calls[-1][1] == "Helios calibration"
        assert first["resolution"]["memory_context"]["referenced_ids"] == [made["id"]]
        if mutation == "expired":
            async with SessionLocal() as session:
                row = await session.get(UserMemory, UUID(made["id"]))
                row.expires_at = utc_now() - timedelta(seconds=1)
                await session.commit()
        else:
            updated = await client.put(
                f"/api/v1/knowledge-bases/{kb}/memories/{made['id']}",
                headers=auth[0],
                json={
                    "kind": "alias",
                    "key": made["key"],
                    "content": "Nadir inspection",
                    "expected_revision": 0,
                    "status": "disabled" if mutation == "disabled" else "active",
                },
            )
            assert updated.status_code == 200, updated.text
        second = (await send(client, auth[0], cid, "它的周期？", 1)).json()
        assert second["status"] == "needs_clarification" and len(calls) == 1
        assert payloads[-1]["pending_clarification"]["requires_restatement"]
        assert "Helios" not in json.dumps(payloads[-1])
        fresh = (await create(client, auth[0], kb))["id"]
        third = (await send(client, auth[0], fresh, "Nadir inspection")).json()
        assert third["status"] == "completed" and calls[-1][1] == "Nadir inspection"
        assert not third["resolution"]["memory_context"]["selected"]
        assert "Helios" not in third["resolved_question"]


@pytest.mark.asyncio
async def test_pending_user_supplements_and_ordered_updates_reach_search_without_id_numbers():
    task = SimpleNamespace(
        id=uuid4(),
        source_turn_id=uuid4(),
        revision=2,
        state={
            "goal": "Which policy applies?",
            "memory_dependencies": [],
            "pending_intent": {
                "question": "Which policy applies?",
                "supplements": ["Lumen 2024 domestic"],
            },
            "user_updates": [{"turn_id": str(uuid4()), "user_input": "Use 2025, not 2024."}],
        },
    )

    class Session:
        async def get(self, cls, key):
            assert cls is TaskMemory and key == task.id
            return task

    result = await resume_context(Session(), task.id, "Continue", {})
    assert result["status"] == "resolved"
    assert "Lumen 2024 domestic" in result["question"]
    assert "Use 2025, not 2024." in result["question"]
    assert result["question"].index("Lumen 2024 domestic") < result["question"].index("Use 2025")
    from knowflow.app.query_plan import numeric_tokens

    assert set(numeric_tokens(result["question"])) == {"2024", "2025"}
    assert str(task.source_turn_id) not in result["question"]
