import asyncio
import json
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException
from knowflow.app import conversation_api as api
from knowflow.app import conversation_context as context
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.main import app
from knowflow.app.models import (
    Conversation,
    ConversationTurn,
    Document,
    KnowledgeBase,
    User,
    utc_now,
)
from knowflow.app.schemas import QueryResponse
from knowflow.app.security import create_access_token
from knowflow.app.token_budget import get_token_counter
from sqlalchemy import delete, select, update


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["quota", "provider_error"])
async def test_round_ledger_persists_across_refresh_and_idempotent_retry(monkeypatch, outcome):
    from knowflow.app import api as online
    from knowflow.app import round_budget as rb

    async with workspace(monkeypatch) as (client, auth, kb, _):
        monkeypatch.setattr(get_settings(), "query_max_provider_calls", 1)
        sent = []

        async def pipeline(*args):
            async def operation():
                sent.append(True)
                return {}

            await rb.invoke("first", "chat", "fixture", 100, 100, operation)
            if outcome == "provider_error":
                raise RuntimeError("PRIVATE-PROVIDER-TEXT")
            await rb.invoke("second", "chat", "fixture", 100, 100, operation)

        monkeypatch.setattr(online, "run_query", pipeline)
        monkeypatch.setattr(api, "query_knowledge_base", online.query_knowledge_base)
        cid = (await create(client, auth[0], kb))["id"]
        key = str(uuid4())
        first = (await send(client, auth[0], cid, "Independent quota control", 0, key)).json()
        duplicate = (await send(client, auth[0], cid, "Independent quota control", 0, key)).json()
        assert first == duplicate and len(sent) == 1
        assert first["status"] == ("completed" if outcome == "quota" else "failed")
        snapshot = first["resolution"]["round_budget"]
        assert snapshot["attempted_calls"] == 1
        assert snapshot["blocked_calls"] == (1 if outcome == "quota" else 0)
        assert snapshot["usage_unknown_calls"] == 1 and snapshot["observed_total_tokens"] is None
        assert "PRIVATE-PROVIDER-TEXT" not in json.dumps(first)
        if outcome == "quota":
            assert first["response"]["refused"] and not first["response"]["citations"]
        restored = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert restored["turns"][0]["resolution"]["round_budget"] == snapshot
        assert rb.CURRENT.get() is None


@asynccontextmanager
async def workspace(monkeypatch):
    ids = [uuid4(), uuid4()]
    kb = uuid4()
    calls = []

    async def answer(kb_id, body, user, session):
        calls.append((str(kb_id), body.question))
        return QueryResponse(
            answer="Fresh retrieved answer", refused=False, strategy=body.strategy, citations=[]
        )

    monkeypatch.setattr(api, "query_knowledge_base", answer)

    async def independent_resolution(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        # This fixture is for state tests, not semantic classification. Cold inputs
        # entering the classifier use a scripted clarification; explicit standalone
        # cold controls provide their own scripted supplier in the scenario suite.
        cold = payload.get("no_prior_intents", False)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "clarify" if cold else "standalone",
                                "question": "请补充具体对象。"
                                if cold
                                else payload["current_question"],
                                "reference_turn_ids": [],
                            }
                        )
                    ),
                )
            ]
        )

    fake = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=independent_resolution))
    )
    fake.with_options = lambda **_: fake
    # These integration tests isolate persistence from the remote classifier.
    # Scenarios needing semantic outputs override this explicitly below.
    monkeypatch.setattr(context, "chat_client", lambda: fake)
    async with SessionLocal() as session:
        session.add_all(
            [User(id=i, email=f"conversation-{i}@example.com", password_hash="unused") for i in ids]
        )
        await session.flush()
        session.add(KnowledgeBase(id=kb, owner_id=ids[0], name="Conversation fixture"))
        await session.commit()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            auth = [{"Authorization": f"Bearer {create_access_token(i)}"} for i in ids]
            yield client, auth, str(kb), calls
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.id.in_(ids)))
            await session.commit()


async def create(client, auth, kb):
    result = await client.post(f"/api/v1/knowledge-bases/{kb}/conversations", headers=auth)
    assert result.status_code == 201, result.text
    return result.json()


async def send(client, auth, cid, question="Describe the document", revision=0, key=None):
    return await client.post(
        f"/api/v1/conversations/{cid}/turns",
        headers=auth,
        json={
            "question": question,
            "strategy": "keyword",
            "expected_revision": revision,
            "request_id": key or str(uuid4()),
        },
    )


@pytest.mark.asyncio
async def test_history_owner_isolation_idempotency_and_revision(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        conv = await create(client, auth[0], kb)
        cid, key = conv["id"], str(uuid4())
        first = await send(client, auth[0], cid, key=key)
        assert first.status_code == 200 and first.json()["status"] == "completed"
        duplicate = await send(client, auth[0], cid, key=key)
        assert duplicate.json() == first.json() and len(calls) == 1
        assert (await send(client, auth[0], cid, "Different", key=key)).status_code == 409
        assert (await send(client, auth[0], cid, "Stale revision")).status_code == 409
        path = f"/api/v1/conversations/{cid}"
        history = (await client.get(path, headers=auth[0])).json()
        assert history["revision"] == 1 and history["active_turn_id"] is None
        assert history["turns"][0]["response"]["answer"] == "Fresh retrieved answer"
        for method, route in [
            ("GET", path),
            ("POST", path + "/turns"),
            ("GET", f"/api/v1/knowledge-bases/{kb}/conversations"),
        ]:
            kwargs = (
                {"json": {"question": "test", "request_id": str(uuid4()), "expected_revision": 1}}
                if method == "POST"
                else {}
            )
            assert (
                await client.request(method, route, headers=auth[1], **kwargs)
            ).status_code == 404
        assert (await client.get(path)).status_code == 401
        second = await create(client, auth[0], kb)
        assert (await client.get(f"/api/v1/conversations/{second['id']}", headers=auth[0])).json()[
            "turns"
        ] == []


@pytest.mark.asyncio
async def test_followup_is_fresh_query_and_ambiguity_does_not_retrieve(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        await send(client, auth[0], cid, "TableFormer cell decoder")

        async def resolve(question, history, settings):
            assert len(history) == 1
            assert history[0].question == "TableFormer cell decoder"
            assert "answer" not in history[0].response
            return {
                "status": "resolved",
                "question": "How does TableFormer cell decoder work?",
                "reference_turn_ids": [str(history[0].id)],
            }

        monkeypatch.setattr(api, "resolve_question", resolve)
        follow = (await send(client, auth[0], cid, "它如何工作？", 1)).json()
        assert follow["resolved_question"] == calls[-1][1]
        assert len(calls) == 2 and follow["question"] == "它如何工作？"
        monkeypatch.setattr(api, "resolve_question", context.resolve_question)
        empty = (await create(client, auth[0], kb))["id"]
        ambiguous = (await send(client, auth[0], empty, "这笔9000是什么？")).json()
        assert ambiguous["status"] == "needs_clarification"
        assert ambiguous["response"]["citations"] == [] and len(calls) == 2


@pytest.mark.asyncio
async def test_short_claim_concurrency_expired_recovery_and_late_fence(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        entered, release = asyncio.Event(), asyncio.Event()

        async def blocked(*_):
            entered.set()
            await release.wait()
            return QueryResponse(answer="Late", refused=False, strategy="keyword", citations=[])

        monkeypatch.setattr(api, "query_knowledge_base", blocked)
        key = str(uuid4())
        running = asyncio.create_task(send(client, auth[0], cid, key=key))
        await asyncio.wait_for(entered.wait(), 5)
        # A different request reads the committed revision while model call is still blocked.
        other = await asyncio.wait_for(send(client, auth[0], cid, revision=1), 5)
        assert other.status_code == 409
        duplicate = await send(client, auth[0], cid, key=key)
        assert duplicate.json()["status"] == "pending"
        async with SessionLocal() as session:
            await session.execute(
                update(Conversation)
                .where(Conversation.id == UUID(cid))
                .values(active_until=utc_now() - timedelta(seconds=1))
            )
            await session.commit()
        recovered = await send(client, auth[0], cid, key=key)
        assert recovered.json()["status"] == "interrupted"
        release.set()
        assert (await running).status_code == 409
        history = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert history["turns"][0]["response"] is None
        assert history["turns"][0]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_failure_and_cancellation_are_durable_and_sanitized(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]

        async def fail(*_):
            raise HTTPException(503, "secret-provider-detail")

        monkeypatch.setattr(api, "query_knowledge_base", fail)
        result = (await send(client, auth[0], cid)).json()
        assert result["status"] == "failed" and result["error"]["code"] == "HTTP_503"
        assert "secret-provider-detail" not in json.dumps(result)

        async def cancel(*_):
            raise asyncio.CancelledError

        monkeypatch.setattr(api, "query_knowledge_base", cancel)
        with pytest.raises(asyncio.CancelledError):
            await send(client, auth[0], cid, revision=1)
        history = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert history["active_turn_id"] is None
        assert history["turns"][-1]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_source_identity_invalidated_on_change_and_during_generation(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            doc = Document(
                knowledge_base_id=UUID(kb),
                uploaded_by=conv.owner_id,
                filename="source.md",
                storage_key=str(uuid4()),
                media_type="text/plain",
                file_size=10,
                sha256="1" * 64,
                status="processed",
            )
            session.add(doc)
            await session.commit()
            did = str(doc.id)

        async def answer(*_):
            return QueryResponse(
                answer="Fresh",
                refused=False,
                strategy="keyword",
                citations=[
                    {
                        "chunk_id": str(uuid4()),
                        "document_id": did,
                        "filename": "source.md",
                        "page_number": None,
                        "section": "A",
                        "quote": "Evidence",
                    }
                ],
            )

        monkeypatch.setattr(api, "query_knowledge_base", answer)
        await send(client, auth[0], cid)
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            assert len((await api.recent_intents(session, conv))[0].response["citations"]) == 1
            await session.execute(
                update(Document)
                .where(Document.id == UUID(did))
                .values(updated_at=utc_now() + timedelta(seconds=1))
            )
            await session.commit()
            assert (await api.recent_intents(session, conv))[0].response["citations"] == []

        async def change_during_query(*args):
            async with SessionLocal() as session:
                await session.execute(
                    update(Document)
                    .where(Document.id == UUID(did))
                    .values(updated_at=utc_now() + timedelta(seconds=2))
                )
                await session.commit()
            return await answer(*args)

        monkeypatch.setattr(api, "query_knowledge_base", change_during_query)
        await send(client, auth[0], cid, "Independent document question", 1)
        async with SessionLocal() as session:
            rows = list(
                await session.scalars(
                    select(ConversationTurn)
                    .where(ConversationTurn.conversation_id == UUID(cid))
                    .order_by(ConversationTurn.sequence)
                )
            )
            assert rows[-1].source_versions == {}


def test_context_budget_complete_intents_no_assistant_facts_and_numeric_validation():
    counter = get_token_counter(get_settings().chunk_tokenizer_path)
    turn = SimpleNamespace(
        id=uuid4(),
        question="Report 2025 payment 9000",
        resolved_question=None,
        response={"answer": "UNTRUSTED_FACT", "citations": []},
    )
    payload = context.context_payload("这笔9000呢", [turn], counter)
    assert "UNTRUSTED_FACT" not in json.dumps(payload)
    raw = {
        "status": "resolved",
        "question": "Report 2025 payment 9000 details?",
        "reference_turn_ids": [str(turn.id)],
    }
    assert context.validate_resolution(json.dumps(raw), payload)["status"] == "resolved"
    for bad in [
        raw | {"question": "payment 10000"},
        raw | {"reference_turn_ids": ["invented"]},
        raw | {"reference_turn_ids": []},
    ]:
        with pytest.raises(ValueError):
            context.validate_resolution(json.dumps(bad), payload)
    too_big = SimpleNamespace(
        id=uuid4(), question="huge " * 9000, resolved_question=None, response={}
    )
    selected = context.context_payload("question", [too_big, turn], counter)
    assert selected["previous_intents"] == []
    assert selected["required_recent_intent_omitted"]["turn_id"] == str(too_big.id)


@pytest.mark.asyncio
async def test_resolution_provider_failure_clarifies_and_standalone_skips_provider(monkeypatch):
    def broken():
        raise RuntimeError("private provider detail")

    monkeypatch.setattr(context, "chat_client", broken)
    settings = get_settings()
    assert (await context.resolve_question("Explain TableFormer", [], settings))[
        "status"
    ] == "standalone"
    turn = SimpleNamespace(id=uuid4(), question="TableFormer", resolved_question=None, response={})
    result = await context.resolve_question("它呢", [turn], settings)
    assert result["status"] == "clarify" and result["error_type"] == "RuntimeError"
    assert "private provider detail" not in json.dumps(result)


@pytest.mark.asyncio
async def test_valid_reference_resolution_uses_only_bounded_intents(monkeypatch):
    turn = SimpleNamespace(
        id=uuid4(),
        question="TableFormer architecture",
        resolved_question=None,
        response={"answer": "SECRET_OLD_ANSWER", "citations": []},
    )

    async def completion(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        assert "SECRET_OLD_ANSWER" not in json.dumps(payload)
        assert payload["previous_intents"][0]["turn_id"] == str(turn.id)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "resolved",
                                "question": "TableFormer architecture components?",
                                "reference_turn_ids": [str(turn.id)],
                            }
                        )
                    ),
                )
            ]
        )

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=completion))

        def with_options(self, **options):
            assert options == {"max_retries": 0}
            return self

    monkeypatch.setattr(context, "chat_client", Client)
    result = await context.resolve_question("它的组成？", [turn], get_settings())
    assert result["status"] == "resolved" and result["context_tokens"] <= 3000
    assert result["tokenizer_sha256"]


@pytest.mark.asyncio
async def test_turn_deadline_and_history_pagination(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        for revision in range(3):
            assert (
                await send(client, auth[0], cid, f"Document topic {revision}", revision)
            ).json()["status"] == "completed"
        route = f"/api/v1/conversations/{cid}"
        page = (await client.get(route + "?limit=2", headers=auth[0])).json()
        assert [t["sequence"] for t in page["turns"]] == [2, 3] and page["has_more"]
        older = (await client.get(route + "?before=2&limit=2", headers=auth[0])).json()
        assert [t["sequence"] for t in older["turns"]] == [1] and not older["has_more"]

        async def blocked(*_):
            await asyncio.Event().wait()

        monkeypatch.setattr(api, "query_knowledge_base", blocked)
        monkeypatch.setattr(get_settings(), "query_timeout_seconds", 0.05)
        result = (await send(client, auth[0], cid, "Deadline test", 3)).json()
        assert result["status"] == "failed" and result["error"]["code"] == "timeout"
        assert (await client.get(route, headers=auth[0])).json()["active_turn_id"] is None


@pytest.mark.asyncio
async def test_clarification_survives_reload_retry_and_resolves_before_retrieval(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        first = (await send(client, auth[0], cid, "它在2025年如何工作？")).json()
        assert first["status"] == "needs_clarification" and calls == []
        route = f"/api/v1/conversations/{cid}"
        reloaded = (await client.get(route, headers=auth[0])).json()
        assert reloaded["turns"][0]["resolution"]["pending_intent"]["question"] == first["question"]
        requests = []

        async def completion(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            requests.append(payload)
            assert payload["pending_clarification"]["turn_id"] == first["id"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "status": "resolved",
                                    "question": "检索服务在2025年如何工作？",
                                    "reference_turn_ids": [first["id"]],
                                }
                            )
                        ),
                    )
                ]
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        fake.with_options = lambda **_: fake
        monkeypatch.setattr(context, "chat_client", lambda: fake)
        key = str(uuid4())
        second = (await send(client, auth[0], cid, "检索服务", 1, key)).json()
        duplicate = (await send(client, auth[0], cid, "检索服务", 1, key)).json()
        assert second == duplicate and second["status"] == "completed"
        assert len(requests) == len(calls) == 1
        assert calls[0][1] == "检索服务在2025年如何工作？"
        assert second["question"] == "检索服务" and second["resolved_question"] == calls[0][1]
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            history = await api.recent_intents(session, conv)
            assert context.pending_clarification(history) is None
        assert (await client.get(route, headers=auth[1])).status_code == 404


@pytest.mark.asyncio
async def test_failed_reply_does_not_consume_persisted_clarification(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        first = (await send(client, auth[0], cid, "它是什么？")).json()

        async def failure(*_):
            raise TimeoutError

        monkeypatch.setattr(api, "resolve_question", failure)
        failed = (await send(client, auth[0], cid, "检索服务", 1)).json()
        assert failed["status"] == "failed" and not calls
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            history = await api.recent_intents(session, conv)
            assert context.pending_clarification(history)["turn_id"] == first["id"]


@pytest.mark.asyncio
async def test_topic_switch_clears_pending_without_inheriting_old_numeric_constraints(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        await send(client, auth[0], cid, "它在2025年的配额？")

        async def completion(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "status": "standalone",
                                    "question": payload["current_question"],
                                    "reference_turn_ids": [],
                                }
                            )
                        ),
                    )
                ]
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        fake.with_options = lambda **_: fake
        monkeypatch.setattr(context, "chat_client", lambda: fake)
        topic = "Explain PDF OCR"
        switched = (await send(client, auth[0], cid, topic, 1)).json()
        assert switched["status"] == "completed" and calls[-1][1] == topic
        assert "2025" not in switched["resolved_question"]
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            assert context.pending_clarification(await api.recent_intents(session, conv)) is None


@pytest.mark.asyncio
async def test_internal_user_intent_is_passed_to_query_and_persists_in_turn(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        original = "仅测试环境，不要生产环境。"
        intent = {
            "version": "user-intent-preservation-v1",
            "current_user_input": "它的限额？",
            "referenced_user_inputs": [{"turn_id": "root", "question": original}],
        }

        async def resolve(*_):
            return {"status": "resolved", "question": "gateway limits", "intent_context": intent}

        async def query(_kb, body, *_):
            assert body.question == "gateway limits" and body._intent_context == intent
            return QueryResponse(
                answer="synthetic answer", refused=False, strategy=body.strategy, citations=[]
            )

        monkeypatch.setattr(api, "resolve_question", resolve)
        monkeypatch.setattr(api, "query_knowledge_base", query)
        result = (await send(client, auth[0], cid, "它的限额？")).json()
        assert result["status"] == "completed"
        saved = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert saved["turns"][0]["resolution"]["intent_context"] == intent


@pytest.mark.asyncio
async def test_ellipsis_inherits_raw_exclusions_and_independent_topic_clears_them(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]
        raw = "PineVault only in staging, excluding production."
        first = (await send(client, auth[0], cid, raw)).json()
        requests, query_inputs = [], []

        async def completion(**kwargs):
            payload = json.loads(kwargs["messages"][1]["content"])
            requests.append(payload)
            standalone = payload["current_question"].startswith("Different topic:")
            output = {
                "status": "standalone" if standalone else "resolved",
                "question": payload["current_question"]
                if standalone
                else "PineVault staging retention",
                "reference_turn_ids": [] if standalone else [first["id"]],
            }
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop", message=SimpleNamespace(content=json.dumps(output))
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=90, completion_tokens=12, total_tokens=102),
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        fake.with_options = lambda **_: fake
        monkeypatch.setattr(context, "chat_client", lambda: fake)

        async def query(_kb, body, *_):
            query_inputs.append((body.question, body._intent_context))
            return QueryResponse(
                answer="Synthetic transport control",
                refused=False,
                strategy=body.strategy,
                citations=[],
            )

        monkeypatch.setattr(api, "query_knowledge_base", query)
        key = str(uuid4())
        second = (await send(client, auth[0], cid, "How long?", 1, key)).json()
        duplicate = (await send(client, auth[0], cid, "How long?", 1, key)).json()
        assert second == duplicate and len(requests) == len(query_inputs) == 1
        intent = query_inputs[0][1]
        assert intent["current_user_input"] == "How long?"
        assert intent["referenced_user_inputs"] == [{"turn_id": first["id"], "question": raw}]
        assert second["resolution"]["routing_reason"] == "history_present"
        topic = "Different topic: Describe CedarHub backup policy."
        third = (await send(client, auth[0], cid, topic, 2)).json()
        assert third["resolution"]["status"] == "standalone"
        assert query_inputs[-1] == (topic, None) and len(requests) == 2
        saved = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
        assert saved["turns"][-1]["resolution"]["request_attempted"]
        assert saved["turns"][-1]["resolution"]["provider_usage"]["completion_tokens"] == 12
