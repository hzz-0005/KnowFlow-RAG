"""Coverage accounting must preserve source restrictions and disclose sampling."""

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import api
from knowflow.app.db import SessionLocal
from knowflow.app.document_reference import resolve_document_reference
from knowflow.app.models import Conversation, Document, KnowledgeBase
from knowflow.app.schemas import QueryRequest
from test_conversations import create, send, workspace
from test_document_topic_switch import seed_documents


@pytest.mark.asyncio
@pytest.mark.parametrize("carried", [False, True])
async def test_sweep_stays_inside_explicit_or_carried_document_scope(monkeypatch, carried):
    seen = []

    async def retrieve(*args, **kwargs):
        seen.extend(kwargs["document_ids"])
        return []

    monkeypatch.setattr(api, "retrieve_per_document", retrieve)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            query = QueryRequest(
                question=(
                    "这份文档列出了哪些量化指标？"
                    if carried
                    else f"只看 {docs[1].filename}，列出里面的量化指标。"
                ),
                strategy="keyword",
            )
            query._coverage = "sweep"
            query._request_kind = "fact_lookup"
            if carried:
                query._source_document_ids = (docs[1].id,)
            result = await api.run_query(kb, query, SimpleNamespace(id=owner), session)
            assert seen == [docs[1].id]
            coverage = result.query_plan["coverage"]
            assert coverage["checked"] == 1
            assert coverage["documents"][0]["document_id"] == str(docs[1].id)


@pytest.mark.asyncio
async def test_sweep_directory_limit_is_disclosed_instead_of_calling_twenty_all(monkeypatch):
    async def retrieve(*args, **kwargs):
        return []

    monkeypatch.setattr(api, "retrieve_per_document", retrieve)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            for index in range(21):
                session.add(
                    Document(
                        knowledge_base_id=kb,
                        uploaded_by=owner,
                        filename=f"document-{index:02}.md",
                        storage_key=str(uuid4()),
                        media_type="text/markdown",
                        file_size=10,
                        sha256=uuid4().hex * 2,
                        status="processed",
                    )
                )
            await session.commit()
            query = QueryRequest(question="列出所有文档提及的指标", strategy="keyword")
            query._coverage = "sweep"
            query._request_kind = "fact_lookup"
            result = await api.run_query(kb, query, SimpleNamespace(id=owner), session)
            coverage = result.query_plan["coverage"]
            assert coverage["eligible_documents"] == 21
            assert coverage["checked"] == 20
            assert coverage["omitted_documents"] == 1
            assert coverage["scope_complete"] is False
            assert "20/21" in result.answer


@pytest.mark.asyncio
async def test_one_title_alias_shared_by_two_versions_is_not_two_explicitly_named_files(
    monkeypatch,
):
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            other = Document(
                knowledge_base_id=kb,
                uploaded_by=owner,
                filename="TerraShape_Agent另一版本.md",
                storage_key=str(uuid4()),
                media_type="text/markdown",
                file_size=10,
                sha256=uuid4().hex * 2,
                status="processed",
            )
            session.add(other)
            await session.commit()
            scope = await resolve_document_reference(session, kb, "TerraShape_Agent是做什么的？")
            assert scope.trace["status"] == "ambiguous"
            # A comparison naming both complete filenames remains legitimate.
            scope = await resolve_document_reference(
                session, kb, f"对比 {docs[1].filename} 和 {other.filename} 的结论"
            )
            assert set(scope.document_ids) == {docs[1].id, other.id}


@pytest.mark.asyncio
async def test_excluded_literal_filename_uses_same_scope_semantics_as_its_title(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            scope = await resolve_document_reference(
                session, kb, f"除了 {docs[1].filename}，其他项目用了哪些工具？"
            )
            assert scope.trace["status"] == "filtered"
            assert set(scope.document_ids) == {doc.id for doc in docs if doc.id != docs[1].id}


@pytest.mark.asyncio
async def test_new_unknown_name_with_pronoun_does_not_reuse_the_previous_file(monkeypatch):
    from knowflow.app import conversation_api

    captured = []

    async def resolution(question, history, settings, **kwargs):
        captured.append(question)
        return {"status": "standalone", "question": question, "reference_turn_ids": []}

    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            owner = (await session.get(KnowledgeBase, UUID(kb))).owner_id
            docs = await seed_documents(session, UUID(kb), owner)
            conv = await session.get(Conversation, UUID(cid))
            conv.context_state = {"source_document_ids": [str(docs[0].id)]}
            await session.commit()
        monkeypatch.setattr(conversation_api, "resolve_question", resolution)
        result = (await send(client, auth[0], cid, "OrionLedger这个项目，它的计划是什么？")).json()
        assert captured
        assert result["resolution"].get("routing_reason") != "prior_document_followup"
        assert result["resolution"]["status"] == "standalone"


@pytest.mark.asyncio
async def test_a_pending_other_object_is_not_overridden_by_the_last_cited_file(monkeypatch):
    from knowflow.app import conversation_api

    calls = []

    async def clarify(question, history, settings, **kwargs):
        calls.append(question)
        return {
            "status": "clarify",
            "question": "请指定另一个对象。",
            "reference_turn_ids": [],
            "pending_intent": {"question": question, "supplements": []},
        }

    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            owner = (await session.get(KnowledgeBase, UUID(kb))).owner_id
            docs = await seed_documents(session, UUID(kb), owner)
            conv = await session.get(Conversation, UUID(cid))
            conv.context_state = {"source_document_ids": [str(docs[0].id)]}
            await session.commit()
        monkeypatch.setattr(conversation_api, "resolve_question", clarify)
        await send(client, auth[0], cid, "另一个项目的计划呢？")
        result = (await send(client, auth[0], cid, "它的计划呢？", 1)).json()
        assert result["status"] == "needs_clarification"
        assert len(calls) == 2


@pytest.mark.asyncio
async def test_empty_per_file_scope_never_falls_back_to_unrestricted_retrieval(monkeypatch):
    from knowflow.app import retrieval

    async def forbidden(*args, **kwargs):
        raise AssertionError("Empty scope must not search the whole KB or call embedding")

    monkeypatch.setattr(retrieval, "retrieve", forbidden)
    monkeypatch.setattr(retrieval, "embed_texts", forbidden)
    assert (
        await retrieval.retrieve_per_document(None, uuid4(), "query", "vector", 5, document_ids=())
        == []
    )


def test_only_scoped_filename_numbers_leave_the_retrieval_text():
    from knowflow.app.source_scope import SourceScope, retrieval_text

    filename = "CedarLedger_总结_20260101.md"
    scope = SourceScope({"status": "filtered", "documents": [{"filename": filename}]}, (uuid4(),))
    original = f"只看 {filename}，查询2025年的预算是否超过9000，别用2024年的数字。"
    result = retrieval_text(original, scope)
    assert filename not in result and "20260101" not in result
    assert all(term in result for term in ("2025", "9000", "别用2024"))
    assert filename in original
    assert retrieval_text(f"为什么文件名叫 {filename}？", scope) == f"为什么文件名叫 {filename}？"


@pytest.mark.asyncio
async def test_retrieval_uses_content_question_but_answer_intent_keeps_full_input(monkeypatch):
    seen = []

    async def retrieve(*args, **kwargs):
        seen.append((args[2], kwargs["document_ids"]))
        return []

    monkeypatch.setattr(api, "retrieve_per_document", retrieve)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            original = f"只看 {docs[1].filename}，查询2025年两项量化成果。"
            request = QueryRequest(question=original, strategy="keyword")
            request._request_kind = "fact_lookup"
            response = await api.run_query(kb, request, SimpleNamespace(id=owner), session)
            assert request.question == original
            assert docs[1].filename not in seen[0][0]
            assert "2025" in seen[0][0] and seen[0][1] == (docs[1].id,)
            assert response.query_plan["retrieval_input"][
                "original_user_input_preserved_for_answer_and_review"
            ]


@pytest.mark.asyncio
async def test_point_comparison_discloses_the_file_missing_from_its_final_citations(monkeypatch):
    from knowflow.app.models import Chunk
    from knowflow.app.retrieval import RetrievalHit
    from sqlalchemy import select

    async def generate(question, evidence, *args, **kwargs):
        return f"One source supplied this part. [{evidence[0]['id']}]"

    monkeypatch.setattr(api, "generate_answer", generate)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            chunks = list(
                (
                    await session.scalars(
                        select(Chunk)
                        .where(Chunk.document_id.in_([docs[1].id, docs[2].id]))
                        .order_by(Chunk.document_id)
                    )
                ).all()
            )

            async def retrieve(*args, **kwargs):
                return [
                    RetrievalHit(
                        chunk=c, filename=next(d.filename for d in docs if d.id == c.document_id)
                    )
                    for c in chunks
                ]

            monkeypatch.setattr(api, "retrieve_per_document", retrieve)
            request = QueryRequest(
                question="对比 TerraShape_Agent 和 OpsHarbor 的证据处理。", strategy="keyword"
            )
            request._coverage = "point"
            response = await api.run_query(kb, request, SimpleNamespace(id=owner), session)
            coverage = response.query_plan["coverage"]
            assert coverage["checked"] == 2 and coverage["cited_documents"] == 1
            uncited = next(d for d in coverage["documents"] if not d["cited_in_answer"])
            assert uncited["filename"] in response.answer
            assert "不能判断文件是否包含" in response.answer
            assert all(d["context_chunks"] == 1 for d in coverage["documents"])


@pytest.mark.asyncio
async def test_english_expansion_cannot_consume_the_next_documents_quota(monkeypatch):
    from knowflow.app import retrieval
    from knowflow.app.config import get_settings
    from knowflow.app.models import Chunk
    from knowflow.app.retrieval import RetrievalHit

    docs, kb = [uuid4(), uuid4()], uuid4()
    groups = {}
    for doc in docs:
        groups[doc] = [
            RetrievalHit(
                Chunk(
                    id=uuid4(),
                    document_id=doc,
                    knowledge_base_id=kb,
                    ordinal=index,
                    content=f"document {doc} passage {index}",
                    search_text="",
                ),
                str(doc),
                retrieval_stage="base" if index < 2 else "text_expansion",
            )
            for index in range(4)
        ]

    async def embed(*args, **kwargs):
        return [[1.0] + [0.0] * (get_settings().embedding_dimensions - 1)]

    async def per_file(*args, **kwargs):
        return groups[kwargs["document_ids"][0]]

    monkeypatch.setattr(retrieval, "embed_texts", embed)
    monkeypatch.setattr(retrieval, "retrieve", per_file)
    hits = await retrieval.retrieve_per_document(
        None, kb, "compare both", "hybrid", 4, document_ids=docs
    )
    assert len(hits) == 4
    assert all(sum(h.chunk.document_id == doc for h in hits) == 2 for doc in docs)
    assert len({h.chunk.id for h in hits}) == 4
