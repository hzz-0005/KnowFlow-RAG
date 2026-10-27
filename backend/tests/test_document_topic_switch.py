"""Document identity is current input, not the last successful citation."""

from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import api
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk, Document
from knowflow.app.schemas import QueryRequest
from test_conversations import create, send, workspace


async def seed_documents(session, kb, owner):
    rows = []
    for filename, content in [
        ("校园二手交易平台_项目总结.md", "Campus market uses a local database."),
        ("TerraShape_Agent项目详解_20261026.md", "TerraShape_Agent reconstructs geometry."),
        ("OpsHarbor_项目详解.md", "OpsHarbor manages incident evidence."),
        ("ClinicalRiver_Graph_项目详解.md", "ClinicalRiver_Graph organizes research."),
    ]:
        doc = Document(
            knowledge_base_id=kb,
            uploaded_by=owner,
            filename=filename,
            storage_key=str(uuid4()),
            media_type="text/markdown",
            file_size=100,
            sha256=uuid4().hex * 2,
            status="processed",
        )
        session.add(doc)
        await session.flush()
        session.add(
            Chunk(
                document_id=doc.id,
                knowledge_base_id=kb,
                ordinal=0,
                content=content,
                search_text=content,
                embedding=[1.0] + [0.0] * (get_settings().embedding_dimensions - 1),
                embedding_model=get_settings().embedding_model,
            )
        )
        rows.append(doc)
    await session.commit()
    return rows


@pytest.mark.asyncio
async def test_explicit_new_named_project_summary_does_not_read_previous_citations(monkeypatch):
    sent = []

    async def generate(question, evidence, *args, **kwargs):
        sent.extend(evidence)
        return f"Project overview [{evidence[0]['id']}]"

    monkeypatch.setattr(api, "generate_answer", generate)
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            query = QueryRequest(question="TerraShape_Agent是干什么的，后续有什么计划？")
            query._request_kind = "document_summary"
            query._context_document_ids = (docs[0].id,)
            query._allow_context_document = False
            result = await api.run_query(kb, query, SimpleNamespace(id=owner), session)
            assert result.document_summary["document"]["id"] == str(docs[1].id)
            assert sent and all("Campus market" not in row["content"] for row in sent)


@pytest.mark.asyncio
async def test_standalone_unknown_summary_never_silently_selects_previous_document(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            query = QueryRequest(question="介绍一下 OceanDesign 项目")
            query._request_kind = "document_summary"
            query._context_document_ids = (docs[0].id,)
            query._allow_context_document = False

            async def never_generate(*args, **kwargs):
                raise AssertionError("An independent unknown object must not read the old file")

            monkeypatch.setattr(api, "generate_answer", never_generate)
            result = await api.run_query(kb, query, SimpleNamespace(id=owner), session)
            assert result.refused and result.document_summary["document"] is None
            # The reply reports a catalog + keyword check; it never asserts that
            # the project itself does not exist.
            assert result.document_summary["named_object"] == "OceanDesign"
            assert "named_probe" in result.document_summary
            assert "OceanDesign" in result.answer and "没有与" in result.answer
            assert "不存在" not in result.answer


@pytest.mark.asyncio
async def test_a_real_followup_summary_still_reads_the_previous_document(monkeypatch):
    sent = []

    async def generate(question, evidence, *args, **kwargs):
        sent.extend(evidence)
        return f"Follow-up overview [{evidence[0]['id']}]"

    monkeypatch.setattr(api, "generate_answer", generate)
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            query = QueryRequest(question="这份文档讲了什么？")
            query._request_kind = "document_summary"
            query._context_document_ids = (docs[0].id,)
            query._allow_context_document = True
            result = await api.run_query(kb, query, SimpleNamespace(id=owner), session)

            assert result.document_summary["document"]["id"] == str(docs[0].id)
            assert sent
            assert all("Campus market" in row["content"] for row in sent)


def test_prior_document_is_only_for_resolved_followups():
    from knowflow.app.document_reference import allows_prior_document

    followup = {"status": "resolved", "reference_turn_ids": ["turn-1"]}
    assert allows_prior_document("这份文档讲了什么", followup) is True
    assert allows_prior_document("这份文档讲了什么", {"status": "standalone"}) is False
    assert allows_prior_document("这份文档讲了什么", followup | {"reference_turn_ids": []}) is False
    assert allows_prior_document("介绍一下 GeoModel 项目", followup) is False
    assert (
        allows_prior_document(
            "这份文档讲了什么",
            {**followup, "binding_decision": {"current_literal_name": "GeoModel"}},
        )
        is False
    )


def test_chinese_partial_title_matches_only_distinctive_fragments():
    from knowflow.app.document_reference import alias_matches, title_aliases

    aliases = title_aliases("校园二手交易平台_项目总结.md")
    assert "校园二手交易平台" in aliases
    hits = alias_matches("介绍一下二手交易平台这个项目", "校园二手交易平台")
    assert [m.group(0) for m in hits] == ["二手交易平台"]
    # A generic document word never selects a file.
    assert alias_matches("介绍一下项目总结", "校园二手交易平台") == []


@pytest.mark.asyncio
async def test_a_chinese_partial_title_selects_its_document(monkeypatch):
    sent = []

    async def generate(question, evidence, *args, **kwargs):
        sent.extend(evidence)
        return f"Scoped answer [{evidence[0]['id']}]"

    monkeypatch.setattr(api, "generate_answer", generate)
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            docs = await seed_documents(session, kb, owner)
            query = QueryRequest(question="二手交易平台这份文档讲了什么？")
            result = await api.run_query(kb, query, SimpleNamespace(id=owner), session)

            assert [c.filename for c in result.citations] == [docs[0].filename]
            assert sent and all("Campus market" in row["content"] for row in sent)


def test_prior_document_followup_is_server_side_and_needs_exactly_one_document():
    from knowflow.app.document_reference import allows_prior_document, prior_document_followup

    followup = prior_document_followup("这份文档讲了什么？", (uuid4(),), "v-test")
    assert followup["status"] == "resolved"
    assert followup["request_kind"] == "document_summary"
    assert allows_prior_document("这份文档讲了什么？", followup) is True

    assert prior_document_followup("这份文档讲了什么？", (uuid4(), uuid4()), "v-test") is None
    assert prior_document_followup("介绍一下 TerraFlow", (uuid4(),), "v-test") is None


@pytest.mark.asyncio
async def test_an_unknown_latin_name_is_searched_instead_of_asking_to_clarify(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]

        result = await send(client, auth[0], cid, "介绍一下 OrionLedger 项目")

        body = result.json()
        assert body["resolution"]["status"] == "standalone"
        assert body["status"] == "completed"
        assert len(calls) == 1


def test_aggregate_list_questions_are_not_singular_referents():
    from knowflow.app.conversation_context import aggregate_list_question

    assert aggregate_list_question("列出这些项目总结里提到的量化指标。") is True
    assert aggregate_list_question("哪些项目没有使用 Temporal？") is True
    assert aggregate_list_question("另一个项目是怎么做证据管理的？") is False
    assert aggregate_list_question("AegisOps 的审批怎么落地？") is False


@pytest.mark.asyncio
async def test_a_list_request_is_searched_instead_of_asking_which_referent(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        cid = (await create(client, auth[0], kb))["id"]

        result = await send(client, auth[0], cid, "列出这些项目总结里提到的量化指标。")

        body = result.json()
        assert body["resolution"]["status"] == "standalone"
        assert len(calls) == 1


@pytest.mark.asyncio
async def test_two_named_documents_are_scoped_together_for_a_comparison(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.document_reference import resolve_document_reference
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            await seed_documents(session, kb, owner)
            scope = await resolve_document_reference(
                session, kb, "校园二手交易平台 和 TerraShape_Agent 在证据管理上有什么区别？"
            )

            assert scope.trace["status"] == "filtered"
            assert scope.trace["selection_basis"] == "multiple_literal_titles"
            assert len(scope.document_ids) == 2


@pytest.mark.asyncio
async def test_excluding_one_named_document_keeps_the_rest_of_the_base(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.document_reference import resolve_document_reference
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            await seed_documents(session, kb, owner)
            scope = await resolve_document_reference(
                session, kb, "不要 校园二手交易平台，其他项目怎么做证据管理？"
            )

            assert scope.trace["status"] == "filtered"
            assert scope.trace["selection_basis"] == "excluded_named_documents"
            assert scope.trace["excluded_filenames"] == ["校园二手交易平台_项目总结.md"]
            assert len(scope.document_ids) == 3


@pytest.mark.asyncio
async def test_sweep_coverage_checks_every_processed_document(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.models import KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            await seed_documents(session, kb, owner)
            body = QueryRequest(question="哪些项目没有使用 Temporal？", strategy="keyword")
            body._coverage = "sweep"

            result = await api.run_query(kb, body, SimpleNamespace(id=owner), session)

            coverage = result.query_plan["coverage"]
            assert coverage["mode"] == "sweep"
            assert coverage["checked"] == 4
            assert len(coverage["documents"]) == 4


@pytest.mark.asyncio
async def test_sweep_skips_per_chunk_scoring_and_asks_per_file(monkeypatch):
    seen = {}
    chunk_id = uuid4()

    async def assessment(*args, **kwargs):
        seen["assessment"] = True
        from knowflow.app.schemas import RetrievalDiagnostics

        return {}, RetrievalDiagnostics(status="complete")

    async def generate(question, evidence, *args, **kwargs):
        seen["question"] = question
        return f"答案 [{evidence[0]['id']}]"

    async def fake_retrieval(*args, **kwargs):
        from knowflow.app.retrieval import RetrievalHit

        chunk = Chunk(
            id=chunk_id,
            document_id=document_id,
            knowledge_base_id=kb_id,
            ordinal=0,
            content="AegisOps uses Temporal for durable workflow.",
            search_text="AegisOps uses Temporal for durable workflow.",
            embedding=[0.0] * get_settings().embedding_dimensions,
            embedding_model="offline_fixture",
        )
        return [RetrievalHit(chunk=chunk, filename="aegis.md")]

    monkeypatch.setattr(api, "assess_retrieval", assessment)
    monkeypatch.setattr(api, "generate_answer", generate)

    async with workspace(monkeypatch) as (_, _, kb, _):
        from knowflow.app.models import Document, KnowledgeBase

        async with SessionLocal() as session:
            kb = UUID(kb)
            kb_id = kb
            owner = (await session.get(KnowledgeBase, kb)).owner_id
            doc = Document(
                knowledge_base_id=kb,
                uploaded_by=owner,
                filename="aegis.md",
                storage_key=str(uuid4()),
                media_type="text/markdown",
                file_size=20,
                sha256="a" * 64,
                status="processed",
            )
            session.add(doc)
            await session.flush()
            document_id = doc.id
            other = Document(
                knowledge_base_id=kb,
                uploaded_by=owner,
                filename="silent.md",
                storage_key=str(uuid4()),
                media_type="text/markdown",
                file_size=20,
                sha256="b" * 64,
                status="processed",
            )
            session.add(other)
            await session.commit()

            monkeypatch.setattr(api, "retrieve_per_document", fake_retrieval)
            body = QueryRequest(question="哪些项目使用 Temporal？", strategy="keyword")
            body._coverage = "sweep"
            result = await api.run_query(kb, body, SimpleNamespace(id=owner), session)

            assert "assessment" not in seen
            assert "按文件分条" in seen["question"]
            assert result.citations
            # Coverage reports what the answer stands behind, not what was fetched.
            assert "本轮未涉及的文件" in result.answer
            assert "silent.md" in result.answer
            coverage = result.query_plan["coverage"]
            assert coverage["scope"] == "all_processed_documents"
            assert coverage["checked"] == 2
            assert coverage["answered"] == 1
            by_name = {row["filename"]: row for row in coverage["documents"]}
            assert by_name["aegis.md"]["cited_in_answer"] is True
            assert by_name["silent.md"]["cited_in_answer"] is False
