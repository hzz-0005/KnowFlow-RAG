"""Source-selection safety and real pre-limit SQL controls, not answer accuracy."""

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import api, reranker, retrieval
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk, Document, EntityMention, KnowledgeBase, Relation
from knowflow.app.schemas import QueryRequest
from knowflow.app.source_scope import ScopedReadSession, resolve_source_scope
from sqlalchemy import select
from test_conversations import workspace


async def materials(session, kb):
    owner = (await session.get(KnowledgeBase, kb)).owner_id
    docs = []
    for name, status in [
        ("ledger-07.pdf", "processed"),
        ("other.md", "processed"),
        ("中文 说明.pdf", "processed"),
        ("waiting.pdf", "pending"),
    ]:
        d = Document(
            knowledge_base_id=kb,
            uploaded_by=owner,
            filename=name,
            storage_key=str(uuid4()),
            media_type="application/pdf",
            file_size=10,
            sha256=uuid4().hex * 2,
            status=status,
        )
        session.add(d)
        await session.flush()
        docs.append(d)
        cfg = get_settings()
        session.add(
            Chunk(
                knowledge_base_id=kb,
                document_id=d.id,
                ordinal=0,
                content="retained backups policy 17 days",
                search_text="retained backups policy 17 days",
                embedding=[1.0] + [0.0] * (cfg.embedding_dimensions - 1),
                embedding_model=cfg.embedding_model,
                source_spans=[
                    {"row_key": "backups", "table_label": "1", "cell_keys": ["backups", "17 days"]}
                ],
            )
        )
    await session.commit()
    return docs, owner


@pytest.mark.asyncio
async def test_literal_scope_unknown_exclusion_duplicates_unicode_and_current_kb(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            docs, owner = await materials(session, UUID(kb))
            cases = [
                ("在 ledger-07.pdf 中查保留期", "filtered", (docs[0].id,)),
                ("LEDGER-07.PDF 里呢？", "filtered", (docs[0].id,)),
                ("只查“中文 说明.pdf”里的内容", "filtered", (docs[2].id,)),
                ("不要查 other.md，只查 ledger-07.pdf", "filtered", (docs[0].id,)),
                ("对比 ledger-07.pdf 和 other.md", "filtered", None),
                ("只看不存在的 missing-12.pdf", "unavailable", ()),
                ("只看缺失说明.pdf", "unavailable", ()),
                ("缺失说明.pdf里的数据呢？", "unavailable", ()),
                ("不要 ledger-07.pdf", "ambiguous", ()),
                ("ledger-07.pdf，不看 ledger-07.pdf", "ambiguous", ()),
                ("waiting.pdf 查一下", "unavailable", ()),
                ("保留几天？", "unscoped", None),
                ("xledger-07.pdf 查一下", "unavailable", ()),
            ]
            for question, status, ids in cases:
                result = await resolve_source_scope(session, UUID(kb), question)
                assert result.trace["status"] == status, (question, result.trace)
                if ids is not None:
                    assert result.document_ids == ids
            result = await resolve_source_scope(session, uuid4(), "ledger-07.pdf 中呢？")
            assert result.trace["status"] == "unavailable" and not result.trace["documents"]
            duplicate = Document(
                knowledge_base_id=UUID(kb),
                uploaded_by=owner,
                filename=docs[0].filename,
                storage_key=str(uuid4()),
                media_type="application/pdf",
                file_size=10,
                sha256=uuid4().hex * 2,
                status="processed",
            )
            session.add(duplicate)
            await session.commit()
            result = await resolve_source_scope(session, UUID(kb), "ledger-07.pdf 保留几天？")
            assert result.trace["reason"] == "duplicate_filename" and result.document_ids == ()


@pytest.mark.asyncio
async def test_all_strategies_prefilter_no_post_topk_loss_and_no_context_leak(monkeypatch):
    async def embeddings(texts, settings):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    async def rerank(query, hits, limit, settings):
        return hits[:limit]

    monkeypatch.setattr(retrieval, "embed_texts", embeddings)
    monkeypatch.setattr(reranker, "rerank_hits", rerank)
    async with workspace(monkeypatch) as (_, _, kb, _):
        kb = UUID(kb)
        async with SessionLocal() as session:
            docs, _ = await materials(session, kb)
            cfg = get_settings().model_copy(update={"english_keyword_retrieval_enabled": True})
            # Same scores across documents: selected document must survive SQL LIMIT 1.
            for strategy in (
                "keyword",
                "vector",
                "hybrid",
                "hybrid_rerank",
                "graph",
                "graph_hybrid",
            ):
                hits = await retrieval.retrieve(
                    session,
                    kb,
                    "retained backups policy Table 1",
                    strategy,
                    1,
                    cfg,
                    document_ids=(docs[1].id,),
                )
                assert all(h.chunk.document_id == docs[1].id for h in hits), strategy
                if strategy != "graph":
                    assert hits, strategy
            assert not await retrieval.retrieve(
                session, uuid4(), "retained backups", "keyword", 1, cfg, document_ids=(docs[1].id,)
            )
            assert not await retrieval.retrieve(
                session, kb, "retained backups", "hybrid", 1, cfg, document_ids=()
            )
            selected_ids = [d.id for d in docs[:2]]

        async def separate_request(did):
            async with SessionLocal() as reader:
                return await retrieval.retrieve(
                    reader, kb, "retained backups", "keyword", 1, cfg, document_ids=(did,)
                )

        a, b = await asyncio.gather(*(separate_request(did) for did in selected_ids))
        assert a[0].chunk.document_id == selected_ids[0]
        assert b[0].chunk.document_id == selected_ids[1]


def test_graph_seed_mention_and_edge_queries_receive_metadata_constraints():
    kb, did = uuid4(), uuid4()
    reader = ScopedReadSession(None, kb, (did,))
    statements = [
        select(EntityMention.entity_id).join(Document, Document.id == EntityMention.document_id),
        select(Chunk)
        .join(EntityMention, EntityMention.chunk_id == Chunk.id)
        .join(Document, Document.id == EntityMention.document_id),
        select(Relation, Chunk)
        .join(Chunk, Chunk.id == Relation.source_chunk_id)
        .join(Document, Document.id == Relation.document_id),
    ]
    for statement in statements:
        compiled = reader.statement(statement.limit(1)).compile(
            compile_kwargs={"literal_binds": True}
        )
        sql = str(compiled)
        assert str(did).replace("-", "") in sql.replace("-", "")
        assert "documents.knowledge_base_id" in sql and "documents.status" in sql
        assert sql.index("documents.id IN") < sql.index("LIMIT")


@pytest.mark.asyncio
async def test_unavailable_scope_blocks_before_any_provider_or_retriever(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail("Unknown file must not fall back to unscoped search")

    monkeypatch.setattr(api, "plan_query", forbidden)
    monkeypatch.setattr(api, "retrieve", forbidden)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            _, owner = await materials(session, UUID(kb))
            response = await api.run_query(
                UUID(kb),
                QueryRequest(question="only missing-12.pdf", strategy="hybrid"),
                SimpleNamespace(id=owner),
                session,
            )
            assert response.refused and not response.citations
            assert response.query_plan["status"] == "source_scope_blocked"
            assert "未扩大" in response.answer


@pytest.mark.asyncio
async def test_empty_scoped_pool_recovers_k_with_review_but_never_other_document(monkeypatch):
    async def embeddings(texts, settings):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    monkeypatch.setattr(retrieval, "embed_texts", embeddings)
    async with workspace(monkeypatch) as (_, _, kb, _):
        kb = UUID(kb)
        async with SessionLocal() as session:
            docs, _ = await materials(session, kb)
            cfg = get_settings().model_copy(
                update={
                    "rag_max_cosine_distance": -1,
                    "english_keyword_retrieval_enabled": False,
                    "table_row_retrieval_enabled": False,
                }
            )
            for ordinal in (1, 2):
                session.add(Chunk(
                    knowledge_base_id=kb, document_id=docs[0].id, ordinal=ordinal,
                    content="A separate evidence unit", search_text="",
                    embedding=[1.0] + [0.0] * (cfg.embedding_dimensions - 1),
                    embedding_model=cfg.embedding_model,
                ))
            await session.commit()
            scoped = await retrieval.retrieve(
                session, kb, "zzqk unrelated", "hybrid", 2, cfg, document_ids=(docs[0].id,)
            )
            global_hits = await retrieval.retrieve(session, kb, "zzqk unrelated", "hybrid", 2, cfg)
            assert len(scoped) == 2 and len(global_hits) == 1
            assert all(
                h.chunk.document_id == docs[0].id and h.distance_gate_fallback for h in scoped
            )
            no_review = cfg.model_copy(update={"retrieval_assessment_enabled": False})
            assert not await retrieval.retrieve(
                session, kb, "zzqk unrelated", "hybrid", 2, no_review,
                document_ids=(docs[0].id,),
            )


@pytest.mark.asyncio
async def test_online_relevance_receives_real_filename_separate_from_content(monkeypatch):
    from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics

    calls = []

    async def assess(question, passages, settings):
        calls.append(passages)
        assert passages and all(p["filename"] == "ledger-07.pdf" for p in passages)
        assert all("ledger-07.pdf" not in p["content"] for p in passages)
        return (
            {p["id"]: RelevanceAssessment(reason="未审核控制") for p in passages},
            RetrievalDiagnostics(status="unavailable"),
        )

    monkeypatch.setattr(api, "assess_retrieval", assess)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            _, owner = await materials(session, UUID(kb))
            result = await api.run_query(
                UUID(kb),
                QueryRequest(question="ledger-07.pdf retained backups", strategy="keyword"),
                SimpleNamespace(id=owner), session,
            )
            assert result.refused and len(calls) == 1
