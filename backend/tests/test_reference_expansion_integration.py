from uuid import uuid4

import httpx
import pytest
from knowflow.app import api
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.document_links import rebuild_document_links
from knowflow.app.main import app
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentBlock,
    DocumentLink,
    DocumentStatus,
    KnowledgeBase,
    User,
)
from knowflow.app.reference_expansion import expand_references
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics
from sqlalchemy import delete, func, select


@pytest.mark.asyncio
async def test_two_hop_expansion_ambiguity_scope_trace_and_rebuild_preserves_chunks(monkeypatch):
    emails = [f"reference-{uuid4()}@example.com" for _ in range(2)]
    settings = get_settings()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            auth = []
            for email in emails:
                response = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": "integration-password-123"},
                )
                auth.append({"Authorization": f"Bearer {response.json()['access_token']}"})
            async with SessionLocal() as session:
                user_id = await session.scalar(select(User.id).where(User.email == emails[0]))
                kb = KnowledgeBase(owner_id=user_id, name="References")
                session.add(kb)
                await session.flush()
                doc = Document(
                    knowledge_base_id=kb.id,
                    uploaded_by=user_id,
                    filename="chain.pdf",
                    storage_key=f"{uuid4()}.pdf",
                    file_size=1,
                    media_type="application/pdf",
                    sha256=uuid4().hex,
                    status=DocumentStatus.PROCESSED,
                    parsing_report={"parser": "docling", "page_count": 8},
                )
                session.add(doc)
                await session.flush()
                blocks, chunks = [], []
                for ordinal, (page, title, text) in enumerate(
                    [
                        (2, "2 部署范围", "G1 对应 Atlas 网关；仅适用于测试环境。"),
                        (4, "4 执行条件", "复核周期为每隔17天；执行范围遵循第2节。"),
                        (8, "8 归档路由", "G1 写入 Cedar 索引，复核周期与执行范围见第4节。"),
                    ]
                ):
                    keys = [f"heading-{page}", f"body-{page}"]
                    spans = [{"block_key": key, "page_number": page} for key in keys]
                    blocks.extend(
                        [
                            DocumentBlock(
                                document_id=doc.id,
                                block_key=keys[0],
                                ordinal=ordinal * 2,
                                kind="section_header",
                                text=title,
                                section=title,
                                source_spans=[spans[0]],
                                structure={},
                            ),
                            DocumentBlock(
                                document_id=doc.id,
                                block_key=keys[1],
                                ordinal=ordinal * 2 + 1,
                                kind="text",
                                text=text,
                                section=title,
                                source_spans=[spans[1]],
                                structure={},
                            ),
                        ]
                    )
                    chunks.append(
                        Chunk(
                            document_id=doc.id,
                            knowledge_base_id=kb.id,
                            ordinal=ordinal,
                            content=f"{title}\n{text}",
                            search_text=text,
                            embedding=[1.0] + [0.0] * (settings.embedding_dimensions - 1),
                            embedding_model="test",
                            page_number=page,
                            section=title,
                            source_spans=spans,
                        )
                    )
                session.add_all(blocks + chunks)
                await session.flush()
                await rebuild_document_links(session, doc)
                await session.commit()
                doc_id, kb_id = doc.id, kb.id
                chunk_ids = [chunk.id for chunk in chunks]
                seed = RetrievalHit(chunks[2], doc.filename, keyword_rank=1)
                expanded, trace = await expand_references(
                    session, kb.id, "G1 的索引、周期与执行范围是什么？", [seed]
                )
                assert [h.chunk.page_number for h in expanded] == [8, 4, 2]
                assert trace["base_count"] == 1 and trace["added_count"] == 2
                assert expanded[1].reference_paths[0]["hop"] == 1
                assert expanded[2].reference_paths[0]["hop"] == 2
                assert (await expand_references(session, uuid4(), "G1", [seed]))[1][
                    "added_count"
                ] == 0
                assert (await expand_references(session, kb.id, "G1", [seed], False))[0] == [seed]
                # A split before the actual reference must not expand the whole block.
                original = chunks[2].content
                chunks[2].content = "只包含 G1，没有编号引用的切片。"
                assert (await expand_references(session, kb.id, "G1", [seed]))[1][
                    "added_count"
                ] == 0
                chunks[2].content = original

            endpoint = f"/api/v1/documents/{doc_id}/references"
            assert (await client.get(endpoint)).status_code == 401
            assert (await client.get(endpoint, headers=auth[1])).status_code == 404
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/reference-index", headers=auth[1])
            ).status_code == 404
            inspected = (await client.get(endpoint, headers=auth[0])).json()
            assert inspected["total"] == 2
            assert inspected["items"][0]["source"]["source_spans"][0]["page_number"] == 4
            assert (await client.get(endpoint + "?limit=101", headers=auth[0])).status_code == 422
            assert (
                await client.get(endpoint + "?status_filter=ambiguous", headers=auth[0])
            ).json()["total"] == 0
            await client.post(f"/api/v1/documents/{doc_id}/reference-index", headers=auth[0])
            await client.post(f"/api/v1/documents/{doc_id}/reference-index", headers=auth[0])
            async with SessionLocal() as session:
                assert (
                    await session.scalar(
                        select(func.count())
                        .select_from(DocumentLink)
                        .where(DocumentLink.document_id == doc_id)
                    )
                    == 2
                )
                assert set(
                    await session.scalars(select(Chunk.id).where(Chunk.document_id == doc_id))
                ) == set(chunk_ids)

            async def retrieve_seed(*args):
                # Fresh hit avoids carrying trace annotations between requests.
                return [RetrievalHit(chunks[2], "chain.pdf", keyword_rank=1)]

            async def assess(question, evidence, *args):
                return {
                    row["id"]: RelevanceAssessment(
                        verdict="partial",
                        reason="提供一个必要子事实",
                        supporting_quotes=[row["content"]],
                    )
                    for row in evidence
                }, RetrievalDiagnostics(status="complete")

            async def answer(question, evidence, *args, **kwargs):
                return "有三处证据。" + " ".join(f"[{row['id']}]" for row in evidence)

            monkeypatch.setattr(api, "retrieve", retrieve_seed)
            monkeypatch.setattr(api, "assess_retrieval", assess)
            monkeypatch.setattr(api, "generate_answer", answer)
            asked = await client.post(
                f"/api/v1/knowledge-bases/{kb_id}/query",
                headers=auth[0],
                json={"question": "G1 的索引、周期与范围？", "strategy": "keyword", "top_k": 1},
            )
            result = asked.json()
            assert asked.status_code == 200
            assert result["refused"] is False
            assert result["reference_expansion"]["added_count"] == 2
            assert {c["page_number"] for c in result["citations"]} == {2, 4, 8}
            assert all(c["included_in_context"] for c in result["retrieval_evidence"])
            disabled = (
                await client.post(
                    f"/api/v1/knowledge-bases/{kb_id}/query",
                    headers=auth[0],
                    json={"question": "G1?", "strategy": "keyword", "expand_references": False},
                )
            ).json()
            assert len(disabled["retrieval_evidence"]) == 1

            async with SessionLocal() as session:
                doc = await session.get(Document, doc_id)
                session.add(
                    DocumentBlock(
                        document_id=doc_id,
                        block_key="duplicate-scope",
                        ordinal=99,
                        kind="section_header",
                        text="2 Other Scope",
                        section="2 Other Scope",
                        source_spans=[{"page_number": 5}],
                        structure={},
                    )
                )
                await session.flush()
                await rebuild_document_links(session, doc)
                await session.commit()
                seed = RetrievalHit(chunks[2], "chain.pdf")
                expanded, trace = await expand_references(session, kb_id, "G1", [seed])
                assert [h.chunk.page_number for h in expanded] == [8, 4]
                assert trace["unresolved"][0]["status"] == "ambiguous"
                # Failed sources never produce new evidence, even if old links remain.
                doc.status = DocumentStatus.FAILED
                await session.flush()
                assert (await expand_references(session, kb_id, "G1", [seed]))[1][
                    "added_count"
                ] == 0
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()
