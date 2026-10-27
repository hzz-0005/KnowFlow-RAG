from uuid import uuid4

import httpx
import pytest
from knowflow.app import diagnostics_api
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.main import app
from knowflow.app.models import Chunk, Document, DocumentBlock, DocumentStatus, User
from sqlalchemy import delete, select


@pytest.mark.asyncio
async def test_diagnostics_are_owner_scoped_read_only_and_budgeted(monkeypatch):
    emails = [f"diagnostics-{uuid4()}@example.com" for _ in range(2)]
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            headers = []
            for email in emails:
                registered = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": "diagnostics-integration-password"},
                )
                assert registered.status_code == 201
                headers.append({"Authorization": f"Bearer {registered.json()['access_token']}"})
            kb = (
                await client.post(
                    "/api/v1/knowledge-bases", headers=headers[0], json={"name": "Diagnostics test"}
                )
            ).json()["id"]
            async with SessionLocal() as session:
                user = await session.scalar(select(User).where(User.email == emails[0]))
                doc = Document(
                    knowledge_base_id=kb,
                    uploaded_by=user.id,
                    filename="no-file.pdf",
                    storage_key=uuid4().hex,
                    media_type="application/pdf",
                    file_size=10,
                    sha256="0" * 64,
                    status=DocumentStatus.PROCESSED,
                    parsing_report={"page_count": 1},
                )
                session.add(doc)
                await session.flush()
                session.add(
                    DocumentBlock(
                        document_id=doc.id,
                        block_key="b1",
                        ordinal=0,
                        kind="text",
                        text="diagnostics without source file",
                        source_spans=[],
                        structure={},
                    )
                )
                chunk = Chunk(
                    document_id=doc.id,
                    knowledge_base_id=kb,
                    ordinal=0,
                    content="diagnostics without source file",
                    search_text="diagnostics",
                    embedding=[1.0] + [0.0] * (get_settings().embedding_dimensions - 1),
                    embedding_model="offline-fixture",
                    source_spans=[{"block_key": "b1", "page_number": 1, "chunk_range": [0, 31]}],
                )
                session.add(chunk)
                await session.commit()
                doc_id, chunk_id, updated_at = str(doc.id), chunk.id, doc.updated_at
            path = f"/api/v1/documents/{doc_id}/diagnostics"
            assert (await client.get(path)).status_code == 401
            assert (await client.get(path, headers=headers[1])).status_code == 404
            response = await client.get(path, headers=headers[0])
            assert response.status_code == 200, response.text
            assert response.headers["Cache-Control"] == "private, no-store"
            data = response.json()
            assert data["scope"]["total"] == {"blocks": 1, "chunks": 1}
            assert data["observations"]["token_budget"]["unknown"] == 1
            assert "embedding" not in response.text
            repeated = await client.get(path, headers=headers[0])
            assert repeated.json()["snapshot_sha256"] == data["snapshot_sha256"]
            # Exercise the actual second-phase source-body read, rather than only
            # the pure comparator or a span without character mapping.
            async with SessionLocal() as session:
                stored_chunk = await session.get(Chunk, chunk_id)
                stored_chunk.source_spans = [
                    {
                        "block_key": "b1",
                        "page_number": 1,
                        "text_range": [0, 31],
                        "chunk_range": [0, 31],
                    }
                ]
                stored_block = await session.scalar(
                    select(DocumentBlock).where(DocumentBlock.document_id == doc_id)
                )
                stored_block.text = "X" * 31
                await session.commit()
            compared = (await client.get(path, headers=headers[0])).json()
            obs = compared["observations"]["source_consistency"]
            assert obs["spans"]["mismatch"] == 1
            assert obs["examples"][0]["source"] == "X" * 31
            assert compared["scope"]["read_bytes"] > data["scope"]["read_bytes"]
            assert compared["scope"]["source_texts_complete"]
            monkeypatch.setitem(diagnostics_api.LIMITS, "block_text_characters", 5)
            truncated = (await client.get(path, headers=headers[0])).json()
            assert not truncated["scope"]["source_texts_complete"]
            assert truncated["observations"]["source_consistency"]["reasons"] == {
                "source_text_truncated": 1
            }
            assert truncated["observations"]["source_consistency"]["spans"]["mismatch"] == 0
            monkeypatch.setitem(diagnostics_api.LIMITS, "inspection_bytes", 1)
            limited = (await client.get(path, headers=headers[0])).json()
            assert not limited["scope"]["complete"]
            assert any(f["code"] == "inspection_incomplete" for f in limited["findings"])
            async with SessionLocal() as session:
                stored = await session.get(Document, doc_id)
                assert stored.updated_at == updated_at
                assert (
                    await session.get(Chunk, chunk_id)
                ).content == "diagnostics without source file"
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()
