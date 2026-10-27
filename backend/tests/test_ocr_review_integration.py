import hashlib
from datetime import timedelta
from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from knowflow.app import api, ocr_review
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.main import app
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentBlock,
    DocumentStatus,
    KnowledgeBase,
    OcrReview,
    User,
    utc_now,
)
from knowflow.app.structured_parsing import Block, ParsedDocument, ParsingFailure
from pypdf import PdfWriter
from sqlalchemy import delete, select


@pytest.mark.asyncio
async def test_ocr_candidate_isolation_persistence_expiry_and_index_preservation(
    monkeypatch, tmp_path
):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "pdf_parser", "docling")
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=300)
    writer.add_blank_page(width=200, height=300)
    output = BytesIO()
    writer.write(output)
    data = output.getvalue()
    storage_key = f"{uuid4().hex}.pdf"
    (tmp_path / storage_key).write_bytes(data)
    emails = [f"ocr-{uuid4()}@example.com" for _ in range(2)]
    calls = []

    async def candidate(filename, content, config, **kwargs):
        calls.append(kwargs)
        page = kwargs["page_range"][0]
        return ParsedDocument(
            [],
            [
                Block(
                    "#/texts/0", "text", "候选17天，覆盖83%。", source_spans=[{"page_number": page}]
                )
            ],
            {"parser": "docling", "warnings": []},
        )

    monkeypatch.setattr(ocr_review, "parse_for_ingestion", candidate)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            auth = []
            for email in emails:
                registration = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": "integration-password-123"},
                )
                auth.append({"Authorization": f"Bearer {registration.json()['access_token']}"})
            async with SessionLocal() as session:
                user_id = await session.scalar(select(User.id).where(User.email == emails[0]))
                kb = KnowledgeBase(owner_id=user_id, name="OCR comparison test")
                session.add(kb)
                await session.flush()
                doc = Document(
                    knowledge_base_id=kb.id,
                    uploaded_by=user_id,
                    filename="scan.pdf",
                    storage_key=storage_key,
                    media_type="application/pdf",
                    file_size=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                    status=DocumentStatus.PROCESSED,
                    parsing_report={"parser": "docling", "page_count": 2},
                )
                session.add(doc)
                await session.flush()
                block = DocumentBlock(
                    document_id=doc.id,
                    block_key="#/texts/0",
                    ordinal=0,
                    kind="text",
                    text="当前17天，覆盖83%。\ufffd",
                    source_spans=[{"page_number": 2}],
                    structure={},
                )
                chunk = Chunk(
                    document_id=doc.id,
                    knowledge_base_id=kb.id,
                    ordinal=0,
                    content=block.text,
                    search_text=block.text,
                    embedding=[1.0] + [0.0] * (settings.embedding_dimensions - 1),
                    embedding_model="test",
                    page_number=2,
                    source_spans=[{"page_number": 2}],
                )
                session.add_all([block, chunk])
                await session.commit()
                doc_id, block_id, chunk_id = doc.id, block.id, chunk.id
            path = f"/api/v1/documents/{doc_id}/ocr-review?page_number=2"
            empty = (await client.get(path, headers=auth[0])).json()
            assert empty["status"] == "not_started"
            assert empty["can_start"] is True
            assert (await client.get(path)).status_code == 401
            assert (await client.get(path, headers=auth[1])).status_code == 404
            assert (await client.post(path, headers=auth[1])).status_code == 404
            assert (await client.post(path.replace("=2", "=0"), headers=auth[0])).status_code == 422
            assert (await client.post(path.replace("=2", "=3"), headers=auth[0])).status_code == 422
            started = await client.post(path, headers=auth[0])
            assert started.status_code == 202
            assert calls == [{"force_ocr": True, "page_range": (2, 2)}]
            response = await client.get(path, headers=auth[0])
            assert response.headers["cache-control"] == "private, no-store"
            saved = response.json()
            assert saved["status"] == "succeeded"
            assert saved["baseline_changed"] is False
            assert saved["result"]["production_index_modified"] is False
            assert saved["result"]["candidate"]["replacement_characters"] == 0
            assert (await client.get(path, headers=auth[0])).json()["review_id"] == saved[
                "review_id"
            ]
            async with SessionLocal() as session:
                assert (await session.get(Chunk, chunk_id)).content == "当前17天，覆盖83%。\ufffd"
                assert (
                    await session.get(DocumentBlock, block_id)
                ).text == "当前17天，覆盖83%。\ufffd"
                assert (await session.get(Document, doc_id)).status == DocumentStatus.PROCESSED
                changed = await session.get(DocumentBlock, block_id)
                changed.text = "新版本17天。"
                await session.commit()
            assert (await client.get(path, headers=auth[0])).json()["baseline_changed"] is True

            async def broken(*args, **kwargs):
                raise ParsingFailure("解析服务不可用", {"parser": "docling"})

            monkeypatch.setattr(ocr_review, "parse_for_ingestion", broken)
            await client.post(path, headers=auth[0])
            failed = (await client.get(path, headers=auth[0])).json()
            assert failed["status"] == "failed"
            assert failed["error_message"] == "解析服务不可用"

            pending = []

            async def pending_job(review_id):
                pending.append(review_id)

            monkeypatch.setattr(api, "run_ocr_review", pending_job)
            first = (await client.post(path, headers=auth[0])).json()
            repeated = (await client.post(path, headers=auth[0])).json()
            assert first["review_id"] == repeated["review_id"]
            assert len(pending) == 1
            assert (await client.post(path.replace("=2", "=1"), headers=auth[0])).status_code == 409
            async with SessionLocal() as session:
                job = await session.get(OcrReview, pending[0])
                job.expires_at = utc_now() - timedelta(seconds=1)
                await session.commit()
            assert (await client.get(path, headers=auth[0])).json()["status"] == "failed"
            await ocr_review.run_ocr_review(pending[0])
            assert (await client.get(path, headers=auth[0])).json()["status"] == "failed"
            monkeypatch.setattr(settings, "pdf_parser", "native")
            assert (await client.post(path, headers=auth[0])).status_code == 409
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()
