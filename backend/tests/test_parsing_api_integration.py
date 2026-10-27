from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from knowflow.app import ingestion
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.document_processing import ParsedSection
from knowflow.app.main import app
from knowflow.app.models import Chunk, DocumentBlock, User
from knowflow.app.parse_artifacts import artifact_path
from knowflow.app.structured_parsing import Block, ParsedDocument, ParsingFailure
from pypdf import PdfWriter
from sqlalchemy import delete, select


@pytest.mark.asyncio
async def test_parse_report_page_preview_ownership_and_failed_rebuild_preserves_sources(
    monkeypatch, tmp_path
):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    span = {
        "block_key": "#/texts/0",
        "page_number": 1,
        "bbox": {"left": 0.1, "top": 0.1, "right": 0.8, "bottom": 0.3},
    }
    report = {
        "parser": "docling",
        "status": "needs_review",
        "page_count": 1,
        "warnings": ["需人工核对"],
        "ocr_requested": True,
    }

    async def parsed(*args):
        return ParsedDocument(
            [ParsedSection("来源文字用于检索和引文验证。", 1, "标题", [span])],
            [Block("#/texts/0", "text", "来源文字用于检索和引文验证。", source_spans=[span])],
            report,
        )

    async def embeddings(texts, *_):
        vector = [1.0] + [0.0] * (settings.embedding_dimensions - 1)
        return [vector for _ in texts]

    monkeypatch.setattr(ingestion, "parse_for_ingestion", parsed)
    monkeypatch.setattr(ingestion, "embed_texts", embeddings)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=300)
    data = BytesIO()
    writer.write(data)
    emails = [f"parse-{uuid4()}@example.com" for _ in range(2)]
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
            kb = (
                await client.post(
                    "/api/v1/knowledge-bases", headers=auth[0], json={"name": "Parse test"}
                )
            ).json()["id"]
            uploaded = await client.post(
                f"/api/v1/knowledge-bases/{kb}/documents",
                headers=auth[0],
                files={"file": ("scan.pdf", data.getvalue(), "application/pdf")},
            )
            doc_id = uploaded.json()["id"]
            detail = (await client.get(f"/api/v1/documents/{doc_id}", headers=auth[0])).json()
            assert detail["status"] == "processed"
            assert {key: detail["parsing_report"][key] for key in report} == report
            assert detail["parsing_report"]["reference_index"]["total"] == 0
            blocks_path = f"/api/v1/documents/{doc_id}/blocks"
            blocks = await client.get(blocks_path, headers=auth[0])
            assert blocks.json()[0]["source_spans"] == [span]
            assert (await client.get(blocks_path, headers=auth[1])).status_code == 404
            assert (
                await client.get(blocks_path + "?limit=201", headers=auth[0])
            ).status_code == 422
            segments_path = f"/api/v1/documents/{doc_id}/segments"
            inspected = (await client.get(segments_path, headers=auth[0])).json()
            assert inspected["total"] == 1
            assert inspected["statistics"]["count"] == 1
            located = inspected["items"][0]["source_spans"][0]
            assert {key: located[key] for key in span} == span
            assert located["chunk_range"] == [0, len(inspected["items"][0]["content"])]
            assert inspected["items"][0]["processing_metadata"]["input_tokens"] > 0
            assert (await client.get(segments_path, headers=auth[1])).status_code == 404
            assert (
                await client.get(segments_path + "?offset=-1", headers=auth[0])
            ).status_code == 422
            assert (await client.get(segments_path + "?q=不存在", headers=auth[0])).json()[
                "total"
            ] == 0
            assert (await client.get(segments_path + "?page_number=2", headers=auth[0])).json()[
                "total"
            ] == 0
            blocks_view = (
                await client.get(segments_path + "?view=blocks&q=来源&limit=1", headers=auth[0])
            ).json()
            assert blocks_view["total"] == 1
            assert blocks_view["items"][0]["kind"] == "text"
            preview_path = f"/api/v1/documents/{doc_id}/pages/1"
            preview = await client.get(preview_path, headers=auth[0])
            assert preview.status_code == 200
            assert preview.content.startswith(b"\x89PNG")
            assert preview.headers["cache-control"] == "private, no-store"
            assert (await client.get(preview_path, headers=auth[1])).status_code == 404
            assert (await client.get(preview_path)).status_code == 401
            assert (await client.get(preview_path[:-1] + "2", headers=auth[0])).status_code == 404

            async def broken(*args):
                raise ParsingFailure("OCR 服务超时", {"parser": "docling"})

            monkeypatch.setattr(ingestion, "parse_for_ingestion", broken)
            # Re-chunking must succeed even when the OCR/parser is unavailable.
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/rechunk", headers=auth[1])
            ).status_code == 404
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/rechunk", headers=auth[0])
            ).status_code == 202
            rebuilt = (await client.get(f"/api/v1/documents/{doc_id}", headers=auth[0])).json()
            assert rebuilt["status"] == "processed"
            assert rebuilt["parsing_report"]["parse_artifact"]["reused"] is True

            async def provider_down(*args):
                raise RuntimeError("temporary embedding outage")

            monkeypatch.setattr(ingestion, "embed_texts", provider_down)
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/rechunk", headers=auth[0])
            ).status_code == 202
            failed_index = (await client.get(f"/api/v1/documents/{doc_id}", headers=auth[0])).json()
            assert failed_index["status"] == "failed"
            assert failed_index["parsing_report"]["parse_artifact"]["available"] is True
            # Recovery retries embedding from the cache while OCR is still unavailable.
            monkeypatch.setattr(ingestion, "embed_texts", embeddings)
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/rechunk", headers=auth[0])
            ).status_code == 202
            recovered = (await client.get(f"/api/v1/documents/{doc_id}", headers=auth[0])).json()
            assert recovered["status"] == "processed"
            assert recovered["parsing_report"]["parse_artifact"]["reused"] is True
            async with SessionLocal() as session:
                from knowflow.app.models import Document

                stored = await session.get(Document, doc_id)
                cache = artifact_path(stored.storage_key, settings)
            raw_cache = cache.read_bytes()
            cache.write_bytes(b"corrupt")
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/rechunk", headers=auth[0])
            ).status_code == 409
            cache.write_bytes(raw_cache)
            assert (
                await client.post(f"/api/v1/documents/{doc_id}/reprocess", headers=auth[0])
            ).status_code == 202
            failed = (await client.get(f"/api/v1/documents/{doc_id}", headers=auth[0])).json()
            assert failed["status"] == "failed"
            assert failed["parsing_report"]["status"] == "failed"
            # Previous derived sources survive failure, and are excluded from active retrieval.
            async with SessionLocal() as session:
                assert await session.scalar(select(Chunk.id).where(Chunk.document_id == doc_id))
                assert await session.scalar(
                    select(DocumentBlock.id).where(DocumentBlock.document_id == doc_id)
                )
            assert (
                await client.delete(f"/api/v1/documents/{doc_id}", headers=auth[0])
            ).status_code == 204
            assert not cache.exists()
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()
