from io import BytesIO
from uuid import uuid4

import httpx
import pytest
from knowflow.app import api, ingestion
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.main import app
from knowflow.app.models import Chunk, User
from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics
from openpyxl import Workbook
from sqlalchemy import delete, select


@pytest.mark.parametrize("extension", ["csv", "xlsx"])
@pytest.mark.asyncio
async def test_tabular_upload_source_ownership_cache_citations_and_failed_rebuild(
    monkeypatch, tmp_path, extension
):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))

    async def embeddings(texts, *_):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    async def answer(question, evidence, *_, **kwargs):
        return f"测试夹具回答 [{evidence[0]['id']}]。"

    async def assessment(question, evidence, *_):
        return {
            row["id"]: RelevanceAssessment(
                verdict="direct", reason="fixture", supporting_quotes=[row["content"]]
            )
            for row in evidence
        }, RetrievalDiagnostics(status="complete")

    monkeypatch.setattr(ingestion, "embed_texts", embeddings)
    monkeypatch.setattr(api, "generate_answer", answer)
    monkeypatch.setattr(api, "assess_retrieval", assessment)
    content = "编号,库存\nZX001,0\nZX002,9".encode()
    if extension == "xlsx":
        book = Workbook()
        book.active.title = "库存"
        for row in [["编号", "库存"], ["ZX001", 0], ["ZX002", 9]]:
            book.active.append(row)
        raw = BytesIO()
        book.save(raw)
        content = raw.getvalue()
    emails = [f"tabular-{uuid4()}@example.com" for _ in range(2)]
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
                assert response.status_code == 201
                auth.append({"Authorization": f"Bearer {response.json()['access_token']}"})
            kb = (
                await client.post(
                    "/api/v1/knowledge-bases", headers=auth[0], json={"name": "Tables"}
                )
            ).json()["id"]
            upload = await client.post(
                f"/api/v1/knowledge-bases/{kb}/documents",
                headers=auth[0],
                files={"file": (f"data.{extension}", content)},
            )
            assert upload.status_code == 202
            doc_id = upload.json()["id"]
            base = f"/api/v1/documents/{doc_id}"
            detail = (await client.get(base, headers=auth[0])).json()
            assert detail["status"] == "processed", detail
            assert detail["parsing_report"]["parser"] == "tabular"
            assert detail["parsing_report"]["parse_artifact"]["available"] is True
            blocks = (await client.get(base + "/segments?view=blocks", headers=auth[0])).json()
            assert blocks["items"][0]["tabular_source"]["cell_count"] == 6
            path = base + "/table-cells?block_key=sheet-0-table-0"
            first = (await client.get(path + "&limit=3", headers=auth[0])).json()
            assert first["total"] == 6
            assert first["cells"][2]["raw_value"] == "ZX001"
            last = (await client.get(path + "&offset=3", headers=auth[0])).json()
            assert last["cells"][0]["coordinate"] == "B2"
            assert last["cells"][0]["raw_value"] in (0, "0")
            assert (await client.get(path, headers=auth[1])).status_code == 404
            assert (await client.get(path)).status_code == 401
            assert (await client.get(path + "&limit=101", headers=auth[0])).status_code == 422
            assert (
                await client.get(base + "/table-cells?block_key=missing", headers=auth[0])
            ).status_code == 404
            assert (await client.get(base + "/pages/1", headers=auth[0])).status_code == 415
            response = await client.post(
                f"/api/v1/knowledge-bases/{kb}/query",
                headers=auth[0],
                json={"question": "ZX001", "strategy": "keyword"},
            )
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["refused"] is False, result
            assert result["citations"][0]["page_number"] is None
            assert result["citations"][0]["source_spans"][0]["sheet_name"] in {"库存", "CSV"}
            assert result["citations"][0]["source_spans"][0]["cell_range"]

            # Cached parsing can re-chunk without touching the parser or a paid provider.
            async def unavailable(*_):
                raise RuntimeError("fixture provider unavailable")

            monkeypatch.setattr(ingestion, "parse_for_ingestion", unavailable)
            assert (await client.post(base + "/rechunk", headers=auth[0])).status_code == 202
            assert (await client.get(base, headers=auth[0])).json()["status"] == "processed"
            async with SessionLocal() as session:
                ids_before = set(
                    await session.scalars(select(Chunk.id).where(Chunk.document_id == doc_id))
                )
            monkeypatch.setattr(ingestion, "embed_texts", unavailable)
            assert (await client.post(base + "/rechunk", headers=auth[0])).status_code == 202
            assert (await client.get(base, headers=auth[0])).json()["status"] == "failed"
            async with SessionLocal() as session:
                assert (
                    set(await session.scalars(select(Chunk.id).where(Chunk.document_id == doc_id)))
                    == ids_before
                )
            assert (await client.delete(base, headers=auth[0])).status_code == 204
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()
