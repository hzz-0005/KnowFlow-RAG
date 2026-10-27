from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from knowflow.app import diagnostics_api
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.main import app
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentBlock,
    DocumentStatus,
    IngestionBatch,
    IngestionJob,
    User,
)
from sqlalchemy import delete, select, update


@asynccontextmanager
async def workspace():
    emails = [f"batch-diagnostic-{uuid4()}@example.com" for _ in range(2)]
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            auth, kbs = [], []
            for email in emails:
                response = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": "batch-diagnostic-integration-password"},
                )
                assert response.status_code == 201
                headers = {"Authorization": f"Bearer {response.json()['access_token']}"}
                auth.append(headers)
                kbs.append(
                    UUID(
                        (
                            await client.post(
                                "/api/v1/knowledge-bases",
                                headers=headers,
                                json={"name": "Batch diagnostics"},
                            )
                        ).json()["id"]
                    )
                )
            async with SessionLocal() as session:
                owners = [
                    await session.scalar(select(User.id).where(User.email == e)) for e in emails
                ]
                docs = []
                for i, status in enumerate(
                    [
                        DocumentStatus.PROCESSED,
                        DocumentStatus.FAILED,
                        DocumentStatus.PENDING,
                        DocumentStatus.PROCESSED,
                    ]
                ):
                    owner = int(i == 3)
                    doc = Document(
                        knowledge_base_id=kbs[owner],
                        uploaded_by=owners[owner],
                        filename="PRIVATE_OTHER_OWNER.md" if owner else f"doc-{i}.md",
                        storage_key=uuid4().hex,
                        media_type="text/plain",
                        file_size=8,
                        sha256=uuid4().hex * 2,
                        status=status,
                        parsing_report={},
                    )
                    session.add(doc)
                    await session.flush()
                    docs.append(doc)
                session.add(
                    DocumentBlock(
                        document_id=docs[0].id,
                        block_key="b",
                        ordinal=0,
                        kind="text",
                        text="evidence",
                        source_spans=[],
                        structure={},
                    )
                )
                session.add(
                    Chunk(
                        document_id=docs[0].id,
                        knowledge_base_id=kbs[0],
                        ordinal=0,
                        content="evidence",
                        search_text="evidence",
                        embedding_model="offline",
                        embedding=[1.0] + [0.0] * (get_settings().embedding_dimensions - 1),
                        source_spans=[{"block_key": "b", "chunk_range": [0, 8]}],
                    )
                )
                batch = IngestionBatch(
                    knowledge_base_id=kbs[0], request_key=uuid4().hex, request_fingerprint="0" * 64
                )
                session.add(batch)
                await session.flush()
                jobs = []
                for ordinal, (doc, status) in enumerate(
                    [
                        (docs[0], "completed"),
                        (docs[0], "skipped"),
                        (docs[1], "failed"),
                        (None, "rejected"),
                        (None, "completed"),
                        (docs[2], "queued"),
                    ]
                ):
                    job = IngestionJob(
                        batch_id=batch.id,
                        document_id=doc.id if doc else None,
                        filename=f"input-{ordinal}.md",
                        ordinal=ordinal,
                        status=status,
                        stage="completed" if status == "completed" else "admission",
                    )
                    session.add(job)
                    jobs.append(job)
                await session.commit()
                ids = {
                    "batch": batch.id,
                    "docs": [d.id for d in docs],
                    "job": jobs[0].id,
                    "updated_at": docs[0].updated_at,
                }
            yield client, auth, ids
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()


async def get(client, auth, ids):
    return await client.get(
        f"/api/v1/ingestion-batches/{ids['batch']}/diagnostics", headers=auth[0]
    )


@pytest.mark.asyncio
async def test_batch_retains_every_input_deduplicates_sources_and_is_read_only():
    async with workspace() as (client, auth, ids):
        path = f"/api/v1/ingestion-batches/{ids['batch']}/diagnostics"
        assert (await client.get(path)).status_code == 401
        assert (await client.get(path, headers=auth[1])).status_code == 404
        response = await get(client, auth, ids)
        assert response.status_code == 200, response.text
        assert response.headers["Cache-Control"] == "private, no-store"
        data = response.json()
        assert len(data["items"]) == 6
        assert data["summary"]["inspection_status_counts"] == {
            "inspected": 3,
            "unavailable": 1,
            "not_applicable": 1,
            "deferred": 1,
            "error": 0,
        }
        assert data["summary"]["n_unique_inspected_documents"] == 2
        assert data["items"][1]["reused"] is True
        assert data["summary"]["finding_documents_by_code"]["token_policy_unknown"] == 1
        assert not data["changed_during_inspection"]
        assert 0 < data["budget"]["read_bytes"] <= data["budget"]["inspection_bytes"]
        async with SessionLocal() as session:
            assert (await session.get(Document, ids["docs"][0])).updated_at == ids["updated_at"]
            assert (await session.get(IngestionJob, ids["job"])).status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", ["bytes", "time"])
async def test_batch_budget_keeps_deferred_rows_without_fabricating_passes(monkeypatch, budget):
    key = "inspection_bytes" if budget == "bytes" else "admission_seconds"
    monkeypatch.setitem(diagnostics_api.BATCH_LIMITS, key, 0)
    async with workspace() as (client, auth, ids):
        data = (await get(client, auth, ids)).json()
        assert len(data["items"]) == 6 and not data["documents"]
        assert data["summary"]["inspection_status_counts"]["deferred"] == 4
        assert data["items"][0]["reason"] == "batch_budget_exhausted"
        assert data["items"][5]["reason"] == "document_in_progress"


@pytest.mark.asyncio
async def test_changed_index_charges_budget_and_does_not_drop_remaining_inputs(monkeypatch):
    calls = []

    async def interrupted(session, doc, *, inspection_bytes):
        calls.append(doc.id)
        raise diagnostics_api.IndexChanged(inspection_bytes)

    monkeypatch.setattr(diagnostics_api, "inspect_document", interrupted)
    monkeypatch.setitem(diagnostics_api.BATCH_LIMITS, "inspection_bytes", 100)
    async with workspace() as (client, auth, ids):
        data = (await get(client, auth, ids)).json()
        assert data["budget"]["read_bytes"] == 100
        assert len(calls) == 1
        assert data["changed_during_inspection"] is True
        assert data["summary"]["inspection_status_counts"]["error"] == 2
        assert data["items"][2]["reason"] == "batch_budget_exhausted"
        assert len(data["items"]) == 6


@pytest.mark.asyncio
async def test_cross_knowledge_base_reference_does_not_reveal_other_document():
    async with workspace() as (client, auth, ids):
        async with SessionLocal() as session:
            await session.execute(
                update(IngestionJob)
                .where(IngestionJob.id == ids["job"])
                .values(document_id=ids["docs"][3])
            )
            await session.commit()
        response = await get(client, auth, ids)
        assert response.status_code == 200
        assert "PRIVATE_OTHER_OWNER" not in response.text
        assert response.json()["items"][0]["inspection_status"] == "unavailable"


@pytest.mark.asyncio
async def test_late_document_and_job_changes_are_reported_without_rewriting_snapshot(monkeypatch):
    original = diagnostics_api.inspect_document
    async with workspace() as (client, auth, ids):

        async def external_change(session, doc, **kwargs):
            result = await original(session, doc, **kwargs)
            if doc.id == ids["docs"][0]:
                async with SessionLocal() as other:
                    await other.execute(
                        update(Document)
                        .where(Document.id == doc.id)
                        .values(updated_at=doc.updated_at + timedelta(seconds=1))
                    )
                    await other.execute(
                        update(IngestionJob)
                        .where(IngestionJob.id == ids["job"])
                        .values(status="failed")
                    )
                    await other.commit()
            return result

        monkeypatch.setattr(diagnostics_api, "inspect_document", external_change)
        data = (await get(client, auth, ids)).json()
        assert data["changed_during_inspection"] is True
        assert data["documents"][str(ids["docs"][0])]["stale"] is True
        assert data["items"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_oversized_legacy_batch_is_rejected_before_scanning():
    async with workspace() as (client, auth, ids):
        async with SessionLocal() as session:
            session.add_all(
                [
                    IngestionJob(
                        batch_id=ids["batch"],
                        filename=f"extra-{i}",
                        ordinal=i,
                        status="rejected",
                        stage="admission",
                    )
                    for i in range(6, 21)
                ]
            )
            await session.commit()
        assert (await get(client, auth, ids)).status_code == 422
