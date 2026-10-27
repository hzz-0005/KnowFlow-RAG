import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from uuid import UUID, uuid4

import httpx
import pytest
from knowflow.app import batch_api, ingestion
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.ingestion_jobs import LeaseLost, claim_one, renew, run_claim
from knowflow.app.main import app
from knowflow.app.models import Chunk, Document, IngestionJob, User, utc_now
from knowflow.app.structured_parsing import ParsingFailure
from sqlalchemy import delete, select, update


@asynccontextmanager
async def workspace(monkeypatch, tmp_path):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(settings, "pdf_parser", "native")

    async def embed(texts, *_):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    monkeypatch.setattr(ingestion, "embed_texts", embed)
    emails = [f"batch-{uuid4()}@example.com" for _ in range(2)]
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            headers = []
            for email in emails:
                response = await client.post(
                    "/api/v1/auth/register",
                    json={"email": email, "password": "batch-integration-password"},
                )
                assert response.status_code == 201
                headers.append({"Authorization": f"Bearer {response.json()['access_token']}"})
            kb = (
                await client.post(
                    "/api/v1/knowledge-bases", headers=headers[0], json={"name": "Batch test"}
                )
            ).json()["id"]
            yield client, headers, kb, embed
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email.in_(emails)))
            await session.commit()


async def upload(client, headers, kb, files=None, key=None):
    response = await client.post(
        f"/api/v1/knowledge-bases/{kb}/ingestion-batches",
        headers=headers | ({"Idempotency-Key": key} if key else {}),
        files=[("files", item) for item in (files or [("good.txt", b"persistent batch content")])],
    )
    assert response.status_code == 202, response.text
    return response.json()


async def inspect(client, headers, batch):
    return (await client.get(f"/api/v1/ingestion-batches/{batch['id']}", headers=headers)).json()


async def due(batch):
    async with SessionLocal() as session:
        await session.execute(
            update(IngestionJob)
            .where(IngestionJob.batch_id == UUID(batch["id"]), IngestionJob.status == "retry_wait")
            .values(available_at=utc_now() - timedelta(seconds=1))
        )
        await session.commit()


async def one(batch):
    control = await claim_one(UUID(batch["id"]))
    assert control is not None
    await run_claim(control)
    return control


@pytest.mark.asyncio
async def test_mixed_batch_keeps_every_input_isolates_failures_and_owners(monkeypatch, tmp_path):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):
        batch = await upload(
            client,
            auth[0],
            kb,
            [
                ("good.txt", b"queue document"),
                ("good.csv", b"id,value\n0001,0"),
                ("bad.pdf", b"broken PDF"),
                ("duplicate.txt", b"queue document"),
                ("bad.exe", b"unsupported"),
                ("empty.md", b""),
            ],
        )
        assert batch["total"] == 6
        assert batch["counts"]["queued"] == 3
        assert batch["counts"]["skipped"] == 1
        assert batch["counts"]["rejected"] == 2
        for _ in range(3):
            await one(batch)
        result = await inspect(client, auth[0], batch)
        assert result["status"] == "completed_with_issues"
        assert result["counts"]["completed"] == 2
        assert result["counts"]["failed"] == 1
        failure = next(j for j in result["items"] if j["filename"] == "bad.pdf")
        assert failure["failure_pages_known"] is False
        assert failure["failure_pages"] == []
        assert failure["stage"] == "parsing"
        assert len(result["attempts"]) == 3
        path = f"/api/v1/ingestion-batches/{batch['id']}"
        assert (await client.get(path, headers=auth[1])).status_code == 404
        assert (await client.get(path)).status_code == 401
        assert (await client.post(path + "/retry", headers=auth[1])).status_code == 404
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb}/ingestion-batches", headers=auth[1])
        ).status_code == 404
        assert (
            await client.get(
                f"/api/v1/knowledge-bases/{kb}/ingestion-batches?limit=31", headers=auth[0]
            )
        ).status_code == 422


@pytest.mark.asyncio
async def test_idempotency_concurrency_payload_conflicts_limits_and_backpressure(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):
        batches = await asyncio.gather(
            *[upload(client, auth[0], kb, key="same-request") for _ in range(2)]
        )
        assert batches[0]["id"] == batches[1]["id"]
        assert len((await inspect(client, auth[0], batches[0]))["items"]) == 1
        path = f"/api/v1/knowledge-bases/{kb}/ingestion-batches"
        conflict = await client.post(
            path,
            headers=auth[0] | {"Idempotency-Key": "same-request"},
            files=[("files", ("different.txt", b"different"))],
        )
        assert conflict.status_code == 409
        too_many = await client.post(
            path, headers=auth[0], files=[("files", (f"{i}.txt", b"x")) for i in range(21)]
        )
        assert too_many.status_code == 422
        duplicate = await upload(client, auth[0], kb)
        assert duplicate["counts"]["skipped"] == 1
        monkeypatch.setattr(batch_api, "MAX_QUEUED", 0)
        assert (
            await client.post(path, headers=auth[0], files=[("files", ("full.txt", b"full"))])
        ).status_code == 429


@pytest.mark.asyncio
async def test_embedding_retry_reuses_parse_and_retains_lineage_without_paid_calls(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, embed):
        batch = await upload(client, auth[0], kb)

        async def unavailable(*_):
            raise TimeoutError("test provider timeout")

        monkeypatch.setattr(ingestion, "embed_texts", unavailable)
        await one(batch)
        waiting = await inspect(client, auth[0], batch)
        assert waiting["items"][0]["status"] == "retry_wait"
        assert waiting["attempts"][0]["stage"] == "embedding"
        doc_id = waiting["items"][0]["document_id"]
        assert (
            await client.post(f"/api/v1/documents/{doc_id}/reprocess", headers=auth[0])
        ).status_code == 409

        async def parser_must_not_run(*_):
            pytest.fail("retry should reuse verified parse cache")

        monkeypatch.setattr(ingestion, "parse_for_ingestion", parser_must_not_run)
        monkeypatch.setattr(ingestion, "embed_texts", embed)
        await due(batch)
        await one(batch)
        result = await inspect(client, auth[0], batch)
        assert result["counts"]["completed"] == 1
        assert result["items"][0]["cache_reused"] is True
        assert result["attempts"][1]["retry_of_attempt_id"] == result["attempts"][0]["id"]
        assert result["attempts"][1]["execution_config"]["source_sha256"]
        assert "key" not in json_text(result["attempts"][1]["execution_config"])


def json_text(value):
    import json

    return json.dumps(value)


@pytest.mark.asyncio
async def test_admission_budget_rejects_remaining_files_without_losing_earlier_valid_file(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):
        monkeypatch.setattr(batch_api, "MAX_BATCH_BYTES", 14)
        batch = await upload(
            client,
            auth[0],
            kb,
            [("长" * 300 + ".txt", b"okay"), ("large.txt", b"x" * 30), ("last.txt", b"last")],
        )
        assert [j["status"] for j in batch["items"]] == ["queued", "rejected", "rejected"]
        assert batch["items"][0]["filename"].endswith(".txt")
        assert len(batch["items"][0]["filename"]) == 255
        await one(batch)
        assert (await inspect(client, auth[0], batch))["counts"]["completed"] == 1


@pytest.mark.asyncio
async def test_attempt_deadline_and_deleted_document_leave_durable_terminal_or_retry_state(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):

        async def blocked(*_):
            await asyncio.Event().wait()

        monkeypatch.setattr(ingestion, "embed_texts", blocked)
        monkeypatch.setattr(get_settings(), "ingestion_attempt_timeout_seconds", 0.5)
        batch = await upload(client, auth[0], kb)
        await one(batch)
        item = (await inspect(client, auth[0], batch))["items"][0]
        assert item["status"] == "retry_wait"
        assert item["error_code"].endswith("_timeout")
        assert (
            await client.delete(f"/api/v1/documents/{item['document_id']}", headers=auth[0])
        ).status_code == 204
        await due(batch)
        assert await claim_one(UUID(batch["id"])) is None
        gone = (await inspect(client, auth[0], batch))["items"][0]
        assert gone["status"] == "failed" and gone["error_code"] == "document_deleted"


@pytest.mark.asyncio
async def test_crash_recovery_budget_is_bounded_and_manual_retry_obeys_backpressure(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):
        batch = await upload(client, auth[0], kb)
        for _ in range(3):
            control = await claim_one(UUID(batch["id"]))
            assert control
            async with SessionLocal() as session:
                await session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == control.job_id)
                    .values(lease_expires_at=utc_now() - timedelta(seconds=1))
                )
                await session.commit()
        assert await claim_one(UUID(batch["id"])) is None
        result = await inspect(client, auth[0], batch)
        assert result["items"][0]["status"] == "failed"
        assert len(result["attempts"]) == 3
        assert all(a["status"] == "interrupted" for a in result["attempts"])
        monkeypatch.setattr(batch_api, "MAX_QUEUED", 0)
        assert (
            await client.post(f"/api/v1/ingestion-batches/{batch['id']}/retry", headers=auth[0])
        ).status_code == 429
        assert (await inspect(client, auth[0], batch))["items"][0]["status"] == "failed"


@pytest.mark.asyncio
async def test_retry_budgets_are_finite_even_after_explicit_manual_retries(monkeypatch, tmp_path):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):

        async def unavailable(*_):
            raise TimeoutError()

        monkeypatch.setattr(ingestion, "embed_texts", unavailable)
        batch = await upload(client, auth[0], kb)
        for round_number in range(3):
            for _ in range(3):
                await due(batch)
                await one(batch)
            result = await inspect(client, auth[0], batch)
            assert result["counts"]["failed"] == 1
            assert len(result["attempts"]) == (round_number + 1) * 3
            response = await client.post(
                f"/api/v1/ingestion-batches/{batch['id']}/retry", headers=auth[0]
            )
            assert response.status_code == (409 if round_number == 2 else 202)
        assert await claim_one(UUID(batch["id"])) is None


@pytest.mark.asyncio
async def test_expired_owner_cannot_overwrite_new_owner_atomic_publication(monkeypatch, tmp_path):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, embed):
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def blocked_first(texts, *_):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return await embed(texts)

        monkeypatch.setattr(ingestion, "embed_texts", blocked_first)
        batch = await upload(client, auth[0], kb)
        old = await claim_one(UUID(batch["id"]))
        assert await renew(old) is True
        assert await claim_one(UUID(batch["id"])) is None
        task = asyncio.create_task(ingestion.process_document(old.document_id, control=old))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            async with SessionLocal() as session:
                await session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == old.job_id)
                    .values(lease_expires_at=utc_now() - timedelta(seconds=1))
                )
                await session.commit()
            new = await claim_one(UUID(batch["id"]))
            assert new.token != old.token
            await run_claim(new)
            async with SessionLocal() as session:
                before = set(
                    await session.scalars(
                        select(Chunk.id).where(Chunk.document_id == old.document_id)
                    )
                )
                assert before
            release.set()
            with pytest.raises(LeaseLost):
                await task
            assert await renew(old) is False
            async with SessionLocal() as session:
                assert (
                    set(
                        await session.scalars(
                            select(Chunk.id).where(Chunk.document_id == old.document_id)
                        )
                    )
                    == before
                )
                assert (await session.get(Document, old.document_id)).status == "processed"
            result = await inspect(client, auth[0], batch)
            assert [a["status"] for a in result["attempts"]] == ["interrupted", "completed"]
            assert result["counts"]["completed"] == 1
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_graceful_shutdown_persists_interruption_and_recovers_from_cache(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, embed):
        started = asyncio.Event()

        async def blocked(*_):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(ingestion, "embed_texts", blocked)
        batch = await upload(client, auth[0], kb)
        control = await claim_one(UUID(batch["id"]))
        task = asyncio.create_task(run_claim(control))
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        result = await inspect(client, auth[0], batch)
        assert result["items"][0]["status"] == "retry_wait"
        assert result["attempts"][0]["status"] == "interrupted"
        monkeypatch.setattr(ingestion, "embed_texts", embed)
        await due(batch)
        await one(batch)
        assert (await inspect(client, auth[0], batch))["items"][0]["cache_reused"] is True


@pytest.mark.asyncio
async def test_explicit_failed_pages_are_preserved_and_provider_errors_redacted(
    monkeypatch, tmp_path
):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):

        async def broken(*_):
            raise ParsingFailure("fixture page conversion failed", {"failed_pages": [2, 4]})

        monkeypatch.setattr(ingestion, "parse_for_ingestion", broken)
        batch = await upload(client, auth[0], kb)
        await one(batch)
        result = await inspect(client, auth[0], batch)
        assert result["items"][0]["failure_pages_known"] is True
        assert [p["page_number"] for p in result["items"][0]["failure_pages"]] == [2, 4]
        from knowflow.app.ingestion_jobs import failure_info

        response = httpx.Response(503, request=httpx.Request("POST", "https://example.invalid"))
        code, message, retryable = failure_info(
            httpx.HTTPStatusError("SYNTHETIC-SECRET", request=response.request, response=response),
            "embedding",
        )
        assert code == "embedding_http_503" and retryable
        assert "SYNTHETIC-SECRET" not in message


@pytest.mark.asyncio
async def test_docling_preflight_corrupt_pdf_is_an_input_failure(monkeypatch, tmp_path):
    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, _):
        monkeypatch.setattr(get_settings(), "pdf_parser", "docling")
        batch = await upload(
            client, auth[0], kb, [("broken.pdf", b"%PDF-1.7\nintentionally broken")]
        )
        await one(batch)
        result = await inspect(client, auth[0], batch)
        assert result["items"][0]["error_code"] == "parsing_invalid_pdf"
        assert result["items"][0]["retryable"] is False
        assert result["items"][0]["failure_pages_known"] is False


@pytest.mark.asyncio
async def test_lost_heartbeat_cancels_old_work_before_reclaim(monkeypatch, tmp_path):
    from knowflow.app import ingestion_jobs

    async with workspace(monkeypatch, tmp_path) as (client, auth, kb, embed):
        started = asyncio.Event()

        async def blocked(*_):
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(ingestion, "embed_texts", blocked)
        monkeypatch.setattr(ingestion_jobs, "HEARTBEAT_SECONDS", 0.02)
        batch = await upload(client, auth[0], kb)
        old = await claim_one(UUID(batch["id"]))
        task = asyncio.create_task(run_claim(old))
        try:
            await asyncio.wait_for(started.wait(), timeout=5)
            async with SessionLocal() as session:
                await session.execute(
                    update(IngestionJob)
                    .where(IngestionJob.id == old.job_id)
                    .values(lease_expires_at=utc_now() - timedelta(seconds=1))
                )
                await session.commit()
            await asyncio.wait_for(task, timeout=3)
            monkeypatch.setattr(ingestion, "embed_texts", embed)
            await one(batch)
            assert (await inspect(client, auth[0], batch))["counts"]["completed"] == 1
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
