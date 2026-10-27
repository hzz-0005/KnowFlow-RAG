"""Bounded multi-file admission and owner-scoped durable batch history."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import APIRouter, File, Header, HTTPException, Query, UploadFile
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from knowflow.app.api import UserDep, owned_knowledge_base
from knowflow.app.config import get_settings
from knowflow.app.document_processing import SUPPORTED_EXTENSIONS
from knowflow.app.ingestion_jobs import ACTIVE, MAX_TOTAL_ATTEMPTS
from knowflow.app.models import (
    Document,
    DocumentStatus,
    IngestionAttempt,
    IngestionBatch,
    IngestionJob,
    utc_now,
)
from knowflow.app.security import SessionDep

router = APIRouter(prefix="/api/v1")
MAX_FILES = 20
MAX_BATCH_BYTES = 100 * 1024 * 1024
MAX_QUEUED = 200


async def owned_batch(session, batch_id, user):
    batch = await session.get(IngestionBatch, batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    await owned_knowledge_base(session, batch.knowledge_base_id, user.id)
    return batch


def batch_summary(batch, jobs):
    counts = Counter(job.status for job in jobs)
    active = sum(counts[s] for s in ACTIVE)
    status = (
        "running"
        if active
        else "completed_with_issues"
        if counts["failed"] or counts["rejected"]
        else "completed"
    )
    return {
        "id": batch.id,
        "created_at": batch.created_at,
        "knowledge_base_id": batch.knowledge_base_id,
        "status": status,
        "total": len(jobs),
        "active": active,
        "counts": {
            s: counts[s]
            for s in [
                "queued",
                "running",
                "retry_wait",
                "completed",
                "failed",
                "skipped",
                "rejected",
            ]
        },
        "can_retry": any(
            j.status == "failed"
            and j.document_id is not None
            and j.attempt_count < MAX_TOTAL_ATTEMPTS
            for j in jobs
        ),
        "items": [
            {
                "id": j.id,
                "document_id": j.document_id,
                "filename": j.filename,
                "ordinal": j.ordinal,
                "status": j.status,
                "stage": j.stage,
                "attempt_count": j.attempt_count,
                "max_attempts": j.max_attempts,
                "retryable": j.retryable,
                "error_code": j.error_code,
                "error_message": j.error_message,
                "failure_pages": j.failure_pages,
                "failure_pages_known": j.failure_pages_known,
                "cache_reused": j.cache_reused,
            }
            for j in jobs
        ],
    }


async def summary(session, batch):
    jobs = list(
        await session.scalars(
            select(IngestionJob)
            .where(IngestionJob.batch_id == batch.id)
            .order_by(IngestionJob.ordinal)
        )
    )
    return batch_summary(batch, jobs)


async def prepare_files(files):
    settings = get_settings()
    prepared, read_bytes = [], 0
    for upload in files:
        original = (upload.filename or "document").replace("\\", "/").split("/")[-1]
        suffix = Path(original).suffix.lower()
        name = original[:255]
        if suffix in SUPPORTED_EXTENSIONS and len(original) > 255:
            name = original[: 255 - len(suffix)] + suffix
        item = {
            "filename": name,
            "suffix": suffix,
            "media_type": upload.content_type or "application/octet-stream",
        }
        if suffix not in SUPPORTED_EXTENSIONS:
            item.update(error_code="unsupported_format", error_message="不支持此文件类型。")
        elif read_bytes >= MAX_BATCH_BYTES:
            item.update(
                error_code="batch_bytes_exceeded",
                error_message="本批已达到 100 MiB 读取预算，请另开一批。",
            )
        else:
            limit = min(settings.max_upload_mb * 1024 * 1024, MAX_BATCH_BYTES - read_bytes)
            data = await upload.read(limit + 1)
            read_bytes += len(data)
            if len(data) > limit:
                item.update(
                    error_code="upload_size_exceeded",
                    error_message="文件超过单文件限制或本批剩余 100 MiB 预算。",
                )
            elif not data:
                item.update(error_code="empty_file", error_message="文件为空。")
            else:
                item.update(data=data, sha256=hashlib.sha256(data).hexdigest())
        prepared.append(item)
    fingerprint = hashlib.sha256(
        json.dumps(
            [
                {k: item[k] for k in ["filename", "sha256", "error_code"] if k in item}
                for item in prepared
            ],
            ensure_ascii=True,
        ).encode()
    ).hexdigest()
    return prepared, fingerprint


@router.post("/knowledge-bases/{knowledge_base_id}/ingestion-batches", status_code=202)
async def upload_batch(
    knowledge_base_id: UUID,
    user: UserDep,
    session: SessionDep,
    files: Annotated[list[UploadFile], File()],
    idempotency_key: Annotated[str | None, Header(pattern=r"^[a-zA-Z0-9_-]{1,64}$")] = None,
):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    if not 1 <= len(files) <= MAX_FILES:
        raise HTTPException(422, "每批需要 1–20 个文件。")
    prepared, fingerprint = await prepare_files(files)
    # Serialize queue admission, not model work, so the queue cap is not a racy count.
    await session.execute(text("SELECT pg_advisory_xact_lock(1262897518)"))
    key = idempotency_key or uuid4().hex
    existing = await session.scalar(
        select(IngestionBatch).where(
            IngestionBatch.knowledge_base_id == knowledge_base_id, IngestionBatch.request_key == key
        )
    )
    if existing:
        if existing.request_fingerprint != fingerprint:
            raise HTTPException(409, "同一请求键对应了不同文件，请使用新的请求键。")
        return await summary(session, existing)
    queued = await session.scalar(
        select(func.count()).select_from(IngestionJob).where(IngestionJob.status.in_(ACTIVE))
    )
    if queued + len(files) > MAX_QUEUED:
        raise HTTPException(429, "摄入队列已满，请等待已有任务处理后再上传。")
    batch = IngestionBatch(
        id=uuid4(),
        knowledge_base_id=knowledge_base_id,
        request_key=key,
        request_fingerprint=fingerprint,
    )
    session.add(batch)
    await session.flush()
    paths = []
    try:
        for ordinal, item in enumerate(prepared):
            job = IngestionJob(batch_id=batch.id, filename=item["filename"], ordinal=ordinal)
            if "error_code" in item:
                job.status, job.stage = "rejected", "admission"
                job.error_code, job.error_message = item["error_code"], item["error_message"]
            else:
                existing_document = await session.scalar(
                    select(Document.id).where(
                        Document.knowledge_base_id == knowledge_base_id,
                        Document.sha256 == item["sha256"],
                    )
                )
                if existing_document:
                    job.document_id, job.status, job.stage = (
                        existing_document,
                        "skipped",
                        "admission",
                    )
                    job.error_code, job.error_message = (
                        "duplicate",
                        "此知识库已有相同文件，未重复入库。",
                    )
                else:
                    path = Path(get_settings().upload_dir) / f"{uuid4().hex}{item['suffix']}"
                    try:
                        async with session.begin_nested():
                            document = Document(
                                id=uuid4(),
                                knowledge_base_id=knowledge_base_id,
                                uploaded_by=user.id,
                                filename=item["filename"],
                                storage_key=path.name,
                                media_type=item["media_type"],
                                file_size=len(item["data"]),
                                sha256=item["sha256"],
                                status=DocumentStatus.PENDING,
                            )
                            session.add(document)
                            await session.flush()
                            paths.append(path)
                            path.parent.mkdir(parents=True, exist_ok=True)
                            await asyncio.to_thread(path.write_bytes, item["data"])
                            job.document_id = document.id
                    except IntegrityError:
                        job.document_id = await session.scalar(
                            select(Document.id).where(
                                Document.knowledge_base_id == knowledge_base_id,
                                Document.sha256 == item["sha256"],
                            )
                        )
                        if job.document_id is None:
                            raise
                        job.status, job.stage = "skipped", "admission"
                        job.error_code, job.error_message = (
                            "duplicate",
                            "此文件已被另一个请求入库。",
                        )
                    except OSError:
                        path.unlink(missing_ok=True)
                        job.status, job.stage = "rejected", "admission"
                        job.error_code, job.error_message = (
                            "storage_error",
                            "文件写入失败，请检查存储后重新提交。",
                        )
            if job.status in {"skipped", "rejected"}:
                job.finished_at = utc_now()
            session.add(job)
            await session.flush()
        await session.commit()
    except BaseException:
        await session.rollback()
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return await summary(session, batch)


@router.get("/knowledge-bases/{knowledge_base_id}/ingestion-batches")
async def list_batches(
    knowledge_base_id: UUID,
    user: UserDep,
    session: SessionDep,
    limit: int = Query(default=10, ge=1, le=30),
):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    batches = list(
        await session.scalars(
            select(IngestionBatch)
            .where(IngestionBatch.knowledge_base_id == knowledge_base_id)
            .order_by(IngestionBatch.created_at.desc())
            .limit(limit)
        )
    )
    return [await summary(session, batch) for batch in batches]


@router.get("/ingestion-batches/{batch_id}")
async def inspect_batch(batch_id: UUID, user: UserDep, session: SessionDep):
    batch = await owned_batch(session, batch_id, user)
    result = await summary(session, batch)
    attempts = await session.scalars(
        select(IngestionAttempt)
        .join(IngestionJob)
        .where(IngestionJob.batch_id == batch.id)
        .order_by(IngestionAttempt.started_at, IngestionAttempt.number)
    )
    result["attempts"] = [
        {
            "id": a.id,
            "job_id": a.job_id,
            "number": a.number,
            "retry_of_attempt_id": a.retry_of_attempt_id,
            "status": a.status,
            "stage": a.stage,
            "error_code": a.error_code,
            "error_message": a.error_message,
            "started_at": a.started_at,
            "finished_at": a.finished_at,
            "execution_config": a.execution_config,
        }
        for a in attempts
    ]
    return result


@router.post("/ingestion-batches/{batch_id}/retry", status_code=202)
async def retry_batch(batch_id: UUID, user: UserDep, session: SessionDep):
    batch = await owned_batch(session, batch_id, user)
    await session.execute(text("SELECT pg_advisory_xact_lock(1262897518)"))
    await session.refresh(batch, with_for_update=True)
    jobs = list(
        await session.scalars(
            select(IngestionJob).where(IngestionJob.batch_id == batch.id).with_for_update()
        )
    )
    count = 0
    for job in jobs:
        if job.status == "failed" and job.document_id and job.attempt_count < MAX_TOTAL_ATTEMPTS:
            document = await session.get(Document, job.document_id, with_for_update=True)
            if document is None or document.status in {
                DocumentStatus.PENDING,
                DocumentStatus.PROCESSING,
            }:
                continue
            job.status, job.available_at, job.finished_at = "queued", utc_now(), None
            job.stage, job.error_code, job.error_message = "uploaded", None, None
            job.failure_pages, job.failure_pages_known = [], False
            job.max_attempts = min(MAX_TOTAL_ATTEMPTS, job.attempt_count + 3)
            document.status, document.error_message = DocumentStatus.PENDING, None
            count += 1
    if not count:
        raise HTTPException(
            409, "没有可重试文件；被拒收文件须修正后重新上传，单文件总尝试上限为 9 次。"
        )
    # Include rows just requeued in this transaction's count (autoflush is enabled).
    queued = await session.scalar(
        select(func.count()).select_from(IngestionJob).where(IngestionJob.status.in_(ACTIVE))
    )
    if queued > MAX_QUEUED:
        raise HTTPException(429, "摄入队列已满，请稍后重试。")
    await session.commit()
    return await summary(session, batch)
