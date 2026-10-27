"""PostgreSQL queue: at-least-once work, fenced atomic index publication."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import timedelta
from uuid import uuid4

import httpx
from pypdf.errors import PyPdfError
from sqlalchemy import or_, select, update
from sqlalchemy.exc import OperationalError

from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Document, DocumentStatus, IngestionAttempt, IngestionJob, utc_now

ACTIVE = {"queued", "running", "retry_wait"}
LEASE_SECONDS = 45
HEARTBEAT_SECONDS = 10
MAX_TOTAL_ATTEMPTS = 9
logger = logging.getLogger(__name__)


class LeaseLost(Exception):
    pass


def execution_config(settings, document):
    fields = [
        "embedding_model",
        "embedding_dimensions",
        "pdf_parser",
        "parse_artifact_revision",
        "chunking_strategy",
        "chunk_max_tokens",
        "chunk_target_tokens",
        "chunk_overlap_tokens",
        "chunk_token_reserve",
    ]
    return {key: getattr(settings, key) for key in fields} | {"source_sha256": document.sha256}


def failure_info(exc, stage):
    """Do not store raw provider bodies, request URLs or credentials in job errors."""
    cause, seen = exc, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if stage == "parsing" and isinstance(cause, PyPdfError):
            return (
                "parsing_invalid_pdf",
                "PDF 文件损坏、加密或无法读取，请修复原件后重新上传。",
                False,
            )
        code = getattr(cause, "status_code", None)
        response = getattr(cause, "response", None)
        code = code or getattr(response, "status_code", None)
        if isinstance(code, int):
            retryable = code in {408, 409, 429} or code >= 500
            return f"{stage}_http_{code}", f"{stage} 服务返回 HTTP {code}。", retryable
        if isinstance(cause, (TimeoutError, httpx.TimeoutException)):
            return f"{stage}_timeout", f"{stage} 阶段超时。", True
        if isinstance(cause, (ConnectionError, httpx.TransportError)):
            return f"{stage}_connection_error", f"{stage} 阶段连接服务失败。", True
        if isinstance(cause, OperationalError):
            return f"{stage}_database_unavailable", f"{stage} 阶段数据库暂时不可用。", True
        if isinstance(cause, InterruptedError):
            return "worker_interrupted", "工作进程停止，等待恢复。", True
        cause = cause.__cause__
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return f"{stage}_invalid_input", str(exc)[:1000], False
    return f"{stage}_internal_error", f"{stage} 阶段发生内部错误，请检查服务日志。", False


def failed_pages(report):
    pages = (report or {}).get("failed_pages")
    if not isinstance(pages, list) or not pages:
        return [], False
    normalized = []
    for item in pages:
        number = (
            item
            if type(item) is int
            else item.get("page_number")
            if isinstance(item, dict)
            else None
        )
        if type(number) is int and number > 0:
            reason = item.get("reason") if isinstance(item, dict) else None
            normalized.append(
                {
                    "page_number": number,
                    "reason": reason[:500]
                    if isinstance(reason, str) and reason
                    else "page_reason_unknown",
                }
            )
    return normalized, len(normalized) == len(pages)


class JobControl:
    def __init__(self, job, attempt):
        self.job_id, self.document_id, self.token = job.id, job.document_id, job.lease_token
        self.attempt_id, self.attempt_number = attempt.id, attempt.number
        self.stage_name = "reading"

    async def locked(self, session):
        job = await session.get(
            IngestionJob, self.job_id, with_for_update=True, populate_existing=True
        )
        if (
            job is None
            or job.status != "running"
            or job.lease_token != self.token
            or job.lease_expires_at is None
            or job.lease_expires_at <= utc_now()
        ):
            raise LeaseLost()
        return job

    async def checkpoint(self, stage, *, cache_reused=None):
        async with SessionLocal() as session:
            job = await self.locked(session)
            job.stage = self.stage_name = stage
            if cache_reused is not None:
                job.cache_reused = cache_reused
            attempt = await session.get(IngestionAttempt, self.attempt_id)
            attempt.stage = stage
            await session.commit()

    async def complete(self, session):
        job = await self.locked(session)
        job.status, job.stage, job.finished_at = "completed", "completed", utc_now()
        job.error_code = job.error_message = None
        job.failure_pages, job.failure_pages_known = [], False
        job.lease_token = job.lease_expires_at = None
        attempt = await session.get(IngestionAttempt, self.attempt_id)
        attempt.status, attempt.stage, attempt.finished_at = "completed", "completed", utc_now()

    async def fail(self, session, exc, report=None):
        job = await self.locked(session)
        code, message, retryable = failure_info(exc, self.stage_name)
        job.error_code, job.error_message, job.retryable = code, message, retryable
        job.failure_pages, job.failure_pages_known = failed_pages(report)
        job.status = (
            "retry_wait" if retryable and job.attempt_count < job.max_attempts else "failed"
        )
        job.available_at = utc_now() + timedelta(seconds=min(60, 2**job.attempt_count))
        job.finished_at = None if job.status == "retry_wait" else utc_now()
        job.lease_token = job.lease_expires_at = None
        attempt = await session.get(IngestionAttempt, self.attempt_id)
        attempt.status = "interrupted" if isinstance(exc, InterruptedError) else "failed"
        attempt.error_code, attempt.error_message, attempt.finished_at = code, message, utc_now()
        document = (
            await session.get(Document, job.document_id, with_for_update=True)
            if job.document_id
            else None
        )
        if document:
            document.status, document.error_message = DocumentStatus.FAILED, message
            if report is not None:
                document.parsing_report = report


async def claim_one(batch_id=None):
    now = utc_now()
    async with SessionLocal() as session:
        query = select(IngestionJob).where(
            or_(
                (IngestionJob.status.in_(["queued", "retry_wait"]))
                & (IngestionJob.available_at <= now),
                (IngestionJob.status == "running") & (IngestionJob.lease_expires_at <= now),
            )
        )
        if batch_id:
            query = query.where(IngestionJob.batch_id == batch_id)
        job = await session.scalar(
            query.order_by(IngestionJob.available_at, IngestionJob.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        if job is None:
            return None
        previous = await session.scalar(
            select(IngestionAttempt)
            .where(IngestionAttempt.job_id == job.id)
            .order_by(IngestionAttempt.number.desc())
            .limit(1)
        )
        if job.status == "running" and previous:
            previous.status, previous.finished_at = "interrupted", now
            previous.error_code, previous.error_message = (
                "lease_expired",
                "租约过期，工作进程可能已中断。",
            )
        document = await session.get(Document, job.document_id) if job.document_id else None
        if document is None or job.attempt_count >= job.max_attempts:
            job.status, job.finished_at = "failed", now
            job.error_code = "document_deleted" if document is None else "retry_budget_exhausted"
            job.error_message = "原文档已删除。" if document is None else "中断恢复已达到尝试上限。"
            job.lease_token = job.lease_expires_at = None
            job.failure_pages, job.failure_pages_known = [], False
            if document:
                document.status, document.error_message = DocumentStatus.FAILED, job.error_message
            await session.commit()
            return None
        job.status, job.stage = "running", "reading"
        job.error_code = job.error_message = None
        job.failure_pages, job.failure_pages_known = [], False
        job.cache_reused, job.finished_at = False, None
        job.attempt_count += 1
        job.lease_token, job.lease_expires_at = uuid4(), now + timedelta(seconds=LEASE_SECONDS)
        attempt = IngestionAttempt(
            id=uuid4(),
            job_id=job.id,
            number=job.attempt_count,
            retry_of_attempt_id=previous.id if previous else None,
            execution_config=execution_config(get_settings(), document),
        )
        session.add(attempt)
        await session.commit()
        return JobControl(job, attempt)


async def renew(control):
    async with SessionLocal() as session:
        result = await session.execute(
            update(IngestionJob)
            .where(
                IngestionJob.id == control.job_id,
                IngestionJob.status == "running",
                IngestionJob.lease_token == control.token,
                IngestionJob.lease_expires_at > utc_now(),
            )
            .values(lease_expires_at=utc_now() + timedelta(seconds=LEASE_SECONDS))
        )
        await session.commit()
        return result.rowcount == 1


async def _heartbeat(control):
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        if not await renew(control):
            return


async def record_failure(control, exc):
    async with SessionLocal() as session:
        with suppress(LeaseLost):
            await control.fail(session, exc)
            await session.commit()


async def run_claim(control):
    from knowflow.app.ingestion import process_document

    heartbeat = asyncio.create_task(_heartbeat(control))
    processing = asyncio.create_task(process_document(control.document_id, control=control))
    try:
        async with asyncio.timeout(get_settings().ingestion_attempt_timeout_seconds):
            done, _ = await asyncio.wait(
                [heartbeat, processing], return_when=asyncio.FIRST_COMPLETED
            )
            if processing in done:
                await processing
            else:
                # A lost heartbeat cannot leave a costly task running until its full deadline.
                await heartbeat
                raise LeaseLost()
    except LeaseLost:
        pass  # A new owner may already be working; the old owner cannot write results.
    except asyncio.CancelledError:
        processing.cancel()
        await asyncio.gather(processing, return_exceptions=True)
        await record_failure(control, InterruptedError())
        raise
    except Exception as exc:
        processing.cancel()
        await asyncio.gather(processing, return_exceptions=True)
        await record_failure(control, exc)
    finally:
        heartbeat.cancel()
        processing.cancel()
        await asyncio.gather(heartbeat, processing, return_exceptions=True)


async def worker_loop():
    while True:
        try:
            control = await claim_one()
            if control:
                await run_claim(control)
                continue
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Ingestion worker cycle failed: %s", type(exc).__name__)
        await asyncio.sleep(1)
