"""Read-only owner-scoped diagnostics. No parser, model or index mutation."""

import asyncio
import json
import time
import uuid
from collections import Counter
from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Response
from sqlalchemy import Text, case, cast, func, select

from knowflow.app.api import SessionDep, UserDep, get_document
from knowflow.app.batch_api import MAX_FILES, owned_batch
from knowflow.app.config import get_settings
from knowflow.app.ingestion_diagnostics import diagnose
from knowflow.app.models import Chunk, Document, DocumentBlock, IngestionJob
from knowflow.app.token_budget import get_token_counter

router = APIRouter(prefix="/api/v1", tags=["diagnostics"])
LIMITS = {
    "blocks": 5000,
    "chunks": 2000,
    "row_characters": 100000,
    "block_text_characters": 100000,
    "source_bytes": 64000,
    "inspection_bytes": 4000000,
}
BATCH_LIMITS = {"inspection_bytes": 8000000, "admission_seconds": 10}


class IndexChanged(HTTPException):
    def __init__(self, read_bytes):
        super().__init__(409, "诊断期间索引发生变化，请重新检查。")
        self.read_bytes = read_bytes


async def read_rows(session, document_id, model, columns, limit, remaining, *, keys=None):
    rows = []
    stmt = (
        select(*columns)
        .where(model.document_id == document_id)
        .order_by(model.ordinal)
        .limit(limit)
        .execution_options(yield_per=16)
    )
    if keys is not None:
        stmt = stmt.where(model.block_key.in_(keys))
    stream = await session.stream(stmt)
    try:
        async for row in stream.mappings():
            item = dict(row)
            size = len(json.dumps(item, default=str, ensure_ascii=False).encode("utf-8"))
            if size > remaining:
                break
            remaining -= size
            rows.append(item)
    finally:
        await stream.close()
    return rows, remaining


async def inspect_document(session, doc, *, inspection_bytes=None):
    """Caller must have checked document ownership. The byte budget may be shared."""
    initial_updated_at = doc.updated_at
    byte_budget = LIMITS["inspection_bytes"] if inspection_bytes is None else inspection_bytes
    totals = {
        name: await session.scalar(
            select(func.count()).select_from(model).where(model.document_id == doc.id)
        )
        for name, model in (("blocks", DocumentBlock), ("chunks", Chunk))
    }
    blocks, remaining = await read_rows(
        session,
        doc.id,
        DocumentBlock,
        [
            DocumentBlock.block_key,
            DocumentBlock.ordinal,
            DocumentBlock.kind,
            func.length(DocumentBlock.text).label("text_length"),
        ],
        LIMITS["blocks"],
        byte_budget,
    )
    spans = case(
        (
            func.octet_length(cast(Chunk.source_spans, Text)) <= LIMITS["source_bytes"],
            Chunk.source_spans,
        ),
        else_=None,
    ).label("source_spans")
    chunks, remaining = await read_rows(
        session,
        doc.id,
        Chunk,
        [
            Chunk.id,
            Chunk.ordinal,
            Chunk.section,
            spans,
            func.substr(Chunk.content, 1, LIMITS["row_characters"]).label("content"),
            func.length(Chunk.content).label("text_length"),
        ],
        LIMITS["chunks"],
        remaining,
    )
    # Phase 2: source text only for blocks referenced by spans (character-mapped
    # consistency). Reads count against the same byte budget; anything not loaded
    # becomes an honest not_comparable, never a guessed match.
    block_keys = sorted(
        {
            span.get("block_key")
            for chunk in chunks
            for span in (chunk.get("source_spans") or [])
            if isinstance(span, dict)
            and isinstance(span.get("block_key"), str)
            and span.get("block_key")
            and isinstance(span.get("text_range"), list)
            and isinstance(span.get("chunk_range"), list)
        }
    )
    block_texts = {}
    if block_keys:
        block_rows, remaining = await read_rows(
            session,
            doc.id,
            DocumentBlock,
            [
                DocumentBlock.block_key,
                func.substr(DocumentBlock.text, 1, LIMITS["block_text_characters"]).label("text"),
            ],
            len(block_keys),
            remaining,
            keys=block_keys,
        )
        block_texts = {row["block_key"]: row["text"] for row in block_rows}
    try:
        counter = await asyncio.to_thread(get_token_counter, get_settings().chunk_tokenizer_path)
    except (OSError, ValueError):
        counter = None
    result = await asyncio.to_thread(
        diagnose,
        {
            "id": str(doc.id),
            "filename": doc.filename,
            "status": doc.status.value,
            "updated_at": doc.updated_at.isoformat(),
            "source_sha256": doc.sha256,
            "parsing_report": doc.parsing_report,
        },
        blocks,
        chunks,
        totals,
        counter=counter,
        limits=LIMITS,
        block_texts=block_texts,
    )
    # READ COMMITTED can cross a concurrent rebuild. Do not claim a stable snapshot.
    latest_updated_at = await session.scalar(
        select(Document.updated_at).where(Document.id == doc.id)
    )
    if latest_updated_at != initial_updated_at:
        raise IndexChanged(byte_budget - remaining)
    result["scope"]["read_bytes"] = byte_budget - remaining
    result["scope"]["limits"] = {**LIMITS, "inspection_bytes": byte_budget}
    result["document_updated_at"] = initial_updated_at.isoformat()
    return result


@router.get("/documents/{document_id}/diagnostics")
async def document_diagnostics(
    document_id: uuid.UUID, user: UserDep, session: SessionDep, response: Response
):
    doc = await get_document(document_id, user, session)
    result = await inspect_document(session, doc)
    response.headers["Cache-Control"] = "private, no-store"
    return result


async def job_snapshot(session, batch_id):
    rows = await session.execute(
        select(
            IngestionJob.id,
            IngestionJob.document_id,
            IngestionJob.filename,
            IngestionJob.ordinal,
            IngestionJob.status,
            IngestionJob.stage,
            IngestionJob.attempt_count,
        )
        .where(IngestionJob.batch_id == batch_id)
        .order_by(IngestionJob.ordinal)
        .limit(MAX_FILES + 1)
    )
    return [dict(row) for row in rows.mappings()]


@router.get("/ingestion-batches/{batch_id}/diagnostics")
async def batch_diagnostics(
    batch_id: uuid.UUID, user: UserDep, session: SessionDep, response: Response
):
    batch = await owned_batch(session, batch_id, user)
    jobs = await job_snapshot(session, batch.id)
    if len(jobs) > MAX_FILES:
        raise HTTPException(422, "批次超过诊断的 20 个输入上限，未执行部分扫描。")
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    remaining = BATCH_LIMITS["inspection_bytes"]
    documents, items, cached = {}, [], {}
    for job in jobs:
        row = {**job, "inspection_status": "unavailable", "reason": "document_missing"}
        items.append(row)
        doc_id = job["document_id"]
        if doc_id is None:
            if job["status"] == "rejected":
                row.update(inspection_status="not_applicable", reason="rejected_no_document")
            continue
        key = str(doc_id)
        if key in cached:
            row.update(cached[key], reused=True)
            continue
        doc = await session.scalar(
            select(Document)
            .where(
                Document.id == doc_id,
                Document.knowledge_base_id == batch.knowledge_base_id,
            )
            .execution_options(populate_existing=True)
        )
        if doc is None:
            cached[key] = {"inspection_status": "unavailable", "reason": "document_missing"}
        elif doc.status.value in {"pending", "processing"}:
            cached[key] = {"inspection_status": "deferred", "reason": "document_in_progress"}
        elif remaining <= 0 or time.monotonic() - started >= BATCH_LIMITS["admission_seconds"]:
            cached[key] = {"inspection_status": "deferred", "reason": "batch_budget_exhausted"}
        else:
            try:
                result = await inspect_document(
                    session, doc, inspection_bytes=min(LIMITS["inspection_bytes"], remaining)
                )
                remaining -= result["scope"]["read_bytes"]
                documents[key] = result
                cached[key] = {"inspection_status": "inspected", "reason": None}
            except IndexChanged as exc:
                remaining -= exc.read_bytes
                cached[key] = {"inspection_status": "error", "reason": "index_changed"}
        row.update(cached[key], reused=False)
    final_jobs = await job_snapshot(session, batch.id)
    versions = await session.execute(
        select(Document.id, Document.updated_at).where(
            Document.knowledge_base_id == batch.knowledge_base_id,
            Document.id.in_([uuid.UUID(key) for key in documents]),
        )
    )
    current_versions = {str(id_): updated.isoformat() for id_, updated in versions}
    stale_ids = [
        key
        for key, doc in documents.items()
        if current_versions.get(key) != doc["document_updated_at"]
    ]
    for key in stale_ids:
        documents[key]["stale"] = True
    counts = Counter(row["inspection_status"] for row in items)
    finding_documents = Counter(f["code"] for d in documents.values() for f in d["findings"])
    response.headers["Cache-Control"] = "private, no-store"
    return {
        "schema_version": "batch-integrity-v1",
        "batch_id": str(batch.id),
        "started_at": started_at,
        "completed_at": datetime.now(UTC).isoformat(),
        "items": items,
        "documents": documents,
        "summary": {
            "n_inputs": len(items),
            "n_unique_inspected_documents": len(documents),
            "inspection_status_counts": {
                k: counts[k]
                for k in ("inspected", "unavailable", "not_applicable", "deferred", "error")
            },
            "finding_documents_by_code": dict(finding_documents),
        },
        "budget": {**BATCH_LIMITS, "read_bytes": BATCH_LIMITS["inspection_bytes"] - remaining},
        "changed_during_inspection": final_jobs != jobs
        or bool(stale_ids)
        or any(row["reason"] == "index_changed" for row in items),
        "stale_document_ids": stale_ids,
        "note": (
            "逐文档只读诊断，不是批次原子快照；检查返回不代表质量通过。"
            "重复输入保留，来源诊断按唯一文档计数。"
        ),
    }
