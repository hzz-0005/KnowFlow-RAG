from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError

from knowflow.app.answer_budget import (
    AnswerInputBudgetError,
    AnswerOutputLimit,
    plan_answer_context,
)
from knowflow.app.answer_guard import guard_answer
from knowflow.app.config import get_settings
from knowflow.app.document_links import rebuild_document_links, reference_excerpt
from knowflow.app.document_processing import SUPPORTED_EXTENSIONS
from knowflow.app.graph_index import index_document_graph, prune_orphan_entities
from knowflow.app.graph_inspection import inspect_graph
from knowflow.app.ingestion import process_document, sha256_bytes
from knowflow.app.ingestion_jobs import ACTIVE as ACTIVE_INGESTION_STATES
from knowflow.app.intent_context import build_routing_input, task_question
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentBlock,
    DocumentLink,
    DocumentStatus,
    IngestionJob,
    KnowledgeBase,
    OcrReview,
    User,
    utc_now,
)
from knowflow.app.ocr_review import expire_reviews, run_ocr_review, stored_snapshot
from knowflow.app.parse_artifacts import artifact_path, load_artifact
from knowflow.app.providers import ANSWER_PROMPT_VERSION, generate_answer
from knowflow.app.query_plan import plan_query
from knowflow.app.reference_expansion import expand_references
from knowflow.app.relevance import ACCEPTED_VERDICTS, assess_retrieval, retain_supported_context
from knowflow.app.retrieval import (
    graph_candidate_diagnostics,
    retrieve,
    retrieve_per_document,
)
from knowflow.app.round_budget import RoundBudgetExceeded, scope
from knowflow.app.schemas import (
    Citation,
    DocumentResponse,
    EvaluationRunSummary,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    QualityEvaluationRequest,
    QueryRequest,
    QueryResponse,
    RegisterRequest,
    RetrievedEvidence,
    TokenResponse,
)
from knowflow.app.security import (
    SessionDep,
    create_access_token,
    current_user,
    hash_password,
    verify_password,
)
from knowflow.app.source_provenance import first_page, quote_spans
from knowflow.app.structured_parsing import render_pdf_page
from knowflow.app.task_planning import should_plan

router = APIRouter(prefix="/api/v1")
UserDep = Annotated[User, Depends(current_user)]
MAX_FILENAME_CHARS = 255
# Also detect malformed UUID-like / unmapped short labels so a mixed answer cannot
# bypass validation by dropping one character from an otherwise plausible marker.
CHUNK_ID_PATTERN = re.compile(r"\[([0-9a-fA-F-]{24,44}|KF[0-9a-zA-Z_-]{0,20})\]", re.I)
SOURCE_BRACKETS = re.compile(r"\[([^\[\]\r\n]*)\]")
CANONICAL_SOURCE_ID = re.compile(
    r"(?:[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}|KF[0-9a-z_-]{0,20})", re.I
)


def normalize_source_markers(text: str) -> str:
    """Split source lists before identity validation, without deleting answer data."""

    def normalize(match: re.Match[str]) -> str:
        parts = [p.strip() for p in re.split(r"[,，;；]", match.group(1))]
        if len(parts) < 2 or not any(CANONICAL_SOURCE_ID.fullmatch(p) for p in parts):
            return match.group(0)
        if len(parts) > 64 or not all(CHUNK_ID_PATTERN.fullmatch(f"[{p}]") for p in parts):
            raise ValueError("Invalid source-marker list")
        return "".join(f"[{p}]" for p in parts)

    return SOURCE_BRACKETS.sub(normalize, text)


def unstated_in_scope(source_scope, question) -> str | None:
    """Name the files that were searched but said nothing, instead of a bare refusal.

    A known scope makes the gap reportable: the answer can say which files were
    checked and for what, which is the honest form of "the documents do not say".
    """
    names = [row["filename"] for row in (source_scope.trace or {}).get("documents", [])]
    if not names or len(names) > 3:
        return None
    subject = re.sub(r"[?？。！!]+$", "", question).strip()
    listed = "、".join(f"《{name}》" for name in names)
    return (
        f"在 {listed} 的本轮检索范围里没有找到与“{subject}”直接相关的原文片段，"
        "因此不能给出有依据的回答；这不代表这些文件里没有相关内容，"
        "可以换个说法、换一个角度，或把范围放宽到整个知识库再问。"
    )


def extended_quote(
    base: str, chunk_text: str, *, cover_threshold: float = 0.4, cap: int = 1200
) -> str:
    """Return optional same-chunk context; never use this to replace the cited witness."""

    def norm(value: str) -> str:
        return re.sub(r"\s+", "", value)

    if cap <= 0:
        return ""
    if not base:
        return chunk_text[:cap]
    if len(norm(base)) >= cover_threshold * max(len(norm(chunk_text)), 1):
        return base
    if len(chunk_text) <= cap:
        return chunk_text
    anchors = [
        (chunk_text.find(q), len(q)) for q in base.split("\n…\n") if q and chunk_text.find(q) >= 0
    ]
    if not anchors:
        return base
    position, size = min(anchors)
    start = max(0, position - max(0, (cap - size) // 2))
    start = min(start, max(0, len(chunk_text) - cap))
    return chunk_text[start : start + cap]


PDF_RENDER_LOCK = asyncio.Lock()


async def owned_knowledge_base(
    session, knowledge_base_id: uuid.UUID, owner_id: uuid.UUID
) -> KnowledgeBase:
    knowledge_base = await session.scalar(
        select(KnowledgeBase).where(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.owner_id == owner_id,
        )
    )
    if knowledge_base is None:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return knowledge_base


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, session: SessionDep) -> TokenResponse:
    user = User(email=body.email.lower(), password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=409, detail="An account with this email already exists"
        ) from exc
    await session.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/auth/token", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], session: SessionDep
) -> TokenResponse:
    user = await session.scalar(select(User).where(User.email == form.username.lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/knowledge-bases", response_model=KnowledgeBaseResponse, status_code=201)
async def create_knowledge_base(body: KnowledgeBaseCreate, user: UserDep, session: SessionDep):
    knowledge_base = KnowledgeBase(owner_id=user.id, name=body.name.strip())
    if not knowledge_base.name:
        raise HTTPException(status_code=422, detail="Knowledge base name cannot be blank")
    session.add(knowledge_base)
    await session.commit()
    await session.refresh(knowledge_base)
    return knowledge_base


@router.get("/knowledge-bases/{knowledge_base_id}/graph")
async def get_knowledge_graph(
    knowledge_base_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
    document_id: uuid.UUID | None = None,
    q: str = Query(default="", max_length=200),
    status: Literal["all", "supported", "excluded", "unknown", "unreviewed"] = "all",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    if document_id:
        document = await session.get(Document, document_id)
        if document is None or document.knowledge_base_id != knowledge_base_id:
            raise HTTPException(status_code=404, detail="Document not found")
    return await inspect_graph(
        session,
        knowledge_base_id,
        document_id=document_id,
        query=q,
        offset=offset,
        limit=limit,
        status=status,
    )


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseResponse])
async def list_knowledge_bases(user: UserDep, session: SessionDep):
    result = await session.scalars(
        select(KnowledgeBase)
        .where(KnowledgeBase.owner_id == user.id)
        .order_by(KnowledgeBase.created_at.desc())
    )
    return list(result)


@router.delete("/knowledge-bases/{knowledge_base_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_knowledge_base(knowledge_base_id: uuid.UUID, user: UserDep, session: SessionDep):
    knowledge_base = await owned_knowledge_base(session, knowledge_base_id, user.id)
    documents = await session.scalars(
        select(Document.storage_key).where(Document.knowledge_base_id == knowledge_base_id)
    )
    settings = get_settings()
    stored_files = [
        path
        for key in documents
        for path in (Path(settings.upload_dir) / key, artifact_path(key, settings))
    ]
    await session.delete(knowledge_base)
    await session.commit()
    for stored_file in stored_files:
        await asyncio.to_thread(stored_file.unlink, missing_ok=True)


@router.post(
    "/knowledge-bases/{knowledge_base_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_document(
    knowledge_base_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    user: UserDep,
    session: SessionDep,
    file: Annotated[UploadFile, File()],
):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    original_filename = (file.filename or "document").replace("\\", "/").split("/")[-1]
    extension = Path(original_filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415, detail="Supported file types: TXT, MD, PDF, DOCX, CSV, XLSX"
        )

    settings = get_settings()
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {settings.max_upload_mb} MB upload limit"
        )
    if not data:
        raise HTTPException(status_code=422, detail="The uploaded document is empty")

    sha256 = sha256_bytes(data)
    existing = await session.scalar(
        select(Document.id).where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.sha256 == sha256,
        )
    )
    if existing is not None:
        raise HTTPException(
            status_code=409, detail=f"This file already exists as document {existing}"
        )

    document = Document(
        knowledge_base_id=knowledge_base_id,
        uploaded_by=user.id,
        filename=original_filename[:MAX_FILENAME_CHARS],
        storage_key=f"{uuid.uuid4().hex}{extension}",
        media_type=file.content_type or "application/octet-stream",
        file_size=len(data),
        sha256=sha256,
        status=DocumentStatus.PENDING,
    )
    target_path = Path(settings.upload_dir) / document.storage_key
    target_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(target_path.write_bytes, data)
    session.add(document)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="This file was uploaded concurrently") from exc
    await session.refresh(document)
    background_tasks.add_task(process_document, document.id)
    return document


@router.get("/knowledge-bases/{knowledge_base_id}/documents", response_model=list[DocumentResponse])
async def list_documents(knowledge_base_id: uuid.UUID, user: UserDep, session: SessionDep):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    result = await session.scalars(
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.created_at.desc())
    )
    return list(result)


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: uuid.UUID, user: UserDep, session: SessionDep):
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await owned_knowledge_base(session, document.knowledge_base_id, user.id)
    return document


@router.get("/documents/{document_id}/blocks")
async def get_document_blocks(
    document_id: uuid.UUID, user: UserDep, session: SessionDep, offset: int = 0, limit: int = 100
):
    document = await get_document(document_id, user, session)
    if offset < 0 or not 1 <= limit <= 200:
        raise HTTPException(status_code=422, detail="Invalid block pagination")
    blocks = await session.scalars(
        select(DocumentBlock)
        .where(DocumentBlock.document_id == document.id)
        .order_by(DocumentBlock.ordinal)
        .offset(offset)
        .limit(limit)
    )
    return [
        {
            "block_key": b.block_key,
            "kind": b.kind,
            "text": b.text,
            "section": b.section,
            "parent_key": b.parent_key,
            "source_spans": b.source_spans,
            "structure": b.structure,
        }
        for b in blocks
    ]


@router.get("/documents/{document_id}/pages/{page_number}")
async def get_source_page(
    document_id: uuid.UUID, page_number: int, user: UserDep, session: SessionDep
):
    document = await get_document(document_id, user, session)
    if not document.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="此文档没有 PDF 原页预览。")
    if page_number < 1:
        raise HTTPException(status_code=404, detail="Page not found")
    path = Path(get_settings().upload_dir) / document.storage_key
    try:
        async with PDF_RENDER_LOCK:
            data = await asyncio.to_thread(render_pdf_page, path, page_number)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail="Page not found") from exc
    return Response(
        content=data, media_type="image/png", headers={"Cache-Control": "private, no-store"}
    )


@router.get("/documents/{document_id}/table-cells")
async def inspect_table_cells(
    document_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
    block_key: str = Query(max_length=200),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=100),
):
    document = await get_document(document_id, user, session)
    block = await session.scalar(
        select(DocumentBlock).where(
            DocumentBlock.document_id == document.id,
            DocumentBlock.block_key == block_key,
            DocumentBlock.kind == "table",
        )
    )
    if block is None or block.structure.get("source_format") not in {"csv", "xlsx"}:
        raise HTTPException(status_code=404, detail="Table source not found")
    cells = block.structure.get("table_cells", [])
    return {
        "document_id": document.id,
        "document_status": document.status,
        "block_key": block.block_key,
        "sheet_name": block.structure.get("sheet_name"),
        "total": len(cells),
        "offset": offset,
        "limit": limit,
        "cells": cells[offset : offset + limit],
    }


@router.get("/documents/{document_id}/segments")
async def inspect_document_segments(
    document_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
    view: Literal["chunks", "blocks"] = "chunks",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    q: str = Query(default="", max_length=200),
    page_number: int | None = Query(default=None, ge=1),
):
    document = await get_document(document_id, user, session)
    model = Chunk if view == "chunks" else DocumentBlock
    text = Chunk.content if view == "chunks" else DocumentBlock.text
    base = model.document_id == document.id
    conditions = [base]
    if q.strip():
        needle = q.strip().lower()
        conditions.append(
            or_(
                func.strpos(func.lower(text), needle) > 0,
                func.strpos(func.lower(func.coalesce(model.section, "")), needle) > 0,
            )
        )
    if page_number is not None:
        page_match = model.source_spans.contains([{"page_number": page_number}])
        conditions.append(
            or_(Chunk.page_number == page_number, page_match) if view == "chunks" else page_match
        )
    length = func.length(text)
    stats = (
        await session.execute(
            select(func.count(), func.min(length), func.max(length), func.avg(length))
            .select_from(model)
            .where(base)
        )
    ).one()
    total = await session.scalar(select(func.count()).select_from(model).where(*conditions))
    rows = await session.scalars(
        select(model).where(*conditions).order_by(model.ordinal).offset(offset).limit(limit)
    )
    return {
        "document_id": document.id,
        "filename": document.filename,
        "document_status": document.status,
        "view": view,
        "offset": offset,
        "limit": limit,
        "total": total,
        "statistics": {
            "count": stats[0],
            "min_characters": stats[1],
            "max_characters": stats[2],
            "average_characters": round(float(stats[3]), 1) if stats[3] is not None else None,
        },
        "items": [
            {
                "id": str(row.id),
                "ordinal": row.ordinal,
                "kind": "chunk" if view == "chunks" else row.kind,
                "content": row.content if view == "chunks" else row.text,
                "characters": len(row.content if view == "chunks" else row.text),
                "section": row.section,
                "source_spans": row.source_spans,
                "page_number": row.page_number if view == "chunks" else None,
                "block_key": None if view == "chunks" else row.block_key,
                "processing_metadata": row.processing_metadata if view == "chunks" else {},
                "tabular_source": (
                    {
                        "sheet_name": row.structure.get("sheet_name"),
                        "cell_count": len(row.structure.get("table_cells", [])),
                    }
                    if view == "blocks" and row.structure.get("source_format") in {"csv", "xlsx"}
                    else None
                ),
            }
            for row in rows
        ],
    }


@router.post("/documents/{document_id}/reference-index")
async def build_reference_index(document_id: uuid.UUID, user: UserDep, session: SessionDep):
    document = await get_document(document_id, user, session)
    document = await session.get(
        Document, document.id, with_for_update=True, populate_existing=True
    )
    if document.status != DocumentStatus.PROCESSED:
        raise HTTPException(status_code=409, detail="请等待文档处理完成后再建立引用索引。")
    report = await rebuild_document_links(session, document)
    await session.commit()
    return report


@router.get("/documents/{document_id}/references")
async def inspect_references(
    document_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    status_filter: Literal[
        "all", "resolved", "missing", "ambiguous", "external_unresolved"
    ] = "all",
):
    document = await get_document(document_id, user, session)
    conditions = [DocumentLink.document_id == document.id]
    if status_filter != "all":
        conditions.append(DocumentLink.status == status_filter)
    total = await session.scalar(select(func.count()).select_from(DocumentLink).where(*conditions))
    links = list(
        await session.scalars(
            select(DocumentLink)
            .join(
                DocumentBlock,
                (DocumentBlock.document_id == DocumentLink.document_id)
                & (DocumentBlock.block_key == DocumentLink.source_block_key),
            )
            .where(*conditions)
            .order_by(DocumentBlock.ordinal, DocumentLink.start_offset)
            .offset(offset)
            .limit(limit)
        )
    )
    keys = {key for link in links for key in [link.source_block_key, *link.candidate_keys]}
    blocks = (
        {
            b.block_key: b
            for b in await session.scalars(
                select(DocumentBlock).where(
                    DocumentBlock.document_id == document.id, DocumentBlock.block_key.in_(keys)
                )
            )
        }
        if keys
        else {}
    )

    def block_info(key, offset=None):
        block = blocks.get(key)
        if block is None:
            return None
        text, spans = (
            reference_excerpt(block, offset)
            if offset is not None
            else (block.text[:1200], block.source_spans)
        )
        return {
            "block_key": block.block_key,
            "kind": block.kind,
            "text": text,
            "source_spans": spans,
            "caption_recovery": block.structure.get("caption_recovery")
            if block.kind == "table"
            else None,
        }

    return {
        "filename": document.filename,
        "document_id": document.id,
        "document_status": document.status,
        "report": document.parsing_report.get("reference_index"),
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": [
            {
                "id": link.id,
                "kind": link.kind,
                "label": link.label,
                "status": link.status,
                "source_quote": link.source_quote,
                "source": block_info(link.source_block_key, link.start_offset),
                "targets": [block_info(key) for key in link.candidate_keys if key in blocks],
            }
            for link in links
        ],
    }


async def ocr_review_response(session, document, review, page_number):
    try:
        current, fingerprint = await stored_snapshot(session, document, page_number)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "document_id": document.id,
        "filename": document.filename,
        "page_number": page_number,
        "parser_available": get_settings().pdf_parser == "docling",
        "can_start": get_settings().pdf_parser == "docling"
        and document.status == DocumentStatus.PROCESSED
        and document.parsing_report.get("parser") == "docling",
        "status": review.status if review else "not_started",
        "review_id": review.id if review else None,
        "created_at": review.created_at if review else None,
        "finished_at": review.finished_at if review else None,
        "expires_at": review.expires_at if review else None,
        "baseline_changed": bool(review and review.baseline_fingerprint != fingerprint),
        "document_status": document.status,
        "error_message": review.error_message if review else None,
        "result": review.result if review else {"baseline": current},
    }


@router.get("/documents/{document_id}/ocr-review")
async def get_page_ocr_review(
    document_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
    response: Response,
    page_number: int = Query(ge=1),
):
    document = await get_document(document_id, user, session)
    if not document.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="仅 PDF 支持原页 OCR 对照。")
    pages = document.parsing_report.get("source_page_count") or document.parsing_report.get(
        "page_count", 0
    )
    if pages and page_number > pages:
        raise HTTPException(status_code=422, detail="请求页码超出原文件范围。")
    await expire_reviews(session)
    await session.commit()
    review = await session.scalar(
        select(OcrReview)
        .where(
            OcrReview.document_id == document_id,
            OcrReview.page_number == page_number,
        )
        .order_by(OcrReview.created_at.desc())
        .limit(1)
    )
    response.headers["Cache-Control"] = "private, no-store"
    return await ocr_review_response(session, document, review, page_number)


@router.post("/documents/{document_id}/ocr-review", status_code=202)
async def start_page_ocr_review(
    document_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
    background_tasks: BackgroundTasks,
    response: Response,
    page_number: int = Query(ge=1),
):
    document = await get_document(document_id, user, session)
    settings = get_settings()
    if not document.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="仅 PDF 支持原页 OCR 对照。")
    if settings.pdf_parser != "docling":
        raise HTTPException(status_code=409, detail="请先启用 Docling 结构化解析服务。")
    if document.status != DocumentStatus.PROCESSED:
        raise HTTPException(status_code=409, detail="请等待文档处理成功后再进行 OCR 对照。")
    if document.parsing_report.get("parser") != "docling":
        raise HTTPException(
            status_code=409, detail="请先重新解析此 PDF，保存结构化来源后再进行对照。"
        )
    pages = document.parsing_report.get("source_page_count") or document.parsing_report.get(
        "page_count", 0
    )
    if page_number > pages:
        raise HTTPException(status_code=422, detail="请求页码超出原文件范围。")
    # Serialize admission across API workers, keeping at most one OCR comparison active.
    await session.execute(select(func.pg_advisory_xact_lock(740211)))
    await expire_reviews(session)
    active = await session.scalar(select(OcrReview).where(OcrReview.status == "running").limit(1))
    if active:
        if active.document_id == document_id and active.page_number == page_number:
            await session.commit()
            response.headers["Cache-Control"] = "private, no-store"
            return await ocr_review_response(session, document, active, page_number)
        raise HTTPException(status_code=409, detail="已有单页 OCR 对照正在运行，请完成后再试。")
    try:
        baseline, fingerprint = await stored_snapshot(session, document, page_number)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    review = OcrReview(
        document_id=document_id,
        page_number=page_number,
        baseline_fingerprint=fingerprint,
        result={"baseline": baseline},
        expires_at=utc_now() + timedelta(seconds=settings.docling_timeout_seconds + 30),
    )
    session.add(review)
    await session.commit()
    background_tasks.add_task(run_ocr_review, review.id)
    response.headers["Cache-Control"] = "private, no-store"
    return await ocr_review_response(session, document, review, page_number)


async def ensure_no_batch_work(session, document_id):
    active = await session.scalar(
        select(IngestionJob.id)
        .where(
            IngestionJob.document_id == document_id,
            IngestionJob.status.in_(ACTIVE_INGESTION_STATES),
        )
        .limit(1)
    )
    if active:
        raise HTTPException(409, "此文档仍有批量摄入任务，请在批任务中查看或重试。")


@router.post("/documents/{document_id}/reprocess", response_model=DocumentResponse, status_code=202)
async def reprocess_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    user: UserDep,
    session: SessionDep,
):
    document = await session.get(Document, document_id, with_for_update=True)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await owned_knowledge_base(session, document.knowledge_base_id, user.id)
    await ensure_no_batch_work(session, document.id)
    if document.status in {DocumentStatus.PENDING, DocumentStatus.PROCESSING}:
        raise HTTPException(status_code=409, detail="Document processing is already in progress")
    document.status = DocumentStatus.PENDING
    document.error_message = None
    document.graph_status = "not_requested"
    document.graph_error = None
    await session.commit()
    await session.refresh(document)
    background_tasks.add_task(process_document, document.id)
    return document


@router.post("/documents/{document_id}/rechunk", response_model=DocumentResponse, status_code=202)
async def rechunk_document(
    document_id: uuid.UUID, background_tasks: BackgroundTasks, user: UserDep, session: SessionDep
):
    document = await session.get(Document, document_id, with_for_update=True)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await owned_knowledge_base(session, document.knowledge_base_id, user.id)
    await ensure_no_batch_work(session, document.id)
    if document.status not in {DocumentStatus.PROCESSED, DocumentStatus.FAILED}:
        raise HTTPException(status_code=409, detail="请等待文档就绪后重新分块。")
    try:
        await asyncio.to_thread(
            load_artifact, document.storage_key, document.filename, document.sha256, get_settings()
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    document.status = DocumentStatus.PENDING
    document.error_message = None
    document.graph_status = "not_requested"
    document.graph_error = None
    await session.commit()
    await session.refresh(document)
    background_tasks.add_task(process_document, document.id, reuse_parse=True)
    return document


@router.post(
    "/documents/{document_id}/graph-index", response_model=DocumentResponse, status_code=202
)
async def build_document_graph(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    user: UserDep,
    session: SessionDep,
):
    document = await session.get(Document, document_id, with_for_update=True)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await owned_knowledge_base(session, document.knowledge_base_id, user.id)
    if document.status != DocumentStatus.PROCESSED:
        raise HTTPException(
            status_code=409, detail="Document must be processed before graph indexing"
        )
    if document.graph_status in {"pending", "processing"}:
        raise HTTPException(status_code=409, detail="Graph indexing is already in progress")
    document.graph_status = "pending"
    document.graph_error = None
    await session.commit()
    await session.refresh(document)
    background_tasks.add_task(index_document_graph, document.id)
    return document


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, user: UserDep, session: SessionDep):
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    await owned_knowledge_base(session, document.knowledge_base_id, user.id)
    file_path = Path(get_settings().upload_dir) / document.storage_key
    parsed_path = artifact_path(document.storage_key, get_settings())
    knowledge_base_id = document.knowledge_base_id
    await session.delete(document)
    await session.flush()
    await prune_orphan_entities(session, knowledge_base_id)
    await session.commit()
    await asyncio.to_thread(file_path.unlink, missing_ok=True)
    await asyncio.to_thread(parsed_path.unlink, missing_ok=True)


@router.get(
    "/knowledge-bases/{knowledge_base_id}/evaluations",
    response_model=list[EvaluationRunSummary],
)
async def list_evaluation_runs(
    knowledge_base_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    results_dir = Path(get_settings().evaluation_results_dir)
    if not results_dir.is_dir():
        return []
    runs: list[EvaluationRunSummary] = []
    seen = set()
    for path in sorted(
        results_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict):
                continue
            if report.get("knowledge_base_id") == str(knowledge_base_id):
                if report.get("run_id") in seen:
                    continue
                report = evaluation_runtime_state(report)
                runs.append(EvaluationRunSummary.model_validate(report))
                seen.add(report.get("run_id"))
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if len(runs) == 25:
            break
    return runs


def evaluation_runtime_state(report):
    if report.get("kind") == "rag_quality" and report.get("status") == "running":
        from knowflow.app.evaluate_quality import ACTIVE_RUNS

        if report.get("run_id") not in ACTIVE_RUNS:
            if report.get("execution_environment") == "cli":
                try:
                    age = (
                        datetime.now(UTC) - datetime.fromisoformat(report["heartbeat_at"])
                    ).total_seconds()
                    if 0 <= age < 120:
                        return report
                except (ValueError, TypeError, KeyError):
                    pass
            report = {
                **report,
                "status": "interrupted",
                "error": "评测执行进程已结束，保留已完成题目；尚未实现重启自动续作。",
            }
    return report


@router.post("/knowledge-bases/{knowledge_base_id}/quality-evaluations", status_code=202)
async def start_quality_evaluation(
    knowledge_base_id: uuid.UUID,
    body: QualityEvaluationRequest,
    background_tasks: BackgroundTasks,
    user: UserDep,
    session: SessionDep,
):
    from knowflow.app.evaluate_quality import (
        ACTIVE_KBS,
        ACTIVE_RUNS,
        DATASETS,
        PROJECT_ROOT,
        execute_quality,
        initial_report,
        save_report,
        validate_dataset,
    )

    await owned_knowledge_base(session, knowledge_base_id, user.id)
    if str(knowledge_base_id) in ACTIVE_KBS:
        raise HTTPException(status_code=409, detail="当前知识库已有质量验证正在执行。")
    try:
        dataset = validate_dataset(
            body.custom_dataset
            if body.dataset == "custom"
            else json.loads((PROJECT_ROOT / DATASETS[body.dataset]).read_text("utf-8"))
        )
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise HTTPException(
            status_code=422, detail="题集格式无效，请检查版本、问题、参考答案和标准证据。"
        ) from exc
    required = {g["filename"] for c in dataset["cases"] for g in c["relevant_evidence"]}
    ready = set(
        await session.scalars(
            select(Document.filename).where(
                Document.knowledge_base_id == knowledge_base_id, Document.status == "processed"
            )
        )
    )
    if missing := sorted(required - ready):
        raise HTTPException(status_code=422, detail="题集对应文档未就绪：" + "、".join(missing))
    strategies = list(dict.fromkeys(body.strategies))
    baseline = (
        await get_evaluation_run(knowledge_base_id, body.baseline_run_id, user, session)
        if body.baseline_run_id
        else None
    )
    report = initial_report(knowledge_base_id, dataset, strategies, body.top_k, body.judge)
    if baseline:
        report["baseline_run_id"] = baseline["run_id"]
    report["execution_environment"] = "api"
    save_report(report, get_settings().evaluation_results_dir)
    ACTIVE_RUNS.add(report["run_id"])
    ACTIVE_KBS.add(str(knowledge_base_id))
    background_tasks.add_task(
        execute_quality,
        knowledge_base_id,
        dataset,
        strategies,
        body.top_k,
        body.judge,
        report,
        baseline,
    )
    return report


@router.get("/knowledge-bases/{knowledge_base_id}/evaluations/{run_id}")
async def get_evaluation_run(
    knowledge_base_id: uuid.UUID,
    run_id: uuid.UUID,
    user: UserDep,
    session: SessionDep,
):
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    results_dir = Path(get_settings().evaluation_results_dir).resolve()
    result_path = (results_dir / f"{run_id}.json").resolve()
    if result_path.parent != results_dir or not result_path.is_file():
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    try:
        report = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=404, detail="Evaluation run not found") from exc
    if not isinstance(report, dict) or report.get("knowledge_base_id") != str(knowledge_base_id):
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return evaluation_runtime_state(report)


@router.post("/knowledge-bases/{knowledge_base_id}/query", response_model=QueryResponse)
async def query_knowledge_base(
    knowledge_base_id: uuid.UUID,
    body: QueryRequest,
    user: UserDep,
    session: SessionDep,
) -> QueryResponse:
    with scope(get_settings()) as budget:
        try:
            async with asyncio.timeout(budget.remaining()):
                result = await run_query(knowledge_base_id, body, user, session)
        except RoundBudgetExceeded:
            result = QueryResponse(
                answer="本轮累计调用或预留token预算不足，已停止后续请求；未交付未经核验的草稿。请缩小问题范围后重试。",
                refused=True,
                strategy=body.strategy,
                citations=[],
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail=("本次检索与依据核验超时，未返回未经核验的答案。请稍后重试或缩小问题范围。"),
            ) from exc
        result.round_budget = budget.snapshot()
        return result


async def run_query(knowledge_base_id, body, user, session, audit=None) -> QueryResponse:
    """The online pipeline, with optional internal capture for offline quality evaluation."""
    await owned_knowledge_base(session, knowledge_base_id, user.id)
    settings = get_settings()
    if body.strategy == "hybrid_rerank":
        from knowflow.app.reranker import require_rerank_configuration

        require_rerank_configuration(settings)
    intent_context = getattr(body, "_intent_context", None)
    from knowflow.app.document_reference import resolve_document_reference
    from knowflow.app.source_scope import bind_candidate_scope, retrieval_text

    source_scope = await resolve_document_reference(session, knowledge_base_id, body.question)
    bound_ids = getattr(body, "_source_document_ids", None)
    if bound_ids is None:
        # A true follow-up continues inside the file the last answer used, so the
        # search stays there instead of silently widening to the whole library.
        bound_ids = (
            tuple(getattr(body, "_context_document_ids", None) or ())
            if getattr(body, "_allow_context_document", False)
            else None
        )
    if bound_ids is not None and source_scope.trace["status"] not in {"unavailable", "ambiguous"}:
        source_scope = await bind_candidate_scope(
            session, knowledge_base_id, source_scope, bound_ids
        )
    if source_scope.trace["status"] in {"unavailable", "ambiguous"}:
        return QueryResponse(
            answer=(
                "指定的文件在当前知识库中不可用或尚未处理完成，请核对文件名与处理状态；本轮未扩大到其他文档搜索。"
                if source_scope.trace["status"] == "unavailable"
                else (
                    "指定文件存在同名、排除条件冲突或缺少要查的文件，请明确文件及版本；"
                    "本轮未扩大到全库搜索。"
                )
            ),
            refused=True,
            strategy=body.strategy,
            citations=[],
            query_plan={
                "status": "source_scope_blocked",
                "search_queries": [],
                "source_scope": source_scope.trace,
            },
        )
    answer_question = task_question(body.question, intent_context)
    retrieval_question = retrieval_text(body.question, source_scope)
    from knowflow.app.document_summary import (
        coverage_statement,
        read_document,
        requested,
        resolve_target,
        summary_settings,
    )

    chinese = any("\u4e00" <= c <= "\u9fff" for c in body.question)
    summary_mode = requested(
        body, [doc["filename"] for doc in source_scope.trace.get("documents", [])]
    )
    answer_settings = summary_settings(settings) if summary_mode else settings
    summary_trace = None
    summary_hits = None
    summary_evidence = None
    summary_coverage = None
    if summary_mode:
        target, summary_trace = await resolve_target(
            session,
            knowledge_base_id,
            source_scope.document_ids,
            getattr(body, "_context_document_ids", None)
            if getattr(body, "_allow_context_document", False)
            else None,
            question=body.question,
        )
        if target is None:
            reason = summary_trace.get("reason")
            hints = None
            if reason in {
                "multiple_documents_available",
                "conversation_has_multiple_documents",
                "multiple_documents_named",
            }:
                from knowflow.app.clarification_hints import document_hints

                hints = await document_hints(session, knowledge_base_id)
            named = summary_trace.get("named_object")
            catalog = summary_trace.get("catalog_size")
            probe = None
            if named and session is not None:
                try:
                    found = await retrieve(
                        session, knowledge_base_id, named, "keyword", 3, settings
                    )
                except Exception:
                    found = []
                probe = {
                    "query": named,
                    "searched": "keyword",
                    "hits": [
                        {
                            "document_id": str(hit.chunk.document_id),
                            "filename": hit.filename,
                            "chunk_id": str(hit.chunk.id),
                        }
                        for hit in found
                    ],
                }
                summary_trace["named_probe"] = probe
            if named and probe and probe["hits"]:
                files = "、".join(dict.fromkeys(row["filename"] for row in probe["hits"]))
                answer = (
                    f"我先核对了当前知识库的 {catalog} 份文件名，没有与「{named}」直接匹配的文档；"
                    f"又按「{named}」做了一次关键词检索，命中 {len(probe['hits'])} 个片段"
                    f"（来自：{files}）。请确认要总结哪一份文件；我不会把这些片段拼成全文总结。"
                    if chinese
                    else f"No filename in this knowledge base ({catalog} documents) matches "
                    f"'{named}'. A keyword search for it found {len(probe['hits'])} passages "
                    f"(from: {files}). Which document should I summarize? I will not splice "
                    "passages into a whole-document summary."
                )
            elif named:
                answer = (
                    f"我先核对了当前知识库的 {catalog} 份文件名，没有与「{named}」直接匹配的文档；"
                    f"按「{named}」做关键词检索也没有命中。请确认名称，或从下面选择要总结的文档。"
                    if chinese
                    else f"No filename in this knowledge base ({catalog} documents) matches "
                    f"'{named}', and a keyword search for it returned no passages. Confirm the "
                    "name or choose a document below."
                )
            else:
                answer = (
                    "请先指明要总结哪一份文档；我不会把全库检索到的零散片段当成某份文档的总结。"
                    if chinese
                    else "Which document should I summarize? I will not present retrieved "
                    "snippets as a summary of one document."
                )
            return QueryResponse(
                answer=answer,
                refused=True,
                strategy=body.strategy,
                citations=[],
                clarification=hints,
                document_summary=summary_trace,
                query_plan={
                    "status": "summary_document_required",
                    "search_queries": [named] if named else [],
                    "source_scope": source_scope.trace,
                },
            )
        answer_question += (
            "\nTask: summarize the selected document's main contents using ALL supplied "
            "parsed text, with citations. Organize its topics and key points, not search-result "
            "snippets. Cite inline after each point, never as a standalone list of labels. "
            "Keep the overview to at most 12 bullets and roughly 700 Chinese characters "
            "(350 English words). "
            "Do not claim unread pages/images or unsupported details were reviewed."
        )
    if getattr(body, "_response_style", None) in {"concise", "standard", "detailed"}:
        answer_question += (
            "\nUser-saved presentation preference (not evidence): response_style="
            + body._response_style
            + ". Apply only if compatible with the CURRENT request. Never omit required facts, "
            "conditions or citations to satisfy this preference."
        )
    routing_options = (
        {"routing_input": build_routing_input(answer_question, intent_context)}
        if intent_context
        else {}
    )
    _, answer_budget = plan_answer_context(answer_question, [], answer_settings, **routing_options)
    if not answer_budget["estimated_feasible"]:
        return QueryResponse(
            answer="原始问题与条件加上回答预留已超过本次运行预算，请缩小问题或重新概括；未截断原始条件。",
            refused=True,
            strategy=body.strategy,
            citations=[],
            context_budget=answer_budget,
        )
    if summary_mode:
        summary_hits, read_trace = await read_document(
            session, knowledge_base_id, uuid.UUID(summary_trace["document"]["id"])
        )
        summary_trace = {
            **read_trace,
            "document": summary_trace["document"],
            "selection_basis": summary_trace.get("selection_basis"),
        }
        if not summary_hits:
            return QueryResponse(
                answer=(
                    "这份文档目前没有可读取的解析文本，无法给出文档总结；"
                    "我没有用检索片段替代全文阅读。"
                    if chinese
                    else "This document has no readable parsed text, so no document summary "
                    "is given; retrieved snippets were not substituted for it."
                ),
                refused=True,
                strategy=body.strategy,
                citations=[],
                document_summary=summary_trace,
                query_plan={
                    "status": "summary_scope_blocked",
                    "search_queries": [],
                    "source_scope": source_scope.trace,
                },
            )
        rows = [{"id": str(hit.chunk.id), "content": hit.chunk.content} for hit in summary_hits]
        complete, preflight = plan_answer_context(
            answer_question, rows, answer_settings, **routing_options
        )
        if len(complete) == len(summary_hits):
            summary_trace.update(
                context_chunks=len(complete),
                loaded_chunks=len(summary_hits),
                read_chunks=0,
                segments=1,
                planned_segments=1,
                summarized_segments=0,
                coverage="parsed_text_prepared_not_generated",
                mode="whole_document",
                answer_window_tokens=answer_settings.rag_summary_context_token_budget,
            )
            summary_coverage = ""
        else:
            from knowflow.app.document_summary import segmented_summary

            mapped = await segmented_summary(
                answer_question,
                summary_hits,
                summary_trace,
                answer_settings,
                routing_options.get("routing_input"),
            )
            summary_trace.update(mapped["updates"], mode="segmented_map_reduce")
            if mapped["status"] != "ok":
                summary_trace.update(status="blocked", reason="segmented_summary_unavailable")
                return QueryResponse(
                    answer=(
                        "这份文档超过单轮上下文预算，分段汇总本轮未能完成；"
                        "我没有用少量片段冒充全文总结，请稍后重试或指定章节。"
                        if chinese
                        else "This document exceeds the single-round context budget and the "
                        "segmented summary did not complete; no snippet subset is presented as "
                        "a full summary. Retry later or name a section."
                    ),
                    refused=True,
                    strategy=body.strategy,
                    citations=[],
                    document_summary=summary_trace,
                    context_budget=preflight,
                )
            summary_trace.update(
                status="read",
                mode="segmented_map_reduce",
                coverage=(
                    "complete_parsed_text_segmented"
                    if summary_trace["summarized_segments"] == summary_trace["segments"]
                    else "partial_parsed_text_segmented"
                ),
            )
            summary_evidence = (mapped["evidence"], mapped["budget"])
            answer_question = mapped["question"]
            summary_coverage = coverage_statement(summary_trace, chinese)
        query_plan = {
            "status": "not_requested",
            "search_queries": [],
            "note": "文档总结直接读取选中文档解析文本，没有执行top-k检索。",
        }
    elif body.strategy in {"hybrid", "hybrid_rerank", "graph_hybrid"}:
        planning_options = (
            {"decompose": True, "intent_context": intent_context}
            if should_plan(body.question, intent_context, settings)
            else {}
        )
        query_plan = await plan_query(retrieval_question, settings, **planning_options)
    else:
        query_plan = {"status": "not_requested", "search_queries": []}
    query_plan["source_scope"] = source_scope.trace
    if retrieval_question != body.question:
        query_plan["retrieval_input"] = {
            "text": retrieval_question,
            "basis": "content_question_without_sql_filtered_file_labels",
            "original_user_input_preserved_for_answer_and_review": True,
        }
        query_plan["search_queries"] = list(
            dict.fromkeys(retrieval_text(q, source_scope) for q in query_plan["search_queries"])
        )
    if intent_context:
        query_plan["intent_context"] = intent_context
        query_plan["note"] = "改写仅用于召回；原始用户条件同时进入相关性、生成及依据复核。"
    from knowflow.app.conversation_context import aggregate_list_question

    coverage_mode = getattr(body, "_coverage", None)
    if coverage_mode not in {"point", "sweep"}:
        coverage_mode = "sweep" if aggregate_list_question(body.question) else "point"
    sweep_document_ids: tuple[uuid.UUID, ...] | None = None
    sweep_rows: list = []
    multiple_sources = len(source_scope.document_ids or ()) > 1
    if (coverage_mode == "sweep" or multiple_sources) and not summary_mode:
        from knowflow.app.coverage_scope import plan_coverage

        sweep_rows, query_plan["coverage"] = await plan_coverage(
            session, knowledge_base_id, source_scope
        )
        query_plan["coverage"]["requested_mode"] = coverage_mode
        sweep_document_ids = tuple(row.id for row in sweep_rows)
    retrieval_top_k = body.top_k
    if source_scope.document_ids and len(source_scope.document_ids) > 1:
        # A comparison over N named files needs room for each file's own evidence.
        base_k = body.top_k or settings.rag_top_k
        retrieval_top_k = min(20, max(base_k, base_k * len(source_scope.document_ids)))
    if summary_mode:
        retrieved = summary_hits
    elif sweep_document_ids is not None:
        retrieved = await retrieve_per_document(
            session,
            knowledge_base_id,
            retrieval_question,
            body.strategy,
            max(2, 2 * len(sweep_document_ids)) if coverage_mode == "sweep" else retrieval_top_k,
            settings,
            document_ids=sweep_document_ids,
            search_queries=query_plan["search_queries"] or None,
        )
    elif source_scope.document_ids is not None and source_scope.document_ids:
        # Every known scope also gets its own lexical pass, so a follow-up inside
        # one file is not limited to what the vector found in that file.
        retrieved = await retrieve_per_document(
            session,
            knowledge_base_id,
            retrieval_question,
            body.strategy,
            retrieval_top_k,
            settings,
            document_ids=source_scope.document_ids,
            search_queries=query_plan["search_queries"] or None,
        )
    else:
        retrieved = await retrieve(
            session,
            knowledge_base_id,
            retrieval_question,
            body.strategy,
            retrieval_top_k,
            settings,
            **(
                {"search_queries": query_plan["search_queries"]}
                if query_plan["search_queries"]
                else {}
            ),
        )
    if sweep_document_ids:
        checked_rows = sweep_rows
        per_document: dict[str, int] = {}
        sampled_documents: set[str] = set()
        for hit in retrieved:
            key = str(hit.chunk.document_id)
            per_document[key] = per_document.get(key, 0) + 1
            if hit.distance_gate_fallback:
                sampled_documents.add(key)
        query_plan["coverage"].update(
            {
                "hit": sum(1 for row in checked_rows if per_document.get(str(row.id))),
                "documents": [
                    {
                        "document_id": str(row.id),
                        "filename": row.filename,
                        "evidence_chunks": per_document.get(str(row.id), 0),
                        "distance_gate_fallback": str(row.id) in sampled_documents,
                    }
                    for row in checked_rows
                ],
            }
        )
    if sweep_document_ids:
        coverage = query_plan["coverage"]
        answer_question = (
            answer_question
            + f"\n本轮仅抽样 {coverage['checked']}/{coverage['eligible_documents']} 份范围内文件，"
            + f"另有 {coverage['omitted_documents']} 份未检查；每份只读取少量候选片段，不是全文。"
            + "\n\n请按文件分条回答：每条以“**该文件的文件名**：”开头（文件名取自片段标注，"
            "不要用序号代替），每份写 1–3 条要点并附上该文件片段的来源标记；"
            "只写该文件原文明确支持的内容，不要跨文件推测。"
            "若问的是“哪些文件没有/没做 X”，先列出片段里明确提到 X 的文件及其原文内容，"
            "其余文件写“本轮片段未提及 X，不能据此断定没有”，不要列出空文件清单之外的推断。"
        )
    rerank_trace = next((h.rerank_trace for h in retrieved if h.rerank_trace), None)
    if query_plan.get("task_plan"):
        query_plan["task_plan"]["retrieval_status"] = "shared_retriever_returned"
        query_plan["task_plan"]["shared_candidate_count"] = len(retrieved)
    graph_expansion = (
        graph_candidate_diagnostics(retrieved, body.top_k or settings.rag_top_k)
        if body.strategy == "graph_hybrid"
        else None
    )

    def capture(hits):
        return [
            {
                "chunk_id": str(hit.chunk.id),
                "filename": hit.filename,
                "content": hit.chunk.content,
                "page_number": hit.chunk.page_number,
                "graph_rank": hit.graph_rank,
                "english_keyword_rank": hit.english_keyword_rank,
                "source_spans": hit.chunk.source_spans,
                "retrieval_stage": hit.retrieval_stage,
                "distance_gate_fallback": hit.distance_gate_fallback,
            }
            for hit in hits
        ]

    if audit is not None:
        audit.update(
            base=capture([hit for hit in retrieved if hit.retrieval_stage == "base"]),
            graph_expansion=graph_expansion,
            query_plan=query_plan,
            context_budget={**answer_budget, "status": "pending_evidence"},
            context=[],
            rerank=rerank_trace,
        )
    retrieved, reference_diagnostics = await expand_references(
        session,
        knowledge_base_id,
        body.question,
        retrieved,
        enabled=(
            not summary_mode and body.expand_references and settings.reference_expansion_enabled
        ),
    )
    if audit is not None:
        audit["candidates"] = capture(retrieved)
    if not retrieved:
        _, diagnostics = await assess_retrieval(answer_question, [], settings)
        empty_answer = (
            unstated_in_scope(source_scope, body.question)
            or "我在这个知识库中没有找到足够相关的文档证据，暂时无法可靠回答。"
        )
        if sweep_document_ids:
            # A sweep answer that saw nothing must name the files it checked and
            # admit the sampling gap, instead of a generic relevance refusal.
            names = "、".join(row.filename for row in checked_rows)
            empty_answer = (
                f"本轮跨文档抽样 {query_plan['coverage']['checked']}/"
                f"{query_plan['coverage']['eligible_documents']} 份文件，取回 0 条原文片段"
                f"（文件：{names}），因此无法给出有依据的回答；"
                "这不代表这些文件没有相关内容，可稍后重试或缩小到单个文件提问。"
                if names
                else empty_answer
            )
        return QueryResponse(
            answer=empty_answer,
            refused=True,
            strategy=body.strategy,
            citations=[],
            retrieval_diagnostics=diagnostics,
            reference_expansion=reference_diagnostics,
            graph_expansion=graph_expansion,
            query_plan=query_plan,
            context_budget={**answer_budget, "status": "no_candidates"},
        )

    if summary_mode:
        from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics

        assessments = {
            str(hit.chunk.id): RelevanceAssessment(
                reason="总结按选中文档读取，未做片段相关性筛选；仍需答案依据核验。"
            )
            for hit in retrieved
        }
        diagnostics = RetrievalDiagnostics(
            status="disabled",
            note="文档总结读取全部可容纳的解析文字，未运行片段相关性筛选；答案仍按原文审核。",
        )
    elif sweep_document_ids:
        from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics

        # Sweep mode already bounds evidence per document; per-chunk scoring is
        # replaced by per-file coverage plus the normal answer guard.
        assessments = {
            str(hit.chunk.id): RelevanceAssessment(
                verdict="partial",
                reason="跨文档扫描：每份文档按配额取样，答案仍逐句核验。",
            )
            for hit in retrieved
        }
        diagnostics = RetrievalDiagnostics(
            status="disabled",
            note="跨文档扫描：每份文档各取少量原文，未逐条打分；覆盖范围见 query_plan.coverage。",
        )
    else:
        assessments, diagnostics = await assess_retrieval(
            answer_question,
            [
                {"id": str(hit.chunk.id), "filename": hit.filename, "content": hit.chunk.content}
                for hit in retrieved
            ],
            settings,
        )
    diagnostics.rerank = rerank_trace
    text_added = sum(hit.retrieval_stage == "text_expansion" for hit in retrieved)
    if text_added:
        diagnostics.note += (
            f" 英文词干通道额外补充 {text_added} 条文本候选（上限 k 条），"
            "未替换基础前 k 条；扩展命中不计入基础 Recall@k，仍须通过相关性检查。"
        )
    if any(hit.distance_gate_fallback for hit in retrieved):
        diagnostics.note += (
            " 本次常规召回为空，少量最近邻候选超过距离阈值，已交由相关性检查；"
            "这不是高置信命中，也不保证可回答。"
        )
    source_documents = await session.scalars(
        select(Document).where(Document.id.in_({hit.chunk.document_id for hit in retrieved}))
    )
    for source in source_documents:
        report = source.parsing_report or {}
        if report.get("warnings"):
            diagnostics.parsing_warnings.extend(
                f"{source.filename}：{warning}" for warning in report["warnings"]
            )
        elif source.filename.lower().endswith(".pdf") and not report:
            diagnostics.parsing_warnings.append(f"{source.filename}：旧索引尚无解析质量报告。")
    located_spans = {
        hit.chunk.id: quote_spans(hit.chunk, assessments[str(hit.chunk.id)].supporting_quotes)
        for hit in retrieved
    }
    trace_hits = retrieved
    if summary_evidence is not None:
        # Only the chunks that were actually supplied to the composition step are
        # evidence rows; the rest of the document was read by the map stage only.
        supplied_ids = {row["id"] for row in summary_evidence[0]}
        trace_hits = [hit for hit in retrieved if str(hit.chunk.id) in supplied_ids]
    trace = [
        RetrievedEvidence(
            chunk_id=hit.chunk.id,
            document_id=hit.chunk.document_id,
            filename=hit.filename,
            page_number=first_page(located_spans[hit.chunk.id], hit.chunk.page_number),
            section=hit.chunk.section,
            quote="\n…\n".join(assessments[str(hit.chunk.id)].supporting_quotes)
            or hit.chunk.content[:400],
            cosine_distance=hit.cosine_distance,
            cosine_similarity=1 - hit.cosine_distance if hit.cosine_distance is not None else None,
            vector_rank=hit.vector_rank,
            keyword_rank=hit.keyword_rank,
            fused_score=hit.fused_score,
            graph_rank=hit.graph_rank,
            table_rank=hit.table_rank,
            english_keyword_rank=hit.english_keyword_rank,
            rerank_rank=hit.rerank_rank,
            rerank_score=hit.rerank_score,
            rank=rank,
            assessment=assessments[str(hit.chunk.id)],
            retrieval_stage=hit.retrieval_stage,
            distance_gate_fallback=hit.distance_gate_fallback,
            source_spans=located_spans[hit.chunk.id],
            reference_paths=hit.reference_paths,
        )
        for rank, hit in enumerate(trace_hits, start=1)
    ]
    accepted = [
        hit
        for hit in retrieved
        if summary_mode
        or not settings.retrieval_assessment_enabled
        or assessments[str(hit.chunk.id)].verdict in ACCEPTED_VERDICTS
    ]
    response_trace = {
        "document_summary": summary_trace,
        "retrieval_evidence": trace,
        "retrieval_diagnostics": diagnostics,
        "reference_expansion": reference_diagnostics,
        "graph_expansion": graph_expansion,
        "query_plan": query_plan,
        "context_budget": {**answer_budget, "status": "no_accepted_evidence"},
    }
    if not accepted:
        return QueryResponse(
            answer=(
                "找到了一些候选资料，但暂时无法确认它们与这个问题的对应关系，"
                "先不给出可能有误的回答。可以换个问法或补充对象后再试。"
                if diagnostics.status in {"unavailable", "partial"}
                else unstated_in_scope(source_scope, body.question)
                or "暂时没有找到足以支撑回答的资料。可以换个问法、补充对象或范围后再试；"
                "这不代表文档里一定没有相关内容。"
            ),
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )
    if summary_evidence is not None:
        # Segmented summary: evidence is the per-part notes that were actually read,
        # not a fresh top-k selection of the document.
        evidence, answer_budget = summary_evidence
    else:
        source_rows = [
            {
                "id": str(hit.chunk.id),
                "content": hit.chunk.content,
                "filename": hit.filename,
            }
            for hit in accepted
        ]
        supported = source_rows
        if settings.retrieval_assessment_enabled and not summary_mode:
            supported = retain_supported_context(source_rows, assessments)
        evidence, answer_budget = plan_answer_context(
            answer_question, supported, answer_settings, **routing_options
        )
        supported_ids = {r["id"] for r in supported}
        for row in source_rows:
            if row["id"] not in supported_ids:
                answer_budget["decisions"].append(
                    {
                        "id": row["id"],
                        "decision": "skipped",
                        "reason": "support_quote_missing",
                    }
                )
                answer_budget["skipped_count"] += 1
        if not supported:
            answer_budget["status"] = "no_supported_source"
    response_trace["context_budget"] = answer_budget
    if audit is not None:
        audit["context_budget"] = answer_budget
        metadata = {str(hit.chunk.id): hit.filename for hit in retrieved}
        audit["context"] = [
            {"chunk_id": row["id"], "content": row["content"], "filename": metadata[row["id"]]}
            for row in evidence
        ]
    allowed_citation_ids = {uuid.UUID(row["id"]) for row in evidence}
    for row in trace:
        row.included_in_context = row.chunk_id in allowed_citation_ids
    if query_plan.get("coverage"):
        for document in query_plan["coverage"]["documents"]:
            document["context_chunks"] = sum(
                row.included_in_context and str(row.document_id) == document["document_id"]
                for row in trace
            )
    if not evidence:
        return QueryResponse(
            answer=(
                "候选片段的支持引文未能在完整来源中核对，尚未向模型提供证据。"
                if not supported
                else "找到了相关片段，但本次上下文预算不足，尚未向模型提供证据。"
            ),
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )
    response_trace["generation_route"] = {
        **answer_budget["model_route"],
        "prompt_version": ANSWER_PROMPT_VERSION,
        "input_basis": answer_budget["routing_input_basis"],
        "routing_input_sha256": answer_budget["routing_input_sha256"],
        "routing_input_chars": answer_budget["routing_input_chars"],
    }
    try:
        answer = await generate_answer(
            answer_question,
            evidence,
            answer_settings,
            budget_trace=answer_budget,
            **routing_options,
        )
    except (AnswerInputBudgetError, AnswerOutputLimit) as exc:
        answer_budget["status"] = (
            "output_limit_reached" if isinstance(exc, AnswerOutputLimit) else "blocked"
        )
        return QueryResponse(
            answer="本次生成达到输出或上下文预算限制，未交付截断答案。请缩小问题范围后重试。",
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )
    if audit is not None:
        audit["draft_answer"] = answer
        audit["generation_route"] = response_trace["generation_route"]
    if summary_mode:
        from knowflow.app.document_summary import keep_cited_blocks

        if summary_trace.get("mode") == "whole_document":
            summary_trace.update(read_chunks=len(summary_hits), coverage="complete_parsed_text")
        answer, summary_check = keep_cited_blocks(answer, {row["id"] for row in evidence})
        if summary_trace.get("mode") == "whole_document":
            summary_trace["summarized_segments"] = int(bool(answer))
        validation = {
            "version": "document-summary-citation-blocks-v1",
            "status": "citation_checked" if answer else "rejected",
            **summary_check,
            "note": (
                "文档总结按整篇解析文本生成；这里只校验每个保留段落是否带有已提供的原文引用，"
                "不做逐句模型复核。"
            ),
        }
    else:
        answer, validation = await guard_answer(
            answer_question,
            answer,
            evidence,
            answer_settings,
            **(
                {"current_user_input": intent_context["current_user_input"]}
                if intent_context
                else {"current_user_input": body.question}
                if getattr(body, "_response_style", None)
                else {}
            ),
        )
    response_trace["answer_validation"] = validation
    if not answer:
        return QueryResponse(
            answer="已找到相关资料，但生成内容未通过依据复核，暂时不能提供可靠回答。请查看下方来源及核验状态。",
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )
    try:
        answer = normalize_source_markers(answer)
    except ValueError:
        return QueryResponse(
            answer="回答的来源标记不完整，暂时无法逐项核对，未采用该回答。",
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )
    citations: list[Citation] = []
    cited_ids: set[uuid.UUID] = set()
    valid_citation_markers: dict[str, str] = {}
    invalid_citation = False
    for match in CHUNK_ID_PATTERN.finditer(answer):
        try:
            chunk_id = uuid.UUID(match.group(1))
        except ValueError:
            invalid_citation = True
            continue
        hit = next((item for item in retrieved if item.chunk.id == chunk_id), None)
        if hit is None or chunk_id not in allowed_citation_ids:
            invalid_citation = True
            continue
        if chunk_id in cited_ids:
            continue
        chunk = hit.chunk
        trace_row = next(row for row in trace if row.chunk_id == chunk.id)
        citations.append(
            Citation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=hit.filename,
                page_number=trace_row.page_number,
                section=chunk.section,
                quote=trace_row.quote,
                context_excerpt=(
                    context
                    if (context := extended_quote(trace_row.quote, chunk.content))
                    != trace_row.quote
                    else None
                ),
                cosine_distance=hit.cosine_distance,
                vector_rank=hit.vector_rank,
                keyword_rank=hit.keyword_rank,
                fused_score=hit.fused_score,
                graph_rank=hit.graph_rank,
                table_rank=hit.table_rank,
                english_keyword_rank=hit.english_keyword_rank,
                rerank_rank=hit.rerank_rank,
                rerank_score=hit.rerank_score,
                source_spans=trace_row.source_spans,
                reference_paths=hit.reference_paths,
            )
        )
        cited_ids.add(chunk_id)
        valid_citation_markers[str(chunk_id)] = f"[{len(citations)}]"
    if not citations:
        return QueryResponse(
            answer="生成的回答没有提供可验证的文档引用，因此我不能把它作为有依据的答案返回。",
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )

    if invalid_citation:
        return QueryResponse(
            answer="回答包含未提供给模型的来源引用，未通过引用校验，暂不采用。",
            refused=True,
            strategy=body.strategy,
            citations=[],
            **response_trace,
        )
    for row in trace:
        row.cited_in_answer = row.chunk_id in cited_ids
    if sweep_document_ids and query_plan.get("coverage"):
        # Coverage is what the answer actually stands behind, not what was fetched:
        # a file can be sampled and still never be cited by the guarded answer.
        cited_documents = {
            str(citation.document_id) for citation in citations if citation.document_id is not None
        }
        for row in query_plan["coverage"]["documents"]:
            row["cited_in_answer"] = False
        coverage_documents = {
            row["document_id"]: row for row in query_plan["coverage"]["documents"]
        }
        for row in checked_rows:
            if str(row.id) in cited_documents:
                coverage_documents[str(row.id)]["cited_in_answer"] = True
        query_plan["coverage"]["answered"] = sum(
            1 for row in query_plan["coverage"]["documents"] if row["cited_in_answer"]
        )
        # Legacy answered counts source citations, never semantic task coverage.
        query_plan["coverage"]["cited_documents"] = query_plan["coverage"]["answered"]
        unanswered = [
            row["filename"]
            for row in query_plan["coverage"]["documents"]
            if not row["cited_in_answer"]
        ]
        if unanswered:
            answer = (
                answer.rstrip()
                + (
                    "\n\n**本轮未涉及的文件（答案未引用，不能判断文件是否包含相关内容）**："
                    if chinese
                    else "\n\n**Files not cited in this answer (content availability unknown)**: "
                )
                + "、".join(unanswered)
            )
        coverage = query_plan["coverage"]
        answer += (
            f"\n\n本轮在限定范围内抽样 {coverage['checked']}/"
            f"{coverage['eligible_documents']} 份文件，每份仅读取少量候选片段；"
            f"{coverage['omitted_documents']} 份未检查。不能据此作全文缺席或全库唯一结论。"
            if chinese
            else f"\n\nSampled {coverage['checked']}/{coverage['eligible_documents']} files in the "
            f"requested scope; {coverage['omitted_documents']} files were not checked. "
            "Only candidate excerpts were read, so this does not establish absence or uniqueness."
        )

    def replace_citation(match: re.Match[str]) -> str:
        try:
            citation_id = str(uuid.UUID(match.group(1)))
        except ValueError:
            return ""
        return valid_citation_markers.get(citation_id, "")

    answer = CHUNK_ID_PATTERN.sub(replace_citation, answer)
    if summary_mode:
        summary_coverage = coverage_statement(summary_trace, chinese)
    if summary_coverage:
        answer = f"{answer}\n\n---\n{summary_coverage}"
    return QueryResponse(
        answer=answer,
        refused=False,
        strategy=body.strategy,
        citations=citations,
        **response_trace,
    )
