from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from uuid import UUID

import jieba
from sqlalchemy import delete

from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.document_links import rebuild_document_links
from knowflow.app.document_processing import ParsedSection
from knowflow.app.graph_index import prune_orphan_entities
from knowflow.app.index_text import chunk_index_text
from knowflow.app.ingestion_jobs import LeaseLost
from knowflow.app.models import Chunk, Document, DocumentBlock, DocumentStatus
from knowflow.app.parse_artifacts import load_artifact, save_artifact
from knowflow.app.providers import embed_texts
from knowflow.app.structured_parsing import (
    ParsingFailure,
    chunk_parsed_document,
    parse_for_ingestion,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def searchable_text(text: str) -> str:
    """Segment Chinese text before storing it for PostgreSQL's simple FTS parser."""
    return " ".join(token.strip() for token in jieba.cut_for_search(text) if token.strip())


async def process_document(document_id: UUID, reuse_parse: bool = False, *, control=None) -> None:
    async with SessionLocal() as session:
        if control:
            await control.locked(session)
        document = await session.get(Document, document_id, with_for_update=bool(control))
        if document is None:
            if control:
                raise ValueError("原文档已删除。")
            return
        document.status = DocumentStatus.PROCESSING
        document.error_message = None
        await session.commit()

        parsing_report = None
        try:
            if control:
                await control.checkpoint("reading")
            settings = get_settings()
            path = Path(settings.upload_dir) / document.storage_key
            data = await asyncio.to_thread(path.read_bytes)
            if sha256_bytes(data) != document.sha256:
                raise ValueError("原始文件校验失败，未替换已有索引。")
            if control:
                await control.checkpoint("parsing")
            parsed = None
            if reuse_parse or (control and control.attempt_number > 1):
                try:
                    parsed = await asyncio.to_thread(
                        load_artifact,
                        document.storage_key,
                        document.filename,
                        document.sha256,
                        settings,
                    )
                except ValueError:
                    if reuse_parse:
                        raise
            cache_reused = parsed is not None
            if parsed is None:
                parsed = await parse_for_ingestion(document.filename, data, settings)
                try:
                    parsed.report["parse_artifact"] = await asyncio.to_thread(
                        save_artifact,
                        document.storage_key,
                        document.filename,
                        document.sha256,
                        parsed,
                        settings,
                    )
                except (OSError, ValueError):
                    parsed.report["parse_artifact"] = {"available": False}
                    parsed.report.setdefault("warnings", []).append(
                        "解析产物未缓存，后续重新分块需要重新解析。"
                    )
            parsing_report = parsed.report
            if control:
                await control.checkpoint("chunking", cache_reused=cache_reused)
            chunks = await asyncio.to_thread(chunk_parsed_document, parsed, settings)
            if not chunks:
                raise ValueError("No searchable text was extracted from this document")
            if control:
                await control.checkpoint("embedding")
            vectors = await embed_texts(
                [chunk_index_text(chunk.section, chunk.content) for chunk in chunks], settings
            )

            if control:
                await control.checkpoint("publishing")
                # Keep this fence locked through both index publication and job completion.
                await control.locked(session)
                document = await session.get(
                    Document, document_id, with_for_update=True, populate_existing=True
                )
                if document is None:
                    raise ValueError("原文档已删除。")

            # Replace this document's derived index in one transaction. A provider
            # error before this point leaves the previous valid index untouched.
            await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
            await session.execute(
                delete(DocumentBlock).where(DocumentBlock.document_id == document_id)
            )
            await prune_orphan_entities(session, document.knowledge_base_id)
            session.add_all(
                [
                    DocumentBlock(
                        document_id=document.id,
                        ordinal=ordinal,
                        block_key=block.block_key,
                        kind=block.kind,
                        text=block.text,
                        section=block.section,
                        parent_key=block.parent_key,
                        source_spans=block.source_spans,
                        structure=block.structure,
                    )
                    for ordinal, block in enumerate(parsed.blocks)
                ]
            )
            session.add_all(
                [
                    Chunk(
                        document_id=document.id,
                        knowledge_base_id=document.knowledge_base_id,
                        ordinal=chunk.ordinal,
                        content=chunk.content,
                        search_text=searchable_text(chunk_index_text(chunk.section, chunk.content)),
                        embedding=vector,
                        page_number=chunk.page_number,
                        section=chunk.section,
                        embedding_model=settings.embedding_model,
                        source_spans=chunk.source_spans,
                        processing_metadata=chunk.processing_metadata,
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
            document.status = DocumentStatus.PROCESSED
            document.error_message = None
            document.parsing_report = parsed.report
            await session.flush()
            await rebuild_document_links(session, document)
            if control:
                await control.complete(session)
            await session.commit()
        except LeaseLost:
            await session.rollback()
            raise
        except Exception as exc:
            await session.rollback()
            if control:
                report = (
                    exc.report
                    if isinstance(exc, ParsingFailure)
                    else ({**parsing_report, "indexing_failed": True} if parsing_report else None)
                )
                await control.fail(session, exc, report)
                await session.commit()
                return
            failed_document = await session.get(Document, document_id)
            if failed_document is not None:
                failed_document.status = DocumentStatus.FAILED
                failed_document.error_message = str(exc)[:2000]
                if isinstance(exc, ParsingFailure):
                    failed_document.parsing_report = exc.report
                elif parsing_report is not None:
                    failed_document.parsing_report = {**parsing_report, "indexing_failed": True}
                await session.commit()


def text_sections(text: str) -> list[ParsedSection]:
    """Utility used by tests and future import sources that provide plain text."""
    return [ParsedSection(text)]
