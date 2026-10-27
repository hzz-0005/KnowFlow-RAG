"""Read parsed text; never call a top-k subset a full document summary.

Two execution shapes share one coverage contract:
* whole parsed text fits the answer window -> one grounded generation;
* it does not -> contiguous segments are summarized separately, then composed,
  and every answer states exactly which chunks/pages were read.
"""

import re

from sqlalchemy import func, select
from sqlalchemy.orm import defer

from knowflow.app.models import Chunk, Document, DocumentStatus
from knowflow.app.retrieval import RetrievalHit

VERSION = "selected-document-summary-v4-observed-coverage"
SUMMARY_CUE = re.compile(
    r"概括|总结|综述|主要.{0,8}(?:讲|内容|说|观点|结论)|(?:讲|说).{0,5}什么|"
    r"内容.{0,5}什么|大意|summari[sz]e|\bsummary\b|\boverview\b|\bwhat is this .{0,20}about\b",
    re.I,
)
# "介绍一下 X" asks for the same whole-file overview as "总结一下 X".
OVERVIEW_CUE = re.compile(
    r"介绍|简介|概述|概括|总结|综述|讲(?:一)?下|说说|聊聊|主要.{0,8}(?:讲|内容|说|观点|结论)|"
    r"(?:讲|说).{0,5}什么|大意|summari[sz]e|describe|introduce|\bsummary\b|\boverview\b",
    re.I,
)
# A summary request must point at a document object; a bare "summarize the policy"
# stays a normal retrieval question over the knowledge base.
DOCUMENT_REF = re.compile(
    r"这份|这篇|该文档|这个文档|本文档|这封|这份材料|文档|文件|论文|报告|材料|全文|正文|"
    r"\.(?:pdf|docx?|md|txt|xlsx?|csv|pptx?)\b|"
    r"\b(?:this|the|that)\s+(?:document|file|paper|report|pdf)\b|"
    r"\b(?:document|file|paper|report)\s+(?:summary|overview)\b",
    re.I,
)
NEGATIVE = re.compile(r"不要(?:总结|概括)|别(?:总结|概括)|do not summarize|don't summarize", re.I)
MAX_READ_CHUNKS = 4000


def summary_settings(settings):
    """The document-summary path gets its own reading window.

    Retrieval answers stay inside the small evidence budget; a whole-document
    summary is generated from all parsed chunks in one call when they fit.
    """
    return settings.model_copy(
        update={
            "rag_context_token_budget": settings.rag_summary_context_token_budget,
            "rag_answer_total_token_budget": settings.rag_summary_total_token_budget,
            "rag_answer_output_token_budget": settings.rag_summary_output_token_budget,
            "answer_guard_max_chars": max(settings.answer_guard_max_chars, 24000),
        }
    )


def _trace(**overrides):
    trace = {
        "version": VERSION,
        "status": "blocked",
        "coverage": "not_read",
        "document": None,
        "selection_basis": None,
        "total_chunks": None,
        "read_chunks": 0,
        "context_chunks": 0,
        "segments": 0,
        "summarized_segments": 0,
        "pages": None,
        "note": (
            "只覆盖当前解析出的文字；未提取的页面、图片和图形关系不算已读，"
            "不把 top-k 片段冒充完整总结。"
        ),
    }
    trace.update(overrides)
    return trace


def document_cue(question):
    """Lexical guard shared by the router and the summary entry.

    A summary request must point at a document object; a bare "summarize the
    policy" stays a normal retrieval question over the knowledge base.
    """
    return bool(SUMMARY_CUE.search(question) and DOCUMENT_REF.search(question))


def requested(body, filenames=()):
    question = body.question
    for filename in sorted(set(filenames), key=len, reverse=True):
        question = re.sub(re.escape(filename), "这份文档", question, flags=re.I)
    kind = getattr(body, "_request_kind", None)
    if NEGATIVE.search(question):
        return False
    if kind == "document_summary":
        return True
    scope = tuple(getattr(body, "_source_document_ids", None) or ())
    # An overview request that resolves to exactly one file is a server-side,
    # list-based decision: read that whole file instead of a guessed top-k. A
    # request spanning several files stays a normal comparison answer.
    if len(scope) == 1 and OVERVIEW_CUE.search(question):
        return True
    if kind is not None:
        return False
    return document_cue(question)


LATIN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9._-]{3,}")
NAME_STOP = {
    "document",
    "documents",
    "summary",
    "project",
    "projects",
    "readme",
    "file",
    "files",
    "paper",
    "report",
    "overview",
    "what",
    "which",
    "how",
    "why",
    "the",
    "this",
    "that",
    "these",
    "those",
    "about",
    "please",
    "show",
    "explain",
    "describe",
    "give",
    "tell",
}


def named_object(question, filenames):
    """A Latin identifier the user typed that no current filename covers.

    This is a *catalog check result*, not proof that the project does not exist.
    """
    covered = " ".join(filenames).casefold()
    for token in sorted(set(LATIN_NAME.findall(question)), key=len, reverse=True):
        name = token.strip("._-")
        low = name.casefold()
        if len(name) < 4 or low in NAME_STOP or low in covered:
            continue
        return name
    return None


async def resolve_target(session, kb_id, explicit_ids, context_ids=None, question=None):
    """Pick exactly one document or say why that is impossible."""
    if explicit_ids:
        ids = list(dict.fromkeys(explicit_ids))
        if len(ids) > 1:
            return None, _trace(
                reason="multiple_documents_named", selection_basis="explicit_filename"
            )
        return await _load(session, kb_id, ids[0], "explicit_filename")
    if context_ids:
        ids = list(dict.fromkeys(context_ids))
        if len(ids) == 1:
            return await _load(session, kb_id, ids[0], "conversation_sources")
        if len(ids) > 1:
            return None, _trace(
                reason="conversation_has_multiple_documents", selection_basis="conversation_sources"
            )
    filters = (
        Document.knowledge_base_id == kb_id,
        Document.status == DocumentStatus.PROCESSED,
    )
    total = await session.scalar(select(func.count(Document.id)).where(*filters))
    rows = list(
        await session.execute(
            select(Document.id, Document.filename)
            .where(*filters)
            .order_by(Document.filename, Document.id)
            .limit(2)
        )
    )
    if total == 1 and len(rows) == 1:
        return await _load(session, kb_id, rows[0].id, "only_processed_document")
    return None, _trace(
        reason="no_processed_document" if not total else "multiple_documents_available",
        selection_basis="knowledge_base_scan",
        catalog_size=total or 0,
        named_object=named_object(question, [row.filename for row in rows]) if question else None,
    )


async def _load(session, kb_id, document_id, basis):
    row = (
        await session.execute(
            select(Document.id, Document.filename).where(
                Document.id == document_id,
                Document.knowledge_base_id == kb_id,
                Document.status == DocumentStatus.PROCESSED,
            )
        )
    ).first()
    if row is None:
        return None, _trace(reason="selected_document_unavailable", selection_basis=basis)
    document = {"id": str(row.id), "filename": row.filename}
    return document, _trace(
        status="selected",
        coverage="selected_not_read",
        document=document,
        selection_basis=basis,
    )


async def read_document(session, kb_id, document_id):
    where = (
        Document.knowledge_base_id == kb_id,
        Document.id == document_id,
        Document.status == DocumentStatus.PROCESSED,
        Chunk.knowledge_base_id == kb_id,
    )
    total = await session.scalar(select(func.count(Chunk.id)).join(Document).where(*where))
    if not total:
        return [], _trace(reason="document_has_no_parsed_text", total_chunks=0)
    if total > MAX_READ_CHUNKS:
        return [], _trace(reason="document_exceeds_read_limit", total_chunks=total)
    rows = (
        await session.execute(
            select(Chunk, Document.filename)
            .join(Document)
            .where(*where)
            .options(defer(Chunk.embedding))
            .order_by(Chunk.ordinal)
        )
    ).all()
    hits = [
        RetrievalHit(chunk, filename, retrieval_stage="document_read") for chunk, filename in rows
    ]
    pages = [hit.chunk.page_number for hit in hits if hit.chunk.page_number]
    return hits, _trace(
        status="read",
        coverage="parsed_text_read",
        total_chunks=total,
        read_chunks=len(hits),
        pages=[min(pages), max(pages)] if pages else None,
    )


def plan_segments(hits, counter, token_budget):
    """Contiguous, order-preserving segments that each fit the evidence window."""
    segments, current, used, section = [], [], 0, None
    for hit in hits:
        cost = counter.count(hit.chunk.content) + 12
        boundary = (
            section is not None and hit.chunk.section != section and used >= token_budget // 2
        )
        if current and (used + cost > token_budget or boundary):
            segments.append(current)
            current, used = [], 0
        current.append(hit)
        used += cost
        section = hit.chunk.section
    if current:
        segments.append(current)
    return segments


def pages_of(hits):
    pages = [hit.chunk.page_number for hit in hits if hit.chunk.page_number]
    return [min(pages), max(pages)] if pages else None


MARKER = re.compile(r"\[([0-9a-fA-F][0-9a-fA-F-]{22,42})\]")


def _segment_question(answer_question, filename, index, total):
    return (
        f"Summarize part {index} of {total} of the document '{filename}' using ONLY this part.\n"
        "Write 3-5 short bullets (at most about 120 words) in the user's language and cite "
        "each bullet with the supplied source labels. Do not describe other parts, do not "
        "claim the whole document was read, and never add facts from outside this part.\n"
        f"Original user request:\n{answer_question}"
    )


def _reduce_question(answer_question, filename, summarized, total):
    return (
        f"Compose one overview of the document '{filename}' from the per-part notes supplied "
        f"as evidence. Only {summarized} of {total} parts are represented. In the user's "
        "language: state what the document is about, its main sections or arguments, key "
        "numbers or findings, and any stated limitations. Use short headings and bullets. "
        "Every bullet must cite at least one supplied source label inline; never print a "
        "standalone list of labels. Keep it to at most 4 short headings and 12 bullets, "
        "roughly 700 Chinese characters (350 English words). Never add facts that are not in "
        "the notes, and never describe a part that has no note.\n"
        f"Original user request:\n{answer_question}"
    )


async def segmented_summary(answer_question, hits, trace, settings, routing_input=None):
    """Map each fitting segment, then reduce notes; citations stay chunk-bound."""
    from knowflow.app.answer_budget import measure_request
    from knowflow.app.providers import generate_answer
    from knowflow.app.token_budget import get_token_counter

    counter = get_token_counter(getattr(settings, "chunk_tokenizer_path", None))
    budget = max(1200, min(int(settings.rag_context_token_budget) - 400, 9000))
    segments = plan_segments(hits, counter, budget)
    # Leave room for the composition call, its answer review, and one repair.
    max_segments = max(1, int(settings.query_max_provider_calls) - 3)
    planned = segments[:max_segments]
    filename = (trace.get("document") or {}).get("filename", "")
    notes = []
    for index, segment in enumerate(planned, start=1):
        ids = {str(hit.chunk.id) for hit in segment}
        question = _segment_question(answer_question, filename, index, len(segments))
        try:
            part = await generate_answer(
                question,
                [{"id": str(hit.chunk.id), "content": hit.chunk.content} for hit in segment],
                settings,
                budget_trace={},
                routing_input=routing_input,
            )
        except Exception as exc:  # a failed part is a coverage gap, never a silent pass
            notes.append(
                {"index": index, "text": "", "cited": [], "error_type": type(exc).__name__}
            )
            continue
        text, cited = part, []
        for match in MARKER.finditer(part):
            chunk_id = match.group(1).lower()
            if chunk_id not in ids:
                text = text.replace(match.group(0), "")
            elif chunk_id not in cited:
                cited.append(chunk_id)
        notes.append(
            {"index": index, "text": text.strip(), "cited": cited, "chars": len(text.strip())}
        )

    answered = [note for note in notes if note["text"] and note["cited"]]
    completed_indexes = {note["index"] for note in answered}
    read_hits = [
        hit
        for index, segment in enumerate(planned, 1)
        if index in completed_indexes
        for hit in segment
    ]
    updates = {
        "segments": len(segments),
        "planned_segments": len(planned),
        "summarized_segments": len(answered),
        "read_chunks": len(read_hits),
        "loaded_chunks": len(hits),
        "planned_chunks": sum(len(segment) for segment in planned),
        "coverage_basis": "segments_with_usable_completed_notes_not_planned_requests",
        "missing_segments": [i for i in range(1, len(segments) + 1) if i not in completed_indexes],
        "pages": pages_of(read_hits),
        "failed_segments": len(notes) - len(answered),
    }
    if not answered:
        return {"status": "failed", "updates": updates}
    updates["note_chars_max"] = max(note["chars"] for note in answered)
    question = _reduce_question(answer_question, filename, len(answered), len(segments))

    def build(granularity, limit):
        rows, seen = [], set()
        for note in answered:
            body = note["text"] if note["chars"] <= limit else note["text"][:limit] + "…"
            if granularity == "cited_chunk":
                for chunk_id in note["cited"]:
                    if chunk_id not in seen:
                        seen.add(chunk_id)
                        rows.append(
                            {
                                "id": chunk_id,
                                "content": f"[Part {note['index']}/{len(segments)}]\n{body}",
                            }
                        )
            else:
                chunk_id = note["cited"][0]
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    rows.append(
                        {
                            "id": chunk_id,
                            "content": (
                                f"[Part {note['index']}/{len(segments)}] Sources: "
                                + ", ".join(note["cited"])
                                + f"\n{body}"
                            ),
                        }
                    )
        return rows

    evidence, budget_trace, variant = [], {}, None
    for name, granularity, limit in (
        ("cited_chunk", "cited_chunk", 1600),
        ("segment_anchor", "segment_anchor", 1600),
        ("segment_anchor_short", "segment_anchor", 700),
    ):
        candidate = build(granularity, limit)
        checked = measure_request(question, candidate, settings, routing_input=routing_input)
        if checked["estimated_feasible"]:
            evidence, budget_trace, variant = candidate, checked, name
            break
    if variant is None:
        budget_trace = checked
    updates["reduce_evidence"] = variant
    return {
        "status": "ok" if variant else "failed",
        "updates": updates,
        "evidence": evidence,
        "question": question,
        "budget": budget_trace,
    }


def keep_cited_blocks(text, allowed_ids):
    """Deterministic summary check: no unit review, but every kept block cites a real chunk.

    A document summary is generated from the whole parsed text in one pass, so the
    sentence-level review is not run; instead blocks without a supplied citation are
    dropped and unknown markers are removed before citation validation.
    """
    cleaned = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        for match in list(MARKER.finditer(block)):
            if match.group(1).lower() not in allowed_ids:
                block = block.replace(match.group(0), "")
        cleaned.append(re.sub(r"[ \t]{2,}", " ", block).strip())
    kept = [
        block
        for block in cleaned
        if block and (block.startswith("#") or block.startswith("---") or MARKER.search(block))
    ]
    return "\n\n".join(kept), {
        "total_blocks": len(cleaned),
        "kept_blocks": len(kept),
        "dropped_blocks": len(cleaned) - len(kept),
    }


def coverage_statement(trace, chinese):
    document = (trace.get("document") or {}).get("filename", "")
    total = trace.get("total_chunks") or 0
    read = trace.get("read_chunks") or 0
    segments = trace.get("segments") or 0
    summarized = trace.get("summarized_segments") or 0
    pages = trace.get("pages")
    page_text_zh = f"第 {pages[0]}–{pages[1]} 页" if pages else "页码未知"
    page_text_en = f"pages {pages[0]}–{pages[1]}" if pages else "page numbers unknown"
    if trace.get("coverage") in {"complete_parsed_text", "complete_parsed_text_segmented"}:
        whole = trace.get("mode") == "whole_document"
        if chinese:
            how = "一次通读全文后写成" if whole else "按段汇总后写成"
            return (
                f"阅读范围：已读取《{document}》全部 {read}/{total} 个解析切片（{page_text_zh}），"
                f"{how}；图片、未提取页面和图表关系不在本次读取范围内。"
            )
        how = "generated in one pass over the whole text" if whole else "summarized in segments"
        return (
            f"Coverage: all {read}/{total} parsed chunks of {document} ({page_text_en}) were read, "
            f"{how}; images, unextracted pages and chart relationships were not read."
        )
    if chinese:
        return (
            f"阅读范围：本次成功汇总《{document}》{read}/{total} 个解析切片"
            f"（{page_text_zh}），完成 {summarized}/{segments} 段汇总；"
            f"其余 {max(total - read, 0)} 个切片未读或未形成可用笔记，本回答不代表全文结论。"
        )
    return (
        f"Coverage: only {read}/{total} parsed chunks of {document} produced usable notes "
        f"({page_text_en}) were read, {summarized}/{segments} segments summarized; "
        f"{max(total - read, 0)} chunks were unread or produced no usable note, "
        "so this is not a whole-document conclusion."
    )
