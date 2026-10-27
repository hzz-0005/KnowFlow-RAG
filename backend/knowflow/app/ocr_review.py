"""Read-only page OCR candidates. Signals describe differences, never correctness."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import UUID

from sqlalchemy import select, update

from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Document, DocumentBlock, OcrReview, utc_now
from knowflow.app.structured_parsing import ParsingFailure, _page_parts, parse_for_ingestion

logger = logging.getLogger(__name__)
IGNORED_KINDS = {"picture", "page_header", "page_footer", "group"}
MAX_PAGE_CHARACTERS = 200000
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*|[\u3400-\u9fff]", re.I)
NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:[.,]\d+)*(?:[ \t]*[%％])?")


def page_snapshot(blocks, page_number: int, parser: str) -> dict:
    selected = [
        b
        for b in blocks
        if b.kind not in IGNORED_KINDS
        and any(span.get("page_number") == page_number for span in b.source_spans)
    ]
    texts, shared, partitioned = [], [], []
    for block in selected:
        parts = _page_parts(block.text, block.source_spans) if block.kind != "table" else None
        if parts:
            partitioned.append(block.block_key)
            texts.extend(text for text, spans in parts if spans[0]["page_number"] == page_number)
        else:
            if block.text:
                texts.append(block.text)
            if len({s.get("page_number") for s in block.source_spans}) > 1:
                shared.append(block.block_key)
    text = "\n\n".join(texts)
    if len(text) > MAX_PAGE_CHARACTERS:
        raise ValueError("单页文字超过对照上限，请先检查异常解析块。")
    return {
        "parser": parser,
        "text": text,
        "characters": len(text),
        "replacement_characters": text.count("\ufffd"),
        "block_count": len(selected),
        "table_count": sum(b.kind == "table" for b in selected),
        "shared_page_blocks": shared,
        "partitioned_page_blocks": partitioned,
    }


async def stored_snapshot(session, document: Document, page_number: int) -> tuple[dict, str]:
    blocks = list(
        await session.scalars(
            select(DocumentBlock)
            .where(
                DocumentBlock.document_id == document.id,
                DocumentBlock.source_spans.contains([{"page_number": page_number}]),
            )
            .order_by(DocumentBlock.ordinal)
        )
    )
    snapshot = page_snapshot(blocks, page_number, document.parsing_report.get("parser", "unknown"))
    # IDs deliberately invalidate a comparison after reprocessing even if text is identical.
    version = {
        "source_sha256": document.sha256,
        "blocks": [str(b.id) for b in blocks],
        "snapshot": snapshot,
    }
    fingerprint = hashlib.sha256(
        json.dumps(version, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return snapshot, fingerprint


def compare_snapshots(baseline: dict, candidate: dict) -> dict:
    native = unicodedata.normalize("NFKC", baseline["text"]).casefold()
    ocr = unicodedata.normalize("NFKC", candidate["text"]).casefold()
    original_tokens, candidate_tokens = (
        Counter(TOKEN_PATTERN.findall(native)),
        Counter(TOKEN_PATTERN.findall(ocr)),
    )
    retained = sum((original_tokens & candidate_tokens).values())
    token_count = sum(original_tokens.values())
    before, after = Counter(NUMBER_PATTERN.findall(native)), Counter(NUMBER_PATTERN.findall(ocr))
    missing_numbers, added_numbers = before - after, after - before
    ratio = (
        round(candidate["characters"] / baseline["characters"], 3)
        if baseline["characters"]
        else None
    )
    warnings = []
    if not baseline["text"].strip():
        warnings.append("当前页没有已入库正文，无法计算相对覆盖；需与原页人工核对。")
    if not candidate["text"].strip():
        warnings.append("OCR 候选没有提取出正文。")
    if ratio is not None and ratio < 0.7:
        warnings.append("候选文字量不足当前正文的 70%，可能漏识别，也可能消除了重复；需核对原页。")
    if missing_numbers or added_numbers:
        warnings.append("数字或出现次数发生变化；变化可能来自识别、去重或阅读顺序，请逐项核对。")
    if candidate["table_count"] < baseline["table_count"]:
        warnings.append("候选表格块数量减少，需检查表头、单元格和单位是否丢失。")
    if baseline["shared_page_blocks"] or candidate["shared_page_blocks"]:
        warnings.append("存在跨页来源块，对照文字包含整个块，不能据此判定单页覆盖。")
    if candidate["replacement_characters"]:
        warnings.append("候选仍包含替换字符，需要核对。")
    return {
        "character_ratio": ratio,
        "baseline_token_retention": round(retained / token_count, 3) if token_count else None,
        "missing_numbers": [
            {"value": value, "count": count} for value, count in missing_numbers.most_common(50)
        ],
        "added_numbers": [
            {"value": value, "count": count} for value, count in added_numbers.most_common(50)
        ],
        "number_details_truncated": len(missing_numbers) > 50 or len(added_numbers) > 50,
        "warnings": warnings,
        "decision": "manual_review_required",
        "note": (
            "相对文字重合和数字变化是排查线索，不是 OCR 准确率；"
            "当前正文也可能有错。候选未进入检索或图谱。"
        ),
    }


async def expire_reviews(session) -> None:
    await session.execute(
        update(OcrReview)
        .where(OcrReview.status == "running", OcrReview.expires_at < utc_now())
        .values(
            status="failed",
            error_message="对照已超时或被服务重启中断，请重新运行。",
            finished_at=utc_now(),
        )
    )


async def run_ocr_review(review_id: UUID) -> None:
    settings = get_settings()
    started = perf_counter()
    async with SessionLocal() as session:
        review = await session.get(OcrReview, review_id)
        if review is None or review.status != "running":
            return
        document = await session.get(Document, review.document_id)
        if document is None:
            return
        filename, storage_key, source_sha = document.filename, document.storage_key, document.sha256
        page, baseline = review.page_number, review.result["baseline"]
    try:
        async with asyncio.timeout(settings.docling_timeout_seconds + 10):
            data = await asyncio.to_thread((Path(settings.upload_dir) / storage_key).read_bytes)
            if hashlib.sha256(data).hexdigest() != source_sha:
                raise ValueError("原文件校验失败，请检查上传文件。")
            parsed = await parse_for_ingestion(
                filename, data, settings, force_ocr=True, page_range=(page, page)
            )
            candidate = page_snapshot(parsed.blocks, page, "docling-forced-ocr")
            # Reject invalid location rather than assigning a page number to unlocated text.
            if any(s.get("page_number") != page for b in parsed.blocks for s in b.source_spans):
                raise ValueError("OCR 候选的物理页码与请求不一致。")
            result = {
                "baseline": baseline,
                "candidate": candidate,
                "comparison": compare_snapshots(baseline, candidate),
                "candidate_report": parsed.report,
                "seconds": round(perf_counter() - started, 2),
                "production_index_modified": False,
            }
        error = None
    except Exception as exc:
        logger.exception("Page OCR comparison failed: %s", review_id)
        error = (
            str(exc)
            if isinstance(exc, (ParsingFailure, ValueError))
            else "OCR 对照失败，请检查解析服务后重试。"
        )
        result = None
    async with SessionLocal() as session:
        # A late worker must not revive an expired result or recreate a deleted document.
        values = {
            "status": "failed" if error else "succeeded",
            "error_message": error,
            "finished_at": utc_now(),
        }
        if result is not None:
            values["result"] = result
        await session.execute(
            update(OcrReview)
            .where(
                OcrReview.id == review_id,
                OcrReview.status == "running",
                OcrReview.expires_at >= utc_now(),
            )
            .values(**values)
        )
        await session.commit()
