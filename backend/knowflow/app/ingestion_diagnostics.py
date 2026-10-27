"""Bounded, deterministic observations of persisted sources; never a semantic grade."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter

from knowflow.app.index_text import chunk_index_text
from knowflow.app.source_consistency import (
    CONTRACT as CONSISTENCY_CONTRACT,
)
from knowflow.app.source_consistency import (
    MATCH,
    MISMATCH,
    NORMALIZED_MATCH,
    NOT_COMPARABLE,
    compare_excerpt,
    sample_row,
)

VERSION = "persisted-integrity-v1"
VERSION_V3 = "persisted-integrity-v3"
EXAMPLES = 8
LAYOUT_SENSITIVE_KINDS = {
    "table",
    "table_row",
    "table_cell",
    "caption",
    "figure",
    "picture",
    "page_header",
    "page_footer",
    "formula",
}
CONSISTENCY_EXAMPLES_PER_STATUS = 3
CONSISTENCY_CHARACTER_BUDGET = 4_000_000


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def valid_range(value, length):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(integer(n) for n in value)
        and 0 <= value[0] < value[1] <= length
    )


def valid_cells(value):
    if not isinstance(value, str) or len(value) > 30:
        return False
    cells = value.split(":")
    if len(cells) not in (1, 2):
        return False
    points = []
    for cell in cells:
        match = re.fullmatch(r"([A-Z]{1,3})([1-9][0-9]{0,6})", cell)
        if not match:
            return False
        col = 0
        for char in match[1]:
            col = col * 26 + ord(char) - 64
        row = int(match[2])
        if col > 16384 or row > 1048576:
            return False
        points.append((col, row))
    return all(a <= b for a, b in zip(points[0], points[-1], strict=True))


def diagnose(
    document,
    blocks,
    chunks,
    totals,
    *,
    counter=None,
    limits=None,
    block_texts=None,
    consistency_char_budget=CONSISTENCY_CHARACTER_BUDGET,
):
    """Rows are projections, excluding embeddings. Counts describe inspected rows only.

    block_texts: optional mapping block_key -> loaded source text (may be truncated).
    When provided, source-text consistency runs under CONTRACT; when None the report
    keeps persisted-integrity-v1 semantics (no consistency claims at all).
    """
    raw_report = document.get("parsing_report")
    report = raw_report if isinstance(raw_report, dict) else {}
    raw_saved = report.get("chunking")
    saved = raw_saved if isinstance(raw_saved, dict) else {}
    complete_blocks = len(blocks) == totals["blocks"]
    complete_chunks = len(chunks) == totals["chunks"]
    complete = complete_blocks and complete_chunks
    sources_complete = all(isinstance(c.get("source_spans"), list) for c in chunks)
    findings = {}

    def add(code, severity, message, location=None):
        item = findings.setdefault(
            code,
            {
                "code": code,
                "severity": severity,
                "message": message,
                "count": 0,
                "examples": [],
            },
        )
        item["count"] += 1
        if location is not None and len(item["examples"]) < EXAMPLES:
            item["examples"].append(location)

    by_key = {b["block_key"]: b for b in blocks}
    referenced, pages, seen = set(), set(), {}
    consistency_counts = Counter()
    consistency_reasons = Counter()
    consistency_characters = 0
    chunks_without_span_objects = 0
    source_texts_complete = True
    consistency_examples = {
        status: [] for status in (MISMATCH, NORMALIZED_MATCH, NOT_COMPARABLE, MATCH)
    }
    token_observed = token_unknown = 0
    token_max = None
    limit = saved.get("max_tokens")
    reserve = saved.get("token_reserve")
    token_policy_valid = (
        integer(limit)
        and integer(reserve)
        and 0 <= reserve < limit
        and counter is not None
        and isinstance(saved.get("tokenizer"), dict)
        and bool(saved["tokenizer"].get("sha256"))
        and saved["tokenizer"]["sha256"] == counter.metadata.get("sha256")
    )
    pdf = document["filename"].lower().endswith(".pdf")
    page_count = report.get("page_count")
    known_pages = pdf and integer(page_count) and 0 < page_count <= 100000
    located = 0
    for chunk in chunks:
        location = {"chunk_id": str(chunk["id"]), "ordinal": chunk["ordinal"]}
        text = chunk["content"]
        text_complete = len(text) == chunk["text_length"]
        if not text_complete:
            add("chunk_text_budget", "unknown", "切片正文超出诊断单行预算，未完整检查。", location)
        if not text.strip() and text_complete:
            add("empty_chunk", "error", "切片正文为空。", location)
        if "\ufffd" in text:
            add(
                "replacement_character",
                "warning",
                "切片含 Unicode 替换字符；需核对编码或 OCR。",
                location,
            )
        if text_complete:
            digest = hashlib.sha256(text.encode()).hexdigest()
            if digest in seen:
                add(
                    "duplicate_content",
                    "info",
                    "存在正文完全相同的切片；可能是重复上下文，未自动删除。",
                    {
                        **location,
                        "same_as_ordinal": seen[digest],
                    },
                )
            seen[digest] = chunk["ordinal"]
        spans = chunk.get("source_spans")
        if not isinstance(spans, list) or not spans:
            add("source_missing", "unknown", "没有可检查的来源列表，或来源超过读取预算。", location)
            spans = []
        else:
            located += 1
        if block_texts is not None and not any(isinstance(s, dict) for s in spans):
            chunks_without_span_objects += 1
            add(
                "source_consistency_unmapped_chunk",
                "unknown",
                "此切片没有来源项，无法进行文字比对；该切片不计入来源项分母。",
                location,
            )
        for span in spans:
            if not isinstance(span, dict):
                add("invalid_span", "error", "来源项不是对象。", location)
                continue
            key = span.get("block_key")
            where = {**location, "block_key": key, "page_number": span.get("page_number")}
            if span.get("source_type") == "spreadsheet":
                where.update(sheet_name=span.get("sheet_name"), cell_range=span.get("cell_range"))
                if (
                    not isinstance(span.get("sheet_name"), str)
                    or not span["sheet_name"].strip()
                    or not valid_cells(span.get("cell_range"))
                ):
                    add("invalid_cell_location", "error", "工作表名或单元格范围无效。", where)
            if not isinstance(key, str) or not key:
                add("block_key_missing", "unknown", "来源没有块标识，无法验证块关联。", where)
            elif key not in by_key:
                add(
                    "dangling_block" if complete_blocks else "block_lookup_incomplete",
                    "error" if complete_blocks else "unknown",
                    "来源块不存在。"
                    if complete_blocks
                    else "块读取不完整，无法判断来源块是否存在。",
                    where,
                )
            else:
                referenced.add(key)
                if "text_range" in span and not valid_range(
                    span["text_range"], by_key[key]["text_length"]
                ):
                    add("invalid_text_range", "error", "来源字符区间越界或格式无效。", where)
            if "chunk_range" not in span:
                add("chunk_range_missing", "unknown", "缺少切片字符区间，无法精确定位节选。", where)
            elif not valid_range(span["chunk_range"], chunk["text_length"]):
                add("invalid_chunk_range", "error", "切片字符区间越界或格式无效。", where)
            if block_texts is not None:
                known = isinstance(key, str) and key in by_key
                comparison = compare_excerpt(
                    span,
                    block_text=block_texts.get(key) if known else None,
                    chunk_text=text,
                    block_text_length=by_key[key]["text_length"] if known else 0,
                    chunk_text_length=chunk["text_length"],
                    max_compared_characters=max(
                        0, consistency_char_budget - consistency_characters
                    ),
                )
                consistency_characters += comparison["characters_inspected"]
                if comparison["reason"] in {"source_text_unavailable", "source_text_truncated"}:
                    source_texts_complete = False
                status = comparison["status"]
                consistency_counts[status] += 1
                if comparison["reason"]:
                    consistency_reasons[comparison["reason"]] += 1
                if status == MISMATCH:
                    layout_sensitive = known and by_key[key]["kind"] in LAYOUT_SENSITIVE_KINDS
                    add(
                        "source_excerpt_layout_mismatch"
                        if layout_sensitive
                        else "source_excerpt_mismatch",
                        "warning" if layout_sensitive else "error",
                        (
                            "表格/图注等布局敏感块的节选与来源文字不一致；"
                            "表格重排或 OCR 清洗可能是合法差异，需人工核对。"
                            if layout_sensitive
                            else "切片节选与来源正文不一致（空白折叠与 Unicode 规范化差异已排除）。"
                        ),
                        where,
                    )
                elif status == NORMALIZED_MATCH:
                    add(
                        "source_excerpt_normalized",
                        "info",
                        "节选与来源仅在空白折叠或 Unicode NFC 规范化后一致。",
                        where,
                    )
                elif status == NOT_COMPARABLE:
                    add(
                        "source_consistency_unknown",
                        "unknown",
                        "来源项文字比对无法完成，具体原因见文字对照分类计数；保留未知。",
                        where,
                    )
                bucket = consistency_examples[status]
                if len(bucket) < CONSISTENCY_EXAMPLES_PER_STATUS:
                    bucket.append(
                        sample_row(
                            status,
                            comparison["reason"],
                            comparison["normalizations"],
                            where,
                            comparison["detail"],
                            comparison["chunk_range"],
                            comparison["text_range"],
                        )
                    )
            page = span.get("page_number")
            if page is not None:
                if not integer(page) or page < 1 or (known_pages and page > page_count):
                    add("invalid_page", "error", "来源页码无效或超过已保存的页数。", where)
                else:
                    pages.add(page)
            box = span.get("bbox")
            if box is not None:
                values = (
                    [box.get(k) for k in ("left", "top", "right", "bottom")]
                    if isinstance(box, dict)
                    else []
                )
                if (
                    len(values) != 4
                    or not all(
                        isinstance(v, (int, float))
                        and not isinstance(v, bool)
                        and math.isfinite(v)
                        and 0 <= v <= 1
                        for v in values
                    )
                    or values[0] >= values[2]
                    or values[1] >= values[3]
                ):
                    add("invalid_bbox", "error", "来源区域不是有效的归一化矩形。", where)
        if token_policy_valid and text_complete:
            count = counter.count(chunk_index_text(chunk.get("section"), text))
            token_observed += 1
            token_max = max(token_max or 0, count)
            if count > limit - reserve:
                add(
                    "token_over_budget",
                    "error",
                    "完整输入（含章节前缀）超过保存的分块预算。",
                    {
                        **location,
                        "input_tokens": count,
                        "budget": limit - reserve,
                    },
                )
        else:
            token_unknown += 1

    # Block presence is a structural observation, not character/semantic coverage.
    if complete_chunks and sources_complete:
        for block in blocks:
            if (
                block["text_length"]
                and block["kind"] not in {"picture", "page_header", "page_footer"}
                and block["block_key"] not in referenced
            ):
                add(
                    "unreferenced_block",
                    "warning",
                    "解析块未被切片来源引用；可能被有意排除，需核对。",
                    {
                        "block_key": block["block_key"],
                        "block_ordinal": block["ordinal"],
                    },
                )
    if known_pages and complete_chunks and sources_complete:
        for page in range(1, page_count + 1):
            if page not in pages:
                add(
                    "page_without_chunk_source",
                    "warning",
                    "此页没有切片来源；可能为空白页、图片或解析遗漏。",
                    {"page_number": page},
                )
    if not known_pages and pdf:
        add("page_count_unknown", "unknown", "没有有效的历史页数，不能计算页覆盖缺口。")
    if not complete:
        add(
            "inspection_incomplete",
            "unknown",
            "达到读取预算，只检查部分记录；未发现异常不代表全量通过。",
        )
    if token_unknown:
        add(
            "token_policy_unknown",
            "unknown",
            "缺少匹配的历史 tokenizer/预算或完整正文，不能重算该批切片的历史预算。",
        )
    if document["status"] != "processed":
        add(
            "index_not_current",
            "warning",
            "文档尚未处理完成或本次处理失败；展示的可能是保留的旧索引。",
        )
    if not totals["chunks"]:
        add("no_indexed_chunks", "unknown", "没有已保存切片，不能评估切片质量。")
    snapshot = {"document": document, "blocks": blocks, "chunks": chunks, "totals": totals}
    if block_texts is not None:
        snapshot["block_texts"] = block_texts
    fingerprint = hashlib.sha256(
        json.dumps(
            snapshot, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":")
        ).encode()
    ).hexdigest()
    return {
        "schema_version": VERSION_V3 if block_texts is not None else VERSION,
        "document_id": document["id"],
        "filename": document["filename"],
        "document_status": document["status"],
        "snapshot_sha256": fingerprint,
        "scope": {
            "complete": complete,
            "total": totals,
            "inspected": {"blocks": len(blocks), "chunks": len(chunks)},
            "limits": limits or {},
            "content_complete": all(len(c["content"]) == c["text_length"] for c in chunks),
            "sources_complete": sources_complete,
            "source_texts_complete": source_texts_complete if block_texts is not None else None,
        },
        "observations": {
            "source_presence": {"observed": located, "inspected": len(chunks)},
            "page_coverage": {
                "status": "observed"
                if known_pages and complete_chunks and sources_complete
                else ("unknown" if pdf else "not_applicable"),
                "expected_pages": page_count if known_pages else None,
                "pages_with_chunk_source": len(pages)
                if known_pages and complete_chunks and sources_complete
                else None,
            },
            "token_budget": {
                "observed": token_observed,
                "unknown": token_unknown,
                "max_input_tokens": token_max,
                "saved_budget": limit - reserve if token_policy_valid else None,
            },
            "source_consistency": (
                {
                    "contract": CONSISTENCY_CONTRACT,
                    "status": "observed"
                    if sum(consistency_counts[s] for s in (MATCH, NORMALIZED_MATCH, MISMATCH))
                    else "unknown",
                    "coverage": {
                        "chunks_inspected": len(chunks),
                        "chunks_without_span_objects": chunks_without_span_objects,
                        "span_objects_inspected": sum(consistency_counts.values()),
                    },
                    "work": {
                        "characters_inspected": consistency_characters,
                        "character_budget": consistency_char_budget,
                    },
                    "spans": {
                        "compared": consistency_counts.get(MATCH, 0)
                        + consistency_counts.get(NORMALIZED_MATCH, 0)
                        + consistency_counts.get(MISMATCH, 0),
                        MATCH: consistency_counts.get(MATCH, 0),
                        NORMALIZED_MATCH: consistency_counts.get(NORMALIZED_MATCH, 0),
                        MISMATCH: consistency_counts.get(MISMATCH, 0),
                        NOT_COMPARABLE: consistency_counts.get(NOT_COMPARABLE, 0),
                    },
                    "reasons": dict(consistency_reasons),
                    "examples": [
                        row
                        for status in (MISMATCH, NORMALIZED_MATCH, NOT_COMPARABLE, MATCH)
                        for row in consistency_examples[status]
                    ],
                    "note": (
                        "只证明已保存来源文字与切片节选一致；"
                        "不证明 OCR 正确、引用相关或答案语义支持。"
                    ),
                }
                if block_texts is not None
                else {
                    "contract": None,
                    "status": "not_run",
                    "note": (
                        "本次运行未读取来源正文，未执行一致性比对（persisted-integrity-v1 语义）。"
                    ),
                }
            ),
        },
        "severity_counts": dict(Counter(f["severity"] for f in findings.values())),
        "findings": list(findings.values()),
        "note": (
            "这是已保存索引的机械检查，不是 OCR 正确率、语义完整性或问答成功率；"
            "没有异常也不代表语义正确。"
        ),
    }
