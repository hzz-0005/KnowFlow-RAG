"""Parser-neutral ordered blocks; PDF conversion never silently falls back after failure."""

from __future__ import annotations

import asyncio
import io
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from pypdf import PdfReader

from knowflow.app.config import Settings
from knowflow.app.document_processing import (
    CHUNK_CHARS,
    ParsedSection,
    TextChunk,
    _clean,
    parse_document,
    split_sections,
)


@dataclass(slots=True)
class Block:
    block_key: str
    kind: str
    text: str
    section: str | None = None
    parent_key: str | None = None
    source_spans: list[dict] = field(default_factory=list)
    structure: dict = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    sections: list[ParsedSection]
    blocks: list[Block]
    report: dict


class ParsingFailure(ValueError):
    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = {**report, "status": "failed", "error": message}


def chunk_parsed_document(
    parsed: ParsedDocument, settings: Settings | None = None
) -> list[TextChunk]:
    from knowflow.app.config import get_settings
    from knowflow.app.structure_chunking import structure_chunks

    settings = settings or get_settings()
    if settings.chunking_strategy == "structure_tokens":
        return structure_chunks(parsed, settings)
    chunks = legacy_chunk_parsed_document(parsed)
    parsed.report["chunking"] = {"strategy": "legacy", "chunk_count": len(chunks)}
    return chunks


def legacy_chunk_parsed_document(parsed: ParsedDocument) -> list[TextChunk]:
    """Pack neighboring blocks without discarding their independent source regions.

    Page, section and parent boundaries stay intact. Table rows only pack with
    rows of the same table; code/formula blocks never pack with prose.
    """
    if parsed.report.get("parser") != "docling":
        return split_sections(parsed.sections)
    by_key = {block.block_key: block for block in parsed.blocks}
    packed: list[ParsedSection] = []
    previous_key = None
    for part in parsed.sections:
        block = by_key.get(part.source_spans[0]["block_key"]) if part.source_spans else None
        category = (
            ("table", block.block_key)
            if block and block.kind == "table"
            else (block.kind,)
            if block and block.kind in {"code", "formula"}
            else ("prose",)
        )
        key = (part.page_number, part.section, block.parent_key if block else None, category)
        addition = part.text
        if block and block.kind == "table" and packed and key == previous_key:
            caption = "\n".join(filter(None, block.structure.get("captions", [])))
            caption = caption or block.structure.get("caption_recovery", {}).get("text", "")
            prefix = caption + "\n" if caption else ""
            if prefix and part.text.startswith(prefix) and packed[-1].text.startswith(prefix):
                # Keep the full caption once per packed chunk, rather than once per row.
                addition = part.text[len(prefix) :]
        if (
            packed
            and key == previous_key
            and len(packed[-1].text) + len(addition) + 2 <= CHUNK_CHARS
        ):
            previous = packed[-1]
            spans = list(previous.source_spans)
            spans.extend(span for span in part.source_spans if span not in spans)
            packed[-1] = ParsedSection(
                previous.text + "\n\n" + addition, part.page_number, part.section, spans
            )
        else:
            packed.append(part)
        previous_key = key
    return split_sections(packed)


def inspect_pdf(data: bytes, max_pages: int) -> list[dict]:
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted and not reader.decrypt(""):
        raise ValueError("PDF 已加密，请先提供可读取的文件。")
    if len(reader.pages) > max_pages:
        raise ValueError(f"PDF 超过 {max_pages} 页处理上限，请分卷上传。")
    pages = []
    for number, page in enumerate(reader.pages, 1):
        try:
            characters = len(_clean(page.extract_text() or ""))
        except Exception:
            characters = None
        pages.append(
            {
                "page_number": number,
                "native_characters": characters,
                "width": float(page.mediabox.width),
                "height": float(page.mediabox.height),
            }
        )
    return pages


def _report(parser: str, pages: list[dict], blocks: list[Block], warnings: list[str], **extra):
    replacement_characters = sum(block.text.count("\ufffd") for block in blocks)
    damaged_pages = sorted(
        {
            span["page_number"]
            for block in blocks
            if "\ufffd" in block.text
            for span in block.source_spans
        }
    )
    if replacement_characters:
        affected = ", ".join(map(str, damaged_pages)) or "未知"
        warnings.append(
            f"检测到 {replacement_characters} 个替换字符，涉及原页 {affected}；"
            "这些位置可能丢字，需核对原页或进行针对性 OCR。"
        )
    if any(page["native_characters"] is None for page in pages):
        warnings.append("部分页面的原文字层无法解码；结构化识别结果需与原页核对。")
    indexed = Counter()
    for block in blocks:
        if block.kind in {"picture", "page_header", "page_footer"}:
            continue
        parts = _page_parts(block.text, block.source_spans) if block.kind != "table" else None
        for span in block.source_spans:
            bounds = span.get("text_range") if parts else None
            indexed[span["page_number"]] += bounds[1] - bounds[0] if bounds else len(block.text)
    empty_pages = [page["page_number"] for page in pages if not indexed[page["page_number"]]]
    if empty_pages:
        warnings.append(
            f"第 {', '.join(map(str, empty_pages))} 页没有可检索正文，需核对空白页或漏识别。"
        )
    return {
        "schema_version": 1,
        "parser": parser,
        "status": "needs_review" if warnings else "parsed",
        "page_count": len(pages),
        "block_count": len(blocks),
        "table_count": sum(block.kind == "table" for block in blocks),
        "picture_count": sum(block.kind == "picture" for block in blocks),
        "replacement_characters": replacement_characters,
        "damaged_text_pages": damaged_pages,
        "pages": [{**page, "indexed_characters": indexed[page["page_number"]]} for page in pages],
        "unindexed_pages": empty_pages,
        "warnings": list(dict.fromkeys(warnings)),
        "note": "解析状态和文字覆盖不代表内容正确率；图中关系、跨页续表仍需核验。",
        **extra,
    }


def native_parse(filename: str, data: bytes, settings: Settings) -> ParsedDocument:
    if Path(filename).suffix.lower() in {".csv", ".xlsx"}:
        from knowflow.app.tabular_parsing import parse_tabular

        return parse_tabular(filename, data)
    is_pdf = Path(filename).suffix.lower() == ".pdf"
    pages = inspect_pdf(data, settings.pdf_max_pages) if is_pdf else []
    warnings = (
        ["基础 PDF 文字提取未进行版面分析或 OCR，表格、图片和阅读顺序尚未核验。"] if is_pdf else []
    )
    sections = parse_document(filename, data)
    blocks, located_sections = [], []
    for index, part in enumerate(sections):
        key = f"native-{index}"
        spans = [{"block_key": key, "page_number": part.page_number}] if part.page_number else []
        blocks.append(Block(key, "text", part.text, part.section, source_spans=spans))
        located_sections.append(ParsedSection(part.text, part.page_number, part.section, spans))
    return ParsedDocument(
        located_sections, blocks, _report("native", pages, blocks, warnings, ocr_requested=False)
    )


def _resolve(document: dict, ref: str) -> dict:
    if not isinstance(ref, str) or not ref.startswith("#/"):
        raise ValueError("Invalid document block reference")
    node: Any = document
    for part in ref[2:].split("/"):
        node = node[int(part)] if isinstance(node, list) else node[part]
    if not isinstance(node, dict):
        raise ValueError("Invalid document block")
    return node


def _source_spans(item: dict, key: str, sizes: dict) -> list[dict]:
    result = []
    raw_text = item.get("text", "")
    clean_text = _clean(raw_text)
    for prov in item.get("prov", []):
        page_no = int(prov["page_no"])
        if page_no not in sizes:
            raise ValueError("Parser returned a source page outside the original PDF")
        span = {"block_key": key, "page_number": page_no}
        charspan = prov.get("charspan")
        if (
            isinstance(charspan, (list, tuple))
            and len(charspan) == 2
            and all(type(n) is int for n in charspan)
            and 0 <= charspan[0] < charspan[1] <= len(raw_text)
        ):
            span["raw_charspan"] = list(charspan)
            local = _clean(raw_text[charspan[0] : charspan[1]])
            # Cleaning changes offsets. Keep only an exact, uniquely locatable range.
            start = clean_text.find(local) if local else -1
            if start >= 0 and clean_text.find(local, start + 1) < 0:
                span["text_range"] = [start, start + len(local)]
        bbox = prov.get("bbox")
        width, height = sizes[page_no]
        if bbox and width > 0 and height > 0:
            left, right = sorted([float(bbox["l"]), float(bbox["r"])])
            low, high = sorted([float(bbox["t"]), float(bbox["b"])])
            top, bottom = (
                (height - high, height - low)
                if bbox.get("coord_origin") == "BOTTOMLEFT"
                else (low, high)
            )
            values = [left / width, top / height, right / width, bottom / height]
            if all(math.isfinite(value) and -0.01 <= value <= 1.01 for value in values):
                span["bbox"] = dict(
                    zip(
                        ("left", "top", "right", "bottom"),
                        [max(0, min(1, v)) for v in values],
                        strict=True,
                    )
                )
        result.append(span)
    return result


def _page_parts(text: str, spans: list[dict]) -> list[tuple[str, list[dict]]] | None:
    """Partition only when page provenance accounts for every non-whitespace character."""
    if len({span["page_number"] for span in spans}) < 2:
        return None
    cursor, parts = 0, []
    for span in sorted(spans, key=lambda s: s.get("text_range", [len(text)])[0]):
        bounds = span.get("text_range")
        if not bounds:
            return None
        start, end = bounds
        if start < cursor or text[cursor:start].strip():
            return None
        parts.append((text[start:end], [span]))
        cursor = end
    return parts if not text[cursor:].strip() else None


def _table_rows(item: dict) -> tuple[list[str], dict]:
    data = item.get("data", {})
    cells = data.get("table_cells", [])
    rows, columns = int(data.get("num_rows", 0)), int(data.get("num_cols", 0))
    if rows < 0 or columns < 0 or rows * columns > 200000:
        raise ValueError("Table structure exceeds processing limit")
    grid = [[""] * columns for _ in range(rows)]
    header_rows = set()
    for cell in cells:
        r, c = int(cell["start_row_offset_idx"]), int(cell["start_col_offset_idx"])
        rend, cend = int(cell["end_row_offset_idx"]), int(cell["end_col_offset_idx"])
        if not (0 <= r < rend <= rows and 0 <= c < cend <= columns):
            raise ValueError("Invalid table cell coordinates")
        if cell.get("column_header"):
            header_rows.update(range(r, rend))
        for row in range(r, rend):
            for column in range(c, cend):
                grid[row][column] = _clean(cell.get("text", ""))
    headers = [
        " / ".join(dict.fromkeys(grid[r][c] for r in sorted(header_rows) if grid[r][c]))
        or f"列{c + 1}"
        for c in range(columns)
    ]
    texts = []
    body_rows = []
    for r, row in enumerate(grid):
        if r not in header_rows and any(row):
            texts.append(
                " | ".join(f"{headers[c]}: {value}" for c, value in enumerate(row) if value)
            )
            body_rows.append(
                {
                    "row_number": r + 1,
                    "row_key": row[0],
                    "row_key_method": "first_column_literal",
                    "cell_keys": list(
                        dict.fromkeys(
                            value
                            for value in row
                            if 2 <= len(value) <= 200 and re.search(r"[A-Za-z\u3400-\u9fff]", value)
                        )
                    )[:30],
                }
            )
    if not texts:
        texts = [" | ".join(row) for row in grid if any(row)]
    return texts, {
        "num_rows": rows,
        "num_cols": columns,
        "table_cells": cells,
        "header_rows": sorted(header_rows),
        "body_rows": body_rows,
    }


def docling_parse(payload: dict, pages: list[dict]) -> ParsedDocument:
    status = payload.get("status")
    report_base = {"parser": "docling", "ocr_requested": True, "page_count": len(pages)}
    if status not in {"success", "partial_success"}:
        raise ParsingFailure("结构化转换失败，请检查解析服务日志后重试。", report_base)
    document = payload.get("document", {}).get("json_content")
    if not isinstance(document, dict) or not isinstance(document.get("body"), dict):
        raise ParsingFailure("解析服务没有返回可用的结构化文档。", report_base)
    warnings = []
    if status == "partial_success" or payload.get("errors"):
        warnings.append("解析服务报告部分成功或转换错误，请逐页核对。")
    sizes = {p["page_number"]: (p["width"], p["height"]) for p in pages}
    for number, page in document.get("pages", {}).items():
        size = page.get("size", {})
        if int(number) in sizes and size.get("width") and size.get("height"):
            sizes[int(number)] = (float(size["width"]), float(size["height"]))
    blocks: list[Block] = []
    sections: list[ParsedSection] = []
    visited: set[str] = set()
    headings: list[tuple[int, str]] = []

    def walk(item: dict, parent: str | None = None, depth: int = 0):
        if depth > 100 or len(visited) > 50000:
            raise ValueError("Document tree exceeds processing limit")
        key = item.get("self_ref", "#/body")
        if key in visited:
            return
        visited.add(key)
        kind = str(item.get("label", "group"))
        text = _clean(item.get("text", ""))
        spans = _source_spans(item, key, sizes)
        structure: dict = {}
        if kind in {"title", "section_header"} and text:
            level = max(1, int(item.get("level", 1)))
            headings[:] = [(n, h) for n, h in headings if n < level]
            headings.append((level, text))
        section = " / ".join(title for _, title in headings)[:500] or None
        row_texts = []
        if kind == "table":
            row_texts, structure = _table_rows(item)
            captions = [
                _clean(_resolve(document, cap["$ref"]).get("text", ""))
                for cap in item.get("captions", [])
            ]
            structure["captions"] = captions
            structure["caption_spans"] = [
                span
                for cap in item.get("captions", [])
                for span in _source_spans(_resolve(document, cap["$ref"]), cap["$ref"], sizes)
            ]
            caption = "\n".join(filter(None, captions))
            row_texts = [f"{caption}\n{row}".strip() for row in row_texts]
            text = "\n".join(row_texts)
            if not text:
                warnings.append(f"表格 {key} 未提取出单元格内容。")
        if kind != "group" or text or spans:
            block = Block(key, kind, text, section, parent, spans, structure)
            blocks.append(block)
            if text and kind not in {"page_header", "page_footer", "picture"}:
                if not spans:
                    warnings.append(f"内容块 {key} 缺少原页位置。")
                for row_index, content in enumerate(row_texts or [text]):
                    row_spans = spans
                    if row_texts and row_index < len(structure.get("body_rows", [])):
                        row_spans = [
                            {**span, **structure["body_rows"][row_index]} for span in spans
                        ] + structure.get("caption_spans", [])
                    parts = _page_parts(content, spans) if not row_texts else None
                    if not row_texts and len({s["page_number"] for s in spans}) > 1 and not parts:
                        warnings.append(
                            f"内容块 {key} 跨页字符范围无法完整校验，保留全部来源供核对。"
                        )
                    for part_text, part_spans in parts or [(content, row_spans)]:
                        sections.append(
                            ParsedSection(
                                part_text,
                                part_spans[0]["page_number"] if part_spans else None,
                                section,
                                part_spans,
                            )
                        )
        for child in item.get("children", []):
            walk(_resolve(document, child["$ref"]), key, depth + 1)

    walk(document["body"])
    # Keep furniture for auditing, but never promote it to searchable body text.
    for item in document.get("texts", []):
        if item.get("self_ref") not in visited and item.get("label") in {
            "page_header",
            "page_footer",
        }:
            walk(item)
    unreached = sum(
        item.get("self_ref") not in visited
        for name in ("texts", "tables", "pictures")
        for item in document.get(name, [])
    )
    if unreached:
        warnings.append(f"有 {unreached} 个内容块未被正文树引用，需核对原文。")
    if any(b.kind == "picture" for b in blocks):
        warnings.append("已保存图片区域；图中箭头、嵌套关系和数值图表尚未经过视觉解释。")
    if not sections:
        raise ParsingFailure("未识别到可检索正文，请检查 OCR 和原文件清晰度。", report_base)
    from knowflow.app.table_captions import recover_table_captions, table_label

    caption_report = recover_table_captions(blocks)
    if caption_report["truncated"]:
        warnings.append("表注恢复达到处理预算，部分候选未检查；请结合原页核对。")
    if caption_report["recovered"]:
        warnings.append("部分表注从页内字符范围与版面位置恢复关联；这是结构候选，仍需核对原页。")
        recovered = {
            block.block_key: block.structure["caption_recovery"]
            for block in blocks
            if block.structure.get("caption_recovery")
        }
        # A recovered caption accompanies each table row just as converter captions do.
        # Preserve both regions so the combined retrieval text remains attributable.
        sections = [
            ParsedSection(
                f"{recovered[part.source_spans[0]['block_key']]['text']}\n{part.text}",
                part.page_number,
                part.section,
                [*part.source_spans, recovered[part.source_spans[0]["block_key"]]["source_span"]],
            )
            if part.source_spans and part.source_spans[0]["block_key"] in recovered
            else part
            for part in sections
        ]
    labels = {b.block_key: table_label(b.structure) for b in blocks if b.kind == "table"}
    sections = [
        ParsedSection(
            part.text,
            part.page_number,
            part.section,
            [
                {**span, "table_label": labels[span["block_key"]]}
                if labels.get(span["block_key"])
                else span
                for span in part.source_spans
            ],
        )
        for part in sections
    ]
    report = _report(
        "docling",
        pages,
        blocks,
        warnings,
        converter_status=status,
        document_schema_version=document.get("version"),
        ocr_requested=True,
        processing_time=payload.get("processing_time"),
        table_caption_recovery=caption_report,
    )
    return ParsedDocument(sections, blocks, report)


async def parse_for_ingestion(
    filename: str,
    data: bytes,
    settings: Settings,
    *,
    force_ocr: bool = False,
    page_range: tuple[int, int] | None = None,
) -> ParsedDocument:
    if Path(filename).suffix.lower() != ".pdf" or settings.pdf_parser == "native":
        if force_ocr or page_range:
            raise ParsingFailure(
                "局部 OCR 需要启用 Docling PDF 解析器。", {"parser": settings.pdf_parser}
            )
        try:
            return await asyncio.to_thread(native_parse, filename, data, settings)
        except Exception as exc:
            parser = "tabular" if Path(filename).suffix.lower() in {".csv", ".xlsx"} else "native"
            raise ParsingFailure(str(exc), {"parser": parser, "ocr_requested": False}) from exc
    report = {"parser": "docling", "ocr_requested": True}
    try:
        pages = await asyncio.to_thread(inspect_pdf, data, settings.pdf_max_pages)
        report["page_count"] = len(pages)
        source_page_count = len(pages)
        if page_range:
            start, end = page_range
            if not 1 <= start <= end <= len(pages):
                raise ParsingFailure("请求页码超出原文件范围。", report)
            pages = [page for page in pages if start <= page["page_number"] <= end]
        options = {
            "to_formats": "json",
            "do_ocr": "true",
            "force_ocr": str(force_ocr).lower(),
            "ocr_lang": ["iso:zh"],
            "do_table_structure": "true",
            "table_mode": "accurate",
            "include_images": "false",
            "image_export_mode": "placeholder",
            "document_timeout": str(settings.docling_timeout_seconds - 5),
        }
        if page_range:
            options["page_range"] = [str(page_range[0]), str(page_range[1])]
        async with (
            asyncio.timeout(settings.docling_timeout_seconds),
            httpx.AsyncClient(timeout=settings.docling_timeout_seconds, trust_env=False) as client,
        ):
            response = await client.post(
                f"{settings.docling_base_url}/v1/convert/file/async",
                files={"files": (filename, data, "application/pdf")},
                data=options,
            )
            response.raise_for_status()
            task = response.json()
            task_id = str(UUID(task["task_id"]))
            while True:
                response = await client.get(f"{settings.docling_base_url}/v1/status/poll/{task_id}")
                response.raise_for_status()
                task_status = response.json().get("task_status")
                if task_status == "failure":
                    raise ParsingFailure(
                        "OCR 或版面分析失败，请检查 Docling 服务日志后重试。", report
                    )
                if task_status == "success":
                    break
                if task_status not in {"pending", "started"}:
                    raise ValueError("Unexpected conversion task status")
                await asyncio.sleep(2)
            response = await client.get(f"{settings.docling_base_url}/v1/result/{task_id}")
            response.raise_for_status()
            payload = response.json()
        parsed = await asyncio.to_thread(docling_parse, payload, pages)
        parsed.report.update(
            force_ocr_requested=force_ocr,
            source_page_count=source_page_count,
            requested_page_range=list(page_range) if page_range else None,
        )
        return parsed
    except ParsingFailure:
        raise
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise ParsingFailure(
            "结构化解析超时，未降级为可能漏页的基础解析，请重试或分卷上传。", report
        ) from exc
    except httpx.HTTPError as exc:
        raise ParsingFailure("结构化解析服务不可用，请检查 Docling 服务后重试。", report) from exc
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise ParsingFailure(
            "结构化解析结果无效或 PDF 无法读取，请检查原文件和解析服务。", report
        ) from exc


def render_pdf_page(path: Path, page_number: int) -> bytes:
    # PDFium is not thread-safe. Rendering is serialized by the API's async lock.
    import pypdfium2 as pdfium

    with pdfium.PdfDocument(path) as document:
        if not 1 <= page_number <= len(document):
            raise ValueError("Page number outside document")
        page = document[page_number - 1]
        try:
            scale = min(1.5, 1600 / max(page.get_size()))
            bitmap = page.render(scale=scale)
            try:
                output = io.BytesIO()
                bitmap.to_pil().save(output, format="PNG")
                return output.getvalue()
            finally:
                bitmap.close()
        finally:
            page.close()
