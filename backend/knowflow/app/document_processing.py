from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any

from docx import Document as WordDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as WordTable
from docx.text.paragraph import Paragraph as WordParagraph
from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class ParsedSection:
    text: str
    page_number: int | None = None
    section: str | None = None
    source_spans: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    content: str
    page_number: int | None
    section: str | None
    source_spans: list[dict[str, Any]] = field(default_factory=list)
    processing_metadata: dict[str, Any] = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}
CHUNK_CHARS = 1000
CHUNK_OVERLAP = 120


class UnsupportedDocumentError(ValueError):
    pass


def parse_document(filename: str, data: bytes) -> list[ParsedSection]:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDocumentError(
            "Supported document formats are TXT, Markdown, PDF, DOCX, CSV, and XLSX"
        )
    if not data:
        raise ValueError("The uploaded document is empty")
    if suffix in {".csv", ".xlsx"}:
        from knowflow.app.tabular_parsing import parse_tabular

        return parse_tabular(filename, data).sections

    if suffix in {".txt", ".md"}:
        text = data.decode("utf-8-sig", errors="replace")
        return _markdown_sections(text) if suffix == ".md" else [ParsedSection(_clean(text))]

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            sections = [
                ParsedSection(_clean(page.extract_text() or ""), page_number=index)
                for index, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError(f"Could not read PDF: {exc}") from exc
        if not any(section.text for section in sections):
            raise ValueError("No text was extracted. This may be a scanned PDF and needs OCR.")
        return sections

    try:
        document = WordDocument(io.BytesIO(data))
        sections: list[ParsedSection] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        def flush_section() -> None:
            content = _clean("\n".join(current_lines))
            if content:
                sections.append(ParsedSection(content, section=current_heading))
            current_lines.clear()

        for element in document.element.body.iterchildren():
            if isinstance(element, CT_P):
                paragraph = WordParagraph(element, document)
                text = _clean(paragraph.text)
                if not text:
                    continue
                if paragraph.style and paragraph.style.name.startswith("Heading"):
                    flush_section()
                    current_heading = text[:500]
                current_lines.append(text)
            elif isinstance(element, CT_Tbl):
                table = WordTable(element, document)
                for row in table.rows:
                    cells = [" | ".join(cell.text.split()) for cell in row.cells]
                    row_text = _clean(" | ".join(cells))
                    if row_text:
                        current_lines.append(row_text)
        flush_section()
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise ValueError("Could not read DOCX document") from exc
    if not sections:
        raise ValueError("No text was extracted from the DOCX document")
    return sections


def split_sections(
    sections: list[ParsedSection], chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[TextChunk]:
    if chunk_chars < 100 or overlap < 0 or overlap >= chunk_chars:
        raise ValueError(
            "Chunk size must be at least 100 and overlap must be smaller than its size"
        )

    chunks: list[TextChunk] = []
    for source in sections:
        text = _clean(source.text)
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            if end < len(text):
                boundary = max(
                    text.rfind(mark, start + chunk_chars // 2, end) for mark in "。！？；\n.?! "
                )
                if boundary > start:
                    end = boundary + 1
            part = _clean(text[start:end])
            if part:
                chunks.append(
                    TextChunk(
                        ordinal=len(chunks),
                        content=part,
                        page_number=source.page_number,
                        section=source.section,
                        source_spans=source.source_spans,
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return chunks


def _markdown_sections(text: str) -> list[ParsedSection]:
    sections: list[ParsedSection] = []
    heading_path: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    fence_marker: str | None = None

    def flush_section() -> None:
        content = _clean("\n".join(current_lines))
        if content:
            sections.append(ParsedSection(content, section=current_heading))
        current_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()
        fence_match = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            current_lines.append(stripped)
            continue

        heading_match = None if fence_marker else re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading_match:
            flush_section()
            level = len(heading_match.group(1))
            heading_path = [(depth, title) for depth, title in heading_path if depth < level]
            heading_path.append((level, heading_match.group(2)[:500]))
            current_heading = " / ".join(title for _, title in heading_path)[:500]
            current_lines.append(stripped)
        elif stripped:
            current_lines.append(stripped)
        elif current_lines and current_lines[-1]:
            current_lines.append("")

    flush_section()
    return sections


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    return re.sub(r"[\t\r ]+", " ", text).strip()
