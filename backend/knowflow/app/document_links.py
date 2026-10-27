"""Resolve explicit numbering only. These are structural links, not knowledge facts."""

from __future__ import annotations

import argparse
import asyncio
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import delete, select

from knowflow.app.db import SessionLocal
from knowflow.app.models import Document, DocumentBlock, DocumentLink, DocumentStatus
from knowflow.app.table_captions import validated_recovery

INDEX_VERSION = "numbered-references-v1"
MAX_REFERENCES = 2000
NUMBER = r"(?:[A-Za-z]\.?\d{1,6}|\d{1,6})(?:\.\d{1,6}){0,5}"
REFERENCE = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<en>tables?|fig(?:ure)?s?\.?|sections?|secs?\.?)\s*(?P<en_num>{NUMBER})(?![A-Za-z0-9_]|\.\d)"
    rf"|(?P<cn>[表图])\s*(?P<cn_num>{NUMBER})(?![A-Za-z0-9_]|\.\d)"
    rf"|第\s*(?P<section_num>{NUMBER})\s*[章节]",
    re.IGNORECASE,
)
HEADING_NUMBER = re.compile(rf"^\s*(?P<num>{NUMBER})(?:[.)]?\s+|(?=[\u3400-\u9fff]))")
EXTERNAL = re.compile(
    r"(?:external|another document|other document|外部|另一份|其他文档|其它文档)", re.I
)
IGNORED_SOURCE_KINDS = {"table", "picture", "page_header", "page_footer", "group"}


def normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def reference_label(match) -> tuple[str, str]:
    name = (match.group("en") or match.group("cn") or "section").casefold()
    kind = (
        "table"
        if name.startswith("tab") or name == "表"
        else "figure"
        if name.startswith("fig") or name == "图"
        else "section"
    )
    number = match.group("en_num") or match.group("cn_num") or match.group("section_num")
    return kind, f"{kind}:{unicodedata.normalize('NFKC', number).casefold()}"


def heading_number(block) -> str | None:
    if block.kind != "section_header":
        return None
    match = REFERENCE.match(block.text)
    if match and reference_label(match)[0] == "section":
        return reference_label(match)[1].split(":", 1)[1]
    match = HEADING_NUMBER.match(block.text)
    return unicodedata.normalize("NFKC", match.group("num")).casefold() if match else None


@dataclass(slots=True)
class LinkSpec:
    source_block_key: str
    target_block_key: str | None
    kind: str
    label: str
    source_quote: str
    start_offset: int
    status: str
    candidate_keys: list[str]


def reference_excerpt(block, offset):
    spans = block.source_spans
    located = [
        s
        for s in spans
        if s.get("text_range") and s["text_range"][0] <= offset < s["text_range"][1]
    ]
    start, end = max(0, offset - 200), offset + 600
    if located:
        spans = located
        start = max(start, min(s["text_range"][0] for s in located))
        end = min(end, max(s["text_range"][1] for s in located))
    return block.text[start:end], spans


def extract_links(blocks) -> tuple[list[LinkSpec], dict]:
    targets = defaultdict(set)
    definitions = set()
    by_key = {block.block_key: block for block in blocks}
    recovered_targets = []
    for block in blocks:
        if block.kind == "table":
            for caption in block.structure.get("captions", []):
                match = REFERENCE.match(caption.strip())
                if match and reference_label(match)[0] == "table":
                    targets[reference_label(match)[1]].add(block.block_key)
            recovery = validated_recovery(block, by_key)
            if recovery:
                match = REFERENCE.match(recovery["text"])
                if match and reference_label(match)[0] == "table":
                    targets[reference_label(match)[1]].add(block.block_key)
                    definitions.add((recovery["source_block_key"], recovery["text_range"][0]))
                    recovered_targets.append(reference_label(match)[1])
        elif block.kind == "caption":
            match = REFERENCE.match(block.text)
            if match:
                kind, label = reference_label(match)
                definitions.add((block.block_key, match.start()))
                # Table captions alone cannot stand in for the actual table body.
                if kind == "figure":
                    targets[label].add(block.block_key)
        elif block.kind == "section_header":
            match = REFERENCE.match(block.text)
            if match and reference_label(match)[0] == "section":
                targets[reference_label(match)[1]].add(block.block_key)
                definitions.add((block.block_key, match.start()))
            else:
                match = HEADING_NUMBER.match(block.text)
                if match:
                    targets[f"section:{heading_number(block)}"].add(block.block_key)

    links = []
    truncated = False
    for block in blocks:
        if block.kind in IGNORED_SOURCE_KINDS:
            continue
        for match in REFERENCE.finditer(block.text):
            if (block.block_key, match.start()) in definitions:
                continue
            if len(match.group()) > 200:
                truncated = True
                continue
            if len(links) >= MAX_REFERENCES:
                truncated = True
                break
            kind, label = reference_label(match)
            candidates = sorted(targets.get(label, set()) - {block.block_key})
            if len(candidates) > 20:
                truncated = True
            before = re.split(r"[.!?;。！？；\n]", block.text[: match.start()])[-1][-35:]
            after = re.split(r"[.!?;。！？；\n]", block.text[match.end() :])[0][:35]
            context = before + match.group() + after
            external = bool(EXTERNAL.search(context))
            status = (
                "external_unresolved"
                if external
                else "resolved"
                if len(candidates) == 1
                else "ambiguous"
                if candidates
                else "missing"
            )
            links.append(
                LinkSpec(
                    block.block_key,
                    candidates[0] if status == "resolved" else None,
                    kind,
                    label,
                    match.group(),
                    match.start(),
                    status,
                    candidates[:20],
                )
            )
        if len(links) >= MAX_REFERENCES:
            break
    return links, {
        "version": INDEX_VERSION,
        "total": len(links),
        "statuses": dict(Counter(link.status for link in links)),
        "truncated": truncated,
        "recovered_table_targets": recovered_targets,
        "note": "仅解析文内明确编号；唯一匹配不代表关系语义正确，图引用仅定位文字图注。",
    }


async def rebuild_document_links(session, document: Document) -> dict:
    blocks = list(
        await session.scalars(
            select(DocumentBlock)
            .where(DocumentBlock.document_id == document.id)
            .order_by(DocumentBlock.ordinal)
        )
    )
    specs, report = extract_links(blocks)
    await session.execute(delete(DocumentLink).where(DocumentLink.document_id == document.id))
    session.add_all(
        [
            DocumentLink(
                document_id=document.id,
                source_block_key=spec.source_block_key,
                target_block_key=spec.target_block_key,
                kind=spec.kind,
                label=spec.label,
                source_quote=spec.source_quote,
                start_offset=spec.start_offset,
                status=spec.status,
                candidate_keys=spec.candidate_keys,
            )
            for spec in specs
        ]
    )
    document.parsing_report = {**document.parsing_report, "reference_index": report}
    return report


async def main(document_id):
    async with SessionLocal() as session:
        document = await session.get(Document, document_id, with_for_update=True)
        if document is None or document.status != DocumentStatus.PROCESSED:
            raise ValueError("文档不存在或尚未就绪。")
        report = await rebuild_document_links(session, document)
        await session.commit()
        print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", type=UUID, required=True)
    asyncio.run(main(parser.parse_args().document_id))
