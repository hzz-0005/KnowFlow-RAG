"""Conservative page-local caption proposals, with auditable source offsets.

Never repair table cells or join continuation tables. A numbered caption must
begin a page-local text range, and have a mutually unique geometric association.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict

VERSION = "page-range-caption-v1"
CAPTION = re.compile(
    r"^(?:Table\s+|表\s*)(?P<number>(?:[A-Za-z]\.?\d{1,6}|\d{1,6})"
    r"(?:\.\d{1,6}){0,5})\s*[:：]\s*\S",
    re.I,
)
IGNORED = {"table", "picture", "group", "page_header", "page_footer", "unspecified"}
TABLE_LABEL = re.compile(
    r"^(?:Table\s+|表\s*)(?P<number>(?:[A-Za-z]\.?\d{1,6}|\d{1,6})"
    r"(?:\.\d{1,6}){0,5})(?![A-Za-z0-9_]|\.\d)",
    re.I,
)


def table_label(structure):
    captions = [
        *structure.get("captions", []),
        structure.get("caption_recovery", {}).get("text", ""),
    ]
    labels = {
        "table:" + unicodedata.normalize("NFKC", match.group("number")).casefold()
        for text in captions
        if (match := TABLE_LABEL.match(text.strip()))
    }
    return next(iter(labels)) if len(labels) == 1 else None


def valid_box(span):
    box = span.get("bbox", {})
    values = [box.get(name) for name in ("left", "top", "right", "bottom")]
    return bool(
        all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in values)
        and values[0] < values[2]
        and values[1] < values[3]
    )


def horizontal_overlap(a, b):
    return max(0, min(a["right"], b["right"]) - max(a["left"], b["left"])) / min(
        a["right"] - a["left"], b["right"] - b["left"]
    )


def candidate_regions(blocks):
    for block in blocks:
        if block.kind in IGNORED:
            continue
        spans = block.source_spans or []
        for span in spans:
            if not valid_box(span):
                continue
            bounds = span.get("text_range")
            if bounds is None and len(spans) == 1:
                bounds = [0, len(block.text)]
            if (
                not isinstance(bounds, list)
                or len(bounds) != 2
                or not all(type(n) is int for n in bounds)
                or not 0 <= bounds[0] < bounds[1] <= len(block.text)
            ):
                continue
            text = block.text[bounds[0] : bounds[1]]
            if len(text) > 1500 or not CAPTION.match(text):
                continue
            yield block, span, bounds, text


def recover_table_captions(blocks):
    # Recompute our own metadata, keeping the converter's original captions intact.
    tables = [b for b in blocks if b.kind == "table"]
    for table in tables:
        table.structure = {k: v for k, v in table.structure.items() if k != "caption_recovery"}
    regions = list(candidate_regions(blocks))
    proposals = defaultdict(list)
    rejected = Counter()
    page_blocks, page_tables = defaultdict(list), defaultdict(list)
    for block in blocks:
        for page in {s.get("page_number") for s in block.source_spans or []}:
            page_blocks[page].append(block)
            if block.kind == "table":
                page_tables[page].append(block)
    truncated = len(regions) > 2000
    if truncated:
        # Do not resolve from a partial caption population (uniqueness is unknown).
        return {
            "version": VERSION,
            "candidates": len(regions),
            "recovered": 0,
            "skipped": {"caption_budget": len(regions)},
            "truncated": True,
        }
    for block, span, bounds, text in regions:
        matches = []
        a = span["bbox"]
        page = span.get("page_number")
        if len(page_tables[page]) > 30 or len(page_blocks[page]) > 2000:
            rejected["page_geometry_budget"] += 1
            truncated = True
            continue
        for table in page_tables[page]:
            for target in table.source_spans or []:
                if target.get("page_number") != span.get("page_number") or not valid_box(target):
                    continue
                b = target["bbox"]
                if horizontal_overlap(a, b) < 0.8:
                    continue
                gap = max(a["top"], b["top"]) - min(a["bottom"], b["bottom"])
                if not 0 <= gap <= 0.065:
                    continue
                corridor = (min(a["bottom"], b["bottom"]), max(a["top"], b["top"]))
                blocked = any(
                    other.block_key not in {block.block_key, table.block_key}
                    and other.kind not in {"page_header", "page_footer", "group", "unspecified"}
                    and any(
                        s.get("page_number") == span.get("page_number")
                        and valid_box(s)
                        and s["bbox"]["top"] < corridor[1]
                        and s["bbox"]["bottom"] > corridor[0]
                        and horizontal_overlap(s["bbox"], b) >= 0.5
                        for s in other.source_spans or []
                    )
                    for other in page_blocks[page]
                )
                if not blocked:
                    matches.append(table)
                    break
        # Existing explicit captions participate in ambiguity; never choose the nearest.
        if len(matches) != 1:
            rejected["ambiguous_geometry" if matches else "no_adjacent_table"] += 1
        elif matches[0].structure.get("captions"):
            rejected["converter_caption_present"] += 1
        else:
            proposals[matches[0].block_key].append((block, span, bounds, text))
    recovered = 0
    for table in tables:
        options = proposals[table.block_key]
        if len(options) > 1:
            rejected["multiple_captions"] += len(options)
        elif options:
            block, span, bounds, text = options[0]
            table.structure = {
                **table.structure,
                "caption_recovery": {
                    "version": VERSION,
                    "method": "page_text_range_and_unique_adjacent_table",
                    "source_block_key": block.block_key,
                    "text_range": bounds,
                    "text": text,
                    "source_span": span,
                },
            }
            recovered += 1
    return {
        "version": VERSION,
        "candidates": len(regions),
        "recovered": recovered,
        "skipped": dict(rejected),
        "truncated": truncated,
        "note": "仅恢复同页明确编号表注与唯一相邻表格；跨页续表、歧义和图内关系未自动修复。",
    }


def validated_recovery(table, by_key):
    """Do not trust stale or hand-edited derived offsets when rebuilding links."""
    recovery = table.structure.get("caption_recovery", {})
    if recovery.get("version") != VERSION:
        return None
    source = by_key.get(recovery.get("source_block_key"))
    bounds = recovery.get("text_range")
    if (
        source is None
        or not isinstance(bounds, list)
        or len(bounds) != 2
        or not all(type(n) is int for n in bounds)
        or not 0 <= bounds[0] < bounds[1] <= len(source.text)
    ):
        return None
    quote = source.text[bounds[0] : bounds[1]]
    span = recovery.get("source_span", {})
    if (
        quote != recovery.get("text")
        or not CAPTION.match(quote)
        or span not in (source.source_spans or [])
        or not any(
            s.get("page_number") == span.get("page_number") for s in table.source_spans or []
        )
    ):
        return None
    return recovery
