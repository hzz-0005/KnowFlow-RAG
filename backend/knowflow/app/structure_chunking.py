"""Deterministic structure/token packing, independent of page breaks or LLM calls."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from knowflow.app.document_processing import TextChunk
from knowflow.app.index_text import chunk_index_text
from knowflow.app.table_provenance import table_row_sources
from knowflow.app.token_budget import get_token_counter

VERSION = "structure-tokens-v4"
MAPPING_REVISION = "piece-and-table-rows-v2"


@dataclass
class Unit:
    text: str
    section: str | None
    spans: list[dict]
    key: tuple
    block: object | None
    caption: str = ""
    caption_spans: list[dict] | None = None
    source_start: int | None = None


def _units(parsed):
    by_key = {block.block_key: block for block in parsed.blocks}
    recovered = {
        (r["source_block_key"], tuple(r["text_range"]))
        for block in parsed.blocks
        if (r := block.structure.get("caption_recovery"))
    }
    # A caption adjacent to a code/formula block binds to it as a prefix in
    # either order, mirroring table captions; prose never absorbs either part.
    parts = parsed.sections
    bound = {}
    for index, part in enumerate(parts):
        block = by_key.get(part.source_spans[0]["block_key"]) if part.source_spans else None
        if block is None or block.kind not in {"code", "formula"}:
            continue
        for neighbor in (index - 1, index + 1):
            if neighbor in bound or not 0 <= neighbor < len(parts):
                continue
            other = parts[neighbor]
            other_spans = other.source_spans
            other_block = by_key.get(other_spans[0]["block_key"]) if other_spans else None
            if (
                other.text.strip()
                and other_block is not None
                and other_block.kind == "caption"
                and other.section == part.section
            ):
                bound[neighbor] = index
    attached = defaultdict(list)
    for caption_index, target in bound.items():
        attached[target].append(caption_index)
    units = []
    for ordinal, part in enumerate(parts):
        if not part.text.strip():
            continue
        if ordinal in bound:
            continue
        block = by_key.get(part.source_spans[0]["block_key"]) if part.source_spans else None
        if block is None and parsed.report.get("parser") == "native":
            block = by_key.get(f"native-{ordinal}")
        kind = block.kind if block else "text"
        parent = block.parent_key if block else None
        caption, caption_spans = "", []
        spans = part.source_spans or ([{"block_key": block.block_key}] if block else [])
        text = part.text
        if kind == "table":
            caption = "\n".join(filter(None, block.structure.get("captions", [])))
            caption = caption or block.structure.get("caption_recovery", {}).get("text", "")
            caption_spans = [s for s in spans if s.get("block_key") != block.block_key]
            if caption and text.startswith(caption + "\n"):
                text = text[len(caption) + 1 :]
            else:
                caption = ""
            spans = [s for s in spans if s.get("block_key") == block.block_key]
            category = ("table", block.block_key)
        elif kind in {"code", "formula", "caption"}:
            category = (kind, block.block_key)
            if attached.get(ordinal):
                caption = "\n".join(parts[j].text.strip() for j in sorted(attached[ordinal]))
                caption_spans = [
                    s for j in sorted(attached[ordinal]) for s in parts[j].source_spans
                ]
        elif any((s.get("block_key"), tuple(s.get("text_range", []))) in recovered for s in spans):
            category = ("caption", block.block_key)
        else:
            # A picture ancestor keeps its extracted text separate from document prose.
            ancestor, seen = by_key.get(parent), set()
            while ancestor and ancestor.block_key not in seen and ancestor.kind != "picture":
                seen.add(ancestor.block_key)
                ancestor = by_key.get(ancestor.parent_key)
            category = (
                ("picture_text", ancestor.block_key)
                if ancestor and ancestor.kind == "picture"
                else ("prose",)
            )
        units.append(
            Unit(
                text,
                part.section,
                spans,
                (part.section, parent, *category),
                block,
                caption,
                caption_spans,
            )
        )
    tables = defaultdict(list)
    for unit in units:
        if unit.key[2] == "table":
            tables[unit.block.block_key].append(unit)
    for group in tables.values():
        mappings = table_row_sources(group[0].block, [u.text for u in group])
        if mappings is not None:
            for unit, mapping in zip(group, mappings, strict=True):
                unit.source_start = mapping["text_range"][0]
                unit.spans = [{**s, **mapping} for s in unit.spans]
    yield from units


def _piece_spans(unit, start, end):
    spans = []
    if unit.source_start is None:
        source_start = unit.block.text.find(unit.text) if unit.block else -1
        unique = source_start >= 0 and unit.block.text.find(unit.text, source_start + 1) < 0
        unit.source_start = source_start if unique else -1
    source_start = unit.source_start
    for span in unit.spans:
        copied = dict(span)
        # This range is relative to the rendered piece, never the original block.
        # Unknown mappings retain broad extents. Tables require full serialization
        # validation, not a substring match (identical rows need distinct origins).
        copied["chunk_range"] = [0, end - start]
        if source_start >= 0 and (unit.key[2] != "table" or span.get("source_mapping")):
            piece_start = source_start + start
            lo, hi = piece_start, source_start + end
            if bounds := span.get("text_range"):
                lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
            if lo >= hi:
                continue
            copied["text_range"] = [lo, hi]
            copied["chunk_range"] = [lo - piece_start, hi - piece_start]
            if "table_cells" in span:
                copied["table_cells"] = [
                    {
                        **cell,
                        "text_range": [a, b],
                        "chunk_range": [a - piece_start, b - piece_start],
                    }
                    for cell in span["table_cells"]
                    if (a := max(lo, cell["text_range"][0])) < (b := min(hi, cell["text_range"][1]))
                ]
        spans.append(copied)
    return spans


def _render(pieces, caption, caption_spans):
    content, spans = "", []
    if caption:
        content = caption + "\n"
        spans.extend({**s, "chunk_range": [0, len(caption)]} for s in caption_spans)
    for unit, start, end in pieces:
        if content and not content.endswith("\n"):
            content += "\n\n"
        offset = len(content)
        text = unit.text[start:end]
        content += text
        for span in _piece_spans(unit, start, end):
            span["chunk_range"] = [offset + n for n in span["chunk_range"]]
            if "table_cells" in span:
                span["table_cells"] = [
                    {**c, "chunk_range": [offset + n for n in c["chunk_range"]]}
                    for c in span["table_cells"]
                ]
            spans.append(span)
    return content, spans


def structure_chunks(parsed, settings):
    counter = get_token_counter(settings.chunk_tokenizer_path)
    hard = settings.chunk_max_tokens - settings.chunk_token_reserve
    target = min(settings.chunk_target_tokens, hard)
    overlap = settings.chunk_overlap_tokens
    if hard < 64 or overlap >= hard // 2:
        raise ValueError("分块安全余量或重叠预算配置不合理。")
    units = list(_units(parsed))
    # Truncating repeated context must not remove the only copy of a long caption.
    extra = []
    for unit in units:
        if unit.caption and counter.count(unit.caption) > min(96, hard // 4):
            if not any(other.text == unit.caption for other in [*units, *extra]):
                extra.append(
                    Unit(
                        unit.caption,
                        unit.section,
                        unit.caption_spans or [],
                        (unit.section, None, "caption", len(extra)),
                        None,
                    )
                )
    units.extend(extra)
    chunks, pending, counts = [], [], []
    coverage = defaultdict(list)
    unit_ids = {id(unit): index for index, unit in enumerate(units)}
    current_key, current_caption, current_caption_spans = None, "", []
    fragmented, caption_excerpted = set(), set()

    def count(section, text):
        return counter.count(chunk_index_text(section, text))

    def flush(pieces=None, caption=None, cap_spans=None):
        nonlocal pending
        active = pending if pieces is None else pieces
        if not active:
            return
        text, spans = _render(
            active,
            current_caption if caption is None else caption,
            current_caption_spans if cap_spans is None else cap_spans,
        )
        section = active[0][0].section
        tokens = count(section, text)
        if tokens > hard:
            raise ValueError("分块自动检查失败：完整 embedding 输入超过 token 预算。")
        pages = sorted({s["page_number"] for s in spans if s.get("page_number")})
        chunks.append(
            TextChunk(
                len(chunks),
                text,
                pages[0] if pages else None,
                section,
                spans,
                {
                    "strategy": VERSION,
                    "source_mapping_revision": MAPPING_REVISION,
                    "input_tokens": tokens,
                    "tokenizer_sha256": counter.metadata["sha256"],
                    "tokenizer_mode": counter.metadata["mode"],
                    "kind": active[0][0].key[2],
                    "pages": pages,
                    "fragmented": any(
                        start > 0 or end < len(unit.text) for unit, start, end in active
                    ),
                },
            )
        )
        counts.append(tokens)
        for unit, start, end in active:
            coverage[unit_ids[id(unit)]].append((start, end))
        if pieces is None:
            pending = []

    for unit in units:
        if unit.key != current_key:
            flush()
            current_key = unit.key
            current_caption = unit.caption
            cap_limit = min(96, hard // 4)
            if current_caption and counter.count(current_caption) > cap_limit:
                current_caption = current_caption[: counter.prefix_end(current_caption, cap_limit)]
                caption_excerpted.add(unit.key)
            current_caption_spans = unit.caption_spans or []
        piece = (unit, 0, len(unit.text))
        if pending:
            present, _ = _render(pending, current_caption, current_caption_spans)
            candidate, _ = _render([*pending, piece], current_caption, current_caption_spans)
            if count(unit.section, present) >= target or count(unit.section, candidate) > hard:
                flush()
        candidate, _ = _render([piece], current_caption, current_caption_spans)
        if count(unit.section, candidate) <= hard:
            pending.append(piece)
            continue
        # Only an overlong element is split, at paragraph/line/sentence boundaries
        # where possible. Repeated context and the heading participate in every check.
        flush()
        start = 0
        fragmented.add(unit_ids[id(unit)])
        while start < len(unit.text):
            while start < len(unit.text) and unit.text[start].isspace():
                start += 1
            if start == len(unit.text):
                break
            prefix = current_caption + "\n" if current_caption else ""
            available = hard - count(unit.section, prefix) - 4
            end = start + counter.prefix_end(unit.text[start:], available)
            if end <= start:
                raise ValueError("标题或表注占满分块预算，无法保留正文；请检查解析结构或配置。")
            if end < len(unit.text):
                region = unit.text[start:end]
                for pattern in (r"\n\s*\n", r"\n", r"[。！？!?；;]|\.\s+", r"\s+"):
                    breaks = [
                        m.end() for m in re.finditer(pattern, region) if m.end() >= len(region) // 2
                    ]
                    if breaks:
                        end = start + breaks[-1]
                        break
            while end > start:
                rendered, _ = _render([(unit, start, end)], current_caption, current_caption_spans)
                if count(unit.section, rendered) <= hard:
                    break
                end -= 1
            if end <= start:
                raise ValueError("单个字符及来源前缀无法放入 token 预算。")
            flush([(unit, start, end)])
            if end == len(unit.text):
                break
            next_start = end
            if overlap and unit.key[2] == "prose":
                encoded = counter.tokenizer.encode(unit.text[start:end], add_special_tokens=False)
                if len(encoded.ids) > overlap:
                    next_start = start + encoded.offsets[-overlap][0]
            start = max(start + 1, next_start)
    flush()
    # Complete non-whitespace coverage is a release gate, not a manual approval.
    for index, unit in enumerate(units):
        cursor = 0
        for start, end in sorted(coverage[index]):
            if start > cursor and unit.text[cursor:start].strip():
                raise ValueError("分块自动检查失败：存在未覆盖的原始内容。")
            cursor = max(cursor, end)
        if unit.text[cursor:].strip():
            raise ValueError("分块自动检查失败：原始内容尾部丢失。")
    fragmented_rows = [i for i in fragmented if units[i].key[2] == "table"]
    parsed.report["chunking"] = {
        "version": VERSION,
        "source_mapping_revision": MAPPING_REVISION,
        "strategy": "structure_tokens",
        "tokenizer": counter.metadata,
        "max_tokens": settings.chunk_max_tokens,
        "token_reserve": settings.chunk_token_reserve,
        "target_tokens": target,
        "overlap_tokens": overlap,
        "chunk_count": len(chunks),
        "min_input_tokens": min(counts, default=0),
        "max_input_tokens": max(counts, default=0),
        "cross_page_chunks": sum(
            len({s.get("page_number") for s in c.source_spans if s.get("page_number")}) > 1
            for c in chunks
        ),
        "source_units": len(units),
        "source_coverage_check": "passed",
        "fragmented_units": len(fragmented),
        "fragmented_table_rows": len(fragmented_rows),
        "excerpted_caption_groups": len(caption_excerpted),
        "note": (
            "预算包含章节前缀；千问同系列 tokenizer 为估算，不是托管模型的官方计费 token。"
            "来源覆盖检查不代表语义正确。"
        ),
    }
    if fragmented_rows:
        parsed.report["warnings"] = [
            *parsed.report.get("warnings", []),
            f"有 {len(fragmented_rows)} 个表格行自身超过分块预算，已分片保留；完整行语义需核对。",
        ]
        parsed.report["status"] = "needs_review"
    return chunks
