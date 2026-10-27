"""Bounded traversal of explicit, uniquely resolved document references."""

from __future__ import annotations

from collections import Counter

from sqlalchemy import or_, select

from knowflow.app.document_links import INDEX_VERSION, heading_number, normalized
from knowflow.app.models import Chunk, Document, DocumentBlock, DocumentLink, DocumentStatus
from knowflow.app.retrieval import RetrievalHit, _keyword_terms

MAX_ADDITIONS = 8
MAX_HOPS = 2
MAX_SEEDS = 3
MAX_CHUNKS_PER_LINK = 4
MAX_BLOCKS_PER_SECTION = 40
MAX_VISITED_LINKS = 32


def source_occurrence_in_chunk(link, chunk):
    if normalized(link.source_quote) not in normalized(chunk.content):
        return False
    return any(
        span.get("block_key") == link.source_block_key
        and (
            not span.get("text_range")
            or span["text_range"][0] <= link.start_offset < span["text_range"][1]
        )
        for span in chunk.source_spans or []
    )


def section_target_keys(target, blocks, label):
    """Use the explicit numbering interval; unnumbered classifier headings may be prose.

    This only proposes bounded candidates, never asserts a semantic parent-child fact.
    """
    number = label.split(":", 1)[1]
    keys, inferred = [target.block_key], False
    for block in blocks:
        if block.ordinal <= target.ordinal:
            continue
        following_number = heading_number(block)
        if following_number and not following_number.startswith(number + "."):
            break
        if block.kind in {"picture", "page_header", "page_footer", "group"}:
            continue
        if block.section != target.section and not (block.section or "").startswith(
            (target.section or "") + " / "
        ):
            inferred = True
        keys.append(block.block_key)
        if len(keys) >= MAX_BLOCKS_PER_SECTION:
            return keys, True, inferred
    return keys, False, inferred


async def expand_references(session, knowledge_base_id, query, hits, enabled=True):
    diagnostics = {
        "enabled": enabled,
        "base_count": len(hits),
        "added_count": 0,
        "max_hops": MAX_HOPS,
        "unresolved": [],
        "unindexed_documents": [],
        "inferred_section_scopes": [],
        "recovered_table_targets": [],
        "truncated": False,
        "note": "展开仅沿文内明确且唯一匹配的编号引用；不是知识关系推断，仍需逐片核验。",
    }
    if not enabled or not hits:
        return hits, diagnostics
    result = list(hits)
    by_id = {hit.chunk.id: hit for hit in hits}
    frontier = hits[:MAX_SEEDS]
    examined = set()
    unresolved_seen = set()
    terms = _keyword_terms(query)
    blocks_by_doc = {}
    for hop in range(1, MAX_HOPS + 1):
        next_frontier = []
        for source in frontier:
            keys = {s.get("block_key") for s in source.chunk.source_spans or []} - {None}
            if not keys:
                continue
            document_id = source.chunk.document_id
            # Both source and target must still belong to this active, authorized scope.
            document = await session.scalar(
                select(Document).where(
                    Document.id == document_id,
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.status == DocumentStatus.PROCESSED,
                )
            )
            if document is None:
                continue
            if document.parsing_report.get("reference_index", {}).get("version") != INDEX_VERSION:
                if source.filename not in diagnostics["unindexed_documents"]:
                    diagnostics["unindexed_documents"].append(source.filename)
                continue
            links = list(
                await session.scalars(
                    select(DocumentLink)
                    .where(
                        DocumentLink.document_id == document_id,
                        DocumentLink.source_block_key.in_(keys),
                    )
                    .order_by(DocumentLink.source_block_key, DocumentLink.start_offset)
                    .limit(51)
                )
            )
            if len(links) > 50:
                diagnostics["truncated"] = True
            for link in links[:50]:
                if (source.chunk.id, link.id) in examined:
                    continue
                if len(examined) >= MAX_VISITED_LINKS:
                    diagnostics["truncated"] = True
                    return result, diagnostics
                examined.add((source.chunk.id, link.id))
                if not source_occurrence_in_chunk(link, source.chunk):
                    continue
                if link.status != "resolved":
                    key = (document_id, link.label, link.status)
                    if key not in unresolved_seen and len(diagnostics["unresolved"]) < 20:
                        diagnostics["unresolved"].append(
                            {
                                "filename": source.filename,
                                "label": link.label,
                                "status": link.status,
                            }
                        )
                        unresolved_seen.add(key)
                    continue
                if document_id not in blocks_by_doc:
                    blocks_by_doc[document_id] = list(
                        await session.scalars(
                            select(DocumentBlock)
                            .where(DocumentBlock.document_id == document_id)
                            .order_by(DocumentBlock.ordinal)
                        )
                    )
                blocks = blocks_by_doc[document_id]
                target = next((b for b in blocks if b.block_key == link.target_block_key), None)
                if target is None:
                    continue
                if target.structure.get("caption_recovery"):
                    if link.label not in diagnostics["recovered_table_targets"]:
                        diagnostics["recovered_table_targets"].append(link.label)
                target_keys = [target.block_key]
                if link.kind == "section":
                    target_keys, truncated, inferred = section_target_keys(
                        target, blocks, link.label
                    )
                    diagnostics["truncated"] |= truncated
                    if inferred and link.label not in diagnostics["inferred_section_scopes"]:
                        diagnostics["inferred_section_scopes"].append(link.label)
                candidates = list(
                    await session.scalars(
                        select(Chunk)
                        .where(
                            Chunk.document_id == document_id,
                            Chunk.knowledge_base_id == knowledge_base_id,
                            or_(
                                *(
                                    Chunk.source_spans.contains([{"block_key": key}])
                                    for key in target_keys
                                )
                            ),
                        )
                        .order_by(Chunk.ordinal)
                        .limit(201)
                    )
                )
                if len(candidates) > MAX_CHUNKS_PER_LINK:
                    diagnostics["truncated"] = True
                # Query overlap guides budget allocation; it is not a confidence score.
                candidates.sort(
                    key=lambda c: (
                        -sum(term.casefold() in c.content.casefold() for term in terms),
                        c.ordinal,
                    )
                )
                for chunk in candidates[:MAX_CHUNKS_PER_LINK]:
                    if chunk.id == source.chunk.id:
                        continue
                    path = {
                        "source_chunk_id": str(source.chunk.id),
                        "label": link.label,
                        "source_quote": link.source_quote,
                        "hop": hop,
                    }
                    if chunk.id in by_id:
                        current = by_id[chunk.id]
                        if path not in current.reference_paths and len(current.reference_paths) < 3:
                            current.reference_paths.append(path)
                        if current not in next_frontier:
                            next_frontier.append(current)
                        continue
                    if diagnostics["added_count"] >= MAX_ADDITIONS:
                        diagnostics["truncated"] = True
                        continue
                    expanded = RetrievalHit(
                        chunk,
                        source.filename,
                        reference_paths=[path],
                        retrieval_stage="reference_expansion",
                    )
                    result.append(expanded)
                    by_id[chunk.id] = expanded
                    next_frontier.append(expanded)
                    diagnostics["added_count"] += 1
        frontier = next_frontier
        if not frontier:
            break
    diagnostics["unresolved_counts"] = dict(
        Counter(row["status"] for row in diagnostics["unresolved"])
    )
    return result, diagnostics
