"""Rebuild literal table-row hints from saved cells without calling models or changing chunks."""

from __future__ import annotations

import argparse
import asyncio
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk, Document, DocumentBlock
from knowflow.app.structured_parsing import _table_rows
from knowflow.app.table_captions import table_label


async def rebuild_table_hints(session: AsyncSession, document_id: UUID) -> dict:
    document = await session.get(Document, document_id)
    if document is None:
        raise ValueError("Document not found")
    tables = await session.scalars(
        select(DocumentBlock).where(
            DocumentBlock.document_id == document_id, DocumentBlock.kind == "table"
        )
    )
    rows_by_table = {}
    for table in tables:
        texts, structure = _table_rows({"data": table.structure})
        label = table_label(table.structure)
        rows_by_table[table.block_key] = [
            (text, {**meta, **({"table_label": label} if label else {})})
            for text, meta in zip(texts, structure["body_rows"], strict=False)
        ]
    chunks = await session.scalars(select(Chunk).where(Chunk.document_id == document_id))
    updated, matched = 0, 0
    for chunk in chunks:
        spans = []
        content = " ".join(chunk.content.split())
        for span in chunk.source_spans or []:
            rows = rows_by_table.get(span.get("block_key"), [])
            matches = [meta for text, meta in rows if " ".join(text.split()) in content]
            if matches:
                for meta in matches:
                    enriched = {**span, **meta}
                    if enriched not in spans:
                        spans.append(enriched)
                        matched += 1
            elif span not in spans:
                spans.append(span)
        if spans != (chunk.source_spans or []):
            chunk.source_spans = spans
            updated += 1
    document.parsing_report = {
        **(document.parsing_report or {}),
        "table_row_hint_version": "literal-cell-table-scope-v2",
    }
    return {"updated_chunks": updated, "matched_row_hints": matched, "tables": len(rows_by_table)}


async def _main(document_id: UUID):
    async with SessionLocal() as session:
        result = await rebuild_table_hints(session, document_id)
        await session.commit()
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id", type=UUID, required=True)
    asyncio.run(_main(parser.parse_args().document_id))
