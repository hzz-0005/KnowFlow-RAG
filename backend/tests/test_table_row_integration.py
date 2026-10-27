from uuid import uuid4

import pytest
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk, Document, DocumentBlock, DocumentStatus, KnowledgeBase, User
from knowflow.app.retrieval import _table_row_search
from knowflow.app.table_hints import rebuild_table_hints
from sqlalchemy import delete, select


@pytest.mark.asyncio
async def test_table_row_recall_backfill_idempotence_and_kb_status_isolation():
    settings = get_settings()
    owner_id, kb_id, other_kb_id = uuid4(), uuid4(), uuid4()
    vector = [1.0] + [0.0] * (settings.embedding_dimensions - 1)
    async with SessionLocal() as session:
        try:
            session.add(
                User(id=owner_id, email=f"table-{owner_id}@example.com", password_hash="unused")
            )
            await session.flush()
            session.add_all(
                [
                    KnowledgeBase(id=kb_id, owner_id=owner_id, name="Tables"),
                    KnowledgeBase(id=other_kb_id, owner_id=owner_id, name="Other"),
                ]
            )
            await session.flush()
            docs = [
                Document(
                    id=uuid4(),
                    knowledge_base_id=kb,
                    uploaded_by=owner_id,
                    filename=f"table-{i}.pdf",
                    storage_key=uuid4().hex,
                    media_type="application/pdf",
                    file_size=1,
                    sha256=f"{i:064x}",
                    status=status,
                )
                for i, (kb, status) in enumerate(
                    [
                        (kb_id, DocumentStatus.PROCESSED),
                        (other_kb_id, DocumentStatus.PROCESSED),
                        (kb_id, DocumentStatus.FAILED),
                    ]
                )
            ]
            session.add_all(docs)
            await session.flush()
            cells = []
            for r, values in enumerate(
                [["Class", "Count"], ["Table", "34733"], ["Text", "510377"]]
            ):
                for c, text in enumerate(values):
                    cells.append(
                        {
                            "text": text,
                            "column_header": r == 0,
                            "start_row_offset_idx": r,
                            "end_row_offset_idx": r + 1,
                            "start_col_offset_idx": c,
                            "end_col_offset_idx": c + 1,
                        }
                    )
            session.add(
                DocumentBlock(
                    document_id=docs[0].id,
                    block_key="#/tables/0",
                    ordinal=0,
                    kind="table",
                    text="original table",
                    source_spans=[{"page_number": 4}],
                    structure={"num_rows": 3, "num_cols": 2, "table_cells": cells},
                )
            )
            original_ids = []
            for doc in docs:
                for ordinal, (label, number) in enumerate([("Table", "34733"), ("Text", "510377")]):
                    chunk_id = uuid4()
                    if doc == docs[0]:
                        original_ids.append(chunk_id)
                    spans = [{"block_key": "#/tables/0", "page_number": 4}]
                    if doc != docs[0]:
                        spans[0]["row_key"] = label
                    session.add(
                        Chunk(
                            id=chunk_id,
                            document_id=doc.id,
                            knowledge_base_id=doc.knowledge_base_id,
                            ordinal=ordinal,
                            content=f"A shared table caption\nClass: {label} | Count: {number}",
                            search_text="Class Count",
                            embedding=vector,
                            page_number=4,
                            source_spans=spans,
                            embedding_model=settings.embedding_model,
                        )
                    )
            await session.commit()
            result = await rebuild_table_hints(session, docs[0].id)
            assert result["updated_chunks"] == 2
            await session.commit()
            assert (await rebuild_table_hints(session, docs[0].id))["updated_chunks"] == 0
            hits = await _table_row_search(session, kb_id, "Table 类和 Text类的 Count 是多少？", 8)
            assert {hit.chunk.id for hit in hits} == set(original_ids)
            assert [hit.table_rank for hit in hits] == [1, 2]
            assert all(hit.chunk.document_id == docs[0].id for hit in hits)
            # An older index without table numbers retains its row-key behavior.
            legacy = await _table_row_search(session, kb_id, "Table 1 的 Text 类 Count", 8)
            assert legacy and all(hit.chunk.document_id == docs[0].id for hit in legacy)
            # Add same-named rows in a different table, then backfill explicit labels.
            table = await session.scalar(
                select(DocumentBlock).where(DocumentBlock.document_id == docs[0].id)
            )
            table.structure = {**table.structure, "captions": ["Table 5: Results"]}
            await rebuild_table_hints(session, docs[0].id)
            decoy = Chunk(
                document_id=docs[0].id,
                knowledge_base_id=kb_id,
                ordinal=9,
                content="Class: Text | Count: 999",
                search_text="Text",
                embedding=vector,
                embedding_model=settings.embedding_model,
                source_spans=[{"row_key": "Text", "cell_keys": ["Text"], "table_label": "table:1"}],
            )
            session.add(decoy)
            await session.commit()
            scoped = await _table_row_search(session, kb_id, "Table 5 的 Text 类 Count", 8)
            assert scoped[0].chunk.id == original_ids[1]
            assert decoy.id not in {hit.chunk.id for hit in scoped}
        finally:
            await session.rollback()
            await session.execute(delete(User).where(User.id == owner_id))
            await session.commit()
