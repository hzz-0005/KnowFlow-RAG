from uuid import uuid4

import pytest
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk, Document, DocumentStatus, KnowledgeBase, User
from knowflow.app.retrieval import _english_keyword_search, _keyword_search
from sqlalchemy import delete


@pytest.mark.asyncio
async def test_stemming_recovers_inflections_and_preserves_kb_and_status_isolation():
    settings = get_settings()
    owner, kb, other = uuid4(), uuid4(), uuid4()
    async with SessionLocal() as session:
        try:
            session.add(User(id=owner, email=f"stem-{owner}@example.com", password_hash="unused"))
            await session.flush()
            session.add_all(
                [KnowledgeBase(id=k, owner_id=owner, name="Stemming") for k in [kb, other]]
            )
            await session.flush()
            expected = None
            for i, (k, status) in enumerate(
                [
                    (kb, DocumentStatus.PROCESSED),
                    (other, DocumentStatus.PROCESSED),
                    (kb, DocumentStatus.FAILED),
                ]
            ):
                doc = Document(
                    knowledge_base_id=k,
                    uploaded_by=owner,
                    filename=f"{i}.txt",
                    storage_key=uuid4().hex,
                    sha256=f"{i:064x}",
                    media_type="text/plain",
                    file_size=1,
                    status=status,
                )
                session.add(doc)
                await session.flush()
                chunk = Chunk(
                    document_id=doc.id,
                    knowledge_base_id=k,
                    ordinal=0,
                    content="Tables were filtered.",
                    search_text="Tables were filtered.",
                    embedding=[1.0] + [0.0] * (settings.embedding_dimensions - 1),
                    embedding_model=settings.embedding_model,
                )
                session.add(chunk)
                await session.flush()
                if i == 0:
                    expected = chunk.id
            await session.commit()
            assert not await _keyword_search(session, kb, "filter table", 5)
            hits = await _english_keyword_search(session, kb, "filter table", 5)
            assert [hit.chunk.id for hit in hits] == [expected]
            assert hits[0].english_keyword_rank == 1
            assert not await _english_keyword_search(session, kb, "the and", 5)
            assert not await _english_keyword_search(session, uuid4(), "filter table", 5)
        finally:
            await session.rollback()
            await session.execute(delete(User).where(User.id == owner))
            await session.commit()
