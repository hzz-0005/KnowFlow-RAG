import os

import pytest
from knowflow.app.config import get_settings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_postgres_has_pgvector_and_full_text_search() -> None:
    """Exercise the real pgvector and PostgreSQL search operators on the Compose DB."""
    database_url = os.getenv("KNOWFLOW_TEST_DATABASE_URL", get_settings().database_url)
    engine = create_async_engine(database_url, pool_pre_ping=True)
    try:
        async with engine.connect() as connection:
            vector_distance = await connection.scalar(
                text("SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector")
            )
            chinese_word_match = await connection.scalar(
                text(
                    "SELECT to_tsvector('simple', '知识 图谱 retrieval') "
                    "@@ plainto_tsquery('simple', '图谱')"
                )
            )
            assert vector_distance == pytest.approx(0.0)
            assert chinese_word_match is True
    finally:
        await engine.dispose()
