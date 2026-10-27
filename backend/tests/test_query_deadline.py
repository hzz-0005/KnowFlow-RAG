import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from knowflow.app import api
from knowflow.app.config import get_settings
from knowflow.app.schemas import QueryRequest


@pytest.mark.asyncio
async def test_interactive_deadline_cancels_work_and_returns_an_explicit_status(monkeypatch):
    stopped = []

    async def slow(*_):
        try:
            await asyncio.sleep(10)
        finally:
            stopped.append(True)

    monkeypatch.setattr(api, "run_query", slow)
    monkeypatch.setattr(get_settings(), "query_timeout_seconds", 0.01)
    with pytest.raises(HTTPException) as caught:
        await api.query_knowledge_base(
            uuid4(), QueryRequest(question="A?"), SimpleNamespace(), None
        )
    assert stopped and caught.value.status_code == 504
    assert "未经核验" in caught.value.detail
