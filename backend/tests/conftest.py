"""New paid pipeline stages are mocked/off in the ordinary offline test suite."""

import pytest
from knowflow.app.config import get_settings


@pytest.fixture(autouse=True)
def disable_unmocked_paid_pipeline_stages(monkeypatch):
    from knowflow.app import referent_candidates

    def unmocked_candidate_client():
        raise RuntimeError("Candidate provider must be mocked in offline tests")

    async def unmocked_candidate_retrieval(*args, **kwargs):
        raise RuntimeError("Candidate retrieval must be mocked in offline tests")

    monkeypatch.setattr(referent_candidates, "chat_client", unmocked_candidate_client)
    monkeypatch.setattr(referent_candidates, "retrieve", unmocked_candidate_retrieval)
    settings = get_settings()
    monkeypatch.setattr(settings, "query_rewrite_enabled", False)
    monkeypatch.setattr(settings, "answer_guard_enabled", False)
    monkeypatch.setattr(settings, "answer_repair_enabled", False)
    monkeypatch.setattr(settings, "english_keyword_retrieval_enabled", False)
