"""A quota block must never turn an unreviewed draft into a delivered answer."""

from types import SimpleNamespace

import pytest
from knowflow.app import answer_guard, answer_review
from knowflow.app.config import Settings
from knowflow.app.round_budget import invoke, scope


@pytest.mark.asyncio
async def test_exhausted_round_does_not_deliver_unreviewed_draft_or_dispatch_repair(monkeypatch):
    async def forbidden(**kwargs):
        pytest.fail("Budget exhausted: no remote review or repair is allowed")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
    fake.with_options = lambda **kw: fake
    monkeypatch.setattr(answer_guard, "chat_client", lambda: fake)
    monkeypatch.setattr(answer_review, "chat_client", lambda: fake)
    settings = Settings(_env_file=None, query_max_provider_calls=1)

    async def already_sent():
        return {}

    with scope(settings) as budget:
        await invoke("answer_generation", "chat", "fixture", 100, 100, already_sent)
        text, validation = await answer_guard.guard_answer(
            "How long?",
            "UNREVIEWED claim is 999 days.",
            [{"id": "00000000-0000-0000-0000-000000000001", "content": "Retained for 17 days."}],
            settings,
        )
        assert not text
        assert validation["status"] != "complete"
        assert budget.snapshot()["attempted_calls"] == 1
        assert budget.snapshot()["blocked_calls"] >= 1
        assert all(not row.get("response_received") for row in budget.rows[1:])
