"""Request admission controls, not supplier billing/semantic quality estimates."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from knowflow.app import round_budget as rb
from knowflow.app.config import Settings


def settings(**kw):
    return Settings(_env_file=None, **kw)


def client(*, response=None, error=None):
    calls, options = [], []

    async def create(**kw):
        calls.append(kw)
        if error:
            raise error
        return response or SimpleNamespace(usage=None)

    result = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        embeddings=SimpleNamespace(create=create),
    )
    result.with_options = lambda **kw: options.append(kw) or result
    return result, calls, options


MESSAGES = [{"role": "user", "content": "Do not include production 007."}]


@pytest.mark.asyncio
@pytest.mark.parametrize("dimension", ["calls", "tokens"])
async def test_exact_boundary_next_request_blocked_and_used_reservation_not_refunded(dimension):
    cfg = settings(
        query_max_provider_calls=1 if dimension == "calls" else 12, query_total_reserved_tokens=1024
    )
    executed = []

    async def operation():
        executed.append(True)
        return {"usage": {"total_tokens": 1}}

    with rb.scope(cfg) as budget:
        await rb.invoke("test", "chat", "fixture", 512, 512, operation)
        assert budget.committed == 1024
        with pytest.raises(rb.RoundBudgetExceeded) as exc:
            await rb.invoke("next", "chat", "fixture", 1, 0, operation)
        assert exc.value.reason == (
            "provider_calls_exhausted" if dimension == "calls" else "reserved_tokens_exhausted"
        )
        snapshot = budget.snapshot()
        assert (
            snapshot["attempted_calls"] == 1
            and snapshot["blocked_calls"] == 1
            and len(executed) == 1
        )
        assert snapshot["observed_total_tokens"] == 1 and snapshot["status"] == "stopped"
    assert rb.CURRENT.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("SECRET-ERROR"), asyncio.CancelledError()])
async def test_failed_cancelled_requests_keep_reservation_and_usage_unknown(error):
    async def operation():
        raise error

    with rb.scope(settings()) as budget:
        with pytest.raises(type(error)):
            await rb.invoke("first", "chat", "m", 400, 600, operation)
        snap = budget.snapshot()
        assert snap["committed_reserved_tokens"] == 1000 and snap["attempted_calls"] == 1
        assert snap["observed_total_tokens"] is None and snap["usage_unknown_calls"] == 1
        assert "SECRET-ERROR" not in json.dumps(snap)


@pytest.mark.asyncio
async def test_reported_above_reservation_is_charged_and_global_overrun_stops_next_request():
    async def operation():
        return {"usage": {"total_tokens": 1100, "private": "DO-NOT-SAVE"}}

    with rb.scope(settings(query_total_reserved_tokens=1024)) as budget:
        await rb.invoke("first", "embedding", "m", 100, 0, operation)
        assert budget.committed == 1100
        with pytest.raises(rb.RoundBudgetExceeded):
            await rb.invoke("next", "chat", "m", 0, 1, operation)
        snap = budget.snapshot()
        assert snap["calls"][0]["reported_above_reservation"] == 1000
        assert snap["known_reported_tokens_subtotal"] == 1100
        assert "DO-NOT-SAVE" not in json.dumps(snap)


@pytest.mark.asyncio
async def test_unknown_usage_does_not_erase_known_subtotal_or_refund_budget():
    async def known():
        return {"usage": {"total_tokens": 10}}

    async def unknown():
        return {}

    with rb.scope(settings()) as budget:
        await rb.invoke("one", "chat", "m", 100, 100, known)
        await rb.invoke("two", "rerank", "m", 100, 0, unknown)
        snap = budget.snapshot()
        assert (
            snap["observed_total_tokens"] is None and snap["known_reported_tokens_subtotal"] == 10
        )
        assert snap["usage_unknown_calls"] == 1 and snap["committed_reserved_tokens"] == 300


@pytest.mark.asyncio
async def test_first_denial_latches_stop_even_if_a_smaller_later_request_could_fit():
    executed = []

    async def operation():
        executed.append(True)
        return {}

    with rb.scope(settings(query_total_reserved_tokens=1024)) as budget:
        await rb.invoke("first", "chat", "m", 100, 100, operation)
        with pytest.raises(rb.RoundBudgetExceeded):
            await rb.invoke("too_large", "chat", "m", 1000, 1000, operation)
        with pytest.raises(rb.RoundBudgetExceeded):
            await rb.invoke("smaller", "embedding", "m", 1, 0, operation)
        assert len(executed) == 1 and budget.snapshot()["blocked_calls"] == 2


@pytest.mark.asyncio
async def test_chat_embedding_rerank_share_one_scope_and_force_zero_sdk_retries():
    fake, calls, options = client(
        response=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, total_tokens=12))
    )
    with rb.scope(settings(query_max_provider_calls=3)) as budget:
        await rb.chat_request("query_rewrite", fake, model="m", messages=MESSAGES, max_tokens=100)
        await rb.embedding_request(fake, model="e", input=["text"], dimensions=4)
        await rb.rerank_request(
            {"model": "r", "input": {"query": "Q", "documents": ["Doc"]}, "parameters": {}},
            lambda: fake.chat.completions.create(),
        )
        with pytest.raises(rb.RoundBudgetExceeded):
            await rb.chat_request(
                "answer_generation", fake, model="m", messages=MESSAGES, max_tokens=100
            )
        assert len(calls) == 3 and options == [{"max_retries": 0}, {"max_retries": 0}]
        assert [r["kind"] for r in budget.rows] == ["chat", "embedding", "rerank", "chat"]
        assert (
            budget.rows[0]["response_received"]
            and budget.rows[0]["provider_usage"]["total_tokens"] == 12
        )


@pytest.mark.asyncio
async def test_concurrent_admission_shares_limit_without_a_race():
    started = asyncio.Event()
    release = asyncio.Event()

    async def operation():
        started.set()
        await release.wait()
        return {}

    with rb.scope(settings(query_max_provider_calls=1)) as budget:
        first = asyncio.create_task(rb.invoke("one", "chat", "m", 1, 1, operation))
        await started.wait()
        with pytest.raises(rb.RoundBudgetExceeded):
            await rb.invoke("two", "chat", "m", 1, 1, operation)
        release.set()
        await first
        assert budget.snapshot()["attempted_calls"] == 1


@pytest.mark.asyncio
async def test_independent_tasks_are_isolated_nested_scope_reuses_deadline():
    async def worker():
        with rb.scope(settings()) as budget:

            async def operation():
                await asyncio.sleep(0)
                return {}

            with rb.scope(settings(query_max_provider_calls=1)) as nested:
                assert nested is budget and nested.deadline == budget.deadline
                await rb.invoke("one", "chat", "m", 1, 1, operation)
            return budget

    left, right = await asyncio.gather(worker(), worker())
    assert left is not right and len(left.rows) == len(right.rows) == 1 and rb.CURRENT.get() is None


@pytest.mark.asyncio
async def test_deadline_does_not_reset_between_stages_and_cancels_inflight_request():
    ticks = [0.0]
    budget = rb.RoundBudget(settings(query_timeout_seconds=1), clock=lambda: ticks[0])
    token = rb.CURRENT.set(budget)
    try:
        ticks[0] = 1.01
        with pytest.raises(rb.RoundBudgetExceeded) as exc:
            await rb.invoke("late", "chat", "m", 1, 1, lambda: pytest.fail("no dispatch"))
        assert exc.value.reason == "deadline_exhausted"
    finally:
        rb.CURRENT.reset(token)

    async def stuck():
        await asyncio.Event().wait()

    with rb.scope(settings(query_timeout_seconds=0.01)) as budget:
        with pytest.raises(TimeoutError):
            await rb.invoke("inflight", "chat", "m", 1, 1, stuck)
        assert budget.rows[0]["reason"] == "deadline_exhausted"
        assert budget.snapshot()["usage_unknown_calls"] == 1


@pytest.mark.asyncio
async def test_missing_output_limit_is_blocked_but_offline_path_retains_original_call_options():
    fake, calls, options = client()
    with rb.scope(settings()) as budget:
        with pytest.raises(rb.RoundBudgetExceeded):
            await rb.chat_request("unknown", fake, model="m", messages=MESSAGES)
        assert not calls and budget.snapshot()["blocked_calls"] == 1
    await rb.chat_request("offline", fake, model="m", messages=MESSAGES)
    assert len(calls) == 1 and not options


@pytest.mark.asyncio
async def test_single_query_quota_refusal_includes_ledger_and_resets_context(monkeypatch):
    from knowflow.app import api
    from knowflow.app.schemas import QueryRequest

    async def pipeline(*_):
        async def operation():
            return {}

        await rb.invoke("first", "chat", "m", 1, 1, operation)
        await rb.invoke("second", "chat", "m", 1, 1, operation)

    monkeypatch.setattr(api, "run_query", pipeline)
    monkeypatch.setattr(api, "get_settings", lambda: settings(query_max_provider_calls=1))
    result = await api.query_knowledge_base(None, QueryRequest(question="Q"), None, None)
    assert result.refused and not result.citations
    assert result.round_budget["attempted_calls"] == 1 and result.round_budget["blocked_calls"] == 1
    assert rb.CURRENT.get() is None
