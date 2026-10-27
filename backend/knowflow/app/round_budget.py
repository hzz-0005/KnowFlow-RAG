"""Online request-local admission ledger. Estimates are not billing guarantees."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from knowflow.app.request_budget import estimate_messages
from knowflow.app.token_budget import get_token_counter

VERSION = "online-round-admission-v1"
CURRENT = ContextVar("knowflow_round_budget", default=None)


class RoundBudgetExceeded(ValueError):
    def __init__(self, reason, stage):
        self.reason, self.stage = reason, stage
        super().__init__(reason)


def usage_of(response):
    usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        usage = {
            k: getattr(usage, k, None)
            for k in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "input_tokens",
                "output_tokens",
            )
        }
    return {
        k: v
        for k, v in usage.items()
        if k
        in {"prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"}
        and type(v) is int
        and v >= 0
    } or None


class RoundBudget:
    def __init__(self, settings, *, clock=None):
        self.settings = settings
        self.clock = clock or perf_counter
        self.started = self.clock()
        self.deadline = self.started + settings.query_timeout_seconds
        self.calls_limit = settings.query_max_provider_calls
        self.tokens_limit = settings.query_total_reserved_tokens
        self.committed = 0
        self.stop_reason = None
        self.rows = []
        self.counter = get_token_counter(settings.chunk_tokenizer_path)

    def remaining(self):
        return max(0, self.deadline - self.clock())

    def admit(self, stage, kind, model, input_tokens, output_tokens):
        reserve = input_tokens + output_tokens
        reason = self.stop_reason or (
            "deadline_exhausted"
            if self.remaining() <= 0
            else "provider_calls_exhausted"
            if sum(r["status"] != "blocked" for r in self.rows) >= self.calls_limit
            else "reserved_tokens_exhausted"
            if self.committed + reserve > self.tokens_limit
            else None
        )
        row = {
            "sequence": len(self.rows) + 1,
            "stage": stage,
            "kind": kind,
            "model": model,
            "status": "blocked" if reason else "started",
            "reason": reason,
            "input_tokens_estimate": input_tokens,
            "output_reserved_tokens": output_tokens,
            "reserved_tokens": reserve,
            "provider_usage": None,
            "response_received": False,
            "latency_ms": 0,
            "max_retries": 0,
        }
        self.rows.append(row)
        if reason:
            self.stop_reason = reason
            raise RoundBudgetExceeded(reason, stage)
        self.committed += reserve
        return row

    def snapshot(self):
        attempted = [r for r in self.rows if r["status"] != "blocked"]
        totals = [(r.get("provider_usage") or {}).get("total_tokens") for r in attempted]
        return {
            "version": VERSION,
            "scope": "online_turn_provider_requests",
            "status": "stopped"
            if any(r["status"] == "blocked" for r in self.rows)
            or self.remaining() <= 0
            or self.committed > self.tokens_limit
            else "within_limits",
            "call_limit": self.calls_limit,
            "token_reservation_limit": self.tokens_limit,
            "deadline_seconds": self.settings.query_timeout_seconds,
            "elapsed_ms": round((self.clock() - self.started) * 1000, 2),
            "remaining_ms": round(self.remaining() * 1000, 2),
            "attempted_calls": len(attempted),
            "blocked_calls": len(self.rows) - len(attempted),
            "committed_reserved_tokens": self.committed,
            "observed_total_tokens": sum(totals) if all(v is not None for v in totals) else None,
            "known_reported_tokens_subtotal": sum(v for v in totals if v is not None),
            "usage_unknown_calls": sum(v is None for v in totals),
            "tokenizer_sha256": self.counter.metadata["sha256"],
            "billing_amount": None,
            "billing_status": "not_calculated",
            "calls": [dict(r) for r in self.rows],
        }


@contextmanager
def scope(settings):
    previous = CURRENT.get()
    if previous is not None:
        yield previous
        return
    budget = RoundBudget(settings)
    token = CURRENT.set(budget)
    try:
        yield budget
    finally:
        CURRENT.reset(token)


async def invoke(stage, kind, model, input_tokens, output_tokens, operation):
    budget = CURRENT.get()
    if budget is None:
        return await operation()
    row = budget.admit(stage, kind, model, input_tokens, output_tokens)
    started = budget.clock()
    try:
        async with asyncio.timeout(budget.remaining()):
            response = await operation()
        row.update(
            status="response_received", response_received=True, provider_usage=usage_of(response)
        )
        actual = (row["provider_usage"] or {}).get("total_tokens")
        if actual is not None and actual > row["reserved_tokens"]:
            extra = actual - row["reserved_tokens"]
            row["reported_above_reservation"] = extra
            budget.committed += extra
        return response
    except asyncio.CancelledError:
        row.update(
            status="cancelled",
            reason="deadline_exhausted" if budget.remaining() <= 0 else "cancelled",
        )
        raise
    except Exception as exc:
        row.update(
            status="failed",
            reason="deadline_exhausted" if budget.remaining() <= 0 else "provider_error",
            error_type=type(exc).__name__,
        )
        raise
    finally:
        row["latency_ms"] = round((budget.clock() - started) * 1000, 2)


async def chat_request(stage, client, **kwargs):
    budget = CURRENT.get()
    if budget is None:
        return await client.chat.completions.create(**kwargs)
    used = estimate_messages(kwargs["messages"], budget.counter)
    output = kwargs.get("max_completion_tokens", kwargs.get("max_tokens"))
    if type(output) is not int or output < 1:
        # Unknown output reservation must not escape admission control.
        row = {
            "stage": stage,
            "kind": "chat",
            "status": "blocked",
            "reason": "output_reservation_missing",
        }
        budget.rows.append(row)
        budget.stop_reason = "output_reservation_missing"
        raise RoundBudgetExceeded("output_reservation_missing", stage)
    return await invoke(
        stage,
        "chat",
        kwargs.get("model"),
        used,
        output,
        lambda: client.with_options(max_retries=0).chat.completions.create(**kwargs),
    )


async def embedding_request(client, **kwargs):
    budget = CURRENT.get()
    if budget is None:
        return await client.embeddings.create(**kwargs)
    values = kwargs["input"]
    used = 64 + sum(budget.counter.count(t) for t in values)
    return await invoke(
        "query_embedding",
        "embedding",
        kwargs.get("model"),
        used,
        0,
        lambda: client.with_options(max_retries=0).embeddings.create(**kwargs),
    )


async def rerank_request(payload, operation):
    budget = CURRENT.get()
    if budget is None:
        return await operation()
    texts = [
        payload["input"]["query"],
        *payload["input"]["documents"],
        payload["parameters"].get("instruct", ""),
    ]
    used = 128 + sum(budget.counter.count(t) for t in texts)
    return await invoke("rerank", "rerank", payload.get("model"), used, 0, operation)
