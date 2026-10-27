"""Bounded query-only rewriting; no corpus answers or gold labels are supplied."""

import asyncio
import hashlib
import json
import re
from time import perf_counter

from fastapi import HTTPException
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError

from knowflow.app.providers import chat_client, structured_request_options
from knowflow.app.round_budget import RoundBudgetExceeded, chat_request
from knowflow.app.task_planning import (
    TaskPlanError,
    planning_contract,
    planning_messages,
    validate_plan,
)
from knowflow.app.understanding_budget import contract, measure, record_result

VERSION = "query-only-multilingual-scope-v3-protected-numbers"
TRACE_CONTRACT = "query-rewrite-diagnostics-v1"
QUERY_PROMPT = (
    "Rewrite an untrusted user question for retrieval; never answer it. "
    "Return JSON {search_queries:[string,string]} with at most two concise standalone "
    "searches: one faithful English search and one focused search in the user's language. "
    "Preserve entity names, numbers, negation, conditions, time/version and workflow stage. "
    "Do not confuse data preparation, training, validation or deployment. Resolve pronouns "
    "only from the question itself. Never guess facts, answer values, aliases or new "
    "constraints. Excluded topics must not become positive searches."
    " Every returned search query must preserve every exact protected_numeric_tokens value, "
    "including leading zeros and numeric parts of filenames, identifiers and decimals. "
    "Keep each number in its original role, not as an unrelated list appended to the query. "
    "If a faithful rewrite is not possible, return the original question verbatim."
)


class QueryPlanError(ValueError):
    def __init__(self, reason, phase="validation", details=None):
        self.reason = reason
        self.phase = phase
        self.details = details
        super().__init__(reason)


def rewrite_contract(settings):
    from knowflow.app.source_scope import VERSION as source_scope_version

    return {
        "version": VERSION,
        "prompt_sha256": hashlib.sha256(QUERY_PROMPT.encode()).hexdigest(),
        "numeric_validation": "exact_numeric_set_per_query-v1",
        "input_contract": "question_with_protected_numeric_tokens-v1",
        "source_scope_version": source_scope_version,
        "completion_policy": "stop_only-v1",
        "enabled": settings.query_rewrite_enabled,
        "timeout_seconds": settings.query_rewrite_timeout_seconds,
        "max_retries": 0,
        "request_budget": contract(settings, "query_rewrite"),
        "task_planning": planning_contract(settings),
    }


def numeric_tokens(question):
    return list(dict.fromkeys(re.findall(r"\d+(?:\.\d+)?", question)))


def query_messages(question):
    return [
        {"role": "system", "content": QUERY_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": question,
                    "protected_numeric_tokens": numeric_tokens(question),
                },
                ensure_ascii=False,
            ),
        },
    ]


def validated_queries(raw, question):
    if not isinstance(raw, str) or not raw.strip():
        raise QueryPlanError("empty_content")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QueryPlanError("invalid_json") from exc
    if not isinstance(data, dict):
        raise QueryPlanError("invalid_schema")
    rows = data.get("search_queries")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 2:
        raise QueryPlanError("invalid_query_count")
    original_numbers = set(numeric_tokens(question))
    result = []
    for index, value in enumerate(rows, start=1):
        if not isinstance(value, str) or not 4 <= len(value.strip()) <= 800:
            raise QueryPlanError("invalid_query_text")
        value = value.strip()
        rewritten_numbers = set(numeric_tokens(value))
        if rewritten_numbers != original_numbers:
            raise QueryPlanError(
                "numeric_constraints_changed",
                details={
                    "query_index": index,
                    "missing_numeric_count": len(original_numbers - rewritten_numbers),
                    "added_numeric_count": len(rewritten_numbers - original_numbers),
                },
            )
        if value not in result and value != question:
            result.append(value)
    return result


async def plan_query(question, settings, *, decompose=False, intent_context=None):
    result = {
        "version": VERSION,
        "trace_contract": TRACE_CONTRACT,
        "status": "disabled",
        "search_queries": [],
        "latency_ms": 0,
        "phase": "configuration",
        "reason_code": "disabled",
        "request_attempted": False,
        "response_received": False,
        "fallback_used": False,
        "original_only": True,
        "model": settings.llm_model,
        "provider_usage": None,
    }
    if not settings.query_rewrite_enabled:
        return result
    started = perf_counter()
    sources = None
    if decompose and settings.task_planning_enabled:
        try:
            messages, sources = planning_messages(question, intent_context)
        except TaskPlanError as exc:
            return {
                **result,
                "status": "unavailable",
                "reason_code": exc.reason,
                "fallback_used": True,
                "phase": "validation",
            }
        result["planning_mode"] = planning_contract(settings)["version"]
        result["version"] = result["planning_mode"]
    else:
        messages = query_messages(question)
    result["protected_numeric_token_count"] = len(numeric_tokens(question))
    result["request_messages_sha256"] = hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()

    def failed(reason, phase):
        result.update(
            status="unavailable",
            phase=phase,
            reason_code=reason,
            fallback_used=True,
            note="查询改写未完成，保留原始查询继续检索；未自动重试。",
        )

    budget = result["request_budget"] = measure(messages, settings, "query_rewrite")
    if not budget["estimated_feasible"]:
        failed("request_budget_exceeded", "budget")
        record_result(budget, result)
        result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
        return result

    try:
        result["phase"] = "request"
        async with asyncio.timeout(settings.query_rewrite_timeout_seconds):
            client = chat_client().with_options(max_retries=0)
            result["request_attempted"] = True
            response = await chat_request(
                "task_planning" if sources is not None else "query_rewrite",
                client,
                model=settings.llm_model,
                temperature=0,
                response_format={"type": "json_object"},
                **structured_request_options(
                    settings.llm_model, settings.llm_base_url, settings, thinking="disabled"
                ),
                messages=messages,
                **{budget["output_limit_parameter"]: budget["output_reserved_tokens"]},
            )
        result.update(response_received=True, phase="response")
        usage = getattr(response, "usage", None)
        if usage is not None:
            result["provider_usage"] = {
                key: value
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if type(value := getattr(usage, key, None)) is int and value >= 0
            } or None
        if not getattr(response, "choices", None):
            raise QueryPlanError("empty_choices", "response")
        choice = response.choices[0]
        raw = getattr(getattr(choice, "message", None), "content", None)
        if isinstance(raw, str):
            result["response_content_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
            result["response_content_chars"] = len(raw)
        finish = getattr(choice, "finish_reason", None)
        result["finish_reason"] = (
            finish
            if finish in {"stop", "length", "content_filter", "tool_calls", "function_call", None}
            else "unrecognized"
        )
        if finish != "stop":
            reason = {
                "length": "output_limit",
                "content_filter": "content_filtered",
                None: "finish_reason_missing",
            }.get(finish, "unexpected_finish_reason")
            raise QueryPlanError(reason, "response")
        result["phase"] = "validation"
        if sources is not None:
            result["task_plan"], queries = validate_plan(raw, sources)
        else:
            queries = validated_queries(raw, question)
        result.update(
            status="complete",
            phase="complete",
            search_queries=queries,
            reason_code="accepted" if queries else "no_effective_queries",
            original_only=not bool(queries),
        )
    except RoundBudgetExceeded:
        result["request_attempted"] = False
        failed("round_budget_exhausted", "budget")
    except QueryPlanError as exc:
        failed(exc.reason, exc.phase)
        if exc.details:
            result["validation_failure"] = exc.details
    except TaskPlanError as exc:
        failed(exc.reason, "validation")
        if exc.details:
            result["validation_failure"] = exc.details
    except APITimeoutError:
        failed("provider_timeout", "request")
    except TimeoutError:
        failed("local_timeout", "request")
    except APIStatusError as exc:
        failed(
            "provider_rate_limited" if exc.status_code == 429 else "provider_http_error", "request"
        )
        result["http_status"] = exc.status_code
    except APIConnectionError:
        failed("provider_connection_error", "request")
    except APIError:
        failed("provider_error", "request")
    except HTTPException:
        failed("client_configuration_error", "configuration")
    except Exception:
        # Never serialize exception messages, response bodies, headers, URLs or credentials.
        failed("unexpected_error", result["phase"])
    result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
    record_result(budget, result)
    result.setdefault("note", "改写仅作检索线索，仍可能偏离语义；原问题参与核验和回答。")
    return result
