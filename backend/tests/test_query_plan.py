import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from knowflow.app import query_plan
from knowflow.app.config import Settings
from openai import APIConnectionError, APIStatusError, APITimeoutError


@pytest.mark.parametrize("rewrite", ["Use the 2026 policy", "Use the current policy"])
def test_rewrite_rejects_invented_or_dropped_numbers(rewrite):
    with pytest.raises(ValueError):
        query_plan.validated_queries(json.dumps({"search_queries": [rewrite]}), "Use 2025 policy")


@pytest.mark.asyncio
async def test_query_planning_uses_only_the_question_and_does_not_retry_invalid_constraints(
    monkeypatch,
):
    requests = []

    async def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"search_queries": ["2026 production timeout"]})
                    ),
                )
            ]
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **_: fake
    monkeypatch.setattr(query_plan, "chat_client", lambda: fake)
    settings = Settings(_env_file=None)
    result = await query_plan.plan_query("2025 年生产版本的超时？", settings)
    assert result["status"] == "unavailable" and not result["search_queries"]
    assert result["reason_code"] == "numeric_constraints_changed"
    assert result["phase"] == "validation" and result["response_received"]
    assert len(requests) == 1
    supplied = json.loads(requests[0]["messages"][1]["content"])
    assert supplied == {"question": "2025 年生产版本的超时？", "protected_numeric_tokens": ["2025"]}


def fake_client(monkeypatch, *, raw=None, finish="stop", error=None, empty_choices=False):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(
            choices=[]
            if empty_choices
            else [SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=raw))],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=12,
                total_tokens=112,
                unsafe_unknown_field="DO_NOT_STORE",
            ),
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **kwargs: calls.append({"options": kwargs}) or fake
    monkeypatch.setattr(query_plan, "chat_client", lambda: fake)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (None, "empty_content"),
        ("  ", "empty_content"),
        ("bad JSON", "invalid_json"),
        ("[]", "invalid_schema"),
        ("null", "invalid_schema"),
        ('{"search_queries":[]}', "invalid_query_count"),
        ('{"search_queries":"Atlas policy"}', "invalid_query_count"),
        ('{"search_queries":["one","two","three"]}', "invalid_query_count"),
        ('{"search_queries":[12]}', "invalid_query_text"),
        ('{"search_queries":["ABC"]}', "invalid_query_text"),
        ('{"search_queries":["Atlas policy 2026"]}', "numeric_constraints_changed"),
        ('{"search_queries":["Atlas policy"]}', "numeric_constraints_changed"),
    ],
)
async def test_validation_reasons_are_specific_and_no_bad_query_escapes(monkeypatch, raw, reason):
    calls = fake_client(monkeypatch, raw=raw)
    result = await query_plan.plan_query("Atlas policy 2025", Settings(_env_file=None))
    assert (result["status"], result["reason_code"], result["phase"]) == (
        "unavailable",
        reason,
        "validation",
    )
    assert result["fallback_used"] and result["original_only"]
    assert result["search_queries"] == [] and len(calls) == 2
    assert calls[0] == {"options": {"max_retries": 0}}
    assert result["provider_usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 12,
        "total_tokens": 112,
    }
    assert "DO_NOT_STORE" not in json.dumps(result)
    if isinstance(raw, str):
        assert result["response_content_sha256"] == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish", "reason"),
    [
        ("length", "output_limit"),
        ("content_filter", "content_filtered"),
        (None, "finish_reason_missing"),
        ("tool_calls", "unexpected_finish_reason"),
    ],
)
async def test_truncated_or_nonfinal_json_is_not_used_even_when_it_parses(
    monkeypatch, finish, reason
):
    raw = '{"search_queries":["Atlas 2025 policy"]}'
    calls = fake_client(monkeypatch, raw=raw, finish=finish)
    result = await query_plan.plan_query("2025 Atlas policy", Settings(_env_file=None))
    assert result["reason_code"] == reason and result["phase"] == "response"
    assert result["search_queries"] == [] and len(calls) == 2


@pytest.mark.asyncio
async def test_empty_choices_keeps_usage_and_falls_back(monkeypatch):
    fake_client(monkeypatch, empty_choices=True)
    result = await query_plan.plan_query("Atlas policy", Settings(_env_file=None))
    assert result["reason_code"] == "empty_choices" and result["response_received"]
    assert result["provider_usage"]["completion_tokens"] == 12


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("queries", "reason", "original_only"),
    [
        (["Atlas policy 2025", "Atlas policy 2025"], "no_effective_queries", True),
        (["2025 Atlas policy", "2025 Atlas policy"], "accepted", False),
    ],
)
async def test_unchanged_query_is_not_mislabeled_as_expansion(
    monkeypatch, queries, reason, original_only
):
    calls = fake_client(monkeypatch, raw=json.dumps({"search_queries": queries}))
    result = await query_plan.plan_query("Atlas policy 2025", Settings(_env_file=None))
    assert result["status"] == "complete" and result["reason_code"] == reason
    assert result["original_only"] is original_only and not result["fallback_used"]
    assert result["search_queries"] == ([] if original_only else ["2025 Atlas policy"])
    assert (
        result["request_messages_sha256"]
        == hashlib.sha256(
            json.dumps(calls[1]["messages"], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )


@pytest.mark.asyncio
async def test_mixed_good_and_bad_queries_fail_as_a_group(monkeypatch):
    fake_client(
        monkeypatch, raw=json.dumps({"search_queries": ["2025 Atlas policy", "2026 Atlas policy"]})
    )
    result = await query_plan.plan_query("Atlas policy 2025", Settings(_env_file=None))
    assert result["search_queries"] == [] and result["reason_code"] == "numeric_constraints_changed"
    assert result["validation_failure"] == {
        "query_index": 2,
        "missing_numeric_count": 1,
        "added_numeric_count": 1,
    }


@pytest.mark.parametrize(
    "question",
    [
        "Compare policy-007.pdf revision 2025 with 2024: limit 12.50, excluding stage 2.",
        "只看 ０７ 号文件，2025 年 12.50 单位，排除2024，保留第二个2025。",
        "What does the policy say, excluding production?",
    ],
)
def test_protected_tokens_preserve_literals_without_adding_context_or_special_case(question):
    messages = query_plan.query_messages(question)
    payload = json.loads(messages[1]["content"])
    assert payload == {
        "question": question,
        "protected_numeric_tokens": query_plan.numeric_tokens(question),
    }
    assert len(payload["protected_numeric_tokens"]) == len(set(payload["protected_numeric_tokens"]))
    assert query_plan.validated_queries(json.dumps({"search_queries": [question]}), question) == []
    assert "Every returned search query" in messages[0]["content"]
    assert "newspaper" not in messages[0]["content"]


def test_missing_identifier_numbers_are_not_silently_patched_into_model_queries():
    with pytest.raises(query_plan.QueryPlanError) as caught:
        query_plan.validated_queries(
            '{"search_queries":["2025 Atlas policy"]}', "2025 Atlas policy-007.pdf"
        )
    assert caught.value.reason == "numeric_constraints_changed"
    assert caught.value.details == {
        "query_index": 1,
        "missing_numeric_count": 1,
        "added_numeric_count": 0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["local_timeout", "provider_timeout", "connection", "http", "rate", "internal"]
)
async def test_request_failures_never_serialize_secret_messages_or_bodies(monkeypatch, kind):
    secret = "sensitive-test-marker"
    request = httpx.Request("POST", f"https://example.invalid/{secret}")
    errors = {
        "local_timeout": TimeoutError(secret),
        "provider_timeout": APITimeoutError(request=request),
        "connection": APIConnectionError(message=secret, request=request),
        "http": APIStatusError(secret, response=httpx.Response(401, request=request), body=secret),
        "rate": APIStatusError(secret, response=httpx.Response(429, request=request), body=secret),
        "internal": RuntimeError(secret),
    }
    reasons = {
        "connection": "provider_connection_error",
        "http": "provider_http_error",
        "rate": "provider_rate_limited",
        "internal": "unexpected_error",
    }
    calls = fake_client(monkeypatch, error=errors[kind])
    result = await query_plan.plan_query("Atlas policy", Settings(_env_file=None))
    assert result["reason_code"] == reasons.get(kind, kind)
    assert result["phase"] == "request" and not result["response_received"]
    assert result["provider_usage"] is None and result["fallback_used"]
    assert secret not in json.dumps(result) and len(calls) == 2


@pytest.mark.asyncio
async def test_real_local_deadline_cancels_once_without_swallowing_external_cancellation(
    monkeypatch,
):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        await asyncio.Event().wait()

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **_: fake
    monkeypatch.setattr(query_plan, "chat_client", lambda: fake)
    result = await query_plan.plan_query(
        "Atlas policy", Settings(_env_file=None, query_rewrite_timeout_seconds=0.01)
    )
    assert result["reason_code"] == "local_timeout" and len(calls) == 1

    async def cancelled(**_):
        raise asyncio.CancelledError

    fake.chat.completions.create = cancelled
    with pytest.raises(asyncio.CancelledError):
        await query_plan.plan_query("Atlas policy", Settings(_env_file=None))


@pytest.mark.asyncio
async def test_disabled_and_missing_configuration_are_distinct_without_a_request(monkeypatch):
    def not_configured():
        raise HTTPException(503, detail="SECRET-CONFIG")

    monkeypatch.setattr(query_plan, "chat_client", not_configured)
    disabled = await query_plan.plan_query(
        "Atlas policy", Settings(_env_file=None, query_rewrite_enabled=False)
    )
    assert disabled["status"] == "disabled" and not disabled["request_attempted"]
    failed = await query_plan.plan_query("Atlas policy", Settings(_env_file=None))
    assert failed["reason_code"] == "client_configuration_error"
    assert not failed["request_attempted"] and failed["phase"] == "configuration"
    assert "SECRET-CONFIG" not in json.dumps(failed)


def test_query_contract_is_recorded_and_changed_or_missing_contract_is_not_comparable():
    from copy import deepcopy

    from knowflow.app.evaluate_quality import initial_report
    from knowflow.app.quality_metrics import regression_check

    report = initial_report("kb", {"version": "fixture", "cases": []}, ["hybrid"], 5, False)
    assert report["query_rewrite_contract"]["version"] == query_plan.VERSION
    assert report["query_rewrite_contract"]["completion_policy"] == "stop_only-v1"
    for change in ("missing", "different"):
        other = deepcopy(report)
        if change == "missing":
            other.pop("query_rewrite_contract")
        else:
            other["query_rewrite_contract"]["version"] = "earlier"
        assert regression_check(report, other)["status"] == "not_comparable"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["numeric", "timeout"])
async def test_pipeline_falls_back_to_original_and_keeps_diagnostics_with_no_evidence(
    monkeypatch, failure
):
    from uuid import uuid4

    from knowflow.app import api
    from knowflow.app.config import get_settings
    from knowflow.app.schemas import QueryRequest, RetrievalDiagnostics

    settings = get_settings()
    monkeypatch.setattr(settings, "query_rewrite_enabled", True)
    original = "2025 Atlas policy, excluding production"
    fake_client(
        monkeypatch,
        raw='{"search_queries":["2026 Atlas policy"]}',
        error=TimeoutError("hidden") if failure == "timeout" else None,
    )
    inputs = []

    async def owned(*_):
        pass

    async def retrieve(_session, _kb, question, *_args, **kwargs):
        inputs.append((question, kwargs))
        return []

    async def assess(*_):
        return {}, RetrievalDiagnostics(status="empty")

    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "retrieve", retrieve)
    monkeypatch.setattr(api, "assess_retrieval", assess)
    response = await api.run_query(
        uuid4(),
        QueryRequest(question=original, strategy="hybrid", expand_references=False),
        SimpleNamespace(id=uuid4()),
        SimpleNamespace(),
    )
    assert inputs == [(original, {})]
    assert response.refused and not response.citations
    assert response.query_plan["fallback_used"]
    assert response.query_plan["reason_code"] == (
        "local_timeout" if failure == "timeout" else "numeric_constraints_changed"
    )
    assert response.query_plan["trace_contract"] == query_plan.TRACE_CONTRACT
