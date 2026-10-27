import hashlib
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from knowflow.app import relevance
from knowflow.app.config import Settings
from knowflow.app.providers import task_route
from knowflow.app.quality_metrics import regression_check
from knowflow.app.relevance_budget import SYSTEM_PROMPT, budget_contract, measure, messages, plan
from knowflow.app.request_budget import output_limit


def settings(**changes):
    return Settings(
        _env_file=None, llm_model="fixture", llm_complex_model="complex-fixture", **changes
    )


def item(number=1, content="The archival period is 23 days."):
    return {"id": str(UUID(int=number)), "content": content}


def pack(question, candidates, configured=None):
    configured = configured or settings()
    route = task_route("relevance", configured, question)
    return plan(question, candidates, configured, route)


def fake_reply(candidates, *, finish="stop", raw=None, usage=None):
    return SimpleNamespace(
        model="fixture-reported",
        usage=usage,
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(
                    content=raw
                    or json.dumps(
                        {
                            "passages": [
                                {
                                    "chunk_id": row["id"],
                                    "verdict": "direct",
                                    "reason": "原文明确给出。",
                                    "quotes": [row["content"]],
                                }
                                for row in candidates
                            ]
                        }
                    )
                ),
            )
        ],
    )


def install(monkeypatch, create):
    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )


def test_budget_extraction_preserves_prior_rules_except_versioned_source_identity_addition():
    from knowflow.app.relevance_budget import SOURCE_METADATA_POLICY

    # Fingerprint of the prior base prompt; no private review directory dependency.
    expected = "c8e362e935bc25d0dd24e764883abd53d0f83eb4507c989f94c02211532568fc"
    assert SOURCE_METADATA_POLICY in SYSTEM_PROMPT
    base_prompt = SYSTEM_PROMPT.replace(SOURCE_METADATA_POLICY, "", 1)
    assert hashlib.sha256(base_prompt.encode()).hexdigest() == expected


def test_full_raw_conditions_and_whole_tail_are_preserved():
    question = "Only for test. Never infer the production value. " * 12
    candidate = item(content=" \nThe period is 23 days. " * 20 + "Only for test. \n")
    selected, trace = pack(question, [candidate])
    assert selected == [candidate]
    payload = json.loads(messages(question, selected)[1]["content"])
    assert payload == {"question": question, "passages": [candidate]}
    assert (
        trace["prompt_sha256"]
        == hashlib.sha256(
            json.dumps(messages(question, selected), ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )
    assert trace["estimated_total_tokens"] <= trace["configured_total_tokens"]


def test_large_first_passage_is_not_cut_and_later_small_passage_can_fit():
    first, second = item(content="Long preamble " * 500 + "ONLY for test."), item(2)
    selected, trace = pack(
        "How long?", [first, second], settings(retrieval_assessment_token_budget=90)
    )
    assert selected == [second]
    assert trace["decisions"][0]["reason"] == "evidence_cap"
    assert trace["skipped_count"] == 1 and trace["unknown_count"] == 2


@pytest.mark.asyncio
async def test_fixed_input_exhaustion_blocks_sdk_without_cutting_user_conditions(monkeypatch):
    monkeypatch.setattr(relevance, "chat_client", lambda: pytest.fail("Must not initialize SDK"))
    question = "Current scope has to be preserved. " * 1000
    result, diagnostics = await relevance.assess_retrieval(
        question, [item()], settings(retrieval_assessment_total_token_budget=1024)
    )
    trace = diagnostics.context_budget
    assert trace["reason_code"] == "fixed_input_exhausted"
    assert not trace["request_attempted"] and trace["provider_usage"] is None
    assert trace["question_sha256"] == hashlib.sha256(question.encode()).hexdigest()
    assert result[item()["id"]].verdict == "unknown"


@pytest.mark.asyncio
async def test_no_whole_candidate_fits_means_no_model_and_unknown(monkeypatch):
    monkeypatch.setattr(relevance, "chat_client", lambda: pytest.fail("Must not initialize SDK"))
    result, diagnostic = await relevance.assess_retrieval(
        "Why?", [item()], settings(retrieval_assessment_token_budget=1)
    )
    assert result[item()["id"]].verdict == "unknown"
    assert diagnostic.context_budget["reason_code"] == "no_whole_passage_fits"


def test_passage_limit_empty_and_duplicate_identity_keep_explicit_decisions():
    candidates = [item(i) for i in range(1, 32)] + [item(40, ""), item(50), item(50, "Conflicting")]
    selected, trace = pack("How long?", candidates)
    assert len(selected) == 30
    assert {row["id"] for row in selected} == {str(UUID(int=i)) for i in range(1, 31)}
    assert [row["reason"] for row in trace["decisions"][-4:]] == [
        "passage_limit",
        "empty",
        "duplicate_id",
        "duplicate_id",
    ]
    # Equal contents under distinct identities are not silently de-duplicated.
    assert len({row["content"] for row in selected}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "finish,reason",
    [
        ("length", "output_limit_reached"),
        ("content_filter", "content_filtered"),
        (None, "finish_reason_missing"),
        ("tool_calls", "nonfinal_model_output"),
    ],
)
async def test_valid_looking_json_from_nonfinal_response_never_promotes_evidence(
    monkeypatch, finish, reason
):
    async def create(**kwargs):
        return fake_reply([item()], finish=finish)

    install(monkeypatch, create)
    result, diagnostics = await relevance.assess_retrieval("How long?", [item()], settings())
    assert result[item()["id"]].verdict == "unknown"
    assert diagnostics.status == "unavailable"
    assert diagnostics.context_budget["reason_code"] == reason
    assert diagnostics.context_budget["response_received"]


@pytest.mark.asyncio
async def test_sdk_prompt_output_limit_and_trace_agree_with_full_request(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return fake_reply([item()], usage={"prompt_tokens": 1001, "completion_tokens": 55})

    install(monkeypatch, create)
    configured = settings(llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    configured.llm_model = configured.llm_complex_model = "qwen3.8-flash"
    result, diagnostic = await relevance.assess_retrieval("How long?", [item()], configured)
    trace = diagnostic.context_budget
    assert len(calls) == 1 and calls[0]["max_completion_tokens"] == 4096
    assert calls[0]["messages"] == messages("How long?", [item()])
    assert result[item()["id"]].verdict == "direct" and diagnostic.status == "complete"
    assert trace["observed_count"] == 1 and trace["unknown_count"] == 0
    assert trace["provider_usage"] == {"prompt_tokens": 1001, "completion_tokens": 55}
    assert trace["input_tokens_estimate"] != trace["provider_usage"]["prompt_tokens"]
    assert trace["output_limit_scope"] == "reasoning_and_answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid_json", "provider_error", "missing_row"])
async def test_failures_and_missing_rows_preserve_unknown_including_budget_omissions(
    monkeypatch, failure
):
    large = item(2, "Too long " * 1000)

    async def create(**kwargs):
        if failure == "provider_error":
            raise RuntimeError("secret-key must not leak")
        return fake_reply(
            [item()], raw="invalid" if failure == "invalid_json" else '{"passages":[]}'
        )

    install(monkeypatch, create)
    result, diagnostic = await relevance.assess_retrieval(
        "How long?", [item(), large], settings(retrieval_assessment_token_budget=90)
    )
    assert all(row.verdict == "unknown" for row in result.values())
    assert "未截断送检" in result[large["id"]].reason
    assert diagnostic.context_budget["unknown_count"] == 2
    assert "secret-key" not in diagnostic.model_dump_json()


def test_total_window_limit_is_separate_from_evidence_cap():
    configured = settings(
        retrieval_assessment_output_token_budget=64, retrieval_assessment_safety_tokens=0
    )
    question = "Specify the complete conditions. " * 100
    route = task_route("relevance", configured, question)
    base = measure(question, [], configured, route)
    configured.retrieval_assessment_total_token_budget = base["estimated_total_tokens"]
    selected, trace = plan(question, [item()], configured, route)
    assert selected == [] and trace["decisions"][0]["reason"] == "window_budget"


def test_budget_contract_changes_or_missing_history_are_not_same_strategy():
    contract = budget_contract(settings())
    new = {"relevance_budget_contract": contract}
    assert regression_check(new, {})["status"] == "not_comparable"
    changed = {**contract, "output_tokens": contract["output_tokens"] + 1}
    assert (
        regression_check(new, {"relevance_budget_contract": changed})["status"] == "not_comparable"
    )


def test_compatible_custom_endpoint_does_not_claim_verified_output_semantics():
    assert output_limit("qwen3.8-flash", "https://private.example/v1") == (
        "max_tokens",
        "unverified",
    )
