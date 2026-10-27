import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from knowflow.app import relevance
from knowflow.app.config import Settings
from knowflow.app.providers import ModelRoute
from knowflow.app.relevance import assess_retrieval, retain_supported_context, validate_assessments
from knowflow.app.schemas import RelevanceAssessment

SOURCE_ID = str(UUID(int=1))
EVIDENCE = [
    {"id": SOURCE_ID, "content": "Mercury production runs in Beijing. Development runs in Paris."}
]


def response_row(**overrides):
    return {
        "chunk_id": SOURCE_ID,
        "verdict": "direct",
        "reason": "明确给出了生产部署地点。",
        "quotes": ["Mercury production runs in Beijing."],
        **overrides,
    }


@pytest.mark.parametrize(
    "rows",
    [
        [response_row(quotes=["Mercury production runs in Paris."])],
        [response_row(quotes=[])],
        [response_row(), response_row()],
        [response_row(chunk_id=str(UUID(int=2)))],
        [response_row(quotes=["Mercury production runs in Beijing.", "Invented quotation"])],
    ],
)
def test_unsupported_missing_duplicate_and_forged_sources_never_get_promoted(rows):
    result = validate_assessments(json.dumps({"passages": rows}), EVIDENCE)
    assert result[SOURCE_ID].verdict == "unknown"
    assert result[SOURCE_ID].calibrated_probability is None


def test_partial_multi_hop_evidence_is_kept_but_not_promoted_to_full_answer():
    result = validate_assessments(
        json.dumps({"passages": [response_row(verdict="partial")]}), EVIDENCE
    )
    assert result[SOURCE_ID].verdict == "partial"
    assert result[SOURCE_ID].supporting_quotes == ["Mercury production runs in Beijing."]


def test_source_filename_is_sent_but_cannot_supply_a_supporting_quote():
    from knowflow.app.relevance_budget import messages

    rows = [{**EVIDENCE[0], "filename": "audit-retention-17-days.pdf"}]
    payload = json.loads(messages("这份文件说明什么？", rows)[1]["content"])
    assert payload["passages"][0]["filename"] == rows[0]["filename"]
    result = validate_assessments(
        json.dumps({"passages": [response_row(quotes=[rows[0]["filename"]])]}), rows
    )
    assert result[SOURCE_ID].verdict == "unknown"


def test_truncation_cannot_claim_the_support_reached_the_generator():
    assessments = {
        SOURCE_ID: RelevanceAssessment(
            verdict="direct", reason="末尾含支持句。", supporting_quotes=["The limit is 4000."]
        )
    }
    truncated = [{"id": SOURCE_ID, "content": "Long introduction without the answer."}]
    assert retain_supported_context(truncated, assessments) == []
    complete = [{"id": SOURCE_ID, "content": "Long introduction. The limit is 4000."}]
    assert retain_supported_context(complete, assessments) == complete


def test_configured_top_k_maximum_can_be_assessed_without_schema_failure():
    evidence = [
        {"id": str(UUID(int=index)), "content": "A background passage."} for index in range(1, 31)
    ]
    raw = json.dumps(
        {
            "passages": [
                response_row(chunk_id=row["id"], verdict="background", quotes=[])
                for row in evidence
            ]
        }
    )
    result = validate_assessments(raw, evidence)
    assert len(result) == 30
    assert all(row.verdict == "background" for row in result.values())


@pytest.mark.parametrize("raw", ['{"passages":[{}]}', '{"passages":"wrong"}', "not json"])
def test_invalid_provider_schemas_fail_closed(raw):
    with pytest.raises(ValueError, match="Invalid relevance"):
        validate_assessments(raw, EVIDENCE)


@pytest.mark.asyncio
async def test_provider_error_returns_unknown_without_leaking_secrets(monkeypatch):
    async def create(**kwargs):
        raise RuntimeError("secret-key and private source text")

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    result, diagnostics = await assess_retrieval("在哪里部署？", EVIDENCE, Settings())
    assert diagnostics.status == "unavailable"
    assert result[SOURCE_ID].verdict == "unknown"
    assert "secret" not in str(result)


@pytest.mark.asyncio
async def test_budget_omissions_are_unknown_and_judge_is_called_once(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps({"passages": [response_row()]}))
                )
            ]
        )

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    # Only the first source fits; the second cannot be mislabeled as checked.
    settings = Settings(retrieval_assessment_token_budget=75)
    extra_id = str(UUID(int=2))
    result, diagnostics = await assess_retrieval(
        "生产部署地点？", [*EVIDENCE, {"id": extra_id, "content": "Other content " * 100}], settings
    )
    assert len(calls) == 1
    assert result[extra_id].verdict == "unknown"
    assert diagnostics.status == "partial"


@pytest.mark.asyncio
async def test_disabled_check_does_not_call_model(monkeypatch):
    monkeypatch.setattr(relevance, "chat_client", lambda: pytest.fail("Unexpected model call"))
    result, diagnostics = await assess_retrieval(
        "question", EVIDENCE, Settings(retrieval_assessment_enabled=False)
    )
    assert diagnostics.status == "disabled"
    assert result[SOURCE_ID].verdict == "unknown"


@pytest.mark.asyncio
async def test_one_malformed_reply_is_retried_before_failing_closed(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        content = (
            "not json"
            if len(calls) == 1
            else json.dumps({"passages": [response_row()]})
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content=content)
                )
            ]
        )

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    result, diagnostics = await assess_retrieval("生产部署地点？", EVIDENCE, Settings())

    assert len(calls) == 2
    assert diagnostics.status == "complete"
    assert result[SOURCE_ID].verdict != "unknown"


@pytest.mark.asyncio
async def test_two_malformed_replies_still_fail_closed(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="not json")
                )
            ]
        )

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    result, diagnostics = await assess_retrieval("生产部署地点？", EVIDENCE, Settings())

    assert len(calls) == 2
    assert diagnostics.status == "unavailable"
    assert result[SOURCE_ID].verdict == "unknown"


@pytest.mark.asyncio
async def test_one_transient_provider_error_is_retried(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("temporary provider failure")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"passages": [response_row()]})
                    ),
                )
            ]
        )

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    result, diagnostics = await assess_retrieval("生产部署地点？", EVIDENCE, Settings())

    assert len(calls) == 2
    assert diagnostics.status == "complete"
    assert result[SOURCE_ID].verdict != "unknown"


@pytest.mark.asyncio
async def test_two_provider_errors_still_fail_closed(monkeypatch):
    async def create(**kwargs):
        raise RuntimeError("still failing")

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    result, diagnostics = await assess_retrieval("生产部署地点？", EVIDENCE, Settings())

    assert diagnostics.status == "unavailable"
    assert result[SOURCE_ID].verdict == "unknown"


@pytest.mark.asyncio
async def test_long_candidate_lists_are_reviewed_in_batches(monkeypatch):
    calls = []
    evidence = [
        {"id": str(UUID(int=index + 1)), "content": f"Passage {index} about deployment."}
        for index in range(25)
    ]

    async def create(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        rows = [
            {
                "chunk_id": passage["id"],
                "verdict": "background",
                "reason": "fixture",
                "quotes": [],
            }
            for passage in payload["passages"]
        ]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=json.dumps({"passages": rows})),
                )
            ]
        )

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )

    result, diagnostics = await assess_retrieval("部署在哪里？", evidence, Settings())

    assert len(calls) == 3
    assert diagnostics.status == "complete"
    assert all(row.verdict == "background" for row in result.values())


@pytest.mark.asyncio
async def test_exhausted_free_quota_falls_back_to_the_configured_model(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("403 FreeTierOnly: Free quota exhausted")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps({"passages": [response_row()]})
                    ),
                )
            ]
        )

    monkeypatch.setattr(
        relevance,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr(
        relevance,
        "task_route",
        lambda *args, **kwargs: ModelRoute("big-model", "enabled", "test"),
    )
    settings = Settings(llm_model="fallback-model", llm_complex_model="big-model")

    result, diagnostics = await assess_retrieval("生产部署地点？", EVIDENCE, settings)

    assert len(calls) == 2
    assert calls[1]["model"] == "fallback-model"
    assert diagnostics.status == "complete"
    assert diagnostics.context_budget["model_fallback"] == "fallback-model"
    assert result[SOURCE_ID].verdict != "unknown"


def test_quota_errors_are_recognised_only_for_quota_messages():
    from knowflow.app.providers import is_quota_error

    assert is_quota_error(RuntimeError("403 FreeTierOnly: Free quota exhausted")) is True
    assert is_quota_error(RuntimeError("connection reset by peer")) is False


def test_quota_blocked_model_falls_back_for_later_routes():
    from knowflow.app import providers

    settings = Settings(llm_model="fallback-model", llm_complex_model="big-model")
    providers.reset_quota_blocked()
    try:
        assert providers.task_route("relevance", settings, "x" * 4000).model == "big-model"
        providers.mark_quota_blocked("big-model")
        assert providers.task_route("relevance", settings, "x" * 4000).model == "fallback-model"
    finally:
        providers.reset_quota_blocked()


def test_quota_block_expires_so_a_top_up_can_use_the_larger_model_again():
    from knowflow.app import providers

    settings = Settings(llm_model="fallback-model", llm_complex_model="big-model")
    providers.reset_quota_blocked()
    try:
        providers.mark_quota_blocked("big-model", now=0.0)
        assert providers._available("big-model", settings, now=1.0) == "fallback-model"
        later = providers.QUOTA_BLOCK_TTL_SECONDS + 1
        # The block is temporary: an exhausted free tier can be topped up.
        assert providers._available("big-model", settings, now=later) == "big-model"
        assert providers._available("big-model", settings, now=later) == "big-model"
    finally:
        providers.reset_quota_blocked()
