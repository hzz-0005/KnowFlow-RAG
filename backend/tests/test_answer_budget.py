import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import answer_budget as budget
from knowflow.app import api, providers
from knowflow.app.config import Settings
from knowflow.app.intent_context import task_question
from knowflow.app.models import Chunk
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import QueryRequest, RelevanceAssessment, RetrievalDiagnostics


def test_hand_calculated_window_skips_whole_large_sources_then_fits_smaller_ones(monkeypatch):
    class Counter:
        metadata = {"sha256": "synthetic-character-counter"}

        def count(self, text):
            return len(text)

    monkeypatch.setattr(budget, "get_token_counter", lambda *_: Counter())
    monkeypatch.setattr(
        budget,
        "answer_messages",
        lambda q, rows: [
            {"role": "system", "content": "SS"},
            {"role": "user", "content": q + "".join(r["content"] for r in rows)},
        ],
    )
    settings = SimpleNamespace(
        llm_model="test",
        rag_answer_total_token_budget=120,
        rag_answer_output_token_budget=30,
        rag_answer_safety_tokens=10,
        rag_context_token_budget=50,
    )
    candidates = [
        {"id": "large", "content": "a" * 60},
        {"id": "middle", "content": "b" * 30},
        {"id": "window", "content": "c" * 20},
        {"id": "small", "content": "d" * 18},
        {"id": "duplicate", "content": "d" * 18},
        {"id": "empty", "content": "  "},
    ]
    selected, trace = budget.plan_answer_context("Q", candidates, settings)
    assert [r["id"] for r in selected] == ["middle", "small"]
    assert selected == [candidates[1], candidates[3]]  # no prefix cut or text rewrite
    assert trace["fixed_input_tokens_estimate"] == 32
    assert trace["input_tokens_estimate"] == 80
    assert trace["estimated_total_tokens"] == 120
    assert [r["reason"] for r in trace["decisions"]] == [
        "evidence_cap",
        None,
        "window_budget",
        None,
        "duplicate",
        "empty",
    ]


@pytest.mark.parametrize(
    "model,url,parameter,scope",
    [
        (
            "qwen3.8-max",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "max_completion_tokens",
            "reasoning_and_answer",
        ),
        (
            "qwen3.7-flash",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "max_completion_tokens",
            "reasoning_and_answer",
        ),
        (
            "mimo-v2.6-pro",
            "https://api.xiaomimimo.com/v1",
            "max_completion_tokens",
            "reasoning_and_answer",
        ),
        ("deepseek-flash", "https://api.deepseek.com", "max_tokens", "reasoning_and_answer"),
        ("qwen3.8-max", "https://unverified.example/v1", "max_tokens", "unverified"),
    ],
)
def test_output_parameter_scope_is_provider_specific_not_just_model_name(
    model, url, parameter, scope
):
    settings = Settings(_env_file=None, llm_model=model, llm_base_url=url)
    trace = budget.measure_request("Q", [], settings)
    assert trace["output_limit_parameter"] == parameter
    assert trace["output_limit_scope"] == scope
    assert trace["provider_tokenizer_match"] == "unverified"
    assert trace["provider_usage"] is None


def test_mandatory_user_intent_is_not_shortened_to_fit():
    settings = Settings(
        _env_file=None,
        rag_answer_total_token_budget=1024,
        rag_answer_output_token_budget=512,
        rag_answer_safety_tokens=128,
    )
    original = "仅测试环境，不要生产环境；使用2025，不是2024。" * 100
    question = task_question(
        "a bad short hint",
        {
            "current_user_input": original,
            "referenced_user_inputs": [{"turn_id": "prior", "question": "never archived"}],
        },
    )
    selected, trace = budget.plan_answer_context(
        question, [{"id": "source", "content": "tiny"}], settings
    )
    assert not selected and not trace["estimated_feasible"]
    assert trace["decisions"][0]["reason"] == "fixed_input_exhausted"
    assert trace["question_sha256"] == hashlib.sha256(question.encode()).hexdigest()
    assert original in question and "never archived" in question


def test_case_and_indentation_are_not_collapsed_as_duplicate_source_text():
    rows = [
        {"id": "A", "content": "Limit=1"},
        {"id": "B", "content": "limit=1"},
        {"id": "C", "content": "    limit=1"},
    ]
    selected, trace = budget.plan_answer_context("Which variable?", rows, Settings(_env_file=None))
    assert selected == rows and trace["selected_count"] == 3


def test_new_quality_reports_include_budget_contract_and_do_not_guess_old_defaults(monkeypatch):
    from knowflow.app import evaluate_quality
    from knowflow.app.quality_metrics import regression_check

    settings = Settings(_env_file=None)
    monkeypatch.setattr(evaluate_quality, "get_settings", lambda: settings)
    report = evaluate_quality.initial_report(
        uuid4(), {"version": "control", "cases": []}, ["keyword"], 5, False
    )
    assert report["answer_budget_contract"] == budget.budget_contract(settings)
    old = dict(report)
    old.pop("answer_budget_contract")
    assert regression_check(report, old)["status"] == "not_comparable"


def test_budget_changes_are_not_reported_as_same_budget_quality_regressions():
    from knowflow.app.quality_metrics import regression_check

    contract = budget.budget_contract(Settings(_env_file=None))
    left = {"answer_budget_contract": contract, "variants": {}}
    right = {"answer_budget_contract": {**contract, "output_tokens": 4096}, "variants": {}}
    assert regression_check(left, right)["status"] == "not_comparable"


@pytest.mark.asyncio
async def test_provider_request_matches_budgeted_prompt_and_records_actual_usage(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            model="reported-alias-not-a-pin",
            usage=SimpleNamespace(
                model_dump=lambda: {
                    "prompt_tokens": 123,
                    "completion_tokens": 45,
                    "total_tokens": 168,
                }
            ),
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="Answer [KF1]")
                )
            ],
        )

    monkeypatch.setattr(
        providers,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    settings = Settings(
        _env_file=None,
        llm_model="qwen3.8-flash",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    evidence = [{"id": str(uuid4()), "content": "条件完整保留，不把未说明当允许。"}]
    trace = {}
    answer = await providers.generate_answer("请说明条件", evidence, settings, budget_trace=trace)
    assert str(evidence[0]["id"]) in answer and len(calls) == 1
    assert calls[0]["messages"] == providers.answer_messages("请说明条件", evidence)
    assert (
        trace["prompt_sha256"]
        == hashlib.sha256(
            json.dumps(calls[0]["messages"], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )
    assert calls[0]["max_completion_tokens"] == settings.rag_answer_output_token_budget
    assert "max_tokens" not in calls[0]
    assert trace["provider_usage"]["prompt_tokens"] == 123


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["partial answer [KF1]", ""])
async def test_truncated_output_is_not_returned_or_silently_retried(monkeypatch, content):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="length", message=SimpleNamespace(content=content))
            ]
        )

    monkeypatch.setattr(
        providers,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    trace = {}
    with pytest.raises(budget.AnswerOutputLimit):
        await providers.generate_answer(
            "Q",
            [{"id": "source", "content": "evidence"}],
            Settings(_env_file=None),
            budget_trace=trace,
        )
    assert len(calls) == 1 and trace["status"] == "output_limit_reached"


@pytest.mark.asyncio
async def test_fixed_budget_preflight_stops_before_paid_retrieval(monkeypatch):
    async def owned(*_):
        pass

    async def forbidden(*_, **kwargs):
        raise AssertionError("No paid stage should run")

    settings = Settings(
        _env_file=None,
        rag_answer_total_token_budget=1024,
        rag_answer_output_token_budget=512,
        rag_answer_safety_tokens=128,
    )
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "plan_query", forbidden)
    monkeypatch.setattr(api, "retrieve", forbidden)
    result = await api.run_query(
        uuid4(),
        QueryRequest(question="条件" * 1000, strategy="hybrid"),
        SimpleNamespace(id=uuid4()),
        None,
    )
    assert result.refused and result.context_budget["status"] == "blocked"
    assert not result.citations


@pytest.mark.asyncio
async def test_pipeline_output_limit_keeps_trace_and_does_not_audit_partial_answer(monkeypatch):
    kb, doc, cid = uuid4(), uuid4(), uuid4()
    chunk = Chunk(
        id=cid,
        document_id=doc,
        knowledge_base_id=kb,
        ordinal=0,
        content="Complete supporting evidence.",
        source_spans=[],
        page_number=None,
        section=None,
    )

    async def owned(*_):
        pass

    async def retrieve(*_, **kwargs):
        return [RetrievalHit(chunk, "control.txt")]

    async def assess(_question, rows, *_):
        return {
            r["id"]: RelevanceAssessment(
                verdict="direct", reason="control", supporting_quotes=[r["content"]]
            )
            for r in rows
        }, RetrievalDiagnostics(status="complete")

    async def truncated(*_, **kwargs):
        raise budget.AnswerOutputLimit

    async def forbidden(*_, **kwargs):
        raise AssertionError("Partial draft must not reach final answer guard")

    class Session:
        async def scalars(self, *_):
            return []

    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "retrieve", retrieve)
    monkeypatch.setattr(api, "assess_retrieval", assess)
    monkeypatch.setattr(api, "generate_answer", truncated)
    monkeypatch.setattr(api, "guard_answer", forbidden)
    result = await api.run_query(
        kb,
        QueryRequest(question="Q", strategy="keyword", expand_references=False),
        SimpleNamespace(id=uuid4()),
        Session(),
    )
    assert result.refused and not result.citations
    assert result.context_budget["status"] == "output_limit_reached"
    assert result.retrieval_evidence[0].included_in_context
