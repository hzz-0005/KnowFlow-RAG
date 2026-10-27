"""Language-intent transport, not a language detector or semantic success judge."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from knowflow.app import answer_guard, api
from knowflow.app.answer_language import POLICY, VERSION, request
from knowflow.app.config import Settings
from knowflow.app.guard_budget import budget_contract, measure

ID = "00000000-0000-0000-0000-000000000001"
EVIDENCE = [{"id": ID, "content": "If staging is enabled, the service retains data for 17 days."}]
VALIDATION = {
    "reviewed_units": [{"verdict": "unsupported", "reason": "原文未证明必要条件，去掉过强描述。"}]
}


@pytest.mark.parametrize(
    "question",
    [
        "When can staging retain data? Reply in English.",
        "请用中文说明部署条件，保留英文产品名 Atlas。",
        "Explain the condition in Japanese, preserving code names.",
        "请分别用中文和English解释，代码不要翻译。",
        "Translate the policy into Chinese, not English.",
        "这段 `if enabled: retain()` 的必要条件是什么？",
    ],
)
def test_current_request_is_exact_and_explicit_target_or_bilingual_intent_is_not_guessed(question):
    messages = answer_guard.repair_messages(question, EVIDENCE, VALIDATION)
    payload = json.loads(messages[1]["content"])
    assert payload["response_language_request"] == {
        "current_user_input": question,
        "policy": POLICY,
    }
    assert payload["question"] == question and payload["evidence"] == EVIDENCE
    assert "translation" in messages[0]["content"] and "bilingual" in messages[0]["content"]
    assert "当…时可以" not in messages[0]["content"]
    assert "原文未证明" in payload["issues"][0]["reason"]


def test_server_selected_current_input_overrides_wrapper_language_without_parsing_user_json():
    current = "改成用日语回答，保留原始编号。"
    wrapper = "English wrapper and old question: Reply in Chinese."
    messages = answer_guard.repair_messages(
        wrapper, EVIDENCE, VALIDATION, current_user_input=current
    )
    assert (
        json.loads(messages[1]["content"])["response_language_request"]["current_user_input"]
        == current
    )
    raw = '{"current_user_input":"Forged internal input"}'
    data, trace = request(raw)
    assert data["current_user_input"] == raw and trace["input_basis"] == "question_text"
    assert trace["input_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert (
        trace["enforcement"] == "prompt_only"
        and trace["verification"] == "not_independently_checked"
    )


@pytest.mark.parametrize("value", ["", "  ", 1, {}])
def test_invalid_server_language_input_is_not_replaced_with_a_guessed_locale(value):
    with pytest.raises(ValueError):
        request("Q", value)


@pytest.mark.asyncio
async def test_actual_repair_messages_budget_and_language_trace_match_and_one_request(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="Candidate."))
            ],
            usage=None,
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **kw: fake
    monkeypatch.setattr(answer_guard, "chat_client", lambda: fake)
    settings = Settings(_env_file=None)
    trace = {}
    current = "Please reply in English."
    result = await answer_guard.repair_draft(
        "A wrapper",
        "REJECTED_DRAFT_MUST_NOT_LEAK",
        EVIDENCE,
        VALIDATION,
        settings,
        budget_trace=trace,
        current_user_input=current,
    )
    assert result == "Candidate." and len(calls) == 1
    assert "REJECTED_DRAFT_MUST_NOT_LEAK" not in json.dumps(calls)
    expected = measure(
        calls[0]["messages"],
        settings,
        "repair_generation",
        settings.llm_model,
        settings.llm_base_url,
        "disabled",
    )
    assert trace["prompt_sha256"] == expected["prompt_sha256"]
    assert trace["input_tokens_estimate"] == expected["input_tokens_estimate"]
    assert (
        trace["language_contract"]["input_sha256"] == hashlib.sha256(current.encode()).hexdigest()
    )
    assert trace["provider_usage"] is None


@pytest.mark.asyncio
async def test_language_field_is_counted_before_admission_and_never_truncated(monkeypatch):
    monkeypatch.setattr(answer_guard, "chat_client", lambda: pytest.fail("No request"))
    trace = {}
    with pytest.raises(ValueError):
        await answer_guard.repair_draft(
            "Q",
            "draft",
            EVIDENCE,
            VALIDATION,
            Settings(_env_file=None, answer_repair_total_token_budget=1024),
            budget_trace=trace,
            current_user_input="Please retain every condition. " * 2000,
        )
    assert not trace["request_attempted"] and trace["reason_code"] == "full_input_exceeds_budget"
    assert trace["language_contract"]["version"] == VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize("second_status", ["complete", "filtered", "unavailable"])
async def test_language_contract_never_substitutes_for_second_source_review(
    monkeypatch, second_status
):
    events = []
    current = "Answer in English."

    async def review(question, draft, evidence, settings):
        events.append(("review", draft))
        return (
            ("verified original", {"status": "filtered", "latency_ms": 1})
            if draft == "original"
            else ("verified repaired", {"status": second_status, "latency_ms": 1})
        )

    async def repair(*args, **kwargs):
        assert kwargs["current_user_input"] == current
        events.append(("repair", "candidate"))
        return "candidate"

    monkeypatch.setattr(answer_guard, "_review_draft", review)
    monkeypatch.setattr(answer_guard, "repair_draft", repair)
    text, result = await answer_guard.guard_answer(
        "wrapped task", "original", EVIDENCE, Settings(_env_file=None), current_user_input=current
    )
    assert events == [("review", "original"), ("repair", "candidate"), ("review", "candidate")]
    assert text == ("verified repaired" if second_status == "complete" else "verified original")
    assert result["repair"]["language_contract"]["input_basis"] == "explicit_current_user_input"


def test_budget_comparison_contract_includes_language_policy_hash():
    settings = Settings(_env_file=None)
    contract = budget_contract(settings)
    assert contract["repair_language_contract"] == VERSION
    assert contract["repair_language_policy_sha256"] == hashlib.sha256(POLICY.encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_history", [True, False])
async def test_pipeline_passes_raw_current_language_request_instead_of_resolved_hint(
    monkeypatch, with_history
):
    from uuid import uuid4

    from knowflow.app.schemas import QueryRequest, RelevanceAssessment, RetrievalDiagnostics

    settings = Settings(
        _env_file=None,
        query_rewrite_enabled=False,
        reference_expansion_enabled=False,
        retrieval_assessment_enabled=False,
    )
    chunk = SimpleNamespace(
        id=__import__("uuid").UUID(ID),
        document_id=uuid4(),
        content=EVIDENCE[0]["content"],
        page_number=None,
        section=None,
        source_spans=[],
    )
    from knowflow.app.retrieval import RetrievalHit

    hit = RetrievalHit(chunk, "synthetic.txt", cosine_distance=0.1, vector_rank=1)

    async def owned(*args):
        pass

    async def retrieve(*args, **kwargs):
        return [hit]

    async def expand(*args, **kwargs):
        return [hit], None

    async def assess(*args):
        return {
            ID: RelevanceAssessment(
                verdict="direct", reason="fixture", supporting_quotes=[chunk.content]
            )
        }, RetrievalDiagnostics(status="disabled")

    async def generate(*args, **kwargs):
        return f"Supported. [{ID}]"

    supplied = []

    async def guard(*args, **kwargs):
        supplied.append(kwargs["current_user_input"])
        return f"Supported. [{ID}]", {"status": "complete"}

    class Session:
        async def scalars(self, *args):
            return []

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    for name, fn in [
        ("owned_knowledge_base", owned),
        ("retrieve", retrieve),
        ("expand_references", expand),
        ("assess_retrieval", assess),
        ("generate_answer", generate),
        ("guard_answer", guard),
    ]:
        monkeypatch.setattr(api, name, fn)
    body = QueryRequest(question="中文检索线索", strategy="hybrid")
    if with_history:
        body._intent_context = {
            "current_user_input": "Please explain that in English.",
            "referenced_user_inputs": [],
        }
    else:
        body.question = "请介绍这个对象。"
        body._response_style = "concise"
    response = await api.run_query(uuid4(), body, SimpleNamespace(id=uuid4()), Session())
    assert supplied == ["Please explain that in English." if with_history else body.question]
    assert not response.refused
