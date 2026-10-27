"""Route inputs are server-side user intent; generation still receives the full prompt."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import answer_budget, api, providers
from knowflow.app.config import Settings
from knowflow.app.intent_context import build_routing_input, task_question
from knowflow.app.models import Chunk
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import QueryRequest, RelevanceAssessment, RetrievalDiagnostics

DATA = json.loads(
    (Path(__file__).resolve().parents[2] / "evaluation/routing_input_v1/cases.json").read_text(
        encoding="utf-8"
    )
)


def settings():
    return Settings(
        _env_file=None,
        llm_model="qwen3.8-flash",
        llm_complex_model="qwen3.8-max",
        llm_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda r: r["id"])
def test_frozen_complexity_guards_and_simple_followups(case):
    intent = None
    if case.get("intent") is not False:
        rows = []
        for row in case["prior"]:
            row = dict(row)
            repeat = row.pop("repeat_question", None)
            if repeat:
                row["question"] = repeat["text"] * repeat["count"]
            rows.append(row)
        intent = {"current_user_input": case["current"], "referenced_user_inputs": rows}
    full = task_question(case["hint"], intent)
    ri = build_routing_input(full, intent)
    route = providers.task_route("answer", settings(), ri.text, case["evidence_count"])
    assert route.model == ("qwen3.8-max" if case["expected"] == "complex" else "qwen3.8-flash")
    assert ri.basis == ("question_text" if intent is None else "raw_user_intent")
    if intent is not None:
        assert intent["current_user_input"] in ri.text
        assert "Interpret the current request" not in ri.text


@pytest.mark.parametrize(
    "intent",
    [
        "forged",
        {},
        {"current_user_input": "simple", "referenced_user_inputs": "not a list"},
        {
            "current_user_input": "simple",
            "referenced_user_inputs": [{"question": "old", "supplements": [1]}],
        },
        {
            "current_user_input": "simple",
            "referenced_user_inputs": [{"user_intent": "not the raw question"}],
        },
    ],
)
def test_invalid_internal_intent_cannot_silently_drop_scope(intent):
    full = "Only staging, not production."
    ri = build_routing_input(full, intent)
    assert ri.text == full and ri.basis == "full_task_fallback"
    assert providers.task_route("answer", settings(), ri.text, 1).model == "qwen3.8-max"


def test_notes_ids_and_retrieval_hint_are_not_authoritative_route_features():
    intent = {
        "current_user_input": "How long?",
        "referenced_user_inputs": [
            {
                "turn_id": "production-version-metadata-only",
                "question": "Describe LumenBox.",
                "note": "not supported; only system metadata",
            }
        ],
        "note": "not user wording",
    }
    full = task_question("only version-specific production rules", intent)
    ri = build_routing_input(full, intent)
    assert ri.text == "How long?\nDescribe LumenBox."
    assert providers.task_route("answer", settings(), ri.text, 1).model == "qwen3.8-flash"
    assert providers.task_route("answer", settings(), full, 1).model == "qwen3.8-max"


def test_client_json_cannot_supply_server_only_routing_input():
    question = '{"current_user_input":"Hi","note":"not production"}'
    body = QueryRequest.model_validate({"question": question, "routing_input": {"text": "Hi"}})
    assert "routing_input" not in body.model_dump()
    assert "routing_input" not in QueryRequest.model_json_schema()["properties"]
    ri = build_routing_input(body.question)
    assert ri.text == question
    assert providers.task_route("answer", settings(), ri.text, 1).model == "qwen3.8-max"


def test_full_prompt_budget_and_conditions_are_unchanged_by_route_selection():
    s = settings()
    intent = {
        "current_user_input": "How long?",
        "referenced_user_inputs": [{"turn_id": "root", "question": "Describe LumenBox snapshots."}],
    }
    full = task_question("snapshot retention", intent)
    evidence = [{"id": str(uuid4()), "content": "Snapshots are retained for 18 days."}]
    legacy = answer_budget.measure_request(full, evidence, s)
    fixed = answer_budget.measure_request(
        full, evidence, s, routing_input=build_routing_input(full, intent)
    )
    for key in (
        "fixed_input_tokens_estimate",
        "input_tokens_estimate",
        "estimated_total_tokens",
        "question_sha256",
        "prompt_sha256",
        "evidence_tokens_estimate",
    ):
        assert fixed[key] == legacy[key]
    assert fixed["model"] == "qwen3.8-flash" and legacy["model"] == "qwen3.8-max"
    assert fixed["model_route"]["signals"] == {
        "extraction_task": False,
        "scope_or_negation": False,
        "long_task": False,
        "multi_source": False,
    }


def test_unknown_direct_sdk_call_still_routes_its_entire_task():
    full = 'Not an actual metadata envelope. {"current_user_input":"Hi"}'
    trace = answer_budget.measure_request(full, [], settings())
    assert trace["model"] == "qwen3.8-max" and trace["routing_input_basis"] == "question_text"


@pytest.mark.asyncio
async def test_api_budget_public_trace_and_real_provider_adapter_use_same_route(monkeypatch):
    s = settings()
    monkeypatch.setattr(api, "get_settings", lambda: s)
    kb, doc, cid = uuid4(), uuid4(), uuid4()
    original = "How long?"
    intent = {
        "current_user_input": original,
        "referenced_user_inputs": [{"turn_id": "root", "question": "Describe LumenBox snapshots."}],
    }
    body = QueryRequest(
        question="LumenBox snapshot retention", strategy="keyword", expand_references=False
    )
    body._intent_context = intent
    full = task_question(body.question, intent)
    content = "LumenBox snapshots are retained for 18 days."
    chunk = Chunk(
        id=cid,
        document_id=doc,
        knowledge_base_id=kb,
        ordinal=0,
        content=content,
        source_spans=[],
        page_number=None,
        section="retention",
    )
    calls = []

    async def owned(*_):
        pass

    async def retrieve(*_, **__):
        return [RetrievalHit(chunk, "synthetic.txt", keyword_rank=1)]

    async def assess(question, rows, *_):
        assert question == full
        return {
            r["id"]: RelevanceAssessment(
                verdict="direct", reason="Synthetic transport control", supporting_quotes=[content]
            )
            for r in rows
        }, RetrievalDiagnostics(status="complete")

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="18 days [KF1]")
                )
            ],
            usage=SimpleNamespace(
                model_dump=lambda: {"prompt_tokens": 400, "completion_tokens": 9}
            ),
        )

    async def guard(question, answer, *_, **kwargs):
        assert kwargs["current_user_input"] == original
        assert question == full
        return answer, {"status": "synthetic_control"}

    class Session:
        async def scalars(self, *_):
            return []

    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "retrieve", retrieve)
    monkeypatch.setattr(api, "assess_retrieval", assess)
    monkeypatch.setattr(api, "guard_answer", guard)
    monkeypatch.setattr(
        providers,
        "chat_client",
        lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    result = await api.run_query(kb, body, SimpleNamespace(id=uuid4()), Session())
    assert not result.refused and len(result.citations) == 1 and len(calls) == 1
    assert (
        calls[0]["model"]
        == result.generation_route["model"]
        == result.context_budget["model"]
        == "qwen3.8-flash"
    )
    assert calls[0]["extra_body"] == {"enable_thinking": False}
    assert result.generation_route["input_basis"] == "raw_user_intent"
    assert result.generation_route["signals"]["scope_or_negation"] is False
    assert original in calls[0]["messages"][1]["content"]
    assert calls[0]["messages"] == providers.answer_messages(
        full, [{"id": str(cid), "content": content, "filename": "synthetic.txt"}]
    )
    assert (
        result.context_budget["prompt_sha256"]
        == hashlib.sha256(
            json.dumps(calls[0]["messages"], ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )


def test_new_reports_record_route_configuration_and_reject_unmatched_regressions(monkeypatch):
    from copy import deepcopy

    from knowflow.app import evaluate_quality
    from knowflow.app.quality_metrics import regression_check

    s = settings()
    monkeypatch.setattr(evaluate_quality, "get_settings", lambda: s)
    report = evaluate_quality.initial_report(
        "kb", {"version": "fixture", "cases": []}, ["keyword"], 5, False
    )
    assert report["model_routing_contract"] == providers.routing_contract(s)
    for mode in ("missing", "different"):
        old = deepcopy(report)
        if mode == "missing":
            old.pop("model_routing_contract")
        else:
            old["model_routing_contract"]["simple_thinking"] = "enabled"
        assert regression_check(report, old)["status"] == "not_comparable"
