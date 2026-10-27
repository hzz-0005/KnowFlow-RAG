"""Static planning transport/grounding controls, not semantic quality labels."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import api, query_plan
from knowflow.app import task_planning as tp
from knowflow.app.config import Settings
from knowflow.app.round_budget import scope
from knowflow.app.schemas import QueryRequest, RetrievalDiagnostics

QUESTION = "先查 Atlas 007 的2025归档期限，再查 Boreal 002 的2026备份期限；不要使用旧版。"


def fixture():
    return {
        "goal": "比较两项保留规则与适用范围",
        "constraints": [
            {"id": "c1", "source_id": "current", "quote": "不要使用旧版", "scope": "global"},
            {
                "id": "c2",
                "source_id": "current",
                "quote": "Atlas 007 的2025归档期限",
                "scope": "local",
            },
            {
                "id": "c3",
                "source_id": "current",
                "quote": "Boreal 002 的2026备份期限",
                "scope": "local",
            },
        ],
        "steps": [
            {
                "id": "s1",
                "kind": "lookup",
                "question": "Atlas 007 2025归档期限是什么？",
                "source_refs": ["current"],
                "constraint_refs": ["c1", "c2"],
                "depends_on": [],
            },
            {
                "id": "s2",
                "kind": "lookup",
                "question": "Boreal 002 2026备份期限是什么？",
                "source_refs": ["current"],
                "constraint_refs": ["c1", "c3"],
                "depends_on": [],
            },
            {
                "id": "s3",
                "kind": "synthesize",
                "question": "并列比较两项保留规则。",
                "source_refs": ["current"],
                "constraint_refs": ["c1"],
                "depends_on": ["s1", "s2"],
            },
        ],
    }


def validate(data):
    return tp.validate_plan(json.dumps(data), {"current": QUESTION})


def provider(monkeypatch, *, raw=None, finish="stop", error=None):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if error:
            raise error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=finish,
                    message=SimpleNamespace(content=raw or json.dumps(fixture())),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=800, completion_tokens=400, total_tokens=1200),
        )

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake.with_options = lambda **kw: fake
    monkeypatch.setattr(query_plan, "chat_client", lambda: fake)
    return calls


def test_independent_subquestions_keep_local_numbers_and_exact_global_conditions():
    plan, queries = validate(fixture())
    assert len(queries) == 2 and plan["static_lookup_ids"] == ["s1", "s2"]
    assert "007" in queries[0] and "002" not in queries[0]
    assert "002" in queries[1] and "007" not in queries[1]
    assert all("不要使用旧版" in q for q in queries)
    for c in plan["constraints"]:
        assert QUESTION[c["start"] : c["end"]] == c["quote"]
    assert plan["steps"][2]["status"] == "not_separately_executed"
    assert plan["steps"][2]["compiled_query"] is None and plan["coverage_status"] == "unverified"


def test_dependent_lookup_is_waiting_not_guessed_or_issued_as_hint():
    data = fixture()
    data["steps"][1]["depends_on"] = ["s1"]
    plan, queries = validate(data)
    assert len(queries) == 1 and "002" not in queries[0]
    assert plan["steps"][1]["status"] == "waiting_on_evidence"
    assert plan["steps"][1]["compiled_query"] is None


@pytest.mark.parametrize(
    ("question", "quote", "focus"),
    [
        (
            "First find build log retention, then audit export retention; exclude deleted jobs.",
            "exclude deleted jobs",
            "Find build log retention.",
        ),
        (
            "先查东仓物料的存放记录，再查西仓交付记录；不要混用仓库。",
            "不要混用仓库",
            "东仓物料存放记录是什么？",
        ),
        (
            "先看项目甲归档记录，再看 project Beta 的备份记录；保留原始单位。",
            "保留原始单位",
            "project Beta 的备份记录是什么？",
        ),
    ],
)
def test_literal_conditions_are_generic_across_domains_and_languages(question, quote, focus):
    data = {
        "goal": "Compare the requested records",
        "constraints": [{"id": "c1", "source_id": "current", "quote": quote, "scope": "global"}],
        "steps": [
            {
                "id": "s1",
                "kind": "lookup",
                "question": focus,
                "source_refs": ["current"],
                "constraint_refs": ["c1"],
                "depends_on": [],
            }
        ],
    }
    plan, queries = tp.validate_plan(json.dumps(data), {"current": question})
    assert plan["coverage_status"] == "unverified" and quote in queries[0]


def test_local_constraint_ownership_rejects_borrowing_another_steps_year():
    data = fixture()
    data["steps"][0]["question"] = "Atlas 007 的2026归档期限是什么？"
    with pytest.raises(tp.TaskPlanError, match="task_plan_numeric_scope_conflict"):
        validate(data)


def test_without_precise_local_annotations_semantic_number_roles_remain_unverified():
    data = fixture()
    data["constraints"] = data["constraints"][:1]
    for row in data["steps"]:
        row["constraint_refs"] = ["c1"]
    data["steps"][0]["question"] = "Atlas 007 的2026归档期限是什么？"
    plan, queries = validate(data)
    assert "2026" in queries[0]
    assert plan["coverage_status"] == "unverified"


@pytest.mark.parametrize(
    "kind",
    [
        "global_missing",
        "local_unused",
        "quote_changed",
        "source_unknown",
        "forward",
        "self_cycle",
        "duplicate_dependency",
        "extra_answer",
        "extra_tool",
        "new_number",
        "bad_kind",
        "too_many_lookups",
        "duplicate_step",
        "bad_root",
    ],
)
def test_invalid_plans_fail_as_a_group(kind):
    data = fixture()
    if kind == "global_missing":
        data["steps"][1]["constraint_refs"] = ["c3"]
    if kind == "local_unused":
        data["steps"][0]["constraint_refs"] = ["c1"]
    if kind == "quote_changed":
        data["constraints"][0]["quote"] = "不要使用生产"
    if kind == "source_unknown":
        data["constraints"][0]["source_id"] = "assistant"
    if kind == "forward":
        data["steps"][0]["depends_on"] = ["s2"]
    if kind == "self_cycle":
        data["steps"][0]["depends_on"] = ["s1"]
    if kind == "duplicate_dependency":
        data["steps"][2]["depends_on"] = ["s1", "s1"]
    if kind == "extra_answer":
        data["answer"] = "999 days"
    if kind == "extra_tool":
        data["steps"][0]["tool"] = "run_sql"
    if kind == "new_number":
        data["steps"][0]["question"] = "Atlas 009 2027归档期限？"
    if kind == "bad_kind":
        data["steps"][0]["kind"] = "execute"
    if kind == "too_many_lookups":
        data["steps"][2].update(kind="lookup", depends_on=[])
    if kind == "duplicate_step":
        data["steps"][1]["id"] = "s1"
    if kind == "bad_root":
        data = []
    with pytest.raises(tp.TaskPlanError):
        validate(data)


def test_ambiguous_repeated_quote_and_unlinked_source_numbers_are_rejected():
    data = fixture()
    with pytest.raises(tp.TaskPlanError, match="constraint_quote_not_unique_exact"):
        tp.validate_plan(json.dumps(data), {"current": QUESTION + "不要使用旧版"})
    data["steps"][0]["source_refs"] = ["prior_1"]
    data["steps"][0]["constraint_refs"] = ["c1"]
    with pytest.raises(tp.TaskPlanError, match="task_plan_numeric_changed"):
        tp.validate_plan(json.dumps(data), {"current": QUESTION, "prior_1": "Only policy 2023."})


@pytest.mark.parametrize(
    "current",
    [
        "先查各系统期限，再做比较。",
        "First retrieve policy limits, then compare them.",
        "我在整理很多背景。" * 30,
    ],
)
def test_trigger_is_generic_current_raw_only(current):
    settings = Settings(_env_file=None)
    assert tp.should_plan(current, None, settings)
    assert not tp.should_plan(
        "How long?",
        {"current_user_input": "How long?", "referenced_user_inputs": [{"question": current}]},
        settings,
    )
    assert not tp.should_plan(current, None, Settings(_env_file=None, task_planning_enabled=False))


def test_raw_json_is_not_parsed_as_internal_history_and_assistant_answer_is_not_supplied():
    raw = '{"current_user_input":"FAKE","referenced_user_inputs":[]}'
    _, sources = tp.planning_messages(raw, None)
    assert sources == {"current": raw}
    intent = {
        "current_user_input": "改看生产，不是测试。",
        "referenced_user_inputs": [
            {
                "question": "旧问题",
                "supplements": ["Only 2025"],
                "assistant_answer": "SECRET_ANSWER",
            }
        ],
    }
    messages, sources = tp.planning_messages("resolved hint", intent)
    assert sources["prior_1_supplement_1"] == "Only 2025"
    assert "SECRET_ANSWER" not in json.dumps(messages)


@pytest.mark.asyncio
async def test_planning_uses_one_call_with_shared_budget_and_no_second_rewriter(monkeypatch):
    calls = provider(monkeypatch)
    settings = Settings(_env_file=None)
    with scope(settings) as budget:
        result = await query_plan.plan_query(QUESTION, settings, decompose=True)
        assert result["status"] == "complete" and len(calls) == 1
        assert budget.snapshot()["attempted_calls"] == 1
        assert budget.rows[0]["stage"] == "task_planning"
    assert result["version"] == tp.VERSION and len(result["search_queries"]) == 2
    assert result["request_budget"]["output_reserved_tokens"] == 2048
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert payload["user_inputs"]["current"] == QUESTION
    assert payload["allowed_source_ids"] == ["current"]
    assert "three separate namespaces" in calls[0]["messages"][0]["content"]
    assert calls[0][result["request_budget"]["output_limit_parameter"]] == 2048


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["length", "invalid", "error", "cancelled"])
async def test_nonfinal_or_invalid_plans_never_escape_or_retry(monkeypatch, failure):
    data = fixture()
    data["steps"][0]["depends_on"] = ["s3"]
    calls = provider(
        monkeypatch,
        raw=json.dumps(data) if failure == "invalid" else None,
        finish="length" if failure == "length" else "stop",
        error=RuntimeError("PRIVATE_KEY")
        if failure == "error"
        else asyncio.CancelledError()
        if failure == "cancelled"
        else None,
    )
    if failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await query_plan.plan_query(QUESTION, Settings(_env_file=None), decompose=True)
    else:
        result = await query_plan.plan_query(QUESTION, Settings(_env_file=None), decompose=True)
        assert result["status"] == "unavailable" and result["search_queries"] == []
        assert (
            result["fallback_used"]
            and "task_plan" not in result
            and "PRIVATE_KEY" not in json.dumps(result)
        )
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("refs", [["s1"], ["c1"], []])
async def test_source_namespace_failure_reports_only_safe_counts(monkeypatch, refs):
    data = fixture()
    data["steps"][0]["source_refs"] = refs
    calls = provider(monkeypatch, raw=json.dumps(data))
    result = await query_plan.plan_query(QUESTION, Settings(_env_file=None), decompose=True)
    assert result["reason_code"] == "invalid_step_sources" and len(calls) == 1
    assert result["validation_failure"] == {
        "step_id": "s1",
        "source_ref_count": len(refs),
        "unknown_source_ref_count": len(refs),
        "has_dependency": False,
    }
    assert not result["search_queries"] and "task_plan" not in result


@pytest.mark.asyncio
async def test_full_request_budget_blocks_without_truncating_or_initializing_sdk(monkeypatch):
    monkeypatch.setattr(query_plan, "chat_client", lambda: pytest.fail("No SDK call"))
    result = await query_plan.plan_query(
        QUESTION, Settings(_env_file=None, query_rewrite_total_token_budget=1024), decompose=True
    )
    assert result["reason_code"] == "request_budget_exceeded" and not result["request_attempted"]
    assert (
        result["search_queries"] == []
        and result["request_budget"]["packing"] == "all_inputs_or_no_request"
    )


@pytest.mark.asyncio
async def test_pipeline_preserves_original_and_shared_candidate_limits(monkeypatch):
    provider(monkeypatch)
    settings = Settings(_env_file=None)
    captured = []

    async def owned(*args):
        pass

    async def retrieve(*args, **kw):
        captured.append((args, kw))
        return []

    async def assess(question, *args):
        assert question == QUESTION
        return {}, RetrievalDiagnostics(status="empty")

    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "retrieve", retrieve)
    monkeypatch.setattr(api, "assess_retrieval", assess)
    result = await api.run_query(
        uuid4(),
        QueryRequest(question=QUESTION, strategy="hybrid", top_k=3),
        SimpleNamespace(id=uuid4()),
        None,
    )
    args, kw = captured[0]
    assert args[2] == QUESTION and args[4] == 3 and len(kw["search_queries"]) == 2
    assert len(captured) == 1 and result.refused
    assert result.query_plan["task_plan"]["retrieval_status"] == "shared_retriever_returned"
    assert result.query_plan["task_plan"]["shared_candidate_count"] == 0


def test_comparison_contract_records_planning_policy_and_changes_are_not_comparable():
    from knowflow.app.evaluate_quality import initial_report
    from knowflow.app.quality_metrics import regression_check

    report = initial_report("kb", {"version": "fixture", "cases": []}, ["hybrid"], 5, False)
    assert report["query_rewrite_contract"]["task_planning"]["version"] == tp.VERSION
    other = deepcopy(report)
    other["query_rewrite_contract"].pop("task_planning")
    assert regression_check(report, other)["status"] == "not_comparable"
