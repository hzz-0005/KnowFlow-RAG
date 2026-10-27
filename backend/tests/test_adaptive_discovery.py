"""Controls for routing/grounding; these are not semantic or held-out accuracy labels."""

import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import conversation_context as context
from knowflow.app import referent_candidates as rc
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk
from knowflow.app.schemas import QueryRequest
from knowflow.app.source_scope import SourceScope, bind_candidate_scope
from test_clarification_context import provider, turn
from test_conversations import workspace
from test_source_scope import materials


def prior_payload():
    return {
        "current_question": "切回 MapleSync，改查正式环境，不用测试的数。",
        "previous_intents": [
            {
                "turn_id": "prior",
                "user_input": "MapleSync测试归档保留多久？",
                "user_intent": "MapleSync staging retention?",
            }
        ],
        "pending_clarification": {
            "turn_id": "pending",
            "original_question": "另一套备份呢？",
            "user_supplements": [],
            "requires_restatement": False,
        },
    }


def prior_result():
    return {
        "status": "resolved",
        "question": "MapleSync正式环境归档保留多久？排除测试环境。",
        "reference_turn_ids": ["prior"],
        "pending_action": "resume_prior",
        "return_anchor": "MapleSync",
    }


def test_explicit_prior_return_keeps_only_selected_user_intent_and_new_scope():
    payload, result = prior_payload(), prior_result()
    validated = context.validate_resolution(json.dumps(result), payload)
    assert validated["pending_action"] == "follow_prior_named_object"
    assert validated["superseded_pending_turn_id"] == "pending"
    intent = context.build_intent_context(payload, validated["reference_turn_ids"])
    assert [r["turn_id"] for r in intent["referenced_user_inputs"]] == ["prior"]
    assert intent["current_user_input"] == payload["current_question"]
    assert "另一套" not in json.dumps(intent, ensure_ascii=False)


@pytest.mark.parametrize(
    "question",
    [
        "那MapleSync的正式环境天数，别用测试天数，给个出处",
        "MapleSync正式环境呢？测试的不要",
        "And MapleSync in production, not staging?",
    ],
)
def test_named_object_follows_prior_without_command_keywords_or_action_field(question):
    payload, result = prior_payload(), prior_result()
    payload["current_question"] = question
    result.pop("pending_action")
    result.pop("return_anchor")
    result["current_object_name"] = "MapleSync"
    validated = context.validate_resolution(json.dumps(result), payload)
    assert validated["pending_action"] == "follow_prior_named_object"


@pytest.mark.parametrize(
    "fault",
    [
        "no_return",
        "negated_return",
        "forged_anchor",
        "pronoun",
        "unknown_ref",
        "memory",
        "restatement",
        "new_number",
    ],
)
def test_prior_return_never_relaxes_existing_identity_memory_and_numeric_gates(fault):
    payload, result = prior_payload(), prior_result()
    if fault == "no_return":
        payload["current_question"] = "正式环境呢？"
    if fault == "negated_return":
        payload["current_question"] = "不要再切回 MapleSync，先等一下。"
    if fault == "forged_anchor":
        result["return_anchor"] = "NotInUserInput"
    if fault == "pronoun":
        result["return_anchor"] = "那个"
    if fault == "unknown_ref":
        result["reference_turn_ids"] = ["not_supplied"]
    if fault == "memory":
        payload["previous_intents"][0]["turn_id"] = "memory:clue"
        result["reference_turn_ids"] = ["memory:clue"]
    if fault == "restatement":
        payload["pending_clarification"]["requires_restatement"] = True
    if fault == "new_number":
        result["question"] += "999天"
    with pytest.raises(context.ResolutionError):
        context.validate_resolution(json.dumps(result), payload)


@pytest.mark.asyncio
async def test_real_resolver_preserves_prior_return_in_actual_retrieval_question(monkeypatch):
    old = turn("MapleSync测试归档保留多久？")
    pending = turn("另一套备份呢？", status="needs_clarification")

    def output(payload):
        actual = next(r for r in payload["previous_intents"] if "MapleSync" in r["user_input"])
        return {
            **prior_result(),
            "reference_turn_ids": [actual["turn_id"]],
            "current_object_name": "MapleSync",
        }

    requests = provider(monkeypatch, [output])
    result = await context.resolve_question(
        "那MapleSync的正式环境天数，别用测试天数，给个出处", [pending, old], get_settings()
    )
    assert result["status"] == "resolved" and "正式环境" in result["question"]
    assert result["reference_turn_ids"] == [str(old.id)] and "pending_intent" not in result
    assert "NEVER_USE_OLD_ANSWER" not in json.dumps(requests)


def test_alternative_anchor_identity_is_not_lost_when_actor_kind_differs():
    from test_referent_candidates import fixture

    sources, data = fixture()
    data["target_kind"] = "organization"
    data["candidates"][0]["kind"] = "product"
    data["candidates"][1]["kind"] = "organization"
    result = rc.validate_options(
        json.dumps(data), sources, "Kestrel之外另一家呢？", [], "all_processed_text_supplied"
    )
    assert result["status"] == "conditional_unique"
    assert result["excluded_names"] == ["kestrel"]
    assert [r["name"] for r in result["options"]] == ["Morrow"]


@pytest.mark.parametrize("phrase,accepted", [("报纸", True), ("invented type", False), ("", False)])
def test_discovery_type_must_be_literal_current_user_input(phrase, accepted):
    payload = {"current_question": "那份报纸怎么不印了？", "previous_intents": []}
    result = context.validate_resolution(
        json.dumps(
            {
                "status": "clarify",
                "question": "哪个？",
                "reference_turn_ids": [],
                "candidate_discovery": {"target_phrase": phrase},
            }
        ),
        payload,
    )
    assert ("candidate_discovery" in result) == accepted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question", ["那款平板能用多久？", "that publication stopped printing?", "切回刚才的对象呢？"]
)
async def test_cold_demonstratives_and_return_without_history_reach_classifier(
    monkeypatch, question
):
    requests = provider(
        monkeypatch, [{"status": "clarify", "question": "请补充对象。", "reference_turn_ids": []}]
    )
    result = await context.resolve_question(question, [], get_settings())
    assert result["status"] == "clarify" and len(requests) == 1
    assert requests[0]["no_prior_intents"]


def product_sources():
    did = str(uuid4())
    return [
        {
            "source_id": "s1",
            "document_id": did,
            "filename": "catalog.txt",
            "text": "JadePad is a tablet product.",
            "version": {"sha256": "x"},
        },
        {
            "source_id": "s2",
            "document_id": did,
            "filename": "catalog.txt",
            "text": "JadePad is a tablet product.",
            "version": {"sha256": "x"},
        },
    ]


def product_data(sources):
    return {
        "target_kind": "product",
        "candidates": [
            {
                "name": "JadePad",
                "kind": "product",
                "source_id": s["source_id"],
                "quote": "JadePad is a tablet product.",
            }
            for s in sources
        ],
    }


@pytest.mark.parametrize("variant", ["single", "multiple", "homonym", "comparison", "forged"])
def test_generic_candidates_conditional_singleton_deduplication_and_ambiguity(variant):
    sources = product_sources()
    data = product_data(sources)
    question = "那款平板能用多久？"
    if variant == "multiple":
        sources[1]["text"] = "QuartzPad is a tablet product."
        data["candidates"][1].update(name="QuartzPad", quote=sources[1]["text"])
    if variant == "homonym":
        sources[1]["document_id"] = str(uuid4())
    if variant == "comparison":
        question = "比较那两款平板"
    if variant == "forged":
        data["candidates"][0]["name"] = "InventedPad"
    if variant == "forged":
        with pytest.raises(ValueError):
            rc.validate_options(
                json.dumps(data),
                sources,
                question,
                [],
                "retrieved_subset",
                {"target_phrase": "平板"},
            )
        return
    result = rc.validate_options(
        json.dumps(data), sources, question, [], "retrieved_subset", {"target_phrase": "平板"}
    )
    assert result["status"] == (
        "conditional_candidate" if variant == "single" else "confirm_options"
    )
    assert result["semantic_exhaustiveness"] == "unverified"
    if variant == "single":
        plan = rc.conditional_plan(question, result, [])
        assert plan["compiled_query"] == "JadePad\n" + question
        assert "tablet product" not in plan["compiled_query"]
        assert plan["source_document_ids"] == [sources[0]["document_id"]]
        assert not plan["full_request_complete"]


@pytest.mark.asyncio
async def test_large_scope_discovery_uses_type_not_an_answer_as_search_input(monkeypatch):
    did = uuid4()
    doc = SimpleNamespace(
        id=did,
        filename="inventory.txt",
        sha256="x",
        updated_at=SimpleNamespace(isoformat=lambda: "t"),
    )
    chunks = [
        SimpleNamespace(id=uuid4(), document_id=did, content="JadePad is a tablet product.")
        for _ in range(9)
    ]

    class Reader:
        async def execute(self, stmt):
            return [(c, doc) for c in chunks]

        async def scalars(self, stmt):
            return [doc]

        async def scalar(self, stmt):
            return 20

    calls = []

    async def rewrite(query, cfg):
        assert query == "平板"
        return {"status": "complete", "search_queries": ["tablet"]}

    async def retrieve(*args, **kwargs):
        calls.append((args[2], kwargs["search_queries"]))
        return [SimpleNamespace(chunk=chunks[0])]

    monkeypatch.setattr(rc, "plan_query", rewrite)
    monkeypatch.setattr(rc, "retrieve", retrieve)
    trace = {}
    sources, coverage = await rc.collect_sources(
        Reader(), uuid4(), "那款平板能用多久？", get_settings(), {"target_phrase": "平板"}, trace
    )
    assert calls == [("平板", ["tablet"])] and coverage == "retrieved_subset" and sources
    assert trace["query"] == "平板"


@pytest.mark.asyncio
async def test_candidate_binding_intersects_file_scope_and_never_accepts_foreign_ids(monkeypatch):
    assert (
        QueryRequest.model_validate(
            {"question": "q", "_source_document_ids": [str(uuid4())]}
        )._source_document_ids
        is None
    )
    async with workspace(monkeypatch) as (_, _, kb, _):
        kb = UUID(kb)
        async with SessionLocal() as session:
            docs, _ = await materials(session, kb)
            unscoped = SourceScope({"status": "unscoped"})
            bound = await bind_candidate_scope(session, kb, unscoped, (docs[0].id,))
            assert bound.document_ids == (docs[0].id,)
            wrong = await bind_candidate_scope(session, uuid4(), unscoped, (docs[0].id,))
            assert wrong.trace["status"] == "unavailable" and wrong.document_ids == ()
            conflict = await bind_candidate_scope(
                session, kb, SourceScope({"status": "filtered"}, (docs[1].id,)), (docs[0].id,)
            )
            assert conflict.document_ids == ()


@pytest.mark.asyncio
async def test_small_library_type_discovery_cannot_be_monopolized_by_long_document(monkeypatch):
    async def forbidden(*args, **kwargs):
        pytest.fail(
            "Small-library document discovery must sample each document before global top-k"
        )

    monkeypatch.setattr(rc, "retrieve", forbidden)
    monkeypatch.setattr(rc, "plan_query", forbidden)
    async with workspace(monkeypatch) as (_, _, kb, _):
        kb = UUID(kb)
        async with SessionLocal() as session:
            docs, _ = await materials(session, kb)
            cfg = get_settings()
            for ordinal in range(1, 12):
                session.add(
                    Chunk(
                        knowledge_base_id=kb,
                        document_id=docs[0].id,
                        ordinal=ordinal,
                        content="A long manual dominates global chunk ranking.",
                        search_text="long manual",
                        embedding=[1.0] + [0.0] * (cfg.embedding_dimensions - 1),
                        embedding_model=cfg.embedding_model,
                    )
                )
            await session.commit()
            trace = {}
            sources, coverage = await rc.collect_sources(
                session, kb, "那份刊物呢？", cfg, {"target_phrase": "刊物"}, trace
            )
            assert coverage == "document_samples"
            assert {s["document_id"] for s in sources} == {str(d.id) for d in docs[:3]}
            assert max(sum(s["document_id"] == str(d.id) for s in sources) for d in docs[:3]) <= 2
            assert (
                trace["documents_considered"] == 3
                and trace["policy"] == "bounded_document_intros_v1"
            )
