"""Literal current sources need no history rewrite; history-dependent facts still do."""

from uuid import uuid4

import pytest
from knowflow.app.document_reference import named_document_resolution
from knowflow.app.source_scope import SourceScope


def named_scope(count=2, **changes):
    names = ["TerraShape_项目详解.md", "OpsHarbor_项目详解.md"][:count]
    ids = tuple(uuid4() for _ in names)
    return SourceScope(
        {
            "status": "filtered",
            "selection_basis": "multiple_literal_titles",
            "documents": [
                {"document_id": str(i), "filename": n} for i, n in zip(ids, names, strict=True)
            ],
            **changes,
        },
        ids,
    )


@pytest.mark.parametrize(
    "question",
    [
        "再看 TerraShape 和 OpsHarbor，它们都用了哪些数据库？按项目分开说，别带上前一个项目。",
        "分别列出 TerraShape 和 OpsHarbor 使用的存储组件。",
        "Compare TerraShape and OpsHarbor. Show their databases without the previous project.",
    ],
)
def test_current_plural_sources_keep_the_original_question(question):
    result = named_document_resolution(question, named_scope(), "test-version")
    assert result is not None
    assert result["status"] == "standalone" and result["question"] == question
    assert result["reference_turn_ids"] == [] and not result["request_attempted"]
    assert result["request_kind"] == "fact_lookup"


@pytest.mark.parametrize(
    "question",
    [
        "按刚才的规则比较 TerraShape 和 OpsHarbor，哪个更好？",
        "TerraShape 和 OpsHarbor 都按之前的阈值检查，结果是多少？",
        "比较 TerraShape 和 OpsHarbor，前者是否满足后者的要求？",
        "TerraShape 和 OpsHarbor 之外，另一份文档怎么处理？",
        "Show TerraShape and OpsHarbor using the same threshold as before.",
        "再看 TerraShape 和 OpsHarbor，别带上前一个项目，但按刚才的条件比较。",
    ],
)
def test_literal_names_do_not_resolve_a_missing_history_constraint(question):
    assert named_document_resolution(question, named_scope(), "test-version") is None


@pytest.mark.parametrize("status", ["ambiguous", "unavailable", "unscoped"])
def test_uncertain_source_identity_still_blocks_the_local_route(status):
    assert (
        named_document_resolution(
            "TerraShape和OpsHarbor用了哪些数据库？", named_scope(status=status), "test-version"
        )
        is None
    )


def test_excluding_a_name_does_not_make_the_rest_explicitly_named():
    assert (
        named_document_resolution(
            "排除 TerraShape，其他项目用了哪些数据库？",
            named_scope(selection_basis="excluded_named_documents"),
            "test-version",
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("pending", [False, True])
async def test_current_two_sources_override_old_topic_before_the_model(monkeypatch, pending):
    from types import SimpleNamespace
    from uuid import UUID

    from knowflow.app import conversation_api
    from knowflow.app.db import SessionLocal
    from knowflow.app.models import KnowledgeBase
    from knowflow.app.schemas import QueryResponse
    from test_conversations import create, send, workspace
    from test_document_topic_switch import seed_documents

    async with workspace(monkeypatch) as (client, auth, kb, _calls):
        async with SessionLocal() as session:
            owner = (await session.get(KnowledgeBase, UUID(kb))).owner_id
            docs = await seed_documents(session, UUID(kb), owner)
        cid = (await create(client, auth[0], kb))["id"]
        initial = "另一个项目呢？" if pending else "TerraShape_Agent是什么？"
        first = (await send(client, auth[0], cid, initial)).json()
        assert first["status"] == ("needs_clarification" if pending else "completed")

        async def forbidden_history(*args, **kwargs):
            raise AssertionError("Current explicit sources must not be rewritten by history")

        seen = []

        async def answer(_kb, body, _user, _session):
            seen.append(
                SimpleNamespace(
                    question=body.question,
                    ids=body._source_document_ids,
                    allow_prior=body._allow_context_document,
                )
            )
            return QueryResponse(
                answer="route control", refused=False, strategy=body.strategy, citations=[]
            )

        monkeypatch.setattr(conversation_api, "resolve_question", forbidden_history)
        monkeypatch.setattr(conversation_api, "query_knowledge_base", answer)
        question = "再看校园二手交易平台和 OpsHarbor，它们用了哪些数据库？别带上前一个项目。"
        result = (await send(client, auth[0], cid, question, revision=1)).json()
        assert result["status"] == "completed"
        assert result["resolution"]["routing_reason"] == "current_named_documents"
        assert len(seen) == 1 and seen[0].question == question and not seen[0].allow_prior
        assert set(seen[0].ids) == {docs[0].id, docs[2].id}
