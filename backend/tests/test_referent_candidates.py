"""Grounding, coverage, isolation and conditional lookup controls; not entity accuracy."""

import json
from copy import deepcopy
from uuid import UUID, uuid4

import pytest
from knowflow.app import conversation_api as api
from knowflow.app import referent_candidates as rc
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import Chunk, Conversation, Document
from knowflow.app.schemas import QueryResponse
from test_clarification_context import provider, turn
from test_conversations import create, send, workspace


def fixture():
    sources = [
        {
            "source_id": "s",
            "document_id": "d",
            "filename": "rules.txt",
            "text": "Kestrel archive policy. Morrow backup policy. Finch backup policy.",
            "version": {"sha256": "x", "updated_at": "t"},
        }
    ]
    data = {
        "target_kind": "system",
        "candidates": [
            {"name": name, "kind": "system", "source_id": "s", "quote": f"{name} {kind} policy."}
            for name, kind in [("Kestrel", "archive"), ("Morrow", "backup")]
        ],
    }
    return sources, data


def test_scoped_singleton_is_conditional_never_global_uniqueness():
    sources, data = fixture()
    result = rc.validate_options(
        json.dumps(data), sources, "Kestrel之外另一套？", [], "all_processed_text_supplied"
    )
    assert result["status"] == "conditional_unique" and result["options"][0]["name"] == "Morrow"
    assert result["semantic_exhaustiveness"] == "unverified"
    plan = rc.conditional_plan("另一套收费多少？", result, [turn("Kestrel的期限？")])
    assert not plan["intent_context"]["referent_assumption"]["confirmed_by_user"]
    assert "Morrow" in plan["compiled_query"] and plan["full_request_complete"] is False
    result["options"][0]["quote"] = "Morrow backups last 14 days."
    plan = rc.conditional_plan("另一套收费多少？", result, [])
    assert "14" not in plan["compiled_query"]
    assert plan["compiled_query"].startswith("Morrow\n")
    assert "kestrel" not in plan["compiled_query"].casefold()
    assert "source_quote" not in plan["intent_context"]["referent_assumption"]


@pytest.mark.parametrize("case", ["subset", "multiple", "no_anchor", "two_anchors", "homonym"])
def test_incomplete_or_ambiguous_inventory_only_lists_options(case):
    sources, data = fixture()
    question, coverage = "Kestrel之外另一套？", "all_processed_text_supplied"
    if case == "subset":
        coverage = "retrieved_subset"
    if case == "multiple":
        data["candidates"].append(
            {"name": "Finch", "kind": "system", "source_id": "s", "quote": "Finch backup policy."}
        )
    if case == "no_anchor":
        question = "另一套规则？"
    if case == "two_anchors":
        question = "Kestrel和Morrow之外另一套？"
    if case == "homonym":
        sources.append({**sources[0], "source_id": "s2", "document_id": "d2"})
        data["candidates"].append({**data["candidates"][1], "source_id": "s2"})
    result = rc.validate_options(json.dumps(data), sources, question, [], coverage)
    assert result["status"] == "confirm_options"


@pytest.mark.parametrize("case", ["name", "quote", "source", "blank"])
def test_unprovided_entities_or_witnesses_are_rejected(case):
    sources, data = fixture()
    if case == "blank":
        data["candidates"][0]["name"] = " "
    else:
        data["candidates"][0][{"name": "name", "quote": "quote", "source": "source_id"}[case]] = (
            "INVENTED"
        )
    with pytest.raises(ValueError):
        rc.validate_options(json.dumps(data), sources, "另一套？", [], "retrieved_subset")


@pytest.mark.asyncio
async def test_candidate_snapshot_is_scoped_and_literal_no_provider(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            d = Document(
                knowledge_base_id=UUID(kb),
                uploaded_by=conv.owner_id,
                filename="local.txt",
                storage_key=str(uuid4()),
                media_type="text/plain",
                file_size=10,
                sha256="1" * 64,
                status="processed",
            )
            session.add(d)
            await session.flush()
            session.add(
                Chunk(
                    document_id=d.id,
                    knowledge_base_id=UUID(kb),
                    ordinal=0,
                    content="Kestrel archive. Morrow backup.",
                    search_text="Kestrel archive Morrow backup",
                    embedding=[0.0] * get_settings().embedding_dimensions,
                    embedding_model="offline_fixture",
                )
            )
            await session.commit()

            sources, coverage = await rc.collect_sources(
                session, UUID(kb), "另一套？", get_settings()
            )
            assert len(sources) == 1 and coverage == "all_processed_text_supplied"
            assert sources[0]["filename"] == "local.txt"
            foreign, _ = await rc.collect_sources(session, uuid4(), "另一套？", get_settings())
            assert not foreign


@pytest.mark.asyncio
async def test_conditional_candidate_replaces_partial_query_preserves_budget_and_pending(
    monkeypatch,
):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        async with SessionLocal() as session:
            conv = await session.get(Conversation, UUID(cid))
            d = Document(
                knowledge_base_id=UUID(kb),
                uploaded_by=conv.owner_id,
                filename="local.txt",
                storage_key=str(uuid4()),
                media_type="text/plain",
                file_size=10,
                sha256="1" * 64,
                status="processed",
            )
            session.add(d)
            await session.commit()
            did, version = d.id, {"sha256": d.sha256, "updated_at": d.updated_at.isoformat()}
        options = {
            "version": rc.VERSION,
            "status": "conditional_unique",
            "coverage": "all_processed_text_supplied",
            "excluded_names": ["kestrel"],
            "source_versions": {str(did): version},
            "options": [
                {
                    "name": "Morrow",
                    "quote": "Morrow backup policy.",
                    "filename": "local.txt",
                    "document_id": str(did),
                    "source_id": str(uuid4()),
                }
            ],
        }

        async def find(*args):
            return deepcopy(options)

        monkeypatch.setattr(rc, "find_options", find)
        provider(
            monkeypatch,
            [{"status": "clarify", "question": "另一套是谁？", "reference_turn_ids": []}],
        )
        calls = []

        async def query(kb_id, body, user, session):
            calls.append((body.question, body._intent_context))
            return QueryResponse(
                answer="按Morrow理解，保留14天 [1]。",
                refused=False,
                strategy="keyword",
                citations=[
                    dict(
                        chunk_id=uuid4(),
                        document_id=did,
                        filename="local.txt",
                        page_number=None,
                        section=None,
                        quote="14 days",
                    )
                ],
            )

        monkeypatch.setattr(api, "query_knowledge_base", query)
        text = "另一套备份多久？"
        key = str(uuid4())
        row = (await send(client, auth[0], cid, text, 0, key)).json()
        assert row["status"] == "needs_clarification" and not row["resolved_question"]
        assert len(calls) == 1 and "Morrow" in calls[0][0] and text in calls[0][0]
        assert calls[0][1]["referent_assumption"]["confirmed_by_user"] is False
        assert row["resolution"]["pending_intent"]["question"] == text
        assert row["response"]["referent_candidates"]["status"] == "conditional_unique"
        assert (await send(client, auth[0], cid, text, 0, key)).json() == row and len(calls) == 1
        async with SessionLocal() as session:
            from knowflow.app.conversation_context import pending_clarification

            conv = await session.get(Conversation, UUID(cid))
            history = await api.recent_intents(session, conv)
            assert pending_clarification(history)["referent_options"][0]["name"] == "Morrow"
            doc = await session.get(Document, did)
            doc.sha256 = "2" * 64
            await session.commit()
            history = await api.recent_intents(session, conv)
            assert pending_clarification(history)["referent_options"] == []


@pytest.mark.asyncio
async def test_new_topic_and_stale_memory_do_not_probe(monkeypatch):

    async def forbidden(*args):
        raise AssertionError("must not read sources")

    monkeypatch.setattr(rc, "collect_sources", forbidden)
    for resolution in (
        {"status": "standalone"},
        {"status": "clarify", "pending_intent": {"requires_restatement": True}},
    ):
        assert (
            await rc.find_options(None, uuid4(), "另一套系统？", [], resolution, get_settings())
            is None
        )


def test_project_kind_is_a_valid_referent_kind():
    sources, data = fixture()
    data["target_kind"] = "project"
    data["candidates"] = [
        {"name": "Kestrel", "kind": "project", "source_id": "s", "quote": "Kestrel archive policy."}
    ]

    result = rc.validate_options(
        json.dumps(data), sources, "另一个项目是怎么做归档的？", [], "all_processed_text_supplied"
    )

    assert result["target_kind"] == "project"
    assert result["status"] in {"conditional_candidate", "confirm_options"}
    assert [row["name"] for row in result["options"]] == ["Kestrel"]


@pytest.mark.asyncio
async def test_comparison_questions_sample_every_document(monkeypatch):
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            from knowflow.app.models import KnowledgeBase

            owner = (await session.get(KnowledgeBase, UUID(kb))).owner_id
            for index in range(3):
                doc = Document(
                    knowledge_base_id=UUID(kb),
                    uploaded_by=owner,
                    filename=f"project-{index}.md",
                    storage_key=str(uuid4()),
                    media_type="text/markdown",
                    file_size=10,
                    sha256=str(index) * 64,
                    status="processed",
                )
                session.add(doc)
                await session.flush()
                for ordinal in range(4):
                    content = f"Project {index} opening {ordinal} describes its evidence model."
                    session.add(
                        Chunk(
                            document_id=doc.id,
                            knowledge_base_id=UUID(kb),
                            ordinal=ordinal,
                            content=content,
                            search_text=content,
                            embedding=[0.0] * get_settings().embedding_dimensions,
                            embedding_model="offline_fixture",
                        )
                    )
            await session.commit()

            sources, _ = await rc.collect_sources(
                session, UUID(kb), "另一个项目是怎么做证据管理的？", get_settings()
            )

            assert len({row["document_id"] for row in sources}) == 3


def test_line_break_differences_do_not_break_a_grounded_quote():
    sources, data = fixture()
    data["candidates"] = [
        {
            "name": "Kestrel",
            "kind": "system",
            "source_id": "s",
            "quote": "Kestrel   archive\npolicy.",
        }
    ]

    result = rc.validate_options(
        json.dumps(data), sources, "另一个系统怎么做归档？", [], "all_processed_text_supplied"
    )

    assert [row["name"] for row in result["options"]] == ["Kestrel"]


def test_invented_text_is_still_rejected_after_normalization():
    sources, data = fixture()
    data["candidates"] = [
        {
            "name": "Kestrel",
            "kind": "system",
            "source_id": "s",
            "quote": "Kestrel archive retention.",
        }
    ]

    with pytest.raises(ValueError, match="ungrounded_candidate"):
        rc.validate_options(
            json.dumps(data), sources, "另一个系统怎么做归档？", [], "all_processed_text_supplied"
        )
