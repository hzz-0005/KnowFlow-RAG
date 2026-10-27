"""Document summary routing, segmentation and citation hygiene (offline)."""

from uuid import uuid4

import pytest
from knowflow.app import document_summary, providers
from knowflow.app.config import Settings
from knowflow.app.document_summary import plan_segments, requested
from knowflow.app.models import Chunk
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import QueryRequest


def body(question, kind=None):
    request = QueryRequest(question=question)
    request._request_kind = kind
    return request


@pytest.mark.parametrize(
    "question",
    [
        "Summarize the maintenance policy.",
        "总结一下各站点的检修安排",
        "What does the report say about 2023 revenue?",
        "不要总结这份文档，直接给结论",
    ],
)
def test_a_summary_cue_without_a_document_object_stays_a_normal_question(question):
    assert requested(body(question)) is False


@pytest.mark.parametrize(
    "question",
    [
        "这份文档讲了什么？",
        "总结一下这份报告的主要内容",
        "give me an overview of this paper",
        "这篇论文说了什么",
    ],
)
def test_document_overview_requests_enter_summary_mode(question):
    assert requested(body(question)) is True


def test_classifier_kind_overrides_the_keyword_fallback():
    assert requested(body("Whatever", kind="document_summary")) is True
    assert requested(body("这份文档讲了什么", kind="fact_lookup")) is False


@pytest.mark.parametrize(
    "filename",
    [
        "CedarLedger_项目总结_20261027.md",
        "实验总结与案例说明.pdf",
        "Research_summary.md",
    ],
)
def test_a_summary_word_in_a_resolved_filename_does_not_override_specific_fact_lookup(filename):
    request = body(f"再看看 {filename}，只列两项量化验证结果。", kind="fact_lookup")
    request._source_document_ids = (uuid4(),)
    assert requested(request, [filename]) is False
    assert requested(body(f"总结一下 {filename} 的主要内容"), [filename]) is True


def test_explicit_no_summary_keeps_priority_over_a_wrong_classifier_kind():
    assert requested(body("不要总结这份文档，只给数字。", kind="document_summary")) is False


def hits(sizes, section=None):
    document_id, kb_id = uuid4(), uuid4()
    rows = []
    for ordinal, size in enumerate(sizes):
        chunk = Chunk(
            id=uuid4(),
            document_id=document_id,
            knowledge_base_id=kb_id,
            ordinal=ordinal,
            content=f"part-{ordinal} " + "字" * size,
            search_text="",
            section=section or f"section-{ordinal // 2}",
            page_number=ordinal + 1,
        )
        rows.append(RetrievalHit(chunk, "long.pdf", retrieval_stage="document_read"))
    return rows


class Counter:
    def count(self, text):
        return len(text)


def test_segments_keep_order_and_cover_every_chunk_once():
    rows = hits([40] * 9)
    segments = plan_segments(rows, Counter(), token_budget=100)

    assert [hit.chunk.ordinal for segment in segments for hit in segment] == list(range(9))
    assert all(sum(len(hit.chunk.content) + 12 for hit in segment) <= 100 for segment in segments)
    assert len(segments) > 1


@pytest.mark.asyncio
async def test_segmented_summary_keeps_only_supplied_citations(monkeypatch):
    rows = hits([600] * 6)
    # Four chunks per segment at this token budget; the fake model cites one local
    # chunk and one foreign chunk in every part.
    settings = Settings(
        _env_file=None,
        rag_context_token_budget=2600,
        query_max_provider_calls=12,
        rag_answer_total_token_budget=32768,
        rag_answer_output_token_budget=1024,
    )
    foreign = str(uuid4())

    async def fake_generate(question, evidence, settings=None, **_kwargs):
        local = evidence[0]["id"]
        return f"- Part point [{local}] and a foreign marker [{foreign}]"

    monkeypatch.setattr(providers, "generate_answer", fake_generate)

    result = await document_summary.segmented_summary(
        "这份文档讲了什么",
        rows,
        {"document": {"filename": "long.pdf"}},
        settings,
    )

    supplied = {str(hit.chunk.id) for hit in rows}
    assert result["status"] == "ok"
    assert result["evidence"] and {row["id"] for row in result["evidence"]} <= supplied
    assert foreign not in {row["id"] for row in result["evidence"]}
    assert result["updates"]["segments"] > 1
    assert result["updates"]["summarized_segments"] == result["updates"]["planned_segments"]


@pytest.mark.asyncio
async def test_a_failed_part_is_reported_as_a_coverage_gap(monkeypatch):
    rows = hits([600] * 6)
    settings = Settings(
        _env_file=None,
        rag_context_token_budget=2600,
        query_max_provider_calls=12,
        rag_answer_total_token_budget=32768,
        rag_answer_output_token_budget=1024,
    )
    calls = {"n": 0}

    async def flaky(question, evidence, settings=None, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError
        return f"- Point [{evidence[0]['id']}]"

    monkeypatch.setattr(providers, "generate_answer", flaky)

    result = await document_summary.segmented_summary(
        "这份文档讲了什么",
        rows,
        {"document": {"filename": "long.pdf"}},
        settings,
    )

    assert result["status"] == "ok"
    assert result["updates"]["failed_segments"] == 1
    assert result["updates"]["read_chunks"] < len(rows)
    assert result["updates"]["planned_chunks"] == len(rows)
    assert result["updates"]["missing_segments"] == [1]
    trace = {
        "document": {"filename": "long.pdf"},
        "total_chunks": 6,
        "read_chunks": 3,
        "segments": 2,
        "summarized_segments": result["updates"]["summarized_segments"],
        "coverage": "partial_parsed_text_segmented",
        "pages": [1, 4],
    }
    note = document_summary.coverage_statement(trace, chinese=True)
    assert "未读" in note and "不代表全文结论" in note


@pytest.mark.asyncio
async def test_all_failed_segments_cannot_claim_the_planned_text_was_read(monkeypatch):
    async def unavailable(*args, **kwargs):
        raise TimeoutError("unavailable")

    monkeypatch.setattr(providers, "generate_answer", unavailable)
    rows = hits([600] * 6)
    result = await document_summary.segmented_summary(
        "总结这份文档",
        rows,
        {"document": {"filename": "long.pdf"}},
        Settings(_env_file=None, rag_context_token_budget=2600),
    )
    assert result["status"] == "failed"
    assert result["updates"]["read_chunks"] == 0
    assert result["updates"]["loaded_chunks"] == len(rows)
    assert result["updates"]["missing_segments"]


@pytest.mark.asyncio
async def test_whole_document_generation_limit_does_not_claim_completed_reading(monkeypatch):
    from types import SimpleNamespace
    from uuid import UUID

    from knowflow.app import api
    from knowflow.app.answer_budget import AnswerOutputLimit
    from knowflow.app.db import SessionLocal
    from test_conversations import workspace
    from test_source_scope import materials

    async def output_limit(*args, **kwargs):
        raise AnswerOutputLimit("simulated provider limit")

    monkeypatch.setattr(api, "generate_answer", output_limit)
    async with workspace(monkeypatch) as (_, _, kb, _):
        async with SessionLocal() as session:
            _, owner = await materials(session, UUID(kb))
            result = await api.run_query(
                UUID(kb), body("总结 ledger-07.pdf"), SimpleNamespace(id=owner), session
            )
            assert result.refused
            assert result.document_summary["loaded_chunks"] == 1
            assert result.document_summary["read_chunks"] == 0
            assert result.document_summary["summarized_segments"] == 0
            assert result.document_summary["coverage"] == "parsed_text_prepared_not_generated"


def test_coverage_statement_says_how_much_was_read():
    complete = {
        "document": {"filename": "a.pdf"},
        "total_chunks": 12,
        "read_chunks": 12,
        "segments": 3,
        "summarized_segments": 3,
        "coverage": "complete_parsed_text_segmented",
        "pages": [1, 8],
    }
    text = document_summary.coverage_statement(complete, chinese=True)
    assert "12/12" in text and "第 1–8 页" in text and "图片" in text


def test_segment_prompt_never_promises_a_whole_document():
    prompt = document_summary._segment_question("原问题", "a.pdf", 2, 5)
    assert "part 2 of 5" in prompt and "do not claim the whole document" in prompt


@pytest.mark.asyncio
async def test_cold_document_summary_skips_the_classifier():
    from knowflow.app.conversation_context import resolve_question

    settings = Settings(_env_file=None)
    overview = await resolve_question("这份文档讲了什么？", [], settings)
    assert overview["status"] == "standalone"
    assert overview["request_kind"] == "document_summary"
    assert overview["request_attempted"] is False

    fact = await resolve_question("Summarize the maintenance policy.", [], settings)
    assert "request_kind" not in fact and fact["request_attempted"] is False


def test_summary_path_gets_its_own_reading_window():
    settings = Settings(_env_file=None)
    window = document_summary.summary_settings(settings)

    assert window.rag_context_token_budget == settings.rag_summary_context_token_budget
    assert window.rag_context_token_budget > settings.rag_context_token_budget
    # Retrieval answers keep the small evidence budget.
    assert settings.rag_context_token_budget == 6000
    assert settings.rag_summary_context_token_budget >= 32000


def test_summary_blocks_without_a_real_citation_are_dropped():
    first, second = str(uuid4()), str(uuid4())
    text = (
        "## 概述\n\n本文提出 TableFormer。[" + first + "]\n\n"
        "这一段没有任何引用。\n\n"
        f"另一段引用不存在的来源 [{uuid4()}]。"
    )
    kept, check = document_summary.keep_cited_blocks(text, {first, second})

    assert first in kept and "这一段没有任何引用" not in kept
    assert check["dropped_blocks"] == 2
    assert check["total_blocks"] == check["kept_blocks"] + check["dropped_blocks"]


def test_overview_request_naming_one_document_reads_that_document():
    from knowflow.app.document_summary import requested
    from knowflow.app.schemas import QueryRequest

    document = uuid4()
    other = uuid4()

    def body(question, ids):
        request = QueryRequest(question=question)
        request._source_document_ids = tuple(ids)
        return request

    # One named file plus an overview cue is a whole-file read, not a top-k guess.
    assert requested(body("介绍一下 AegisOps_项目全解_简历与面试.md", [document])) is True
    assert requested(body("概述 AegisOps_项目全解_简历与面试.md", [document])) is True
    # Several files stay a comparison; an explicit fact kind is never overridden.
    assert requested(body("介绍一下 AegisOps 和 GeoModel", [document, other])) is False
    fact = body("介绍一下 AegisOps_项目全解_简历与面试.md", [document])
    fact._request_kind = "fact_lookup"
    assert requested(fact) is True
    comparison = body("AegisOps 的告警关联是怎么实现的？", [])
    comparison._request_kind = "fact_lookup"
    assert requested(comparison) is False
