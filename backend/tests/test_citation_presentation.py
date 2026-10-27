"""D-package: citation quote coverage, raw source id leaks and gate wording."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import api
from knowflow.app.api import CHUNK_ID_PATTERN, extended_quote, normalize_source_markers
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import QueryRequest, RelevanceAssessment, RetrievalDiagnostics


def test_source_lists_become_individually_validated_markers_without_deleting_words():
    ids = "[cb6ddb47-90f4-46cd-a33e-05c2a09c1d2b, 385a3a66-01f3-4711-94e5-cac9a8111af9]"
    cleaned = normalize_source_markers(f"共发行了 26 年 {ids}。")
    assert cleaned == (
        "共发行了 26 年 [cb6ddb47-90f4-46cd-a33e-05c2a09c1d2b]"
        "[385a3a66-01f3-4711-94e5-cac9a8111af9]。"
    )
    single = "[0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0]"
    assert single in normalize_source_markers(f"kept {single}")


def test_chunk_id_pattern_still_matches_single_ids_only():
    assert CHUNK_ID_PATTERN.fullmatch("[0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0]")
    assert not CHUNK_ID_PATTERN.fullmatch("[a, 0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0]")


def test_short_excerpts_expose_the_whole_chunk_for_verification():
    filler = "第一句与问题无关。" * 10
    base = "26 Jahre lang hat euch diese Zeitung Gesprächsstoff geliefert."
    chunk = filler + base + " Kosten fuer Druck und Verteilung."
    assert extended_quote(base, chunk) == chunk[:1200]
    assert extended_quote(base, "短块 " + base) == base  # already covers enough
    assert extended_quote("", chunk) == chunk[:1200]


def test_extended_quote_keeps_bounded_size_for_long_chunks():
    chunk = "内容" * 2000
    assert len(extended_quote("内容", chunk)) == 1200


@pytest.mark.parametrize(
    "data",
    [
        "[10000000000, 20000000000]",
        "[123456, 456789, 777777]",
        "[deadbeef, f00d]",
        "[1.00, -2.75, 3.20]",
    ],
)
def test_real_data_arrays_are_never_silently_deleted(data):
    assert normalize_source_markers("原始数据 " + data) == "原始数据 " + data


def test_long_source_lists_are_not_limited_to_two_ids():
    values = [str(uuid4()) for _ in range(5)]
    assert normalize_source_markers("[" + ", ".join(values) + "]") == "".join(
        f"[{value}]" for value in values
    )
    with pytest.raises(ValueError, match="source-marker"):
        normalize_source_markers(f"[{values[0]}, broken-label]")


def test_context_window_keeps_a_late_witness_and_never_invents_source_text():
    witness = "The approved policy applies for eighteen days."
    chunk = "Unrelated background. " * 100 + witness
    context = extended_quote(witness, chunk)
    assert witness in context and context in chunk and len(context) <= 1200
    assert extended_quote("Quote absent from this source", chunk) == "Quote absent from this source"


@pytest.mark.asyncio
@pytest.mark.parametrize("foreign", [False, True])
async def test_pipeline_validates_all_list_ids_and_preserves_reviewed_witness(monkeypatch, foreign):
    witness = "Approved source statement."
    hits = [
        RetrievalHit(
            chunk=SimpleNamespace(
                id=uuid4(),
                document_id=uuid4(),
                content=f"{label} background. " * 160 + witness,
                page_number=None,
                section=None,
                source_spans=[],
            ),
            filename="manual.txt",
        )
        for label in ("Alpha", "Beta")
    ]

    async def owned(*_):
        return None

    async def retrieve(*_):
        return hits

    async def references(*args, **_):
        return args[3], {"enabled": False}

    async def assess(_question, evidence, _settings):
        return {
            r["id"]: RelevanceAssessment(
                verdict="direct", reason="scripted", supporting_quotes=[witness]
            )
            for r in evidence
        }, RetrievalDiagnostics(status="complete")

    async def generate(*_, **__):
        second = uuid4() if foreign else hits[1].chunk.id
        return f"Data [10000000000, 20000000000] [{hits[0].chunk.id}, {second}]"

    async def guard(_question, draft, *_):
        return draft, {"status": "complete"}

    class Session:
        async def scalars(self, _):
            return []

    for name, value in (
        ("owned_knowledge_base", owned),
        ("retrieve", retrieve),
        ("expand_references", references),
        ("assess_retrieval", assess),
        ("generate_answer", generate),
        ("guard_answer", guard),
    ):
        monkeypatch.setattr(api, name, value)
    response = await api.run_query(
        uuid4(),
        QueryRequest(question="Explain source data", strategy="keyword"),
        SimpleNamespace(id=uuid4()),
        Session(),
    )
    if foreign:
        assert response.refused and not response.citations
    else:
        assert not response.refused and response.answer.endswith("[1][2]")
        assert "[10000000000, 20000000000]" in response.answer
        assert all(c.quote == witness and witness in c.context_excerpt for c in response.citations)
        assert all(str(h.chunk.id) not in response.answer for h in hits)
