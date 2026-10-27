from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import api
from knowflow.app.retrieval import RetrievalHit
from knowflow.app.schemas import QueryRequest, RelevanceAssessment, RetrievalDiagnostics


@pytest.mark.asyncio
@pytest.mark.parametrize("guard_keeps", [True, False])
async def test_graph_addition_reaches_assessment_and_citation_but_not_base_top_k(
    monkeypatch, guard_keeps
):
    doc_id, kb_id = uuid4(), uuid4()
    hits = [
        RetrievalHit(
            chunk=SimpleNamespace(
                id=uuid4(),
                document_id=doc_id,
                content=text,
                page_number=None,
                section=None,
                source_spans=[],
            ),
            filename="manual.md",
            retrieval_stage=stage,
            graph_rank=1 if stage == "graph_expansion" else None,
        )
        for text, stage in [("A routes to B.", "base"), ("B is hosted on C.", "graph_expansion")]
    ]

    async def owned(*_):
        return None

    async def retrieved(*_):
        return hits

    async def references(*args, **_):
        return args[3], {"enabled": False}

    inspected = []

    async def assessment(_question, evidence, _settings):
        inspected.extend(row["id"] for row in evidence)
        return {
            row["id"]: RelevanceAssessment(
                verdict="partial",
                reason="One link of the chain",
                supporting_quotes=[row["content"]],
            )
            for row in evidence
        }, RetrievalDiagnostics(status="complete")

    async def answer(_question, evidence, _settings, **kwargs):
        assert len(evidence) == 2
        return "A reaches C via B " + " ".join(f"[{row['id']}]" for row in evidence)

    class Session:
        async def scalars(self, _):
            return []

    monkeypatch.setattr(api, "owned_knowledge_base", owned)
    monkeypatch.setattr(api, "retrieve", retrieved)
    monkeypatch.setattr(api, "expand_references", references)
    monkeypatch.setattr(api, "assess_retrieval", assessment)
    monkeypatch.setattr(api, "generate_answer", answer)

    async def guard(_question, draft, evidence, _settings):
        assert len(evidence) == 2
        return (draft if guard_keeps else ""), {"status": "complete" if guard_keeps else "rejected"}

    monkeypatch.setattr(api, "guard_answer", guard)
    audit = {"top_k": 1}
    response = await api.run_query(
        kb_id,
        QueryRequest(question="Where is A hosted?", strategy="graph_hybrid", top_k=1),
        SimpleNamespace(id=uuid4()),
        Session(),
        audit,
    )
    assert audit["base"][0]["chunk_id"] == str(hits[0].chunk.id)
    assert len(audit["base"]) == 1 and len(audit["candidates"]) == 2
    assert audit["draft_answer"]
    if not guard_keeps:
        assert response.refused and not response.citations
        assert response.answer_validation["status"] == "rejected"
        assert response.generation_route["prompt_version"]
        return
    assert len(inspected) == 2 and len(response.citations) == 2
    assert response.graph_expansion["added_count"] == 1
    assert response.retrieval_evidence[1].retrieval_stage == "graph_expansion"
    assert response.retrieval_evidence[1].included_in_context
