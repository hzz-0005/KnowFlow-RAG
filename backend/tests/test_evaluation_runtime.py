import pytest
from knowflow.app import evaluate_retrieval


def test_runtime_without_git_can_still_finish_paid_retrieval_evaluation(monkeypatch):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git is not installed in the API image")

    monkeypatch.setattr(evaluate_retrieval.subprocess, "run", no_git)
    assert evaluate_retrieval._code_revision() == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["hybrid", "graph_hybrid"])
async def test_answer_evaluation_does_not_pay_for_unrequested_graph_review(
    monkeypatch, tmp_path, strategy
):
    from types import SimpleNamespace
    from uuid import uuid4

    from knowflow.app import api, evaluate_quality
    from knowflow.app.config import Settings
    from knowflow.app.schemas import QueryResponse

    settings = Settings(_env_file=None, evaluation_results_dir=str(tmp_path))
    owner = SimpleNamespace(id=uuid4())
    snapshot = {
        "corpus_sha256": "fixed-corpus",
        "index_sha256": "fixed-index",
        "graph_sha256": "fixed-graph",
        "owner_id": owner.id,
        "documents": [],
        "indexed": [],
        "parsed": [],
        "graph": {"edges": [{"id": "existing-edge"}], "eligible_edge_count": 1},
    }
    calls = []

    async def source(*_):
        return snapshot

    async def graph_review(*_):
        calls.append("paid-graph-review")
        return {"status": "complete"}

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def get(self, *_):
            return owner

    async def query(_kb, body, _owner, _session, audit):
        audit.update(base=[], candidates=[], context=[])
        return QueryResponse(answer="资料不足", refused=True, strategy=body.strategy, citations=[])

    monkeypatch.setattr(evaluate_quality, "get_settings", lambda: settings)
    monkeypatch.setattr(evaluate_quality, "source_snapshot", source)
    monkeypatch.setattr(evaluate_quality, "review_graph_edges", graph_review)
    monkeypatch.setattr(evaluate_quality, "SessionLocal", Session)
    monkeypatch.setattr(api, "run_query", query)
    dataset = {
        "version": "scope-test",
        "source": "fixture",
        "cases": [
            {
                "id": "no-evidence",
                "question": "Unknown?",
                "should_refuse": True,
                "reference_answer": "Unknown",
                "relevant_evidence": [],
            }
        ],
    }
    report = await evaluate_quality.execute_quality(uuid4(), dataset, [strategy], 3, judge=True)
    assert report["status"] == "completed"
    assert calls == (["paid-graph-review"] if strategy == "graph_hybrid" else [])
    assert report["review_scope"]["graph_relations"] is (strategy == "graph_hybrid")
