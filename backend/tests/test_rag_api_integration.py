import json
import uuid

import httpx
import pytest
from knowflow.app import api, graph_index, graph_semantics, ingestion, reranker, retrieval
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.graph_index import ChunkGraphFacts, EntityFact, RelationFact
from knowflow.app.main import app
from knowflow.app.models import User
from knowflow.app.schemas import RelevanceAssessment, RetrievalDiagnostics
from sqlalchemy import delete


def _test_embedding() -> list[float]:
    vector = [0.0] * get_settings().embedding_dimensions
    vector[0] = 1.0
    return vector


@pytest.mark.asyncio
async def test_upload_retrieval_citations_and_knowledge_base_isolation(
    monkeypatch, tmp_path
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    result_dir = tmp_path / "evaluation-results"
    result_dir.mkdir()
    # Audit artifacts share this directory but are not necessarily evaluation objects.
    (result_dir / "array-audit.json").write_text("[]", encoding="utf-8")
    (result_dir / "null-audit.json").write_text("null", encoding="utf-8")
    monkeypatch.setattr(settings, "evaluation_results_dir", str(result_dir))

    async def fake_embeddings(texts, _settings=None):
        return [_test_embedding() for _ in texts]

    async def fake_answer(question, evidence, _settings=None, **kwargs):
        return f"根据文档，{question} 的答案是已索引内容 [{evidence[0]['id']}]。"

    async def fake_assessment(question, evidence, _settings=None):
        return {
            row["id"]: RelevanceAssessment(
                verdict="direct",
                reason="片段含有所问事实。",
                supporting_quotes=[row["content"]],
            )
            for row in evidence
        }, RetrievalDiagnostics(status="complete" if evidence else "empty")

    async def fake_graph_extraction(_content):
        return ChunkGraphFacts(
            entities=[
                EntityFact(name="数据库", evidence="数据库"),
                EntityFact(name="连接配置", evidence="连接配置"),
            ],
            relations=[
                RelationFact(
                    source="数据库",
                    target="连接配置",
                    predicate="描述",
                    description="文档描述数据库连接配置。",
                    evidence="数据库连接配置",
                    confidence=0.95,
                )
            ],
        )

    async def fake_rerank(_query, hits, top_k, _settings=None):
        for rank, hit in enumerate(hits, start=1):
            hit.rerank_rank = rank
            hit.rerank_score = 1.0 / rank
        return hits[:top_k]

    monkeypatch.setattr(retrieval, "embed_texts", fake_embeddings)
    monkeypatch.setattr(ingestion, "embed_texts", fake_embeddings)
    monkeypatch.setattr(api, "generate_answer", fake_answer)
    monkeypatch.setattr(api, "assess_retrieval", fake_assessment)
    monkeypatch.setattr(graph_index, "extract_chunk_graph", fake_graph_extraction)

    async def graph_reviewer(graph, _settings, **_):
        return {
            "units": [
                {"id": e["id"], "verdict": "supported", "reason": "fixture"} for e in graph["edges"]
            ]
        }

    monkeypatch.setattr(graph_semantics, "review_graph_edges", graph_reviewer)
    monkeypatch.setattr(reranker, "rerank_hits", fake_rerank)
    # This integration scenario mocks the provider; configuration preflight is
    # covered separately with an absent service and forbidden paid stages.
    monkeypatch.setattr(reranker, "require_rerank_configuration", lambda settings: None)

    transport = httpx.ASGITransport(app=app)
    emails = [f"a-{uuid.uuid4()}@example.com", f"b-{uuid.uuid4()}@example.com"]
    async with httpx.AsyncClient(transport=transport, base_url="http://knowflow.test") as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        register_a = await client.post(
            "/api/v1/auth/register",
            json={"email": emails[0], "password": "correct-horse-battery"},
        )
        register_b = await client.post(
            "/api/v1/auth/register",
            json={"email": emails[1], "password": "correct-horse-battery"},
        )
        assert register_a.status_code == 201
        assert register_b.status_code == 201
        headers_a = {"Authorization": f"Bearer {register_a.json()['access_token']}"}
        headers_b = {"Authorization": f"Bearer {register_b.json()['access_token']}"}

        created_kb = await client.post(
            "/api/v1/knowledge-bases", json={"name": "Runbook"}, headers=headers_a
        )
        assert created_kb.status_code == 201
        kb_id = created_kb.json()["id"]
        run_id = uuid.uuid4()
        (result_dir / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "run_id": str(run_id),
                    "created_at": "2026-10-21T00:00:00+00:00",
                    "dataset_version": "test-v1",
                    "code_revision": "uncommitted",
                    "strategy": "hybrid",
                    "top_k": 5,
                    "metrics": {"mrr": 0.75},
                    "knowledge_base_id": kb_id,
                    "cases": [],
                }
            ),
            encoding="utf-8",
        )
        evaluation_runs = await client.get(
            f"/api/v1/knowledge-bases/{kb_id}/evaluations", headers=headers_a
        )
        assert evaluation_runs.status_code == 200
        assert evaluation_runs.json()[0]["run_id"] == str(run_id)
        evaluation_detail = await client.get(
            f"/api/v1/knowledge-bases/{kb_id}/evaluations/{run_id}", headers=headers_a
        )
        assert evaluation_detail.status_code == 200
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb_id}/evaluations", headers=headers_b)
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb_id}/documents", headers=headers_b)
        ).status_code == 404

        uploaded = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            headers=headers_a,
            files={"file": ("handbook.txt", "已索引内容包括数据库连接配置。", "text/plain")},
        )
        assert uploaded.status_code == 202, uploaded.text
        document = uploaded.json()
        processed = await client.get(f"/api/v1/documents/{document['id']}", headers=headers_a)
        assert processed.json()["status"] == "processed"
        duplicate = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents",
            headers=headers_a,
            files={"file": ("another-name.txt", "已索引内容包括数据库连接配置。", "text/plain")},
        )
        assert duplicate.status_code == 409
        reprocessed = await client.post(
            f"/api/v1/documents/{document['id']}/reprocess", headers=headers_a
        )
        assert reprocessed.status_code == 202
        processed_again = await client.get(f"/api/v1/documents/{document['id']}", headers=headers_a)
        assert processed_again.json()["status"] == "processed"

        question = "数据库连接配置是什么？"
        answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question},
        )
        assert answer.status_code == 200, answer.text
        assert answer.json()["refused"] is False
        assert answer.json()["strategy"] == "vector"
        assert answer.json()["citations"][0]["document_id"] == document["id"]
        assert "数据库连接配置" in answer.json()["citations"][0]["quote"]
        assert answer.json()["retrieval_evidence"][0]["assessment"]["verdict"] == "direct"
        assert answer.json()["retrieval_evidence"][0]["included_in_context"] is True
        assert answer.json()["retrieval_evidence"][0]["cited_in_answer"] is True
        assert (
            answer.json()["retrieval_evidence"][0]["assessment"]["calibrated_probability"] is None
        )

        keyword_answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question, "strategy": "keyword"},
        )
        assert keyword_answer.status_code == 200, keyword_answer.text
        assert keyword_answer.json()["strategy"] == "keyword"
        assert keyword_answer.json()["citations"][0]["keyword_rank"] == 1

        hybrid_answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question, "strategy": "hybrid"},
        )
        assert hybrid_answer.status_code == 200, hybrid_answer.text
        assert hybrid_answer.json()["strategy"] == "hybrid"
        hybrid_citation = hybrid_answer.json()["citations"][0]
        assert hybrid_citation["vector_rank"] == 1
        assert hybrid_citation["keyword_rank"] == 1
        assert hybrid_citation["fused_score"] > 0

        reranked_answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question, "strategy": "hybrid_rerank"},
        )
        assert reranked_answer.status_code == 200, reranked_answer.text
        assert reranked_answer.json()["strategy"] == "hybrid_rerank"
        assert reranked_answer.json()["citations"][0]["rerank_rank"] == 1
        assert reranked_answer.json()["citations"][0]["rerank_score"] == 1

        graph_indexed = await client.post(
            f"/api/v1/documents/{document['id']}/graph-index", headers=headers_a
        )
        assert graph_indexed.status_code == 202
        graph_status = await client.get(f"/api/v1/documents/{document['id']}", headers=headers_a)
        assert graph_status.json()["graph_status"] == "processed"

        graph_answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question, "strategy": "graph"},
        )
        assert graph_answer.status_code == 200, graph_answer.text
        assert graph_answer.json()["strategy"] == "graph"
        assert graph_answer.json()["citations"][0]["graph_rank"] == 1

        graph_hybrid_answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question, "strategy": "graph_hybrid"},
        )
        assert graph_hybrid_answer.status_code == 200, graph_hybrid_answer.text
        graph_hybrid_citation = graph_hybrid_answer.json()["citations"][0]
        assert graph_hybrid_citation["graph_rank"] == 1
        assert graph_hybrid_citation["vector_rank"] == 1
        pool = graph_hybrid_answer.json()["graph_expansion"]
        assert pool["base_count"] == 1 and pool["added_count"] == 0
        assert pool["shared_count"] == 1
        assert graph_hybrid_answer.json()["retrieval_evidence"][0]["retrieval_stage"] == "base"

        graph_view = await client.get(f"/api/v1/knowledge-bases/{kb_id}/graph", headers=headers_a)
        assert graph_view.status_code == 200, graph_view.text
        assert graph_view.json()["node_count"] == 2
        assert graph_view.json()["filtered_edge_count"] == 1
        assert graph_view.json()["edges"][0]["quote_in_source"] is True
        assert graph_view.json()["edges"][0]["semantic_correctness"] == "supported"
        assert graph_view.json()["edges"][0]["retrieval_eligible"] is True
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb_id}/graph", headers=headers_b)
        ).status_code == 404
        assert (
            await client.get(f"/api/v1/knowledge-bases/{kb_id}/graph?q=%25", headers=headers_a)
        ).json()["edges"] == []

        from knowflow.app import evaluate_quality

        dataset_path = tmp_path / "quality.json"
        dataset_path.write_text(
            json.dumps(
                {
                    "version": "test-quality-v1",
                    "source": "unit fixture",
                    "cases": [
                        {
                            "id": "a",
                            "question": question,
                            "should_refuse": False,
                            "reference_answer": "数据库连接配置",
                            "relevant_evidence": [
                                {
                                    "id": "source",
                                    "filename": "handbook.txt",
                                    "contains": "数据库连接配置",
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setitem(evaluate_quality.DATASETS, "engineering", str(dataset_path))
        quality = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/quality-evaluations",
            headers=headers_a,
            json={
                "dataset": "engineering",
                "strategies": ["hybrid", "graph_hybrid"],
                "judge": False,
                "top_k": 3,
            },
        )
        assert quality.status_code == 202, quality.text
        quality_id = quality.json()["run_id"]
        detail = (
            await client.get(
                f"/api/v1/knowledge-bases/{kb_id}/evaluations/{quality_id}", headers=headers_a
            )
        ).json()
        assert detail["status"] == "completed", detail
        assert detail["source_snapshot_unchanged"] is True
        assert detail["variants"]["hybrid"]["cases"][0]["evidence_checks"][0]["ranks"]["base"] == [
            1
        ]
        assert detail["variants"]["graph_hybrid"]["metrics"]["context_complete_evidence"] == 1
        custom = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/quality-evaluations",
            headers=headers_a,
            json={
                "dataset": "custom",
                "custom_dataset": json.loads(dataset_path.read_text("utf-8")),
                "strategies": ["hybrid"],
                "top_k": 3,
                "judge": False,
                "baseline_run_id": quality_id,
            },
        )
        assert custom.status_code == 202, custom.text
        custom_detail = (
            await client.get(
                f"/api/v1/knowledge-bases/{kb_id}/evaluations/{custom.json()['run_id']}",
                headers=headers_a,
            )
        ).json()
        assert custom_detail["baseline_check"]["status"] == "no_regression_found"
        assert (
            await client.post(
                f"/api/v1/knowledge-bases/{kb_id}/quality-evaluations",
                headers=headers_a,
                json={"dataset": "custom", "custom_dataset": {}},
            )
        ).status_code == 422
        assert (
            await client.get(
                f"/api/v1/knowledge-bases/{kb_id}/evaluations/{quality_id}", headers=headers_b
            )
        ).status_code == 404

        async def forged_citation(question, evidence, _settings=None, **kwargs):
            return "Unsupported source [11111111-1111-1111-1111-111111111111]"

        monkeypatch.setattr(api, "generate_answer", forged_citation)
        forged = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question},
        )
        assert forged.json()["refused"] is True

        async def mixed_citations(question, evidence, _settings=None, **kwargs):
            return (
                f"有依据 [{evidence[0]['id']}]，"
                "但还编造了来源 [11111111-1111-1111-1111-111111111111]"
            )

        monkeypatch.setattr(api, "generate_answer", mixed_citations)
        mixed = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question},
        )
        assert mixed.json()["refused"] is True
        assert mixed.json()["citations"] == []

        for broken_marker in ["2354f803-7f33-4cd3-aeb-63e5e6951966", "KF9"]:

            async def malformed_citation(
                question, evidence, _settings=None, *, marker=broken_marker, **kwargs
            ):
                return f"有依据 [{evidence[0]['id']}]，但还有错误来源 [{marker}]"

            monkeypatch.setattr(api, "generate_answer", malformed_citation)
            malformed = await client.post(
                f"/api/v1/knowledge-bases/{kb_id}/query",
                headers=headers_a,
                json={"question": question},
            )
            assert malformed.json()["refused"] is True
            assert malformed.json()["citations"] == []

        async def reject_evidence(question, evidence, _settings=None):
            return {
                row["id"]: RelevanceAssessment(
                    verdict="irrelevant", reason="主题相似但不是所问事实。"
                )
                for row in evidence
            }, RetrievalDiagnostics(status="complete")

        async def must_not_generate(*args):
            raise AssertionError("Rejected evidence must not reach generation")

        monkeypatch.setattr(api, "assess_retrieval", reject_evidence)
        monkeypatch.setattr(api, "generate_answer", must_not_generate)
        rejected = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question},
        )
        assert rejected.json()["refused"] is True
        assert len(rejected.json()["retrieval_evidence"]) == 1
        assert rejected.json()["retrieval_evidence"][0]["included_in_context"] is False

        async def unavailable_assessment(question, evidence, _settings=None):
            return {
                row["id"]: RelevanceAssessment(reason="检测不可用") for row in evidence
            }, RetrievalDiagnostics(status="unavailable")

        monkeypatch.setattr(api, "assess_retrieval", unavailable_assessment)
        unavailable = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question},
        )
        assert unavailable.json()["refused"] is True
        assert unavailable.json()["retrieval_diagnostics"]["status"] == "unavailable"
        monkeypatch.setattr(api, "assess_retrieval", fake_assessment)
        monkeypatch.setattr(api, "generate_answer", fake_answer)

        leaked_answer = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_b,
            json={"question": question},
        )
        assert leaked_answer.status_code == 404

        deleted = await client.delete(f"/api/v1/documents/{document['id']}", headers=headers_a)
        assert deleted.status_code == 204
        no_evidence = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question},
        )
        assert no_evidence.status_code == 200
        assert no_evidence.json()["refused"] is True
        assert no_evidence.json()["citations"] == []
        no_graph_evidence = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/query",
            headers=headers_a,
            json={"question": question, "strategy": "graph"},
        )
        assert no_graph_evidence.status_code == 200
        assert no_graph_evidence.json()["refused"] is True

    async with SessionLocal() as session:
        await session.execute(delete(User).where(User.email.in_(emails)))
        await session.commit()
