from uuid import UUID, uuid4

import httpx
import pytest
from knowflow.app import graph_index, graph_semantics, ingestion
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.graph_index import ChunkGraphFacts, EntityFact, RelationFact, RelationQualifier
from knowflow.app.main import app
from knowflow.app.models import Relation, User
from sqlalchemy import delete, update


@pytest.mark.asyncio
async def test_graph_index_persists_same_endpoint_predicate_with_different_explicit_scopes(
    monkeypatch, tmp_path
):
    settings = get_settings()
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    email = f"scopes-{uuid4()}@example.com"
    content = "In staging Alpha may write to Redis. In production Alpha must not write to Redis."

    async def embeddings(texts, *_):
        return [[1.0] + [0.0] * (settings.embedding_dimensions - 1) for _ in texts]

    async def extraction(*_):
        return ChunkGraphFacts(
            entities=[EntityFact(name=n, evidence=n) for n in ["Alpha", "Redis"]],
            relations=[
                RelationFact(
                    source="Alpha",
                    target="Redis",
                    predicate="writes_to",
                    description=quote,
                    evidence=quote,
                    qualifiers=[
                        RelationQualifier(kind="environment", text=scope),
                        RelationQualifier(kind="modality", text=modal),
                    ],
                )
                for quote, scope, modal in [
                    ("In staging Alpha may write to Redis.", "In staging", "may"),
                    ("In production Alpha must not write to Redis.", "In production", "must not"),
                ]
            ],
        )

    monkeypatch.setattr(ingestion, "embed_texts", embeddings)
    monkeypatch.setattr(graph_index, "extract_chunk_graph", extraction)

    review_calls = []

    async def reviewer(graph, _settings, **_):
        review_calls.append(len(graph["edges"]))
        return {
            "units": [
                {"id": e["id"], "verdict": "supported", "reason": "fixture"} for e in graph["edges"]
            ]
        }

    monkeypatch.setattr(graph_semantics, "review_graph_edges", reviewer)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            token = (
                await client.post(
                    "/api/v1/auth/register", json={"email": email, "password": "scope-password-123"}
                )
            ).json()["access_token"]
            headers = {"Authorization": "Bearer " + token}
            kb = (
                await client.post(
                    "/api/v1/knowledge-bases", headers=headers, json={"name": "Scopes"}
                )
            ).json()["id"]
            doc = (
                await client.post(
                    f"/api/v1/knowledge-bases/{kb}/documents",
                    headers=headers,
                    files={"file": ("scopes.md", content.encode(), "text/markdown")},
                )
            ).json()["id"]
            assert (
                await client.post(f"/api/v1/documents/{doc}/graph-index", headers=headers)
            ).status_code == 202
            edges = (
                await client.get(f"/api/v1/knowledge-bases/{kb}/graph", headers=headers)
            ).json()["edges"]
            assert len(edges) == 2
            assert {e["qualifiers"]["items"][0]["text"] for e in edges} == {
                "In staging",
                "In production",
            }
            assert all(e["qualifiers"]["status"] == "reported" for e in edges)
            assert review_calls == [2]
            await client.post(f"/api/v1/documents/{doc}/graph-index", headers=headers)
            assert review_calls == [2]
            async with SessionLocal() as session:
                await session.execute(
                    update(Relation)
                    .where(Relation.document_id == UUID(doc))
                    .values(semantic_status="unsupported")
                )
                await session.commit()
            view = (
                await client.get(
                    f"/api/v1/knowledge-bases/{kb}/graph?status=excluded", headers=headers
                )
            ).json()
            assert view["filtered_edge_count"] == 2 and view["eligible_edge_count"] == 0
            assert all(not e["retrieval_eligible"] for e in view["edges"])
            assert (
                await client.get(
                    f"/api/v1/knowledge-bases/{kb}/graph?status=supported", headers=headers
                )
            ).json()["edges"] == []
    finally:
        async with SessionLocal() as session:
            await session.execute(delete(User).where(User.email == email))
            await session.commit()
