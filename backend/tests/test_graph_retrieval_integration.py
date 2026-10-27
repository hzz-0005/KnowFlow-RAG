import uuid

import pytest
from knowflow.app.config import get_settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import (
    Chunk,
    Document,
    DocumentStatus,
    Entity,
    EntityMention,
    KnowledgeBase,
    Relation,
    User,
)
from knowflow.app.retrieval import _graph_search, retrieve
from sqlalchemy import delete, update


def _embedding() -> list[float]:
    vector = [0.0] * get_settings().embedding_dimensions
    vector[0] = 1.0
    return vector


@pytest.mark.asyncio
async def test_graph_retrieval_follows_two_hop_path_across_documents() -> None:
    email = f"graph-{uuid.uuid4()}@example.com"
    async with SessionLocal() as session:
        user = User(email=email, password_hash="unused")
        knowledge_base = KnowledgeBase(owner=user, name="Graph test")
        session.add_all([user, knowledge_base])
        await session.flush()

        documents = [
            Document(
                knowledge_base_id=knowledge_base.id,
                uploaded_by=user.id,
                filename="nova-service.md",
                storage_key=f"{uuid.uuid4().hex}.md",
                media_type="text/markdown",
                file_size=32,
                sha256=uuid.uuid4().hex,
                status=DocumentStatus.PROCESSED,
            ),
            Document(
                knowledge_base_id=knowledge_base.id,
                uploaded_by=user.id,
                filename="hosting.md",
                storage_key=f"{uuid.uuid4().hex}.md",
                media_type="text/markdown",
                file_size=36,
                sha256=uuid.uuid4().hex,
                status=DocumentStatus.PROCESSED,
            ),
        ]
        session.add_all(documents)
        await session.flush()
        chunks = [
            Chunk(
                document_id=documents[0].id,
                knowledge_base_id=knowledge_base.id,
                ordinal=0,
                content="Nova uses the API Gateway.",
                search_text="Nova uses the API Gateway",
                embedding=_embedding(),
                embedding_model=get_settings().embedding_model,
            ),
            Chunk(
                document_id=documents[1].id,
                knowledge_base_id=knowledge_base.id,
                ordinal=0,
                content="The API Gateway is hosted on Orion Cloud.",
                search_text="API Gateway hosted Orion Cloud",
                embedding=_embedding(),
                embedding_model=get_settings().embedding_model,
            ),
        ]
        session.add_all(chunks)
        await session.flush()

        nova = Entity(
            knowledge_base_id=knowledge_base.id,
            normalized_name="nova",
            display_name="Nova",
        )
        gateway = Entity(
            knowledge_base_id=knowledge_base.id,
            normalized_name="api gateway",
            display_name="API Gateway",
        )
        cloud = Entity(
            knowledge_base_id=knowledge_base.id,
            normalized_name="orion cloud",
            display_name="Orion Cloud",
        )
        session.add_all([nova, gateway, cloud])
        await session.flush()
        session.add_all(
            [
                EntityMention(
                    knowledge_base_id=knowledge_base.id,
                    document_id=documents[0].id,
                    chunk_id=chunks[0].id,
                    entity_id=nova.id,
                    surface_name="Nova",
                    evidence="Nova",
                ),
                EntityMention(
                    knowledge_base_id=knowledge_base.id,
                    document_id=documents[0].id,
                    chunk_id=chunks[0].id,
                    entity_id=gateway.id,
                    surface_name="API Gateway",
                    evidence="API Gateway",
                ),
                EntityMention(
                    knowledge_base_id=knowledge_base.id,
                    document_id=documents[1].id,
                    chunk_id=chunks[1].id,
                    entity_id=gateway.id,
                    surface_name="API Gateway",
                    evidence="API Gateway",
                ),
                EntityMention(
                    knowledge_base_id=knowledge_base.id,
                    document_id=documents[1].id,
                    chunk_id=chunks[1].id,
                    entity_id=cloud.id,
                    surface_name="Orion Cloud",
                    evidence="Orion Cloud",
                ),
                Relation(
                    knowledge_base_id=knowledge_base.id,
                    document_id=documents[0].id,
                    source_chunk_id=chunks[0].id,
                    source_entity_id=nova.id,
                    target_entity_id=gateway.id,
                    predicate="uses",
                    semantic_status="supported",
                    description="Nova uses the API Gateway.",
                    evidence="Nova uses the API Gateway.",
                    confidence=0.95,
                ),
                Relation(
                    knowledge_base_id=knowledge_base.id,
                    document_id=documents[1].id,
                    source_chunk_id=chunks[1].id,
                    source_entity_id=gateway.id,
                    target_entity_id=cloud.id,
                    predicate="hosted_on",
                    semantic_status="supported",
                    description="The API Gateway is hosted on Orion Cloud.",
                    evidence="The API Gateway is hosted on Orion Cloud.",
                    confidence=0.9,
                ),
            ]
        )
        await session.commit()

        hits = await retrieve(
            session,
            knowledge_base.id,
            "Where is Nova hosted?",
            "graph",
            top_k=5,
        )
        by_filename = {hit.filename: hit for hit in hits}
        assert set(by_filename) == {"nova-service.md", "hosting.md"}
        assert by_filename["hosting.md"].graph_rank == 2
        assert "Orion Cloud" in by_filename["hosting.md"].chunk.content
        # A paraphrase with no literal entity name can start from an already
        # retrieved source. This follows stored identities rather than guessing
        # that a short query name is an alias for a longer entity.
        paraphrase = "执行链最终落在哪里？"
        assert not await _graph_search(session, knowledge_base.id, paraphrase, 5)
        seeded = await _graph_search(
            session, knowledge_base.id, paraphrase, 5, seed_chunk_ids=[chunks[0].id]
        )
        assert "hosting.md" in {hit.filename for hit in seeded}
        assert not await _graph_search(
            session, knowledge_base.id, paraphrase, 5, seed_chunk_ids=[uuid.uuid4()]
        )
        # An unknown second hop cannot extend an otherwise approved first hop.
        await session.execute(
            update(Relation)
            .where(Relation.document_id == documents[1].id)
            .values(semantic_status="unknown")
        )
        hits = await retrieve(session, knowledge_base.id, "Where is Nova hosted?", "graph", top_k=5)
        assert {h.filename for h in hits} == {"nova-service.md"}
        seeded = await _graph_search(
            session, knowledge_base.id, paraphrase, 5, seed_chunk_ids=[chunks[0].id]
        )
        assert {hit.filename for hit in seeded} == {"nova-service.md"}
        # Rejecting the first edge also excludes entity-only candidates; mentions
        # must not bypass the semantic gate and silently restore the bad path.
        await session.execute(
            update(Relation)
            .where(Relation.document_id == documents[0].id)
            .values(semantic_status="unsupported")
        )
        assert not await retrieve(
            session, knowledge_base.id, "Where is Nova hosted?", "graph", top_k=5
        )

    async with SessionLocal() as session:
        await session.execute(delete(User).where(User.email == email))
        await session.commit()
