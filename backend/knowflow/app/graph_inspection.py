"""Inspect actual stored graph edges and their source evidence, scoped to a KB."""

from sqlalchemy import func, or_, select
from sqlalchemy.orm import aliased

from knowflow.app.graph_index import _supported_quote
from knowflow.app.models import Chunk, Document, Entity, EntityMention, Relation


async def inspect_graph(
    session, kb_id, *, document_id=None, query="", offset=0, limit=30, status="all"
):
    source, target = aliased(Entity), aliased(Entity)
    conditions = [Relation.knowledge_base_id == kb_id, Document.status == "processed"]
    if document_id:
        conditions.append(Relation.document_id == document_id)
    if query:
        # Autoescape literal user text rather than treating %/_ as wildcards.
        conditions.append(
            or_(
                source.display_name.icontains(query, autoescape=True),
                target.display_name.icontains(query, autoescape=True),
                Relation.predicate.icontains(query, autoescape=True),
            )
        )
    statement = (
        select(Relation, source.display_name, target.display_name, Chunk, Document.filename)
        .join(source, source.id == Relation.source_entity_id)
        .join(target, target.id == Relation.target_entity_id)
        .join(Chunk, Chunk.id == Relation.source_chunk_id)
        .join(Document, Document.id == Relation.document_id)
        .where(*conditions)
    )
    review_counts = dict(
        (
            await session.execute(
                select(Relation.semantic_status, func.count())
                .select_from(Relation)
                .join(source, source.id == Relation.source_entity_id)
                .join(target, target.id == Relation.target_entity_id)
                .join(Document, Document.id == Relation.document_id)
                .where(*conditions)
                .group_by(Relation.semantic_status)
            )
        ).all()
    )
    if status == "excluded":
        statement = statement.where(Relation.semantic_status.in_(["unsupported", "contradicted"]))
    elif status != "all":
        statement = statement.where(Relation.semantic_status == status)
    total = await session.scalar(select(func.count()).select_from(statement.subquery()))
    records = (
        await session.execute(
            statement.order_by(Document.filename, Chunk.ordinal, Relation.id)
            .offset(offset)
            .limit(limit)
        )
    ).all()
    node_query = (
        select(func.count(func.distinct(EntityMention.entity_id)))
        .join(Document, Document.id == EntityMention.document_id)
        .where(EntityMention.knowledge_base_id == kb_id, Document.status == "processed")
    )
    if document_id:
        node_query = node_query.where(EntityMention.document_id == document_id)
    documents = list(
        await session.scalars(select(Document).where(Document.knowledge_base_id == kb_id))
    )
    rows = []
    for relation, source_name, target_name, chunk, filename in records:
        rows.append(
            {
                "id": str(relation.id),
                "source": source_name,
                "target": target_name,
                "predicate": relation.predicate,
                "description": relation.description,
                "evidence": relation.evidence,
                "filename": filename,
                "document_id": str(relation.document_id),
                "chunk_id": str(chunk.id),
                "page_number": chunk.page_number,
                "source_spans": chunk.source_spans,
                "quote_in_source": _supported_quote(chunk.content, relation.evidence),
                "endpoints_in_source": _supported_quote(chunk.content, source_name)
                and _supported_quote(chunk.content, target_name),
                "semantic_correctness": relation.semantic_status,
                "semantic_review": relation.semantic_review,
                "retrieval_eligible": relation.semantic_status == "supported",
                "qualifiers": relation.qualifiers,
                "source_text": chunk.content,
                "source_section": chunk.section,
            }
        )
    return {
        "node_count": await session.scalar(node_query),
        "filtered_edge_count": total,
        "eligible_edge_count": review_counts.get("supported", 0),
        "review_counts": review_counts,
        "offset": offset,
        "limit": limit,
        "edges": rows,
        "document_states": [
            {
                "filename": d.filename,
                "status": d.status,
                "graph_status": d.graph_status,
                "graph_error": d.graph_error,
                "graph_index": (d.parsing_report or {}).get("graph_index"),
            }
            for d in documents
            if not document_id or d.id == document_id
        ],
        "note": (
            "节点是抽取的实体名称，边是带原文来源的关系候选。引文存在不等于关系方向、"
            "条件或谓词正确。仅模型判支持且来源校验通过的关系参与图遍历；其他候选保留供检查。"
            "模型判定仍可能出错，不保证同名消歧或 PDF 图片箭头理解。"
        ),
    }
