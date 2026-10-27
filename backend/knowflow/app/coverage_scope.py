"""Freeze a bounded document sampling scope without widening caller restrictions."""

from sqlalchemy import func, select

from knowflow.app.models import Document, DocumentStatus

MAX_DOCUMENTS = 20


async def plan_coverage(session, kb_id, source_scope):
    predicates = [
        Document.knowledge_base_id == kb_id,
        Document.status == DocumentStatus.PROCESSED,
    ]
    excluded = source_scope.trace.get("excluded_filenames") or []
    if excluded:
        predicates.append(Document.filename.not_in(excluded))
    if source_scope.document_ids is not None:
        predicates.append(Document.id.in_(source_scope.document_ids))
        scope = (
            "all_minus_excluded"
            if source_scope.trace.get("selection_basis") == "excluded_named_documents"
            else "explicit_documents"
        )
    else:
        scope = "all_processed_documents"
    total = await session.scalar(select(func.count()).select_from(Document).where(*predicates))
    rows = list(
        (
            await session.execute(
                select(Document.id, Document.filename)
                .where(*predicates)
                .order_by(Document.filename, Document.id)
                .limit(MAX_DOCUMENTS)
            )
        ).all()
    )
    trace = {
        "version": "bounded-document-sampling-v2",
        "mode": "sweep",
        "scope": scope,
        "eligible_documents": total,
        "checked": len(rows),
        "omitted_documents": max(0, total - len(rows)),
        "scope_complete": len(rows) == total,
        "content_exhaustive": False,
        "sampling_limit": MAX_DOCUMENTS,
        "hit": 0,
        "cited_documents": 0,
        "documents": [
            {
                "document_id": str(row.id),
                "filename": row.filename,
                "evidence_chunks": 0,
                "cited_in_answer": False,
            }
            for row in rows
        ],
        "note": (
            "这是限定文件范围内的有界片段抽样，不是全文阅读。"
            "未召回或未被答案引用不代表文件没有相关内容；"
            "超过文件上限的部分未检查，distance_gate_fallback表示阈值外近邻取样。"
        ),
    }
    return rows, trace
