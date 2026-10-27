"""Server-owned clarification metadata; never a source of entity bindings or facts."""

from sqlalchemy import select

from knowflow.app.models import Document, utc_now

VERSION = "clarification-hints-v1"
MEMORY_REASONS = {
    "memory_unavailable": (
        "此前引用的记忆已删除或不可访问",
        "A previously used memory is unavailable",
    ),
    "memory_scope_changed": (
        "此前引用的记忆已不在当前权限范围内",
        "A previously used memory is outside the current scope",
    ),
    "memory_disabled": ("此前引用的记忆已停用", "A previously used memory has been disabled"),
    "memory_expired": ("此前引用的记忆已过期", "A previously used memory has expired"),
    "memory_revised": (
        "此前引用的记忆已更正，旧指代不再沿用",
        "A previously used memory was revised; its old binding is no longer used",
    ),
    "memory_inactive": (
        "此前引用的记忆未处于启用状态",
        "A previously used memory is no longer active",
    ),
}


def memory_invalidation(memory, dependency, owner_id, kb_id):
    """One deterministic reason per dependency; disclose neither old nor new content."""
    if memory is None:
        return "memory_unavailable"
    if memory.owner_id != owner_id or memory.namespace not in {"user", f"kb:{kb_id}"}:
        return "memory_scope_changed"
    if memory.status == "disabled":
        return "memory_disabled"
    if memory.status != "active":
        return "memory_inactive"
    if memory.expires_at is not None and memory.expires_at <= utc_now():
        return "memory_expired"
    if memory.revision != dependency["revision"]:
        return "memory_revised"
    return None


def restatement_reasons(intent):
    # Whitelist saved fields: historical/free-form errors must never become prompt text.
    return sorted(
        {
            r
            for r in intent.get("restatement_reasons", [])
            if isinstance(r, str) and r in MEMORY_REASONS
        }
    )


def memory_explanation(reasons, question):
    english = not any("\u4e00" <= c <= "\u9fff" for c in question)
    return ("; " if english else "；").join(MEMORY_REASONS[r][int(english)] for r in reasons)


async def document_hints(session, kb_id):
    """Caller has already checked KB ownership; no excerpts, search or provider calls."""
    rows = list(
        await session.execute(
            select(Document.id, Document.filename)
            .where(Document.knowledge_base_id == kb_id, Document.status == "processed")
            .order_by(Document.filename, Document.id)
            .limit(6)
        )
    )
    return {
        "version": VERSION,
        "kind": "available_documents_not_relevance_ranked",
        "scope": "current_knowledge_base",
        "limit": 5,
        "has_more": len(rows) > 5,
        "documents": [{"document_id": str(r.id), "filename": r.filename} for r in rows[:5]],
    }
