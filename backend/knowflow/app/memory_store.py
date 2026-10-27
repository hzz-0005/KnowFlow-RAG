"""Bounded deterministic recall. Memory is a user clue, never document evidence."""

import json
import re
from types import SimpleNamespace

from sqlalchemy import func, or_, select

from knowflow.app.memory_models import UserMemory
from knowflow.app.models import utc_now
from knowflow.app.token_budget import get_token_counter

VERSION = "explicit-scoped-memory-v1"
STYLE = {"concise", "standard", "detailed"}


def dependencies(resolution):
    """Carry transitive alias versions through user-intent references, not only direct recall."""
    result = {}
    trace = resolution.get("memory_context", {})
    used = trace.get("referenced_ids")
    for row in trace.get("selected", []):
        if used is None or row["id"] in used:
            result[row["id"]] = {"id": row["id"], "revision": row["revision"]}
    rows = resolution.get("intent_context", {}).get("referenced_user_inputs", [])
    for row in rows:
        parts = str(row.get("turn_id", "")).split(":")
        if len(parts) == 3 and parts[0] == "memory":
            try:
                revision = int(parts[2])
            except ValueError:
                revision = -1  # invalid lineage must become stale, never trusted
            result[parts[1]] = {"id": parts[1], "revision": revision}
    return list(result.values())


def namespace(kb_id, scope):
    return "user" if scope == "user" else f"kb:{kb_id}"


def visible(user_id, kb_id):
    return (UserMemory.owner_id == user_id) & UserMemory.namespace.in_(["user", f"kb:{kb_id}"])


def matches(key, question):
    # Latin identifiers need word boundaries; CJK and punctuation aliases match literally.
    if re.fullmatch(r"[\w -]+", key, re.ASCII):
        return bool(re.search(r"(?<!\w)" + re.escape(key) + r"(?!\w)", question, re.I))
    return key.casefold() in question.casefold()


async def recall(session, user_id, kb_id, question, settings):
    now = utc_now()
    # Match in SQL before LIMIT; old unrelated rows must not evict a relevant alias.
    rows = list(
        await session.scalars(
            select(UserMemory)
            .where(
                visible(user_id, kb_id),
                UserMemory.status == "active",
                or_(UserMemory.expires_at.is_(None), UserMemory.expires_at > now),
                or_(
                    UserMemory.kind == "preference",
                    # Literal substring, no user-controlled LIKE wildcards.
                    func.strpos(func.lower(question), func.lower(UserMemory.key)) > 0,
                ),
            )
            .order_by(UserMemory.updated_at.desc(), UserMemory.id)
            .limit(101)
        )
    )
    matched = [r for r in rows[:100] if r.kind == "preference" or matches(r.key, question)]
    by_key = {}
    for row in matched:
        category = "preference" if row.kind == "preference" else "clue"
        by_key.setdefault((category, row.key.casefold()), []).append(row)
    conflicted = [group for group in by_key.values() if len({r.content for r in group}) > 1]
    conflicting = {r.id for group in conflicted for r in group}
    trace = {
        "version": VERSION,
        "method": "literal_key_and_typed_preference",
        "budget_tokens": 800,
        "max_items": 4,
        "selected": [],
        "skipped": [],
        "conflicts": [],
        "search_truncated": len(rows) > 100,
        "evidence_status": "not_document_evidence",
    }
    if len(rows) > 100:
        trace["skipped"].append({"reason": "recall_limit_requires_narrower_keys"})
        return [], trace, None
    for group in conflicted:
        trace["conflicts"].append({"key": group[0].key, "ids": [str(r.id) for r in group]})
    selected, styles = [], []
    counter = get_token_counter(settings.chunk_tokenizer_path)
    for row in matched:
        meta = {
            "id": str(row.id),
            "revision": row.revision,
            "kind": row.kind,
            "namespace": row.namespace,
        }
        if row.id in conflicting:
            trace["skipped"].append({**meta, "reason": "conflicting_values"})
            continue
        if any(
            r["kind"] == row.kind and r["key"].casefold() == row.key.casefold() for r in selected
        ):
            continue
        item = {**meta, "key": row.key, "content": row.content}
        tokens = counter.count(json.dumps([*selected, item], ensure_ascii=False))
        if len(selected) >= 4 or tokens > 800:
            trace["skipped"].append({**meta, "reason": "memory_budget"})
            continue
        selected.append(item)
        trace["selected"].append(meta)
        if row.kind == "preference":
            styles.append(row.content)
    trace["selected_tokens"] = counter.count(json.dumps(selected, ensure_ascii=False))
    return selected, trace, styles[0] if styles else None


def memory_intents(selected):
    return [
        SimpleNamespace(
            id=f"memory:{r['id']}:{r['revision']}",
            status="completed",
            question=(f"User-saved {r['kind']} clue (not evidence): {r['key']} = {r['content']}"),
            resolved_question=None,
            resolution={"memory_identity": {k: r[k] for k in ("id", "revision", "namespace")}},
            response={"citations": []},
        )
        for r in selected
        if r["kind"] in {"alias", "note"}
    ]


async def remember_candidate(session, conversation, turn):
    match = re.match(r"\s*(?:记住|remember)\s*[:：]\s*(\S[\s\S]{0,799})$", turn.question, re.I)
    if not match:
        return
    key = f"turn-{turn.id}"
    row = UserMemory(
        owner_id=conversation.owner_id,
        knowledge_base_id=conversation.knowledge_base_id,
        namespace=f"kb:{conversation.knowledge_base_id}",
        key=key,
        kind="note",
        status="candidate",
        content=match.group(1),
        source={
            "type": "explicit_remember_prefix",
            "turn_id": str(turn.id),
            "quote": match.group(1),
            "verification": "user_asserted_not_document_verified",
        },
    )
    session.add(row)
