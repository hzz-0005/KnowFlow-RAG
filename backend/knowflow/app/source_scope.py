"""Literal source selection and request-local, SQL-before-LIMIT read constraints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.orm import with_loader_criteria

from knowflow.app.models import Chunk, Document, DocumentStatus, EntityMention, Relation

VERSION = "literal-document-scope-v1"
EXT = r"(?:pdf|md|txt|docx?|xlsx?|csv|tsv|pptx?|html?|json|rtf|odt|ods)"
TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.-])[A-Za-z0-9_][A-Za-z0-9_.-]*\." + EXT + r"(?![A-Za-z0-9_])", re.I
)
UNICODE_TOKEN = re.compile(r"[^\W\d][\w.-]*\." + EXT + r"(?![A-Za-z0-9_])", re.I)
QUOTED = re.compile(r'["“`「]([^"”`」\n]+\.' + EXT + r')["”`」]', re.I)
# "除了 A，其他项目…" excludes A just as "排除 A" does, so the two phrasings must
# not be treated differently by the scope resolver.
NEGATIVE = re.compile(
    r"(?:不要(?:查|看|搜|检索)?|不看|不查|排除|不包括|而非|不是|除了|除去|剔除|不算|不计|"
    r"except|exclude|not\s+(?:in|from)|other\s+than)\s*[\"“`「]*$",
    re.I,
)


@dataclass
class SourceScope:
    trace: dict
    document_ids: tuple[UUID, ...] | None = None


def retrieval_text(question, source_scope):
    """File labels already enforced in SQL must not dominate lexical/vector ranking.

    Only exact catalog filenames are removed. The original user input, numeric
    conditions outside those labels, generation and review remain untouched.
    """
    if source_scope.trace.get("status") != "filtered" or re.search(
        r"文件名|命名|\b(?:filename|file name|path)\b", question, re.I
    ):
        return question
    labels = [row["filename"] for row in source_scope.trace.get("documents", [])]
    labels.extend(source_scope.trace.get("excluded_filenames") or [])
    text = question
    for label in sorted(set(labels), key=len, reverse=True):
        for match in reversed(occurrences(text, label)):
            text = text[: match.start()] + " " + text[match.end() :]
    text = re.sub(r"\s+", " ", text).strip()
    return text if re.search(r"[\w\u4e00-\u9fff]", text) else question


def occurrences(question, filename):
    pattern = re.escape(filename)
    if filename[0].isascii():
        pattern = r"(?<![A-Za-z0-9_.-])" + pattern
    return list(re.finditer(pattern + r"(?![A-Za-z0-9_.-])", question, re.I))


async def resolve_source_scope(session, kb_id, question):
    trace = {
        "version": VERSION,
        "status": "unscoped",
        "documents": [],
        "excluded_filenames": [],
        "unknown_filenames": [],
        "note": "仅按检索问题中逐字出现的文件名限定来源；不是语义相关性或版本权威判断。",
    }
    if not re.search(r"\." + EXT + r"(?![A-Za-z0-9_])", question, re.I):
        return SourceScope(trace)
    # Match metadata in the current KB only; no corpus text or provider request.
    rows = (
        await session.execute(
            select(Document.id, Document.filename, Document.status)
            .where(
                Document.knowledge_base_id == kb_id,
                func.strpos(func.lower(question), func.lower(Document.filename)) > 0,
            )
            .order_by(Document.id)
            .limit(65)
        )
    ).all()
    if len(rows) > 64:
        return SourceScope(
            {**trace, "status": "ambiguous", "reason": "too_many_matching_documents"}, ()
        )
    known = {}
    spans = []
    for did, filename, status in rows:
        matches = occurrences(question, filename)
        if not matches:
            continue
        known.setdefault(filename.casefold(), []).append((did, filename, status))
        spans.extend((m.start(), m.end()) for m in matches)
    unknown = []
    for match in [
        *QUOTED.finditer(question),
        *TOKEN.finditer(question),
        *UNICODE_TOKEN.finditer(question),
    ]:
        name = match.group(1) if match.re is QUOTED else match.group()
        if name.casefold() not in known and not any(
            (a <= match.start() and match.end() <= b) or (match.start() <= a and match.end() == b)
            for a, b in spans
        ):
            unknown.append(name)
    trace["unknown_filenames"] = list(dict.fromkeys(unknown))[:8]
    if unknown:
        return SourceScope(
            {**trace, "status": "unavailable", "reason": "filename_not_in_current_knowledge_base"},
            (),
        )
    if not known:
        return SourceScope(trace)
    selected = []
    for variants in known.values():
        filename = variants[0][1]
        polarities = set()
        for match in occurrences(question, filename):
            prefix = re.split(r"[，,。；;?!？！\n]", question[: match.start()])[-1]
            polarities.add(bool(NEGATIVE.search(prefix)))
        if len(polarities) > 1:
            return SourceScope(
                {**trace, "status": "ambiguous", "reason": "conflicting_file_selection"}, ()
            )
        if True in polarities:
            trace["excluded_filenames"].append(filename)
            continue
        if len(variants) > 1:
            return SourceScope(
                {
                    **trace,
                    "status": "ambiguous",
                    "reason": "duplicate_filename",
                    "filename": filename,
                },
                (),
            )
        did, filename, status = variants[0]
        if status != DocumentStatus.PROCESSED:
            return SourceScope(
                {
                    **trace,
                    "status": "unavailable",
                    "reason": "selected_document_not_processed",
                    "filename": filename,
                },
                (),
            )
        selected.append((did, filename))
    if not selected:
        return SourceScope(
            {**trace, "status": "ambiguous", "reason": "only_exclusions_supplied"}, ()
        )
    trace.update(
        status="filtered",
        documents=[{"document_id": str(did), "filename": name} for did, name in selected],
    )
    return SourceScope(trace, tuple(did for did, _ in selected))


async def bind_candidate_scope(session, kb_id, source_scope, document_ids):
    """Intersect server-held candidate identities with any explicit file restriction."""
    ids = set(document_ids)
    if source_scope.document_ids is not None:
        ids.intersection_update(source_scope.document_ids)
    rows = (
        await session.execute(
            select(Document.id, Document.filename).where(
                Document.id.in_(ids),
                Document.knowledge_base_id == kb_id,
                Document.status == DocumentStatus.PROCESSED,
            )
        )
    ).all()
    if not ids or len(rows) != len(ids):
        return SourceScope(
            {
                **source_scope.trace,
                "status": "unavailable",
                "reason": "candidate_source_unavailable_or_scope_conflict",
            },
            (),
        )
    return SourceScope(
        {
            **source_scope.trace,
            "status": "filtered",
            "selection_basis": "grounded_candidate_document",
            "note": (
                "按有原文支持的候选文档限定来源；与用户明确文件限制取交集，"
                "不代表相关性或语义审核通过。"
            ),
            "documents": [{"document_id": str(did), "filename": name} for did, name in rows],
        },
        tuple(ids),
    )


class ScopedReadSession:
    """Explicit per-call reader: no session events, globals, mutation or fallback.

    ORM loader criteria affect WHERE/ON clauses before ordering and LIMIT,
    including graph seeds, text expansions and nearest-distance recovery.
    """

    def __init__(self, session, kb_id, document_ids):
        self.session = session
        self.kb_id = kb_id
        self.document_ids = tuple(document_ids)

    def statement(self, statement):
        ids, kb = self.document_ids, self.kb_id
        return statement.options(
            with_loader_criteria(
                Document,
                and_(
                    Document.id.in_(ids),
                    Document.knowledge_base_id == kb,
                    Document.status == DocumentStatus.PROCESSED,
                ),
                include_aliases=True,
            ),
            *[
                with_loader_criteria(
                    model,
                    and_(model.document_id.in_(ids), model.knowledge_base_id == kb),
                    include_aliases=True,
                )
                for model in (Chunk, EntityMention, Relation)
            ],
        )

    async def execute(self, statement, *args, **kwargs):
        return await self.session.execute(self.statement(statement), *args, **kwargs)
