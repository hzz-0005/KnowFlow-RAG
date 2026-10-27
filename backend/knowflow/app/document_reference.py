"""Bound literal document identities before asking a model to interpret history.

Metadata selects a source, never proves the document's contents or entity facts.
No fuzzy spelling guesses, corpus-global aliases or previous-answer inference.
"""

import re
from functools import lru_cache
from pathlib import PurePath

from sqlalchemy import select

from knowflow.app.models import Document, DocumentStatus
from knowflow.app.source_scope import NEGATIVE, SourceScope, resolve_source_scope

VERSION = "current-document-reference-v1"
PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_ -][A-Za-z][A-Za-z0-9]*)*")
GENERIC = {"readme", "document", "report", "summary", "notes", "results", "project", "test"}
REQUEST = re.compile(
    r"查询|查一下|查找|介绍|解释|总结|概括|干嘛|什么|多少|多久|为何|为什么|怎么|如何|计划|用途|作用|"
    r"哪些|列出|比较|对比|\b(?:what|why|how|when|describe|explain|summari[sz]e|overview|find|show|list|compare)\b",
    re.I,
)
UNRESOLVED = re.compile(
    r"另(?:一)?(?:个|份|套|篇)|前者|后者|第[一二三\d]+个|\b(?:other|another|former|latter)\b", re.I
)
CORRECTION = re.compile(r"我(?:问|说|指)的是|不是.{0,80}(?:而是|是)|你为什么|你怎么")
HISTORY_DEPENDENCY = re.compile(
    r"刚才|上(?:轮|次|回)|前一个|上一个|前文|沿用|照着|"
    r"之前(?:的)?(?:条件|规则|阈值|要求|限制|回答|问题)|"
    r"\b(?:previous|earlier|as before|same (?:rule|condition|threshold))\b",
    re.I,
)
SOURCE_HISTORY_EXCLUSION = re.compile(
    r"(?:别|不要|不用|不再)(?:带上|引用|检索|查|看)(?:前一个|上一个|之前|刚才)(?:的)?(?:文档|文件|项目)|"
    r"\bwithout (?:the )?(?:previous|last) (?:project|document|file)\b",
    re.I,
)
DOCUMENT_FOLLOWUP = re.compile(
    r"这(?:份|篇|个)(?:文档|文件|报告|论文|材料)?|那(?:份|篇)(?:文档|文件|报告|论文|材料)?|"
    r"该文档|它|\b(?:it|this|that)\b",
    re.I,
)


CJK = re.compile(r"[\u4e00-\u9fff]+")
CJK_STOP = {
    "项目总结",
    "项目详解",
    "项目介绍",
    "使用说明",
    "文档说明",
    "测试报告",
    "验收报告",
    "项目文档",
    "技术方案",
    "设计文档",
    "会议记录",
    "工作总结",
}


@lru_cache(maxsize=2048)
def _cjk_pieces(alias, min_len=5):
    """Distinctive Chinese fragments of a title; generic document words excluded."""
    pieces = set()
    for run in CJK.findall(alias):
        for size in range(len(run), min_len - 1, -1):
            for start in range(len(run) - size + 1):
                piece = run[start : start + size]
                if piece not in CJK_STOP:
                    pieces.add(piece)
    return frozenset(pieces)


def title_aliases(filename):
    """Only whole stems, leading identifiers and separator-delimited titles."""
    stem = PurePath(filename).stem
    aliases = {stem}
    prefix = PREFIX.match(stem)
    if prefix:
        leading = prefix.group().rstrip("_ -")
        aliases.add(leading)
        first = re.split(r"[_ -]", leading)[0]
        if len(first) >= 6:
            aliases.add(first)
    for part in re.split(r"[_\-—·.()（）\[\]【】]+", stem):
        if CJK.search(part) and len(part) >= 4 and part not in CJK_STOP:
            aliases.add(part)
    return sorted(a for a in aliases if len(a) >= 4 and a.casefold() not in GENERIC)


def alias_matches(question, alias):
    if CJK.search(alias):
        pieces = sorted(_cjk_pieces(alias), key=lambda value: (-len(value), value))
        if not pieces:
            return []
        pattern = "|".join(re.escape(piece) for piece in pieces[:400])
        return list(re.finditer(pattern, question))
    parts = re.split(r"[_ -]+", alias)
    pattern = r"[_ -]+".join(re.escape(p) for p in parts)
    return list(re.finditer(r"(?<![A-Za-z0-9_])" + pattern + r"(?![A-Za-z0-9_])", question, re.I))


async def resolve_document_reference(session, kb_id, question):
    literal = await resolve_source_scope(session, kb_id, question)
    only_exclusions = literal.trace.get("reason") == "only_exclusions_supplied"
    if literal.trace["status"] != "unscoped" and not only_exclusions:
        return literal
    if getattr(session, "execute", None) is None:
        # Pipeline unit tests may pass no session or a metadata-only stub; without
        # the KB catalog a title is only matched when written literally with a suffix.
        return literal
    rows = (
        await session.execute(
            select(Document.id, Document.filename, Document.status)
            .where(Document.knowledge_base_id == kb_id)
            .order_by(Document.id)
            .limit(257)
        )
    ).all()
    if len(rows) > 256:
        return literal  # No truncated catalog is treated as a unique match.
    if only_exclusions:
        excluded_names = set(literal.trace["excluded_filenames"])
        remaining = [
            (did, filename)
            for did, filename, status in rows
            if status == DocumentStatus.PROCESSED and filename not in excluded_names
        ]
        if not remaining:
            return literal
        return SourceScope(
            {
                **literal.trace,
                "version": VERSION,
                "status": "filtered",
                "reason": None,
                "selection_basis": "excluded_named_documents",
                "documents": [
                    {"document_id": str(did), "filename": name} for did, name in remaining
                ],
            },
            tuple(did for did, _ in remaining),
        )
    matches, excluded, match_spans = {}, [], {}
    for did, filename, status in rows:
        for alias in title_aliases(filename):
            occurrences = alias_matches(question, alias)
            if not occurrences:
                continue
            positive = []
            for match in occurrences:
                prefix = re.split(r"[，,。；;?!？！\n]", question[: match.start()])[-1]
                if NEGATIVE.search(prefix):
                    excluded.append(filename)
                else:
                    positive.append(match.group())
                    match_spans.setdefault(did, []).append((match.start(), match.end()))
            if positive:
                matches[did] = (filename, status, positive[0])
    from knowflow.app.document_summary import named_object

    unknown_token = named_object(question, [row.filename for row in rows])
    if not matches:
        if excluded:
            excluded_names = set(excluded)
            remaining = [
                (did, filename)
                for did, filename, status in rows
                if status == DocumentStatus.PROCESSED and filename not in excluded_names
            ]
            if remaining:
                return SourceScope(
                    {
                        **literal.trace,
                        "version": VERSION,
                        "status": "filtered",
                        "selection_basis": "excluded_named_documents",
                        "excluded_filenames": sorted(excluded_names),
                        "documents": [
                            {"document_id": str(did), "filename": filename}
                            for did, filename in remaining
                        ],
                    },
                    tuple(did for did, _filename in remaining),
                )
            return SourceScope(
                {
                    **literal.trace,
                    "version": VERSION,
                    "status": "ambiguous",
                    "excluded_filenames": sorted(excluded_names),
                    "reason": "only_exclusions_supplied",
                },
                (),
            )
        return SourceScope(
            {**literal.trace, "version": VERSION, "unknown_title_token": unknown_token},
            literal.document_ids,
        )
    trace = {
        **literal.trace,
        "version": VERSION,
        "selection_basis": "literal_title_identifier",
        "note": "仅绑定当前输入中可唯一匹配的文件标题标识；文件身份不证明正文事实。",
        "excluded_filenames": sorted(set(excluded)),
    }
    if any(name in excluded for name, _, _ in matches.values()):
        return SourceScope(
            {**trace, "status": "ambiguous", "reason": "named_and_excluded_conflict"}, ()
        )
    if len(matches) > 1:
        # Two versions sharing the same literal alias are one ambiguous mention,
        # not two independently named sources. Each selected file needs its own
        # non-overlapping mention; full filenames already resolved above.
        independently_named = all(
            any(
                all(
                    not (start < other_end and other_start < end)
                    for other, spans in match_spans.items()
                    if other != did
                    for other_start, other_end in spans
                )
                for start, end in match_spans.get(did, [])
            )
            for did in matches
        )
        processed = [
            (did, filename, alias)
            for did, (filename, status, alias) in matches.items()
            if status == DocumentStatus.PROCESSED
        ]
        if independently_named and len(processed) == len(matches) and len(processed) <= 8:
            # A comparison question may legitimately name two or three files.
            return SourceScope(
                {
                    **trace,
                    "status": "filtered",
                    "selection_basis": "multiple_literal_titles",
                    "documents": [
                        {"document_id": str(did), "filename": filename}
                        for did, filename, _alias in processed
                    ],
                },
                tuple(did for did, _filename, _alias in processed),
            )
        return SourceScope(
            {**trace, "status": "ambiguous", "reason": "non_unique_title_identifier"}, ()
        )
    did, (filename, status, alias) = next(iter(matches.items()))
    if status != DocumentStatus.PROCESSED:
        return SourceScope(
            {**trace, "status": "unavailable", "reason": "selected_document_not_processed"}, ()
        )
    return SourceScope(
        {
            **trace,
            "status": "filtered",
            "matched_title": alias,
            "documents": [{"document_id": str(did), "filename": filename}],
        },
        (did,),
    )


def named_document_resolution(question, source_scope, version):
    """A self-contained lookup naming current files needs no old-topic rewriting.

    Corrections, missing ordinals and multi-object dependencies still go through
    the history interpreter, now with this source identity supplied.
    """
    if (
        source_scope.trace["status"] != "filtered"
        or not 1 <= len(source_scope.document_ids or ()) <= 8
        or not REQUEST.search(question)
        or UNRESOLVED.search(question)
        or CORRECTION.search(question)
    ):
        return None
    from knowflow.app.document_summary import NEGATIVE as NO_SUMMARY
    from knowflow.app.document_summary import document_cue

    # Excluding the previous source needs no old facts. Positive references to
    # earlier rules/thresholds still require history; do not drop any user text.
    if HISTORY_DEPENDENCY.search(SOURCE_HISTORY_EXCLUSION.sub("", question)):
        return None
    multiple = len(source_scope.document_ids) > 1
    if multiple and not (
        source_scope.trace.get("selection_basis") == "multiple_literal_titles"
        or all(
            doc["filename"].casefold() in question.casefold()
            for doc in source_scope.trace["documents"]
        )
    ):
        return None  # An exclusion-derived set is not a set of current literal names.
    # A filename containing '总结' is not itself a request to summarize.
    text = question
    for doc in source_scope.trace["documents"]:
        text = re.sub(re.escape(doc["filename"]), "这份文档", text, flags=re.I)
    if multiple and document_cue(text) and not NO_SUMMARY.search(text):
        return None  # Keep multi-document whole-summary planning on its existing path.
    return {
        "version": version,
        "status": "standalone",
        "question": question,
        "original_question": question,
        "reference_turn_ids": [],
        "request_kind": "document_summary"
        if document_cue(text) and not NO_SUMMARY.search(text)
        else "fact_lookup",
        "routing_reason": "current_named_documents" if multiple else "current_named_document",
        "latency_ms": 0,
        "request_attempted": False,
        "response_received": False,
        "provider_usage": None,
        "note": "本轮明确文件优先；按原问题重新检索，不继承上轮文件或答案。",
    }


def prior_document_followup(question, prior_ids, version):
    """A genuine 'this document' follow-up uses the single document just discussed.

    Server-side, deterministic: one prior source document + a demonstrative in the
    current question. No model call is spent re-asking which document this is.
    """
    if len(prior_ids or ()) != 1 or not DOCUMENT_FOLLOWUP.search(question):
        return None
    from knowflow.app.document_summary import NEGATIVE as NO_SUMMARY
    from knowflow.app.document_summary import document_cue

    return {
        "version": version,
        "status": "resolved",
        "question": question,
        "original_question": question,
        "reference_turn_ids": [],
        "routing_reason": "prior_document_followup",
        "request_kind": "document_summary"
        if document_cue(question) and not NO_SUMMARY.search(question)
        else "fact_lookup",
        "latency_ms": 0,
        "request_attempted": False,
        "response_received": False,
        "provider_usage": None,
        "note": "本轮是上一轮文档的真追问；只使用上一轮唯一引用的文档，不继承其他来源。",
    }


def allows_prior_document(question, resolution):
    if (resolution.get("binding_decision") or {}).get("current_literal_name"):
        return False
    if resolution.get("routing_reason") == "prior_document_followup":
        return True
    return bool(
        resolution.get("status") == "resolved"
        and resolution.get("reference_turn_ids")
        and DOCUMENT_FOLLOWUP.search(question)
    )
