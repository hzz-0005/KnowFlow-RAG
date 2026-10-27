"""Conservative projection of one independent request from a clarification turn."""

import re

from knowflow.app.intent_context import build_intent_context

VERSION = "quoted-partial-lookup-v2-lossless-scope"
DEPENDENT = re.compile(
    r"比较|相差|差值|差额|比值|之和|合计|总计|如果|除非|取决于|"
    r"\b(?:compare|difference|ratio|sum|total|if|unless|depending)\b",
    re.I,
)


class PartialLookupError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise PartialLookupError(reason)


def numbers(value):
    return set(re.findall(r"\d+(?:\.\d+)?", value))


def validate_partial(value, payload):
    require(
        isinstance(value, dict)
        and set(value)
        == {
            "segments",
            "question",
            "reference_turn_ids",
            "depends_on",
        },
        "partial_invalid_schema",
    )
    pending = payload.get("pending_clarification") or {}
    require(not pending.get("requires_restatement"), "partial_restatement_required")
    require(value["depends_on"] == [], "partial_dependent_lookup")
    question = value["question"]
    require(
        isinstance(question, str) and 4 <= len(question.strip()) <= 800, "partial_invalid_question"
    )
    segments = value["segments"]
    require(isinstance(segments, list) and 2 <= len(segments) <= 12, "partial_invalid_segments")
    for row in segments:
        require(isinstance(row, dict) and set(row) == {"kind", "text"}, "partial_invalid_segments")
        require(
            isinstance(row["kind"], str)
            and row["kind"]
            in {
                "answerable",
                "blocked",
                "shared",
                "connector",
            }
            and isinstance(row["text"], str)
            and bool(row["text"]),
            "partial_invalid_segments",
        )
    require(
        "".join(r["text"] for r in segments) == payload["current_question"],
        "partial_input_not_exact_partition",
    )
    answerable = [r["text"] for r in segments if r["kind"] == "answerable"]
    blocked = [r["text"] for r in segments if r["kind"] == "blocked"]
    require(
        len(answerable) == 1
        and len(answerable[0].strip()) >= 4
        and blocked
        and all(len(s.strip()) >= 2 for s in blocked),
        "partial_missing_scope",
    )
    require(not DEPENDENT.search(payload["current_question"]), "partial_dependency_language")
    refs = value["reference_turn_ids"]
    allowed = {
        r["turn_id"] for r in payload["previous_intents"] if not r["turn_id"].startswith("memory:")
    }
    require(
        isinstance(refs, list)
        and len(refs) <= 3
        and all(isinstance(r, str) and r in allowed for r in refs)
        and len(refs) == len(set(refs)),
        "partial_invalid_reference",
    )
    projection = "\n".join(
        r["text"].strip() for r in segments if r["kind"] != "blocked" and r["text"].strip()
    )
    context = build_intent_context({**payload, "current_question": projection}, refs)
    rows = context["referenced_user_inputs"]
    require(
        not any(str(r.get("turn_id", "")).startswith("memory:") for r in rows),
        "partial_memory_dependency",
    )
    inherited = "\n".join(r["question"] + "\n" + "\n".join(r.get("supplements", [])) for r in rows)
    # With no historical binding to resolve, use literal user words, never an
    # unreferenced model rewrite. Every non-blocked word reaches retrieval too.
    if not refs:
        question = projection
    required, actual = numbers(projection), numbers(question)
    require(required <= actual <= numbers(projection + "\n" + inherited), "partial_numeric_changed")
    require(
        not actual.intersection(numbers(" ".join(blocked)) - required), "partial_blocked_numbers"
    )
    compiled = question if not refs else question.strip() + "\n当前用户原话范围：\n" + projection
    require(len(compiled) <= 4000, "partial_compiled_query_too_large")
    return {
        "version": VERSION,
        "question": question.strip(),
        "compiled_query": compiled,
        "retrieval_basis": "referenced_rewrite_plus_raw_scope"
        if refs
        else "raw_current_projection",
        "segments": segments,
        "reference_turn_ids": refs,
        "depends_on": [],
        "current_scope": projection,
        "blocked_scopes": blocked,
        "intent_context": context,
        "full_request_complete": False,
    }
