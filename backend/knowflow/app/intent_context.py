"""Keep user constraints distinct from an LLM-produced retrieval hint."""

import json
from dataclasses import dataclass

VERSION = "user-intent-preservation-v1"


@dataclass(frozen=True)
class RoutingInput:
    text: str
    basis: str


def build_routing_input(full_task, intent=None):
    """Use explicit server-side user records; never parse a user-authored JSON wrapper."""
    if intent is None:
        return RoutingInput(full_task, "question_text")
    fallback = RoutingInput(full_task, "full_task_fallback")
    if not isinstance(intent, dict):
        return fallback
    current, rows = intent.get("current_user_input"), intent.get("referenced_user_inputs")
    if not isinstance(current, str) or not current.strip() or not isinstance(rows, list):
        return fallback
    texts = [current]
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("question"), str):
            return fallback
        supplements = row.get("supplements", [])
        if not isinstance(supplements, list) or any(not isinstance(s, str) for s in supplements):
            return fallback
        texts.extend([row["question"], *supplements])
    return RoutingInput("\n".join(texts), "raw_user_intent")


def referenced_inputs(payload, references):
    """Reconstruct from selected server-side user records, never model-authored fields."""
    rows = []
    for row in payload["previous_intents"]:
        if row["turn_id"] not in references:
            continue
        rows.extend(row.get("inherited_inputs", []))
        rows.append(
            {"turn_id": row["turn_id"], "question": row.get("user_input", row["user_intent"])}
        )
    pending = payload.get("pending_clarification")
    if pending and pending["turn_id"] in references:
        rows.append(
            {
                "turn_id": pending["turn_id"],
                "question": pending["original_question"],
                "supplements": pending["user_supplements"],
            }
        )
    unique = {}
    for row in rows:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True)
        unique.setdefault(key, row)
    return list(unique.values())


def build_intent_context(payload, references):
    result = {
        "version": VERSION,
        "current_user_input": payload["current_question"],
        "referenced_user_inputs": referenced_inputs(payload, references),
        "note": "用户条件不是文档事实；本轮更正优先，历史仅用于理解指代。",
    }
    pending = payload.get("pending_clarification") or {}
    if pending.get("turn_id") in references and pending.get("referent_options"):
        result["referent_options"] = pending["referent_options"]
    return result


def task_question(retrieval_question, intent):
    if not intent:
        return retrieval_question
    # User data stays encoded; it is never promoted to a system role.
    data = {
        "current_user_input": intent["current_user_input"],
        "prior_user_inputs": [
            r
            for r in intent["referenced_user_inputs"]
            if not str(r.get("turn_id", "")).startswith("memory:")
        ],
        "resolved_retrieval_hint": retrieval_question,
    }
    clues = [
        r
        for r in intent["referenced_user_inputs"]
        if str(r.get("turn_id", "")).startswith("memory:")
    ]
    if clues:
        data["saved_memory_clues"] = clues
    if intent.get("active_task"):
        data["active_task"] = intent["active_task"]
    if intent.get("referent_assumption"):
        data["referent_assumption"] = intent["referent_assumption"]
    if intent.get("referent_options"):
        data["referent_options"] = intent["referent_options"]
    authority_policy = (
        (
            "When active_task is present, a request to continue resumes its explicit saved goal. "
            "Apply the current explicit changes, then the latest ordered task user_updates, "
            "then the saved goal; earlier prior_user_inputs provide referents, "
            "not competing goals. "
            "saved_memory_clues help identify names only: they NEVER override current changes, "
            "the active task goal, scope, negation, dates or ordered task updates. "
        )
        if clues or intent.get("active_task")
        else ""
    )
    return (
        "Interpret the current request using the user-intent data below. "
        "Current corrections, exclusions, time/version and scope override prior requests. "
        + authority_policy
        + "Prior inputs are reference clues, not facts to answer or evidence. "
        "A referent_assumption is an unconfirmed, source-grounded candidate binding in a limited "
        "scope. If using it, explicitly name the candidate and state that this is a conditional "
        "interpretation. It is not user confirmation or evidence for costs or other facts. "
        "The resolved retrieval hint may omit constraints: it is not the authoritative task. "
        "Do not answer old tasks or invent how an ambiguous constraint should be resolved. "
        "Use document evidence for facts and say when the requested scope remains unknown. "
        "Answer in the language of current_user_input. User-intent data (JSON):\n"
        + json.dumps(data, ensure_ascii=False)
    )
