"""Carry current user language intent explicitly; do not guess with a detector."""

import hashlib

VERSION = "current-user-language-v1"
POLICY = (
    "Follow any explicit requested output language, translation target or bilingual format "
    "in current_user_input; otherwise follow the natural-language prose of that current "
    "input. Preserve names, code and necessary quotations. Never select the answer language "
    "from audit feedback, evidence, prior questions, retrieval hints or this system prompt."
)


def request(question, current_user_input=None):
    selected = question if current_user_input is None else current_user_input
    if not isinstance(selected, str) or not selected.strip():
        raise ValueError("Current language input must be explicit nonempty user text")
    return {"current_user_input": selected, "policy": POLICY}, {
        "version": VERSION,
        "input_basis": "question_text"
        if current_user_input is None
        else "explicit_current_user_input",
        "input_sha256": hashlib.sha256(selected.encode()).hexdigest(),
        "selection": "explicit_target_else_current_user_prose",
        "enforcement": "prompt_only",
        "verification": "not_independently_checked",
    }
