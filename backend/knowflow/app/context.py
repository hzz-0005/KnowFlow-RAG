from __future__ import annotations

import re


def estimate_tokens(text: str) -> int:
    """Conservative provider-independent estimate; counts roughly two characters per token."""
    return max(1, (len(text) + 1) // 2)


def build_evidence_context(
    evidence: list[dict[str, str]], token_budget: int
) -> list[dict[str, str]]:
    """Deduplicate evidence and trim it to a bounded, approximate token budget."""
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    remaining = token_budget
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for item in evidence:
        content = item["content"].strip()
        key = re.sub(r"\s+", " ", content).casefold()
        if not content or key in seen:
            continue
        overhead = estimate_tokens(item["id"]) + 8
        available = remaining - overhead
        if available <= 0:
            break
        max_chars = available * 2
        included = content if len(content) <= max_chars else content[:max_chars]
        if not included.strip():
            continue
        result.append({"id": item["id"], "content": included})
        seen.add(key)
        remaining -= overhead + estimate_tokens(included)
        if remaining <= 0:
            break
    return result
