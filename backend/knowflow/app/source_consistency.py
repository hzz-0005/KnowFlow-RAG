"""Deterministic source-text vs chunk-excerpt consistency.

Only compares where an explicit character mapping exists (span text_range into the
saved block text, span chunk_range into the saved chunk content). It proves whether
the saved excerpt matches the saved source text. It does NOT prove OCR correctness,
citation relevance, or that an answer is semantically supported.
"""

from __future__ import annotations

import unicodedata

CONTRACT = "source-consistency-v2"

MATCH = "match"
NORMALIZED_MATCH = "normalized_match"
MISMATCH = "mismatch"
NOT_COMPARABLE = "not_comparable"

REASONS = (
    "missing_chunk_range",
    "missing_text_range",
    "invalid_ranges",
    "source_text_unavailable",
    "source_text_truncated",
    "excerpt_text_truncated",
    "ambiguous_character_mapping",
    "comparison_budget_exhausted",
)

SAMPLE_CHARS = 600


def collapse_whitespace(text):
    """Same basis as runtime quote matching in source_provenance.quote_spans."""
    return " ".join(text.split())


def to_nfc(text):
    return unicodedata.normalize("NFC", text)


def _pair(value):
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in value)
    ):
        return value
    return None


def _slice(text, bounds):
    if not isinstance(text, str) or bounds is None or not 0 <= bounds[0] < bounds[1] <= len(text):
        return None, None
    return text[bounds[0] : min(bounds[1], bounds[0] + SAMPLE_CHARS)], (
        bounds[1] - bounds[0] > SAMPLE_CHARS
    )


def compare_excerpt(
    span,
    *,
    block_text,
    chunk_text,
    block_text_length,
    chunk_text_length,
    max_compared_characters=None,
):
    """Compare one source span's mapped slices. Returns a plain dict.

    block_text / chunk_text are the LOADED portions (possibly truncated by a read
    budget); block_text_length / chunk_text_length are the full stored lengths.
    block_text is None when the source text could not be read at all.
    """
    chunk_range = _pair(span.get("chunk_range"))
    text_range = _pair(span.get("text_range"))
    inspected = 0

    def result(status, reason=None, normalizations=()):
        source, source_truncated = _slice(block_text, text_range)
        excerpt, excerpt_truncated = _slice(chunk_text, chunk_range)
        return {
            "status": status,
            "reason": reason,
            "normalizations": list(normalizations),
            "characters_inspected": inspected,
            "chunk_range": chunk_range,
            "text_range": text_range,
            "detail": {
                "source": source,
                "source_truncated": source_truncated,
                "excerpt": excerpt,
                "excerpt_truncated": excerpt_truncated,
            },
        }

    if chunk_range is None:
        return result(NOT_COMPARABLE, "missing_chunk_range")
    if text_range is None:
        return result(NOT_COMPARABLE, "missing_text_range")
    if block_text is None:
        return result(NOT_COMPARABLE, "source_text_unavailable")
    for lo, hi, length in (
        (text_range[0], text_range[1], block_text_length),
        (chunk_range[0], chunk_range[1], chunk_text_length),
    ):
        if not (0 <= lo < hi <= length):
            return result(NOT_COMPARABLE, "invalid_ranges")
    if text_range[1] > len(block_text):
        return result(NOT_COMPARABLE, "source_text_truncated")
    if chunk_text is None or chunk_range[1] > len(chunk_text):
        return result(NOT_COMPARABLE, "excerpt_text_truncated")
    cost = text_range[1] - text_range[0] + chunk_range[1] - chunk_range[0]
    if max_compared_characters is not None and cost > max_compared_characters:
        return result(NOT_COMPARABLE, "comparison_budget_exhausted")
    inspected = cost
    source = block_text[text_range[0] : text_range[1]]
    excerpt = chunk_text[chunk_range[0] : chunk_range[1]]
    if source == excerpt:
        return result(MATCH)
    if to_nfc(collapse_whitespace(source)) == to_nfc(collapse_whitespace(excerpt)):
        normalizations = []
        if collapse_whitespace(source) != source or collapse_whitespace(excerpt) != excerpt:
            normalizations.append("whitespace")
        if to_nfc(source) != source or to_nfc(excerpt) != excerpt:
            normalizations.append("unicode_nfc")
        return result(NORMALIZED_MATCH, None, normalizations or ["whitespace"])
    # Existing provenance can clip a source range to one page but retain the
    # whole piece's chunk_range. Unequal extents do not establish a 1:1 mapping.
    # Exact/normalised agreement above remains a valid, narrower observation.
    if len(source) != len(excerpt):
        return result(NOT_COMPARABLE, "ambiguous_character_mapping")
    compared = result(MISMATCH)
    first = next(i for i, (a, b) in enumerate(zip(source, excerpt, strict=True)) if a != b)
    start = max(0, first - SAMPLE_CHARS // 3)
    end = min(len(source), start + SAMPLE_CHARS)
    compared["detail"].update(
        source=source[start:end],
        excerpt=excerpt[start:end],
        source_truncated=end - start < len(source),
        excerpt_truncated=end - start < len(excerpt),
        difference={
            "unit": "unicode_codepoint",
            "source_offset": text_range[0] + first,
            "excerpt_offset": chunk_range[0] + first,
            "source_display_range": [text_range[0] + start, text_range[0] + end],
            "excerpt_display_range": [chunk_range[0] + start, chunk_range[0] + end],
        },
    )
    return compared


def sample_row(status, reason, normalizations, location, detail, chunk_range, text_range):
    """Bounded evidence row for page-level side-by-side inspection."""
    return {
        "status": status,
        "reason": reason,
        "normalizations": normalizations,
        **location,
        "chunk_range": chunk_range,
        "text_range": text_range,
        "excerpt": detail.get("excerpt"),
        "excerpt_truncated": detail.get("excerpt_truncated"),
        "source": detail.get("source"),
        "source_truncated": detail.get("source_truncated"),
        "difference": detail.get("difference"),
    }
