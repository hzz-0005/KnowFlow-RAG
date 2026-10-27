"""Locate validated quote text within the source regions of a packed chunk."""

import re


def quote_spans(chunk, quotes):
    spans = chunk.source_spans or []
    if not quotes or not spans or any("chunk_range" not in span for span in spans):
        return spans
    words = list(re.finditer(r"\S+", chunk.content))
    normalized, positions = [], []
    for word in words:
        if normalized:
            normalized.append(" ")
            positions.append(word.start() - 1)
        normalized.extend(word.group())
        positions.extend(range(word.start(), word.end()))
    content = "".join(normalized)
    ranges = []
    for quote in quotes:
        quote = " ".join(quote.split())
        if not quote:
            continue
        start = content.find(quote)
        if start < 0:
            return spans
        while start >= 0:
            ranges.append((positions[start], positions[start + len(quote) - 1] + 1))
            start = content.find(quote, start + 1)
    selected = [
        span
        for span in spans
        if any(
            span["chunk_range"][0] < end and span["chunk_range"][1] > start for start, end in ranges
        )
    ]
    return selected or spans


def first_page(spans, fallback):
    pages = [s["page_number"] for s in spans if s.get("page_number")]
    return min(pages) if pages else fallback
