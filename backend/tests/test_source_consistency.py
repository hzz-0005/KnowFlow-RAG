"""Source-text consistency: deterministic, character-mapped, honest about unknowns."""

import importlib.util
from pathlib import Path

from knowflow.app.ingestion_diagnostics import diagnose
from knowflow.app.source_consistency import (
    CONTRACT,
    MATCH,
    MISMATCH,
    NORMALIZED_MATCH,
    NOT_COMPARABLE,
    compare_excerpt,
)


def span(text_range=None, chunk_range=None, key="b0"):
    item = {"block_key": key}
    if text_range is not None:
        item["text_range"] = text_range
    if chunk_range is not None:
        item["chunk_range"] = chunk_range
    return item


def test_exact_and_normalized_matches():
    exact = compare_excerpt(
        span([0, 5], [0, 5]),
        block_text="hello",
        chunk_text="hello",
        block_text_length=5,
        chunk_text_length=5,
    )
    assert exact["status"] == MATCH
    spaced = compare_excerpt(
        span([0, 5], [0, 6]),
        block_text="a b c",
        chunk_text="a  b c",
        block_text_length=5,
        chunk_text_length=6,
    )
    assert spaced["status"] == NORMALIZED_MATCH
    assert "whitespace" in spaced["normalizations"]
    composed = compare_excerpt(
        span([0, 5], [0, 6]),
        block_text="caf\u00e9x",
        chunk_text="cafe\u0301x",
        block_text_length=5,
        chunk_text_length=6,
    )
    assert composed["status"] == NORMALIZED_MATCH
    assert "unicode_nfc" in composed["normalizations"]


def test_same_length_replacement_is_a_provable_mismatch():
    result = compare_excerpt(
        span([0, 11], [0, 11]),
        block_text="hello world",
        chunk_text="X" * 11,
        block_text_length=11,
        chunk_text_length=11,
    )
    assert result["status"] == MISMATCH
    assert result["detail"]["source"] == "hello world"
    assert result["detail"]["excerpt"] == "X" * 11


def test_not_comparable_reasons_are_distinct():
    base = dict(block_text="hello", chunk_text="hello", block_text_length=5, chunk_text_length=5)
    assert compare_excerpt(span(None, [0, 5]), **base)["reason"] == "missing_text_range"
    assert compare_excerpt(span([0, 5], None), **base)["reason"] == "missing_chunk_range"
    assert compare_excerpt(span([0, 5], [0, 5]), **{**base, "block_text": None})["reason"] == (
        "source_text_unavailable"
    )
    assert compare_excerpt(span([0, 9], [0, 5]), **base)["reason"] == "invalid_ranges"
    truncated = compare_excerpt(span([0, 5], [0, 5]), **{**base, "block_text": "hel"})
    assert truncated["reason"] == "source_text_truncated"
    short_excerpt = compare_excerpt(span([0, 5], [0, 5]), **{**base, "chunk_text": "hel"})
    assert short_excerpt["reason"] == "excerpt_text_truncated"
    for case in (
        compare_excerpt(span(None, [0, 5]), **base),
        compare_excerpt(span([0, 5], None), **base),
    ):
        assert case["status"] == NOT_COMPARABLE


def _fixture(content, source_text=None, spans=None, kind="text"):
    source = content if source_text is None else source_text
    doc = {"id": "d1", "filename": "f.md", "status": "processed", "parsing_report": {}}
    blocks = [{"block_key": "b0", "ordinal": 0, "kind": kind, "text_length": len(source)}]
    chunk_spans = spans or [
        {"block_key": "b0", "text_range": [0, len(source)], "chunk_range": [0, len(content)]}
    ]
    chunks = [
        {
            "id": "c0",
            "ordinal": 0,
            "content": content,
            "text_length": len(content),
            "section": None,
            "source_spans": chunk_spans,
        }
    ]
    totals = {"blocks": 1, "chunks": 1}
    return doc, blocks, chunks, totals, {"b0": source}


def test_diagnose_reports_consistency_and_locates_mismatch():
    doc, blocks, chunks, totals, block_texts = _fixture("X" * 5, source_text="hello")
    result = diagnose(doc, blocks, chunks, totals, block_texts=block_texts)
    assert result["schema_version"] == "persisted-integrity-v3"
    found = {f["code"]: f for f in result["findings"]}
    assert found["source_excerpt_mismatch"]["severity"] == "error"
    assert found["source_excerpt_mismatch"]["count"] == 1
    observation = result["observations"]["source_consistency"]
    assert observation["contract"] == CONTRACT
    assert observation["spans"]["compared"] == 1
    assert observation["spans"][MISMATCH] == 1
    example = observation["examples"][0]
    assert example["excerpt"] == "XXXXX"
    assert example["source"] == "hello"
    assert example["chunk_id"] == "c0"


def test_diagnose_layout_mismatch_is_a_warning_and_v1_mode_claims_nothing():
    doc, blocks, chunks, totals, block_texts = _fixture("X" * 5, source_text="hello", kind="table")
    result = diagnose(doc, blocks, chunks, totals, block_texts=block_texts)
    found = {f["code"]: f for f in result["findings"]}
    assert found["source_excerpt_layout_mismatch"]["severity"] == "warning"
    assert "source_excerpt_mismatch" not in found
    legacy = diagnose(doc, blocks, chunks, totals)
    assert legacy["schema_version"] == "persisted-integrity-v1"
    assert legacy["observations"]["source_consistency"]["status"] == "not_run"
    assert not any(f["code"].startswith("source_excerpt") for f in legacy["findings"])


def test_control_corpora_reproduce_both_contracts():
    path = Path(__file__).resolve().parents[2] / "evaluation/diagnostics_v1/run_controls.py"
    spec = importlib.util.spec_from_file_location("controls_consistency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v1 = module.evaluate()
    assert v1["n_contract_mismatches"] == 0
    v2_root = Path(__file__).resolve().parents[2] / "evaluation/diagnostics_v2"
    v2 = module.evaluate(
        cases_path=v2_root / "cases.json",
        manifest_path=v2_root / "corpus-manifest.json",
        contract="v2",
    )
    assert v2["n_contract_mismatches"] == 0
    rows = {row["id"]: row for row in v2["rows"]}
    blind_codes = {f["code"] for f in rows["same_length_replacement"]["diagnostics"]["findings"]}
    assert "source_excerpt_mismatch" in blind_codes
    for case_id in ("whitespace_rewrite", "unicode_form"):
        codes = {f["code"] for f in rows[case_id]["diagnostics"]["findings"]}
        assert "source_excerpt_normalized" in codes
        assert "source_excerpt_mismatch" not in codes
    table_codes = {f["code"] for f in rows["clean_csv"]["diagnostics"]["findings"]}
    assert "source_excerpt_mismatch" not in table_codes


def test_coarse_cross_page_mapping_is_unknown_without_calling_intact_text_corrupt():
    from knowflow.app.config import Settings
    from knowflow.app.document_processing import ParsedSection
    from knowflow.app.structure_chunking import structure_chunks
    from knowflow.app.structured_parsing import Block, ParsedDocument

    text = "alpha gap omega"
    spans = [
        {"block_key": "b0", "page_number": 1, "text_range": [0, 5]},
        {"block_key": "b0", "page_number": 2, "text_range": [10, 15]},
    ]
    parsed = ParsedDocument(
        [ParsedSection(text, 1, None, spans)],
        [Block("b0", "text", text, source_spans=spans)],
        {"parser": "docling", "page_count": 2},
    )
    pieces = structure_chunks(parsed, Settings(_env_file=None))
    assert len(pieces) == 1
    piece = pieces[0]
    assert piece.content == text
    # Existing stored indexes can retain the old coarse mapping. New chunkers
    # produce precise subranges, but diagnostics must still accept old records.
    legacy_spans = [{**s, "chunk_range": [0, len(text)]} for s in spans]
    doc, blocks, chunks, totals, texts = _fixture(text, spans=legacy_spans)
    result = diagnose(doc, blocks, chunks, totals, block_texts=texts)
    observation = result["observations"]["source_consistency"]
    assert observation["status"] == "unknown"
    assert observation["spans"][MISMATCH] == 0
    assert observation["spans"][NOT_COMPARABLE] == 2
    assert observation["reasons"] == {"ambiguous_character_mapping": 2}
    assert "source_excerpt_mismatch" not in {f["code"] for f in result["findings"]}


def test_difference_after_display_prefix_is_visible_and_located():
    source, excerpt = "A" * 900 + "OLD", "A" * 900 + "NEW"
    result = compare_excerpt(
        span([0, 903], [0, 903]),
        block_text=source,
        chunk_text=excerpt,
        block_text_length=903,
        chunk_text_length=903,
    )
    assert result["status"] == MISMATCH
    detail = result["detail"]
    assert detail["source"].endswith("OLD")
    assert detail["excerpt"].endswith("NEW")
    assert detail["difference"]["source_offset"] == 900
    assert detail["difference"]["excerpt_offset"] == 900
    assert len(detail["source"]) <= 600


def test_invalid_negative_mapping_never_previews_a_python_tail_slice():
    result = compare_excerpt(
        span([-4, 9], [0, 9]),
        block_text="123456789",
        chunk_text="123456789",
        block_text_length=9,
        chunk_text_length=9,
    )
    assert result["reason"] == "invalid_ranges"
    assert result["detail"]["source"] is None


def test_missing_provenance_has_chunk_coverage_and_no_invented_span_denominator():
    doc, blocks, chunks, totals, texts = _fixture("hello")
    for missing in ([], None, [None]):
        chunks[0]["source_spans"] = missing
        result = diagnose(doc, blocks, chunks, totals, block_texts=texts)
        obs = result["observations"]["source_consistency"]
        assert obs["status"] == "unknown"
        assert obs["coverage"] == {
            "chunks_inspected": 1,
            "chunks_without_span_objects": 1,
            "span_objects_inspected": 0,
        }
        assert obs["spans"]["compared"] == 0
        found = {f["code"]: f for f in result["findings"]}
        assert found["source_consistency_unmapped_chunk"]["count"] == 1
        assert "source_consistency_unknown" not in found
    empty = diagnose(doc, [], [], {"blocks": 0, "chunks": 0}, block_texts={})
    assert empty["observations"]["source_consistency"]["status"] == "unknown"


def test_repeated_source_comparisons_stop_at_the_character_budget():
    doc, blocks, chunks, totals, texts = _fixture(
        "hello",
        spans=[span([0, 5], [0, 5])] * 3,
    )
    result = diagnose(doc, blocks, chunks, totals, block_texts=texts, consistency_char_budget=10)
    obs = result["observations"]["source_consistency"]
    assert obs["spans"][MATCH] == 1
    assert obs["spans"][NOT_COMPARABLE] == 2
    assert obs["reasons"] == {"comparison_budget_exhausted": 2}
    assert obs["work"] == {"characters_inspected": 10, "character_budget": 10}
