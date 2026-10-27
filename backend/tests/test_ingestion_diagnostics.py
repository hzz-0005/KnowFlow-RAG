from copy import deepcopy

import pytest
from knowflow.app.ingestion_diagnostics import diagnose, valid_cells


class Counter:
    metadata = {"sha256": "fixed-test-tokenizer"}

    def count(self, text):
        return len(text)


def fixture():
    doc = {
        "id": "doc",
        "filename": "example.pdf",
        "status": "processed",
        "parsing_report": {
            "page_count": 2,
            "chunking": {"max_tokens": 12, "token_reserve": 2, "tokenizer": Counter.metadata},
        },
    }
    blocks = [{"block_key": "b1", "ordinal": 0, "kind": "text", "text_length": 3}]
    chunks = [
        {
            "id": "c1",
            "ordinal": 0,
            "content": "abc",
            "text_length": 3,
            "section": None,
            "source_spans": [
                {"block_key": "b1", "page_number": 1, "text_range": [0, 3], "chunk_range": [0, 3]}
            ],
        }
    ]
    return doc, blocks, chunks


def run(doc=None, blocks=None, chunks=None, totals=None):
    a, b, c = fixture()
    return diagnose(
        doc or a,
        b if blocks is None else blocks,
        c if chunks is None else chunks,
        totals
        or {
            "blocks": len(b if blocks is None else blocks),
            "chunks": len(c if chunks is None else chunks),
        },
        counter=Counter(),
    )


def codes(result):
    return {f["code"]: f for f in result["findings"]}


def test_page_gap_is_a_localized_warning_not_semantic_failure():
    result = run()
    assert result["observations"]["page_coverage"]["pages_with_chunk_source"] == 1
    gap = codes(result)["page_without_chunk_source"]
    assert gap["severity"] == "warning"
    assert gap["examples"] == [{"page_number": 2}]
    assert "可能" in gap["message"]
    assert "不代表语义正确" in result["note"]


def test_legacy_policy_and_ranges_are_unknown_not_zero_or_pass():
    doc, _, chunks = fixture()
    doc["parsing_report"] = {}
    chunks[0]["source_spans"] = [{"block_key": "b1"}]
    result = run(doc=doc, chunks=chunks)
    assert result["observations"]["token_budget"] == {
        "observed": 0,
        "unknown": 1,
        "max_input_tokens": None,
        "saved_budget": None,
    }
    assert codes(result)["chunk_range_missing"]["severity"] == "unknown"
    assert result["observations"]["page_coverage"]["status"] == "unknown"


def test_budget_uses_saved_tokenizer_and_complete_heading_input():
    doc, _, chunks = fixture()
    chunks[0]["section"] = "long heading"
    result = run(chunks=chunks)
    assert codes(result)["token_over_budget"]["examples"][0]["input_tokens"] == 16
    doc["parsing_report"]["chunking"]["tokenizer"] = {"sha256": "other-version"}
    result = run(doc=doc, chunks=chunks)
    assert "token_over_budget" not in codes(result)
    assert result["observations"]["token_budget"]["unknown"] == 1


def test_invalid_locations_do_not_crash_and_are_distinct_from_missing():
    _, _, chunks = fixture()
    chunks[0]["source_spans"] = [
        None,
        {
            "block_key": "absent",
            "page_number": True,
            "bbox": {"left": 0.5, "top": 0, "right": 0.1, "bottom": 1},
            "chunk_range": [0, 999],
        },
        {"block_key": "b1", "text_range": [-1, 3]},
    ]
    found = codes(run(chunks=chunks))
    assert {
        "invalid_span",
        "dangling_block",
        "invalid_page",
        "invalid_bbox",
        "invalid_chunk_range",
        "invalid_text_range",
    } <= found.keys()


def test_read_limits_do_not_create_dangling_blocks_or_full_page_claims():
    result = run(blocks=[], totals={"blocks": 20000, "chunks": 10})
    assert not result["scope"]["complete"]
    assert codes(result)["block_lookup_incomplete"]["severity"] == "unknown"
    assert "dangling_block" not in codes(result)
    assert "unreferenced_block" not in codes(result)
    assert "page_without_chunk_source" not in codes(result)
    assert result["observations"]["page_coverage"]["status"] == "unknown"


def test_truncated_text_is_not_tokenized_as_complete_input():
    _, _, chunks = fixture()
    chunks[0]["text_length"] = 1000000
    result = run(chunks=chunks)
    assert result["observations"]["token_budget"]["max_input_tokens"] is None
    assert not result["scope"]["content_complete"]


def test_withheld_source_metadata_cannot_produce_page_gaps_or_unreferenced_blocks():
    _, _, chunks = fixture()
    chunks[0]["source_spans"] = None
    result = run(chunks=chunks)
    assert not result["scope"]["sources_complete"]
    assert result["observations"]["page_coverage"]["status"] == "unknown"
    assert "page_without_chunk_source" not in codes(result)
    assert "unreferenced_block" not in codes(result)


@pytest.mark.parametrize("report", [[], {"chunking": "invalid-legacy-shape"}])
def test_malformed_legacy_policy_is_unknown(report):
    doc, _, _ = fixture()
    doc["parsing_report"] = report
    assert run(doc=doc)["observations"]["token_budget"]["unknown"] == 1


def test_duplicate_and_replacement_character_preserve_all_rows():
    doc, blocks, chunks = fixture()
    chunks[0]["content"] = "a\ufffdb"
    chunks.append({**deepcopy(chunks[0]), "id": "c2", "ordinal": 1})
    before = deepcopy(chunks)
    result = run(doc, blocks, chunks)
    assert codes(result)["duplicate_content"]["severity"] == "info"
    assert codes(result)["replacement_character"]["count"] == 2
    assert chunks == before
    assert result["snapshot_sha256"] == run(doc, blocks, chunks)["snapshot_sha256"]
    chunks[0]["section"] = "different"
    assert result["snapshot_sha256"] != run(doc, blocks, chunks)["snapshot_sha256"]


def test_failure_with_no_index_is_unknown_not_quality_score():
    doc, _, _ = fixture()
    doc["status"] = "failed"
    result = run(doc=doc, chunks=[], blocks=[])
    assert {"index_not_current", "no_indexed_chunks"} <= codes(result).keys()
    assert "success_rate" not in result


def test_examples_are_bounded_and_counts_are_not_truncated():
    doc, _, _ = fixture()
    doc["parsing_report"]["page_count"] = 100
    finding = codes(run(doc=doc))["page_without_chunk_source"]
    assert finding["count"] == 99
    assert len(finding["examples"]) == 8


@pytest.mark.parametrize(
    "value,valid",
    [
        ("A1", True),
        ("A1:XFD1048576", True),
        ("B2:A1", False),
        ("XFE1", False),
        ("A0", False),
        ("A1048577", False),
        (None, False),
    ],
)
def test_excel_coordinates(value, valid):
    assert valid_cells(value) == valid


def test_tabular_coordinates_have_no_fabricated_pdf_pages():
    doc, _, chunks = fixture()
    doc["filename"] = "data.xlsx"
    chunks[0]["source_spans"][0].update(
        source_type="spreadsheet", sheet_name="数据", cell_range="A1:B2", page_number=None
    )
    result = run(doc=doc, chunks=chunks)
    assert result["observations"]["page_coverage"]["status"] == "not_applicable"
    assert "invalid_cell_location" not in codes(result)
    chunks[0]["source_spans"][0]["cell_range"] = "A0"
    assert codes(run(doc=doc, chunks=chunks))["invalid_cell_location"]["severity"] == "error"
