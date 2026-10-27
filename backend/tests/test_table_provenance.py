from copy import deepcopy

from knowflow.app.document_processing import ParsedSection
from knowflow.app.structure_chunking import _render, _units
from knowflow.app.structured_parsing import Block, ParsedDocument, _table_rows
from knowflow.app.table_provenance import table_row_sources


def table(values, *, headers=True, caption="表一", merged=False):
    cells = []
    for r, row in enumerate(values):
        for c, text in enumerate(row):
            if merged and r == 1 and c == 1:
                continue
            cells.append(
                {
                    "text": text,
                    "start_row_offset_idx": r,
                    "end_row_offset_idx": r + 1,
                    "start_col_offset_idx": c,
                    "end_col_offset_idx": c + (2 if merged and r == 1 and c == 0 else 1),
                    "column_header": headers and r == 0,
                }
            )
    rows, structure = _table_rows(
        {"data": {"num_rows": len(values), "num_cols": len(values[0]), "table_cells": cells}}
    )
    structure["captions"] = [caption] if caption else []
    full = [f"{caption}\n{row}".strip() for row in rows]
    block = Block(
        "b",
        "table",
        "\n".join(full),
        source_spans=[{"block_key": "b", "page_number": 1}],
        structure=structure,
    )
    parsed = ParsedDocument(
        [ParsedSection(t, 1, None, deepcopy(block.source_spans)) for t in full],
        [block],
        {"parser": "docling"},
    )
    return parsed, rows


def test_duplicate_rows_and_repeated_captions_have_distinct_ordered_source_ranges():
    parsed, rows = table([["姓名", "值"], ["甲", "4"], ["甲", "4"]])
    units = list(_units(parsed))
    content, spans = _render([(u, 0, len(u.text)) for u in units], "表一", [])
    assert content == "表一\n" + "\n\n".join(rows)
    assert spans[0]["text_range"] != spans[1]["text_range"]
    for span in spans:
        a, b = span["text_range"]
        x, y = span["chunk_range"]
        assert parsed.blocks[0].text[a:b] == content[x:y]
        assert span["row_number"] in [2, 3]
        for cell in span["table_cells"]:
            a, b = cell["text_range"]
            x, y = cell["chunk_range"]
            assert parsed.blocks[0].text[a:b] == content[x:y]
            assert cell["source_cells"]


def test_merged_cells_and_header_reuse_keep_original_grid_identity():
    parsed, rows = table([["A", "B"], ["合并值", ""], ["乙", "8"]], merged=True)
    mappings = table_row_sources(parsed.blocks[0], rows)
    merged = [c for c in mappings[0]["table_cells"] if c["role"] == "value"]
    assert len(merged) == 2
    assert merged[0]["source_cells"] == merged[1]["source_cells"]
    assert merged[0]["source_cells"][0]["columns"] == [1, 2]
    assert mappings[1]["table_cells"][0]["source_cells"][0]["rows"] == [1, 1]


def test_header_only_fallback_and_empty_columns_preserve_serialization():
    parsed, rows = table([["a", "", "b"]], caption="")
    mappings = table_row_sources(parsed.blocks[0], rows)
    assert mappings[0]["text_range"] == [0, len("a |  | b")]
    assert [c["column_number"] for c in mappings[0]["table_cells"]] == [1, 3]


def test_fragmented_row_clips_cell_ranges_without_changing_saved_offsets():
    parsed, _ = table([["A", "B"], ["αβγδε", "9"]])
    unit = list(_units(parsed))[0]
    before = deepcopy(unit.spans)
    start = unit.text.index("β")
    content, spans = _render([(unit, start, start + 2)], "表一", [])
    assert content == "表一\nβγ"
    assert len(spans[0]["table_cells"]) == 1
    assert spans[0]["table_cells"][0]["chunk_range"] == [3, 5]
    assert unit.spans == before


def test_bad_complete_serialization_order_overlap_or_multipage_never_gets_mapping():
    for fault in ("text", "order", "overlap", "multipage", "missing_structure"):
        parsed, rows = table([["A", "B"], ["甲", "1"], ["乙", "2"]])
        block = parsed.blocks[0]
        if fault == "text":
            block.text += "corrupt"
        elif fault == "order":
            rows.reverse()
        elif fault == "overlap":
            block.structure["table_cells"].append(deepcopy(block.structure["table_cells"][0]))
        elif fault == "multipage":
            block.source_spans.append({"block_key": "b", "page_number": 2})
        else:
            block.structure.pop("table_cells")
        assert table_row_sources(block, rows) is None, fault
