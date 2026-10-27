from types import SimpleNamespace

from knowflow.app.source_provenance import quote_spans
from knowflow.app.structure_chunking import Unit, _render
from knowflow.app.structured_parsing import Block


def unit(text, source=None, spans=None, kind="prose"):
    return Unit(text, None, spans or [], (None, None, kind), Block("b", "text", source or text))


def test_partial_source_ranges_map_only_their_part_and_keep_unmapped_body():
    text = "alpha gap omega"
    part = unit(
        text,
        spans=[
            {"block_key": "b", "page_number": 1, "text_range": [0, 5]},
            {"block_key": "b", "page_number": 2, "text_range": [10, 15]},
        ],
    )
    content, spans = _render([(part, 0, len(text))], "", [])
    assert content == text  # The unmapped middle is retained, never fabricated as evidence.
    assert [s["chunk_range"] for s in spans] == [[0, 5], [10, 15]]
    for span in spans:
        a, b = span["text_range"]
        x, y = span["chunk_range"]
        assert text[a:b] == content[x:y]
    quoted = quote_spans(SimpleNamespace(content=content, source_spans=spans), ["omega"])
    assert [s["page_number"] for s in quoted] == [2]


def test_fragment_offsets_include_prior_piece_and_unicode_codepoints():
    source = "前缀甲乙丙丁后缀"
    part = unit(
        "甲乙丙丁",
        source,
        [
            {"block_key": "b", "text_range": [2, 4], "page_number": 1},
            {"block_key": "b", "text_range": [4, 6], "page_number": 2},
        ],
    )
    leading = unit("lead", spans=[{"block_key": "lead"}])
    content, spans = _render([(leading, 0, 4), (part, 1, 4)], "", [])
    assert content == "lead\n\n乙丙丁"
    assert spans[1]["chunk_range"] == [6, 7]
    assert spans[2]["chunk_range"] == [7, 9]
    assert [s["text_range"] for s in spans[1:]] == [[3, 4], [4, 6]]


def test_table_and_ambiguous_source_text_do_not_gain_invented_character_ranges():
    for part in [
        unit("row", "row row", [{"block_key": "b"}]),
        unit("row", "row", [{"block_key": "b", "cell_range": "A1:B1"}], kind="table"),
    ]:
        content, spans = _render([(part, 0, 3)], "", [])
        assert content == "row"
        assert "text_range" not in spans[0]
        assert spans[0]["chunk_range"] == [0, 3]
