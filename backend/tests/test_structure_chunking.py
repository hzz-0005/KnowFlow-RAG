from types import SimpleNamespace

import pytest
from knowflow.app.config import Settings
from knowflow.app.document_processing import ParsedSection
from knowflow.app.index_text import chunk_index_text
from knowflow.app.source_provenance import first_page, quote_spans
from knowflow.app.structured_parsing import Block, ParsedDocument, chunk_parsed_document
from knowflow.app.token_budget import get_token_counter


def document(rows):
    blocks, sections = [], []
    for i, (text, page, section, kind) in enumerate(rows):
        span = {"block_key": str(i), "page_number": page, "text_range": [0, len(text)]}
        blocks.append(Block(str(i), kind, text, section, "body", [span]))
        sections.append(ParsedSection(text, page, section, [span]))
    return ParsedDocument(sections, blocks, {"parser": "docling", "warnings": []})


def test_page_break_does_not_break_a_section_and_quotes_keep_their_own_physical_page():
    parsed = document(
        [
            ("Atlas 网关的复核周期为", 2, "部署", "text"),
            ("每隔17天；仅适用于测试环境，生产环境不适用。", 3, "部署", "text"),
            ("这里是独立章节。", 3, "其他", "text"),
        ]
    )
    chunks = chunk_parsed_document(parsed, Settings())
    assert len(chunks) == 2
    assert "Atlas" in chunks[0].content and "17天" in chunks[0].content
    assert chunks[0].processing_metadata["pages"] == [2, 3]
    spans = quote_spans(chunks[0], ["每隔17天；仅适用于测试环境"])
    assert {s["page_number"] for s in spans} == {3}
    assert first_page(spans, chunks[0].page_number) == 3
    assert parsed.report["chunking"]["cross_page_chunks"] == 1


def test_full_embedding_input_including_heading_obeys_budget_and_unicode_is_preserved():
    text = "中文🙂压力单位MPa；混合记录设备Cedar42。\n" * 200
    parsed = document([(text, 1, "维护手册 / 状态说明" * 8, "text")])
    settings = Settings(chunk_max_tokens=128, chunk_target_tokens=96, chunk_overlap_tokens=12)
    chunks = chunk_parsed_document(parsed, settings)
    counter = get_token_counter()
    assert len(chunks) > 2 and all("\ufffd" not in c.content for c in chunks)
    assert all(
        counter.count(chunk_index_text(c.section, c.content)) + settings.chunk_token_reserve <= 128
        for c in chunks
    )
    assert parsed.report["chunking"]["source_coverage_check"] == "passed"
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        for span in c.source_spans:
            lo, hi = span["text_range"]
            clo, chi = span["chunk_range"]
            assert c.content[clo:chi] == text[lo:hi]


def test_code_formula_and_caption_are_not_merged_into_prose():
    parsed = document(
        [
            (text, 1, "S", kind)
            for text, kind in [
                ("解释正文", "text"),
                ("select 17;", "code"),
                ("p=17", "formula"),
                ("Table 1: Limits", "caption"),
                ("后续正文", "text"),
            ]
        ]
    )
    chunks = chunk_parsed_document(parsed, Settings())
    # The caption follows the formula block directly and binds to it as a
    # prefix; neither part ever merges into prose.
    assert [c.processing_metadata["kind"] for c in chunks] == [
        "prose",
        "code",
        "formula",
        "prose",
    ]
    assert "Table 1: Limits" in chunks[2].content and chunks[2].content.index(
        "Table 1: Limits"
    ) < chunks[2].content.index("p=17")


def test_caption_adjacent_to_code_binds_in_either_order_with_provenance():
    code = "function add(a, b) { return a + b; }"
    for rows in (
        [("前置说明", "text"), (code, "code"), ("Listing 1: Simple JavaScript Program", "caption")],
        [("Listing 1: Simple JavaScript Program", "caption"), (code, "code"), ("后续正文", "text")],
    ):
        parsed = document([(text, 1, "S", kind) for text, kind in rows])
        chunks = chunk_parsed_document(parsed, Settings())
        kinds = [c.processing_metadata["kind"] for c in chunks]
        assert "caption" not in kinds and kinds.count("code") == 1
        code_chunk = next(c for c in chunks if c.processing_metadata["kind"] == "code")
        assert code_chunk.content.startswith("Listing 1: Simple JavaScript Program\n")
        assert code in code_chunk.content
        expected_keys = {str(i) for i, (_, kind) in enumerate(rows) if kind in {"code", "caption"}}
        assert {s["block_key"] for s in code_chunk.source_spans} == expected_keys
        assert parsed.report["chunking"]["source_coverage_check"] == "passed"


def test_caption_far_from_code_stays_its_own_chunk():
    parsed = document(
        [
            (text, 1, "S", kind)
            for text, kind in [
                ("select 17;", "code"),
                ("插入正文使图注不再紧邻代码块。", "text"),
                ("Listing 2: Detached", "caption"),
            ]
        ]
    )
    chunks = chunk_parsed_document(parsed, Settings())
    kinds = [c.processing_metadata["kind"] for c in chunks]
    assert kinds == ["code", "prose", "caption"]
    assert not chunks[0].content.startswith("Listing 2")


def test_table_rows_repeat_caption_once_and_keep_caption_and_body_regions_separate():
    caption = "Table 1: Limits in MPa."
    capspan = {"block_key": "cap", "page_number": 1, "text_range": [0, len(caption)]}
    table = Block(
        "table",
        "table",
        "设备: 泵A | 压力: 1.8\n设备: 泵B | 压力: 2.4",
        section="S",
        source_spans=[{"block_key": "table", "page_number": 2}],
        structure={"captions": [caption]},
    )
    rows = ["设备: 泵A | 压力: 1.8", "设备: 泵B | 压力: 2.4"]
    parsed = ParsedDocument(
        [
            ParsedSection(
                caption + "\n" + row,
                2,
                "S",
                [{"block_key": "table", "page_number": 2, "row_key": row}, capspan],
            )
            for row in rows
        ],
        [table, Block("cap", "caption", caption, source_spans=[capspan])],
        {"parser": "docling", "warnings": []},
    )
    chunks = chunk_parsed_document(parsed, Settings())
    assert len(chunks) == 1 and chunks[0].content.count(caption) == 1
    assert all(row in chunks[0].content for row in rows)
    assert {s["page_number"] for s in quote_spans(chunks[0], [rows[1]])} == {2}


def test_overlong_table_row_is_retained_and_flagged_instead_of_dropped():
    body = "单元格内容🙂" * 300
    table = Block(
        "t",
        "table",
        body,
        source_spans=[{"block_key": "t", "page_number": 1}],
        structure={"captions": []},
    )
    parsed = ParsedDocument(
        [ParsedSection(body, 1, None, table.source_spans)],
        [table],
        {"parser": "docling", "warnings": []},
    )
    chunks = chunk_parsed_document(
        parsed, Settings(chunk_max_tokens=128, chunk_target_tokens=96, chunk_overlap_tokens=12)
    )
    assert "".join(c.content for c in chunks) == body
    assert parsed.report["chunking"]["fragmented_table_rows"] == 1
    assert parsed.report["status"] == "needs_review"


def test_tokenizer_offset_prefix_never_overruns_a_budget_inside_multitoken_characters():
    counter = get_token_counter()
    text = "🙂漢字🧑🏽‍🚀 repeated token 42。" * 4
    for budget in range(1, 30):
        end = counter.prefix_end(text, budget)
        assert counter.count(text[:end]) <= budget


def test_invalid_budget_is_an_automatic_failure_not_a_silent_overflow():
    parsed = document([("body", 1, "header" * 1000, "text")])
    with pytest.raises(ValueError, match="预算"):
        chunk_parsed_document(parsed, Settings(chunk_max_tokens=128, chunk_overlap_tokens=12))


def test_quote_location_keeps_all_occurrences_when_identical_text_occurs_on_multiple_pages():
    chunk = SimpleNamespace(
        content="same evidence.\n\nsame evidence.",
        source_spans=[
            {"page_number": 1, "chunk_range": [0, 14]},
            {"page_number": 2, "chunk_range": [16, 30]},
        ],
    )
    assert len(quote_spans(chunk, ["same evidence."])) == 2


def test_prefix_of_a_large_section_does_not_tokenize_the_whole_remaining_document():
    from knowflow.app.token_budget import TokenCounter

    counter = TokenCounter()
    tokenizer = counter.tokenizer
    sizes = []

    class RecordingTokenizer:
        def encode(self, text, **kwargs):
            sizes.append(len(text))
            return tokenizer.encode(text, **kwargs)

    counter.tokenizer = RecordingTokenizer()
    text = "设备事件记录42。" * 100000
    end = counter.prefix_end(text, 128)
    assert 0 < end < len(text)
    assert max(sizes) <= 2048
    assert counter.count(text[:end]) <= 128
