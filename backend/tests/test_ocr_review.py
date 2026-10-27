from types import SimpleNamespace

from knowflow.app.ocr_review import compare_snapshots, page_snapshot


def block(text, page=2, kind="text", spans=None):
    return SimpleNamespace(
        kind=kind, text=text, block_key="test", source_spans=spans or [{"page_number": page}]
    )


def test_zero_replacement_does_not_hide_missing_text_numbers_or_tables():
    before = page_snapshot(
        [block("备份周期17天，17天，覆盖83%和11天。\ufffd"), block("表格", kind="table")],
        2,
        "docling",
    )
    after = page_snapshot([block("备份周期17天，覆盖88%。")], 2, "ocr")
    comparison = compare_snapshots(before, after)
    assert after["replacement_characters"] == 0
    assert comparison["decision"] == "manual_review_required"
    assert comparison["character_ratio"] < 0.7
    assert {item["value"]: item["count"] for item in comparison["missing_numbers"]} == {
        "17": 1,
        "83%": 1,
        "11": 1,
    }
    assert comparison["added_numbers"] == [{"value": "88%", "count": 1}]
    assert any("表格" in warning for warning in comparison["warnings"])


def test_empty_baseline_has_unknown_retention_and_shared_page_provenance_is_explicit():
    before = page_snapshot([], 2, "docling")
    after = page_snapshot(
        [block("文字17", spans=[{"page_number": 1}, {"page_number": 2}])], 2, "ocr"
    )
    result = compare_snapshots(before, after)
    assert result["character_ratio"] is None
    assert result["baseline_token_retention"] is None
    assert any("跨页" in warning for warning in result["warnings"])


def test_page_snapshot_excludes_other_pages_furniture_and_pictures():
    result = page_snapshot(
        [
            block("正文"),
            block("另一页", page=1),
            block("页眉", kind="page_header"),
            block("图", kind="picture"),
        ],
        2,
        "docling",
    )
    assert result["text"] == "正文"
    assert result["block_count"] == 1


def test_page_snapshot_uses_complete_ranges_but_warns_when_cross_page_ranges_are_incomplete():
    text = "Earlier prose 17. Table 5: Comparison 83%."
    boundary = text.index("Table")
    item = block(
        text,
        spans=[
            {"page_number": 7, "text_range": [0, boundary]},
            {"page_number": 8, "text_range": [boundary, len(text)]},
        ],
    )
    result = page_snapshot([item], 8, "docling")
    assert result["text"] == "Table 5: Comparison 83%."
    assert result["shared_page_blocks"] == []
    assert result["partitioned_page_blocks"] == ["test"]
    item.source_spans[0]["text_range"][1] -= 3
    fallback = page_snapshot([item], 8, "docling")
    assert fallback["text"] == text
    assert fallback["shared_page_blocks"] == ["test"]
