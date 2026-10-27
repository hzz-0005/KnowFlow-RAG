from copy import deepcopy
from io import BytesIO

import httpx
import pytest
from knowflow.app.config import Settings
from knowflow.app.document_processing import split_sections
from knowflow.app.structured_parsing import (
    ParsingFailure,
    docling_parse,
    native_parse,
    parse_for_ingestion,
)
from knowflow.app.structured_parsing import (
    legacy_chunk_parsed_document as chunk_parsed_document,
)
from pypdf import PdfWriter


def sample_conversion():
    def prov(page, left=10, top=90, right=90, bottom=80):
        return [
            {
                "page_no": page,
                "bbox": {
                    "l": left,
                    "t": top,
                    "r": right,
                    "b": bottom,
                    "coord_origin": "BOTTOMLEFT",
                },
            }
        ]

    texts = [
        {
            "self_ref": "#/texts/0",
            "label": "section_header",
            "level": 1,
            "text": "设备巡检",
            "prov": prov(1),
        },
        {
            "self_ref": "#/texts/1",
            "label": "text",
            "text": "每次检查必须确认压力单位。",
            "prov": prov(1),
        },
        {
            "self_ref": "#/texts/2",
            "label": "caption",
            "text": "表 1 压力上限（MPa）",
            "prov": prov(2),
        },
        {"self_ref": "#/texts/3", "label": "page_header", "text": "公司内部资料", "prov": prov(1)},
    ]
    cells = []
    for r, row in enumerate([["设备", "压力上限（MPa）"], ["泵 A", "1.8"], ["泵 B", "2.4"]]):
        for c, text in enumerate(row):
            cells.append(
                {
                    "text": text,
                    "start_row_offset_idx": r,
                    "end_row_offset_idx": r + 1,
                    "start_col_offset_idx": c,
                    "end_col_offset_idx": c + 1,
                    "column_header": r == 0,
                }
            )
    table = {
        "self_ref": "#/tables/0",
        "label": "table",
        "prov": prov(2),
        "captions": [{"$ref": "#/texts/2"}],
        "data": {"num_rows": 3, "num_cols": 2, "table_cells": cells},
    }
    document = {
        "version": "1.0.0",
        "texts": texts,
        "tables": [table],
        "pictures": [],
        "body": {
            "self_ref": "#/body",
            "children": [
                {"$ref": ref} for ref in ["#/texts/0", "#/texts/1", "#/texts/2", "#/tables/0"]
            ],
        },
    }
    return {"status": "success", "document": {"json_content": document}}


PAGES = [
    {"page_number": n, "native_characters": 0 if n == 2 else 12, "width": 100, "height": 100}
    for n in [1, 2]
]


def test_order_table_headers_units_and_original_positions_survive_chunking():
    parsed = docling_parse(sample_conversion(), PAGES)
    assert parsed.report["status"] == "parsed"
    assert parsed.report["unindexed_pages"] == []
    assert [b.kind for b in parsed.blocks] == [
        "section_header",
        "text",
        "caption",
        "table",
        "page_header",
    ]
    table_rows = [s for s in parsed.sections if "泵" in s.text]
    assert len(table_rows) == 2
    assert "设备: 泵 B | 压力上限（MPa）: 2.4" in table_rows[1].text
    assert table_rows[1].source_spans[0]["row_key"] == "泵 B"
    assert all("表 1" in s.text and s.section == "设备巡检" for s in table_rows)
    assert not any("公司内部资料" in c.content for c in split_sections(parsed.sections))
    source = split_sections(table_rows)[1].source_spans[0]
    assert source["page_number"] == 2
    assert source["bbox"] == {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.2}
    assert parsed.blocks[3].structure["num_rows"] == 3


def test_partial_conversion_and_missing_pages_are_explicit():
    payload = sample_conversion()
    payload["status"] = "partial_success"
    parsed = docling_parse(
        payload, [*PAGES, {"page_number": 3, "native_characters": 0, "width": 100, "height": 100}]
    )
    assert parsed.report["status"] == "needs_review"
    assert parsed.report["unindexed_pages"] == [3]
    assert any("部分成功" in warning for warning in parsed.report["warnings"])


def test_small_blocks_pack_dynamically_but_keep_regions_table_units_and_page_boundaries():
    parsed = docling_parse(sample_conversion(), PAGES)
    chunks = chunk_parsed_document(parsed)
    assert len(chunks) == 3  # page 1 prose, page 2 caption, page 2 table rows
    first = chunks[0]
    assert "设备巡检" in first.content and "压力单位" in first.content
    assert {s["block_key"] for s in first.source_spans} == {"#/texts/0", "#/texts/1"}
    table = chunks[-1]
    assert "泵 A" in table.content and "泵 B" in table.content and "MPa" in table.content
    assert table.page_number == 2
    assert all(len(c.content) <= 1000 for c in chunks)
    assert table.content.count("表 1 压力上限（MPa）") == 1
    assert {s["row_key"] for s in table.source_spans if s.get("row_key")} == {"泵 A", "泵 B"}


def test_unknown_picture_semantics_and_unreachable_content_are_not_silently_accepted():
    payload = sample_conversion()
    doc = payload["document"]["json_content"]
    doc["pictures"] = [{"self_ref": "#/pictures/0", "label": "picture", "prov": [{"page_no": 2}]}]
    doc["body"]["children"].append({"$ref": "#/pictures/0"})
    doc["texts"].append({"self_ref": "#/texts/4", "label": "text", "text": "未链接的关键正文"})
    parsed = docling_parse(payload, PAGES)
    assert parsed.report["status"] == "needs_review"
    assert parsed.report["picture_count"] == 1
    assert any("未被正文树引用" in w for w in parsed.report["warnings"])
    assert any("尚未经过视觉解释" in w for w in parsed.report["warnings"])


def test_outside_source_page_or_corrupt_cell_coordinates_are_rejected():
    payload = sample_conversion()
    payload["document"]["json_content"]["texts"][0]["prov"][0]["page_no"] = 99
    with pytest.raises(ValueError, match="outside"):
        docling_parse(payload, PAGES)
    payload = deepcopy(sample_conversion())
    payload["document"]["json_content"]["tables"][0]["data"]["table_cells"][0][
        "end_row_offset_idx"
    ] = 99
    with pytest.raises(ValueError, match="cell coordinates"):
        docling_parse(payload, PAGES)


def test_decoding_replacement_characters_are_flagged_even_when_all_pages_have_text():
    payload = sample_conversion()
    payload["document"]["json_content"]["texts"][1]["text"] = "已提取但缺字\ufffd，不能算完整。"
    parsed = docling_parse(payload, PAGES)
    assert parsed.report["status"] == "needs_review"
    assert parsed.report["unindexed_pages"] == []
    assert parsed.report["replacement_characters"] == 1
    assert parsed.report["damaged_text_pages"] == [1]


def test_cross_page_charspans_split_cleaned_text_without_moving_caption_to_previous_page():
    payload = sample_conversion()
    item = payload["document"]["json_content"]["texts"][1]
    first, second = "See Table 1 for limits.", "Table 1: Pump limits."
    raw = "  " + first + "  \t" + second + "  "
    item["text"] = raw
    start = raw.index(second)
    item["prov"] = [
        {"page_no": 1, "charspan": [0, start]},
        {"page_no": 2, "charspan": [start, len(raw)]},
    ]
    parsed = docling_parse(payload, PAGES)
    parts = [
        s
        for s in parsed.sections
        if any(p["block_key"] == item["self_ref"] for p in s.source_spans)
    ]
    assert [(s.page_number, s.text) for s in parts] == [(1, first), (2, second)]
    assert all(len(s.source_spans) == 1 for s in parts)
    assert parts[1].source_spans[0]["raw_charspan"] == [start, len(raw)]
    clean = next(b.text for b in parsed.blocks if b.block_key == item["self_ref"])
    lo, hi = parts[1].source_spans[0]["text_range"]
    assert clean[lo:hi] == second
    assert not any(
        c.page_number == 1 and second in c.content for c in chunk_parsed_document(parsed)
    )


def test_recovered_caption_is_searchable_with_table_rows_and_both_source_regions():
    payload = sample_conversion()
    document = payload["document"]["json_content"]
    first, caption = "See Table 1 for pressure limits.", "Table 1: Pump limits (MPa)."
    item = document["texts"][1]
    item["text"] = first + " " + caption
    item["prov"] = [
        {
            "page_no": 1,
            "charspan": [0, len(first)],
            "bbox": {"l": 10, "r": 40, "t": 20, "b": 10, "coord_origin": "BOTTOMLEFT"},
        },
        {
            "page_no": 2,
            "charspan": [len(first) + 1, len(item["text"])],
            "bbox": {"l": 10, "r": 40, "t": 90, "b": 82, "coord_origin": "BOTTOMLEFT"},
        },
    ]
    document["tables"][0]["captions"] = []
    document["tables"][0]["prov"][0]["bbox"].update(l=10, r=40, t=80, b=50)
    document["texts"][2]["text"] = "Unrelated note."
    document["texts"][2]["prov"][0]["bbox"].update(l=60, r=90)
    parsed = docling_parse(payload, PAGES)
    assert parsed.report["table_caption_recovery"]["recovered"] == 1
    rows = [s for s in parsed.sections if "泵 A" in s.text]
    assert len(rows) == 1 and rows[0].text.startswith(caption)
    assert rows[0].page_number == 2
    assert {s["block_key"] for s in rows[0].source_spans} == {"#/texts/1", "#/tables/0"}
    assert all(s["page_number"] == 2 for s in rows[0].source_spans)


@pytest.mark.parametrize("bounds", [[[0, 5], [8, 12]], [[0, 10], [5, 12]], [[0, 100], [0, 0]]])
def test_invalid_incomplete_or_overlapping_page_charspans_preserve_text_with_warning(bounds):
    payload = sample_conversion()
    item = payload["document"]["json_content"]["texts"][1]
    item["text"] = "abcdefghijkl"
    item["prov"] = [
        {"page_no": page, "charspan": bound} for page, bound in zip([1, 2], bounds, strict=True)
    ]
    parsed = docling_parse(payload, PAGES)
    parts = [
        s
        for s in parsed.sections
        if any(p["block_key"] == item["self_ref"] for p in s.source_spans)
    ]
    assert len(parts) == 1 and parts[0].text == item["text"]
    assert len(parts[0].source_spans) == 2
    assert any("跨页字符范围无法完整校验" in warning for warning in parsed.report["warnings"])


@pytest.mark.asyncio
async def test_docling_service_failure_never_silently_falls_back(monkeypatch):
    from knowflow.app import structured_parsing

    monkeypatch.setattr(structured_parsing, "inspect_pdf", lambda *args: PAGES)

    class OfflineClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, *args, **kwargs):
            raise httpx.ConnectError("offline")

    monkeypatch.setattr(structured_parsing.httpx, "AsyncClient", lambda **kwargs: OfflineClient())
    with pytest.raises(ParsingFailure, match="不可用") as error:
        await parse_for_ingestion("mixed.pdf", b"not used", Settings(pdf_parser="docling"))
    assert error.value.report["status"] == "failed"
    assert error.value.report["parser"] == "docling"


def test_native_scan_error_and_page_limit_are_actionable():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    stream = BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match="OCR"):
        native_parse("scan.pdf", stream.getvalue(), Settings())
    with pytest.raises(ValueError, match="上限"):
        native_parse("scan.pdf", stream.getvalue(), Settings(pdf_max_pages=1))


@pytest.mark.asyncio
@pytest.mark.parametrize("task_status", ["success", "failure"])
async def test_async_converter_contract_and_explicit_worker_failure(monkeypatch, task_status):
    from types import SimpleNamespace

    from knowflow.app import structured_parsing

    monkeypatch.setattr(structured_parsing, "inspect_pdf", lambda *args: PAGES)
    requests = []

    def response(data):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    class Service:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            assert url.endswith("/file/async")
            assert kwargs["data"]["ocr_lang"] == ["iso:zh"]
            return response({"task_id": "11111111-1111-1111-1111-111111111111"})

        async def get(self, url):
            requests.append(url)
            return response(
                {"task_status": task_status} if "/status/" in url else sample_conversion()
            )

    monkeypatch.setattr(structured_parsing.httpx, "AsyncClient", lambda **kwargs: Service())
    if task_status == "failure":
        with pytest.raises(ParsingFailure, match="版面分析失败"):
            await parse_for_ingestion("source.pdf", b"unused", Settings(pdf_parser="docling"))
        assert (
            len(requests) == 1
        )  # Failed jobs have no result; avoid misreporting their 404 as offline.
    else:
        parsed = await parse_for_ingestion("source.pdf", b"unused", Settings(pdf_parser="docling"))
        assert parsed.report["parser"] == "docling"
        assert len(requests) == 2


@pytest.mark.asyncio
async def test_forced_ocr_requires_structured_parser():
    with pytest.raises(ParsingFailure, match="启用 Docling"):
        await parse_for_ingestion(
            "source.pdf", b"unused", Settings(pdf_parser="native"), force_ocr=True
        )


@pytest.mark.asyncio
async def test_requested_page_range_is_validated_before_conversion(monkeypatch):
    from knowflow.app import structured_parsing

    monkeypatch.setattr(structured_parsing, "inspect_pdf", lambda *args: PAGES)
    with pytest.raises(ParsingFailure, match="超出原文件"):
        await parse_for_ingestion(
            "source.pdf", b"unused", Settings(pdf_parser="docling"), page_range=(3, 3)
        )


@pytest.mark.asyncio
async def test_partial_page_conversion_keeps_physical_page_numbers(monkeypatch):
    from types import SimpleNamespace

    from knowflow.app import structured_parsing

    monkeypatch.setattr(structured_parsing, "inspect_pdf", lambda *args: PAGES)
    document = {
        "body": {"self_ref": "#/body", "children": [{"$ref": "#/texts/0"}]},
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "第二页扫描内容。",
                "prov": [{"page_no": 2}],
            }
        ],
    }

    def response(data):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    class Service:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            assert kwargs["data"]["page_range"] == ["2", "2"]
            assert kwargs["data"]["force_ocr"] == "true"
            return response({"task_id": "11111111-1111-1111-1111-111111111111"})

        async def get(self, url):
            return response(
                {"task_status": "success"}
                if "/status/" in url
                else {"status": "success", "document": {"json_content": document}}
            )

    monkeypatch.setattr(structured_parsing.httpx, "AsyncClient", lambda **kwargs: Service())
    parsed = await parse_for_ingestion(
        "source.pdf", b"unused", Settings(pdf_parser="docling"), force_ocr=True, page_range=(2, 2)
    )
    assert parsed.sections[0].page_number == 2
    assert parsed.report["pages"][0]["page_number"] == 2
    assert parsed.report["page_count"] == 1
    assert parsed.report["source_page_count"] == 2
