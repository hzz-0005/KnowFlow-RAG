from copy import deepcopy
from types import SimpleNamespace

import pytest
from knowflow.app.document_links import extract_links, reference_excerpt
from knowflow.app.reference_expansion import source_occurrence_in_chunk
from knowflow.app.structured_parsing import Block
from knowflow.app.table_captions import recover_table_captions


def span(key, page, top, bottom, left=0.1, right=0.45, bounds=None):
    result = {
        "block_key": key,
        "page_number": page,
        "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
    }
    if bounds:
        result["text_range"] = bounds
    return result


def merged_blocks():
    first = "See Table 5 for the comparison."
    caption = "Table 5: Comparison of Orion and Cedar."
    text = first + " " + caption
    return [
        Block(
            "merged",
            "text",
            text,
            source_spans=[
                span("merged", 7, 0.8, 0.9, bounds=[0, len(first)]),
                span("merged", 8, 0.1, 0.19, bounds=[len(first) + 1, len(text)]),
            ],
        ),
        Block(
            "table",
            "table",
            "Orion: 83",
            source_spans=[span("table", 8, 0.22, 0.43)],
            structure={"captions": []},
        ),
        Block(
            "later",
            "text",
            "As shown in Table 5, results differ.",
            source_spans=[span("later", 8, 0.6, 0.7)],
        ),
    ]


def test_merged_caption_recovery_uses_page_range_keeps_body_and_excludes_definition():
    blocks = merged_blocks()
    texts = [b.text for b in blocks]
    report = recover_table_captions(blocks)
    links, index = extract_links(blocks)
    assert report["recovered"] == 1
    recovery = blocks[1].structure["caption_recovery"]
    assert recovery["source_span"]["page_number"] == 8
    assert recovery["text"].startswith("Table 5:")
    assert [b.text for b in blocks] == texts
    assert blocks[1].structure["captions"] == []
    assert len(links) == 2 and index["statuses"] == {"resolved": 2}
    assert all(link.target_block_key == "table" for link in links)
    excerpt, source_spans = reference_excerpt(blocks[0], links[0].start_offset)
    assert excerpt == "See Table 5 for the comparison."
    assert [s["page_number"] for s in source_spans] == [7]
    # Same label in the p8 caption must not follow the p7 prose occurrence.
    caption_chunk = SimpleNamespace(
        content=recovery["text"], source_spans=[recovery["source_span"]]
    )
    assert not source_occurrence_in_chunk(links[0], caption_chunk)
    assert source_occurrence_in_chunk(
        links[0], SimpleNamespace(content=blocks[0].text, source_spans=blocks[0].source_spans[:1])
    )
    before = deepcopy(recovery)
    recover_table_captions(blocks)
    assert blocks[1].structure["caption_recovery"] == before


@pytest.mark.parametrize(
    "change",
    [
        "other_page",
        "other_column",
        "overlap",
        "distant",
        "no_range",
        "prose",
        "intervening",
        "duplicate_table",
        "two_captions",
    ],
)
def test_unsafe_caption_associations_remain_unconnected(change):
    blocks = merged_blocks()
    table = blocks[1]
    if change == "other_page":
        table.source_spans[0]["page_number"] = 9
    elif change == "other_column":
        table.source_spans[0]["bbox"].update(left=0.55, right=0.9)
    elif change == "overlap":
        table.source_spans[0]["bbox"]["top"] = 0.15
    elif change == "distant":
        table.source_spans[0]["bbox"]["top"] = 0.4
    elif change == "no_range":
        for s in blocks[0].source_spans:
            s.pop("text_range")
    elif change == "prose":
        blocks[0].text = blocks[0].text.replace("Table 5:", "Table 5,")
    elif change == "intervening":
        blocks.append(
            Block(
                "gap", "text", "Unrelated limitation", source_spans=[span("gap", 8, 0.195, 0.215)]
            )
        )
    elif change == "duplicate_table":
        other = deepcopy(table)
        other.block_key = "second"
        blocks.append(other)
    elif change == "two_captions":
        blocks.append(
            Block(
                "caption2",
                "caption",
                "Table 9: Another definition.",
                source_spans=[span("caption2", 8, 0.45, 0.48)],
            )
        )
    assert recover_table_captions(blocks)["recovered"] == 0
    links, _ = extract_links(blocks)
    assert all(link.status == "missing" for link in links)


def test_stale_caption_offsets_and_same_number_tables_do_not_produce_false_unique_target():
    blocks = merged_blocks()
    recover_table_captions(blocks)
    other = Block("other", "table", "row", structure={"captions": ["Table 5: Different table"]})
    links, _ = extract_links([*blocks, other])
    assert all(link.status == "ambiguous" for link in links)
    blocks[0].text = blocks[0].text.replace("Cedar", "Vesta")
    links, _ = extract_links(blocks)
    assert all(link.status == "missing" for link in links)
