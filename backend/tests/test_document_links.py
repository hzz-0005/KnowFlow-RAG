from types import SimpleNamespace

from knowflow.app.document_links import extract_links
from knowflow.app.reference_expansion import section_target_keys
from knowflow.app.structured_parsing import Block


def heading(key, text):
    return Block(key, "section_header", text, section=text)


def test_references_resolve_distant_targets_in_both_languages_without_caption_duplicates():
    blocks = [
        heading("scope", "2 Deployment Scope"),
        Block("table", "table", "Cycle: 17 days", structure={"captions": ["Table 1: Cycle"]}),
        Block("caption", "caption", "Table 1: Cycle"),
        Block("fig", "caption", "Figure 3: Example"),
        Block("source", "text", "See Section 2, Table 1 and Fig. 3. 也可参见第2节和表1。"),
    ]
    links, report = extract_links(blocks)
    assert len(links) == 5
    assert report["statuses"] == {"resolved": 5}
    assert {link.target_block_key for link in links} == {"scope", "table", "fig"}
    assert all(link.source_block_key == "source" for link in links)


def test_duplicate_number_missing_table_and_explicit_external_reference_are_not_resolved():
    blocks = [
        heading("scope_a", "2 Scope A"),
        heading("scope_b", "2 Scope B"),
        Block("caption", "caption", "Table 9: A caption with no table body"),
        Block("source", "text", "参见第2节。见表9。See external Figure 4."),
    ]
    links, _ = extract_links(blocks)
    assert [link.status for link in links] == ["ambiguous", "missing", "external_unresolved"]
    assert links[0].candidate_keys == ["scope_a", "scope_b"]
    assert all(link.target_block_key is None for link in links)


def test_reference_sentence_terminators_decimal_numbers_and_subfigure_boundaries():
    blocks = [
        heading("scope", "2.1 Conditions"),
        Block("table", "table", "row", structure={"captions": ["Table 12: Totals"]}),
        Block("source", "text", "See Section 2.1. Table 12. Fig. 3a is not a whole-figure label."),
    ]
    links, _ = extract_links(blocks)
    assert [(link.source_quote, link.target_block_key) for link in links] == [
        ("Section 2.1", "scope"),
        ("Table 12", "table"),
    ]


def test_reference_and_ambiguous_candidate_limits_never_select_a_truncated_candidate():
    blocks = [heading(str(n), "2 Duplicate") for n in range(30)]
    blocks.append(Block("source", "text", "See Section 2. " * 2100))
    links, report = extract_links(blocks)
    assert len(links) == 2000
    assert report["truncated"] is True
    assert all(link.status == "ambiguous" and link.target_block_key is None for link in links)
    assert len(links[0].candidate_keys) == 20


def test_misclassified_unnumbered_heading_does_not_drop_a_numbered_sections_scope():
    blocks = [
        SimpleNamespace(ordinal=i, block_key=str(i), kind=kind, text=text, section=section)
        for i, (kind, text, section) in enumerate(
            [
                ("section_header", "2 部署范围", "2 部署范围"),
                ("text", "G1 对应 Atlas。", "2 部署范围"),
                ("section_header", "部署范围：仅测试环境。", "部署范围：仅测试环境。"),
                ("section_header", "2.1 附加条件", "2.1 附加条件"),
                ("text", "限制说明", "2.1 附加条件"),
                ("section_header", "3 其他章节", "3 其他章节"),
            ]
        )
    ]
    keys, truncated, inferred = section_target_keys(blocks[0], blocks, "section:2")
    assert keys == ["0", "1", "2", "3", "4"]
    assert truncated is False and inferred is True
