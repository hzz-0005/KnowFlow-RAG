from knowflow.app.retrieval import _keyword_terms, table_row_match_score


def test_joint_conditions_in_one_row_outrank_keywords_pooled_from_unrelated_rows():
    query = "Alpha 模型的 Figure 结果是多少？"
    terms = _keyword_terms(query)
    unrelated = [
        {"cell_keys": ["Alpha", "Table"]},
        {"cell_keys": ["Beta", "Figure"]},
        {"cell_keys": ["Gamma", "Text"]},
    ]
    target = [{"cell_keys": ["Alpha", "Figure"]}]
    assert table_row_match_score(query, target, terms) > table_row_match_score(
        query, unrelated, terms
    )


def test_aggregate_coverage_still_breaks_ties_for_multiple_requested_rows():
    query = "Table 3 的 Pump 和 Valve 的 Count 各是多少？"
    terms = _keyword_terms(query)
    pump = {"cell_keys": ["Pump"]}
    valve = {"cell_keys": ["Valve"]}
    assert table_row_match_score(query, [pump, valve], terms) > table_row_match_score(
        query, [pump], terms
    )
