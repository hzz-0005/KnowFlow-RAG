from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import retrieval
from knowflow.app.config import get_settings
from knowflow.app.retrieval import (
    RetrievalHit,
    _keyword_terms,
    _query_variants,
    append_graph_candidates,
    graph_candidate_diagnostics,
    mentioned_table_keys,
    reciprocal_rank_fusion,
)


@pytest.mark.parametrize("limit", [1, 3, 5])
@pytest.mark.parametrize("reverse", [False, True])
def test_graph_pool_cannot_evict_text_top_k_and_has_a_separate_bounded_budget(limit, reverse):
    # Identifiers and graph order vary; no document names or question IDs drive protection.
    base = [_hit(uuid4()) for _ in range(limit)]
    graph = [_hit(uuid4()) for _ in range(limit * 3)]
    shared = _hit(base[-1].chunk.id)
    graph.insert(1, shared)
    if reverse:
        graph.reverse()
    for rank, hit in enumerate(graph, 1):
        hit.graph_rank = rank
    original = [hit.chunk.id for hit in base]
    result = append_graph_candidates(base, graph, limit)
    assert [hit.chunk.id for hit in result[:limit]] == original
    assert len(result) == 2 * limit
    assert len({hit.chunk.id for hit in result}) == len(result)
    assert all(hit.retrieval_stage == "base" for hit in result[:limit])
    assert all(hit.retrieval_stage == "graph_expansion" for hit in result[limit:])
    assert result[limit - 1].graph_rank == shared.graph_rank
    assert graph_candidate_diagnostics(result, limit)["added_count"] == limit


def test_graph_pool_handles_empty_shared_and_repeated_candidates_without_fabricating_additions():
    base = [_hit(uuid4())]
    shared = _hit(base[0].chunk.id)
    shared.graph_rank = 1
    assert append_graph_candidates(base, [], 3) == base
    result = append_graph_candidates(base, [shared, shared], 3)
    assert len(result) == 1
    assert graph_candidate_diagnostics(result, 3)["shared_count"] == 1
    only_graph = append_graph_candidates([], [shared, shared], 3)
    assert len(only_graph) == 1
    assert only_graph[0].retrieval_stage == "graph_expansion"


@pytest.mark.asyncio
async def test_online_graph_hybrid_uses_same_text_prefix_and_one_embedding_pass(monkeypatch):
    from knowflow.app.config import Settings

    base = [_hit(uuid4()) for _ in range(3)]
    extra = [_hit(uuid4()) for _ in range(4)]
    embedding_calls = []

    async def vectors(*args):
        embedding_calls.append(1)
        return [base]

    async def keywords(*args):
        return base

    async def graph(*args, **kwargs):
        assert kwargs["seed_chunk_ids"] == [hit.chunk.id for hit in hybrid]
        for rank, hit in enumerate(extra, 1):
            hit.graph_rank = rank
        return extra

    monkeypatch.setattr(retrieval, "_vector_search_many", vectors)
    monkeypatch.setattr(retrieval, "_keyword_search", keywords)
    monkeypatch.setattr(retrieval, "_graph_search", graph)
    settings = Settings(table_row_retrieval_enabled=False, english_keyword_retrieval_enabled=False)
    hybrid = await retrieval.retrieve(None, uuid4(), "A route?", "hybrid", 3, settings)
    embedding_calls.clear()
    augmented = await retrieval.retrieve(None, uuid4(), "A route?", "graph_hybrid", 3, settings)
    assert [h.chunk.id for h in augmented[:3]] == [h.chunk.id for h in hybrid]
    assert [h.fused_score for h in augmented[:3]] == [h.fused_score for h in hybrid]
    assert len(augmented) == 6 and len(embedding_calls) == 1


async def _no_lexical(*_args, **_kwargs):
    return []


def _hit(identifier):
    return RetrievalHit(chunk=SimpleNamespace(id=identifier), filename="manual.md")


@pytest.mark.asyncio
async def test_english_expansion_preserves_base_order_and_separate_limit(monkeypatch):
    from knowflow.app.config import Settings

    base = [_hit(uuid4()), _hit(uuid4())]
    shared, extra = _hit(base[1].chunk.id), [_hit(uuid4()) for _ in range(5)]
    ranking = [shared, *extra, extra[0]]
    for rank, hit in enumerate(ranking, 1):
        hit.english_keyword_rank = rank

    async def search(*_):
        return ranking

    monkeypatch.setattr(retrieval, "_english_keyword_search", search)
    original = [h.chunk.id for h in base]
    result = await retrieval._append_english_candidates(
        None, uuid4(), ["filter table"], base, 2, Settings(_env_file=None)
    )
    assert [h.chunk.id for h in result[:2]] == original
    assert result[1].english_keyword_rank == 1
    assert len(result) == 4 and len({h.chunk.id for h in result}) == 4
    assert all(h.retrieval_stage == "text_expansion" for h in result[2:])


def test_rrf_rewards_documents_appearing_in_both_rankings() -> None:
    shared, vector_only, keyword_only = uuid4(), uuid4(), uuid4()
    vector_hits = [_hit(shared), _hit(vector_only)]
    vector_hits[0].cosine_distance = 0.12
    vector_hits[0].vector_rank = 1
    vector_hits[1].cosine_distance = 0.22
    vector_hits[1].vector_rank = 2
    keyword_hits = [_hit(shared), _hit(keyword_only)]
    keyword_hits[0].keyword_rank = 1
    keyword_hits[1].keyword_rank = 2

    result = reciprocal_rank_fusion([vector_hits, keyword_hits])

    assert result[0].chunk.id == shared
    assert {hit.chunk.id for hit in result[1:]} == {vector_only, keyword_only}
    assert result[0].vector_rank == 1
    assert result[0].keyword_rank == 1
    assert result[0].cosine_distance == 0.12
    assert result[0].fused_score == pytest.approx(2 / 61)
    by_id = {hit.chunk.id: hit for hit in result}
    assert by_id[vector_only].vector_rank == 2
    assert by_id[vector_only].keyword_rank is None
    assert by_id[keyword_only].keyword_rank == 2
    assert by_id[keyword_only].vector_rank is None


def test_rrf_rejects_non_positive_smoothing_constant() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([], k=0)


@pytest.mark.asyncio
@pytest.mark.parametrize("assessment_enabled", [True, False])
async def test_empty_hybrid_pool_recovers_bounded_candidates_only_with_mandatory_review(
    monkeypatch, assessment_enabled
):
    from knowflow.app.config import Settings

    recovered = _hit(uuid4())
    calls = []

    async def vectors(_session, _kb, queries, limit, settings):
        calls.append((queries, limit, settings.rag_max_cosine_distance))
        return [[recovered] for _ in queries] if settings.rag_max_cosine_distance == 2 else [[]]

    async def keywords(*_):
        return []

    monkeypatch.setattr(retrieval, "_vector_search_many", vectors)
    monkeypatch.setattr(retrieval, "_keyword_search", keywords)
    question = "谁负责转交？数据存放在哪里？"
    settings = Settings(
        table_row_retrieval_enabled=False,
        retrieval_assessment_enabled=assessment_enabled,
        english_keyword_retrieval_enabled=False,
    )
    hits = await retrieval.retrieve(None, uuid4(), question, "hybrid", 3, settings)
    if assessment_enabled:
        assert len(calls) == 2 and calls[-1][1] == 1
        assert calls[-1][0][0] == question
        assert len(hits) == 1 and hits[0].distance_gate_fallback
    else:
        assert len(calls) == 1 and not hits
    assert settings.rag_max_cosine_distance == 0.35


def test_query_variants_split_chinese_subquestions_without_splitting_single_question() -> None:
    query = "pgvector 的 <=> 运算符计算什么？余弦相似度如何从余弦距离得到？"

    assert _query_variants(query) == [
        "pgvector 的 <=> 运算符计算什么",
        "余弦相似度如何从余弦距离得到",
    ]
    assert _query_variants("余弦相似度如何计算？") == ["余弦相似度如何计算？"]


def test_keyword_terms_ignore_operator_punctuation() -> None:
    terms = _keyword_terms("pgvector 的 <=> 运算符计算什么？")

    assert "pgvector" in terms
    assert "运算符" in terms
    assert all(any(character.isalnum() for character in term) for term in terms)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "按照 2025 年生产策略，超时是多少？不是问 2026 版。",
            ["按照 2025 年生产策略，超时是多少"],
        ),
        ("Find the 2025 policy? Not asking about 2026?", ["Find the 2025 policy"]),
        (
            "Production 是否禁止写入？Staging 是否允许写入？",
            ["Production 是否禁止写入", "Staging 是否允许写入"],
        ),
        ("不要查旧版本", ["不要查旧版本"]),
    ],
)
def test_exclusion_constraints_do_not_become_positive_subqueries(question, expected):
    assert _query_variants(question) == expected


def test_literal_table_keys_match_requested_labels_without_matching_context_as_text():
    spans = [{"row_key": value} for value in ["Text", "Table", "甲设备", "other", ""]]
    assert mentioned_table_keys("Table 类和 Text类的 Count？甲设备是多少？", spans) == {
        "text",
        "table",
        "甲设备",
    }
    assert mentioned_table_keys("context tables and otherworld", spans) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_table_hint_ablation_switch_controls_only_the_extra_retrieval_path(
    monkeypatch, enabled
):
    from knowflow.app.config import Settings

    base, table = _hit(uuid4()), _hit(uuid4())
    table.table_rank = 1
    calls = []

    async def vectors(*args):
        return [[base]]

    async def keywords(*args):
        return []

    async def table_rows(*args):
        calls.append("table")
        return [table]

    monkeypatch.setattr(retrieval, "_vector_search_many", vectors)
    monkeypatch.setattr(retrieval, "_keyword_search", keywords)
    monkeypatch.setattr(retrieval, "_table_row_search", table_rows)
    hits = await retrieval.retrieve(
        None,
        uuid4(),
        "Table Count?",
        "hybrid",
        5,
        Settings(table_row_retrieval_enabled=enabled, english_keyword_retrieval_enabled=False),
    )
    assert bool(calls) is enabled
    assert {hit.chunk.id for hit in hits} == (
        {base.chunk.id, table.chunk.id} if enabled else {base.chunk.id}
    )


@pytest.mark.asyncio
async def test_vector_retrieval_fuses_evidence_for_each_subquestion(monkeypatch) -> None:
    query = "pgvector 的 <=> 运算符计算什么？余弦相似度如何从余弦距离得到？"
    knowledge_base_id = uuid4()
    operator = RetrievalHit(
        chunk=SimpleNamespace(id=uuid4(), content="<=> is cosine distance"),
        filename="pgvector.md",
        cosine_distance=0.1,
        vector_rank=1,
    )
    formula = RetrievalHit(
        chunk=SimpleNamespace(id=uuid4(), content="Cosine similarity is 1 - cosine distance"),
        filename="pgvector.md",
        cosine_distance=0.2,
        vector_rank=1,
    )

    async def fake_vector_search_many(_session, _kb_id, queries, _limit, _settings):
        assert queries == [
            "pgvector 的 <=> 运算符计算什么",
            "余弦相似度如何从余弦距离得到",
        ]
        return [[operator], [formula]]

    monkeypatch.setattr(retrieval, "_vector_search_many", fake_vector_search_many)

    hits = await retrieval.retrieve(
        session=None,
        knowledge_base_id=knowledge_base_id,
        query=query,
        strategy="vector",
        top_k=5,
        settings=get_settings(),
    )

    assert {hit.chunk.id for hit in hits} == {operator.chunk.id, formula.chunk.id}
    by_id = {hit.chunk.id: hit for hit in hits}
    assert by_id[operator.chunk.id].vector_rank == 1
    assert by_id[formula.chunk.id].vector_rank == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("assessment_enabled", [True, False])
async def test_empty_vector_pool_recovers_flagged_nearest_only_with_mandatory_review(
    monkeypatch, assessment_enabled
):
    from knowflow.app.config import Settings

    recovered = _hit(uuid4())
    calls = []

    async def vectors(_session, _kb, queries, limit, settings, **kwargs):
        calls.append((tuple(queries), limit, settings.rag_max_cosine_distance))
        return [[recovered]] if settings.rag_max_cosine_distance == 2 else [[]]

    monkeypatch.setattr(retrieval, "_vector_search_many", vectors)
    settings = Settings(
        _env_file=None, retrieval_assessment_enabled=assessment_enabled
    )
    question = "哪些项目没有使用 Temporal？"
    hits = await retrieval.retrieve(None, uuid4(), question, "vector", 3, settings)
    if assessment_enabled:
        assert len(calls) == 2 and calls[-1][1] == 1
        assert calls[-1][0] == (question,)
        assert len(hits) == 1 and hits[0].distance_gate_fallback
    else:
        assert len(calls) == 1 and not hits
    assert settings.rag_max_cosine_distance == 0.35


@pytest.mark.asyncio
async def test_per_document_sweep_embeds_the_question_once_and_reuses_it(monkeypatch):
    from knowflow.app.config import Settings

    embed_calls, search_calls = [], []

    async def embed(queries, _settings):
        embed_calls.append(list(queries))
        return [[0.1, 0.2] for _ in queries]

    async def search(_session, _kb, queries, limit, _settings, **kwargs):
        search_calls.append(kwargs.get("vectors") is not None)
        return [[_hit(uuid4()) for _ in range(limit)] for _ in queries]

    monkeypatch.setattr(retrieval, "embed_texts", embed)
    monkeypatch.setattr(retrieval, "_vector_search_many", search)
    monkeypatch.setattr(retrieval, "_keyword_search", _no_lexical)
    question = "哪些项目没有使用 Temporal？"
    documents = [uuid4() for _ in range(5)]
    hits = await retrieval.retrieve_per_document(
        None,
        uuid4(),
        question,
        "vector",
        10,
        Settings(_env_file=None),
        document_ids=documents,
    )
    # Five per-file searches must not spend five identical embedding requests.
    assert embed_calls == [[question]]
    assert len(search_calls) == 5 and all(search_calls)
    assert len(hits) == 10
    assert len({hit.chunk.id for hit in hits}) == 10


@pytest.mark.asyncio
async def test_per_document_sweep_flags_only_the_files_sampled_after_the_gate(monkeypatch):
    from knowflow.app.config import Settings

    documents = [uuid4() for _ in range(3)]
    silent = documents[0]

    async def search(session, _kb, _queries, limit, settings, **kwargs):
        # Only the first file empties under the distance gate and needs a sample.
        if not (
            settings.rag_max_cosine_distance == 2 or silent not in session.document_ids
        ):
            return [[]]
        return [[_hit(uuid4()) for _ in range(limit)]]

    monkeypatch.setattr(retrieval, "_vector_search_many", search)
    monkeypatch.setattr(retrieval, "_keyword_search", _no_lexical)
    settings = Settings(_env_file=None)
    hits = await retrieval.retrieve_per_document(
        None,
        uuid4(),
        "每个项目的量化成果分别是什么？",
        "vector",
        6,
        settings,
        document_ids=documents,
    )
    flagged = [hit for hit in hits if hit.distance_gate_fallback]
    # Every file still contributes its quota; only the gated one is marked.
    assert len(hits) == 6
    assert len(flagged) == 2
    # Widening the gate must stay a per-call copy, never a shared mutation.
    assert settings.rag_max_cosine_distance == 0.35


@pytest.mark.asyncio
async def test_per_document_sweep_adds_each_files_own_lexical_evidence(monkeypatch):
    from knowflow.app.config import Settings

    documents = [uuid4(), uuid4()]
    lexical_id = uuid4()
    lexical_calls = []

    async def vectors(_session, _kb, _queries, limit, _settings, **kwargs):
        # Nearest neighbours alone: these two are the sample, never a lexical match.
        return [[_hit(uuid4()) for _ in range(limit)]]

    async def keyword(session, _kb, _query, limit, **kwargs):
        # A per-file lexical pass relaxes the match rule for that one file.
        assert kwargs.get("minimum_matches") == 1
        lexical_calls.append(tuple(session.document_ids))
        hit = _hit(lexical_id)
        hit.keyword_rank = 1
        return [hit] if session.document_ids[0] == documents[1] else []

    monkeypatch.setattr(retrieval, "_vector_search_many", vectors)
    monkeypatch.setattr(retrieval, "_keyword_search", keyword)
    hits = await retrieval.retrieve_per_document(
        None,
        uuid4(),
        "哪些项目没有做真实用户测试？",
        "vector",
        2,
        Settings(_env_file=None),
        document_ids=documents,
    )
    # The file whose own wording matches contributes it, the other one still contributes.
    assert lexical_calls == [(documents[0],), (documents[1],)]
    assert lexical_id in {hit.chunk.id for hit in hits}
    assert len(hits) == 2
