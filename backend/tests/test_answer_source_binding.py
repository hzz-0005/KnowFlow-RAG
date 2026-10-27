from uuid import uuid4

import pytest
from knowflow.app import answer_guard
from knowflow.app.config import Settings


@pytest.mark.asyncio
async def test_filtered_block_cannot_take_away_surviving_verified_source(monkeypatch):
    a, b = str(uuid4()), str(uuid4())

    async def reviewer(units, *_, **_budget_kwargs):
        return {
            "status": "complete",
            "units": [
                {
                    **units[0],
                    "verdict": "supported",
                    "evidence": [{"chunk_id": a, "quote": "Alpha"}],
                },
                {
                    **units[1],
                    "verdict": "unsupported",
                    "evidence": [{"chunk_id": b, "quote": "Beta"}],
                },
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    answer, info = await answer_guard.guard_answer(
        "Q",
        f"Alpha.\n\nUnsupported elaboration [{b}].",
        [{"id": a, "content": "Alpha"}, {"id": b, "content": "Beta"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    assert answer == f"Alpha.\n\n[{a}]"
    assert b not in answer and info["status"] == "filtered"
    assert info["citation_binding"]["added"][0]["source_ids"] == [a]


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing", "duplicate", "unknown", "forged_quote"])
async def test_binding_never_rescues_unverified_atomic_block(monkeypatch, problem):
    key = str(uuid4())

    async def reviewer(units, *_, **_budget_kwargs):
        row = {
            **units[0],
            "verdict": "supported",
            "evidence": [{"chunk_id": key, "quote": "Alpha"}],
        }
        if problem == "unknown":
            row["verdict"] = "unknown"
        if problem == "forged_quote":
            row["evidence"][0]["quote"] = "FORGED"
        return {
            "status": "complete",
            "units": []
            if problem == "missing"
            else [row, row]
            if problem == "duplicate"
            else [row],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    answer, info = await answer_guard.guard_answer(
        "Q",
        "Alpha.",
        [{"id": key, "content": "Alpha"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    assert answer == "" and info["citation_binding"]["added"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "draft", ["```mermaid\ngraph LR\nA-->B\n```", "| A | B |\n|---|---|\n|1|2|"]
)
async def test_binding_keeps_diagram_and_table_body_intact(monkeypatch, draft):
    key = str(uuid4())

    async def reviewer(units, *_, **_budget_kwargs):
        return {
            "status": "complete",
            "units": [
                {**u, "verdict": "supported", "evidence": [{"chunk_id": key, "quote": "Alpha"}]}
                for u in units
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    answer, _ = await answer_guard.guard_answer(
        "Q",
        draft,
        [{"id": key, "content": "Alpha"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    assert answer == draft + f"\n\n[{key}]"


@pytest.mark.asyncio
async def test_existing_forged_marker_is_preserved_for_api_rejection(monkeypatch):
    allowed, forged = str(uuid4()), str(uuid4())

    async def reviewer(units, *_, **_budget_kwargs):
        return {
            "status": "complete",
            "units": [
                {**u, "verdict": "supported", "evidence": [{"chunk_id": allowed, "quote": "Alpha"}]}
                for u in units
            ],
        }

    monkeypatch.setattr(answer_guard, "review_units", reviewer)
    draft = f"Alpha [{forged}]."
    answer, info = await answer_guard.guard_answer(
        "Q",
        draft,
        [{"id": allowed, "content": "Alpha"}],
        Settings(_env_file=None, answer_repair_enabled=False),
    )
    assert answer == draft and info["citation_binding"]["added"] == []
