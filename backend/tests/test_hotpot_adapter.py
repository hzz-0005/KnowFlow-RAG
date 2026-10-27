import copy
import hashlib
import json
import runpy
from pathlib import Path

import pytest

ADAPTER = runpy.run_path(str(Path(__file__).resolve().parents[2] / "evaluation/prepare_hotpot.py"))


def entry(key, kind):
    return {
        "row_idx": key,
        "truncated_cells": [],
        "row": {
            "id": str(key),
            "type": kind,
            "level": "medium",
            "question": "Original question?",
            "answer": "Original answer",
            "context": {
                "title": [f"Target {key}", f"Distractor {key}"],
                "sentences": [
                    ["First sentence. ", "Supporting sentence."],
                    ["Unrelated sentence."],
                ],
            },
            "supporting_facts": {"title": [f"Target {key}"], "sent_id": [1]},
        },
    }


def test_selection_is_independent_of_input_order_and_answers():
    rows = [entry(i, "bridge" if i % 2 else "comparison") for i in range(20)]

    def ids(selected):
        return [e["row"]["id"] for e in selected]

    selected = ids(ADAPTER["select_rows"]({"rows": rows}))
    changed = copy.deepcopy(rows[::-1])
    for item in changed:
        item["row"]["answer"] = "changed without changing selection"
    assert ids(ADAPTER["select_rows"]({"rows": changed})) == selected
    assert len(selected) == 4


def test_conversion_keeps_distractors_and_exact_official_labels_without_answer_leakage():
    item = entry(1, "bridge")
    docs, _, cases = ADAPTER["convert_rows"]([item])
    assert len(docs) == 2
    assert any("Unrelated sentence." in text for text in docs.values())
    assert all(
        "Original answer" not in text and "Original question?" not in text for text in docs.values()
    )
    assert cases[0]["question"] == item["row"]["question"]
    assert cases[0]["reference_answer"] == item["row"]["answer"]
    fact = cases[0]["relevant_evidence"][0]
    assert fact["contains"] == "Supporting sentence."
    assert fact["contains"] in docs[fact["filename"]]


def test_invalid_official_sentence_index_is_rejected_without_replacing_question():
    item = entry(1, "bridge")
    item["row"]["supporting_facts"]["sent_id"] = [100]
    with pytest.raises(ValueError):
        ADAPTER["convert_rows"]([item])


def test_preparation_cannot_overwrite_a_frozen_trial(tmp_path):
    source = tmp_path / "rows.json"
    source.write_text(
        json.dumps({"rows": [entry(i, "bridge" if i % 2 else "comparison") for i in range(100)]}),
        encoding="utf-8",
    )
    target = tmp_path / "trial"
    ADAPTER["prepare"](source, target)
    before = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        ADAPTER["prepare"](source, target)
    after = {
        str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in target.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_preparation_rejects_a_changed_window_before_creating_artifacts(tmp_path):
    source = tmp_path / "rows.json"
    source.write_text(
        json.dumps({"rows": [entry(1, "bridge"), entry(2, "comparison")]}), encoding="utf-8"
    )
    target = tmp_path / "trial"
    with pytest.raises(ValueError, match="first 100"):
        ADAPTER["prepare"](source, target)
    assert not target.exists()
