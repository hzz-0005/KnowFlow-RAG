"""Mechanical dataset checks must not turn draft annotation into semantic gold."""

import json
import socket
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from knowflow.app.evaluation_replay.common import ReplayError, digest, read_json
from m2_synthetic_fixture import build_control

from evaluation.m2_v1.validate_dataset import jsonl, validate

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "evaluation/m2_v1"


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def rows_write(path, rows):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows), encoding="utf-8"
    )


@pytest.fixture
def bundle(tmp_path):
    return build_control(tmp_path)


def test_draft_integrity_does_not_release_baseline(bundle):
    root, base = bundle
    result = validate(root, base)
    assert result["mechanical_validation"] == "passed"
    assert not result["baseline_ready"] and result["semantic_task_success"] is None
    assert "independent_task_review_pending" in result["release_blockers"]
    assert result["task_count"] == 7 and result["equivalence_group_count"] == 6


@pytest.mark.parametrize(
    "change",
    [
        "quote",
        "page",
        "unknown_reference",
        "negative_sign",
        "currency",
        "equivalence",
        "unanswerable",
    ],
)
def test_bad_annotation_is_rejected(bundle, change):
    root, base = bundle
    evidence = jsonl(base / "evidence-draft.jsonl")
    cases = jsonl(base / "tasks-draft.jsonl")
    if change == "quote":
        evidence[0]["quote"] += "invented"
    elif change == "page":
        evidence[0]["physical_page"] = 9999
    elif change == "unknown_reference":
        cases[0]["required_claims"][0]["evidence_alternatives"][0]["all_of"].append("missing")
    elif change == "negative_sign":
        row = next(c for c in cases if c["id"] == "control-roe-change")
        row["numeric_assertions"][-1]["value"] = "1.73"
    elif change == "currency":
        row = next(c for c in cases if c["id"] == "control-profit-difference")
        row["numeric_assertions"][0]["currency"] = "USD"
    elif change == "equivalence":
        cases[1]["numeric_assertions"][0]["value"] = "337489"
    else:
        cases[0]["answerability"] = "unanswerable"
        cases[0]["required_claims"] = []
        cases[0]["expected_answer"] = None
    rows_write(base / "evidence-draft.jsonl", evidence)
    rows_write(base / "tasks-draft.jsonl", cases)
    with pytest.raises(ReplayError):
        validate(root, base)


def test_duplicate_bytes_cannot_become_a_new_holdout_group(bundle):
    root, base = bundle
    corpus = read_json(base / "corpus-manifest.json")
    copy = deepcopy(corpus["documents"][0])
    copy.update(
        document_id="different-name", group_id="different-group", split="reserved_candidate"
    )
    corpus["documents"].append(copy)
    write(base / "corpus-manifest.json", corpus)
    protocol = read_json(base / "baseline-protocol.json")
    protocol["corpus_manifest_sha256"] = digest(base / "corpus-manifest.json")
    write(base / "baseline-protocol.json", protocol)
    with pytest.raises(ReplayError, match="bytes cross splits"):
        validate(root, base)


def test_source_hash_drift_is_rejected(bundle):
    root, base = bundle
    doc = read_json(base / "corpus-manifest.json")["documents"][0]
    with (root / doc["path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ReplayError, match="SHA256 mismatch"):
        validate(root, base)


def test_snapshots_cannot_replace_original_page_text(bundle):
    root, base = bundle
    evidence = jsonl(base / "evidence-draft.jsonl")
    path = root / evidence[0]["text_snapshot"]
    path.write_text(path.read_text(encoding="utf-8") + "invented", encoding="utf-8")
    for e in evidence:
        if e["text_snapshot"] == evidence[0]["text_snapshot"]:
            e["text_snapshot_sha256"] = digest(path)
    rows_write(base / "evidence-draft.jsonl", evidence)
    with pytest.raises(ReplayError, match="not from original page"):
        validate(root, base)


def test_reserved_content_is_not_inspected(bundle):
    root, base = bundle
    corpus = read_json(base / "corpus-manifest.json")
    corpus["documents"][0].update(split="reserved_candidate", content_inspection_allowed=False)
    splits = read_json(base / "split-assignment.json")
    splits["assignments"][0].update(split="reserved_candidate", content_inspection_allowed=False)
    write(base / "split-assignment.json", splits)
    corpus["split_manifest_sha256"] = digest(base / "split-assignment.json")
    write(base / "corpus-manifest.json", corpus)
    protocol = read_json(base / "baseline-protocol.json")
    protocol["corpus_manifest_sha256"] = digest(base / "corpus-manifest.json")
    write(base / "baseline-protocol.json", protocol)
    with pytest.raises(ReplayError, match="Reserved content"):
        validate(root, base)


def test_duplicate_and_nonfinite_jsonl_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    for text in ['{"id":"x","id":"y"}', '{"id":"x","score":NaN}']:
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ReplayError):
            jsonl(path)


def test_offline_and_release_exit_code(bundle, monkeypatch):
    root, base = bundle

    def denied(*args, **kwargs):
        raise AssertionError("Preparation must stay offline")

    monkeypatch.setattr(socket.socket, "connect", denied)
    assert not validate(root, base)["baseline_ready"]
    argv = [
        sys.executable,
        "-X",
        "utf8",
        str(SOURCE / "validate_dataset.py"),
        "--root",
        str(root),
        "--base",
        str(base),
    ]
    normal = subprocess.run([*argv, "--out", str(root / "normal.json")], capture_output=True)
    assert normal.returncode == 0, normal.stderr
    strict = subprocess.run(
        [*argv, "--out", str(root / "strict.json"), "--require-ready"], capture_output=True
    )
    assert strict.returncode == 3
    duplicate = subprocess.run([*argv, "--out", str(root / "normal.json")], capture_output=True)
    assert duplicate.returncode == 2


def test_real_preparation_keeps_reserved_candidate_unannotated():
    if (
        not (SOURCE / "corpus-manifest.json").is_file()
        and (ROOT / "SOURCE_MANIFEST.json").is_file()
    ):
        pytest.skip(
            "Private real-corpus archive is intentionally not distributed; synthetic gates run"
        )
    result = validate(ROOT, SOURCE)
    assert result["documents_by_split"] == {
        "development": 3,
        "reserved_candidate": 1,
        "regression": 3,
        "control": 2,
    }
    assert result["task_count"] == 15 and result["equivalence_group_count"] == 13
    assert result["evidence_count"] == 25 and not result["baseline_ready"]


@pytest.mark.parametrize("kind", ["evidence", "task", "m1_acceptance"])
def test_review_flag_alone_cannot_establish_acceptance(bundle, kind):
    root, base = bundle
    if kind == "m1_acceptance":
        data = read_json(base / "baseline-protocol.json")
        data.update(m1_external_review_status="accepted", m1_external_review_evidence="yes")
        write(base / "baseline-protocol.json", data)
    else:
        path = base / ("evidence-draft.jsonl" if kind == "evidence" else "tasks-draft.jsonl")
        rows = jsonl(path)
        field = "review_status" if kind == "evidence" else "annotation_status"
        rows[0].update({field: "independently_reviewed", "review_record": "yes"})
        rows_write(path, rows)
    with pytest.raises(ReplayError, match="pinned review file"):
        validate(root, base)
