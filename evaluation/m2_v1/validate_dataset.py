"""Validate M2 preparation without running models or changing a knowledge base."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from knowflow.app.evaluation_replay.common import ReplayError, digest, identity, inside, read_json
from pypdf import PdfReader


def jsonl(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ReplayError(f"Duplicate field {key}: {path.name}")
            result[key] = value
        return result

    rows = [
        json.loads(
            line,
            object_pairs_hook=unique,
            parse_constant=lambda v: require(False, f"Nonfinite JSON: {v}"),
        )
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(
        not isinstance(r, dict) or not isinstance(r.get("id"), str) or not r["id"] for r in rows
    ):
        raise ReplayError(f"Invalid row identity: {path.name}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ReplayError(f"Duplicate row identity: {path.name}")
    return rows


def require(condition, message):
    if not condition:
        raise ReplayError(message)


def decimal(value):
    require(isinstance(value, str), "Annotation numbers must be explicit decimal strings")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ReplayError("Invalid annotated number") from exc
    require(result.is_finite(), "Nonfinite annotated number")
    return result


def validate(root: Path, base: Path):
    root, base = root.resolve(), base.resolve()
    require(base.is_relative_to(root), "Preparation directory escapes project root")
    inputs = {}

    def source(relative, expected=None):
        path = inside(root, relative)
        actual = digest(path)
        require(expected is None or expected == actual, f"SHA256 mismatch: {relative}")
        inputs[relative] = actual
        return path

    def local(name):
        return source((base / name).relative_to(root).as_posix())

    def reviewed(row, kind):
        ref = row.get("review_record")
        require(isinstance(ref, dict), "Reviewed annotation needs a pinned review file")
        review = read_json(source(ref["path"], ref["sha256"]))
        require(
            review.get("schema") == "knowflow.m2.annotation_review.v1"
            and review.get("reviewer_type") == "human"
            and bool(review.get("reviewer")),
            "Model estimate or unidentified reviewer is not independent semantic gold",
        )
        payload = {
            k: v
            for k, v in row.items()
            if k not in {"review_record", "review_status", "annotation_status"}
        }
        require(
            review.get("target_kind") == kind
            and review.get("target_id") == row["id"]
            and review.get("target_sha256") == identity(payload)
            and review.get("decision") == "accepted",
            "Review target changed or unaccepted",
        )

    corpus = read_json(local("corpus-manifest.json"))
    splits = read_json(local("split-assignment.json"))
    protocol = read_json(local("baseline-protocol.json"))
    require(corpus.get("schema") == "knowflow.m2.corpus.v1", "Unknown corpus schema")
    require(splits.get("schema") == "knowflow.m2.split.v1", "Unknown split schema")
    require(protocol.get("schema") == "knowflow.m2.protocol.v1", "Unknown protocol schema")
    require(
        corpus["split_manifest_sha256"] == digest(base / "split-assignment.json"),
        "Split assignment changed",
    )
    require(
        protocol["corpus_manifest_sha256"] == digest(base / "corpus-manifest.json"),
        "Baseline protocol refers to a different corpus revision",
    )
    documents, groups, readers, hash_splits = {}, {}, {}, {}
    for doc in corpus["documents"]:
        did = doc["document_id"]
        require(did not in documents, "Duplicate document identity")
        require(
            doc["split"] in {"development", "regression", "control", "reserved_candidate"},
            "Unknown document split",
        )
        require(doc["download_status"] == "completed", f"Unavailable document: {did}")
        source(doc["path"], doc["sha256"])
        require(
            doc["sha256"] not in hash_splits or hash_splits[doc["sha256"]] == doc["split"],
            "Same document bytes cross splits",
        )
        hash_splits[doc["sha256"]] = doc["split"]
        group = doc["group_id"]
        require(
            group not in groups or groups[group] == doc["split"], "Document family crosses splits"
        )
        groups[group] = doc["split"]
        documents[did] = doc
    for assignment in splits["assignments"]:
        doc = documents[assignment["document_id"]]
        require(
            all(
                doc[k] == assignment[k] for k in ("group_id", "split", "content_inspection_allowed")
            ),
            "Split assignment drift",
        )
    evidence = {e["id"]: e for e in jsonl(local("evidence-draft.jsonl"))}
    for eid, e in evidence.items():
        doc = documents[e["document_id"]]
        require(
            doc["content_inspection_allowed"] and doc["split"] != "reserved_candidate",
            f"Reserved content entered annotation: {eid}",
        )
        require(e["document_sha256"] == doc["sha256"], f"Evidence PDF mismatch: {eid}")
        require(doc["format"] == "pdf", "Current evidence adapter supports PDF physical pages")
        page = e["physical_page"]
        require(type(page) is int and 1 <= page <= doc["physical_pages"], "Invalid physical page")
        snapshot = source(e["text_snapshot"], e["text_snapshot_sha256"])
        text = snapshot.read_text(encoding="utf-8")
        start, end = e["char_start"], e["char_end"]
        require(
            type(start) is int and type(end) is int and 0 <= start < end <= len(text),
            f"Invalid text range: {eid}",
        )
        require(text[start:end] == e["quote"], f"Evidence quote mismatch: {eid}")
        if doc["document_id"] not in readers:
            readers[doc["document_id"]] = PdfReader(root / doc["path"])
        reader = readers[doc["document_id"]]
        require(len(reader.pages) == doc["physical_pages"], "PDF page count changed")
        require(
            reader.pages[page - 1].extract_text() == text, f"Snapshot not from original page: {eid}"
        )
        source(e["page_preview"], e["page_preview_sha256"])
        require(
            e["review_status"] in {"draft", "independently_reviewed"}, "Unknown evidence review"
        )
        if e["review_status"] == "independently_reviewed":
            reviewed(e, "evidence")

    cases = jsonl(local("tasks-draft.jsonl"))
    equivalents = {}
    for case in cases:
        cid = case["id"]
        require(case["split"] == "development", "Reserved tasks must not enter development")
        require(
            isinstance(case["question"], str) and bool(case["question"].strip()), "Empty question"
        )
        scope = case["scope_document_ids"]
        require(bool(scope) and len(set(scope)) == len(scope), "Invalid task scope")
        require(
            all(d in documents and documents[d]["split"] == case["split"] for d in scope),
            f"Task scope crosses splits: {cid}",
        )
        require(
            case["answerability"] in {"answerable", "unanswerable", "unknown"},
            "Unknown answerability",
        )
        require(case["annotation_status"] in {"draft", "independently_reviewed"}, "Unknown review")
        if case["annotation_status"] == "independently_reviewed":
            reviewed(case, "task")
        claims = case["required_claims"]
        require(len({c["id"] for c in claims}) == len(claims), "Duplicate claim identity")
        if case["answerability"] == "answerable":
            require(bool(claims) and bool(case["expected_answer"]), "Answerable task lacks claims")
        elif case["answerability"] == "unknown":
            require(
                case["expected_answer"] is None and bool(case.get("missing_reason")),
                "Unknown answerability cannot invent an expected answer",
            )
        else:
            require(
                bool(case.get("negative_scope_review")),
                "Empty evidence does not establish an unanswerable task",
            )
        for claim in claims:
            alternatives = claim["evidence_alternatives"]
            require(bool(alternatives), f"Claim lacks alternatives: {cid}")
            for alternative in alternatives:
                ids = alternative["all_of"]
                require(bool(ids) and len(set(ids)) == len(ids), "Invalid AND evidence group")
                require(
                    all(i in evidence and evidence[i]["document_id"] in scope for i in ids),
                    f"Unknown/out-of-scope evidence: {cid}",
                )
        numbers = {n["id"]: n for n in case["numeric_assertions"]}
        require(len(numbers) == len(case["numeric_assertions"]), "Duplicate numeric assertion")
        for n in numbers.values():
            require(
                all(k in n for k in ("entity", "metric", "period", "basis", "currency", "unit")),
                "Number lacks entity/period/basis/currency/unit",
            )
            decimal(n["value"])
        calculation = case.get("calculation")
        if calculation:
            require(
                calculation["operator"] == "subtract", "Unsupported calculation; never eval text"
            )
            operands = calculation["operand_ids"]
            require(len(operands) == 2 and all(i in numbers for i in operands), "Missing operands")
            a, b = [numbers[i] for i in operands]
            require(
                all(a[k] == b[k] for k in ("entity", "metric", "currency", "unit")),
                "Calculation mixes units, entities or bases",
            )
            if a["basis"] != b["basis"]:
                require(
                    calculation.get("allowed_basis_pair") == [a["basis"], b["basis"]]
                    and bool(calculation.get("basis_justification")),
                    "Different reporting bases require an explicit comparison definition",
                )
            result = numbers[calculation["result_id"]]
            require(
                all(result[k] == a[k] for k in ("entity", "currency")),
                "Calculation result changes entity/currency",
            )
            expected_unit = "percentage_points" if a["unit"] == "percent" else a["unit"]
            require(result["unit"] == expected_unit, "Incorrect difference unit")
            require(
                decimal(a["value"]) - decimal(b["value"]) == decimal(result["value"]),
                f"Arithmetic/sign mismatch: {cid}",
            )
        key = case["equivalence_group"]
        signature = identity(
            {
                k: case[k]
                for k in (
                    "scope_document_ids",
                    "answerability",
                    "required_claims",
                    "numeric_assertions",
                )
            }
        )
        require(
            key not in equivalents or equivalents[key] == signature, "Equivalent questions drift"
        )
        equivalents[key] = signature

    blockers = []
    if protocol["execution_enabled"] is not True:
        blockers.append("baseline_execution_not_released")
    if protocol.get("m1_external_review_status") != "accepted":
        blockers.append("m1_external_review_pending")
    else:
        ref = protocol.get("m1_external_review_evidence")
        require(isinstance(ref, dict), "M1 acceptance needs a pinned review file")
        record = read_json(source(ref["path"], ref["sha256"]))
        require(
            record.get("decision") == "accepted" and bool(record.get("reviewer")),
            "M1 review is not accepted",
        )
        audit = source("evaluation/results/unified/m1-review-20261023-v2/audit.json")
        require(
            record.get("target_audit_sha256") == digest(audit),
            "M1 review refers to another implementation",
        )
    if any(c["annotation_status"] != "independently_reviewed" for c in cases):
        blockers.append("independent_task_review_pending")
    if any(e["review_status"] != "independently_reviewed" for e in evidence.values()):
        blockers.append("independent_evidence_review_pending")
    if any(c["answerability"] == "unknown" for c in cases):
        blockers.append("answerability_review_pending")
    if not protocol.get("runtime_snapshot"):
        blockers.append("effective_runtime_budget_snapshot_missing")
    else:
        ref = protocol["runtime_snapshot"]
        require(isinstance(ref, dict), "Runtime snapshot needs a pinned file")
        snapshot = read_json(source(ref["path"], ref["sha256"]))
        require(snapshot.get("schema") == "knowflow.m2.runtime.v1", "Unknown runtime snapshot")
        require(
            all(
                snapshot.get(k) is not None
                for k in (
                    "candidate_budget",
                    "context_token_budget",
                    "models",
                    "config_sha256",
                    "prompt_sha256",
                    "code_sha256",
                    "index_sha256",
                )
            ),
            "Incomplete runtime snapshot",
        )
    for relative, expected in inputs.items():
        require(digest(root / relative) == expected, f"Input changed during validation: {relative}")
    return {
        "schema": "knowflow.m2.validation.v1",
        "mechanical_validation": "passed",
        "baseline_ready": not blockers,
        "release_blockers": blockers,
        "documents_by_split": dict(Counter(d["split"] for d in documents.values())),
        "task_count": len(cases),
        "equivalence_group_count": len(equivalents),
        "tasks_by_type": dict(Counter(c["question_type"] for c in cases)),
        "evidence_count": len(evidence),
        "semantic_task_success": None,
        "note": "Mechanical integrity and arithmetic only; not semantic approval or RAG quality.",
        "inputs": [{"path": p, "sha256": s} for p, s in inputs.items()],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    started = datetime.now(UTC).isoformat()
    try:
        require(not args.out.exists(), "Output already exists; use a new validation path")
        result = validate(args.root, args.base)
        code = 3 if args.require_ready and not result["baseline_ready"] else 0
        result.update(
            started_at=started,
            completed_at=datetime.now(UTC).isoformat(),
            argv=list(sys.orig_argv),
            cwd=str(Path.cwd()),
            exit_code=code,
            validator_sha256=digest(Path(__file__)),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ("mechanical_validation", "baseline_ready", "release_blockers")
                },
                ensure_ascii=True,
            )
        )
        return code
    except (ReplayError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"M2 invalid preparation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
