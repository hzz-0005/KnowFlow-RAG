"""Offline review snapshots and literal annotation checks; never score answer semantics."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

VERSION = "conversation-review-bundle-v1"
MAX_TURNS = 100
MAX_BYTES = 2 * 1024 * 1024
DIMENSIONS = (
    "referent_resolution",
    "scope_and_corrections",
    "topic_boundary",
    "task_completeness",
    "source_support",
    "abstention_appropriateness",
)
VERDICTS = {"unreviewed", "satisfied", "violated", "unknown", "not_applicable"}


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_json(path):
    path = Path(path)
    if path.stat().st_size > 8 * MAX_BYTES:
        raise ValueError("Input exceeds 16MiB; not read")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON value: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def template(snapshot):
    if not snapshot.get("turns"):
        raise ValueError("Empty history")
    target = snapshot["turns"][-1]
    return {
        "snapshot_sha256": digest(snapshot),
        "target_turn_id": target["id"],
        "declared_method": "unknown",
        "reviewer_id": None,
        "reviewed_at": None,
        "criteria": [
            {
                "dimension": dimension,
                "verdict": "unreviewed",
                "reason": "",
                "request_ref": None,
                "answer_quote": "",
                "citation_refs": [],
            }
            for dimension in DIMENSIONS
        ],
    }


def make_bundle(snapshot, exported_at):
    bundle = {
        "schema_version": VERSION,
        "exported_at": exported_at,
        "snapshot": snapshot,
        "snapshot_sha256": digest(snapshot),
        "review_template": template(snapshot),
        "limits": {"max_turns": MAX_TURNS, "max_canonical_bytes": MAX_BYTES},
        "notice": "Developer-observed export, not blind gold. Saved citation excerpts only; "
        "source completeness and task success unknown. Model judgments omitted. "
        "SHA256 checks integrity, not authenticity or independent human review.",
    }
    if len(canonical(bundle)) > MAX_BYTES:
        raise ValueError("Review export exceeds 2MiB; no partial bundle emitted")
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema_version") != VERSION:
        raise ValueError("Unsupported review bundle schema")
    if len(canonical(bundle)) > MAX_BYTES:
        raise ValueError("Review export exceeds 2MiB")
    snapshot = bundle["snapshot"]
    if digest(snapshot) != bundle["snapshot_sha256"]:
        raise ValueError("Snapshot fingerprint mismatch")
    turns = snapshot["turns"]
    if not turns or len(turns) > MAX_TURNS:
        raise ValueError("Empty or oversized history")
    sequences = [r["sequence"] for r in turns]
    if any(type(n) is not int for n in sequences) or sequences != list(range(1, len(turns) + 1)):
        raise ValueError("Incomplete or unordered history; no silent truncation")
    if len({r["id"] for r in turns}) != len(turns):
        raise ValueError("Duplicate turn identity")
    target = turns[-1]
    if (
        target["id"] != snapshot["target_turn_id"]
        or target["sequence"] != snapshot["through_sequence"]
    ):
        raise ValueError("Target does not match history boundary")
    for row in turns:
        if row["turn_sha256"] != digest({k: v for k, v in row.items() if k != "turn_sha256"}):
            raise ValueError("Turn fingerprint mismatch")
        if (
            not isinstance(row["question"], str)
            or row["answer"] is not None
            and not isinstance(row["answer"], str)
        ):
            raise ValueError("Invalid saved user/answer text")
        if row["task_inputs"]["requirement_sources"].get("current") != row["question"]:
            raise ValueError("Current input differs from original question")
        if not all(isinstance(v, str) for v in row["task_inputs"]["requirement_sources"].values()):
            raise ValueError("Invalid requirement source")
        citations = row["citations"]
        if [c["id"] for c in citations] != [
            f"{row['id']}:c{i}" for i in range(1, len(citations) + 1)
        ]:
            raise ValueError("Invalid citation identity")
        if any(not isinstance(c["quote"], str) for c in citations):
            raise ValueError("Invalid citation excerpt")
    return target


def validate_annotations(bundle, annotations):
    target = validate_bundle(bundle)
    expected = {
        "snapshot_sha256",
        "target_turn_id",
        "declared_method",
        "reviewer_id",
        "reviewed_at",
        "criteria",
    }
    if not isinstance(annotations, dict) or set(annotations) != expected:
        raise ValueError("Unexpected annotation fields")
    if (
        annotations["snapshot_sha256"] != bundle["snapshot_sha256"]
        or annotations["target_turn_id"] != target["id"]
    ):
        raise ValueError("Annotations belong to another snapshot or turn")
    method = annotations["declared_method"]
    if method not in {"unknown", "human", "model", "model_assisted_human"}:
        raise ValueError("Invalid declared annotation method")
    criteria = annotations["criteria"]
    if (
        not isinstance(criteria, list)
        or len(criteria) != len(DIMENSIONS)
        or {r["dimension"] for r in criteria} != set(DIMENSIONS)
    ):
        raise ValueError("Criteria must contain each declared dimension exactly once")
    any_reviewed = any(row["verdict"] != "unreviewed" for row in criteria)
    if any_reviewed:
        if (
            method == "unknown"
            or not isinstance(annotations["reviewer_id"], str)
            or not annotations["reviewer_id"].strip()
        ):
            raise ValueError("Reviewed criteria need declared method and reviewer identity")
        when = datetime.fromisoformat(annotations["reviewed_at"])
        if when.utcoffset() is None:
            raise ValueError("Review timestamp needs timezone")
    sources = {
        **target["task_inputs"]["requirement_sources"],
        **{f"turn:{r['id']}": r["question"] for r in bundle["snapshot"]["turns"]},
    }
    citations = {c["id"]: c["quote"] for c in target["citations"]}
    rows = []
    for row in criteria:
        if set(row) != {
            "dimension",
            "verdict",
            "reason",
            "request_ref",
            "answer_quote",
            "citation_refs",
        }:
            raise ValueError("Unexpected criterion fields")
        verdict = row["verdict"]
        if verdict not in VERDICTS or not isinstance(row["reason"], str):
            raise ValueError("Invalid criterion verdict/reason")
        if verdict != "unreviewed" and not row["reason"].strip():
            raise ValueError("Reviewed criterion needs a reason, including unknown/not_applicable")
        ref = row["request_ref"]
        if ref is not None:
            if (
                set(ref) != {"source_id", "quote"}
                or ref["source_id"] not in sources
                or not isinstance(ref["quote"], str)
                or not ref["quote"].strip()
                or ref["quote"] not in sources[ref["source_id"]]
            ):
                raise ValueError("Invalid original request excerpt")
        quote = row["answer_quote"]
        if not isinstance(quote, str) or quote and quote not in (target["answer"] or ""):
            raise ValueError("Invalid final answer excerpt")
        refs = row["citation_refs"]
        if not isinstance(refs, list) or len(refs) > len(citations):
            raise ValueError("Invalid citation references")
        seen = set()
        for ref_c in refs:
            if (
                set(ref_c) != {"citation_id", "quote"}
                or ref_c["citation_id"] not in citations
                or ref_c["citation_id"] in seen
                or not isinstance(ref_c["quote"], str)
                or not ref_c["quote"].strip()
                or ref_c["quote"] not in citations[ref_c["citation_id"]]
            ):
                raise ValueError("Invalid saved citation excerpt")
            seen.add(ref_c["citation_id"])
        if verdict in {"satisfied", "violated"} and (ref is None or not quote.strip()):
            raise ValueError("Semantic claims require request and answer witnesses")
        if (
            row["dimension"] == "source_support"
            and verdict in {"satisfied", "violated"}
            and not refs
        ):
            raise ValueError("Source support claims need saved citation witnesses")
        rows.append(
            {
                **row,
                "observation": "unknown"
                if verdict in {"unreviewed", "unknown"}
                else "declared_annotation_only",
            }
        )
    return {
        "schema_version": VERSION,
        "snapshot_sha256": bundle["snapshot_sha256"],
        "target_turn_id": target["id"],
        "history_turns": len(bundle["snapshot"]["turns"]),
        "declared_method": method,
        "reviewer_id": annotations["reviewer_id"],
        "annotation_status": "pending"
        if not any_reviewed
        else "declared_not_independently_verified",
        "unreviewed_dimensions": [r["dimension"] for r in criteria if r["verdict"] == "unreviewed"],
        "independent_review_status": "not_established",
        "task_success": None,
        "source_coverage": "saved_citations_only_unknown_completeness",
        "criteria": rows,
        "note": (
            "Literal checks do not establish entailment, coverage, authenticity, "
            "independence or benchmark readiness."
        ),
    }
