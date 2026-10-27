"""Native parser -> real structure/token packer -> controlled persisted-data faults.

Contracts:
  --contract v1 (default): persisted-integrity-v1 semantics, no source-text
    consistency claims. Reproduces the frozen v1 expectations byte-for-byte.
  --contract v2: passes block source text into the checker (persisted-integrity-v2 +
    source-consistency-v1) and evaluates diagnostics_v2/cases.json.
"""

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from knowflow.app.config import Settings
from knowflow.app.ingestion_diagnostics import diagnose
from knowflow.app.structure_chunking import structure_chunks
from knowflow.app.structured_parsing import native_parse
from knowflow.app.token_budget import get_token_counter

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "diagnostics_v2"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def apply_mutation(mutation, doc, blocks, chunks, block_texts, span):
    if mutation == "missing_source":
        chunks[0]["source_spans"] = []
    elif mutation == "dangling_block":
        span["block_key"] = "deliberately-missing"
    elif mutation == "chunk_range":
        span["chunk_range"] = [0, chunks[0]["text_length"] + 1]
    elif mutation == "text_range":
        span["text_range"] = [-1, 5]
    elif mutation == "invalid_region":
        span["bbox"] = {"left": 0.8, "top": 0, "right": 0.2, "bottom": 1}
    elif mutation == "over_budget":
        doc["parsing_report"]["chunking"]["max_tokens"] = 128
    elif mutation == "bad_cell":
        span["cell_range"] = "A0"
    elif mutation == "old_tokenizer":
        doc["parsing_report"]["chunking"]["tokenizer"]["sha256"] = "unmatched-version"
    elif mutation == "withheld_sources":
        chunks[0]["source_spans"] = None
    elif mutation == "partial_blocks":
        blocks.clear()
    elif mutation == "duplicate_context":
        chunks.append({**deepcopy(chunks[0]), "id": "c-duplicate", "ordinal": len(chunks)})
    elif mutation == "blank_page_projection":
        doc["filename"] = "synthetic-two-page-projection.pdf"
        doc["parsing_report"]["page_count"] = 2
        for chunk in chunks:
            for source in chunk["source_spans"]:
                source["page_number"] = 1
    elif mutation == "same_length_replacement":
        chunks[0]["content"] = "X" * chunks[0]["text_length"]
    elif mutation == "caption_layout_replacement":
        blocks[0]["kind"] = "caption"
        chunks[0]["content"] = "X" * chunks[0]["text_length"]
    elif mutation == "whitespace_rewrite":
        content = chunks[0]["content"]
        if " " in content:
            chunks[0]["content"] = content.replace(" ", "\n")
        elif "\n" in content:
            chunks[0]["content"] = content.replace("\n", " ")
        else:
            raise ValueError("whitespace_rewrite: control chunk has no whitespace to rewrite")
    elif mutation == "partial_coverage":
        span["chunk_range"] = [span["chunk_range"][0] + 1, span["chunk_range"][1] - 1]
        span["text_range"] = [span["text_range"][0] + 1, span["text_range"][1] - 1]
    elif mutation == "cross_page_spans":
        if len(chunks) < 2 or not chunks[1]["source_spans"]:
            raise ValueError("cross_page_spans: control needs two spans")
        doc["filename"] = "synthetic-two-page-projection.pdf"
        doc["parsing_report"]["page_count"] = 2
        chunks[0]["source_spans"][0]["page_number"] = 1
        chunks[1]["source_spans"][0]["page_number"] = 2
    elif mutation == "section_prefix":
        chunks[0]["section"] = "第九章 附加说明与跨页引用"
    elif mutation == "remove_chunk_range":
        del span["chunk_range"]
    elif mutation == "truncated_source_text":
        key = span["block_key"]
        block_texts[key] = block_texts.get(key, "")[:3]
    elif mutation == "unicode_form":
        target = None
        for chunk in chunks:
            for source in chunk["source_spans"]:
                key = source.get("block_key")
                bounds = source.get("text_range")
                if not bounds or key not in block_texts:
                    continue
                referenced = sum(
                    1
                    for other in chunks
                    for item in other["source_spans"] or []
                    if item.get("block_key") == key
                )
                slice_ = block_texts[key][bounds[0] : bounds[1]]
                if referenced == 1 and unicodedata.normalize("NFD", slice_) != slice_:
                    target = (key, source, slice_)
                    break
            if target:
                break
        if not target:
            raise ValueError("unicode_form: no singly-referenced decomposable slice")
        key, source, slice_ = target
        nfd = unicodedata.normalize("NFD", slice_)
        delta = len(nfd) - len(slice_)
        block_texts[key] = (
            block_texts[key][: source["text_range"][0]]
            + nfd
            + block_texts[key][source["text_range"][1] :]
        )
        source["text_range"][1] += delta
        for block in blocks:
            if block["block_key"] == key:
                block["text_length"] += delta
    elif mutation != "none":
        raise ValueError(f"Unsupported control: {mutation}")


def evaluate(cases_path=None, manifest_path=None, contract="v1"):
    manifest_path = Path(manifest_path) if manifest_path else HERE / "corpus-manifest.json"
    cases_path = Path(cases_path) if cases_path else HERE / "cases.json"
    manifest_raw = manifest_path.read_bytes()
    cases_raw = cases_path.read_bytes()
    cases = json.loads(cases_raw)["cases"]
    manifest = json.loads(manifest_raw)
    settings = Settings(
        _env_file=None,
        chunk_max_tokens=512,
        chunk_token_reserve=16,
        chunk_target_tokens=384,
        chunk_overlap_tokens=48,
        chunk_tokenizer_path=None,
    )
    counter = get_token_counter()
    bases, rows = {}, []
    for file in manifest["files"]:
        rel = file.get("path") or f"corpus/{file['filename']}"
        raw = (manifest_path.parent / rel).read_bytes()
        assert digest(raw) == file["sha256"], file["filename"]
        parsed = native_parse(file["filename"], raw, settings)
        chunks = structure_chunks(parsed, settings)
        bases[file["filename"]] = (
            {
                "id": file["filename"],
                "filename": file["filename"],
                "status": "processed",
                "source_sha256": file["sha256"],
                "parsing_report": parsed.report,
            },
            [
                {
                    "block_key": b.block_key,
                    "ordinal": i,
                    "kind": b.kind,
                    "text_length": len(b.text),
                }
                for i, b in enumerate(parsed.blocks)
            ],
            [
                {
                    "id": f"c-{i}",
                    "ordinal": i,
                    "content": c.content,
                    "text_length": len(c.content),
                    "section": c.section,
                    "source_spans": c.source_spans,
                }
                for i, c in enumerate(chunks)
            ],
            {b.block_key: b.text for b in parsed.blocks},
        )
    for case in cases:
        doc, blocks, chunks, block_texts = deepcopy(bases[case["source"]])
        totals = {"blocks": len(blocks), "chunks": len(chunks)}
        mutation = case["mutation"]
        span = chunks[0]["source_spans"][0]
        apply_mutation(mutation, doc, blocks, chunks, block_texts, span)
        if mutation == "duplicate_context":
            totals["chunks"] += 1
        elif mutation == "partial_blocks":
            totals["blocks"] = len(bases[case["source"]][1])
        result = diagnose(
            doc,
            blocks,
            chunks,
            totals,
            counter=counter,
            block_texts=block_texts if contract == "v2" else None,
        )
        found = {f["code"]: f for f in result["findings"]}
        violations = []
        if case.get("expect_code"):
            finding = found.get(case["expect_code"])
            if not finding or finding["severity"] != case["expect_severity"]:
                violations.append("expected_finding_not_observed")
        if set(case.get("forbid_codes", [])) & found.keys():
            violations.append("forbidden_finding_observed")
        if any(f["severity"] in case.get("forbid_severities", []) for f in found.values()):
            violations.append("forbidden_severity_observed")
        rows.append(
            {
                "id": case["id"],
                "source": case["source"],
                "role": case["role"],
                "mutation": mutation,
                "contract_matched": not violations,
                "violations": violations,
                "diagnostics": result,
                "note": case.get("note"),
            }
        )
    return {
        "schema": f"knowflow.integrity-controls-report.{'v2' if contract == 'v2' else 'v1'}",
        "contract": contract,
        "created_at": datetime.now(UTC).isoformat(),
        "cases_sha256": digest(cases_raw),
        "corpus_manifest_sha256": digest(manifest_raw),
        "tokenizer_sha256": counter.metadata["sha256"],
        "settings": {
            key: getattr(settings, key)
            for key in (
                "chunk_max_tokens",
                "chunk_token_reserve",
                "chunk_target_tokens",
                "chunk_overlap_tokens",
            )
        },
        "n_cases": len(rows),
        "roles": dict(Counter(row["role"] for row in rows)),
        "n_contract_mismatches": sum(not row["contract_matched"] for row in rows),
        "rows": rows,
        "note": (
            "合成软件控制，不是独立语义金标或自然脏文档检出率；"
            "unsupported_semantics 单列，不能作为质量通过。"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", choices=("v1", "v2"), default="v1")
    parser.add_argument("--cases", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    if args.contract == "v2" and (args.cases is None or args.manifest is None):
        args.cases = args.cases or V2 / "cases.json"
        args.manifest = args.manifest or V2 / "corpus-manifest.json"
    result = evaluate(cases_path=args.cases, manifest_path=args.manifest, contract=args.contract)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=True, indent=2)
    print(json.dumps({key: result[key] for key in ("n_cases", "roles", "n_contract_mismatches")}))
    return int(result["n_contract_mismatches"] != 0)


if __name__ == "__main__":
    sys.exit(main())
