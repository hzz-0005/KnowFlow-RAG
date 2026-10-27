"""Provider-free replay of checksummed parser/diagnostic inputs. Never writes indexes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from knowflow.app.config import Settings
from knowflow.app.document_processing import ParsedSection
from knowflow.app.ingestion_diagnostics import diagnose
from knowflow.app.structure_chunking import structure_chunks
from knowflow.app.structured_parsing import Block, ParsedDocument
from knowflow.app.token_budget import get_token_counter

MAX_INPUT_BYTES = 8_000_000
MAX_EXPANDED_BYTES = 24_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_checked(base, row, *, compressed=True):
    path = (base / row["path"]).resolve()
    if not path.is_relative_to(base.resolve()) or path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("Input outside manifest directory or above byte budget")
    raw = path.read_bytes()
    if digest(raw) != row["sha256"]:
        raise ValueError("Input SHA-256 mismatch")
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(MAX_EXPANDED_BYTES + 1)
        if len(raw) > MAX_EXPANDED_BYTES:
            raise ValueError("Expanded input exceeds byte budget")
    return json.loads(raw.decode("utf-8-sig"))


def compare(snapshot, parsed_data):
    if snapshot.get("schema") != "knowflow.diagnostic-snapshot.v1":
        raise ValueError("Unsupported diagnostic snapshot")
    parsed_blocks = {b["block_key"]: b for b in parsed_data["blocks"]}
    for block in snapshot["blocks"]:
        source = parsed_blocks.get(block["block_key"])
        loaded = snapshot["block_texts"].get(block["block_key"])
        if (
            source is None
            or len(source["text"]) != block["text_length"]
            or (loaded is not None and not source["text"].startswith(loaded))
        ):
            raise ValueError("Cached parser text differs from frozen index projections")
    saved = snapshot["document"]["parsing_report"]["chunking"]
    settings = Settings(
        _env_file=None,
        chunking_strategy="structure_tokens",
        chunk_max_tokens=saved["max_tokens"],
        chunk_token_reserve=saved["token_reserve"],
        chunk_target_tokens=saved["target_tokens"],
        chunk_overlap_tokens=saved["overlap_tokens"],
    )
    counter = get_token_counter(settings.chunk_tokenizer_path)
    if counter.metadata["sha256"] != saved["tokenizer"]["sha256"]:
        raise ValueError("Saved tokenizer fingerprint unavailable; comparison not performed")
    args = (snapshot["document"], snapshot["blocks"], snapshot["chunks"], snapshot["totals"])
    before = diagnose(
        *args, counter=counter, limits=snapshot["limits"], block_texts=snapshot["block_texts"]
    )
    if before["snapshot_sha256"] != snapshot["expected_snapshot_sha256"]:
        raise ValueError("Frozen projections do not reproduce the captured live fingerprint")
    parsed = ParsedDocument(
        [ParsedSection(**part) for part in parsed_data["sections"]],
        [Block(**block) for block in parsed_data["blocks"]],
        deepcopy(parsed_data["report"]),
    )
    rebuilt = structure_chunks(parsed, settings)
    stored = snapshot["chunks"]
    if len(rebuilt) != len(stored) or any(
        a.ordinal != b["ordinal"] or a.content != b["content"] or a.section != b["section"]
        for a, b in zip(rebuilt, stored, strict=True)
    ):
        raise ValueError(
            "Chunk content/section/boundaries changed; provenance-only comparison refused"
        )
    after_snapshot = deepcopy(snapshot)
    after_snapshot["document"]["parsing_report"]["chunking"] = parsed.report["chunking"]
    for chunk, rebuilt_chunk in zip(after_snapshot["chunks"], rebuilt, strict=True):
        chunk["source_spans"] = rebuilt_chunk.source_spans
    after = diagnose(
        after_snapshot["document"],
        after_snapshot["blocks"],
        after_snapshot["chunks"],
        after_snapshot["totals"],
        counter=counter,
        limits=snapshot["limits"],
        block_texts=snapshot["block_texts"],
    )
    after_snapshot["expected_snapshot_sha256"] = after["snapshot_sha256"]
    after_snapshot["live_report"] = None
    after_snapshot["scope"] = "offline provenance-only rebuild, not a live ingestion result"
    row = {
        "filename": snapshot["document"]["filename"],
        "source_sha256": snapshot["document"]["source_sha256"],
        "status": "observed",
        "body_and_embedding_input_unchanged": True,
        "chunks": len(stored),
        "before": before,
        "after": after,
        "source_span_records_changed": sum(
            a["source_spans"] != b.source_spans for a, b in zip(stored, rebuilt, strict=True)
        ),
        "not_measured": [
            "OCR accuracy",
            "reading order",
            "formula accuracy",
            "table structure",
            "semantic support",
            "retrieval quality",
            "false positive rate",
        ],
    }
    return row, after_snapshot


def run(manifest_path, output):
    manifest_path, output = Path(manifest_path), Path(output)
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest.decode("utf-8-sig"))
    if manifest.get("schema") != "knowflow.natural-diagnostics.inputs.v1":
        raise ValueError("Unsupported input manifest")
    base = manifest_path.parent
    for key in ("protocol", "classification"):
        load_checked(base, manifest[key], compressed=False)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, source in enumerate(manifest["documents"]):
        try:
            snapshot = load_checked(base, source["snapshot"])
            parsed = load_checked(base, source["parsed"])
            if snapshot["document"]["source_sha256"] != source["source_sha256"]:
                raise ValueError("Source identity mismatch")
            row, after = compare(snapshot, parsed)
            row["origin_kind"] = source["origin_kind"]
            target = output / f"{index:02d}-after.snapshot.json.gz"
            target.write_bytes(
                gzip.compress(json.dumps(after, ensure_ascii=True).encode(), mtime=0)
            )
            row["after_snapshot"] = {"path": target.name, "sha256": digest(target.read_bytes())}
        except (KeyError, ValueError, OSError, TypeError, EOFError) as exc:
            row = {"filename": source.get("filename"), "status": "unknown", "reason": str(exc)}
        rows.append(row)
    report = {
        "schema": "knowflow.diagnostic-replay.v1",
        "completed_at": datetime.now(UTC).isoformat(),
        "manifest_sha256": digest(raw_manifest),
        "documents": rows,
        "provider_calls": 0,
        "index_writes": 0,
        "semantic_task_success": None,
        "observed_documents": sum(r["status"] == "observed" for r in rows),
        "unknown_documents": sum(r["status"] == "unknown" for r in rows),
        "code_sha256": {
            name: digest(Path(__file__).with_name(name).read_bytes())
            for name in (
                "structure_chunking.py",
                "table_provenance.py",
                "ingestion_diagnostics.py",
                "source_consistency.py",
                "diagnostic_replay.py",
            )
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    lines = [
        "# 来源映射离线对照",
        "",
        "这是固定开发样本的机械观测，不是语义成功率或自然文档准确率。",
        "",
        "| 文件 | 来源分类 | 切片数 | 可比对前→后 | 无法比较前→后 | 状态 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] != "observed":
            lines.append(f"| {row['filename']} | 未知 | — | — | — | unknown |")
            continue
        a = row["before"]["observations"]["source_consistency"]["spans"]
        b = row["after"]["observations"]["source_consistency"]["spans"]
        lines.append(
            f"| {row['filename']} | {row['origin_kind']} | {row['chunks']} "
            f"| {a['compared']}→{b['compared']} "
            f"| {a['not_comparable']}→{b['not_comparable']} | observed |"
        )
    lines += [
        "",
        "未覆盖的范围、原页 OCR/阅读顺序/表格/公式均未因此验证；没有独立标注就不计算误报率。",
        "没有调用模型，没有写回数据库；任何正文或章节前缀变化都会拒绝来源映射专用对照。",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run(args.manifest, args.output)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "observed_documents",
                    "unknown_documents",
                    "provider_calls",
                    "index_writes",
                )
            }
        )
    )
    return bool(report["unknown_documents"])


if __name__ == "__main__":
    raise SystemExit(main())
