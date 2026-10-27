"""Checksummed normalized parse outputs, reusable for automatic re-chunking."""

import gzip
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from knowflow.app.document_processing import ParsedSection
from knowflow.app.structured_parsing import Block, ParsedDocument

VERSION = 1
MAX_BYTES = 200 * 1024 * 1024


def artifact_path(storage_key, settings):
    if not storage_key or Path(storage_key).name != storage_key or storage_key in {".", ".."}:
        raise ValueError("Invalid storage key")
    return Path(settings.upload_dir) / ".parse-artifacts" / (storage_key + ".json.gz")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def fingerprint(filename, source_sha, settings):
    # Chunk options are intentionally absent: changing them never requires OCR.
    if Path(filename).suffix.lower() in {".csv", ".xlsx"}:
        from knowflow.app.tabular_parsing import VERSION as TABULAR_VERSION

        return _digest(
            {
                "schema": VERSION,
                "source_sha256": source_sha,
                "extension": Path(filename).suffix.lower(),
                "parser": "tabular",
                "tabular_revision": TABULAR_VERSION,
                "parser_revision": settings.parse_artifact_revision,
            }
        )

    return _digest(
        {
            "schema": VERSION,
            "source_sha256": source_sha,
            "extension": Path(filename).suffix.lower(),
            "parser": settings.pdf_parser,
            "parser_revision": settings.parse_artifact_revision,
            "endpoint": settings.docling_base_url if settings.pdf_parser == "docling" else None,
            "max_pages": settings.pdf_max_pages,
        }
    )


def save_artifact(storage_key, filename, source_sha, parsed, settings):
    payload = asdict(parsed)
    checksum = _digest(payload)
    key = fingerprint(filename, source_sha, settings)
    envelope = {"schema": VERSION, "fingerprint": key, "checksum": checksum, "parsed": payload}
    raw = json.dumps(envelope, ensure_ascii=False).encode()
    if len(raw) > MAX_BYTES:
        raise ValueError("解析产物超过缓存大小上限。")
    path = artifact_path(storage_key, settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(gzip.compress(raw, mtime=0))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"available": True, "schema_version": VERSION, "sha256": checksum, "fingerprint": key}


def load_artifact(storage_key, filename, source_sha, settings):
    try:
        with gzip.open(artifact_path(storage_key, settings), "rb") as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("解析产物超过大小上限。")
        envelope = json.loads(raw)
        if envelope["schema"] != VERSION or envelope["fingerprint"] != fingerprint(
            filename, source_sha, settings
        ):
            raise ValueError("原文件或解析配置已变化。")
        payload = envelope["parsed"]
        if _digest(payload) != envelope["checksum"]:
            raise ValueError("解析产物完整性校验失败。")
        parsed = ParsedDocument(
            [ParsedSection(**part) for part in payload["sections"]],
            [Block(**block) for block in payload["blocks"]],
            payload["report"],
        )
        parsed.report["parse_artifact"] = {
            "available": True,
            "schema_version": VERSION,
            "sha256": envelope["checksum"],
            "fingerprint": envelope["fingerprint"],
            "reused": True,
        }
        return parsed
    except (OSError, EOFError, ValueError, KeyError, TypeError) as exc:
        raise ValueError("没有有效的可复用解析产物，请先重新解析文档。") from exc
