import gzip
import json

import pytest
from knowflow.app.config import Settings
from knowflow.app.document_processing import ParsedSection
from knowflow.app.parse_artifacts import artifact_path, load_artifact, save_artifact
from knowflow.app.structured_parsing import Block, ParsedDocument


def test_rechunk_cache_survives_chunk_policy_changes_but_rejects_source_or_parser_changes(tmp_path):
    settings = Settings(upload_dir=str(tmp_path))
    parsed = ParsedDocument(
        [ParsedSection("正文17天")], [Block("b", "text", "正文17天")], {"parser": "native"}
    )
    meta = save_artifact("sample.pdf", "report.pdf", "source-hash", parsed, settings)
    settings.chunk_max_tokens = 256
    loaded = load_artifact("sample.pdf", "report.pdf", "source-hash", settings)
    assert loaded.sections == parsed.sections
    assert loaded.report["parse_artifact"]["sha256"] == meta["sha256"]
    for sha, revision in [("different", "v1"), ("source-hash", "different-parser")]:
        settings.parse_artifact_revision = revision
        with pytest.raises(ValueError, match="重新解析"):
            load_artifact("sample.pdf", "report.pdf", sha, settings)


def test_corrupt_or_tampered_cache_is_not_silently_used(tmp_path):
    settings = Settings(upload_dir=str(tmp_path))
    parsed = ParsedDocument([ParsedSection("正确数值17")], [], {})
    save_artifact("file.pdf", "file.pdf", "sha", parsed, settings)
    path = artifact_path("file.pdf", settings)
    data = json.loads(gzip.decompress(path.read_bytes()))
    data["parsed"]["sections"][0]["text"] = "错误数值99"
    path.write_bytes(gzip.compress(json.dumps(data).encode()))
    with pytest.raises(ValueError, match="重新解析"):
        load_artifact("file.pdf", "file.pdf", "sha", settings)
