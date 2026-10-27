import gzip
import hashlib
import json

import pytest
from knowflow.app import diagnostic_replay


def test_replay_rejects_tampered_and_escaping_inputs_before_parsing(tmp_path):
    base = tmp_path / "inputs"
    base.mkdir()
    path = base / "snapshot.gz"
    raw = gzip.compress(b'{"schema":"test"}')
    path.write_bytes(raw)
    row = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
    assert diagnostic_replay.load_checked(base, row)["schema"] == "test"
    path.write_bytes(raw + b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        diagnostic_replay.load_checked(base, row)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    with pytest.raises(ValueError, match="outside"):
        diagnostic_replay.load_checked(base, {"path": "../outside.json", "sha256": "x"})


def test_replay_rejects_compressed_expansion_over_budget(tmp_path, monkeypatch):
    raw = gzip.compress(json.dumps({"text": "x" * 1000}).encode())
    (tmp_path / "large.gz").write_bytes(raw)
    monkeypatch.setattr(diagnostic_replay, "MAX_EXPANDED_BYTES", 100)
    with pytest.raises(ValueError, match="Expanded"):
        diagnostic_replay.load_checked(
            tmp_path,
            {
                "path": "large.gz",
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        )


def test_replay_requires_original_parser_and_index_text_to_agree():
    snapshot = {
        "schema": "knowflow.diagnostic-snapshot.v1",
        "blocks": [{"block_key": "b", "text_length": 3}],
        "block_texts": {"b": "old"},
    }
    with pytest.raises(ValueError, match="differs"):
        diagnostic_replay.compare(snapshot, {"blocks": [{"block_key": "b", "text": "new"}]})
