"""Release/config checks use dummy secrets and temporary directories only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from knowflow.app.config import Settings

from scripts.check_release import scan
from scripts.prepare_release import build
from scripts.public_files import allowed

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("value", ["short", "replace-this-with-an-example-secret-long-enough"])
def test_production_rejects_example_or_short_signing_secrets(value):
    with pytest.raises(ValueError, match="unique JWT_SECRET"):
        Settings(_env_file=None, app_env="production", jwt_secret=value)


def test_generated_config_is_unique_and_never_overwrites_or_prints_secrets(tmp_path):
    generated = []
    for index in range(2):
        path = tmp_path / f"config-{index}.env"
        command = [sys.executable, str(ROOT / "scripts/init_env.py"), "--out", str(path)]
        first = subprocess.run(command, capture_output=True)
        assert first.returncode == 0
        data = path.read_bytes()
        line = next(row for row in data.splitlines() if row.startswith(b"JWT_SECRET="))
        secret = line.split(b"=", 1)[1]
        assert len(secret) >= 48 and secret not in first.stdout + first.stderr
        generated.append(secret)
        retry = subprocess.run(command, capture_output=True)
        assert retry.returncode != 0 and path.read_bytes() == data
    assert generated[0] != generated[1]


def test_public_policy_excludes_user_data_archives_and_generated_environments():
    for name in [
        ".env",
        "data/uploads/test.pdf",
        "review/history.py",
        "总结/report.md",
        "frontend/node_modules/private.js",
        "frontend/dist/answer-view.js",
        "evaluation/m2_v1/corpus/private.pdf",
        "to-delete/old.py",
    ]:
        assert not allowed(name), name
    assert allowed("backend/tests/m2_synthetic_fixture.py")
    assert allowed("evaluation/conversation_v1/scenarios.json")


def test_export_refuses_recursive_source_destination_and_existing_output(tmp_path):
    with pytest.raises(ValueError, match="outside exported"):
        build(tmp_path, tmp_path / "backend" / "recursive")
    target = tmp_path / "already-there"
    target.mkdir()
    with pytest.raises(FileExistsError):
        build(tmp_path, target)


def test_scanner_reports_location_but_not_dummy_credential_value(tmp_path):
    private = tmp_path / "private.env"
    secret = "dummy-" + "x" * 40
    private.write_text("LLM_API_KEY=" + secret, encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text(secret, encoding="utf-8")
    (source / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"files": [{"path": "README.md", "sha256": "wrong"}]}), encoding="utf-8"
    )
    result = scan(source, private)
    assert not result["passed"]
    assert any(r["rule"] == "matches_local_secret" for r in result["findings"])
    assert any(r["rule"] == "fingerprint_mismatch" for r in result["findings"])
    assert secret not in json.dumps(result)
