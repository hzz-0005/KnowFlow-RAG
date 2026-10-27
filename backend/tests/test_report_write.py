import json

import pytest
from knowflow.app import evaluate_quality as quality


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_windows_contention_has_bounded_replace_retry_and_complete_json(
    monkeypatch, tmp_path, winerror
):
    report = {"run_id": "fixture", "status": "new"}
    path = tmp_path / "fixture.json"
    path.write_text('{"status":"old"}', encoding="utf-8")
    replace = quality.os.replace
    calls, delays = [], []

    def busy_then_replace(src, dst):
        calls.append(True)
        if len(calls) < 3:
            assert json.loads(path.read_text(encoding="utf-8"))["status"] == "old"
            exc = PermissionError("mocked contention")
            exc.winerror = winerror
            raise exc
        replace(src, dst)

    monkeypatch.setattr(quality.os, "replace", busy_then_replace)
    monkeypatch.setattr(quality.time, "sleep", delays.append)
    assert quality.save_report(report, tmp_path) == path
    assert len(calls) == 3 and delays == [0.01, 0.05]
    assert json.loads(path.read_text(encoding="utf-8")) == report


@pytest.mark.parametrize("code,count", [(32, 4), (None, 1)])
def test_persistent_error_is_never_waived_or_in_place_overwritten(
    monkeypatch, tmp_path, code, count
):
    path = tmp_path / "fixture.json"
    path.write_bytes(b'{"status":"old"}')
    calls = []

    def fail(*_):
        calls.append(True)
        exc = PermissionError("mocked permanent failure")
        if code is not None:
            exc.winerror = code
        raise exc

    monkeypatch.setattr(quality.os, "replace", fail)
    monkeypatch.setattr(quality.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        quality.save_report({"run_id": "fixture", "status": "new"}, tmp_path)
    assert len(calls) == count
    assert path.read_bytes() == b'{"status":"old"}'
