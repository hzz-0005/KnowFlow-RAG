import copy
import json
from uuid import UUID, uuid4

import pytest
from knowflow.app.conversation_review import (
    DIMENSIONS,
    digest,
    load_json,
    make_bundle,
    validate_annotations,
    validate_bundle,
)
from knowflow.app.db import SessionLocal
from knowflow.app.models import ConversationTurn
from knowflow.app.review_conversation_bundle import run
from test_conversations import create, send, workspace


def fixture():
    row = {
        "id": "t1",
        "sequence": 1,
        "question": "只查生产。",
        "answer": "生产保留90天。",
        "task_inputs": {"requirement_sources": {"current": "只查生产。"}},
        "citations": [{"id": "t1:c1", "quote": "生产保留90天。"}],
    }
    row["turn_sha256"] = digest(row)
    return make_bundle(
        {"target_turn_id": "t1", "through_sequence": 1, "turns": [row]}, "2026-10-25T00:00:00+00:00"
    )


def annotated(bundle):
    labels = copy.deepcopy(bundle["review_template"])
    labels.update(
        declared_method="human",
        reviewer_id="developer-control-not-independent",
        reviewed_at="2026-10-25T01:00:00+00:00",
    )
    labels["criteria"][0].update(
        verdict="satisfied",
        reason="Synthetic validator control",
        request_ref={"source_id": "current", "quote": "生产"},
        answer_quote="生产保留90天。",
    )
    return labels


def test_blank_template_keeps_all_dimensions_unreviewed_and_no_semantic_score():
    bundle = fixture()
    report = validate_annotations(bundle, bundle["review_template"])
    assert report["unreviewed_dimensions"] == list(DIMENSIONS)
    assert report["task_success"] is None
    assert report["independent_review_status"] == "not_established"
    assert {r["observation"] for r in report["criteria"]} == {"unknown"}


def test_valid_declared_human_witness_does_not_establish_independence():
    bundle = fixture()
    report = validate_annotations(bundle, annotated(bundle))
    assert report["annotation_status"] == "declared_not_independently_verified"
    assert report["task_success"] is None
    assert report["independent_review_status"] == "not_established"


def test_human_can_reference_raw_history_the_system_failed_to_select():
    snapshot = fixture()["snapshot"]
    row = {
        "id": "t2",
        "sequence": 2,
        "question": "那它呢？",
        "answer": "生产保留90天。",
        "task_inputs": {"requirement_sources": {"current": "那它呢？"}},
        "citations": [],
    }
    row["turn_sha256"] = digest(row)
    snapshot["turns"].append(row)
    snapshot.update(target_turn_id="t2", through_sequence=2)
    bundle = make_bundle(snapshot, "2026-10-25T00:00:00+00:00")
    labels = annotated(bundle)
    labels["criteria"][0]["request_ref"] = {"source_id": "turn:t1", "quote": "生产"}
    assert validate_annotations(bundle, labels)["criteria"][0]["verdict"] == "satisfied"


def test_size_and_unknown_reason_bounds_are_explicit():
    bundle = fixture()
    bundle["notice"] = "x" * (2 * 1024 * 1024)
    with pytest.raises(ValueError, match="exceeds"):
        validate_bundle(bundle)
    bundle = fixture()
    labels = annotated(bundle)
    labels["criteria"][0].update(verdict="unknown", reason="")
    with pytest.raises(ValueError, match="reason"):
        validate_annotations(bundle, labels)


def test_source_support_requires_citation_and_independent_method_is_not_inferred():
    bundle = fixture()
    labels = annotated(bundle)
    row = next(r for r in labels["criteria"] if r["dimension"] == "source_support")
    row.update(
        verdict="satisfied",
        reason="literal control",
        request_ref={"source_id": "current", "quote": "生产"},
        answer_quote="90天",
    )
    with pytest.raises(ValueError, match="citation witnesses"):
        validate_annotations(bundle, labels)
    row["citation_refs"] = [{"citation_id": "t1:c1", "quote": "90天"}]
    assert validate_annotations(bundle, labels)["task_success"] is None
    labels["declared_method"] = "unknown"
    with pytest.raises(ValueError, match="method"):
        validate_annotations(bundle, labels)


@pytest.mark.parametrize(
    "field,value",
    [
        ("answer_quote", "invented"),
        ("request_ref", {"source_id": "memory:future", "quote": "production"}),
        ("request_ref", {"source_id": "current", "quote": "invented"}),
        ("citation_refs", [{"citation_id": "other-turn:c1", "quote": "生产"}]),
        ("citation_refs", [{"citation_id": "t1:c1", "quote": "invented"}]),
        ("verdict", "passed"),
        ("reason", ""),
    ],
)
def test_invalid_annotation_witnesses_and_identity_are_rejected(field, value):
    bundle = fixture()
    labels = annotated(bundle)
    labels["criteria"][0][field] = value
    with pytest.raises(ValueError):
        validate_annotations(bundle, labels)


def test_labels_cannot_migrate_to_another_answer_or_snapshot():
    bundle = fixture()
    labels = annotated(bundle)
    bundle["snapshot"]["turns"][0]["answer"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_annotations(bundle, labels)
    labels["snapshot_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        validate_annotations(fixture(), labels)


def test_full_history_and_turn_fingerprints_are_required():
    bundle = fixture()
    bundle["snapshot"]["turns"][0]["sequence"] = 2
    bundle["snapshot_sha256"] = digest(bundle["snapshot"])
    with pytest.raises(ValueError, match="history"):
        validate_bundle(bundle)
    bundle = fixture()
    bundle["snapshot"]["turns"][0]["question"] = "forged"
    bundle["snapshot_sha256"] = digest(bundle["snapshot"])
    with pytest.raises(ValueError, match="Turn fingerprint"):
        validate_bundle(bundle)


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path, bad):
    file = tmp_path / "bad.json"
    file.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        load_json(file)


def test_cli_unknown_gate_utf8_readback_and_no_overwrite(tmp_path, capsys):
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(fixture(), ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "out"
    assert run(["--bundle", str(bundle_path), "--out", str(output), "--require-annotations"]) == 3
    assert "unknown不是0分" in (output / "summary.md").read_text(encoding="utf-8")
    original = (output / "report.json").read_bytes()
    assert run(["--bundle", str(bundle_path), "--out", str(output)]) == 2
    assert (output / "report.json").read_bytes() == original
    assert "exists" in capsys.readouterr().err
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]", encoding="utf-8")
    assert run(["--bundle", str(malformed), "--out", str(tmp_path / "never-created")]) == 2
    assert not (tmp_path / "never-created").exists()


@pytest.mark.asyncio
async def test_api_exports_only_history_through_target_without_model_judgments_or_writes(
    monkeypatch,
):
    async with workspace(monkeypatch) as (client, auth, kb, calls):
        conversation = await create(client, auth[0], kb)
        first = await send(client, auth[0], conversation["id"], "Production only")
        assert first.status_code == 200
        await send(
            client, auth[0], conversation["id"], "Future correction must be absent", revision=1
        )
        view = (
            await client.get(f"/api/v1/conversations/{conversation['id']}", headers=auth[0])
        ).json()
        tid = view["turns"][0]["id"]
        async with SessionLocal() as session:
            row = await session.get(ConversationTurn, UUID(tid))
            row.resolution = {
                **row.resolution,
                "task_alignment_checks": [{"outcome": "model_claim_should_be_absent"}],
            }
            row.response = {
                **row.response,
                "answer_validation": {"status": "model_claim_should_be_absent"},
                "generation_route": {"model": "hidden-model-identity"},
            }
            await session.commit()
        before = (
            await client.get(f"/api/v1/conversations/{conversation['id']}", headers=auth[0])
        ).json()
        call_count = len(calls)
        path = f"/api/v1/conversations/{conversation['id']}/turns/{tid}/review-export"
        result = await client.get(path, headers=auth[0])
        assert result.status_code == 200, result.text
        assert result.headers["cache-control"] == "no-store"
        bundle = result.json()
        assert len(bundle["snapshot"]["turns"]) == 1
        assert bundle["snapshot"]["turns"][0]["question"] == "Production only"
        assert "Future correction" not in result.text
        assert "model_claim_should_be_absent" not in result.text
        assert "hidden-model-identity" not in result.text
        assert len(calls) == call_count
        again = await client.get(path, headers=auth[0])
        assert again.json()["snapshot_sha256"] == bundle["snapshot_sha256"]
        after = (
            await client.get(f"/api/v1/conversations/{conversation['id']}", headers=auth[0])
        ).json()
        assert after == before
        assert (await client.get(path, headers=auth[1])).status_code == 404
        other = await create(client, auth[0], kb)
        assert (
            await client.get(
                f"/api/v1/conversations/{other['id']}/turns/{tid}/review-export", headers=auth[0]
            )
        ).status_code == 404


@pytest.mark.asyncio
async def test_api_pending_failed_and_history_limit_are_explicit(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        tids = [uuid4() for _ in range(101)]
        async with SessionLocal() as session:
            for i, tid in enumerate(tids, 1):
                session.add(
                    ConversationTurn(
                        id=tid,
                        conversation_id=UUID(cid),
                        sequence=i,
                        request_id=uuid4(),
                        request_sha256="a" * 64,
                        question="Raw question",
                        status="pending" if i == 1 else "failed",
                        resolution={},
                        response=None,
                    )
                )
            await session.commit()
        root = f"/api/v1/conversations/{cid}/turns"
        assert (
            await client.get(f"{root}/{tids[0]}/review-export", headers=auth[0])
        ).status_code == 409
        failed = await client.get(f"{root}/{tids[1]}/review-export", headers=auth[0])
        assert failed.status_code == 200, failed.text
        row = failed.json()["snapshot"]["turns"][-1]
        assert row["status"] == "failed" and row["answer"] is None
        assert (
            await client.get(f"{root}/{tids[-1]}/review-export", headers=auth[0])
        ).status_code == 413
