"""Scripted full-API trajectories: persistence/transport controls, not semantic model scores."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from knowflow.app import conversation_api, conversation_context
from knowflow.app.conversation_review import DIMENSIONS, digest, validate_bundle
from knowflow.app.schemas import QueryResponse
from test_conversations import create, send, workspace

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evaluation/conversation_v1/scenarios.json"
SUITE = json.loads(DATA.read_text(encoding="utf-8"))


def test_scenario_contract_is_development_only_and_references_only_prior_turns():
    assert SUITE["independent_review_status"] == "not_established"
    assert SUITE["semantic_task_success"] is None
    assert SUITE["sample_role"] == "developer_authored_synthetic_controls"
    assert len({row["id"] for row in SUITE["scenarios"]}) == len(SUITE["scenarios"])
    covered = set()
    for scenario in SUITE["scenarios"]:
        covered.update(scenario["dimensions"])
        for number, turn in enumerate(scenario["turns"], 1):
            assert turn["status"] in {"standalone", "resolved", "clarify"}
            assert len(turn["references"]) == len(set(turn["references"]))
            assert all(type(n) is int and 1 <= n < number for n in turn["references"])
            assert all(1 <= n < number for n in turn["required_raw_turns"])
            if turn["status"] == "resolved":
                assert turn["references"] and turn["scripted_question"]
            else:
                assert not turn["references"]
    assert covered == set(DIMENSIONS)
    for control in SUITE["memory_state_controls"]:
        path, function = control["test"].split("::")
        assert f"async def {function}(" in (ROOT / path).read_text(encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", SUITE["scenarios"], ids=lambda row: row["id"])
async def test_scripted_multiturn_trajectory_survives_reload_and_exports_every_turn(
    monkeypatch, scenario
):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        active = {}
        saved = []
        query_calls, model_calls = [], []

        async def completion(**kwargs):
            turn = active["turn"]
            payload = json.loads(kwargs["messages"][1]["content"])
            serialized = json.dumps(payload, ensure_ascii=False)
            assert "SCRIPTED_ASSISTANT_NOT_EVIDENCE" not in serialized
            for text in turn.get("forbidden_payload_text", []):
                assert text not in serialized
            model_calls.append(payload)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "status": turn["status"],
                                    "question": turn.get("scripted_question", turn["question"]),
                                    "reference_turn_ids": [
                                        saved[n - 1]["id"] for n in turn["references"]
                                    ],
                                },
                                ensure_ascii=False,
                            )
                        ),
                    )
                ]
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=completion)))
        fake.with_options = lambda **kw: fake if kw == {"max_retries": 0} else pytest.fail("retry")
        monkeypatch.setattr(conversation_context, "chat_client", lambda: fake)

        async def query(kb_id, body, user, session):
            query_calls.append({"question": body.question, "intent": body._intent_context})
            if active["turn"].get("query_failure"):
                raise RuntimeError("SCRIPTED_PRIVATE_PROVIDER_ERROR")
            return QueryResponse(
                answer="SCRIPTED_ASSISTANT_NOT_EVIDENCE",
                refused=False,
                strategy=body.strategy,
                citations=[],
            )

        monkeypatch.setattr(conversation_api, "query_knowledge_base", query)
        cid = (await create(client, auth[0], kb))["id"]
        bundles = []
        for sequence, turn in enumerate(scenario["turns"], 1):
            active["turn"] = turn
            query_before = len(query_calls)
            result = await send(client, auth[0], cid, turn["question"], sequence - 1)
            assert result.status_code == 200, result.text
            row = result.json()
            saved.append(row)
            assert row["resolution"]["status"] == turn["status"]
            assert row["resolution"]["reference_turn_ids"] == [
                saved[n - 1]["id"] for n in turn["references"]
            ]
            expected_status = (
                "needs_clarification"
                if turn["status"] == "clarify"
                else "failed"
                if turn.get("query_failure")
                else "completed"
            )
            assert row["status"] == expected_status
            assert "SCRIPTED_PRIVATE_PROVIDER_ERROR" not in json.dumps(row)
            assert len(query_calls) - query_before == (turn["status"] != "clarify")
            intent = row["resolution"].get("intent_context")
            if turn["status"] == "resolved":
                assert intent["current_user_input"] == turn["question"]
                inherited = {r["turn_id"]: r for r in intent["referenced_user_inputs"]}
                for prior in turn["required_raw_turns"]:
                    assert (
                        inherited[saved[prior - 1]["id"]]["question"]
                        == (scenario["turns"][prior - 1]["question"])
                    )
                assert query_calls[-1]["intent"] == intent
            else:
                assert intent is None
            history = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()
            assert history["revision"] == sequence and history["turns"] == saved
            response = await client.get(
                f"/api/v1/conversations/{cid}/turns/{row['id']}/review-export", headers=auth[0]
            )
            assert response.status_code == 200, response.text
            bundle = response.json()
            validate_bundle(bundle)
            assert bundle["snapshot"]["through_sequence"] == sequence
            assert [t["question"] for t in bundle["snapshot"]["turns"]] == [
                t["question"] for t in scenario["turns"][:sequence]
            ]
            assert all(c["verdict"] == "unreviewed" for c in bundle["review_template"]["criteria"])
            bundles.append(bundle)
        # Re-read the first snapshot after future turns exist: no future information leaks.
        first = (
            await client.get(
                f"/api/v1/conversations/{cid}/turns/{saved[0]['id']}/review-export", headers=auth[0]
            )
        ).json()
        assert first["snapshot_sha256"] == bundles[0]["snapshot_sha256"]
        if target := os.environ.get("KNOWFLOW_TEST_EVIDENCE_DIR"):
            folder = Path(target) / ("software-trajectories-" + uuid4().hex)
            folder.mkdir(parents=True, exist_ok=False)
            (folder / (scenario["id"] + ".json")).write_text(
                json.dumps(
                    {
                        "scenario_id": scenario["id"],
                        "scenario_sha256": digest(scenario),
                        "execution_mode": "isolated_api_with_scripted_context_and_query",
                        "model_requests_stubbed": len(model_calls),
                        "query_requests_stubbed": len(query_calls),
                        "independent_review_status": "not_established",
                        "task_success": None,
                        "turns": saved,
                        "review_bundles": bundles,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
