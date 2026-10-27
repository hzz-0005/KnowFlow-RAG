import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from knowflow.app import task_alignment as alignment
from knowflow.app import task_alignment_api as api
from knowflow.app.config import Settings
from knowflow.app.db import SessionLocal
from knowflow.app.models import ConversationTurn, utc_now
from knowflow.app.round_budget import scope
from test_conversations import create, send, workspace
from test_memory import saved_task


def payload():
    return {
        "requirement_sources": {"current": "Explain production only"},
        "final_answer": "The staging value is six days.",
        "generator_model": "generator",
    }


def row(**changes):
    data = {
        "id": "r1",
        "source_id": "current",
        "request_quote": "production only",
        "verdict": "violated",
        "answer_quote": "staging value",
        "reason": "scope mismatch",
    }
    return {**data, **changes}


def test_diagnostic_mismatch_is_independent_of_faithfulness_and_has_no_success_score():
    result = alignment.validate(json.dumps({"requirements": [row()]}), payload())
    assert result["outcome"] == "mismatch_detected" and result["status"] == "complete"
    assert result["task_success"] is None and result["coverage"] == "model_selected_unverified"


@pytest.mark.parametrize(
    "change",
    [
        {"source_id": "memory_1"},
        {"request_quote": "unprovided scope"},
        {"answer_quote": "invented answer"},
        {"request_quote": ""},
        {"answer_quote": " "},
    ],
)
def test_invalid_quotes_become_unknown_not_a_negative_or_positive(change):
    result = alignment.validate(json.dumps({"requirements": [row(**change)]}), payload())
    assert result["outcome"] == "unknown" and result["requirements"][0]["verdict"] == "unknown"


def test_all_met_is_no_mismatch_not_task_pass_and_empty_is_unknown():
    assert (
        alignment.validate(json.dumps({"requirements": [row(verdict="met")]}), payload())["outcome"]
        == "no_mismatch_detected"
    )
    result = alignment.validate('{"requirements":[]}', payload())
    assert result["status"] == "partial" and result["outcome"] == "unknown"


@pytest.mark.parametrize(
    "value",
    [
        [],
        {},
        {"requirements": {}, "score": 1},
        {"requirements": [row(id="r2")]},
        {"requirements": [row(verdict="passed")]},
        {"requirements": [row(), row()]},
    ],
)
def test_invalid_schema_and_duplicate_rows_are_not_adopted(value):
    with pytest.raises(ValueError):
        alignment.validate(json.dumps(value), payload())


def test_snapshot_keeps_goal_and_current_separate_from_memory_and_old_answer():
    turn = SimpleNamespace(
        question="Continue",
        resolved_question="hint",
        status="completed",
        resolution={
            "intent_context": {
                "active_task": {"goal": {"user_input": "production"}, "user_updates": []},
                "referenced_user_inputs": [
                    {"turn_id": "memory:x:0", "question": "staging only"},
                    {"turn_id": "older", "question": "old goal"},
                ],
            }
        },
        response={
            "answer": "final",
            "answer_validation": {"status": "complete"},
            "generation_route": {"model": "gen"},
        },
    )
    data = alignment.inputs_for_turn(turn)
    assert data["requirement_sources"] == {
        "current": "Continue",
        "task_goal": "production",
        "history_2": "old goal",
    }
    assert "staging only" not in str(data["requirement_sources"])
    assert data["memory_identity_clues_not_requirements"] == ["staging only"]
    assert data["source_review_status_not_task_success"] == "complete"
    assert data["generator_model"] == "gen"


class FakeClient:
    def __init__(self, finish="stop"):
        self.sent = []
        self.finish = finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **options):
        assert options["max_retries"] == 0
        return self

    async def create(self, **kwargs):
        self.sent.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish,
                    message=SimpleNamespace(content=json.dumps({"requirements": [row()]})),
                )
            ],
            usage=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["stop", "length", "content_filter", None])
async def test_whole_budget_single_call_stop_only_and_missing_usage(monkeypatch, finish):
    client = FakeClient(finish)
    monkeypatch.setattr(alignment, "chat_client", lambda: client)
    settings = Settings(_env_file=None, llm_model="fixture")
    with scope(settings) as budget:
        result = await alignment.review(payload(), settings)
    assert len(client.sent) == 1 and budget.snapshot()["attempted_calls"] == 1
    trace = result["context_budget"]
    assert trace["provider_usage"] is None and trace["output_reserved_tokens"] == 8192
    assert result["outcome"] == ("mismatch_detected" if finish == "stop" else "unknown")
    assert result["status"] == ("complete" if finish == "stop" else "unavailable")


@pytest.mark.asyncio
async def test_budget_blocks_before_constructing_client_and_does_not_trim_input(monkeypatch):
    monkeypatch.setattr(alignment, "chat_client", lambda: pytest.fail("No paid client"))
    settings = Settings(_env_file=None, answer_review_total_token_budget=1024)
    result = await alignment.review(payload(), settings)
    assert (
        result["status"] == "budget_exceeded" and not result["context_budget"]["request_attempted"]
    )
    assert result["outcome"] == "unknown" and result["input_sha256"] == alignment.digest(payload())


def review_result(status="complete"):
    return {
        "version": alignment.VERSION,
        "status": status,
        "outcome": "mismatch_detected" if status == "complete" else "unknown",
        "requirements": [],
        "task_success": None,
        "coverage": "model_selected_unverified",
    }


async def request(client, auth, cid, tid, key=None):
    return await client.post(
        f"/api/v1/conversations/{cid}/turns/{tid}/task-review",
        headers=auth,
        json={"request_id": key or str(uuid4())},
    )


@pytest.mark.asyncio
async def test_owned_idempotent_cache_persistence_and_original_response_unchanged(monkeypatch):
    calls = []

    async def review(data, settings):
        calls.append(data)
        return review_result()

    monkeypatch.setattr(alignment, "review", review)
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        turn = (await send(client, auth[0], cid)).json()
        tid = turn["id"]
        assert (await request(client, auth[1], cid, tid)).status_code == 404
        first = await request(client, auth[0], cid, tid)
        assert first.status_code == 200, first.text
        key = first.json()["check"]["request_id"]
        duplicate = await request(client, auth[0], cid, tid, key)
        new_key = await request(client, auth[0], cid, tid)
        assert duplicate.json()["cached"] and new_key.json()["cached"] and len(calls) == 1
        saved = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()["turns"][
            0
        ]
        assert saved["response"] == turn["response"] and saved["question"] == turn["question"]
        assert len(saved["resolution"]["task_alignment_checks"]) == 1
        other = (await create(client, auth[0], kb))["id"]
        assert (await request(client, auth[0], other, tid)).status_code == 404


@pytest.mark.asyncio
async def test_busy_and_expired_claim_do_not_duplicate_or_accept_late_result(monkeypatch):
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        turn = (await send(client, auth[0], cid)).json()
        tid = UUID(turn["id"])
        async with SessionLocal() as session:
            from knowflow.app.models import Conversation

            uid = (await session.get(Conversation, UUID(cid))).owner_id
            first, _, dispatch, _ = await api.claim(session, UUID(cid), tid, uid, uuid4())
            assert dispatch
        assert (await request(client, auth[0], cid, str(tid))).status_code == 409
        async with SessionLocal() as session:
            saved = await session.get(ConversationTurn, tid)
            checks = [dict(r) for r in saved.resolution["task_alignment_checks"]]
            checks[0]["lease_until"] = (utc_now() - timedelta(seconds=1)).isoformat()
            saved.resolution = {**saved.resolution, "task_alignment_checks": checks}
            await session.commit()

        async def review(*_):
            return review_result()

        monkeypatch.setattr(alignment, "review", review)
        assert (await request(client, auth[0], cid, str(tid))).status_code == 200
        with pytest.raises(Exception) as exc:
            await api.finish(
                UUID(cid),
                tid,
                uid,
                UUID(first["request_id"]),
                first["fingerprint"],
                review_result(),
            )
        assert exc.value.status_code == 409
        saved = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()["turns"][
            0
        ]
        assert [r["status"] for r in saved["resolution"]["task_alignment_checks"]] == [
            "interrupted",
            "complete",
        ]


@pytest.mark.asyncio
async def test_cancel_persists_unknown_and_no_unchecked_answer(monkeypatch):
    async def cancel(*_):
        raise asyncio.CancelledError()

    monkeypatch.setattr(alignment, "review", cancel)
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        turn = (await send(client, auth[0], cid)).json()
        with pytest.raises(asyncio.CancelledError):
            await request(client, auth[0], cid, turn["id"])
        saved = (await client.get(f"/api/v1/conversations/{cid}", headers=auth[0])).json()["turns"][
            0
        ]
        assert saved["resolution"]["task_alignment_checks"][-1]["status"] == "interrupted"
        assert saved["response"] == turn["response"]


@pytest.mark.asyncio
async def test_legacy_task_goal_reconstruction_is_explicit_and_does_not_use_mutable_updates(
    monkeypatch,
):
    from knowflow.app.memory_models import TaskMemory

    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        task, origin = await saved_task(client, auth[0], kb, cid, "Explicit production goal")
        follow = (await send(client, auth[0], cid, "Continue", 1)).json()
        async with SessionLocal() as session:
            record = await session.get(TaskMemory, UUID(task["id"]))
            record.state = {**record.state, "user_updates": [{"user_input": "FUTURE-CHANGES"}]}
            turn = await session.get(ConversationTurn, UUID(follow["id"]))
            turn.resolution = {"task_resume": {"task_id": task["id"]}}
            await session.commit()
        observed = []

        async def review(data, _):
            observed.append(data)
            return review_result()

        monkeypatch.setattr(alignment, "review", review)
        assert (await request(client, auth[0], cid, follow["id"])).status_code == 200
        assert observed[0]["goal_basis"] == "reconstructed_from_task_source_turn_only"
        assert observed[0]["requirement_sources"]["task_goal"] == "Explicit production goal"
        assert "FUTURE-CHANGES" not in str(observed)


@pytest.mark.asyncio
async def test_missing_goal_failed_turn_and_forged_payload_do_not_call_model(monkeypatch):
    monkeypatch.setattr(alignment, "review", lambda *_: pytest.fail("No model call"))
    async with workspace(monkeypatch) as (client, auth, kb, _):
        cid = (await create(client, auth[0], kb))["id"]
        turn = (await send(client, auth[0], cid)).json()
        async with SessionLocal() as session:
            saved = await session.get(ConversationTurn, UUID(turn["id"]))
            saved.resolution = {"task_resume": {"task_id": str(uuid4())}}
            await session.commit()
        result = (await request(client, auth[0], cid, turn["id"])).json()
        assert (
            result["check"]["status"] == "unavailable"
            and result["check"]["reason"] == "missing_task_goal"
        )
        forged = await client.post(
            f"/api/v1/conversations/{cid}/turns/{turn['id']}/task-review",
            headers=auth[0],
            json={"request_id": str(uuid4()), "answer": "FORGED"},
        )
        assert forged.status_code == 422
