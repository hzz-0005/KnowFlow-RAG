"""Topic boundaries are validated server events, not guessed from product names."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from knowflow.app import conversation_context as context
from knowflow.app.config import get_settings
from knowflow.app.token_budget import get_token_counter


def turn(key, question, *, boundary=False, status="completed", resolution=None):
    if resolution is None:
        resolution = (
            {
                "version": "conversation-reference-v5-history-aware-constraints",
                "status": "standalone",
                "question": question,
                "reference_turn_ids": [],
            }
            if boundary
            else {}
        )
    return SimpleNamespace(
        id=key,
        question=question,
        resolved_question=question,
        status=status,
        resolution=resolution,
        response={"answer": "OLD_ANSWER_NEVER_EVIDENCE", "citations": []},
    )


@pytest.mark.parametrize(
    "old,new",
    [
        (
            "PineVault only staging; exclude production.",
            "Different topic: CedarHub backup retention?",
        ),
        ("只看采购台账，不查销售。", "换个问题：通知服务如何验证消息？"),
    ],
)
def test_validated_topic_switch_excludes_old_domain_and_scope(old, new):
    rows = [turn("new", new, boundary=True), turn("old", old, boundary=True)]
    original = deepcopy([vars(r) for r in rows])
    payload = context.context_payload(
        "那怎么办？", rows, get_token_counter(get_settings().chunk_tokenizer_path)
    )
    assert [r["turn_id"] for r in payload["previous_intents"]] == ["new"]
    assert payload["active_topic"]["boundary_turn_id"] == "new"
    assert payload["active_topic"]["older_loaded_turns_omitted"] == 1
    assert payload["active_topic"]["count_scope"] == "loaded_recent_intents_only"
    assert old not in json.dumps(payload, ensure_ascii=False)
    assert "OLD_ANSWER_NEVER_EVIDENCE" not in json.dumps(payload)
    assert [vars(r) for r in rows] == original
    with pytest.raises(context.ResolutionError, match="invalid_reference_turn"):
        context.validate_resolution(
            json.dumps(
                {"status": "resolved", "question": "old domain", "reference_turn_ids": ["old"]}
            ),
            payload,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"version": "historical_unknown"},
        {"reference_turn_ids": ["old"]},
        {"intent_context": {"current_user_input": "retained old task"}},
        {"question": "changed original"},
    ],
)
def test_unvalidated_or_inconsistent_marker_does_not_cut_history(change):
    latest = turn("new", "New complete question?", boundary=True)
    latest.resolution.update(change)
    rows = [latest, turn("old", "Old question?")]
    selected, topic = context.active_topic_history(rows)
    assert selected == rows and topic is None


def test_continuations_after_boundary_are_kept_newest_first_before_budget_packing():
    rows = [
        turn("follow", "What changed?"),
        turn("root", "New topic?", boundary=True),
        turn("old", "Prior topic?"),
    ]
    selected, topic = context.active_topic_history(rows)
    assert [r.id for r in selected] == ["follow", "root"]
    assert topic["older_loaded_turns_omitted"] == 1


@pytest.mark.asyncio
async def test_pending_clarification_survives_boundary_and_does_not_revive_old_scope(monkeypatch):
    root = turn("new", "CedarHub backups?", boundary=True)
    pending = turn(
        "pending",
        "那保存多长时间？",
        status="needs_clarification",
        resolution={
            "question": "Which product?",
            "pending_intent": {"question": "那保存多长时间？", "supplements": []},
        },
    )
    old = turn("old", "PineVault only staging, not production.")
    requests = []

    async def create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        requests.append(payload)
        assert payload["pending_clarification"]["turn_id"] == "pending"
        assert [r["turn_id"] for r in payload["previous_intents"]] == ["new"]
        assert "PineVault" not in json.dumps(payload)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "status": "resolved",
                                "question": "CedarHub backup retention?",
                                "reference_turn_ids": ["pending", "new"],
                            }
                        )
                    ),
                )
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.with_options = lambda **_: client
    monkeypatch.setattr(context, "chat_client", lambda: client)
    result = await context.resolve_question("CedarHub", [pending, root, old], get_settings())
    assert result["status"] == "resolved" and len(requests) == 1
    assert result["topic_scope"]["boundary_turn_id"] == "new"
    assert "PineVault" not in json.dumps(result["intent_context"])


@pytest.mark.asyncio
async def test_required_recent_intent_budget_remains_mandatory_with_topic_boundary(monkeypatch):
    root = turn("root", "CedarHub backups?", boundary=True)
    oversized = turn("huge", "complete user condition " * 9000)

    def forbidden():
        raise AssertionError("Must not guess from older smaller root")

    monkeypatch.setattr(context, "chat_client", forbidden)
    result = await context.resolve_question("How long?", [oversized, root], get_settings())
    assert result["reason"] == "recent_intent_budget_exceeded" and result["status"] == "clarify"
    assert result["topic_scope"]["boundary_turn_id"] == "root"
    assert result["required_recent_intent_omitted"]["turn_id"] == "huge"
