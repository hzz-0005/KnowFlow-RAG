"""Transactional task claim/completion fences. Resumes re-retrieve, never replay side effects."""

from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select

from knowflow.app.memory_models import TaskAttempt, TaskMemory, UserMemory
from knowflow.app.memory_store import dependencies
from knowflow.app.models import Conversation, ConversationTurn, utc_now


def checkpoint(conversation, turn):
    response = turn.response or {}
    intent = (turn.resolution or {}).get("intent_context") or {}
    return TaskMemory(
        owner_id=conversation.owner_id,
        knowledge_base_id=conversation.knowledge_base_id,
        source_turn_id=turn.id,
        title=turn.question[:100],
        status="awaiting_clarification" if turn.status == "needs_clarification" else "ready",
        state={
            "version": "retrieval-task-checkpoint-v1",
            "goal": turn.question,
            "retrieval_hint": turn.resolved_question,
            "memory_dependencies": dependencies(turn.resolution or {}),
            "user_updates": [],
            "user_inputs": intent.get("referenced_user_inputs", []),
            "pending_intent": (turn.resolution or {}).get("pending_intent"),
            "partial_answer": response.get("partial_answer"),
            "plan": (response.get("query_plan") or {}).get("task_plan"),
            "source_versions": turn.source_versions,
            "initial_turn_status": turn.status,
            "note": "恢复重新检索完整只读任务；旧回答不是事实，依赖步骤未自动执行。",
        },
    )


async def claim(session, conversation, turn, body):
    from knowflow.app.memory_api import owned_task

    if body.task_id is None:
        if body.task_revision is not None:
            raise HTTPException(422, "任务修订号需要任务ID。")
        return
    task = await owned_task(
        session, body.task_id, conversation.owner_id, conversation.knowledge_base_id, lock=True
    )
    if task.revision != body.task_revision:
        raise HTTPException(409, "任务已更新，请刷新检查点后恢复。")
    if task.status == "cancelled":
        raise HTTPException(409, "任务已取消，请从原问题另存新任务。")
    if task.active_turn_id:
        active = await session.get(ConversationTurn, task.active_turn_id)
        owner = await session.get(Conversation, active.conversation_id) if active else None
        if (
            owner
            and owner.active_until
            and owner.active_until > utc_now()
            and active.status == "pending"
        ):
            raise HTTPException(409, "此任务仍在运行，不重复派发。")
        old = await session.scalar(
            select(TaskAttempt).where(TaskAttempt.turn_id == task.active_turn_id)
        )
        if old and old.status == "running":
            old.status, old.finished_at = "interrupted", utc_now()
            old.result = {"reason": "lease_expired", "result_quality": "unknown"}
    previous = await session.scalar(
        select(TaskAttempt)
        .where(TaskAttempt.task_id == task.id)
        .order_by(TaskAttempt.created_at.desc(), TaskAttempt.id.desc())
        .limit(1)
    )
    attempt = TaskAttempt(
        task_id=task.id, turn_id=turn.id, retry_of_attempt_id=previous.id if previous else None
    )
    session.add(attempt)
    task.status, task.active_turn_id = "running", turn.id
    task.revision += 1
    task.updated_at = utc_now()


async def resume_context(session, task_id, current, versions, clues=None):
    task = await session.get(TaskMemory, task_id)
    state = task.state
    original = state.get("pending_intent") or {"question": state["goal"], "supplements": []}
    prior = [
        *state.get("user_inputs", []),
        {
            "turn_id": str(task.source_turn_id),
            "question": original["question"],
            "supplements": original.get("supplements", []),
        },
    ]
    # Memory-derived instructions never silently survive a deleted/expired memory.
    prior = [r for r in prior if not str(r.get("turn_id", "")).startswith("memory:")]
    hint_valid = True
    for dep in state.get("memory_dependencies", []):
        memory = await session.get(UserMemory, UUID(dep["id"]))
        if (
            memory is None
            or memory.status != "active"
            or memory.revision != dep["revision"]
            or (memory.expires_at is not None and memory.expires_at <= utc_now())
        ):
            hint_valid = False
            break
    parts = [original["question"]]
    parts.extend("原任务已收到的用户补充：" + text for text in original.get("supplements", []))
    if hint_valid and state.get("retrieval_hint"):
        parts.append("先前检索线索（不是事实，本轮更正优先）：" + state["retrieval_hint"])
    else:
        parts.extend(r["question"] for r in prior)
    parts.extend(row.question for row in (clues or []))
    # These updates already govern the answer. They must also reach retrieval on
    # the next resume, even when the new user text is only "continue". Do not add
    # sequence numbers/IDs to the search text: they are not user numeric conditions.
    parts.extend(
        "已保存的用户补充与更正（从旧到新，后项优先）：" + row["user_input"]
        for row in state.get("user_updates", [])
    )
    parts.append("本轮补充与更正：" + current)
    hint = "\n".join(dict.fromkeys(parts))
    if len(hint) > 4000 or original.get("requires_restatement"):
        return {
            "status": "clarify",
            "reason": "task_resume_context_requires_restatement"
            if original.get("requires_restatement")
            else "task_resume_input_too_large",
            "question": "任务累计条件超出预算或缺少完整条件；请取消本次恢复，"
            "用完整对象和条件新建问题或检查点。原条件已保留，未截断检索。",
            "pending_intent": {"question": "", "supplements": [], "requires_restatement": True},
        }
    changes = [
        key for key, value in state.get("source_versions", {}).items() if versions.get(key) != value
    ]
    return {
        "version": "task-resume-ordered-updates-v2",
        "status": "resolved",
        "question": hint,
        "reference_turn_ids": [str(task.source_turn_id)],
        "intent_context": {
            "version": "user-intent-task-authority-v2",
            "current_user_input": current,
            "referenced_user_inputs": prior,
            "active_task": {
                "goal": {"turn_id": str(task.source_turn_id), "user_input": state["goal"]},
                "user_updates": state.get("user_updates", []),
                "authority": "current > latest_task_updates > goal > memory_clues",
            },
        },
        "task_resume": {
            "version": "task-resume-ordered-retrieval-v3",
            "task_id": str(task.id),
            "revision": task.revision,
            "source_versions_changed": changes,
            "execution": "full_read_only_retrieval_restart",
            "step_coverage": "unverified",
            "retrieval_input_policy": "goal_supplements_hint_clues_ordered_updates_current-v1",
            "saved_update_count": len(state.get("user_updates", [])),
        },
    }


async def finish(session, turn, status, resolution, response, versions):
    attempt = await session.scalar(select(TaskAttempt).where(TaskAttempt.turn_id == turn.id))
    if attempt is None:
        return
    task = await session.scalar(
        select(TaskMemory).where(TaskMemory.id == attempt.task_id).with_for_update()
    )
    attempt.status = status
    attempt.finished_at = utc_now()
    attempt.result = {
        "turn_id": str(turn.id),
        "source_versions": versions,
        "refused": (response or {}).get("refused"),
        "result_identity": str(turn.id),
        "semantic_quality": "unknown",
        "step_coverage": "unverified",
        "task_fence_accepted": task.active_turn_id == turn.id and task.status != "cancelled",
    }
    if not attempt.result["task_fence_accepted"]:
        return
    task.status = (
        "awaiting_clarification"
        if status == "needs_clarification"
        else "completed"
        if status == "completed"
        else "failed"
    )
    task.active_turn_id = None
    task.revision += 1
    task.updated_at = utc_now()
    task.state = {
        **task.state,
        "last_turn_id": str(turn.id),
        "last_result_status": status,
        "source_versions": versions or task.state.get("source_versions", {}),
        "pending_intent": resolution.get("pending_intent"),
    }
    # Corrections are carried across retries, but never the assistant's answer.
    intent = resolution.get("intent_context") or {}
    raw = intent.get("current_user_input")
    if raw:
        prior = [
            *intent.get("referenced_user_inputs", []),
            {"turn_id": str(turn.id), "question": raw},
        ]
        task.state = {
            **task.state,
            "user_inputs": prior,
            "user_updates": [
                *task.state.get("user_updates", []),
                {"turn_id": str(turn.id), "user_input": raw},
            ],
        }
