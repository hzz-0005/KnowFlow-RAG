"""Idempotent, optional review of an immutable saved answer. Does not alter that answer."""

import asyncio
from datetime import timedelta
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from knowflow.app import task_alignment
from knowflow.app.api import UserDep
from knowflow.app.config import get_settings
from knowflow.app.conversation_api import owned_conversation
from knowflow.app.db import SessionLocal
from knowflow.app.memory_models import TaskMemory
from knowflow.app.models import ConversationTurn, utc_now
from knowflow.app.round_budget import scope
from knowflow.app.security import SessionDep

router = APIRouter(prefix="/api/v1", tags=["task alignment"])


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: UUID


async def saved_input(session, conversation, turn):
    goal = None
    resolution = turn.resolution or {}
    task_id = resolution.get("task_resume", {}).get("task_id")
    if task_id and not (resolution.get("intent_context") or {}).get("active_task"):
        try:
            parsed_id = UUID(task_id)
        except (ValueError, TypeError):
            parsed_id = None
        task = await session.scalar(
            select(TaskMemory).where(
                TaskMemory.id == parsed_id,
                TaskMemory.owner_id == conversation.owner_id,
                TaskMemory.knowledge_base_id == conversation.knowledge_base_id,
            )
        )
        if task:
            source = await session.get(ConversationTurn, task.source_turn_id)
            if source:
                source_conversation = await owned_conversation(
                    session, source.conversation_id, conversation.owner_id
                )
                if source_conversation.knowledge_base_id == conversation.knowledge_base_id:
                    goal = {"turn_id": str(source.id), "user_input": source.question}
    payload = task_alignment.inputs_for_turn(turn, goal)
    if task_id and goal is None and not (resolution.get("intent_context") or {}).get("active_task"):
        payload["goal_basis"] = "unavailable_task_goal"
    return payload


async def claim(session, cid, tid, user_id, request_id):
    conversation = await owned_conversation(session, cid, user_id, lock=True)
    turn = await session.get(ConversationTurn, tid)
    if turn is None or turn.conversation_id != cid:
        raise HTTPException(404, "Turn not found")
    if turn.status == "pending":
        raise HTTPException(409, "本轮仍在处理，请等待回答结束后核对。")
    payload = await saved_input(session, conversation, turn)
    policy = task_alignment.contract(get_settings())
    fingerprint = task_alignment.digest({"input": payload, "contract": policy})
    resolution = turn.resolution or {}
    rows = [dict(r) for r in resolution.get("task_alignment_checks", [])]
    for row in rows:
        if row["request_id"] == str(request_id):
            if row["fingerprint"] != fingerprint:
                raise HTTPException(409, "同一核对标识的输入或协议已改变，请刷新后重试。")
            await session.commit()
            return row, payload, False, True
    if cached := next(
        (
            r
            for r in reversed(rows)
            if r["fingerprint"] == fingerprint
            and r["status"] in {"complete", "partial", "not_applicable", "budget_exceeded"}
        ),
        None,
    ):
        await session.commit()
        return cached, payload, False, True
    now = utc_now()
    for row in rows:
        if row["status"] == "running":
            from datetime import datetime

            if datetime.fromisoformat(row["lease_until"]) > now:
                raise HTTPException(409, "本轮任务核对仍在运行，未重复调用模型。")
            row.update(
                status="interrupted", outcome="unknown", reason="lease_expired_result_unknown"
            )
    if len(rows) >= 20:
        raise HTTPException(409, "该轮已达到20条核对记录上限，原记录保留。")
    applicable = bool(payload["final_answer"]) and turn.status == "completed"
    missing_goal = payload["goal_basis"] == "unavailable_task_goal"
    row = {
        "request_id": str(request_id),
        "fingerprint": fingerprint,
        "version": task_alignment.VERSION,
        "status": "running"
        if applicable and not missing_goal
        else "unavailable"
        if missing_goal
        else "not_applicable",
        "outcome": "unknown",
        "input_snapshot": payload,
        "contract": policy,
        "input_sha256": task_alignment.digest(payload),
        "created_at": now.isoformat(),
        "lease_until": (
            now + timedelta(seconds=get_settings().eval_judge_timeout_seconds + 60)
        ).isoformat(),
        "result": None,
        "reason": "missing_task_goal"
        if missing_goal
        else None
        if applicable
        else "no_completed_final_answer",
    }
    rows.append(row)
    turn.resolution = {**resolution, "task_alignment_checks": rows}
    await session.commit()  # No row lock/DB transaction during a remote request.
    return row, payload, applicable and not missing_goal, False


async def finish(cid, tid, uid, request_id, fingerprint, result):
    async with SessionLocal() as session:
        await owned_conversation(session, cid, uid, lock=True)
        turn = await session.get(ConversationTurn, tid)
        if turn is None or turn.conversation_id != cid:
            raise HTTPException(404, "Turn not found")
        resolution = turn.resolution or {}
        rows = [dict(r) for r in resolution.get("task_alignment_checks", [])]
        target = next(
            (
                r
                for r in rows
                if r["request_id"] == str(request_id) and r["fingerprint"] == fingerprint
            ),
            None,
        )
        if target is None or target["status"] != "running":
            raise HTTPException(409, "核对已失去执行权，未覆盖新结果。")
        if any(r["created_at"] > target["created_at"] for r in rows):
            raise HTTPException(409, "已有后续核对，迟到结果未覆盖记录。")
        target.update(
            status=result["status"],
            outcome=result["outcome"],
            result=result,
            completed_at=utc_now().isoformat(),
        )
        turn.resolution = {**resolution, "task_alignment_checks": rows}
        await session.commit()
        return target


@router.post("/conversations/{cid}/turns/{tid}/task-review")
async def review_turn(
    cid: UUID, tid: UUID, body: ReviewRequest, user: UserDep, session: SessionDep
):
    row, payload, dispatch, cached = await claim(session, cid, tid, user.id, body.request_id)
    if not dispatch:
        return {"check": row, "cached": cached}
    with scope(get_settings()) as budget:
        try:
            result = await task_alignment.review(payload, get_settings())
        except asyncio.CancelledError:
            result = {
                "status": "interrupted",
                "outcome": "unknown",
                "task_success": None,
                "reason": "request_cancelled_result_unknown",
                "round_budget": budget.snapshot(),
            }
            await asyncio.shield(
                finish(cid, tid, user.id, body.request_id, row["fingerprint"], result)
            )
            raise
        except Exception as exc:
            result = {
                "status": "unavailable",
                "outcome": "unknown",
                "task_success": None,
                "error_type": type(exc).__name__,
            }
        result["round_budget"] = budget.snapshot()
    completed = await finish(cid, tid, user.id, body.request_id, row["fingerprint"], result)
    return {"check": completed, "cached": False}
