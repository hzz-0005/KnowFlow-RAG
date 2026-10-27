"""Explicit memory lifecycle and resumable read-only retrieval tasks."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from knowflow.app.api import UserDep, owned_knowledge_base
from knowflow.app.conversation_api import owned_conversation
from knowflow.app.memory_models import TaskAttempt, TaskMemory, UserMemory
from knowflow.app.memory_store import STYLE, namespace, visible
from knowflow.app.models import ConversationTurn, utc_now
from knowflow.app.security import SessionDep

router = APIRouter(prefix="/api/v1", tags=["memory"])


class MemoryWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["user", "knowledge_base"] = "knowledge_base"
    kind: Literal["preference", "alias", "note"]
    key: str = Field(min_length=2, max_length=100)
    content: str = Field(min_length=1, max_length=800)
    status: Literal["candidate", "active", "disabled"] = "candidate"
    expires_at: datetime | None = None
    source_turn_id: UUID | None = None
    source_quote: str | None = Field(None, max_length=800)
    expected_revision: int | None = Field(None, ge=0)

    @field_validator("key", "content")
    @classmethod
    def strip_text(cls, value):
        if not value.strip():
            raise ValueError("Empty memory text")
        return value.strip()


def memory_view(row):
    return {
        k: getattr(row, k)
        for k in (
            "id",
            "namespace",
            "knowledge_base_id",
            "kind",
            "key",
            "content",
            "status",
            "revision",
            "source",
            "expires_at",
            "created_at",
            "updated_at",
        )
    }


async def memory_source(session, body, user_id, kb_id):
    if body.kind == "preference" and (body.key != "response_style" or body.content not in STYLE):
        raise HTTPException(422, "表达偏好仅支持 response_style: concise / standard / detailed。")
    if body.expires_at is not None and (
        body.expires_at.tzinfo is None or body.expires_at <= utc_now()
    ):
        raise HTTPException(422, "有效期须为带时区的未来时间。")
    source = {"type": "explicit_user_entry", "verification": "user_asserted_not_document_verified"}
    if body.source_turn_id:
        turn = await session.get(ConversationTurn, body.source_turn_id)
        if turn is None:
            raise HTTPException(404, "Source turn not found")
        conv = await owned_conversation(session, turn.conversation_id, user_id)
        if conv.knowledge_base_id != kb_id:
            raise HTTPException(404, "Source turn not found")
        if not body.source_quote or body.source_quote not in turn.question:
            raise HTTPException(422, "来源必须是该轮用户原话中的连续节选，不能用模型回答充当来源。")
        source.update(type="user_turn_quote", turn_id=str(turn.id), quote=body.source_quote)
    elif body.source_quote is not None:
        raise HTTPException(422, "来源节选需要轮次ID。")
    return source


@router.get("/knowledge-bases/{kb_id}/memories")
async def list_memories(
    kb_id: UUID, user: UserDep, session: SessionDep, offset: int = Query(0, ge=0)
):
    await owned_knowledge_base(session, kb_id, user.id)
    rows = list(
        await session.scalars(
            select(UserMemory)
            .where(visible(user.id, kb_id))
            .order_by(UserMemory.updated_at.desc(), UserMemory.id)
            .offset(offset)
            .limit(51)
        )
    )
    return {
        "items": [memory_view(r) for r in rows[:50]],
        "has_more": len(rows) > 50,
        "next_offset": offset + 50,
    }


@router.post("/knowledge-bases/{kb_id}/memories", status_code=201)
async def create_memory(kb_id: UUID, body: MemoryWrite, user: UserDep, session: SessionDep):
    await owned_knowledge_base(session, kb_id, user.id)
    source = await memory_source(session, body, user.id, kb_id)
    row = UserMemory(
        owner_id=user.id,
        knowledge_base_id=None if body.scope == "user" else kb_id,
        namespace=namespace(kb_id, body.scope),
        kind=body.kind,
        key=body.key,
        content=body.content,
        status=body.status,
        source=source,
        expires_at=body.expires_at,
    )
    session.add(row)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "同一命名空间已存在此键，请选择更正，不覆盖旧值。") from None
    return memory_view(row)


async def owned_memory(session, kb_id, mid, uid):
    await owned_knowledge_base(session, kb_id, uid)
    row = await session.scalar(
        select(UserMemory).where(visible(uid, kb_id), UserMemory.id == mid).with_for_update()
    )
    if row is None:
        raise HTTPException(404, "Memory not found")
    return row


@router.put("/knowledge-bases/{kb_id}/memories/{mid}")
async def update_memory(
    kb_id: UUID, mid: UUID, body: MemoryWrite, user: UserDep, session: SessionDep
):
    row = await owned_memory(session, kb_id, mid, user.id)
    if body.expected_revision != row.revision:
        raise HTTPException(409, "记忆已更新，请刷新后再更正。")
    if namespace(kb_id, body.scope) != row.namespace:
        raise HTTPException(422, "更正不能移动命名空间，请另建记忆。")
    row.source = await memory_source(session, body, user.id, kb_id)
    row.kind, row.key, row.content = body.kind, body.key, body.content
    row.status, row.expires_at = body.status, body.expires_at
    row.revision += 1
    row.updated_at = utc_now()
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(409, "该键已存在，未覆盖。") from None
    return memory_view(row)


@router.delete("/knowledge-bases/{kb_id}/memories/{mid}")
async def delete_memory(
    kb_id: UUID, mid: UUID, user: UserDep, session: SessionDep, expected_revision: int = Query(ge=0)
):
    row = await owned_memory(session, kb_id, mid, user.id)
    if row.revision != expected_revision:
        raise HTTPException(409, "记忆已更新，请刷新后删除。")
    await session.delete(row)
    await session.commit()
    return {
        "deleted": True,
        "note": "后续请求不再召回；已经发送的请求、历史对话与用户原话不会被追溯改写。",
    }


class SaveTask(BaseModel):
    source_turn_id: UUID


def task_view(row):
    return {
        k: getattr(row, k)
        for k in (
            "id",
            "knowledge_base_id",
            "source_turn_id",
            "title",
            "status",
            "revision",
            "active_turn_id",
            "state",
            "created_at",
            "updated_at",
        )
    }


@router.get("/knowledge-bases/{kb_id}/tasks")
async def list_tasks(kb_id: UUID, user: UserDep, session: SessionDep, offset: int = Query(0, ge=0)):
    await owned_knowledge_base(session, kb_id, user.id)
    rows = list(
        await session.scalars(
            select(TaskMemory)
            .where(TaskMemory.owner_id == user.id, TaskMemory.knowledge_base_id == kb_id)
            .order_by(TaskMemory.updated_at.desc(), TaskMemory.id)
            .offset(offset)
            .limit(51)
        )
    )
    return {
        "items": [task_view(r) for r in rows[:50]],
        "has_more": len(rows) > 50,
        "next_offset": offset + 50,
    }


@router.post("/knowledge-bases/{kb_id}/tasks", status_code=201)
async def save_task(kb_id: UUID, body: SaveTask, user: UserDep, session: SessionDep):
    await owned_knowledge_base(session, kb_id, user.id)
    turn = await session.get(ConversationTurn, body.source_turn_id)
    if turn is None:
        raise HTTPException(404, "Turn not found")
    conv = await owned_conversation(session, turn.conversation_id, user.id, lock=True)
    if conv.knowledge_base_id != kb_id:
        raise HTTPException(404, "Turn not found")
    if turn.status == "pending":
        raise HTTPException(409, "请等待本轮结束后保存检查点。")
    row = await session.scalar(select(TaskMemory).where(TaskMemory.source_turn_id == turn.id))
    if row is None:
        from knowflow.app.task_memory import checkpoint

        row = checkpoint(conv, turn)
        session.add(row)
        await session.commit()
    return task_view(row)


async def owned_task(session, tid, uid, kb_id=None, *, lock=False):
    stmt = select(TaskMemory).where(TaskMemory.id == tid, TaskMemory.owner_id == uid)
    row = await session.scalar(stmt.with_for_update() if lock else stmt)
    if row is None or (kb_id is not None and row.knowledge_base_id != kb_id):
        raise HTTPException(404, "Task not found")
    await owned_knowledge_base(session, row.knowledge_base_id, uid)
    return row


@router.get("/tasks/{tid}")
async def get_task(tid: UUID, user: UserDep, session: SessionDep):
    row = await owned_task(session, tid, user.id)
    attempts = list(
        await session.scalars(
            select(TaskAttempt)
            .where(TaskAttempt.task_id == tid)
            .order_by(TaskAttempt.created_at, TaskAttempt.id)
        )
    )
    return {
        **task_view(row),
        "attempts": [
            {
                k: getattr(a, k)
                for k in (
                    "id",
                    "turn_id",
                    "retry_of_attempt_id",
                    "status",
                    "result",
                    "created_at",
                    "finished_at",
                )
            }
            for a in attempts
        ],
    }


class CancelTask(BaseModel):
    expected_revision: int = Field(ge=0)


@router.post("/tasks/{tid}/cancel")
async def cancel_task(tid: UUID, body: CancelTask, user: UserDep, session: SessionDep):
    row = await owned_task(session, tid, user.id, lock=True)
    if row.revision != body.expected_revision:
        raise HTTPException(409, "任务已更新，请刷新后操作。")
    row.status, row.active_turn_id = "cancelled", None
    row.revision += 1
    row.updated_at = utc_now()
    await session.commit()
    return task_view(row)
