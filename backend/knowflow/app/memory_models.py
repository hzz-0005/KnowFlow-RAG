"""Durable task checkpoints and owner-scoped, explicitly activated memories."""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from knowflow.app.models import Base, utc_now


class TaskMemory(Base):
    __tablename__ = "task_memories"
    __table_args__ = (UniqueConstraint("source_turn_id", name="uq_task_source_turn"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    knowledge_base_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )
    source_turn_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation_turns.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(String(120))
    state: Mapped[dict] = mapped_column(JSONB, default=dict)
    status: Mapped[str] = mapped_column(String(30), default="ready")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    active_turn_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class TaskAttempt(Base):
    __tablename__ = "task_memory_attempts"
    __table_args__ = (UniqueConstraint("turn_id", name="uq_task_attempt_turn"),)
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("task_memories.id", ondelete="CASCADE"), index=True
    )
    turn_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversation_turns.id", ondelete="CASCADE")
    )
    retry_of_attempt_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="running")
    result: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        UniqueConstraint("owner_id", "namespace", "key", name="uq_memory_namespace_key"),
        Index("ix_memory_recall", "owner_id", "namespace", "status", "kind"),
    )
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    knowledge_base_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=True
    )
    namespace: Mapped[str] = mapped_column(String(50))
    key: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="candidate")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[dict] = mapped_column(JSONB, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
