"""Task checkpoints and explicit owner/namespace memory. Existing histories untouched."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0012_task_user_memories"
down_revision = "0011_conversations"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "task_memories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "owner_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "knowledge_base_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_turn_id",
            sa.Uuid(),
            sa.ForeignKey("conversation_turns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("state", JSONB(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("active_turn_id", sa.Uuid()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("source_turn_id", name="uq_task_source_turn"),
    )
    op.create_index("ix_task_memories_owner_id", "task_memories", ["owner_id"])
    op.create_index("ix_task_memories_knowledge_base_id", "task_memories", ["knowledge_base_id"])
    op.create_table(
        "task_memory_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "task_id",
            sa.Uuid(),
            sa.ForeignKey("task_memories.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "turn_id",
            sa.Uuid(),
            sa.ForeignKey("conversation_turns.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("retry_of_attempt_id", sa.Uuid()),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("result", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("turn_id", name="uq_task_attempt_turn"),
    )
    op.create_index("ix_task_memory_attempts_task_id", "task_memory_attempts", ["task_id"])
    op.create_table(
        "user_memories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "owner_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column(
            "knowledge_base_id", sa.Uuid(), sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE")
        ),
        sa.Column("namespace", sa.String(50), nullable=False),
        sa.Column("key", sa.String(100), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("source", JSONB(), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("owner_id", "namespace", "key", name="uq_memory_namespace_key"),
    )
    op.create_index(
        "ix_memory_recall", "user_memories", ["owner_id", "namespace", "status", "kind"]
    )


def downgrade():
    op.drop_table("user_memories")
    op.drop_table("task_memory_attempts")
    op.drop_table("task_memories")
