"""Durable ingestion batches, fenced leases and append-only attempt history."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0010_ingestion_jobs"
down_revision = "0009_english_search"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "ingestion_batches",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "knowledge_base_id",
            sa.Uuid(),
            sa.ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("request_key", sa.String(64), nullable=False),
        sa.Column("request_fingerprint", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("knowledge_base_id", "request_key", name="uq_batch_request_key"),
    )
    op.create_index(
        "ix_ingestion_batches_knowledge_base_id", "ingestion_batches", ["knowledge_base_id"]
    )
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "batch_id",
            sa.Uuid(),
            sa.ForeignKey("ingestion_batches.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="SET NULL")),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("stage", sa.String(24), nullable=False),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("max_attempts", sa.Integer(), nullable=False),
        sa.Column("retryable", sa.Boolean(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("error_code", sa.String(60)),
        sa.Column("error_message", sa.Text()),
        sa.Column("failure_pages", pg.JSONB(), nullable=False),
        sa.Column("failure_pages_known", sa.Boolean(), nullable=False),
        sa.Column("cache_reused", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("batch_id", "ordinal", name="uq_ingestion_job_ordinal"),
    )
    op.create_index("ix_ingestion_jobs_batch_id", "ingestion_jobs", ["batch_id"])
    op.create_index("ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"])
    op.create_index("ix_ingestion_job_claim", "ingestion_jobs", ["status", "available_at"])
    op.create_table(
        "ingestion_attempts",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "job_id",
            sa.Uuid(),
            sa.ForeignKey("ingestion_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("retry_of_attempt_id", sa.Uuid()),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("stage", sa.String(24), nullable=False),
        sa.Column("execution_config", pg.JSONB(), nullable=False),
        sa.Column("error_code", sa.String(60)),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("job_id", "number", name="uq_ingestion_attempt_number"),
    )
    op.create_index("ix_ingestion_attempts_job_id", "ingestion_attempts", ["job_id"])


def downgrade():
    op.drop_table("ingestion_attempts")
    op.drop_table("ingestion_jobs")
    op.drop_table("ingestion_batches")
