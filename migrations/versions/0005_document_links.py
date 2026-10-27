"""Explicit document references with auditable unresolved and ambiguous states."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005_document_links"
down_revision = "0004_ocr_reviews"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_links",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_block_key", sa.String(200), nullable=False),
        sa.Column("target_block_key", sa.String(200)),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("label", sa.String(80), nullable=False),
        sa.Column("source_quote", sa.String(200), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("candidate_keys", postgresql.JSONB(), nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id", "source_block_key"],
            ["document_blocks.document_id", "document_blocks.block_key"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_id", "target_block_key"],
            ["document_blocks.document_id", "document_blocks.block_key"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "document_id", "source_block_key", "start_offset", name="uq_document_link_source_offset"
        ),
    )
    op.create_index(
        "ix_document_links_source", "document_links", ["document_id", "source_block_key"]
    )


def downgrade() -> None:
    op.drop_table("document_links")
