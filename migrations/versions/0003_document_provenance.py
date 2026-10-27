"""Persist parsing quality, ordered blocks, and chunk source regions."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0003_document_provenance"
down_revision = "ec96f3749ec5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("parsing_report", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "chunks", sa.Column("source_spans", postgresql.JSONB(), nullable=False, server_default="[]")
    )
    op.create_table(
        "document_blocks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("block_key", sa.String(200), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(60), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("section", sa.String(500)),
        sa.Column("parent_key", sa.String(200)),
        sa.Column("source_spans", postgresql.JSONB(), nullable=False),
        sa.Column("structure", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("document_id", "block_key", name="uq_block_document_key"),
    )
    op.create_index("ix_document_blocks_document_id", "document_blocks", ["document_id"])
    op.create_index("ix_document_blocks_order", "document_blocks", ["document_id", "ordinal"])


def downgrade() -> None:
    op.drop_table("document_blocks")
    op.drop_column("chunks", "source_spans")
    op.drop_column("documents", "parsing_report")
