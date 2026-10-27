"""Persist chunk budgets and structural roles without rewriting older chunks."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006_chunk_metadata"
down_revision = "0005_document_links"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "chunks",
        sa.Column("processing_metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
    )


def downgrade():
    op.drop_column("chunks", "processing_metadata")
