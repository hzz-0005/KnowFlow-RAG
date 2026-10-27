"""Retain semantic review outcomes separately from extraction confidence."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008_relation_review"
down_revision = "0007_relation_scope"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "relations",
        sa.Column("semantic_status", sa.String(20), nullable=False, server_default="unreviewed"),
    )
    op.add_column(
        "relations",
        sa.Column("semantic_review", postgresql.JSONB(), nullable=False, server_default="{}"),
    )


def downgrade():
    op.drop_column("relations", "semantic_review")
    op.drop_column("relations", "semantic_status")
