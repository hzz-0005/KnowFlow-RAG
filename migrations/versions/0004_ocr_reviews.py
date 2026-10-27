"""Persist isolated page OCR comparisons without modifying retrieval indexes."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_ocr_reviews"
down_revision = "0003_document_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ocr_reviews",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("baseline_fingerprint", sa.String(64), nullable=False),
        sa.Column("result", postgresql.JSONB(), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_ocr_reviews_document_page", "ocr_reviews", ["document_id", "page_number"])


def downgrade() -> None:
    op.drop_table("ocr_reviews")
