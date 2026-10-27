"""Add an English stemming search index without reingesting source documents."""

from alembic import op

revision = "0009_english_search"
down_revision = "0008_relation_review"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "CREATE INDEX ix_chunks_search_english ON chunks "
        "USING gin (to_tsvector('english'::regconfig, search_text))"
    )


def downgrade():
    op.drop_index("ix_chunks_search_english", table_name="chunks")
