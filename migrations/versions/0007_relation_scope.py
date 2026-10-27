"""Preserve qualified facts with identical endpoints without collapsing their scopes."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007_relation_scope"
down_revision = "0006_chunk_metadata"
branch_labels = None
depends_on = None
NAME = "uq_relation_chunk_endpoints_predicate"
KEYS = ["source_chunk_id", "source_entity_id", "target_entity_id", "predicate"]


def upgrade():
    op.add_column(
        "relations",
        sa.Column("qualifiers", postgresql.JSONB(), nullable=False, server_default="{}"),
    )
    op.add_column(
        "relations", sa.Column("scope_key", sa.String(64), nullable=False, server_default="")
    )
    op.drop_constraint(NAME, "relations", type_="unique")
    op.create_unique_constraint(NAME, "relations", [*KEYS, "scope_key"])


def downgrade():
    count = op.get_bind().scalar(
        sa.text("""
        SELECT count(*) FROM (
          SELECT 1 FROM relations
          GROUP BY source_chunk_id, source_entity_id, target_entity_id, predicate
          HAVING count(*) > 1
        ) duplicates
    """)
    )
    if count:
        raise RuntimeError(
            "Scoped facts would collide in the old schema; preserve them before downgrade"
        )
    op.drop_constraint(NAME, "relations", type_="unique")
    op.create_unique_constraint(NAME, "relations", KEYS)
    op.drop_column("relations", "scope_key")
    op.drop_column("relations", "qualifiers")
