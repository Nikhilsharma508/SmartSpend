"""add model-selection fields to scenarios

Revision ID: c1f4a2b7d901
Revises: 377cb2ffba37
Create Date: 2026-10-20 12:00:00.000000

Nullable columns: benchmarks made before the multi-model era keep reading
(``model_name`` NULL = answered by the pre-catalog single-model app).
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c1f4a2b7d901'
down_revision: str | Sequence[str] | None = '377cb2ffba37'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'scenarios',
        sa.Column('model_name', sa.String(length=64), nullable=True),
    )
    op.add_column(
        'scenarios',
        sa.Column('model_selector', sa.String(length=16), nullable=True),
    )
    op.add_column(
        'scenarios',
        sa.Column('score_kind', sa.String(length=40), nullable=True),
    )
    op.add_column(
        'scenarios',
        sa.Column('research_mode', sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('scenarios', 'research_mode')
    op.drop_column('scenarios', 'score_kind')
    op.drop_column('scenarios', 'model_selector')
    op.drop_column('scenarios', 'model_name')
