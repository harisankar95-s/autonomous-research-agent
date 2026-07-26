"""add columns row_count entity_columns to dataset_facts

Revision ID: b5702b4ccf58
Revises: 1db47f9492f2
Create Date: 2026-07-25 19:20:28.394412

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'b5702b4ccf58'
down_revision: Union[str, Sequence[str], None] = '1db47f9492f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('dataset_facts', sa.Column('columns', sa.JSON(), nullable=True))
    op.add_column('dataset_facts', sa.Column('row_count', sa.Integer(), nullable=True))
    op.add_column('dataset_facts', sa.Column('entity_columns', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('dataset_facts', 'entity_columns')
    op.drop_column('dataset_facts', 'row_count')
    op.drop_column('dataset_facts', 'columns')
