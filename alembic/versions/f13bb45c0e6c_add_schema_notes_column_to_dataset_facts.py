"""add schema_notes column to dataset_facts

Revision ID: f13bb45c0e6c
Revises: e0465ef53ed7
Create Date: 2026-07-23 17:28:53.670870

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'f13bb45c0e6c'
down_revision: Union[str, Sequence[str], None] = 'e0465ef53ed7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('dataset_facts', sa.Column('schema_notes', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('dataset_facts', 'schema_notes')
