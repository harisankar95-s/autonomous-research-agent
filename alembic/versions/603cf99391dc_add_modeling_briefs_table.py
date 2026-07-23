"""add modeling_briefs table

Revision ID: 603cf99391dc
Revises: f13bb45c0e6c
Create Date: 2026-07-23 18:33:57.043293

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '603cf99391dc'
down_revision: Union[str, Sequence[str], None] = 'f13bb45c0e6c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('modeling_briefs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.String(), nullable=False),
    sa.Column('label_status', sa.String(), nullable=False),
    sa.Column('label_column', sa.String(), nullable=True),
    sa.Column('label_notes', sa.String(), nullable=False),
    sa.Column('feature_set', sa.JSON(), nullable=False),
    sa.Column('preprocessing_rules', sa.JSON(), nullable=False),
    sa.Column('validation_strategy', sa.String(), nullable=False),
    sa.Column('confidence', sa.Float(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('dataset_id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('modeling_briefs')
