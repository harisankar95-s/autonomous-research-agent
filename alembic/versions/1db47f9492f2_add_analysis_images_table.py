"""add analysis_images table

Revision ID: 1db47f9492f2
Revises: 603cf99391dc
Create Date: 2026-07-23 20:59:38.743536

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '1db47f9492f2'
down_revision: Union[str, Sequence[str], None] = '603cf99391dc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('analysis_images',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dataset_id', sa.String(), nullable=False),
    sa.Column('file_path', sa.String(), nullable=False),
    sa.Column('caption', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('analysis_images')
