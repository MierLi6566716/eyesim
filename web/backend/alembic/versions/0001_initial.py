"""initial schema: model_versions, cases

Revision ID: 0001
Revises:
Create Date: 2026-07-28
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tag", sa.String(length=64), nullable=False),
        sa.Column("checkpoint_path", sa.String(length=512), nullable=False),
        sa.Column("val_mae_m", sa.Float(), nullable=False),
        sa.Column("val_mae_j0", sa.Float(), nullable=False),
        sa.Column("val_mae_j45", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_model_versions_tag", "model_versions", ["tag"], unique=True)

    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "mode", sa.Enum("simulate", "diagnose", name="casemode"), nullable=False
        ),
        sa.Column(
            "model_version_id",
            sa.Integer(),
            sa.ForeignKey("model_versions.id"),
            nullable=True,
        ),
        sa.Column("source_image_url", sa.String(length=1024), nullable=False),
        sa.Column("blurred_image_url", sa.String(length=1024), nullable=True),
        sa.Column("predistorted_image_url", sa.String(length=1024), nullable=True),
        sa.Column("S", sa.Float(), nullable=True),
        sa.Column("C", sa.Float(), nullable=True),
        sa.Column("axis", sa.Float(), nullable=True),
        sa.Column("M", sa.Float(), nullable=False),
        sa.Column("J0", sa.Float(), nullable=False),
        sa.Column("J45", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("is_public", sa.Boolean(), nullable=False),
        sa.Column("share_slug", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cases_mode", "cases", ["mode"])
    op.create_index("ix_cases_share_slug", "cases", ["share_slug"], unique=True)
    op.create_index("ix_cases_created_at", "cases", ["created_at"])


def downgrade() -> None:
    op.drop_table("cases")
    op.drop_table("model_versions")
