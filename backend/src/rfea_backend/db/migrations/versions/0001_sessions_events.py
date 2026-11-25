"""Sessions and events

Revision ID: 0001
Revises:
Create Date: 2026-11-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("scene", sa.String(200), nullable=False),
        sa.Column("scene_sha256", sa.String(64), nullable=False),
        sa.Column("params", postgresql.JSONB(), nullable=False),
        sa.Column("model", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_reason", sa.String(20), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(36), sa.ForeignKey("sessions.id"), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("detected_at_s", sa.Float(), nullable=False),
        sa.Column("decided_at_s", sa.Float(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("center_freq_hz", sa.Float(), nullable=False),
        sa.Column("bandwidth_hz", sa.Float(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("label", sa.String(40), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("n_frames", sa.Integer(), nullable=False),
        sa.Column("observation_s", sa.Float(), nullable=False),
        sa.Column("snr_db", sa.Float(), nullable=True),
        sa.Column("bandwidth_out_of_range", sa.Boolean(), nullable=False),
    )
    op.create_index("ix_events_session_id", "events", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_events_session_id", table_name="events")
    op.drop_table("events")
    op.drop_table("sessions")
