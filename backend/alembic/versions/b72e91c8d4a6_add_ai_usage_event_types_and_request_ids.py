"""add event type and request ID to AI usage events

Revision ID: b72e91c8d4a6
Revises: c35f3a0dc859
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "b72e91c8d4a6"
down_revision: Union[str, None] = "c35f3a0dc859"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "ai_usage_events",
        sa.Column("event_type", sa.String(length=16), server_default="llm", nullable=False),
    )
    op.add_column(
        "ai_usage_events",
        sa.Column("request_id", sa.Uuid(as_uuid=True), nullable=True),
    )
    op.create_check_constraint(
        "ck_ai_usage_events_event_type",
        "ai_usage_events",
        "event_type IN ('llm', 'embedding')",
    )
    op.create_index(
        "ix_ai_usage_events_request_id", "ai_usage_events", ["request_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_ai_usage_events_request_id", table_name="ai_usage_events")
    op.drop_constraint(
        "ck_ai_usage_events_event_type", "ai_usage_events", type_="check"
    )
    op.drop_column("ai_usage_events", "request_id")
    op.drop_column("ai_usage_events", "event_type")
