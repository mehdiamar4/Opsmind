"""add document chunks and ingestion metadata

Revision ID: c35f3a0dc859
Revises: a586b260179d
Create Date: 2026-10-10 21:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


revision: str = "c35f3a0dc859"
down_revision: Union[str, None] = "a586b260179d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "status",
            sa.String(length=20),
            server_default="pending",
            nullable=False,
        ),
    )
    op.create_unique_constraint(
        "uq_documents_content_hash", "documents", ["content_hash"]
    )
    op.create_check_constraint(
        "ck_documents_status",
        "documents",
        "status IN ('pending', 'processing', 'ready', 'failed')",
    )

    op.create_table(
        "document_chunks",
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("section_title", sa.Text(), nullable=True),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_chunks_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("document_id", "chunk_index"),
    )


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_constraint("ck_documents_status", "documents", type_="check")
    op.drop_constraint("uq_documents_content_hash", "documents", type_="unique")
    op.drop_column("documents", "status")
    op.drop_column("documents", "content_hash")
    # Keep the shared PostgreSQL extension installed on downgrade.
