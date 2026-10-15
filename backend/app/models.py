from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector

from app.database import Base


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("role IN ('employee', 'admin')", name="ck_users_role"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    role: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_uploaded_by", "uploaded_by"),
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed')",
            name="ck_documents_status",
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    mime_type: Mapped[str] = mapped_column(String, nullable=False)
    uploaded_by: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", name="fk_documents_uploaded_by_users", ondelete="RESTRICT"),
        nullable=False,
    )
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    document_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", name="fk_document_chunks_document_id_documents", ondelete="CASCADE"),
        primary_key=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section_title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AIUsageEvent(Base):
    __tablename__ = "ai_usage_events"
    __table_args__ = (
        CheckConstraint(
            "event_type IN ('llm', 'embedding')",
            name="ck_ai_usage_events_event_type",
        ),
        Index("ix_ai_usage_events_request_id", "request_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    event_type: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default="llm"
    )
    request_id: Mapped[Optional[UUID]] = mapped_column(Uuid(as_uuid=True), nullable=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    request_success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    estimated_cost: Mapped[Optional[Decimal]] = mapped_column(Numeric(12, 8), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
