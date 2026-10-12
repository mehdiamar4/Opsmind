from dataclasses import dataclass
from typing import Callable, Optional

from sqlalchemy import select


TOP_K = 5


@dataclass(frozen=True)
class RetrievedChunk:
    document_id: int
    filename: str
    page_number: int
    section_title: Optional[str]
    chunk_index: int
    content: str
    distance: float


def retrieve_top_chunks(
    query_embedding: list[float],
    *,
    top_k: int = TOP_K,
    document_id: Optional[int] = None,
    session_factory: Optional[Callable[[], object]] = None,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("top_k must be positive")

    from app.database import SessionLocal
    from app.models import Document, DocumentChunk

    session_factory = session_factory or SessionLocal
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(
            DocumentChunk.document_id,
            Document.filename,
            DocumentChunk.page_number,
            DocumentChunk.section_title,
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            distance.label("distance"),
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(Document.status == "ready")
        .order_by(distance.asc())
        .limit(top_k)
    )
    if document_id is not None:
        statement = statement.where(DocumentChunk.document_id == document_id)

    with session_factory() as session:
        rows = session.execute(statement).all()

    return [
        RetrievedChunk(
            document_id=row.document_id,
            filename=row.filename,
            page_number=row.page_number,
            section_title=row.section_title,
            chunk_index=row.chunk_index,
            content=row.content,
            distance=float(row.distance),
        )
        for row in rows
    ]
