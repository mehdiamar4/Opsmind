import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Optional

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.chunking import chunk_pages
from app.embeddings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingConfigurationError,
    create_embeddings,
)


MAX_CHUNK_TOKENS = 800


class DocumentIngestionError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class IngestionResult:
    document_id: int
    filename: str
    status: str
    chunk_count: int
    content_hash: str


def _open_pdf(content: bytes) -> PdfReader:
    if not content:
        raise DocumentIngestionError(400, "The uploaded file is empty.")
    if b"%PDF-" not in content[:1024]:
        raise DocumentIngestionError(415, "The uploaded file is not a PDF.")
    try:
        reader = PdfReader(BytesIO(content), strict=False)
    except Exception:
        raise DocumentIngestionError(415, "The uploaded file is not a valid PDF.") from None
    if reader.is_encrypted:
        raise DocumentIngestionError(422, "Encrypted PDFs are not supported.")
    return reader


def _extract_pages(reader: PdfReader) -> list[tuple[int, str]]:
    pages: list[tuple[int, str]] = []
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append((page_number, page.extract_text() or ""))
    except Exception:
        raise DocumentIngestionError(422, "Text could not be extracted from this PDF.") from None
    if not any(text.strip() for _, text in pages):
        raise DocumentIngestionError(422, "The PDF contains no extractable text.")
    return pages


def ingest_pdf(
    *,
    content: bytes,
    filename: str,
    uploaded_by: int,
    session_factory: Optional[Callable[[], object]] = None,
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
) -> IngestionResult:
    reader = _open_pdf(content)
    content_hash = hashlib.sha256(content).hexdigest()

    from app.database import SessionLocal
    from app.models import Document, DocumentChunk, User

    session_factory = session_factory or SessionLocal
    embedder = embedder or create_embeddings

    try:
        with session_factory() as session:
            if session.get(User, uploaded_by) is None:
                raise DocumentIngestionError(404, "The uploading user was not found.")
            duplicate_id = session.scalar(
                select(Document.id).where(Document.content_hash == content_hash)
            )
            if duplicate_id is not None:
                raise DocumentIngestionError(409, "A document with this content already exists.")
    except DocumentIngestionError:
        raise
    except SQLAlchemyError:
        raise DocumentIngestionError(503, "The database is unavailable.") from None

    chunks = chunk_pages(_extract_pages(reader), max_tokens=MAX_CHUNK_TOKENS)
    if not chunks:
        raise DocumentIngestionError(422, "The PDF contains no extractable text.")

    try:
        vectors = embedder([chunk.content for chunk in chunks])
    except EmbeddingConfigurationError:
        raise DocumentIngestionError(503, "The embedding service is not configured.") from None
    except Exception:
        raise DocumentIngestionError(502, "The embedding provider request failed.") from None

    if len(vectors) != len(chunks) or any(
        len(vector) != EMBEDDING_DIMENSIONS for vector in vectors
    ):
        raise DocumentIngestionError(502, "The embedding provider returned invalid vectors.")

    safe_filename = (filename or "upload.pdf").replace("\\", "/").rsplit("/", 1)[-1]
    if not safe_filename:
        safe_filename = "upload.pdf"

    try:
        with session_factory() as session:
            with session.begin():
                document = Document(
                    filename=safe_filename,
                    mime_type="application/pdf",
                    uploaded_by=uploaded_by,
                    content_hash=content_hash,
                    status="processing",
                )
                session.add(document)
                session.flush()
                session.add_all(
                    [
                        DocumentChunk(
                            document_id=document.id,
                            chunk_index=index,
                            content=chunk.content,
                            page_number=chunk.page_number,
                            section_title=chunk.section_title,
                            embedding=vector,
                        )
                        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                    ]
                )
                document.status = "ready"
                session.flush()
            return IngestionResult(
                document_id=document.id,
                filename=document.filename,
                status=document.status,
                chunk_count=len(chunks),
                content_hash=content_hash,
            )
    except IntegrityError:
        try:
            with session_factory() as session:
                if session.scalar(
                    select(Document.id).where(Document.content_hash == content_hash)
                ) is not None:
                    raise DocumentIngestionError(
                        409, "A document with this content already exists."
                    )
                if session.get(User, uploaded_by) is None:
                    raise DocumentIngestionError(404, "The uploading user was not found.")
        except DocumentIngestionError:
            raise
        except SQLAlchemyError:
            raise DocumentIngestionError(503, "The database is unavailable.") from None
        raise DocumentIngestionError(503, "The document could not be stored.") from None
    except SQLAlchemyError:
        raise DocumentIngestionError(503, "The document could not be stored.") from None
