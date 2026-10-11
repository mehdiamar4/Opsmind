import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import SQLAlchemyError

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://opsmind:opsmind@localhost:5432/opsmind"
)

from app import document_ingestion
from app.document_ingestion import DocumentIngestionError, IngestionResult, ingest_pdf
from app.embeddings import EMBEDDING_DIMENSIONS
from app.main import app


class FakeSession:
    def __init__(self, factory) -> None:
        self.factory = factory
        self.committed = False
        self.rolled_back = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    @contextmanager
    def begin(self):
        try:
            yield self
        except Exception:
            self.rolled_back = True
            raise
        else:
            self.committed = True

    def get(self, model, primary_key):
        if model.__name__ == "User" and self.factory.user_exists:
            return SimpleNamespace(id=primary_key)
        return None

    def scalar(self, statement):
        return self.factory.duplicate_id

    def add(self, value):
        self.factory.added.append(value)
        if value.__class__.__name__ == "Document":
            value.id = 21

    def add_all(self, values):
        self.factory.added.extend(values)

    def flush(self):
        if self.factory.fail_write:
            raise SQLAlchemyError("simulated database write failure")


class FakeSessionFactory:
    def __init__(self, *, user_exists=True, duplicate_id=None, fail_write=False) -> None:
        self.user_exists = user_exists
        self.duplicate_id = duplicate_id
        self.fail_write = fail_write
        self.added = []
        self.sessions = []

    def __call__(self):
        session = FakeSession(self)
        self.sessions.append(session)
        return session


@pytest.fixture
def extracted_pdf(monkeypatch):
    monkeypatch.setattr(document_ingestion, "_open_pdf", lambda content: object())
    monkeypatch.setattr(
        document_ingestion,
        "_extract_pages",
        lambda reader: [(1, "Introduction\n\nA short policy paragraph.")],
    )


def fake_embedder(texts):
    return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]


def make_text_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped_text}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode("ascii"))
        data.extend(obj)
        data.extend(b"\nendobj\n")
    xref_offset = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    data.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(data)


def test_ingestion_success_persists_ready_document_and_chunks(extracted_pdf) -> None:
    sessions = FakeSessionFactory()

    result = ingest_pdf(
        content=b"%PDF-test content",
        filename="/private/path/policy.pdf",
        uploaded_by=7,
        session_factory=sessions,
        embedder=fake_embedder,
    )

    assert result == IngestionResult(
        document_id=21,
        filename="policy.pdf",
        status="ready",
        chunk_count=1,
        content_hash="7db2deee98d33870f7c967e17ebf4663be79567aa81e0e14498cf845106aceb0",
    )
    stored_document = next(item for item in sessions.added if item.__class__.__name__ == "Document")
    stored_chunk = next(item for item in sessions.added if item.__class__.__name__ == "DocumentChunk")
    assert stored_document.status == "ready"
    assert stored_chunk.document_id == 21
    assert stored_chunk.page_number == 1
    assert stored_chunk.section_title == "Introduction"
    assert len(stored_chunk.embedding) == EMBEDDING_DIMENSIONS
    assert sessions.sessions[-1].committed


@pytest.mark.parametrize(
    ("content", "expected_status"),
    [(b"", 400), (b"not a pdf", 415)],
)
def test_ingestion_rejects_empty_or_non_pdf(content, expected_status) -> None:
    with pytest.raises(DocumentIngestionError) as error:
        ingest_pdf(content=content, filename="file.pdf", uploaded_by=7)

    assert error.value.status_code == expected_status


def test_pdf_text_is_extracted_page_by_page() -> None:
    reader = document_ingestion._open_pdf(make_text_pdf("Extracted policy text"))

    assert document_ingestion._extract_pages(reader) == [(1, "Extracted policy text")]


def test_ingestion_rejects_unknown_user_before_embedding(extracted_pdf) -> None:
    sessions = FakeSessionFactory(user_exists=False)

    with pytest.raises(DocumentIngestionError) as error:
        ingest_pdf(
            content=b"%PDF-test content",
            filename="file.pdf",
            uploaded_by=7,
            session_factory=sessions,
            embedder=lambda texts: pytest.fail("embedding must not be called"),
        )

    assert error.value.status_code == 404
    assert len(sessions.sessions) == 1


def test_ingestion_rejects_duplicate_before_embedding(extracted_pdf) -> None:
    sessions = FakeSessionFactory(duplicate_id=9)

    with pytest.raises(DocumentIngestionError) as error:
        ingest_pdf(
            content=b"%PDF-test content",
            filename="file.pdf",
            uploaded_by=7,
            session_factory=sessions,
            embedder=lambda texts: pytest.fail("embedding must not be called"),
        )

    assert error.value.status_code == 409
    assert len(sessions.sessions) == 1


def test_embedding_failure_does_not_write_document(extracted_pdf) -> None:
    sessions = FakeSessionFactory()

    def fail_embedding(texts):
        raise RuntimeError("provider failure")

    with pytest.raises(DocumentIngestionError) as error:
        ingest_pdf(
            content=b"%PDF-test content",
            filename="file.pdf",
            uploaded_by=7,
            session_factory=sessions,
            embedder=fail_embedding,
        )

    assert error.value.status_code == 502
    assert error.value.detail == "The embedding provider request failed."
    assert sessions.added == []
    assert len(sessions.sessions) == 1


def test_storage_failure_rolls_back_document_and_chunks(extracted_pdf) -> None:
    sessions = FakeSessionFactory(fail_write=True)

    with pytest.raises(DocumentIngestionError) as error:
        ingest_pdf(
            content=b"%PDF-test content",
            filename="file.pdf",
            uploaded_by=7,
            session_factory=sessions,
            embedder=fake_embedder,
        )

    assert error.value.status_code == 503
    assert sessions.sessions[-1].rolled_back


def test_documents_endpoint_returns_ingestion_result(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.routes.documents.ingest_pdf",
        lambda **kwargs: IngestionResult(
            document_id=21,
            filename="policy.pdf",
            status="ready",
            chunk_count=2,
            content_hash="a" * 64,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            data={"uploaded_by": "7"},
            files={"file": ("policy.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 201
    assert response.json() == {
        "document_id": 21,
        "filename": "policy.pdf",
        "status": "ready",
        "chunk_count": 2,
        "content_hash": "a" * 64,
    }


def test_documents_endpoint_maps_duplicate_to_conflict(monkeypatch) -> None:
    def duplicate(**kwargs):
        raise DocumentIngestionError(409, "A document with this content already exists.")

    monkeypatch.setattr("app.routes.documents.ingest_pdf", duplicate)

    with TestClient(app) as client:
        response = client.post(
            "/documents",
            data={"uploaded_by": "7"},
            files={"file": ("policy.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "A document with this content already exists."}


@pytest.mark.parametrize(
    ("content", "expected_status"),
    [(b"", 400), (b"plain text", 415)],
)
def test_documents_endpoint_rejects_empty_and_non_pdf(content, expected_status) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            data={"uploaded_by": "7"},
            files={"file": ("upload.pdf", content, "application/pdf")},
        )

    assert response.status_code == expected_status


def test_health_and_version_endpoints_remain_available() -> None:
    with TestClient(app) as client:
        health = client.get("/health")
        version = client.get("/api/version")

    assert health.status_code == 200
    assert health.json() == {"status": "healthy"}
    assert version.status_code == 200
    assert version.json() == {"version": "0.1.0"}
