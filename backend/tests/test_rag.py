import os
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://opsmind:opsmind@localhost:5432/opsmind"
)

from app import main, rag
from app.main import app
from app.rag import RAGResult
from app.retrieval import RetrievedChunk, retrieve_top_chunks


class FakeRetrievalSession:
    def __init__(self, rows):
        self.rows = rows
        self.statement = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, statement):
        self.statement = statement
        return SimpleNamespace(all=lambda: self.rows)


class FakeRetrievalSessionFactory:
    def __init__(self, rows):
        self.session = FakeRetrievalSession(rows)

    def __call__(self):
        return self.session


def test_retrieval_returns_top_five_in_cosine_order() -> None:
    rows = [
        SimpleNamespace(
            document_id=index,
            filename=f"doc-{index}.pdf",
            page_number=index,
            section_title=None,
            chunk_index=0,
            content=f"content-{index}",
            distance=distance,
        )
        for index, distance in enumerate([0.01, 0.04, 0.10, 0.25, 0.80], start=1)
    ]
    sessions = FakeRetrievalSessionFactory(rows)

    chunks = retrieve_top_chunks([0.0] * 1536, session_factory=sessions)
    compiled = sessions.session.statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert [chunk.distance for chunk in chunks] == [0.01, 0.04, 0.10, 0.25, 0.80]
    assert len(chunks) == 5
    assert "<=>" in sql
    assert "ORDER BY" in sql and "<=>" in sql.split("ORDER BY", 1)[1]
    assert "LIMIT" in sql
    assert "documents.status" in sql
    assert "ready" in compiled.params.values()
    assert 5 in compiled.params.values()


def test_retrieval_can_be_limited_to_one_document() -> None:
    sessions = FakeRetrievalSessionFactory([])

    retrieve_top_chunks([0.0] * 1536, document_id=2, session_factory=sessions)
    compiled = sessions.session.statement.compile(dialect=postgresql.dialect())

    assert "document_chunks.document_id" in str(compiled)
    assert 2 in compiled.params.values()


def test_rag_endpoint_returns_answer_and_database_sources(monkeypatch) -> None:
    chunks = [
        RetrievedChunk(
            document_id=12,
            filename="policy.pdf",
            page_number=3,
            section_title="Access Rules",
            chunk_index=2,
            content="Employees need approval.",
            distance=0.12,
        ),
        RetrievedChunk(
            document_id=20,
            filename="handbook.pdf",
            page_number=1,
            section_title=None,
            chunk_index=0,
            content="Managers review requests.",
            distance=0.19,
        ),
    ]
    monkeypatch.setattr(rag, "create_embeddings", lambda texts: [[0.25] * 1536])
    received_document_ids = []

    def fake_retriever(vector, *, document_id=None):
        received_document_ids.append(document_id)
        return chunks

    monkeypatch.setattr(rag, "retrieve_top_chunks", fake_retriever)

    def fake_answerer(question, context):
        assert question == "Who approves access?"
        assert "[1] filename=policy.pdf" in context
        assert "[2] filename=handbook.pdf" in context
        assert "distance=0.120000" in context
        return "Employees need approval [1], and managers review requests [2]."

    monkeypatch.setattr(rag, "_generate_grounded_answer", fake_answerer)

    with TestClient(app) as client:
        response = client.post(
            "/rag/chat",
            json={"question": "Who approves access?", "document_id": 12},
        )

    assert response.status_code == 200
    assert received_document_ids == [12]
    assert response.json() == {
        "answer": "Employees need approval [1], and managers review requests [2].",
        "sources": [
            {
                "document_id": 12,
                "filename": "policy.pdf",
                "page_number": 3,
                "section_title": "Access Rules",
                "chunk_index": 2,
                "distance": 0.12,
            },
            {
                "document_id": 20,
                "filename": "handbook.pdf",
                "page_number": 1,
                "section_title": None,
                "chunk_index": 0,
                "distance": 0.19,
            },
        ],
    }


def test_rag_without_chunks_returns_safe_answer_without_calling_llm() -> None:
    result = rag.answer_question(
        "What does the policy say?",
        embedder=lambda texts: [[0.0] * 1536],
        retriever=lambda vector: [],
        answerer=lambda question, context: (_ for _ in ()).throw(
            AssertionError("LLM should not be called without retrieved chunks")
        ),
    )

    assert result == RAGResult(
        answer="I don't have document evidence to answer that question yet.",
        sources=[],
    )


def test_grounded_llm_prompt_requires_evidence_and_numbered_citations(monkeypatch) -> None:
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="Answer [1].")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(
        rag,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(responses=FakeResponses()),
    )

    answer = rag._generate_grounded_answer("Question?", "[1] Evidence")

    assert answer == "Answer [1]."
    assert captured["model"] == "test-model"
    assert "only from the provided document context" in captured["instructions"]
    assert "evidence is insufficient" in captured["instructions"]
    assert "[1]" in captured["instructions"]
    assert captured["store"] is False


def test_existing_chat_still_works(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "generate_reply",
        lambda message: main.LLMResult(
            answer="Baseline reply", model="test-model", input_tokens=1, output_tokens=2
        ),
    )
    monkeypatch.setattr(main, "_record_usage", lambda **kwargs: True)

    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "Hello"})

    assert response.status_code == 200
    assert response.json() == {"answer": "Baseline reply", "model": "test-model"}
