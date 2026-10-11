from types import SimpleNamespace

from app import embeddings


def test_embedding_calls_are_batched_with_one_model(monkeypatch) -> None:
    calls = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            values = [
                SimpleNamespace(index=index, embedding=[float(index)] * 1536)
                for index, _ in enumerate(kwargs["input"])
            ]
            return SimpleNamespace(data=list(reversed(values)))

    fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(embeddings, "OpenAI", lambda **kwargs: fake_client)

    vectors = embeddings.create_embeddings([f"chunk {index}" for index in range(65)])

    assert [len(call["input"]) for call in calls] == [64, 1]
    assert {call["model"] for call in calls} == {"text-embedding-3-small"}
    assert {call["dimensions"] for call in calls} == {1536}
    assert [vector[0] for vector in vectors[:3]] == [0.0, 1.0, 2.0]
    assert len(vectors) == 65
