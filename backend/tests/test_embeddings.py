from types import SimpleNamespace

import pytest

from app import embeddings


def test_embedding_calls_are_batched_with_one_model(monkeypatch) -> None:
    calls = []
    usage_events = []

    class FakeEmbeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            values = [
                SimpleNamespace(index=index, embedding=[float(index)] * 1536)
                for index, _ in enumerate(kwargs["input"])
            ]
            return SimpleNamespace(
                data=list(reversed(values)),
                model="text-embedding-3-small",
                usage=SimpleNamespace(prompt_tokens=len(kwargs["input"]) * 2),
            )

    fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setattr(embeddings, "OpenAI", lambda **kwargs: fake_client)

    vectors = embeddings.create_embeddings(
        [f"chunk {index}" for index in range(65)],
        on_usage=lambda **event: usage_events.append(event),
    )

    assert [len(call["input"]) for call in calls] == [64, 1]
    assert {call["model"] for call in calls} == {"text-embedding-3-small"}
    assert {call["dimensions"] for call in calls} == {1536}
    assert [vector[0] for vector in vectors[:3]] == [0.0, 1.0, 2.0]
    assert len(vectors) == 65
    assert [event["input_tokens"] for event in usage_events] == [128, 2]
    assert all(event["event_type"] == "embedding" for event in usage_events)
    assert all(event["output_tokens"] == 0 for event in usage_events)
    assert all(event["request_success"] for event in usage_events)
    assert all(event["latency_ms"] >= 0 for event in usage_events)


def test_embedding_provider_failure_records_failed_usage(monkeypatch) -> None:
    usage_events = []

    class FailingEmbeddings:
        def create(self, **kwargs):
            raise RuntimeError("provider failed")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        embeddings,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(embeddings=FailingEmbeddings()),
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        embeddings.create_embeddings(
            ["query"], on_usage=lambda **event: usage_events.append(event)
        )

    assert len(usage_events) == 1
    assert usage_events[0]["event_type"] == "embedding"
    assert usage_events[0]["request_success"] is False
    assert usage_events[0]["input_tokens"] == 0
