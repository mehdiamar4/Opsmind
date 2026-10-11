import tiktoken
import pytest


class LocalTestEncoding:
    """Deterministic offline tokenizer for chunk-boundary unit tests."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


@pytest.fixture(autouse=True)
def use_local_test_encoding(monkeypatch):
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: LocalTestEncoding())
