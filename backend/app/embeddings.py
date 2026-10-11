import os
from typing import Sequence

from openai import OpenAI


EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingConfigurationError(Exception):
    pass


def create_embeddings(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    if not api_key:
        raise EmbeddingConfigurationError
    if not model.startswith("text-embedding-3-"):
        raise EmbeddingConfigurationError

    client = OpenAI(api_key=api_key, max_retries=0)
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = list(texts[start : start + EMBEDDING_BATCH_SIZE])
        response = client.embeddings.create(
            model=model,
            input=batch,
            dimensions=EMBEDDING_DIMENSIONS,
            encoding_format="float",
        )
        batch_embeddings = sorted(response.data, key=lambda item: item.index)
        if len(batch_embeddings) != len(batch):
            raise RuntimeError("Embedding response did not match the input batch")
        for item in batch_embeddings:
            if len(item.embedding) != EMBEDDING_DIMENSIONS:
                raise RuntimeError("Embedding response had an unexpected dimension")
            embeddings.append(item.embedding)
    return embeddings
