import os
from time import perf_counter
from typing import Callable, Optional, Sequence

from openai import OpenAI


EMBEDDING_DIMENSIONS = 1536
EMBEDDING_BATCH_SIZE = 64
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class EmbeddingConfigurationError(Exception):
    pass


def create_embeddings(
    texts: Sequence[str], *, on_usage: Optional[Callable[..., None]] = None
) -> list[list[float]]:
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
        started_at = perf_counter()
        response = None
        try:
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
        except Exception:
            if on_usage is not None:
                usage = getattr(response, "usage", None)
                on_usage(
                    event_type="embedding",
                    model=getattr(response, "model", None) or model,
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=0,
                    latency_ms=int((perf_counter() - started_at) * 1000),
                    request_success=False,
                )
            raise
        if on_usage is not None:
            usage = getattr(response, "usage", None)
            on_usage(
                event_type="embedding",
                model=getattr(response, "model", None) or model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=0,
                latency_ms=int((perf_counter() - started_at) * 1000),
                request_success=True,
            )
        embeddings.extend(item.embedding for item in batch_embeddings)
    return embeddings
