import os
from dataclasses import dataclass

from openai import OpenAI


SYSTEM_INSTRUCTIONS = (
    "You are OpsMind, a concise assistant for workplace operations. "
    "Answer clearly and say when you do not know."
)


class LLMConfigurationError(Exception):
    pass


class LLMUsageError(Exception):
    pass


@dataclass(frozen=True)
class LLMResult:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int


def generate_reply(message: str) -> LLMResult:
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not api_key or not model:
        raise LLMConfigurationError

    client = OpenAI(api_key=api_key, max_retries=0)
    response = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=message,
        store=False,
    )
    if response.usage is None:
        raise LLMUsageError

    return LLMResult(
        answer=response.output_text,
        model=response.model or model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )
