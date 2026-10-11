import os
from time import perf_counter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm import LLMConfigurationError, LLMResult, generate_reply
from app.routes.documents import router as documents_router

app = FastAPI(title="OpsMind API", version="0.1.0")
app.include_router(documents_router)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    model: str


def _record_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    request_success: bool,
) -> bool:
    try:
        from app.database import SessionLocal
        from app.models import AIUsageEvent

        session = SessionLocal()
        try:
            session.add(
                AIUsageEvent(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    request_success=request_success,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            return False
        finally:
            session.close()
    except Exception:
        return False
    return True


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/api/version")
def version() -> dict[str, str]:
    return {"version": "0.1.0"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    configured_model = os.environ.get("OPENAI_MODEL") or "unconfigured"
    started_at = perf_counter()

    try:
        result: LLMResult = generate_reply(request.message)
    except LLMConfigurationError:
        _record_usage(
            model=configured_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((perf_counter() - started_at) * 1000),
            request_success=False,
        )
        raise HTTPException(status_code=503, detail="LLM service is not configured.")
    except Exception:
        _record_usage(
            model=configured_model,
            input_tokens=0,
            output_tokens=0,
            latency_ms=int((perf_counter() - started_at) * 1000),
            request_success=False,
        )
        raise HTTPException(
            status_code=502, detail="LLM provider request failed."
        ) from None

    recorded = _record_usage(
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=int((perf_counter() - started_at) * 1000),
        request_success=True,
    )
    if not recorded:
        raise HTTPException(status_code=503, detail="Usage metrics could not be recorded.")

    return ChatResponse(answer=result.answer, model=result.model)
