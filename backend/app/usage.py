"""Persist AI provider usage for a RAG request."""

from uuid import UUID

from app.pricing import estimate_cost


class UsageRecordingError(Exception):
    pass


def record_usage_event(
    *,
    request_id: UUID,
    event_type: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    request_success: bool,
) -> None:
    try:
        from app.database import SessionLocal
        from app.models import AIUsageEvent

        session = SessionLocal()
        try:
            session.add(
                AIUsageEvent(
                    request_id=request_id,
                    event_type=event_type,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    latency_ms=latency_ms,
                    request_success=request_success,
                    estimated_cost=estimate_cost(
                        event_type=event_type,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        request_success=request_success,
                    ),
                )
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    except Exception as exc:
        raise UsageRecordingError("AI usage metrics could not be recorded") from exc
