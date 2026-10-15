import os
from decimal import Decimal
from uuid import uuid4

os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://opsmind:opsmind@localhost:5432/opsmind"
)

from app import database
from app import main
from app.models import AIUsageEvent
from app.pricing import estimate_cost
from app.usage import record_usage_event


def test_usage_event_persists_provider_metrics_and_request_id(monkeypatch) -> None:
    class FakeSession:
        def __init__(self):
            self.event = None
            self.committed = False
            self.closed = False

        def add(self, event):
            self.event = event

        def commit(self):
            self.committed = True

        def rollback(self):
            raise AssertionError("rollback should not be needed")

        def close(self):
            self.closed = True

    session = FakeSession()
    monkeypatch.setattr(database, "SessionLocal", lambda: session)
    request_id = uuid4()

    record_usage_event(
        request_id=request_id,
        event_type="embedding",
        model="text-embedding-3-small",
        input_tokens=12,
        output_tokens=0,
        latency_ms=35,
        request_success=True,
    )

    assert isinstance(session.event, AIUsageEvent)
    assert session.event.request_id == request_id
    assert session.event.event_type == "embedding"
    assert session.event.model == "text-embedding-3-small"
    assert session.event.input_tokens == 12
    assert session.event.output_tokens == 0
    assert session.event.latency_ms == 35
    assert session.event.request_success is True
    assert session.event.estimated_cost == Decimal("0.00000024")
    assert session.committed and session.closed


def test_llm_usage_cost_and_unknown_model(monkeypatch) -> None:
    events = []

    class FakeSession:
        def add(self, event):
            events.append(event)

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(database, "SessionLocal", FakeSession)

    record_usage_event(
        request_id=uuid4(),
        event_type="llm",
        model="gpt-6-luna",
        input_tokens=1_000_000,
        output_tokens=500_000,
        latency_ms=1,
        request_success=True,
    )
    assert events[0].estimated_cost == Decimal("0.35000000")

    record_usage_event(
        request_id=uuid4(),
        event_type="llm",
        model="unpriced-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=1,
        request_success=True,
    )
    assert events[1].estimated_cost is None
    assert events[1].model == "unpriced-model"
    assert events[1].input_tokens == 100
    assert events[1].output_tokens == 50


def test_baseline_chat_records_estimated_cost(monkeypatch) -> None:
    events = []

    class FakeSession:
        def add(self, event):
            events.append(event)

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(database, "SessionLocal", FakeSession)

    assert main._record_usage(
        model="gpt-6-luna",
        input_tokens=100,
        output_tokens=20,
        latency_ms=40,
        request_success=True,
    )
    assert events[0].estimated_cost == Decimal("0.00002000")


def test_failed_call_without_token_usage_has_no_cost() -> None:
    assert estimate_cost(
        event_type="llm",
        model="gpt-6-luna",
        input_tokens=0,
        output_tokens=0,
        request_success=False,
    ) is None


def test_existing_llm_rows_get_database_default() -> None:
    event_type = AIUsageEvent.__table__.c.event_type
    request_id = AIUsageEvent.__table__.c.request_id

    assert event_type.server_default.arg == "llm"
    assert event_type.nullable is False
    assert request_id.nullable is True
