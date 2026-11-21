"""Event structures for the asynchronous/synchronous event bus."""

from datetime import datetime

from pydantic import BaseModel, field_validator

from hybrid_agent.core.models import Bar, Order, Tick, validate_utc_timestamp


class BaseEvent(BaseModel, frozen=True):
    """Base event marker with strict timezone-aware UTC timestamp."""

    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return validate_utc_timestamp(v)


class MarketTickEvent(BaseEvent, frozen=True):
    """Event emitted upon receiving new tick data."""

    tick: Tick


class MarketBarEvent(BaseEvent, frozen=True):
    """Event emitted upon receiving a closed bar."""

    bar: Bar


class SignalEvent(BaseEvent, frozen=True):
    """Alpha signal produced by strategy or ML model."""

    symbol: str
    score: float
    confidence: float
    direction: int  # +1 Long, -1 Short, 0 Flat


class OrderProposalEvent(BaseEvent, frozen=True):
    """Proposed candidate order submitted to risk firewall."""

    order: Order


class OrderApprovedEvent(BaseEvent, frozen=True):
    """Order cleared by risk engine for paper execution."""

    order: Order


class OrderRejectedEvent(BaseEvent, frozen=True):
    """Order blocked by risk engine with rationale."""

    order: Order
    reason: str
