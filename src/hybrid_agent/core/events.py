"""Event structures for the asynchronous/synchronous event bus."""

from datetime import datetime
from pydantic import BaseModel
from hybrid_agent.core.models import Bar, Order, Tick


class BaseEvent(BaseModel, frozen=True):
    """Base event marker with strict timestamp."""
    timestamp: datetime


class MarketTickEvent(BaseEvent):
    """Event emitted upon receiving new tick data."""
    tick: Tick


class MarketBarEvent(BaseEvent):
    """Event emitted upon receiving a closed bar."""
    bar: Bar


class SignalEvent(BaseEvent):
    """Alpha signal produced by strategy or ML model."""
    symbol: str
    score: float
    confidence: float
    direction: int  # +1 Long, -1 Short, 0 Flat


class OrderProposalEvent(BaseEvent):
    """Proposed candidate order submitted to risk firewall."""
    order: Order


class OrderApprovedEvent(BaseEvent):
    """Order cleared by risk engine for paper execution."""
    order: Order


class OrderRejectedEvent(BaseEvent):
    """Order blocked by risk engine with rationale."""
    order: Order
    reason: str
