"""Immutable domain models for financial telemetry and trading objects."""

from datetime import datetime
from pydantic import BaseModel, Field
from hybrid_agent.core.constants import OrderSide, OrderStatus, OrderType


class Tick(BaseModel, frozen=True):
    """High-resolution market quote snapshot."""
    symbol: str
    timestamp: datetime
    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    bid_volume: float = Field(default=0.0, ge=0.0)
    ask_volume: float = Field(default=0.0, ge=0.0)


class Bar(BaseModel, frozen=True):
    """OHLCV aggregation with strict availability timestamping."""
    symbol: str
    start_time: datetime
    end_time: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)


class Order(BaseModel):
    """Trading order entity with lifecycle tracking."""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0.0)
    limit_price: float | None = Field(default=None)
    stop_price: float | None = Field(default=None)
    status: OrderStatus = Field(default=OrderStatus.PENDING_RISK)
    created_at: datetime
    updated_at: datetime


class Position(BaseModel):
    """Open position entity tracking inventory and unrealized PnL."""
    symbol: str
    side: OrderSide
    quantity: float = Field(ge=0.0)
    average_entry_price: float = Field(gt=0.0)
    current_price: float = Field(gt=0.0)
    unrealized_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)
