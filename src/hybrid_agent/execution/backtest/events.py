"""Event definitions and priority hierarchy for the backtesting engine."""

from datetime import datetime
from enum import IntEnum
from typing import Any

from pydantic import Field, model_validator

from hybrid_agent.core.events import BaseEvent
from hybrid_agent.core.models import Bar, Order, Position, validate_utc_timestamp


class EventPriority(IntEnum):
    """Deterministic intra-timestamp event dispatch priority (1=highest, 8=lowest).

    As established in the accepted Milestone 3B architecture specification:
    1: MarketBarClosedEvent - completed bar arrives at or after T_end
    2: Strategy evaluation - strategy receives closed bar
    3: Order proposal and risk approval - candidate orders evaluated by risk firewall
    4: Order submission and reservation - approved orders registered in OrderManager
    5: MarketBarOpenedEvent - subsequent bar begins at T_start
    6: Order matching and fill processing - orders matched against open prices/quotes
    7: Ledger and position reconciliation - fills update balances and positions
    8: Portfolio mark-to-market - equity snapshot recorded
    """

    MARKET_BAR_CLOSED = 1
    STRATEGY_EVALUATION = 2
    ORDER_PROPOSAL_RISK = 3
    ORDER_SUBMISSION_RESERVATION = 4
    MARKET_BAR_OPENED = 5
    ORDER_MATCHING_FILL = 6
    LEDGER_RECONCILIATION = 7
    PORTFOLIO_MARK_TO_MARKET = 8


class MarketBarClosedEvent(BaseEvent, frozen=True):
    """Event emitted when a historical bar completes and becomes available for strategy consumption.

    Invariant:
        The event's availability timestamp must be strictly greater than or equal to
        the bar's end_time (T_avail >= T_end). A bar is never available before its close.
    """

    bar: Bar

    @model_validator(mode="after")
    def validate_bar_availability(self) -> "MarketBarClosedEvent":
        if self.timestamp < self.bar.end_time:
            raise ValueError(
                f"Data causality violation: MarketBarClosedEvent timestamp ({self.timestamp.isoformat()}) "
                f"cannot be earlier than bar.end_time ({self.bar.end_time.isoformat()})"
            )
        return self


class MarketBarOpenedEvent(BaseEvent, frozen=True):
    """Event emitted at the opening timestamp of a new historical bar.

    Invariant:
        The event's timestamp must correspond exactly to the bar's start_time.
    """

    bar: Bar

    @model_validator(mode="after")
    def validate_bar_opening(self) -> "MarketBarOpenedEvent":
        if self.timestamp != self.bar.start_time:
            raise ValueError(
                f"Temporal boundary mismatch: MarketBarOpenedEvent timestamp ({self.timestamp.isoformat()}) "
                f"must match bar.start_time ({self.bar.start_time.isoformat()})"
            )
        return self


class OrderSubmittedEvent(BaseEvent, frozen=True):
    """Event emitted when an approved order is submitted to the OrderManager.

    Records the explicit submission timestamp to guarantee T_execution >= T_submission.
    """

    order: Order
    submission_time: datetime = Field(default=...)

    @model_validator(mode="before")
    @classmethod
    def set_default_submission_time(cls, data: Any) -> Any:
        if isinstance(data, dict) and "submission_time" not in data and "timestamp" in data:
            data["submission_time"] = data["timestamp"]
        return data

    @model_validator(mode="after")
    def validate_submission_time(self) -> "OrderSubmittedEvent":
        validate_utc_timestamp(self.submission_time)
        return self


class OrderPartiallyFilledEvent(BaseEvent, frozen=True):
    """Event emitted upon partial execution of an order."""

    order: Order
    fill_price: float = Field(gt=0.0)
    fill_quantity: float = Field(gt=0.0)
    fee: float = Field(default=0.0, ge=0.0)
    slippage: float = Field(default=0.0)


class OrderFilledEvent(BaseEvent, frozen=True):
    """Event emitted upon complete execution of an order."""

    order: Order
    fill_price: float = Field(gt=0.0)
    fill_quantity: float = Field(gt=0.0)
    fee: float = Field(default=0.0, ge=0.0)
    slippage: float = Field(default=0.0)


class OrderCancelledEvent(BaseEvent, frozen=True):
    """Event emitted when an order or remaining quantity is cancelled."""

    order: Order
    reason: str


class PositionUpdatedEvent(BaseEvent, frozen=True):
    """Event emitted when position inventory or average entry price is modified."""

    position: Position


class PortfolioMarkedEvent(BaseEvent, frozen=True):
    """Event emitted when portfolio equity and drawdown metrics are marked to market."""

    equity: float
    cash: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float = Field(ge=0.0)


class BacktestCompletedEvent(BaseEvent, frozen=True):
    """Event emitted when simulation replay reaches the end of the market dataset."""

    total_bars: int = Field(ge=0)
    total_trades: int = Field(ge=0)
    final_equity: float
    completed_at: datetime
