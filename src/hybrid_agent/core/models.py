"""Immutable domain models for financial telemetry, market data, and trading objects."""

import math
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from hybrid_agent.core.constants import (
    ALLOWED_ORDER_TRANSITIONS,
    InvalidOrderStateTransitionError,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
    ValuationModel,
)


def validate_utc_timestamp(dt: datetime) -> datetime:
    """Enforces that datetime instances must be timezone-aware and set to UTC."""
    if dt.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware. Naive datetimes are strictly forbidden."
        )
    # Check if offset is UTC (0 seconds)
    offset = dt.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"Timestamp must be in UTC timezone, received offset: {offset}")
    return dt


class Timeframe(str, Enum):
    """Standardized market data aggregation intervals."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        """Returns interval duration in seconds."""
        lookup = {
            Timeframe.M1: 60,
            Timeframe.M5: 300,
            Timeframe.M15: 900,
            Timeframe.H1: 3600,
            Timeframe.H4: 14400,
            Timeframe.D1: 86400,
            Timeframe.W1: 604800,
        }
        return lookup[self]

    @property
    def timedelta(self) -> timedelta:
        """Returns interval duration as a timedelta."""
        return timedelta(seconds=self.seconds)

    @property
    def pandas_freq(self) -> str:
        """Returns standard pandas resample frequency string."""
        lookup = {
            Timeframe.M1: "1min",
            Timeframe.M5: "5min",
            Timeframe.M15: "15min",
            Timeframe.H1: "1h",
            Timeframe.H4: "4h",
            Timeframe.D1: "1D",
            Timeframe.W1: "1W",
        }
        return lookup[self]

    @classmethod
    def from_string(cls, val: str) -> "Timeframe":
        """Parses common string representations into a Timeframe enum."""
        cleaned = val.strip().lower()
        mapping = {
            "1m": cls.M1,
            "m1": cls.M1,
            "1min": cls.M1,
            "5m": cls.M5,
            "m5": cls.M5,
            "5min": cls.M5,
            "15m": cls.M15,
            "m15": cls.M15,
            "15min": cls.M15,
            "1h": cls.H1,
            "h1": cls.H1,
            "60m": cls.H1,
            "4h": cls.H4,
            "h4": cls.H4,
            "1d": cls.D1,
            "d1": cls.D1,
            "daily": cls.D1,
            "1w": cls.W1,
            "w1": cls.W1,
            "weekly": cls.W1,
        }
        if cleaned not in mapping:
            raise ValueError(f"Unsupported timeframe identifier: '{val}'")
        return mapping[cleaned]


class DataSourceType(str, Enum):
    """Classification of market data origin."""

    REAL_LIVE = "REAL_LIVE"
    DELAYED = "DELAYED"
    HISTORICAL = "HISTORICAL"
    SIMULATED = "SIMULATED"


class VolumeType(str, Enum):
    """Classification of volume telemetry semantics."""

    REAL_TRADED = "REAL_TRADED"
    TICK_COUNT = "TICK_COUNT"
    UNAVAILABLE = "UNAVAILABLE"


class DataProvenance(BaseModel, frozen=True):
    """Metadata detailing data origin, provider, and ingestion audit trail."""

    source_name: str
    source_type: DataSourceType
    retrieved_at: datetime
    is_synthetic: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("retrieved_at")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return validate_utc_timestamp(v)


class Quote(BaseModel, frozen=True):
    """Real-time bid/ask quote snapshot with market depth and provenance."""

    symbol: str
    timestamp: datetime
    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    bid_size: float = Field(default=0.0, ge=0.0)
    ask_size: float = Field(default=0.0, ge=0.0)
    provenance: DataProvenance | None = Field(default=None)

    @field_validator("timestamp")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return validate_utc_timestamp(v)

    @model_validator(mode="after")
    def validate_bid_ask(self) -> "Quote":
        if self.bid > self.ask:
            raise ValueError(
                f"Invalid Quote: bid price ({self.bid}) cannot exceed ask price ({self.ask})"
            )
        return self


class Tick(BaseModel, frozen=True):
    """High-resolution market quote snapshot with bid/ask validation."""

    symbol: str
    timestamp: datetime
    bid: float = Field(gt=0.0)
    ask: float = Field(gt=0.0)
    bid_volume: float = Field(default=0.0, ge=0.0)
    ask_volume: float = Field(default=0.0, ge=0.0)
    provenance: DataProvenance | None = Field(default=None)

    @field_validator("timestamp")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return validate_utc_timestamp(v)

    @model_validator(mode="after")
    def validate_bid_ask(self) -> "Tick":
        if self.bid > self.ask:
            raise ValueError(
                f"Invalid Tick: bid price ({self.bid}) cannot exceed ask price ({self.ask})"
            )
        return self


class Bar(BaseModel, frozen=True):
    """OHLCV aggregation with strict availability timestamping and OHLC relationship invariants."""

    symbol: str
    start_time: datetime
    end_time: datetime
    open: float = Field(gt=0.0)
    high: float = Field(gt=0.0)
    low: float = Field(gt=0.0)
    close: float = Field(gt=0.0)
    volume: float = Field(ge=0.0)
    volume_type: VolumeType = Field(default=VolumeType.REAL_TRADED)
    timeframe: Timeframe = Field(default=Timeframe.D1)
    is_synthetic: bool = Field(default=False)
    provenance: DataProvenance | None = Field(default=None)

    @field_validator("start_time", "end_time")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return validate_utc_timestamp(v)

    @model_validator(mode="after")
    def validate_bar_invariants(self) -> "Bar":
        # Temporal ordering
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Invalid Bar: end_time ({self.end_time}) must be strictly later than start_time ({self.start_time})"
            )

        # High / Low boundary invariants
        if self.high < self.low:
            raise ValueError(
                f"Invalid Bar: high ({self.high}) cannot be lower than low ({self.low})"
            )
        if self.high < self.open or self.high < self.close:
            raise ValueError(
                f"Invalid Bar: high ({self.high}) must be >= open ({self.open}) and close ({self.close})"
            )
        if self.low > self.open or self.low > self.close:
            raise ValueError(
                f"Invalid Bar: low ({self.low}) must be <= open ({self.open}) and close ({self.close})"
            )
        return self


class InstrumentSpecification(BaseModel, frozen=True):
    """Explicit market instrument trading specification and valuation metadata."""

    symbol: str
    tick_size: float = Field(gt=0.0)
    contract_size: float = Field(default=1.0, gt=0.0)
    min_volume: float = Field(gt=0.0)
    volume_step: float = Field(gt=0.0)
    price_precision: int = Field(default=2, ge=0)
    profit_currency: str = Field(default="USD", min_length=3, max_length=5)
    quote_currency: str = Field(default="USD", min_length=3, max_length=5)
    base_currency: str = Field(default="USD", min_length=3, max_length=5)
    valuation_model: ValuationModel = Field(default=ValuationModel.LINEAR)

    @model_validator(mode="after")
    def validate_spec_invariants(self) -> "InstrumentSpecification":
        if self.min_volume < self.volume_step:
            raise ValueError(
                f"Invalid InstrumentSpec: min_volume ({self.min_volume}) cannot be less than volume_step ({self.volume_step})"
            )
        # Check volume step alignment
        step_ratio = self.min_volume / self.volume_step
        if not math.isclose(step_ratio, round(step_ratio), rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(
                f"Invalid InstrumentSpec: min_volume ({self.min_volume}) must be an exact multiple of volume_step ({self.volume_step})"
            )
        return self

    def validate_quantity(self, quantity: float) -> None:
        """Validates that a quantity conforms to min volume and volume step requirements."""
        if quantity < self.min_volume:
            raise OrderValidationError(
                f"Quantity {quantity} is below minimum allowed volume {self.min_volume} for {self.symbol}"
            )
        steps = quantity / self.volume_step
        if not math.isclose(steps, round(steps), rel_tol=1e-7, abs_tol=1e-7):
            raise OrderValidationError(
                f"Quantity {quantity} does not align with volume_step {self.volume_step} for {self.symbol}"
            )

    def validate_price(self, price: float) -> None:
        """Validates that a price conforms to tick size and price precision."""
        if price <= 0.0:
            raise OrderValidationError(f"Price {price} must be strictly positive")
        # Check precision
        rounded = round(price, self.price_precision)
        if not math.isclose(price, rounded, rel_tol=1e-7, abs_tol=1e-7):
            raise OrderValidationError(
                f"Price {price} exceeds allowed price precision of {self.price_precision} decimal places"
            )
        # Check tick size multiple
        ticks = price / self.tick_size
        if not math.isclose(ticks, round(ticks), rel_tol=1e-7, abs_tol=1e-7):
            raise OrderValidationError(
                f"Price {price} is not an exact multiple of tick_size {self.tick_size} for {self.symbol}"
            )


class Order(BaseModel):
    """Trading order entity with lifecycle state machine and directional validation."""

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float = Field(gt=0.0)
    limit_price: float | None = Field(default=None)
    stop_loss_price: float | None = Field(default=None)
    take_profit_price: float | None = Field(default=None)
    status: OrderStatus = Field(default=OrderStatus.PENDING_RISK)
    created_at: datetime
    updated_at: datetime

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_initialized", False) and name == "status":
            raise InvalidOrderStateTransitionError(
                "Direct mutation of order 'status' is prohibited. "
                "Use order.transition_to(new_status) to enforce validated lifecycle rules."
            )
        super().__setattr__(name, value)

    @field_validator("created_at", "updated_at")
    @classmethod
    def check_utc(cls, v: datetime) -> datetime:
        return validate_utc_timestamp(v)

    @model_validator(mode="after")
    def validate_directional_prices(self) -> "Order":
        ref_price = self.limit_price

        # Directional validation for stop loss when limit price is defined
        if self.stop_loss_price is not None:
            if self.stop_loss_price <= 0.0:
                raise ValueError("stop_loss_price must be positive")
            if ref_price is not None:
                if self.side == OrderSide.BUY and self.stop_loss_price >= ref_price:
                    raise OrderValidationError(
                        f"Directional Error: BUY stop_loss_price ({self.stop_loss_price}) must be strictly below entry price ({ref_price})"
                    )
                if self.side == OrderSide.SELL and self.stop_loss_price <= ref_price:
                    raise OrderValidationError(
                        f"Directional Error: SELL stop_loss_price ({self.stop_loss_price}) must be strictly above entry price ({ref_price})"
                    )

        # Directional validation for take profit when limit price is defined
        if self.take_profit_price is not None:
            if self.take_profit_price <= 0.0:
                raise ValueError("take_profit_price must be positive")
            if ref_price is not None:
                if self.side == OrderSide.BUY and self.take_profit_price <= ref_price:
                    raise OrderValidationError(
                        f"Directional Error: BUY take_profit_price ({self.take_profit_price}) must be strictly above entry price ({ref_price})"
                    )
                if self.side == OrderSide.SELL and self.take_profit_price >= ref_price:
                    raise OrderValidationError(
                        f"Directional Error: SELL take_profit_price ({self.take_profit_price}) must be strictly below entry price ({ref_price})"
                    )
        return self

    def transition_to(self, new_status: OrderStatus, timestamp: datetime | None = None) -> None:
        """Transitions the order status following strict lifecycle rules."""
        allowed = ALLOWED_ORDER_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise InvalidOrderStateTransitionError(
                f"Illegal order lifecycle transition from '{self.status.value}' to '{new_status.value}'"
            )
        super().__setattr__("status", new_status)
        now = timestamp if timestamp is not None else datetime.now(UTC)
        super().__setattr__("updated_at", validate_utc_timestamp(now))

    def validate_against_instrument(self, spec: InstrumentSpecification) -> None:
        """Validates order specifications against an instrument specification."""
        if self.symbol != spec.symbol:
            raise OrderValidationError(
                f"Order symbol '{self.symbol}' does not match instrument specification symbol '{spec.symbol}'"
            )
        spec.validate_quantity(self.quantity)
        if self.limit_price is not None:
            spec.validate_price(self.limit_price)
        if self.stop_loss_price is not None:
            spec.validate_price(self.stop_loss_price)
        if self.take_profit_price is not None:
            spec.validate_price(self.take_profit_price)


class Position(BaseModel):
    """
    Open position entity separating gross notional exposure from monetary downside risk.
    Assumes linear payoff structure (Spot / Cash Equity / Direct Forex).
    """

    symbol: str
    side: OrderSide
    quantity: float = Field(ge=0.0)
    average_entry_price: float = Field(gt=0.0)
    current_price: float = Field(gt=0.0)
    contract_size: float = Field(default=1.0, gt=0.0)
    stop_loss_price: float | None = Field(default=None)
    unrealized_pnl: float = Field(default=0.0)
    realized_pnl: float = Field(default=0.0)

    @property
    def notional_value(self) -> float:
        """
        Gross exposure of the open position under linear valuation assumption:
        Notional = Quantity * CurrentPrice * ContractSize.
        """
        return self.quantity * self.current_price * self.contract_size

    @property
    def monetary_risk(self) -> float:
        """
        Quantifies downside monetary capital at risk under linear valuation assumption.
        If a stop_loss_price is established: |entry - stop_loss| * qty * contract_size.
        If unhedged:
          - For BUY: Total entry notional (entry price to 0)
          - For SELL: Notional baseline (unbounded theoretically, returns full entry notional baseline)
        """
        if self.stop_loss_price is not None:
            price_delta = abs(self.average_entry_price - self.stop_loss_price)
            return price_delta * self.quantity * self.contract_size
        return self.quantity * self.average_entry_price * self.contract_size
