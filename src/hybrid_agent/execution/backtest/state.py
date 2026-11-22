"""Execution-state foundation data structures for the native backtest execution engine.

Provides:
- ExecutionStatus: dedicated order matching progress enumeration.
- FillRecord: immutable execution fill slice with strict Decimal precision.
- OrderExecutionState: state wrapper around domain Order managing execution matching,
  partial fills, remaining quantities, weighted-average prices, and cancellations.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from types import MappingProxyType

from hybrid_agent.core.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)
from hybrid_agent.core.models import (
    InstrumentSpecification,
    Order,
    validate_utc_timestamp,
)


def _to_decimal(value: Decimal | float | str, name: str) -> Decimal:
    """Safely converts an input value to a finite Decimal.

    Converts float types strictly via str(value) to prevent IEEE-754 precision artifacts.
    """
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, float):
        dec = Decimal(str(value))
    elif isinstance(value, (int, str)):
        dec = Decimal(value)
    else:
        raise TypeError(f"{name} must be a Decimal, float, int, or str, got {type(value).__name__}")

    if not dec.is_finite():
        raise ValueError(f"{name} must be a finite Decimal value, got {value}")
    return dec


class ExecutionStatus(str, Enum):
    """Lifecycle status representing the matching progress and residual quantity of an order.

    Distinction:
    - OrderStatus: High-level risk and submission lifecycle (APPROVED -> SUBMITTED -> FILLED/CANCELLED).
    - ExecutionStatus: Micro execution state tracking matching progress (ACTIVE -> PARTIALLY_FILLED -> FILLED/CANCELLED).
    """

    ACTIVE = "ACTIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class FillRecord:
    """Immutable execution fill record representing a single fill event.

    Uses Decimal for all monetary and quantity figures to enforce mathematical precision.
    """

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    fill_price: Decimal
    fill_quantity: Decimal
    timestamp: datetime
    fee: Decimal = Decimal(0)
    commission_currency: str | None = None
    slippage: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        if not self.fill_id or not isinstance(self.fill_id, str) or not self.fill_id.strip():
            raise ValueError("fill_id must be a non-empty string")
        if not self.order_id or not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if not self.symbol or not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.side, OrderSide):
            raise TypeError(
                f"side must be an instance of OrderSide, got {type(self.side).__name__}"
            )

        # Coerce and validate Decimal fields
        px = _to_decimal(self.fill_price, "fill_price")
        if px <= Decimal(0):
            raise ValueError(f"fill_price must be strictly positive, got {px}")
        object.__setattr__(self, "fill_price", px)

        qty = _to_decimal(self.fill_quantity, "fill_quantity")
        if qty <= Decimal(0):
            raise ValueError(f"fill_quantity must be strictly positive, got {qty}")
        object.__setattr__(self, "fill_quantity", qty)

        fee_dec = _to_decimal(self.fee, "fee")
        if fee_dec < Decimal(0):
            raise ValueError(f"fee must be non-negative, got {fee_dec}")
        object.__setattr__(self, "fee", fee_dec)

        slip = _to_decimal(self.slippage, "slippage")
        object.__setattr__(self, "slippage", slip)

        # Validate commission currency rules
        if fee_dec > Decimal(0):
            if (
                not self.commission_currency
                or not isinstance(self.commission_currency, str)
                or not self.commission_currency.strip()
            ):
                raise ValueError(
                    "commission_currency must be provided as a non-empty string when fee > 0"
                )
            object.__setattr__(
                self, "commission_currency", self.commission_currency.strip().upper()
            )
        elif self.commission_currency is not None:
            if isinstance(self.commission_currency, str) and self.commission_currency.strip():
                object.__setattr__(
                    self, "commission_currency", self.commission_currency.strip().upper()
                )
            else:
                object.__setattr__(self, "commission_currency", None)

        # Validate timestamp
        ts = validate_utc_timestamp(self.timestamp)
        object.__setattr__(self, "timestamp", ts)


@dataclass(frozen=True)
class OrderSnapshot:
    """Immutable public snapshot of an order's contract specification and lifecycle status."""

    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None
    stop_loss_price: float | None
    take_profit_price: float | None
    status: OrderStatus
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_order(cls, order: Order) -> "OrderSnapshot":
        """Constructs an immutable snapshot from a domain Order."""
        return cls(
            id=order.id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            limit_price=order.limit_price,
            stop_loss_price=order.stop_loss_price,
            take_profit_price=order.take_profit_price,
            status=order.status,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )


@dataclass(frozen=True)
class OrderExecutionSnapshot:
    """Immutable public snapshot of an order's execution state and progress.

    Protects OrderManager's internal registries from external mutation while
    exposing all execution telemetry, quantities, prices, fees, and lifecycle states.
    """

    order: OrderSnapshot
    submission_time: datetime
    original_quantity: Decimal
    filled_quantity: Decimal
    unfilled_quantity: Decimal
    leaves_quantity: Decimal
    cancelled_quantity: Decimal
    average_fill_price: Decimal
    cumulative_fees: Mapping[str, Decimal]
    status: ExecutionStatus
    fills: tuple[FillRecord, ...]
    last_update_time: datetime
    cancellation_timestamp: datetime | None = None
    cancellation_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cumulative_fees, MappingProxyType):
            object.__setattr__(
                self, "cumulative_fees", MappingProxyType(dict(self.cumulative_fees))
            )

    @property
    def is_active(self) -> bool:
        """Whether the order is currently active without any fills."""
        return self.status == ExecutionStatus.ACTIVE

    @property
    def is_partially_filled(self) -> bool:
        """Whether the order has received one or more fills but has remaining executable quantity."""
        return self.status == ExecutionStatus.PARTIALLY_FILLED

    @property
    def is_filled(self) -> bool:
        """Whether the order has been completely executed."""
        return self.status == ExecutionStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        """Whether the unfilled remainder was cancelled."""
        return self.status == ExecutionStatus.CANCELLED

    @property
    def is_terminal(self) -> bool:
        """Whether the execution state has reached a terminal state (FILLED or CANCELLED)."""
        return self.status in (ExecutionStatus.FILLED, ExecutionStatus.CANCELLED)


class OrderExecutionState:
    """State wrapper around domain Order managing execution matching, partial fills, and residuals.

    Maintains complete execution telemetry, remaining executable leaves quantity,
    weighted-average execution prices, fee aggregation by currency, and cancellation history.
    """

    def __init__(
        self,
        order: Order,
        submission_time: datetime,
        instrument_spec: InstrumentSpecification | None = None,
    ) -> None:
        if not isinstance(order, Order):
            raise TypeError(f"order must be an instance of Order, got {type(order).__name__}")

        if order.status not in (OrderStatus.APPROVED, OrderStatus.SUBMITTED):
            raise ValueError(
                f"Cannot initialize OrderExecutionState for order in status '{order.status.value}'. "
                f"Order must be in APPROVED or SUBMITTED status."
            )

        sub_time = validate_utc_timestamp(submission_time)
        if sub_time < order.created_at:
            raise ValueError(
                f"submission_time ({sub_time.isoformat()}) cannot be earlier than order.created_at ({order.created_at.isoformat()})"
            )

        orig_qty = _to_decimal(order.quantity, "order.quantity")
        if orig_qty <= Decimal(0):
            raise ValueError(f"Order quantity must be positive, got {orig_qty}")

        if instrument_spec is not None:
            if not isinstance(instrument_spec, InstrumentSpecification):
                raise TypeError(
                    f"instrument_spec must be an InstrumentSpecification, got {type(instrument_spec).__name__}"
                )
            if instrument_spec.symbol != order.symbol:
                raise ValueError(
                    f"Instrument specification symbol '{instrument_spec.symbol}' does not match order symbol '{order.symbol}'"
                )

            # Decimal-based instrument checks
            min_vol = _to_decimal(instrument_spec.min_volume, "min_volume")
            step_vol = _to_decimal(instrument_spec.volume_step, "volume_step")
            if orig_qty < min_vol:
                raise OrderValidationError(
                    f"Quantity {orig_qty} is below minimum allowed volume {min_vol} for {order.symbol}"
                )
            if orig_qty % step_vol != Decimal(0):
                raise OrderValidationError(
                    f"Quantity {orig_qty} does not align with volume_step {step_vol} for {order.symbol}"
                )

            # Validate prices against instrument specification
            if order.limit_price is not None:
                instrument_spec.validate_price(order.limit_price)
            if order.stop_loss_price is not None:
                instrument_spec.validate_price(order.stop_loss_price)
            if order.take_profit_price is not None:
                instrument_spec.validate_price(order.take_profit_price)

        self._order = order
        self._submission_time = sub_time
        self._instrument_spec = instrument_spec
        self._original_quantity = orig_qty
        self._filled_quantity = Decimal(0)
        self._average_fill_price = Decimal(0)
        self._cumulative_fees: dict[str, Decimal] = {}
        self._fills: list[FillRecord] = []
        self._fill_ids: set[str] = set()
        self._status = ExecutionStatus.ACTIVE
        self._last_update_time = sub_time
        self._cancellation_timestamp: datetime | None = None
        self._cancellation_reason: str | None = None
        self._cancelled_remainder: Decimal | None = None

    @property
    def order(self) -> Order:
        """Defensive copy of the wrapped domain Order entity."""
        return self._order.model_copy(deep=True)

    @property
    def submission_time(self) -> datetime:
        """UTC timestamp when the order was submitted for execution."""
        return self._submission_time

    @property
    def instrument_spec(self) -> InstrumentSpecification | None:
        """Instrument specification, if provided."""
        return self._instrument_spec

    @property
    def original_quantity(self) -> Decimal:
        """Total quantity initially requested in the submitted order."""
        return self._original_quantity

    @property
    def filled_quantity(self) -> Decimal:
        """Cumulative quantity successfully executed across all fills."""
        return self._filled_quantity

    @property
    def unfilled_quantity(self) -> Decimal:
        """Historical unfilled quantity (original_quantity - filled_quantity).

        Preserved even after cancellation for auditing and remainder reporting.
        """
        return self._original_quantity - self._filled_quantity

    @property
    def leaves_quantity(self) -> Decimal:
        """Currently executable remaining quantity.

        Equal to unfilled_quantity while active or partially filled; becomes 0 once FILLED or CANCELLED.
        """
        if self._status in (ExecutionStatus.FILLED, ExecutionStatus.CANCELLED):
            return Decimal(0)
        return self._original_quantity - self._filled_quantity

    @property
    def cancelled_quantity(self) -> Decimal:
        """Quantity cancelled without execution, if the order was cancelled."""
        if self._status == ExecutionStatus.CANCELLED and self._cancelled_remainder is not None:
            return self._cancelled_remainder
        return Decimal(0)

    @property
    def average_fill_price(self) -> Decimal:
        """Volume-weighted average execution price across all fills."""
        return self._average_fill_price

    @property
    def cumulative_fees(self) -> dict[str, Decimal]:
        """Cumulative execution fees grouped by fee currency."""
        return dict(self._cumulative_fees)

    @property
    def status(self) -> ExecutionStatus:
        """Current execution lifecycle status."""
        return self._status

    @property
    def fills(self) -> tuple[FillRecord, ...]:
        """Immutable collection of all executed fill records."""
        return tuple(self._fills)

    @property
    def last_update_time(self) -> datetime:
        """UTC timestamp of the latest state mutation."""
        return self._last_update_time

    @property
    def cancellation_timestamp(self) -> datetime | None:
        """UTC timestamp when cancellation occurred, if cancelled."""
        return self._cancellation_timestamp

    @property
    def cancellation_reason(self) -> str | None:
        """Reason provided for cancellation, if cancelled."""
        return self._cancellation_reason

    @property
    def is_active(self) -> bool:
        """Whether the order is currently active without any fills."""
        return self._status == ExecutionStatus.ACTIVE

    @property
    def is_partially_filled(self) -> bool:
        """Whether the order has received one or more fills but has remaining executable quantity."""
        return self._status == ExecutionStatus.PARTIALLY_FILLED

    @property
    def is_filled(self) -> bool:
        """Whether the order has been completely executed."""
        return self._status == ExecutionStatus.FILLED

    @property
    def is_cancelled(self) -> bool:
        """Whether the unfilled remainder was cancelled."""
        return self._status == ExecutionStatus.CANCELLED

    @property
    def is_terminal(self) -> bool:
        """Whether the execution state has reached a terminal state (FILLED or CANCELLED)."""
        return self._status in (ExecutionStatus.FILLED, ExecutionStatus.CANCELLED)

    def record_fill(self, fill: FillRecord) -> None:
        """Atomically records an execution fill slice.

        Validates all preconditions before mutating state. If any validation fails,
        every state field remains unchanged.
        """
        if self.is_terminal:
            raise ValueError(
                f"Cannot record fill on terminal execution state '{self._status.value}'"
            )

        if not isinstance(fill, FillRecord):
            raise TypeError(f"fill must be a FillRecord instance, got {type(fill).__name__}")

        if fill.order_id != self._order.id:
            raise ValueError(
                f"Fill order_id '{fill.order_id}' does not match order id '{self._order.id}'"
            )

        if fill.symbol != self._order.symbol:
            raise ValueError(
                f"Fill symbol '{fill.symbol}' does not match order symbol '{self._order.symbol}'"
            )

        if fill.side != self._order.side:
            raise ValueError(
                f"Fill side '{fill.side}' does not match order side '{self._order.side}'"
            )

        if fill.fill_id in self._fill_ids:
            raise ValueError(f"Duplicate fill_id '{fill.fill_id}' already recorded")

        fill_time = validate_utc_timestamp(fill.timestamp)
        if fill_time < self._submission_time:
            raise ValueError(
                f"Fill timestamp ({fill_time.isoformat()}) cannot be earlier than submission_time ({self._submission_time.isoformat()})"
            )
        if fill_time < self._last_update_time:
            raise ValueError(
                f"Fill timestamp ({fill_time.isoformat()}) cannot be earlier than previous update ({self._last_update_time.isoformat()})"
            )

        if fill.fill_price <= Decimal(0):
            raise ValueError(f"Fill price must be strictly positive, got {fill.fill_price}")

        if fill.fill_quantity <= Decimal(0):
            raise ValueError(f"Fill quantity must be strictly positive, got {fill.fill_quantity}")

        if fill.fee < Decimal(0):
            raise ValueError(f"Fee must be non-negative, got {fill.fee}")

        if fill.fee > Decimal(0) and (
            not fill.commission_currency or not fill.commission_currency.strip()
        ):
            raise ValueError("commission_currency must be provided when fee > 0")

        # Validate volume step alignment for partial fills against instrument specification
        if self._instrument_spec is not None:
            step = _to_decimal(self._instrument_spec.volume_step, "volume_step")
            if fill.fill_quantity % step != Decimal(0):
                raise ValueError(
                    f"Fill quantity {fill.fill_quantity} does not align with volume_step {step} for {self._order.symbol}"
                )

        # Check leaves and cumulative boundaries
        current_leaves = self.leaves_quantity
        if fill.fill_quantity > current_leaves:
            raise ValueError(
                f"Fill quantity {fill.fill_quantity} exceeds executable leaves quantity {current_leaves}"
            )

        new_filled = self._filled_quantity + fill.fill_quantity
        if new_filled > self._original_quantity:
            raise ValueError(
                f"Cumulative filled quantity {new_filled} exceeds original quantity {self._original_quantity}"
            )

        # Calculations (all pure, non-mutating until complete)
        total_executed_notional = (self._average_fill_price * self._filled_quantity) + (
            fill.fill_price * fill.fill_quantity
        )
        new_average_price = total_executed_notional / new_filled

        new_fees = dict(self._cumulative_fees)
        if fill.fee > Decimal(0) and fill.commission_currency is not None:
            ccy = fill.commission_currency
            new_fees[ccy] = new_fees.get(ccy, Decimal(0)) + fill.fee

        new_status = (
            ExecutionStatus.FILLED
            if new_filled == self._original_quantity
            else ExecutionStatus.PARTIALLY_FILLED
        )

        # Atomic commit
        self._fills.append(fill)
        self._fill_ids.add(fill.fill_id)
        self._filled_quantity = new_filled
        self._average_fill_price = new_average_price
        self._cumulative_fees = new_fees
        self._status = new_status
        self._last_update_time = fill_time

    def transition_order_status(self, new_status: OrderStatus, timestamp: datetime) -> None:
        """Transitions the wrapped domain Order lifecycle status.

        Args:
            new_status: Target OrderStatus following ALLOWED_ORDER_TRANSITIONS.
            timestamp: Explicit UTC simulation timestamp of the transition.
        """
        self._order.transition_to(new_status, timestamp=timestamp)

    def cancel_remaining(self, timestamp: datetime, reason: str) -> None:
        """Cancels the remaining unfilled quantity.

        Preserves all historical fills, executed quantities, and average entry price.
        Leaves quantity becomes zero.
        """
        if self.is_terminal:
            raise ValueError(f"Cannot cancel order with execution status '{self._status.value}'")

        cancel_time = validate_utc_timestamp(timestamp)
        if cancel_time < self._submission_time:
            raise ValueError(
                f"Cancellation timestamp ({cancel_time.isoformat()}) cannot be earlier than submission_time ({self._submission_time.isoformat()})"
            )
        if cancel_time < self._last_update_time:
            raise ValueError(
                f"Cancellation timestamp ({cancel_time.isoformat()}) cannot be earlier than previous update ({self._last_update_time.isoformat()})"
            )

        if not reason or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Cancellation reason must be a non-empty string")

        # Atomic commit
        self._cancelled_remainder = self.unfilled_quantity
        self._status = ExecutionStatus.CANCELLED
        self._cancellation_timestamp = cancel_time
        self._cancellation_reason = reason.strip()
        self._last_update_time = cancel_time

    def attach_instrument_spec(self, spec: InstrumentSpecification) -> None:
        """Safely attaches an instrument specification to an unassigned execution state."""
        if not isinstance(spec, InstrumentSpecification):
            raise TypeError(f"spec must be an InstrumentSpecification, got {type(spec).__name__}")
        if spec.symbol != self._order.symbol:
            raise ValueError(
                f"Instrument specification symbol '{spec.symbol}' does not match order symbol '{self._order.symbol}'"
            )
        min_vol = _to_decimal(spec.min_volume, "min_volume")
        step_vol = _to_decimal(spec.volume_step, "volume_step")
        if self._original_quantity < min_vol:
            raise OrderValidationError(
                f"Quantity {self._original_quantity} is below minimum allowed volume {min_vol} for {self._order.symbol}"
            )
        if self._original_quantity % step_vol != Decimal(0):
            raise OrderValidationError(
                f"Quantity {self._original_quantity} does not align with volume_step {step_vol} for {self._order.symbol}"
            )
        if self._order.limit_price is not None:
            spec.validate_price(self._order.limit_price)
        if self._order.stop_loss_price is not None:
            spec.validate_price(self._order.stop_loss_price)
        if self._order.take_profit_price is not None:
            spec.validate_price(self._order.take_profit_price)

        self._instrument_spec = spec

    def to_snapshot(self) -> OrderExecutionSnapshot:
        """Creates an immutable public snapshot of the current execution state."""
        return OrderExecutionSnapshot(
            order=OrderSnapshot.from_order(self._order),
            submission_time=self._submission_time,
            original_quantity=self._original_quantity,
            filled_quantity=self._filled_quantity,
            unfilled_quantity=self.unfilled_quantity,
            leaves_quantity=self.leaves_quantity,
            cancelled_quantity=self.cancelled_quantity,
            average_fill_price=self._average_fill_price,
            cumulative_fees=dict(self._cumulative_fees),
            status=self._status,
            fills=tuple(self._fills),
            last_update_time=self._last_update_time,
            cancellation_timestamp=self._cancellation_timestamp,
            cancellation_reason=self._cancellation_reason,
        )
