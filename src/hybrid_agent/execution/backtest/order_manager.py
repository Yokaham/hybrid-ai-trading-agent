"""Core OrderManager for backtesting order lifecycle, state tracking, and registration."""

from datetime import datetime
from decimal import Decimal

from hybrid_agent.core.constants import ALLOWED_ORDER_TRANSITIONS, OrderStatus
from hybrid_agent.core.models import InstrumentSpecification, Order, validate_utc_timestamp
from hybrid_agent.execution.backtest.events import OrderCancelledEvent, OrderSubmittedEvent
from hybrid_agent.execution.backtest.state import (
    ExecutionStatus,
    FillRecord,
    OrderExecutionState,
)


class OrderManagerError(Exception):
    """Base exception for all OrderManager errors."""


class OrderNotFoundError(OrderManagerError, KeyError):
    """Raised when an order ID is not found in the registry."""


class DuplicateOrderError(OrderManagerError, ValueError):
    """Raised when an order ID is already registered with conflicting parameters."""


class DuplicateFillError(OrderManagerError, ValueError):
    """Raised when a fill ID has already been recorded across any order."""


class InvalidOrderStateError(OrderManagerError, ValueError):
    """Raised when an order operation violates valid execution or lifecycle states."""


class OrderManager:
    """Deterministic order lifecycle and execution-state coordinator for backtesting.

    Maintains permanent registry of orders, maps execution progress via OrderExecutionState,
    indexes active matching orders, records fills atomically, and coordinates cancellation.

    Strict Architectural Boundaries:
    - Maintains ZERO independent cash balances, position inventories, or margin collateral.
    - Performs NO price matching or fill calculation (delegated to FillEngine).
    - Delegates all authoritative accounting to Portfolio.apply_fill().
    """

    def __init__(self) -> None:
        self._orders: dict[str, OrderExecutionState] = {}
        self._order_sequence: list[str] = []
        self._registration_seq: dict[str, int] = {}
        self._seq_counter: int = 0
        self._active_order_ids: set[str] = set()
        self._fill_to_order_id: dict[str, str] = {}
        self._instrument_specs: dict[str, InstrumentSpecification] = {}

    def register_instrument(self, spec: InstrumentSpecification) -> None:
        """Registers an immutable instrument specification for order validation.

        Args:
            spec: The InstrumentSpecification entity.

        Raises:
            TypeError: If spec is not an InstrumentSpecification.
            ValueError: If a conflicting specification is already registered for this symbol.
        """
        if not isinstance(spec, InstrumentSpecification):
            raise TypeError(f"spec must be an InstrumentSpecification, got {type(spec).__name__}")
        if spec.symbol in self._instrument_specs:
            existing = self._instrument_specs[spec.symbol]
            if existing != spec:
                raise ValueError(
                    f"Conflicting instrument specification already registered for symbol '{spec.symbol}'"
                )
            return
        self._instrument_specs[spec.symbol] = spec

    def get_instrument_spec(self, symbol: str) -> InstrumentSpecification | None:
        """Returns the registered instrument specification for a symbol, if any."""
        return self._instrument_specs.get(symbol)

    def register_order(
        self,
        event: OrderSubmittedEvent,
        instrument_spec: InstrumentSpecification | None = None,
    ) -> OrderExecutionState:
        """Registers an approved or submitted order into the execution engine.

        Creates an OrderExecutionState snapshot using the explicit simulation submission timestamp.
        Guarantees that external mutations to the input order object do not alter the managed state.

        Args:
            event: The OrderSubmittedEvent containing the order and submission timestamp.
            instrument_spec: Optional InstrumentSpecification override or registration.

        Returns:
            The newly created or idempotently retrieved OrderExecutionState.

        Raises:
            TypeError: If event or order are of invalid types.
            ValueError: If timestamps are invalid or do not agree, or specifications conflict.
            InvalidOrderStateError: If the order is not in APPROVED or SUBMITTED status.
            DuplicateOrderError: If the order ID is already registered with conflicting parameters.
        """
        if not isinstance(event, OrderSubmittedEvent):
            raise TypeError(f"event must be an OrderSubmittedEvent, got {type(event).__name__}")

        ts = validate_utc_timestamp(event.timestamp)
        sub_time = validate_utc_timestamp(event.submission_time)
        if ts != sub_time:
            raise ValueError(
                f"event.timestamp ({ts.isoformat()}) must match event.submission_time ({sub_time.isoformat()})"
            )

        order = event.order
        if not isinstance(order, Order):
            raise TypeError(f"event.order must be an Order instance, got {type(order).__name__}")

        if not order.id or not isinstance(order.id, str) or not order.id.strip():
            raise ValueError("order.id must be a non-empty string")
        order_id = order.id.strip()

        # Resolve and validate instrument specification
        spec: InstrumentSpecification | None
        if instrument_spec is not None:
            if not isinstance(instrument_spec, InstrumentSpecification):
                raise TypeError(
                    f"instrument_spec must be an InstrumentSpecification, got {type(instrument_spec).__name__}"
                )
            if instrument_spec.symbol != order.symbol:
                raise ValueError(
                    f"Instrument specification symbol '{instrument_spec.symbol}' does not match order symbol '{order.symbol}'"
                )
            if (
                order.symbol in self._instrument_specs
                and self._instrument_specs[order.symbol] != instrument_spec
            ):
                raise ValueError(
                    f"Conflicting instrument specification for symbol '{order.symbol}' already registered"
                )
            self.register_instrument(instrument_spec)
            spec = instrument_spec
        else:
            spec = self._instrument_specs.get(order.symbol)

        # Handle duplicate registration
        if order_id in self._orders:
            existing_state = self._orders[order_id]
            existing_order = existing_state.order
            is_identical = (
                existing_order.symbol == order.symbol
                and existing_order.side == order.side
                and existing_order.order_type == order.order_type
                and Decimal(str(existing_order.quantity)) == Decimal(str(order.quantity))
                and existing_order.limit_price == order.limit_price
                and existing_order.stop_loss_price == order.stop_loss_price
                and existing_order.take_profit_price == order.take_profit_price
                and existing_state.submission_time == sub_time
            )
            if is_identical:
                return existing_state
            raise DuplicateOrderError(
                f"Order ID '{order_id}' is already registered with conflicting order parameters"
            )

        # Validate order lifecycle status
        if order.status not in (OrderStatus.APPROVED, OrderStatus.SUBMITTED):
            raise InvalidOrderStateError(
                f"Cannot register order '{order_id}' in lifecycle status '{order.status.value}'. "
                f"Order must be in APPROVED or SUBMITTED status."
            )

        # Protect original input order by creating a deep copy for manager ownership
        managed_order = order.model_copy(deep=True)
        if managed_order.status == OrderStatus.APPROVED:
            managed_order.transition_to(OrderStatus.SUBMITTED, timestamp=sub_time)

        # Initialize execution state
        state = OrderExecutionState(
            order=managed_order,
            submission_time=sub_time,
            instrument_spec=spec,
        )

        # Commit to internal registries
        self._orders[order_id] = state
        self._order_sequence.append(order_id)
        self._seq_counter += 1
        self._registration_seq[order_id] = self._seq_counter
        self._active_order_ids.add(order_id)

        return state

    def record_fill(self, fill: FillRecord) -> OrderExecutionState:
        """Records an execution fill against the owning order atomically.

        Args:
            fill: The immutable FillRecord to record.

        Returns:
            The updated OrderExecutionState.

        Raises:
            TypeError: If fill is not a FillRecord.
            OrderNotFoundError: If the fill references an unknown order ID.
            DuplicateFillError: If the fill ID was previously recorded.
            InvalidOrderStateError: If the order is already in a terminal state or cannot transition.
            ValueError: If fill parameters (symbol, side, quantity, price, etc.) fail validation.
        """
        if not isinstance(fill, FillRecord):
            raise TypeError(f"fill must be a FillRecord, got {type(fill).__name__}")

        if fill.order_id not in self._orders:
            raise OrderNotFoundError(f"Order '{fill.order_id}' not found in registry")

        if fill.fill_id in self._fill_to_order_id:
            raise DuplicateFillError(
                f"Fill ID '{fill.fill_id}' has already been recorded for order '{self._fill_to_order_id[fill.fill_id]}'"
            )

        state = self._orders[fill.order_id]
        if state.is_terminal:
            raise InvalidOrderStateError(
                f"Cannot record fill on terminal order '{fill.order_id}' with status '{state.status.value}'"
            )

        if fill.symbol != state.order.symbol:
            raise ValueError(
                f"Fill symbol '{fill.symbol}' does not match order symbol '{state.order.symbol}'"
            )

        if fill.side != state.order.side:
            raise ValueError(
                f"Fill side '{fill.side}' does not match order side '{state.order.side}'"
            )

        # Pre-validate core Order lifecycle transition if fill completes the order
        if fill.fill_quantity == state.leaves_quantity:
            allowed = ALLOWED_ORDER_TRANSITIONS.get(state.order.status, set())
            if OrderStatus.FILLED not in allowed:
                raise InvalidOrderStateError(
                    f"Cannot transition order '{state.order.id}' from '{state.order.status.value}' to 'FILLED'"
                )

        # Delegate execution state calculations and validations atomically.
        # If state.record_fill fails, an exception is raised before any manager state is updated.
        state.record_fill(fill)

        # Commit fill index
        self._fill_to_order_id[fill.fill_id] = fill.order_id

        # Synchronize macro lifecycle if complete
        if state.status == ExecutionStatus.FILLED:
            state.transition_order_status(OrderStatus.FILLED, timestamp=fill.timestamp)
            self._active_order_ids.discard(fill.order_id)

        return state

    def cancel_order(
        self,
        order_id: str,
        timestamp: datetime,
        reason: str,
    ) -> OrderCancelledEvent:
        """Cancels the remaining unfilled quantity of an active or partially filled order.

        Preserves all historical fills and executed quantities. Sets executable leaves to zero.
        Transitions the managed core Order to OrderStatus.CANCELLED and removes it from active collections.

        Args:
            order_id: Unique ID of the order to cancel.
            timestamp: Explicit UTC simulation timestamp of cancellation.
            reason: Non-empty description explaining the cancellation reason.

        Returns:
            The OrderCancelledEvent for downstream dispatch.

        Raises:
            OrderNotFoundError: If order_id is not registered.
            InvalidOrderStateError: If the order is already in a terminal state (FILLED or CANCELLED).
            ValueError: If reason is empty or timestamp violates temporal monotonicity.
        """
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order '{order_id}' not found in registry")

        state = self._orders[order_id]
        if state.is_terminal:
            raise InvalidOrderStateError(
                f"Cannot cancel order '{order_id}' with terminal execution status '{state.status.value}'"
            )

        cancel_time = validate_utc_timestamp(timestamp)
        if not reason or not isinstance(reason, str) or not reason.strip():
            raise ValueError("Cancellation reason must be a non-empty string")
        clean_reason = reason.strip()

        # Pre-validate core lifecycle transition
        allowed = ALLOWED_ORDER_TRANSITIONS.get(state.order.status, set())
        if OrderStatus.CANCELLED not in allowed:
            raise InvalidOrderStateError(
                f"Cannot transition order '{order_id}' from '{state.order.status.value}' to 'CANCELLED'"
            )

        # Atomically cancel remaining in execution state
        state.cancel_remaining(timestamp=cancel_time, reason=clean_reason)

        # Synchronize core order macro lifecycle
        state.transition_order_status(OrderStatus.CANCELLED, timestamp=cancel_time)

        # Remove from active collection
        self._active_order_ids.discard(order_id)

        return OrderCancelledEvent(
            timestamp=cancel_time,
            order=state.order,
            reason=clean_reason,
        )

    def get_execution_state(self, order_id: str) -> OrderExecutionState:
        """Retrieves the OrderExecutionState for a registered order.

        Args:
            order_id: The order ID to look up.

        Returns:
            The registered OrderExecutionState.

        Raises:
            OrderNotFoundError: If the order ID is not registered.
        """
        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order '{order_id}' not found in registry")
        return self._orders[order_id]

    def has_order(self, order_id: str) -> bool:
        """Returns True if the order ID has been registered, False otherwise."""
        return order_id in self._orders

    def get_active_orders(self, symbol: str | None = None) -> list[OrderExecutionState]:
        """Returns all currently active and partially filled orders.

        Ordered deterministically by submission_time, registration sequence, and order ID tie-breaker.
        Returned collection is a newly constructed list; modifying it does not alter internal state.

        Args:
            symbol: Optional symbol filter.

        Returns:
            A list of active OrderExecutionState instances.
        """
        active = [
            self._orders[oid]
            for oid in self._active_order_ids
            if self._orders[oid].is_active or self._orders[oid].is_partially_filled
        ]
        if symbol is not None:
            active = [s for s in active if s.order.symbol == symbol]

        active.sort(
            key=lambda s: (s.submission_time, self._registration_seq[s.order.id], s.order.id)
        )
        return active

    def get_all_orders(self, symbol: str | None = None) -> list[OrderExecutionState]:
        """Returns all registered orders including completed and cancelled orders.

        Preserves chronological registration sequence.
        Returned collection is a newly constructed list; modifying it does not alter internal state.

        Args:
            symbol: Optional symbol filter.

        Returns:
            A list of all OrderExecutionState instances.
        """
        all_orders = [self._orders[oid] for oid in self._order_sequence]
        if symbol is not None:
            all_orders = [s for s in all_orders if s.order.symbol == symbol]
        return all_orders

    def get_order_by_fill_id(self, fill_id: str) -> OrderExecutionState | None:
        """Looks up the owning OrderExecutionState by fill ID for historical traceability."""
        order_id = self._fill_to_order_id.get(fill_id)
        if order_id is None:
            return None
        return self._orders.get(order_id)

    @property
    def active_order_count(self) -> int:
        """Count of orders currently active or partially filled."""
        return len(self._active_order_ids)

    @property
    def total_order_count(self) -> int:
        """Count of all orders registered in the manager's lifetime."""
        return len(self._orders)
