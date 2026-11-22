"""Core OrderManager for backtesting order lifecycle, state tracking, and registration."""

from datetime import datetime
from decimal import Decimal

from hybrid_agent.core.constants import (
    ALLOWED_ORDER_TRANSITIONS,
    OrderSide,
    OrderStatus,
    OrderType,
)
from hybrid_agent.core.models import InstrumentSpecification, Order, validate_utc_timestamp
from hybrid_agent.execution.backtest.bracket import (
    Bracket,
    BracketExposureBreachError,
    BracketNotFoundError,
    BracketOrderRole,
    BracketSnapshot,
    BracketStatus,
    DuplicateBracketError,
    InvalidBracketStateError,
)
from hybrid_agent.execution.backtest.events import OrderCancelledEvent, OrderSubmittedEvent
from hybrid_agent.execution.backtest.state import (
    ExecutionStatus,
    FillRecord,
    OrderExecutionSnapshot,
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
    indexes active matching orders, records fills atomically, coordinates cancellation,
    and manages OCO bracket and contingent child-order relationships.

    Strict Architectural Boundaries:
    - Maintains ZERO independent cash balances, position inventories, or margin collateral.
    - Performs NO price matching or fill calculation (delegated to FillEngine).
    - Delegates all authoritative accounting to Portfolio.apply_fill().
    - Returns read-only OrderExecutionSnapshot and BracketSnapshot instances to prevent external mutation of internal registries.
    """

    def __init__(self) -> None:
        self._orders: dict[str, OrderExecutionState] = {}
        self._order_sequence: list[str] = []
        self._registration_seq: dict[str, int] = {}
        self._seq_counter: int = 0
        self._active_order_ids: set[str] = set()
        self._fill_to_order_id: dict[str, str] = {}
        self._instrument_specs: dict[str, InstrumentSpecification] = {}

        # OCO Bracket registries
        self._brackets: dict[str, Bracket] = {}
        self._bracket_sequence: list[str] = []
        self._order_to_bracket_id: dict[str, str] = {}
        self._order_roles: dict[str, BracketOrderRole] = {}

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

    @staticmethod
    def _validate_bracket_prices(
        order: Order, stop_loss_price: float, take_profit_price: float
    ) -> None:
        """Validates directional price rules between parent order and bracket exit legs."""
        if stop_loss_price <= 0.0:
            raise ValueError("stop_loss_price must be positive")
        if take_profit_price <= 0.0:
            raise ValueError("take_profit_price must be positive")

        if order.side == OrderSide.BUY:
            if stop_loss_price >= take_profit_price:
                raise ValueError(
                    f"BUY bracket directional error: stop_loss_price ({stop_loss_price}) must be strictly below take_profit_price ({take_profit_price})"
                )
            if order.limit_price is not None:
                if stop_loss_price >= order.limit_price:
                    raise ValueError(
                        f"BUY bracket directional error: stop_loss_price ({stop_loss_price}) must be strictly below limit_price ({order.limit_price})"
                    )
                if take_profit_price <= order.limit_price:
                    raise ValueError(
                        f"BUY bracket directional error: take_profit_price ({take_profit_price}) must be strictly above limit_price ({order.limit_price})"
                    )
        elif order.side == OrderSide.SELL:
            if take_profit_price >= stop_loss_price:
                raise ValueError(
                    f"SELL bracket directional error: take_profit_price ({take_profit_price}) must be strictly below stop_loss_price ({stop_loss_price})"
                )
            if order.limit_price is not None:
                if stop_loss_price <= order.limit_price:
                    raise ValueError(
                        f"SELL bracket directional error: stop_loss_price ({stop_loss_price}) must be strictly above limit_price ({order.limit_price})"
                    )
                if take_profit_price >= order.limit_price:
                    raise ValueError(
                        f"SELL bracket directional error: take_profit_price ({take_profit_price}) must be strictly below limit_price ({order.limit_price})"
                    )

    def register_order(
        self,
        event: OrderSubmittedEvent,
        instrument_spec: InstrumentSpecification | None = None,
    ) -> OrderExecutionSnapshot:
        """Registers an approved or submitted order into the execution engine.

        Creates an OrderExecutionState snapshot using the explicit simulation submission timestamp.
        Guarantees that external mutations to the input order object do not alter the managed state.
        If both stop_loss_price and take_profit_price are populated, automatically creates and links an OCO bracket.
        Returns an immutable OrderExecutionSnapshot.

        Args:
            event: The OrderSubmittedEvent containing the order and submission timestamp.
            instrument_spec: Optional InstrumentSpecification override or registration.

        Returns:
            An immutable OrderExecutionSnapshot of the newly registered or idempotently retrieved order.

        Raises:
            TypeError: If event or order are of invalid types.
            ValueError: If timestamps are invalid, out of chronological order, or specifications conflict.
            OrderValidationError: If order or bracket prices/quantities violate instrument specification rules.
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
        if order.id != order.id.strip():
            raise ValueError(
                f"Order ID cannot contain leading or trailing whitespace: '{order.id}'"
            )
        order_id = order.id

        # Validate chronological order creation/update vs submission timestamp
        if order.created_at > sub_time:
            raise ValueError(
                f"order.created_at ({order.created_at.isoformat()}) cannot be later than submission_time ({sub_time.isoformat()})"
            )
        if order.updated_at > sub_time:
            raise ValueError(
                f"order.updated_at ({order.updated_at.isoformat()}) cannot be later than submission_time ({sub_time.isoformat()})"
            )

        # Validate order lifecycle status first (before duplicate handling)
        if order.status not in (OrderStatus.APPROVED, OrderStatus.SUBMITTED):
            raise InvalidOrderStateError(
                f"Cannot register order '{order_id}' in lifecycle status '{order.status.value}'. "
                f"Order must be in APPROVED or SUBMITTED status."
            )

        # Resolve candidate instrument specification upfront
        candidate_spec: InstrumentSpecification | None
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
                if order_id in self._orders:
                    raise DuplicateOrderError(
                        f"Order ID '{order_id}' has conflicting instrument specification for symbol '{order.symbol}'"
                    )
                raise ValueError(
                    f"Conflicting instrument specification for symbol '{order.symbol}' already registered"
                )
            candidate_spec = instrument_spec
        else:
            candidate_spec = self._instrument_specs.get(order.symbol)

        # Pre-validate order and bracket parameters against candidate spec before ANY state mutation
        if candidate_spec is not None:
            order.validate_against_instrument(candidate_spec)

        if order.stop_loss_price is not None and order.take_profit_price is not None:
            self._validate_bracket_prices(order, order.stop_loss_price, order.take_profit_price)
            if candidate_spec is not None:
                candidate_spec.validate_price(order.stop_loss_price)
                candidate_spec.validate_price(order.take_profit_price)

        # Handle duplicate registration
        if order_id in self._orders:
            existing_state = self._orders[order_id]
            existing_order = existing_state.order

            if instrument_spec is not None:
                if (
                    existing_state.instrument_spec is not None
                    and existing_state.instrument_spec != instrument_spec
                ):
                    raise DuplicateOrderError(
                        f"Order ID '{order_id}' is already registered with a conflicting instrument specification"
                    )
                if (
                    order.symbol in self._instrument_specs
                    and self._instrument_specs[order.symbol] != instrument_spec
                ):
                    raise DuplicateOrderError(
                        f"Order ID '{order_id}' is already registered with a conflicting instrument specification"
                    )

            is_identical = (
                existing_order.symbol == order.symbol
                and existing_order.side == order.side
                and existing_order.order_type == order.order_type
                and Decimal(str(existing_order.quantity)) == Decimal(str(order.quantity))
                and existing_order.limit_price == order.limit_price
                and existing_order.stop_loss_price == order.stop_loss_price
                and existing_order.take_profit_price == order.take_profit_price
                and existing_order.created_at == order.created_at
                and existing_state.submission_time == sub_time
            )
            if is_identical:
                # Safe validated attachment of late instrument specification
                if instrument_spec is not None and existing_state.instrument_spec is None:
                    existing_state.attach_instrument_spec(instrument_spec)
                    self.register_instrument(instrument_spec)
                return existing_state.to_snapshot()
            raise DuplicateOrderError(
                f"Order ID '{order_id}' is already registered with conflicting order parameters"
            )

        # Register instrument spec in registry if new spec was provided
        spec: InstrumentSpecification | None
        if instrument_spec is not None:
            self.register_instrument(instrument_spec)
            spec = instrument_spec
        else:
            spec = candidate_spec

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

        # Commit to internal registries with transactional rollback protection
        self._orders[order_id] = state
        self._order_sequence.append(order_id)
        self._seq_counter += 1
        self._registration_seq[order_id] = self._seq_counter
        self._active_order_ids.add(order_id)

        try:
            # Auto-register bracket if both SL and TP are populated on the order
            if (
                managed_order.stop_loss_price is not None
                and managed_order.take_profit_price is not None
            ):
                self.register_bracket(
                    parent_order_id=order_id,
                    stop_loss_price=managed_order.stop_loss_price,
                    take_profit_price=managed_order.take_profit_price,
                )
        except Exception:
            # Atomic rollback: revert order registration if bracket creation fails
            self._orders.pop(order_id, None)
            if self._order_sequence and self._order_sequence[-1] == order_id:
                self._order_sequence.pop()
            self._registration_seq.pop(order_id, None)
            self._active_order_ids.discard(order_id)
            if order_id in self._order_to_bracket_id:
                b_id = self._order_to_bracket_id.pop(order_id)
                self._brackets.pop(b_id, None)
                if self._bracket_sequence and self._bracket_sequence[-1] == b_id:
                    self._bracket_sequence.pop()
            self._order_roles.pop(order_id, None)
            raise

        return state.to_snapshot()

    def register_bracket(
        self,
        parent_order_id: str,
        stop_loss_price: float,
        take_profit_price: float,
        bracket_id: str | None = None,
    ) -> BracketSnapshot:
        """Registers a deterministic OCO bracket linking a parent entry order to protective exit legs.

        Protective child legs will not be placed on the market until the parent entry order receives fills.
        Directional price consistency is enforced immediately.

        Args:
            parent_order_id: The ID of an existing non-terminal parent entry order.
            stop_loss_price: Protective stop price (opposes entry direction).
            take_profit_price: Profit target price (opposes entry direction).
            bracket_id: Optional explicit bracket ID. Defaults to 'BRACKET-{parent_order_id}'.

        Returns:
            An immutable BracketSnapshot.

        Raises:
            OrderNotFoundError: If parent_order_id is not registered.
            InvalidBracketStateError: If parent order is cancelled without any fills.
            DuplicateBracketError: If bracket_id or parent_order_id is already registered with conflicting parameters.
            ValueError: If directional prices violate entry direction or instrument rules.
        """
        if (
            not parent_order_id
            or not isinstance(parent_order_id, str)
            or not parent_order_id.strip()
        ):
            raise ValueError("parent_order_id must be a non-empty string")
        if parent_order_id != parent_order_id.strip():
            raise ValueError(
                f"Parent order ID cannot contain leading or trailing whitespace: '{parent_order_id}'"
            )

        if parent_order_id not in self._orders:
            raise OrderNotFoundError(f"Parent order '{parent_order_id}' not found in registry")

        parent_state = self._orders[parent_order_id]
        if (
            parent_state.is_terminal
            and parent_state.status == ExecutionStatus.CANCELLED
            and parent_state.filled_quantity == Decimal(0)
        ):
            raise InvalidBracketStateError(
                f"Cannot attach bracket to cancelled unfilled parent order '{parent_order_id}'"
            )

        # Idempotency / duplicate check for parent order
        if parent_order_id in self._order_to_bracket_id:
            existing_bid = self._order_to_bracket_id[parent_order_id]
            existing_bracket = self._brackets[existing_bid]
            if (
                (bracket_id is None or bracket_id == existing_bracket.bracket_id)
                and existing_bracket.stop_loss_price == stop_loss_price
                and existing_bracket.take_profit_price == take_profit_price
            ):
                return existing_bracket.to_snapshot()
            raise DuplicateBracketError(
                f"Parent order '{parent_order_id}' is already linked to bracket '{existing_bid}' with conflicting parameters"
            )

        # Clean / generate bracket ID
        if bracket_id is not None:
            if not isinstance(bracket_id, str) or not bracket_id.strip():
                raise ValueError("bracket_id must be a non-empty string")
            if bracket_id != bracket_id.strip():
                raise ValueError(
                    f"Bracket ID cannot contain leading or trailing whitespace: '{bracket_id}'"
                )
            clean_bracket_id = bracket_id
            if clean_bracket_id in self._brackets:
                existing_bracket = self._brackets[clean_bracket_id]
                if (
                    existing_bracket.parent_order_id == parent_order_id
                    and existing_bracket.stop_loss_price == stop_loss_price
                    and existing_bracket.take_profit_price == take_profit_price
                ):
                    return existing_bracket.to_snapshot()
                raise DuplicateBracketError(
                    f"Bracket ID '{clean_bracket_id}' is already registered with conflicting parameters"
                )
        else:
            clean_bracket_id = f"BRACKET-{parent_order_id}"
            if clean_bracket_id in self._brackets:
                existing_bracket = self._brackets[clean_bracket_id]
                if (
                    existing_bracket.parent_order_id == parent_order_id
                    and existing_bracket.stop_loss_price == stop_loss_price
                    and existing_bracket.take_profit_price == take_profit_price
                ):
                    return existing_bracket.to_snapshot()
                raise DuplicateBracketError(
                    f"Generated bracket ID '{clean_bracket_id}' is already registered with conflicting parameters"
                )

        # Directional price validation
        self._validate_bracket_prices(parent_state.order, stop_loss_price, take_profit_price)

        # Spec validation for prices if instrument spec is registered
        spec = parent_state.instrument_spec or self._instrument_specs.get(parent_state.order.symbol)
        if spec is not None:
            spec.validate_price(stop_loss_price)
            spec.validate_price(take_profit_price)

        # Create Bracket entity
        bracket = Bracket(
            bracket_id=clean_bracket_id,
            parent_order_id=parent_order_id,
            symbol=parent_state.order.symbol,
            entry_side=parent_state.order.side,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            created_at=parent_state.submission_time,
        )

        self._brackets[clean_bracket_id] = bracket
        self._bracket_sequence.append(clean_bracket_id)
        self._order_to_bracket_id[parent_order_id] = clean_bracket_id
        self._order_roles[parent_order_id] = BracketOrderRole.PARENT_ENTRY

        # If parent order already had fills before bracket registration, activate child orders immediately
        if parent_state.filled_quantity > Decimal(0):
            bracket.allocated_quantity = parent_state.filled_quantity
            try:
                self._spawn_child_orders(
                    bracket,
                    quantity=bracket.residual_quantity,
                    timestamp=parent_state.last_update_time,
                )
            except Exception:
                # Rollback bracket registration if child creation fails
                self._brackets.pop(clean_bracket_id, None)
                if self._bracket_sequence and self._bracket_sequence[-1] == clean_bracket_id:
                    self._bracket_sequence.pop()
                self._order_to_bracket_id.pop(parent_order_id, None)
                self._order_roles.pop(parent_order_id, None)
                raise
            bracket.status = BracketStatus.ACTIVE

        return bracket.to_snapshot()

    def _replace_single_child_leg(
        self,
        bracket: Bracket,
        role: BracketOrderRole,
        quantity: Decimal,
        timestamp: datetime,
    ) -> str:
        """Spawns and registers a single contingent child leg (SL or TP) sized to quantity."""
        if quantity <= Decimal(0):
            raise ValueError(f"Cannot spawn child order with non-positive quantity: {quantity}")

        child_side = OrderSide.SELL if bracket.entry_side == OrderSide.BUY else OrderSide.BUY
        spec = self._instrument_specs.get(bracket.symbol)

        if role == BracketOrderRole.STOP_LOSS:
            count = len(bracket.stop_loss_order_ids) + 1
            order_id = f"{bracket.bracket_id}-SL-{count}"
            order = Order(
                id=order_id,
                symbol=bracket.symbol,
                side=child_side,
                order_type=OrderType.STOP_LOSS,
                quantity=float(quantity),
                stop_loss_price=bracket.stop_loss_price,
                status=OrderStatus.APPROVED,
                created_at=timestamp,
                updated_at=timestamp,
            )
            if spec is not None:
                order.validate_against_instrument(spec)
            event = OrderSubmittedEvent(order=order, timestamp=timestamp, submission_time=timestamp)
            self.register_order(event, instrument_spec=spec)
            self._order_to_bracket_id[order_id] = bracket.bracket_id
            self._order_roles[order_id] = BracketOrderRole.STOP_LOSS
            bracket.active_stop_loss_order_id = order_id
            bracket.stop_loss_order_ids.append(order_id)
            bracket.updated_at = timestamp
            return order_id

        elif role == BracketOrderRole.TAKE_PROFIT:
            count = len(bracket.take_profit_order_ids) + 1
            order_id = f"{bracket.bracket_id}-TP-{count}"
            order = Order(
                id=order_id,
                symbol=bracket.symbol,
                side=child_side,
                order_type=OrderType.LIMIT,
                quantity=float(quantity),
                limit_price=bracket.take_profit_price,
                status=OrderStatus.APPROVED,
                created_at=timestamp,
                updated_at=timestamp,
            )
            if spec is not None:
                order.validate_against_instrument(spec)
            event = OrderSubmittedEvent(order=order, timestamp=timestamp, submission_time=timestamp)
            self.register_order(event, instrument_spec=spec)
            self._order_to_bracket_id[order_id] = bracket.bracket_id
            self._order_roles[order_id] = BracketOrderRole.TAKE_PROFIT
            bracket.active_take_profit_order_id = order_id
            bracket.take_profit_order_ids.append(order_id)
            bracket.updated_at = timestamp
            return order_id

        else:
            raise ValueError(f"Invalid child order role '{role}'")

    def _spawn_child_orders(
        self,
        bracket: Bracket,
        quantity: Decimal,
        timestamp: datetime,
    ) -> None:
        """Spawns both SL and TP child legs sized to the given quantity."""
        if quantity <= Decimal(0):
            return
        self._replace_single_child_leg(bracket, BracketOrderRole.STOP_LOSS, quantity, timestamp)
        self._replace_single_child_leg(bracket, BracketOrderRole.TAKE_PROFIT, quantity, timestamp)

    def _reconcile_bracket_exit_exposure(
        self,
        bracket: Bracket,
        filled_order_id: str,
        timestamp: datetime,
    ) -> None:
        """Reconciles all active protective child legs and parent remainder against bracket residual exposure.

        Enforces:
        - When residual_quantity == 0:
          - The bracket transitions to CLOSED.
          - Every active child order in the bracket is cancelled.
          - If the order that just filled still has executable leaves, its remaining leaves are cancelled.
          - If the parent entry order still has executable leaves, the parent remainder is cancelled.
          - Active child references are cleared only after orders are made non-executable.
        - When residual_quantity > 0:
          - Reconciles both STOP_LOSS and TAKE_PROFIT active legs.
          - If an active child has leaves > residual_quantity, it is cancelled and replaced with residual_quantity.
          - If an active child has reached a terminal state with 0 leaves, a replacement child is spawned for residual_quantity.
          - Guarantees neither active leg has leaves > residual_quantity.
          - Preserves all historical order records and fill history.
        """
        if bracket.residual_quantity == Decimal(0):
            bracket.status = BracketStatus.CLOSED

            # Cancel active stop loss if active
            if (
                bracket.active_stop_loss_order_id
                and bracket.active_stop_loss_order_id in self._active_order_ids
            ):
                self.cancel_order(
                    bracket.active_stop_loss_order_id,
                    timestamp=timestamp,
                    reason="Bracket fully closed by exit fill",
                )

            # Cancel active take profit if active
            if (
                bracket.active_take_profit_order_id
                and bracket.active_take_profit_order_id in self._active_order_ids
            ):
                self.cancel_order(
                    bracket.active_take_profit_order_id,
                    timestamp=timestamp,
                    reason="Bracket fully closed by exit fill",
                )

            # Cancel remaining leaves on filled order if any
            filled_state = self._orders.get(filled_order_id)
            if (
                filled_state is not None
                and not filled_state.is_terminal
                and filled_state.leaves_quantity > Decimal(0)
            ):
                self.cancel_order(
                    filled_order_id,
                    timestamp=timestamp,
                    reason="Bracket fully closed by exit fill",
                )

            # Cancel parent entry remainder if parent order still has executable leaves
            parent_state = self._orders.get(bracket.parent_order_id)
            if (
                parent_state is not None
                and not parent_state.is_terminal
                and parent_state.leaves_quantity > Decimal(0)
            ):
                self.cancel_order(
                    bracket.parent_order_id,
                    timestamp=timestamp,
                    reason="Bracket fully closed by exit fill",
                )

            bracket.active_stop_loss_order_id = None
            bracket.active_take_profit_order_id = None
            bracket.updated_at = timestamp
            return

        target_qty = bracket.residual_quantity

        # 1. Reconcile Stop-Loss leg
        sl_id = bracket.active_stop_loss_order_id
        if sl_id is not None and sl_id in self._orders:
            sl_state = self._orders[sl_id]
            if sl_state.leaves_quantity > target_qty:
                self.cancel_order(
                    sl_id,
                    timestamp=timestamp,
                    reason=f"Bracket residual exposure reduction to {target_qty}",
                )
                self._replace_single_child_leg(
                    bracket, BracketOrderRole.STOP_LOSS, target_qty, timestamp
                )
            elif sl_state.is_terminal and sl_state.leaves_quantity == Decimal(0):
                self._replace_single_child_leg(
                    bracket, BracketOrderRole.STOP_LOSS, target_qty, timestamp
                )
        else:
            self._replace_single_child_leg(
                bracket, BracketOrderRole.STOP_LOSS, target_qty, timestamp
            )

        # 2. Reconcile Take-Profit leg
        tp_id = bracket.active_take_profit_order_id
        if tp_id is not None and tp_id in self._orders:
            tp_state = self._orders[tp_id]
            if tp_state.leaves_quantity > target_qty:
                self.cancel_order(
                    tp_id,
                    timestamp=timestamp,
                    reason=f"Bracket residual exposure reduction to {target_qty}",
                )
                self._replace_single_child_leg(
                    bracket, BracketOrderRole.TAKE_PROFIT, target_qty, timestamp
                )
            elif tp_state.is_terminal and tp_state.leaves_quantity == Decimal(0):
                self._replace_single_child_leg(
                    bracket, BracketOrderRole.TAKE_PROFIT, target_qty, timestamp
                )
        else:
            self._replace_single_child_leg(
                bracket, BracketOrderRole.TAKE_PROFIT, target_qty, timestamp
            )

        bracket.updated_at = timestamp

    def record_fill(
        self,
        fill: FillRecord,
        allow_late_fill: bool = False,
        ignore_duplicate: bool = False,
    ) -> OrderExecutionSnapshot:
        """Records an execution fill against the owning order atomically.

        Coordinates OCO bracket updates, dynamic cancel-and-replace scaling,
        and exposure breach tracking for contingent orders.

        Args:
            fill: The immutable FillRecord to record.
            allow_late_fill: Whether to explicitly permit recording late fills arriving on cancelled orders.
            ignore_duplicate: If True, duplicate fill delivery returns existing snapshot idempotently.

        Returns:
            An immutable OrderExecutionSnapshot reflecting the updated state.

        Raises:
            TypeError: If fill is not a FillRecord.
            OrderNotFoundError: If the fill references an unknown order ID.
            DuplicateFillError: If the fill ID was previously recorded.
            InvalidOrderStateError: If the order is already in a terminal state or cannot transition.
            InvalidBracketStateError: If an entry fill is submitted against a closed bracket.
            BracketExposureBreachError: If exit fills exceed the bracket's allocated entry exposure.
            ValueError: If fill parameters (symbol, side, quantity, price, etc.) fail validation.
        """
        if not isinstance(fill, FillRecord):
            raise TypeError(f"fill must be a FillRecord, got {type(fill).__name__}")

        if fill.fill_id in self._fill_to_order_id:
            if ignore_duplicate and self._fill_to_order_id[fill.fill_id] == fill.order_id:
                return self._orders[fill.order_id].to_snapshot()
            raise DuplicateFillError(
                f"Fill ID '{fill.fill_id}' has already been recorded for order '{self._fill_to_order_id[fill.fill_id]}'"
            )

        if fill.order_id not in self._orders:
            raise OrderNotFoundError(f"Order '{fill.order_id}' not found in registry")

        # Resolve bracket association and role
        bracket_id = self._order_to_bracket_id.get(fill.order_id)
        role = self._order_roles.get(fill.order_id)
        bracket = self._brackets.get(bracket_id) if bracket_id else None

        state = self._orders[fill.order_id]

        # Explicit late-fill permission: cancelled orders reject fills unless caller explicitly permits
        if state.is_terminal and not (allow_late_fill and state.is_cancelled):
            raise InvalidOrderStateError(
                f"Cannot record fill on terminal order '{fill.order_id}' with status '{state.status.value}'"
            )

        # Do not allow additional ordinary parent fills to create new exposure under a closed bracket
        if role == BracketOrderRole.PARENT_ENTRY and bracket is not None and bracket.is_closed:
            raise InvalidBracketStateError(
                f"Cannot record parent entry fill for closed bracket '{bracket.bracket_id}'"
            )

        if fill.symbol != state.order.symbol:
            raise ValueError(
                f"Fill symbol '{fill.symbol}' does not match order symbol '{state.order.symbol}'"
            )

        if fill.side != state.order.side:
            raise ValueError(
                f"Fill side '{fill.side}' does not match order side '{state.order.side}'"
            )

        # Exposure breach evaluation for bracket exit fills
        breach_occurred = False
        breach_message = ""
        if bracket is not None and role in (
            BracketOrderRole.STOP_LOSS,
            BracketOrderRole.TAKE_PROFIT,
        ):
            potential_closed = bracket.closed_quantity + fill.fill_quantity
            if potential_closed > bracket.allocated_quantity:
                breach_occurred = True
                breach_message = (
                    f"Bracket '{bracket.bracket_id}' exposure breach: exit fill '{fill.fill_id}' for quantity "
                    f"{fill.fill_quantity} brings total closed quantity to {potential_closed}, exceeding "
                    f"allocated entry quantity {bracket.allocated_quantity}"
                )

        # Pre-validate core Order lifecycle transition if fill completes the order
        if not state.is_cancelled and fill.fill_quantity == state.leaves_quantity:
            allowed = ALLOWED_ORDER_TRANSITIONS.get(state.order.status, set())
            if OrderStatus.FILLED not in allowed:
                raise InvalidOrderStateError(
                    f"Cannot transition order '{state.order.id}' from '{state.order.status.value}' to 'FILLED'"
                )

        # Delegate execution state calculations and validations atomically.
        state.record_fill(fill, allow_late_fill=allow_late_fill)

        # Commit fill index
        self._fill_to_order_id[fill.fill_id] = fill.order_id

        # Synchronize macro lifecycle if complete
        if state.status == ExecutionStatus.FILLED:
            state.transition_order_status(OrderStatus.FILLED, timestamp=fill.timestamp)
            self._active_order_ids.discard(fill.order_id)

        # Coordinate Bracket lifecycle mutations
        if breach_occurred and bracket is not None:
            bracket.closed_quantity += fill.fill_quantity
            bracket.has_exposure_breach = True
            bracket.status = BracketStatus.BREACHED
            bracket.breach_details = breach_message
            bracket.updated_at = fill.timestamp
            # Cancel active legs to halt ordinary trading on breached bracket
            if (
                bracket.active_stop_loss_order_id
                and bracket.active_stop_loss_order_id in self._active_order_ids
            ):
                self.cancel_order(
                    bracket.active_stop_loss_order_id,
                    timestamp=fill.timestamp,
                    reason="Bracket exposure breach - cancelling active legs",
                )
            if (
                bracket.active_take_profit_order_id
                and bracket.active_take_profit_order_id in self._active_order_ids
            ):
                self.cancel_order(
                    bracket.active_take_profit_order_id,
                    timestamp=fill.timestamp,
                    reason="Bracket exposure breach - cancelling active legs",
                )
            bracket.active_stop_loss_order_id = None
            bracket.active_take_profit_order_id = None
            raise BracketExposureBreachError(breach_message)

        if bracket is not None:
            if role == BracketOrderRole.PARENT_ENTRY:
                bracket.allocated_quantity += fill.fill_quantity
                bracket.updated_at = fill.timestamp
                # If bracket is closed, no children are spawned
                if not bracket.is_closed:
                    # Cancel active existing children for clean cancel-and-replace scale-up
                    if (
                        bracket.active_stop_loss_order_id
                        and bracket.active_stop_loss_order_id in self._active_order_ids
                    ):
                        self.cancel_order(
                            bracket.active_stop_loss_order_id,
                            timestamp=fill.timestamp,
                            reason=f"Bracket scale-up replacement to {bracket.residual_quantity}",
                        )
                    if (
                        bracket.active_take_profit_order_id
                        and bracket.active_take_profit_order_id in self._active_order_ids
                    ):
                        self.cancel_order(
                            bracket.active_take_profit_order_id,
                            timestamp=fill.timestamp,
                            reason=f"Bracket scale-up replacement to {bracket.residual_quantity}",
                        )
                    # Spawn replacement children sized to new residual quantity
                    self._spawn_child_orders(
                        bracket,
                        quantity=bracket.residual_quantity,
                        timestamp=fill.timestamp,
                    )
                    bracket.status = BracketStatus.ACTIVE

            elif role in (BracketOrderRole.STOP_LOSS, BracketOrderRole.TAKE_PROFIT):
                bracket.closed_quantity += fill.fill_quantity
                bracket.updated_at = fill.timestamp
                self._reconcile_bracket_exit_exposure(
                    bracket,
                    filled_order_id=fill.order_id,
                    timestamp=fill.timestamp,
                )

        return state.to_snapshot()

    def cancel_order(
        self,
        order_id: str,
        timestamp: datetime,
        reason: str,
    ) -> OrderCancelledEvent:
        """Cancels the remaining unfilled quantity of an active or partially filled order.

        Preserves all historical fills and executed quantities. Sets executable leaves to zero.
        Transitions the managed core Order to OrderStatus.CANCELLED and removes it from active collections.
        If a parent entry order is cancelled before any fills, the attached bracket is marked CANCELLED.
        If a parent entry order is cancelled after partial fills or after bracket closure, bracket status is preserved.

        Args:
            order_id: Unique ID of the order to cancel.
            timestamp: Explicit UTC simulation timestamp of cancellation.
            reason: Non-empty description explaining the cancellation reason.

        Returns:
            The OrderCancelledEvent for downstream dispatch, containing an independent order copy.

        Raises:
            OrderNotFoundError: If order_id is not registered.
            InvalidOrderStateError: If the order is already in a terminal state (FILLED or CANCELLED).
            ValueError: If reason is empty or timestamp violates temporal monotonicity.
        """
        if not order_id or not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if order_id != order_id.strip():
            raise ValueError(
                f"Order ID cannot contain leading or trailing whitespace: '{order_id}'"
            )

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

        # Coordinate Bracket lifecycle hooks
        bracket_id = self._order_to_bracket_id.get(order_id)
        role = self._order_roles.get(order_id)
        bracket = self._brackets.get(bracket_id) if bracket_id else None

        if bracket is not None:
            if role == BracketOrderRole.PARENT_ENTRY:
                if bracket.status == BracketStatus.CLOSED:
                    # Bracket was already CLOSED: preserve CLOSED status!
                    bracket.updated_at = cancel_time
                elif state.filled_quantity == Decimal(0):
                    bracket.status = BracketStatus.CANCELLED
                    bracket.updated_at = cancel_time
                    if (
                        bracket.active_stop_loss_order_id
                        and bracket.active_stop_loss_order_id in self._active_order_ids
                    ):
                        self.cancel_order(
                            bracket.active_stop_loss_order_id,
                            timestamp=cancel_time,
                            reason=f"Parent entry cancelled: {clean_reason}",
                        )
                    if (
                        bracket.active_take_profit_order_id
                        and bracket.active_take_profit_order_id in self._active_order_ids
                    ):
                        self.cancel_order(
                            bracket.active_take_profit_order_id,
                            timestamp=cancel_time,
                            reason=f"Parent entry cancelled: {clean_reason}",
                        )
                else:
                    # Parent had partial fills: protective child orders remain active
                    bracket.updated_at = cancel_time
            elif (
                role == BracketOrderRole.STOP_LOSS and bracket.active_stop_loss_order_id == order_id
            ):
                bracket.active_stop_loss_order_id = None
                bracket.updated_at = cancel_time
            elif (
                role == BracketOrderRole.TAKE_PROFIT
                and bracket.active_take_profit_order_id == order_id
            ):
                bracket.active_take_profit_order_id = None
                bracket.updated_at = cancel_time

        return OrderCancelledEvent(
            timestamp=cancel_time,
            order=state.order,  # state.order returns a defensive deep copy
            reason=clean_reason,
        )

    def get_execution_state(self, order_id: str) -> OrderExecutionSnapshot:
        """Retrieves an immutable OrderExecutionSnapshot for a registered order.

        Args:
            order_id: The order ID to look up.

        Returns:
            The immutable OrderExecutionSnapshot.

        Raises:
            OrderNotFoundError: If the order ID is not registered.
            ValueError: If order_id contains invalid whitespace.
        """
        if not order_id or not isinstance(order_id, str) or not order_id.strip():
            raise ValueError("order_id must be a non-empty string")
        if order_id != order_id.strip():
            raise ValueError(
                f"Order ID cannot contain leading or trailing whitespace: '{order_id}'"
            )

        if order_id not in self._orders:
            raise OrderNotFoundError(f"Order '{order_id}' not found in registry")
        return self._orders[order_id].to_snapshot()

    def has_order(self, order_id: str) -> bool:
        """Returns True if the order ID has been registered, False otherwise."""
        if not order_id or not isinstance(order_id, str) or order_id != order_id.strip():
            return False
        return order_id in self._orders

    def get_active_orders(self, symbol: str | None = None) -> list[OrderExecutionSnapshot]:
        """Returns immutable snapshots of all currently active and partially filled orders.

        Ordered deterministically by submission_time, registration sequence, and order ID tie-breaker.
        Returned collection is a newly constructed list of immutable snapshots.

        Args:
            symbol: Optional symbol filter.

        Returns:
            A list of active OrderExecutionSnapshot instances.
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
        return [s.to_snapshot() for s in active]

    def get_all_orders(self, symbol: str | None = None) -> list[OrderExecutionSnapshot]:
        """Returns immutable snapshots of all registered orders including completed and cancelled orders.

        Preserves chronological registration sequence.
        Returned collection is a newly constructed list of immutable snapshots.

        Args:
            symbol: Optional symbol filter.

        Returns:
            A list of all OrderExecutionSnapshot instances.
        """
        all_orders = [self._orders[oid] for oid in self._order_sequence]
        if symbol is not None:
            all_orders = [s for s in all_orders if s.order.symbol == symbol]
        return [s.to_snapshot() for s in all_orders]

    def get_order_by_fill_id(self, fill_id: str) -> OrderExecutionSnapshot | None:
        """Looks up the owning OrderExecutionSnapshot by fill ID for historical traceability."""
        order_id = self._fill_to_order_id.get(fill_id)
        if order_id is None:
            return None
        state = self._orders.get(order_id)
        return state.to_snapshot() if state is not None else None

    @property
    def active_order_count(self) -> int:
        """Count of orders currently active or partially filled."""
        return len(self._active_order_ids)

    @property
    def total_order_count(self) -> int:
        """Count of all orders registered in the manager's lifetime."""
        return len(self._orders)

    # Bracket Query APIs

    def get_bracket(self, bracket_id: str) -> BracketSnapshot:
        """Retrieves an immutable BracketSnapshot for a registered bracket.

        Args:
            bracket_id: The bracket ID to look up.

        Returns:
            The immutable BracketSnapshot.

        Raises:
            BracketNotFoundError: If bracket_id is not registered.
            ValueError: If bracket_id contains invalid whitespace.
        """
        if not bracket_id or not isinstance(bracket_id, str) or bracket_id != bracket_id.strip():
            raise ValueError(f"Invalid bracket_id '{bracket_id}'")
        if bracket_id not in self._brackets:
            raise BracketNotFoundError(f"Bracket '{bracket_id}' not found in registry")
        return self._brackets[bracket_id].to_snapshot()

    def has_bracket(self, bracket_id: str) -> bool:
        """Returns True if the bracket ID has been registered, False otherwise."""
        if not bracket_id or not isinstance(bracket_id, str) or bracket_id != bracket_id.strip():
            return False
        return bracket_id in self._brackets

    def get_bracket_for_order(self, order_id: str) -> BracketSnapshot | None:
        """Retrieves the BracketSnapshot for the bracket associated with this order, if any."""
        if not order_id or not isinstance(order_id, str) or order_id != order_id.strip():
            return None
        bracket_id = self._order_to_bracket_id.get(order_id)
        if not bracket_id:
            return None
        return self._brackets[bracket_id].to_snapshot()

    def get_order_role(self, order_id: str) -> BracketOrderRole | None:
        """Retrieves the BracketOrderRole for an order, or None if independent."""
        return self._order_roles.get(order_id)

    def get_active_brackets(self, symbol: str | None = None) -> list[BracketSnapshot]:
        """Returns immutable snapshots of all currently active brackets with open protective child orders.

        Args:
            symbol: Optional symbol filter.

        Returns:
            A list of active BracketSnapshot instances.
        """
        active = [b for b in self._brackets.values() if b.status == BracketStatus.ACTIVE]
        if symbol is not None:
            active = [b for b in active if b.symbol == symbol]
        active.sort(key=lambda b: (b.created_at, b.bracket_id))
        return [b.to_snapshot() for b in active]

    def get_all_brackets(self, symbol: str | None = None) -> list[BracketSnapshot]:
        """Returns immutable snapshots of all registered brackets across all lifecycle states.

        Args:
            symbol: Optional symbol filter.

        Returns:
            A list of all BracketSnapshot instances.
        """
        brackets = [self._brackets[bid] for bid in self._bracket_sequence]
        if symbol is not None:
            brackets = [b for b in brackets if b.symbol == symbol]
        return [b.to_snapshot() for b in brackets]

    @property
    def active_bracket_count(self) -> int:
        """Count of brackets currently in ACTIVE status with open child orders."""
        return sum(1 for b in self._brackets.values() if b.status == BracketStatus.ACTIVE)

    @property
    def total_bracket_count(self) -> int:
        """Count of all brackets registered in the manager's lifetime."""
        return len(self._brackets)
