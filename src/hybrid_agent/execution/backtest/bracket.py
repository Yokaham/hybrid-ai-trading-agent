"""OCO (One-Cancels-the-Other) bracket and contingent order models for native backtesting.

Provides:
- BracketStatus: lifecycle state of an OCO bracket.
- BracketOrderRole: role of an order within an OCO bracket group (parent entry, stop-loss, take-profit).
- BracketSnapshot: immutable public snapshot of a bracket's exposure and active child legs.
- Bracket: internal state entity coordinating parent entry fills and contingent exit orders.
- Bracket exception hierarchy.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from hybrid_agent.core.constants import OrderSide

if TYPE_CHECKING:
    from hybrid_agent.execution.backtest.order_manager import OrderManagerError
else:
    try:
        from hybrid_agent.execution.backtest.order_manager import OrderManagerError
    except ImportError:

        class OrderManagerError(Exception):  # type: ignore[no-redef]
            """Fallback base exception for OrderManager errors."""


class BracketStatus(str, Enum):
    """Lifecycle status of an OCO bracket."""

    PENDING_ENTRY = "PENDING_ENTRY"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"
    BREACHED = "BREACHED"


class BracketOrderRole(str, Enum):
    """Role of an order within an OCO bracket."""

    PARENT_ENTRY = "PARENT_ENTRY"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class BracketError(OrderManagerError):
    """Base exception for bracket errors."""


class BracketNotFoundError(BracketError, KeyError):
    """Raised when a bracket ID is not found."""


class DuplicateBracketError(BracketError, ValueError):
    """Raised when duplicate bracket registration conflicts."""


class InvalidBracketStateError(BracketError, ValueError):
    """Raised on invalid bracket lifecycle transitions or operations."""


class BracketExposureBreachError(BracketError, ValueError):
    """Raised when exit fills exceed the bracket's allocated entry exposure."""


@dataclass(frozen=True)
class BracketSnapshot:
    """Immutable public snapshot of an OCO bracket's state.

    Exposes bracket identity, allocated filled exposure, residual quantity,
    and active contingent child legs without leaking mutable state.
    """

    bracket_id: str
    parent_order_id: str
    symbol: str
    entry_side: OrderSide
    status: BracketStatus
    allocated_quantity: Decimal
    closed_quantity: Decimal
    residual_quantity: Decimal
    stop_loss_price: float
    take_profit_price: float
    active_stop_loss_order_id: str | None
    active_take_profit_order_id: str | None
    stop_loss_order_ids: tuple[str, ...]
    take_profit_order_ids: tuple[str, ...]
    has_exposure_breach: bool
    breach_details: str | None
    created_at: datetime
    updated_at: datetime

    @property
    def is_pending_entry(self) -> bool:
        """Whether the bracket is waiting for the parent entry order to produce fills."""
        return self.status == BracketStatus.PENDING_ENTRY

    @property
    def is_active(self) -> bool:
        """Whether the bracket has active protective child orders on the market."""
        return self.status == BracketStatus.ACTIVE

    @property
    def is_closed(self) -> bool:
        """Whether the bracket has been completely closed by exit fills."""
        return self.status == BracketStatus.CLOSED

    @property
    def is_cancelled(self) -> bool:
        """Whether the bracket was cancelled before receiving any entry fills."""
        return self.status == BracketStatus.CANCELLED

    @property
    def is_breached(self) -> bool:
        """Whether exit fills exceeded the bracket's allocated entry exposure."""
        return self.status == BracketStatus.BREACHED


class Bracket:
    """Internal state entity managing an OCO bracket and its contingent child legs.

    Strict Architectural Boundaries:
    - Maintains only group-level quantity allocation (allocated_quantity, closed_quantity).
    - Contains ZERO cash balances, position accounting, or margin calculations.
    - All authoritative accounting remains strictly with Portfolio.apply_fill().
    """

    def __init__(
        self,
        bracket_id: str,
        parent_order_id: str,
        symbol: str,
        entry_side: OrderSide,
        stop_loss_price: float,
        take_profit_price: float,
        created_at: datetime,
    ) -> None:
        self.bracket_id = bracket_id
        self.parent_order_id = parent_order_id
        self.symbol = symbol
        self.entry_side = entry_side
        self.stop_loss_price = stop_loss_price
        self.take_profit_price = take_profit_price
        self.status = BracketStatus.PENDING_ENTRY
        self.allocated_quantity = Decimal(0)
        self.closed_quantity = Decimal(0)
        self.active_stop_loss_order_id: str | None = None
        self.active_take_profit_order_id: str | None = None
        self.stop_loss_order_ids: list[str] = []
        self.take_profit_order_ids: list[str] = []
        self.has_exposure_breach = False
        self.breach_details: str | None = None
        self.created_at = created_at
        self.updated_at = created_at

    @property
    def residual_quantity(self) -> Decimal:
        """Remaining unclosed entry exposure allocated to this bracket."""
        return max(Decimal(0), self.allocated_quantity - self.closed_quantity)

    @property
    def is_pending_entry(self) -> bool:
        """Whether the bracket is waiting for entry fills."""
        return self.status == BracketStatus.PENDING_ENTRY

    @property
    def is_active(self) -> bool:
        """Whether the bracket has active protective child orders on the market."""
        return self.status == BracketStatus.ACTIVE

    @property
    def is_closed(self) -> bool:
        """Whether the bracket has been completely closed by exit fills."""
        return self.status == BracketStatus.CLOSED

    @property
    def is_cancelled(self) -> bool:
        """Whether the bracket was cancelled before receiving any entry fills."""
        return self.status == BracketStatus.CANCELLED

    @property
    def is_breached(self) -> bool:
        """Whether exit fills exceeded the bracket's allocated entry exposure."""
        return self.status == BracketStatus.BREACHED

    @property
    def is_terminal(self) -> bool:
        """Whether the bracket is in a terminal state."""
        return self.status in (
            BracketStatus.CLOSED,
            BracketStatus.CANCELLED,
            BracketStatus.BREACHED,
        )

    def to_snapshot(self) -> BracketSnapshot:
        """Creates an immutable public snapshot of the bracket."""
        return BracketSnapshot(
            bracket_id=self.bracket_id,
            parent_order_id=self.parent_order_id,
            symbol=self.symbol,
            entry_side=self.entry_side,
            status=self.status,
            allocated_quantity=self.allocated_quantity,
            closed_quantity=self.closed_quantity,
            residual_quantity=self.residual_quantity,
            stop_loss_price=self.stop_loss_price,
            take_profit_price=self.take_profit_price,
            active_stop_loss_order_id=self.active_stop_loss_order_id,
            active_take_profit_order_id=self.active_take_profit_order_id,
            stop_loss_order_ids=tuple(self.stop_loss_order_ids),
            take_profit_order_ids=tuple(self.take_profit_order_ids),
            has_exposure_breach=self.has_exposure_breach,
            breach_details=self.breach_details,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
