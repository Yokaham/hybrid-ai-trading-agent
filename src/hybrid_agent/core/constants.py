"""Core domain constants, enumerations, and state transition rules."""

from enum import Enum


class OrderSide(str, Enum):
    """Direction of an order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order execution classification."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    PENDING_RISK = "PENDING_RISK"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"


# Explicit valid order lifecycle state transitions
ALLOWED_ORDER_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING_RISK: {OrderStatus.APPROVED, OrderStatus.REJECTED},
    OrderStatus.APPROVED: {OrderStatus.SUBMITTED, OrderStatus.CANCELLED},
    OrderStatus.SUBMITTED: {OrderStatus.FILLED, OrderStatus.CANCELLED},
    OrderStatus.FILLED: set(),  # Terminal state
    OrderStatus.REJECTED: set(),  # Terminal state
    OrderStatus.CANCELLED: set(),  # Terminal state
}


class InvalidOrderStateTransitionError(ValueError):
    """Raised when an order attempts an unauthorized state transition."""


class OrderValidationError(ValueError):
    """Raised when an order fails domain specification or directional validation."""
