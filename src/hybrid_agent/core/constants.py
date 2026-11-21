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


class ValuationModel(str, Enum):
    """Valuation model used for calculating notional and risk."""

    LINEAR = "LINEAR"  # Spot, cash equity, direct forex with linear pricing
    INVERSE = "INVERSE"  # Inverse/coin-margined contracts (unsupported -> fail closed)
    DERIVATIVE = "DERIVATIVE"  # Non-linear options/exotics (unsupported -> fail closed)


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
    """Raised when an order attempts an unauthorized state transition or direct status mutation."""


class OrderValidationError(ValueError):
    """Raised when an order fails domain specification or directional validation."""


class CurrencyConversionError(ValueError):
    """Raised when a currency conversion fails due to missing, invalid, or unsupported rates."""


class UnsupportedValuationModelError(ValueError):
    """Raised when an instrument requires a non-linear or unsupported valuation model."""
