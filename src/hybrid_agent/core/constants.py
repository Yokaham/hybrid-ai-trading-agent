"""Core domain constants and enumerations."""

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
