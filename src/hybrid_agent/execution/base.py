"""Execution broker interface definition."""

from abc import ABC, abstractmethod

from hybrid_agent.core.models import Order, Position


class BaseBroker(ABC):
    """Abstract interface defining required broker interaction capabilities."""

    @abstractmethod
    def get_equity(self) -> float:
        """Return total portfolio equity."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Return all active positions."""
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an approved order for simulation or execution."""
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by identifier."""
        raise NotImplementedError
