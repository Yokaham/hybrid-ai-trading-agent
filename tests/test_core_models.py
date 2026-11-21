"""Unit tests for core immutable domain models."""

import unittest
from datetime import datetime, timezone
from pydantic import ValidationError
from hybrid_agent.core.constants import OrderSide, OrderStatus, OrderType
from hybrid_agent.core.models import Bar, Order, Tick


class TestCoreModels(unittest.TestCase):
    def test_tick_creation_and_immutability(self) -> None:
        now = datetime.now(timezone.utc)
        tick = Tick(
            symbol="EURUSD",
            timestamp=now,
            bid=1.0850,
            ask=1.0852,
        )
        self.assertEqual(tick.symbol, "EURUSD")
        with self.assertRaises(ValidationError):
            tick.bid = 1.0900  # type: ignore[misc]

    def test_bar_timestamp_ordering(self) -> None:
        now = datetime.now(timezone.utc)
        bar = Bar(
            symbol="AAPL",
            start_time=now,
            end_time=now,
            open=150.0,
            high=155.0,
            low=149.5,
            close=154.0,
            volume=10000.0,
        )
        self.assertEqual(bar.close, 154.0)
        with self.assertRaises(ValidationError):
            bar.open = 160.0  # type: ignore[misc]

    def test_order_status_transitions(self) -> None:
        now = datetime.now(timezone.utc)
        order = Order(
            id="ord-101",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5.0,
            created_at=now,
            updated_at=now,
        )
        self.assertEqual(order.status, OrderStatus.PENDING_RISK)
        order.status = OrderStatus.APPROVED
        self.assertEqual(order.status, OrderStatus.APPROVED)


if __name__ == "__main__":
    unittest.main()
