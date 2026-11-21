"""Unit tests for the pre-trade risk engine."""

import unittest
from datetime import datetime, timezone
from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.constants import OrderSide, OrderType
from hybrid_agent.core.models import Order
from hybrid_agent.risk.policy import RiskEngine


def create_sample_order(quantity: float = 10.0, limit_price: float = 100.0) -> Order:
    now = datetime.now(timezone.utc)
    return Order(
        id="ord-001",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        limit_price=limit_price,
        created_at=now,
        updated_at=now,
    )


class TestRiskEngine(unittest.TestCase):
    def test_risk_engine_approves_safe_order(self) -> None:
        settings = AppSettings()
        risk = RiskEngine(settings)
        order = create_sample_order(quantity=1.0, limit_price=100.0)  # Cost = 100
        # Portfolio equity = 50,000; 1% max position = 500
        result = risk.evaluate_order(
            order=order,
            portfolio_equity=50000.0,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.005,
        )
        self.assertTrue(result.approved)

    def test_risk_engine_rejects_excessive_position_size(self) -> None:
        settings = AppSettings()
        risk = RiskEngine(settings)
        order = create_sample_order(quantity=10.0, limit_price=100.0)  # Cost = 1,000
        # Portfolio equity = 50,000; 1% max position = 500
        result = risk.evaluate_order(
            order=order,
            portfolio_equity=50000.0,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.005,
        )
        self.assertFalse(result.approved)
        self.assertIn("exceeds max single position", result.reason)

    def test_risk_engine_rejects_drawdown_breach(self) -> None:
        settings = AppSettings()
        risk = RiskEngine(settings)
        order = create_sample_order(quantity=1.0, limit_price=100.0)
        result = risk.evaluate_order(
            order=order,
            portfolio_equity=50000.0,
            current_drawdown_pct=0.06,  # Limit is 0.05
            daily_loss_pct=0.005,
        )
        self.assertFalse(result.approved)
        self.assertIn("Max Drawdown limit", result.reason)


if __name__ == "__main__":
    unittest.main()
