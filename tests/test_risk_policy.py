"""Unit tests for the pre-trade risk engine and monetary risk calculations."""

import unittest
from datetime import UTC, datetime

from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.constants import OrderSide, OrderType
from hybrid_agent.core.models import InstrumentSpecification, Order
from hybrid_agent.risk.policy import RiskEngine, SimpleFixedRateConverter


def create_sample_order(
    quantity: float = 1.0,
    limit_price: float = 1.0850,
    stop_loss_price: float | None = 1.0800,
) -> Order:
    now = datetime.now(UTC)
    return Order(
        id="ord-001",
        symbol="EURUSD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        limit_price=limit_price,
        stop_loss_price=stop_loss_price,
        created_at=now,
        updated_at=now,
    )


class TestRiskEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(
            account_currency="USD",
            max_daily_loss_pct=0.02,
            max_drawdown_limit_pct=0.05,
            max_risk_per_trade_pct=0.01,  # 1% equity risk limit
            max_position_size_pct=0.20,  # 20% max gross notional limit
        )
        self.converter = SimpleFixedRateConverter(rates={"USD": 1.0, "EUR": 1.08})
        self.risk = RiskEngine(settings=self.settings, converter=self.converter)
        self.spec = InstrumentSpecification(
            symbol="EURUSD",
            tick_size=0.0001,
            contract_size=100000.0,
            min_volume=0.01,
            volume_step=0.01,
            price_precision=5,
            profit_currency="USD",
        )

    def test_monetary_risk_calculation(self) -> None:
        # 1 lot EURUSD (100,000 units), entry = 1.0850, stop loss = 1.0800 (50 pips)
        order = create_sample_order(quantity=1.0, limit_price=1.0850, stop_loss_price=1.0800)
        assessment = self.risk.calculate_monetary_risk(
            order=order,
            instrument_spec=self.spec,
            portfolio_equity=100000.0,
        )
        # Gross notional: 1.0 * 1.0850 * 100,000 = $108,500
        self.assertAlmostEqual(assessment.gross_notional, 108500.0)
        # Monetary risk: (1.0850 - 1.0800) * 1.0 * 100,000 = $500
        self.assertAlmostEqual(assessment.monetary_risk, 500.0)
        # Risk % of equity ($500 / $100,000) = 0.5%
        self.assertAlmostEqual(assessment.risk_pct_of_equity, 0.005)

    def test_risk_engine_approves_safe_order(self) -> None:
        # Portfolio equity = $100,000. Max risk = $1,000 (1%). Max notional = $20,000 (20%).
        # Order: 0.1 lot (10,000 units), entry = 1.0850, SL = 1.0800
        # Notional = $10,850 (< $20,000). Risk = $50 (< $1,000).
        order = create_sample_order(quantity=0.1, limit_price=1.0850, stop_loss_price=1.0800)
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=100000.0,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.005,
            instrument_spec=self.spec,
        )
        self.assertTrue(result.approved)
        self.assertIsNotNone(result.assessment)
        self.assertAlmostEqual(result.assessment.monetary_risk, 50.0)  # type: ignore[union-attr]

    def test_risk_engine_rejects_excessive_monetary_risk(self) -> None:
        # Portfolio equity = $50,000. Max risk = $500 (1%).
        # Order: 0.2 lot (20,000 units), entry = 1.0850, SL = 1.0500 (350 pips risk = $700)
        order = create_sample_order(quantity=0.2, limit_price=1.0850, stop_loss_price=1.0500)
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=50000.0,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.005,
            instrument_spec=self.spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("exceeds max risk per trade", result.reason)

    def test_risk_engine_rejects_excessive_gross_notional_exposure(self) -> None:
        # Portfolio equity = $10,000. Max notional = $2,000 (20%).
        # Order: 0.05 lot (5,000 units), entry = 1.0850 -> Notional = $5,425 (> $2,000).
        order = create_sample_order(quantity=0.05, limit_price=1.0850, stop_loss_price=1.0840)
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=10000.0,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.005,
            instrument_spec=self.spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("exceeds position ceiling", result.reason)

    def test_risk_engine_rejects_drawdown_breach(self) -> None:
        order = create_sample_order(quantity=0.01, limit_price=1.0850, stop_loss_price=1.0840)
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=50000.0,
            current_drawdown_pct=0.06,  # Breach (limit is 0.05)
            daily_loss_pct=0.005,
            instrument_spec=self.spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("Max Drawdown limit", result.reason)

    def test_risk_engine_rejects_daily_loss_breach(self) -> None:
        order = create_sample_order(quantity=0.01, limit_price=1.0850, stop_loss_price=1.0840)
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=50000.0,
            current_drawdown_pct=0.01,
            daily_loss_pct=0.025,  # Breach (limit is 0.02)
            instrument_spec=self.spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("Daily Loss limit", result.reason)


if __name__ == "__main__":
    unittest.main()
