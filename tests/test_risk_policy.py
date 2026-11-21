"""Unit tests for the pre-trade risk engine, monetary risk calculations, and fail-closed conversion."""

import unittest
from datetime import UTC, datetime

from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.constants import (
    CurrencyConversionError,
    OrderSide,
    OrderType,
    OrderValidationError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, Order, Tick
from hybrid_agent.risk.policy import (
    RiskEngine,
    SimpleFixedRateConverter,
    resolve_executable_price,
)


def create_sample_order(
    order_type: OrderType = OrderType.LIMIT,
    side: OrderSide = OrderSide.BUY,
    quantity: float = 1.0,
    limit_price: float | None = 1.0850,
    stop_loss_price: float | None = 1.0800,
) -> Order:
    now = datetime.now(UTC)
    return Order(
        id="ord-001",
        symbol="EURUSD",
        side=side,
        order_type=order_type,
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
            quote_currency="USD",
            base_currency="EUR",
            valuation_model=ValuationModel.LINEAR,
        )

    # 1. Market Order Valuation & Executable Reference Prices
    def test_market_order_without_reference_price_rejected(self) -> None:
        # Market order with no limit_price and no reference_price/quote supplied
        market_order = create_sample_order(
            order_type=OrderType.MARKET,
            limit_price=None,
            stop_loss_price=1.0800,
        )
        # In evaluate_order
        result = self.risk.evaluate_order(
            order=market_order,
            portfolio_equity=100000.0,
            current_drawdown_pct=0.0,
            daily_loss_pct=0.0,
            instrument_spec=self.spec,
            reference_price=None,
            current_quote=None,
        )
        self.assertFalse(result.approved)
        self.assertIn("Unable to determine valid executable reference price", result.reason)

        # In calculate_monetary_risk direct call
        with self.assertRaises(ValueError) as ctx:
            self.risk.calculate_monetary_risk(
                order=market_order,
                instrument_spec=self.spec,
                portfolio_equity=100000.0,
                reference_price=None,
            )
        self.assertIn("Market orders require an explicit reference price", str(ctx.exception))

    def test_correct_buy_and_sell_executable_reference_prices_from_quote(self) -> None:
        now = datetime.now(UTC)
        quote = Tick(
            symbol="EURUSD",
            timestamp=now,
            bid=1.0850,
            ask=1.0853,
        )

        # BUY order must resolve to Ask price (1.0853)
        buy_order = create_sample_order(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            limit_price=None,
            stop_loss_price=1.0800,
        )
        buy_exec = resolve_executable_price(order=buy_order, current_quote=quote)
        self.assertEqual(buy_exec, 1.0853)

        # SELL order must resolve to Bid price (1.0850)
        sell_order = create_sample_order(
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            limit_price=None,
            stop_loss_price=1.0900,
        )
        sell_exec = resolve_executable_price(order=sell_order, current_quote=quote)
        self.assertEqual(sell_exec, 1.0850)

    def test_quote_symbol_mismatch_rejected(self) -> None:
        now = datetime.now(UTC)
        mismatched_quote = Tick(
            symbol="GBPUSD",
            timestamp=now,
            bid=1.2500,
            ask=1.2502,
        )
        buy_order = create_sample_order(order_type=OrderType.MARKET, limit_price=None)
        with self.assertRaises(OrderValidationError):
            resolve_executable_price(order=buy_order, current_quote=mismatched_quote)

    # 2. Currency Conversion Fail-Closed Tests
    def test_converter_missing_currency_rate_raises_error(self) -> None:
        converter = SimpleFixedRateConverter(rates={"USD": 1.0, "EUR": 1.08})
        with self.assertRaises(CurrencyConversionError) as ctx:
            converter.convert(100.0, from_currency="GBP", to_currency="USD")
        self.assertIn("Missing exchange rate for source currency 'GBP'", str(ctx.exception))

        with self.assertRaises(CurrencyConversionError) as ctx:
            converter.convert(100.0, from_currency="USD", to_currency="JPY")
        self.assertIn("Missing exchange rate for target currency 'JPY'", str(ctx.exception))

    def test_converter_invalid_rate_initialization_rejected(self) -> None:
        # Zero rate
        with self.assertRaises(CurrencyConversionError):
            SimpleFixedRateConverter(rates={"USD": 1.0, "EUR": 0.0})

        # Negative rate
        with self.assertRaises(CurrencyConversionError):
            SimpleFixedRateConverter(rates={"USD": 1.0, "EUR": -1.08})

        # Non-finite rate (inf or nan)
        with self.assertRaises(CurrencyConversionError):
            SimpleFixedRateConverter(rates={"USD": 1.0, "EUR": float("inf")})

        # Empty currency key
        with self.assertRaises(CurrencyConversionError):
            SimpleFixedRateConverter(rates={"": 1.0})

    def test_converter_invalid_amount_rejected(self) -> None:
        converter = SimpleFixedRateConverter(rates={"USD": 1.0})
        with self.assertRaises(CurrencyConversionError):
            converter.convert(-50.0, from_currency="USD", to_currency="USD")
        with self.assertRaises(CurrencyConversionError):
            converter.convert(float("nan"), from_currency="USD", to_currency="USD")

    def test_risk_evaluation_fails_closed_on_unsupported_currency(self) -> None:
        # Spec with profit_currency "JPY", but converter only has USD/EUR
        jpy_spec = InstrumentSpecification(
            symbol="EURJPY",
            tick_size=0.01,
            contract_size=100000.0,
            min_volume=0.01,
            volume_step=0.01,
            price_precision=3,
            profit_currency="JPY",
            quote_currency="JPY",
            base_currency="EUR",
        )
        order = Order(
            id="ord-jpy",
            symbol="EURJPY",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.1,
            limit_price=160.00,
            stop_loss_price=159.00,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=100000.0,
            current_drawdown_pct=0.0,
            daily_loss_pct=0.0,
            instrument_spec=jpy_spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("Missing exchange rate for source currency 'JPY'", result.reason)

    # 3. Missing & Inconsistent Instrument Specifications
    def test_missing_instrument_specification_rejected(self) -> None:
        order = create_sample_order()
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=100000.0,
            current_drawdown_pct=0.0,
            daily_loss_pct=0.0,
            instrument_spec=None,  # Missing
        )
        self.assertFalse(result.approved)
        self.assertIn("Missing required instrument specification", result.reason)

    def test_instrument_specification_symbol_mismatch_rejected(self) -> None:
        order = create_sample_order()  # symbol EURUSD
        mismatched_spec = InstrumentSpecification(
            symbol="GBPUSD",
            tick_size=0.0001,
            contract_size=100000.0,
            min_volume=0.01,
            volume_step=0.01,
            price_precision=5,
            profit_currency="USD",
        )
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=100000.0,
            current_drawdown_pct=0.0,
            daily_loss_pct=0.0,
            instrument_spec=mismatched_spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("symbol 'GBPUSD' does not match order symbol 'EURUSD'", result.reason)

    # 4. Unsupported Valuation Model Fail-Closed
    def test_unsupported_valuation_model_rejected(self) -> None:
        inverse_spec = InstrumentSpecification(
            symbol="BTCUSD_INV",
            tick_size=0.5,
            contract_size=1.0,
            min_volume=1.0,
            volume_step=1.0,
            price_precision=1,
            profit_currency="USD",
            valuation_model=ValuationModel.INVERSE,
        )
        order = Order(
            id="ord-inv",
            symbol="BTCUSD_INV",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=60000.0,
            stop_loss_price=59000.0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        result = self.risk.evaluate_order(
            order=order,
            portfolio_equity=100000.0,
            current_drawdown_pct=0.0,
            daily_loss_pct=0.0,
            instrument_spec=inverse_spec,
        )
        self.assertFalse(result.approved)
        self.assertIn("Unsupported valuation model 'INVERSE'", result.reason)

    # 5. Invalid Risk Assessment Inputs
    def test_invalid_equity_rejected(self) -> None:
        order = create_sample_order()
        with self.assertRaises(ValueError):
            self.risk.calculate_monetary_risk(
                order=order,
                instrument_spec=self.spec,
                portfolio_equity=-1000.0,  # Negative
            )
        with self.assertRaises(ValueError):
            self.risk.calculate_monetary_risk(
                order=order,
                instrument_spec=self.spec,
                portfolio_equity=0.0,  # Zero
            )

    # 6. Approved and Breach Risk Calculations
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
