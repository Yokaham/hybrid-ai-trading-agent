"""Unit tests for core immutable domain models, specifications, and state transitions."""

import unittest
from datetime import UTC, datetime, timedelta

from pydantic import ValidationError

from hybrid_agent.core.constants import (
    InvalidOrderStateTransitionError,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
    ValuationModel,
)
from hybrid_agent.core.events import MarketTickEvent
from hybrid_agent.core.models import Bar, InstrumentSpecification, Order, Position, Tick


class TestCoreModels(unittest.TestCase):
    def setUp(self) -> None:
        self.now_utc = datetime.now(UTC)
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

    # 1. Timestamps: Timezone-aware UTC enforcement
    def test_naive_timestamp_rejected_in_tick(self) -> None:
        naive_dt = datetime.now()  # No tzinfo
        with self.assertRaises(ValidationError):
            Tick(symbol="EURUSD", timestamp=naive_dt, bid=1.0850, ask=1.0852)

    def test_non_utc_timezone_rejected_in_tick(self) -> None:
        from datetime import timezone

        est_tz = timezone(timedelta(hours=-5))
        est_dt = datetime.now(est_tz)
        with self.assertRaises(ValidationError):
            Tick(symbol="EURUSD", timestamp=est_dt, bid=1.0850, ask=1.0852)

    def test_naive_timestamp_rejected_in_bar(self) -> None:
        naive_start = datetime(2026, 1, 1, 10, 0, 0)
        naive_end = datetime(2026, 1, 1, 10, 1, 0)
        with self.assertRaises(ValidationError):
            Bar(
                symbol="EURUSD",
                start_time=naive_start,
                end_time=naive_end,
                open=1.085,
                high=1.086,
                low=1.084,
                close=1.0855,
                volume=100.0,
            )

    def test_naive_timestamp_rejected_in_event(self) -> None:
        naive_dt = datetime.now()
        tick = Tick(symbol="EURUSD", timestamp=self.now_utc, bid=1.0850, ask=1.0852)
        with self.assertRaises(ValidationError):
            MarketTickEvent(timestamp=naive_dt, tick=tick)

    # 2. Tick bid/ask relationship validation
    def test_tick_bid_exceeding_ask_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Tick(symbol="EURUSD", timestamp=self.now_utc, bid=1.0860, ask=1.0850)

    def test_valid_tick_creation_and_frozen(self) -> None:
        tick = Tick(symbol="EURUSD", timestamp=self.now_utc, bid=1.0850, ask=1.0852)
        self.assertEqual(tick.bid, 1.0850)
        self.assertEqual(tick.ask, 1.0852)
        with self.assertRaises(ValidationError):
            tick.bid = 1.0900  # type: ignore[misc]

    # 3. Bar OHLC and temporal relationships
    def test_bar_temporal_ordering_rejected(self) -> None:
        t1 = self.now_utc
        t0 = t1 - timedelta(minutes=1)
        # end_time <= start_time
        with self.assertRaises(ValidationError):
            Bar(
                symbol="EURUSD",
                start_time=t1,
                end_time=t0,
                open=1.085,
                high=1.086,
                low=1.084,
                close=1.0855,
                volume=10.0,
            )

    def test_bar_invalid_ohlc_high_lower_than_low(self) -> None:
        t0 = self.now_utc
        t1 = t0 + timedelta(minutes=1)
        with self.assertRaises(ValidationError):
            Bar(
                symbol="EURUSD",
                start_time=t0,
                end_time=t1,
                open=1.085,
                high=1.083,  # high < low
                low=1.084,
                close=1.0835,
                volume=10.0,
            )

    def test_bar_invalid_ohlc_high_lower_than_open_or_close(self) -> None:
        t0 = self.now_utc
        t1 = t0 + timedelta(minutes=1)
        with self.assertRaises(ValidationError):
            Bar(
                symbol="EURUSD",
                start_time=t0,
                end_time=t1,
                open=1.087,  # open > high
                high=1.086,
                low=1.084,
                close=1.085,
                volume=10.0,
            )

    def test_bar_invalid_ohlc_low_higher_than_open_or_close(self) -> None:
        t0 = self.now_utc
        t1 = t0 + timedelta(minutes=1)
        with self.assertRaises(ValidationError):
            Bar(
                symbol="EURUSD",
                start_time=t0,
                end_time=t1,
                open=1.085,
                high=1.087,
                low=1.086,  # low > close
                close=1.084,
                volume=10.0,
            )

    # 4. Instrument Specification Validation
    def test_instrument_spec_invalid_min_volume(self) -> None:
        with self.assertRaises(ValidationError):
            InstrumentSpecification(
                symbol="EURUSD",
                tick_size=0.0001,
                contract_size=100000.0,
                min_volume=0.005,  # min_volume < volume_step
                volume_step=0.01,
                price_precision=5,
            )

    def test_instrument_spec_min_volume_not_step_multiple(self) -> None:
        with self.assertRaises(ValidationError):
            InstrumentSpecification(
                symbol="EURUSD",
                tick_size=0.0001,
                contract_size=100000.0,
                min_volume=0.025,  # Not a multiple of 0.01
                volume_step=0.01,
                price_precision=5,
            )

    def test_order_validation_against_instrument(self) -> None:
        # Valid order
        order = Order(
            id="ord-1",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.05,
            limit_price=1.08500,
            stop_loss_price=1.08000,
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        order.validate_against_instrument(self.spec)

        # Quantity below min volume
        small_order = Order(
            id="ord-2",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.005,
            limit_price=1.08500,
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        with self.assertRaises(OrderValidationError):
            small_order.validate_against_instrument(self.spec)

        # Quantity not a multiple of volume step
        fractional_order = Order(
            id="ord-3",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.015,
            limit_price=1.08500,
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        with self.assertRaises(OrderValidationError):
            fractional_order.validate_against_instrument(self.spec)

        # Price violating precision
        precision_order = Order(
            id="ord-4",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.02,
            limit_price=1.0850012,  # 7 decimals, precision is 5
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        with self.assertRaises(OrderValidationError):
            precision_order.validate_against_instrument(self.spec)

    # 5. Order Directional Stop-Loss / Take-Profit Validation
    def test_directional_validation_buy_order_invalid_stop_loss(self) -> None:
        with self.assertRaises((OrderValidationError, ValidationError)) as ctx:
            Order(
                id="ord-sl-err",
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                limit_price=1.0850,
                stop_loss_price=1.0860,  # Invalid: BUY SL >= limit price
                created_at=self.now_utc,
                updated_at=self.now_utc,
            )
        self.assertIn("Directional Error", str(ctx.exception))

    def test_directional_validation_sell_order_invalid_stop_loss(self) -> None:
        with self.assertRaises((OrderValidationError, ValidationError)) as ctx:
            Order(
                id="ord-sl-err-sell",
                symbol="EURUSD",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                limit_price=1.0850,
                stop_loss_price=1.0840,  # Invalid: SELL SL <= limit price
                created_at=self.now_utc,
                updated_at=self.now_utc,
            )
        self.assertIn("Directional Error", str(ctx.exception))

    def test_directional_validation_buy_order_invalid_take_profit(self) -> None:
        with self.assertRaises((OrderValidationError, ValidationError)) as ctx:
            Order(
                id="ord-tp-err",
                symbol="EURUSD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0,
                limit_price=1.0850,
                take_profit_price=1.0840,  # Invalid: BUY TP <= limit price
                created_at=self.now_utc,
                updated_at=self.now_utc,
            )
        self.assertIn("Directional Error", str(ctx.exception))

    # 6. Order State Transitions & Direct Mutation Prevention
    def test_valid_order_state_transitions(self) -> None:
        order = Order(
            id="ord-lifecycle",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0850,
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        self.assertEqual(order.status, OrderStatus.PENDING_RISK)

        # Transition: PENDING_RISK -> APPROVED
        order.transition_to(OrderStatus.APPROVED)
        self.assertEqual(order.status, OrderStatus.APPROVED)

        # Transition: APPROVED -> SUBMITTED
        order.transition_to(OrderStatus.SUBMITTED)
        self.assertEqual(order.status, OrderStatus.SUBMITTED)

        # Transition: SUBMITTED -> FILLED
        order.transition_to(OrderStatus.FILLED)
        self.assertEqual(order.status, OrderStatus.FILLED)

    def test_invalid_order_state_transitions(self) -> None:
        order = Order(
            id="ord-invalid-trans",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0850,
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        # Cannot skip from PENDING_RISK directly to FILLED
        with self.assertRaises(InvalidOrderStateTransitionError):
            order.transition_to(OrderStatus.FILLED)

        # Approve and fill
        order.transition_to(OrderStatus.APPROVED)
        order.transition_to(OrderStatus.SUBMITTED)
        order.transition_to(OrderStatus.FILLED)

        # Cannot transition out of terminal FILLED state
        with self.assertRaises(InvalidOrderStateTransitionError):
            order.transition_to(OrderStatus.CANCELLED)

    def test_direct_order_status_mutation_is_blocked(self) -> None:
        """Proves that direct assignment to order.status raises an exception."""
        order = Order(
            id="ord-direct-mut",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0850,
            created_at=self.now_utc,
            updated_at=self.now_utc,
        )
        # Attempt direct status mutation
        with self.assertRaises(InvalidOrderStateTransitionError) as ctx:
            order.status = OrderStatus.APPROVED
        self.assertIn("Direct mutation of order 'status' is prohibited", str(ctx.exception))

        with self.assertRaises(InvalidOrderStateTransitionError) as ctx:
            order.status = OrderStatus.FILLED
        self.assertIn("Direct mutation of order 'status' is prohibited", str(ctx.exception))

        # Status must remain PENDING_RISK
        self.assertEqual(order.status, OrderStatus.PENDING_RISK)

    # 7. Position Notional vs Monetary Risk Separation
    def test_position_notional_and_monetary_risk_separation(self) -> None:
        pos = Position(
            symbol="EURUSD",
            side=OrderSide.BUY,
            quantity=1.0,
            average_entry_price=1.0850,
            current_price=1.0900,
            contract_size=100000.0,
            stop_loss_price=1.0800,  # 50 pips risk
        )
        # Gross notional = 1.0 * 1.0900 * 100000 = 109,000 USD
        self.assertAlmostEqual(pos.notional_value, 109000.0)
        # Monetary risk = |1.0850 - 1.0800| * 1.0 * 100000 = 500 USD
        self.assertAlmostEqual(pos.monetary_risk, 500.0)


if __name__ == "__main__":
    unittest.main()
