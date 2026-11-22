"""Milestone 3B.4A: Prototype Adapter Translating Execution Engine Messages to Domain Events.

NOTE: This is a translation demonstration and adapter prototype, NOT an adapter proven
against live NautilusTrader event streams. It demonstrates that external execution messages
can map cleanly into our authoritative Portfolio, CashBook, and PositionTracker lifecycle
without maintaining dual portfolios or leaking accounting state.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from hybrid_agent.core.constants import OrderSide, OrderType, ValuationModel
from hybrid_agent.core.models import InstrumentSpecification, Order
from hybrid_agent.execution.backtest.events import (
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
)
from hybrid_agent.execution.backtest.portfolio.currency import (
    TimestampedCurrencyConverter,
)
from hybrid_agent.execution.backtest.portfolio.portfolio import Portfolio
from hybrid_agent.execution.backtest.portfolio.positions import (
    AccountingClassification,
)


@dataclass(frozen=True)
class NautilusOrderFilledMessage:
    """Represents external execution telemetry emitted by a matching engine."""

    client_order_id: str
    instrument_id: str
    side: str  # "BUY" or "SELL"
    last_px: float
    last_qty: float
    leaves_qty: float
    cum_qty: float
    commission_amount: float
    commission_currency: str
    ts_event_ns: int  # UTC nanoseconds


@dataclass(frozen=True)
class NautilusOrderCanceledMessage:
    """Represents cancellation emitted by an external matching engine."""

    client_order_id: str
    instrument_id: str
    reason: str
    ts_event_ns: int


class NautilusEventAdapter:
    """Translates external execution messages into authoritative project DTOs.

    Invariants:
    1. Preserves original order quantity (Order.quantity is immutable).
    2. Validates instrument identity and order side against the registered Order.
    3. Validates that cumulative filled quantity agrees with previous fills plus the latest fill.
    4. Validates that leaves_qty equals original quantity minus cumulative filled quantity.
    5. Preserves explicit commission currency and requires it when commission_amount > 0.
    6. Ensures adapter state is not mutated before all message and event validations succeed.
    7. Converts nanosecond timestamps into microsecond-accurate timezone-aware UTC datetimes.
    8. Emits OrderPartiallyFilledEvent when leaves_qty > 0; OrderFilledEvent when leaves_qty == 0.
    """

    def __init__(self, orders: dict[str, Order]) -> None:
        self._orders = dict(orders)
        self._filled_quantities: dict[str, float] = {k: 0.0 for k in orders}
        self._remaining_quantities: dict[str, float] = {
            k: float(v.quantity) for k, v in orders.items()
        }

    @staticmethod
    def _ns_to_utc_datetime(ts_ns: int) -> datetime:
        """Converts nanosecond integer timestamp to microsecond UTC datetime without float jitter."""
        if ts_ns < 0:
            raise ValueError(f"Timestamp cannot be negative: {ts_ns}")
        seconds = ts_ns // 1_000_000_000
        micros = (ts_ns % 1_000_000_000) // 1000
        return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=micros)

    def translate_fill(
        self, message: NautilusOrderFilledMessage
    ) -> OrderFilledEvent | OrderPartiallyFilledEvent:
        """Translates an external fill message into a validated domain fill event.

        Follows transactional semantics: all validations and event instantiations
        must succeed before internal adapter state is mutated.
        """
        order = self._orders.get(message.client_order_id)
        if order is None:
            raise KeyError(f"Unknown client_order_id: {message.client_order_id}")

        # 1. Validate instrument identity (allows symbol or symbol.venue format)
        expected_sym = order.symbol
        msg_sym = message.instrument_id.split(".")[0]
        if msg_sym != expected_sym:
            raise ValueError(
                f"Instrument identity mismatch: expected {expected_sym}, got {message.instrument_id}"
            )

        # 2. Validate order side
        if message.side != order.side.value:
            raise ValueError(
                f"Order side mismatch: expected {order.side.value}, got {message.side}"
            )

        # 3. Validate execution price and quantities
        if message.last_px <= 0.0:
            raise ValueError(f"Fill price must be strictly positive: {message.last_px}")
        if message.last_qty <= 0.0:
            raise ValueError(f"Fill quantity must be strictly positive: {message.last_qty}")

        current_cum = self._filled_quantities[order.id]
        current_rem = self._remaining_quantities[order.id]

        if message.last_qty > current_rem + 1e-9:
            raise ValueError(
                f"Fill quantity {message.last_qty} exceeds remaining order quantity {current_rem} for order {order.id}"
            )

        # 4. Validate cumulative fill quantity consistency
        expected_cum = current_cum + message.last_qty
        if abs(message.cum_qty - expected_cum) > 1e-7:
            raise ValueError(
                f"Cumulative filled quantity mismatch: message claims {message.cum_qty}, "
                f"expected previous ({current_cum}) + fill ({message.last_qty}) = {expected_cum}"
            )

        if message.cum_qty > order.quantity + 1e-9:
            raise ValueError(
                f"Cumulative filled quantity {message.cum_qty} exceeds original order quantity {order.quantity}"
            )

        # 5. Validate leaves_qty consistency
        expected_leaves = order.quantity - message.cum_qty
        if abs(message.leaves_qty - expected_leaves) > 1e-7:
            raise ValueError(
                f"Leaves quantity mismatch: message claims {message.leaves_qty}, "
                f"expected original ({order.quantity}) - cumulative ({message.cum_qty}) = {expected_leaves}"
            )

        # 6. Validate commission currency
        if message.commission_amount > 0.0 and (
            not message.commission_currency or not message.commission_currency.strip()
        ):
            raise ValueError("commission_currency must be specified when commission_amount > 0")

        # 7. Convert timestamp
        fill_time = self._ns_to_utc_datetime(message.ts_event_ns)

        # 8. Construct immutable domain event (validates Pydantic constraints)
        event: OrderFilledEvent | OrderPartiallyFilledEvent
        if message.leaves_qty > 1e-7:
            event = OrderPartiallyFilledEvent(
                order=order,
                fill_price=message.last_px,
                fill_quantity=message.last_qty,
                fee=message.commission_amount,
                timestamp=fill_time,
            )
        else:
            event = OrderFilledEvent(
                order=order,
                fill_price=message.last_px,
                fill_quantity=message.last_qty,
                fee=message.commission_amount,
                timestamp=fill_time,
            )

        # 9. Commit state mutation ONLY after all validations and event construction succeed
        self._filled_quantities[order.id] = message.cum_qty
        self._remaining_quantities[order.id] = max(0.0, message.leaves_qty)

        return event

    def translate_cancel(self, message: NautilusOrderCanceledMessage) -> OrderCancelledEvent:
        """Translates an external cancellation message into an OrderCancelledEvent."""
        order = self._orders.get(message.client_order_id)
        if order is None:
            raise KeyError(f"Unknown client_order_id: {message.client_order_id}")

        msg_sym = message.instrument_id.split(".")[0]
        if msg_sym != order.symbol:
            raise ValueError(
                f"Instrument identity mismatch: expected {order.symbol}, got {message.instrument_id}"
            )

        if not message.reason or not message.reason.strip():
            raise ValueError("Cancellation reason must be provided")

        cancel_time = self._ns_to_utc_datetime(message.ts_event_ns)

        # Construct domain event
        event = OrderCancelledEvent(order=order, reason=message.reason, timestamp=cancel_time)

        # Commit state mutation
        self._remaining_quantities[order.id] = 0.0
        return event

    def get_remaining_quantity(self, order_id: str) -> float:
        return self._remaining_quantities[order_id]

    def get_filled_quantity(self, order_id: str) -> float:
        return self._filled_quantities[order_id]


def run_prototype_verification() -> None:
    """Verifies that execution messages feed cleanly into our authoritative Portfolio."""
    print("=== Running Milestone 3B.4A Integration Prototype Verification ===")

    conv = TimestampedCurrencyConverter(pivot_currency="USD")
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    portfolio = Portfolio(account_currency="USD", currency_converter=conv)
    portfolio.initialize_cash("USD", Decimal("100000.00"), t0)

    equity_spec = InstrumentSpecification(
        symbol="AAPL",
        tick_size=0.01,
        contract_size=1.0,
        min_volume=1.0,
        volume_step=1.0,
        price_precision=2,
        profit_currency="USD",
        quote_currency="USD",
        base_currency="USD",
        valuation_model=ValuationModel.LINEAR,
    )

    # 1. Create a 100-share Buy Order
    order_aapl = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        id="ord-aapl-100",
        created_at=t0,
        updated_at=t0,
    )

    adapter = NautilusEventAdapter({order_aapl.id: order_aapl})

    # 2. Simulate partial fill: 40 shares @ $150.00 with $2 commission at 10:01 UTC
    t1_ns = 1767261660_000_000_000  # 2026-01-01 10:01:00 UTC in ns
    nautilus_partial = NautilusOrderFilledMessage(
        client_order_id="ord-aapl-100",
        instrument_id="AAPL.NASDAQ",
        side="BUY",
        last_px=150.0,
        last_qty=40.0,
        leaves_qty=60.0,
        cum_qty=40.0,
        commission_amount=2.0,
        commission_currency="USD",
        ts_event_ns=t1_ns,
    )

    partial_evt = adapter.translate_fill(nautilus_partial)
    assert isinstance(partial_evt, OrderPartiallyFilledEvent)
    assert partial_evt.fill_quantity == 40.0
    assert partial_evt.order.quantity == 100.0  # Original order quantity preserved!
    assert adapter.get_filled_quantity(order_aapl.id) == 40.0
    assert adapter.get_remaining_quantity(order_aapl.id) == 60.0

    # Apply to authoritative Portfolio
    portfolio.apply_fill(
        event=partial_evt,
        spec=equity_spec,
        accounting_type=AccountingClassification.CASH_EQUITY,
        commission_currency="USD",
    )

    # Check portfolio state: Cash = 100,000 - (40 * 150) - 2 = 93,998 USD
    assert portfolio.cashbook.get_balance("USD") == Decimal("93998.00")
    pos = portfolio.positions.get_position("AAPL")
    assert pos is not None and pos.quantity == 40.0
    print("[PASS] Partial fill translated and applied: held 40 shares, remaining 60 shares.")

    # 3. Simulate final fill: remaining 60 shares @ $151.00 with $3 commission at 10:02 UTC
    t2_ns = 1767261720_000_000_000  # 2026-01-01 10:02:00 UTC in ns
    nautilus_final = NautilusOrderFilledMessage(
        client_order_id="ord-aapl-100",
        instrument_id="AAPL.NASDAQ",
        side="BUY",
        last_px=151.0,
        last_qty=60.0,
        leaves_qty=0.0,
        cum_qty=100.0,
        commission_amount=3.0,
        commission_currency="USD",
        ts_event_ns=t2_ns,
    )

    final_evt = adapter.translate_fill(nautilus_final)
    assert isinstance(final_evt, OrderFilledEvent)
    assert final_evt.fill_quantity == 60.0
    assert adapter.get_filled_quantity(order_aapl.id) == 100.0
    assert adapter.get_remaining_quantity(order_aapl.id) == 0.0

    portfolio.apply_fill(
        event=final_evt,
        spec=equity_spec,
        accounting_type=AccountingClassification.CASH_EQUITY,
        commission_currency="USD",
    )

    # Cash = 93,998 - (60 * 151) - 3 = 84,935 USD
    assert portfolio.cashbook.get_balance("USD") == Decimal("84935.00")
    pos_final = portfolio.positions.get_position("AAPL")
    assert pos_final is not None and pos_final.quantity == 100.0
    # Average price = (40*150 + 60*151)/100 = 150.60
    assert abs(pos_final.average_entry_price - 150.60) < 1e-4
    print("[PASS] Final fill translated and applied: held 100 shares @ $150.60 average entry.")

    # 4. Mark portfolio at 10:05 UTC with mark price $155.00
    t3 = datetime(2026, 1, 1, 10, 5, tzinfo=UTC)
    marked_evt = portfolio.mark_portfolio(timestamp=t3, current_prices={"AAPL": 155.0})
    # Equity = Cash ($84,935) + Market Value (100 * 155 = $15,500) = $100,435
    assert marked_evt.equity == 100435.0
    print("[PASS] Portfolio mark to market verified: Equity = $100,435.00")

    # 5. Validation Invariant Tests
    print("--- Testing Invariant Enforcement & Transactional Safety ---")

    # Invariant A: Inconsistent cumulative quantity rejected
    try:
        adapter.translate_fill(
            NautilusOrderFilledMessage(
                client_order_id="ord-aapl-100",
                instrument_id="AAPL.NASDAQ",
                side="BUY",
                last_px=150.0,
                last_qty=10.0,
                leaves_qty=0.0,
                cum_qty=999.0,  # Invalid
                commission_amount=0.0,
                commission_currency="USD",
                ts_event_ns=t2_ns,
            )
        )
        raise AssertionError("Should have rejected inconsistent cum_qty")
    except ValueError as e:
        assert "Cumulative filled quantity mismatch" in str(e) or "exceeds" in str(e)
    print("[PASS] Inconsistent cumulative quantity rejected.")

    # Invariant B: Inconsistent leaves quantity rejected
    order_test2 = Order(
        symbol="MSFT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        id="ord-msft-50",
        created_at=t0,
        updated_at=t0,
    )
    adapter2 = NautilusEventAdapter({order_test2.id: order_test2})
    try:
        adapter2.translate_fill(
            NautilusOrderFilledMessage(
                client_order_id="ord-msft-50",
                instrument_id="MSFT.NASDAQ",
                side="BUY",
                last_px=300.0,
                last_qty=20.0,
                leaves_qty=999.0,  # Invalid leaves_qty
                cum_qty=20.0,
                commission_amount=0.0,
                commission_currency="USD",
                ts_event_ns=t1_ns,
            )
        )
        raise AssertionError("Should have rejected inconsistent leaves_qty")
    except ValueError as e:
        assert "Leaves quantity mismatch" in str(e)
    # Check transactional safety: adapter state remained untouched!
    assert adapter2.get_filled_quantity("ord-msft-50") == 0.0
    assert adapter2.get_remaining_quantity("ord-msft-50") == 50.0
    print("[PASS] Inconsistent leaves quantity rejected with state rollback.")

    # Invariant C: Instrument and side mismatch rejected
    try:
        adapter2.translate_fill(
            NautilusOrderFilledMessage(
                client_order_id="ord-msft-50",
                instrument_id="GOOG.NASDAQ",  # Mismatch
                side="BUY",
                last_px=300.0,
                last_qty=20.0,
                leaves_qty=30.0,
                cum_qty=20.0,
                commission_amount=0.0,
                commission_currency="USD",
                ts_event_ns=t1_ns,
            )
        )
        raise AssertionError("Should have rejected instrument mismatch")
    except ValueError as e:
        assert "Instrument identity mismatch" in str(e)
    print("[PASS] Instrument identity mismatch rejected.")

    try:
        adapter2.translate_fill(
            NautilusOrderFilledMessage(
                client_order_id="ord-msft-50",
                instrument_id="MSFT.NASDAQ",
                side="SELL",  # Mismatch
                last_px=300.0,
                last_qty=20.0,
                leaves_qty=30.0,
                cum_qty=20.0,
                commission_amount=0.0,
                commission_currency="USD",
                ts_event_ns=t1_ns,
            )
        )
        raise AssertionError("Should have rejected side mismatch")
    except ValueError as e:
        assert "Order side mismatch" in str(e)
    print("[PASS] Order side mismatch rejected.")

    # Invariant D: Missing commission currency rejected when fee > 0
    try:
        adapter2.translate_fill(
            NautilusOrderFilledMessage(
                client_order_id="ord-msft-50",
                instrument_id="MSFT.NASDAQ",
                side="BUY",
                last_px=300.0,
                last_qty=20.0,
                leaves_qty=30.0,
                cum_qty=20.0,
                commission_amount=5.0,  # Fee > 0
                commission_currency="",  # Empty
                ts_event_ns=t1_ns,
            )
        )
        raise AssertionError("Should have rejected missing commission_currency")
    except ValueError as e:
        assert "commission_currency must be specified" in str(e)
    print("[PASS] Missing commission currency rejected when commission > 0.")

    # Invariant E: Cancellation translation with reason
    cancel_evt = adapter2.translate_cancel(
        NautilusOrderCanceledMessage(
            client_order_id="ord-msft-50",
            instrument_id="MSFT.NASDAQ",
            reason="User cancelled remaining quantity",
            ts_event_ns=t2_ns,
        )
    )
    assert isinstance(cancel_evt, OrderCancelledEvent)
    assert cancel_evt.reason == "User cancelled remaining quantity"
    assert adapter2.get_remaining_quantity("ord-msft-50") == 0.0
    print("[PASS] Cancellation translated with reason and remaining quantity zeroed.")

    print("=== All Integration Prototype Invariants Verified Successfully! ===")


if __name__ == "__main__":
    run_prototype_verification()
