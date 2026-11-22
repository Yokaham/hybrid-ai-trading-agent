"""Milestone 3B.4A: Prototype Adapter Translating Execution Engine Events to Authoritative Domain Models.

Demonstrates that an external matching engine (such as NautilusTrader) can produce
execution events that map cleanly into our authoritative Portfolio, CashBook, and
PositionTracker lifecycle without maintaining dual portfolios.
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
    """Represents the execution telemetry emitted by NautilusTrader OrderFilled."""

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
    """Represents cancellation emitted by NautilusTrader OrderCanceled."""

    client_order_id: str
    instrument_id: str
    ts_event_ns: int


class NautilusEventAdapter:
    """Translates external NautilusTrader execution messages into authoritative project DTOs.

    Invariants:
    1. Preserves original order quantity (Order.quantity is immutable).
    2. Tracks remaining executable quantity via OrderExecutionState / adapter state.
    3. Emits OrderPartiallyFilledEvent when leaves_qty > 0; OrderFilledEvent when leaves_qty == 0.
    4. Translates Unix nanosecond timestamps to authoritative UTC datetime.
    5. Feeds events directly into Portfolio.apply_fill() without dual-portfolio accounting.
    """

    def __init__(self, orders: dict[str, Order]) -> None:
        self._orders = dict(orders)
        self._filled_quantities: dict[str, float] = {k: 0.0 for k in orders}
        self._remaining_quantities: dict[str, float] = {
            k: float(v.quantity) for k, v in orders.items()
        }

    def translate_fill(
        self, message: NautilusOrderFilledMessage
    ) -> OrderFilledEvent | OrderPartiallyFilledEvent:
        order = self._orders.get(message.client_order_id)
        if order is None:
            raise KeyError(f"Unknown client_order_id: {message.client_order_id}")

        rem_qty = self._remaining_quantities[order.id]
        if message.last_qty > rem_qty + 1e-9:
            raise ValueError(
                f"Fill quantity {message.last_qty} exceeds remaining order quantity {rem_qty} for order {order.id}"
            )

        new_cum_qty = self._filled_quantities[order.id] + message.last_qty
        if new_cum_qty > order.quantity + 1e-9:
            raise ValueError(
                f"Cumulative filled quantity {new_cum_qty} exceeds original order quantity {order.quantity}"
            )

        self._filled_quantities[order.id] = new_cum_qty
        self._remaining_quantities[order.id] = max(0.0, order.quantity - new_cum_qty)

        fill_time = datetime.fromtimestamp(message.ts_event_ns / 1_000_000_000, tz=UTC)

        if message.leaves_qty > 0.0:
            return OrderPartiallyFilledEvent(
                order=order,
                fill_price=message.last_px,
                fill_quantity=message.last_qty,
                fee=message.commission_amount,
                timestamp=fill_time,
            )
        else:
            return OrderFilledEvent(
                order=order,
                fill_price=message.last_px,
                fill_quantity=message.last_qty,
                fee=message.commission_amount,
                timestamp=fill_time,
            )

    def translate_cancel(self, message: NautilusOrderCanceledMessage) -> OrderCancelledEvent:
        order = self._orders.get(message.client_order_id)
        if order is None:
            raise KeyError(f"Unknown client_order_id: {message.client_order_id}")

        cancel_time = datetime.fromtimestamp(message.ts_event_ns / 1_000_000_000, tz=UTC)
        self._remaining_quantities[order.id] = 0.0
        return OrderCancelledEvent(order=order, timestamp=cancel_time)


def run_prototype_verification() -> None:
    """Verifies that external execution messages feed cleanly into our authoritative Portfolio."""
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
    t1_ns = int(datetime(2026, 1, 1, 10, 1, tzinfo=UTC).timestamp() * 1e9)
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
    t2_ns = int(datetime(2026, 1, 1, 10, 2, tzinfo=UTC).timestamp() * 1e9)
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

    print("=== All Integration Prototype Invariants Verified Successfully! ===")


if __name__ == "__main__":
    run_prototype_verification()
