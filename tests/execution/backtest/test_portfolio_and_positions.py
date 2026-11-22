"""Unit tests for PositionTracker, cost basis accounting, and Portfolio coordinator."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import (
    OrderSide,
    OrderType,
    TemporalOrderViolationError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, Order, Position
from hybrid_agent.execution.backtest.events import (
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    PortfolioMarkedEvent,
    PositionUpdatedEvent,
)
from hybrid_agent.execution.backtest.portfolio.cashbook import (
    InsufficientFundsError,
)
from hybrid_agent.execution.backtest.portfolio.currency import (
    FXRateObservation,
    MissingFXRateError,
    StaleFXRateError,
    TimestampedCurrencyConverter,
)
from hybrid_agent.execution.backtest.portfolio.ledger import (
    LedgerEntryType,
)
from hybrid_agent.execution.backtest.portfolio.portfolio import (
    Portfolio,
)
from hybrid_agent.execution.backtest.portfolio.positions import (
    AccountingClassification,
    OversellingError,
    PositionReversalError,
    PositionTracker,
)


@pytest.fixture
def equity_spec() -> InstrumentSpecification:
    return InstrumentSpecification(
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


@pytest.fixture
def cfd_spec() -> InstrumentSpecification:
    return InstrumentSpecification(
        symbol="GER40",
        tick_size=1.0,
        contract_size=1.0,
        min_volume=1.0,
        volume_step=1.0,
        price_precision=1,
        profit_currency="EUR",
        quote_currency="EUR",
        base_currency="EUR",
        valuation_model=ValuationModel.LINEAR,
    )


@pytest.fixture
def converter() -> TimestampedCurrencyConverter:
    conv = TimestampedCurrencyConverter(default_max_age=timedelta(days=1), pivot_currency="USD")
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    conv.add_rate(FXRateObservation("EUR", "USD", Decimal("1.1000"), t0))
    conv.add_rate(FXRateObservation("GBP", "USD", Decimal("1.3000"), t0))
    return conv


# ==============================================================================
# 1. PositionTracker & Cost Basis Unit Tests
# ==============================================================================


def test_weighted_average_entry_scaling(equity_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # 1. Buy 100 shares @ $150
    res1 = tracker.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        fill_price=Decimal("150.0"),
        fill_quantity=Decimal("100.0"),
        spec=equity_spec,
        accounting_model=AccountingClassification.CASH_EQUITY,
        timestamp=t0,
    )
    assert res1.new_average_entry_price == Decimal("150.0")
    assert res1.remaining_quantity == Decimal("100.0")
    assert isinstance(res1.position, Position)
    assert res1.position.quantity == 100.0
    assert res1.position.average_entry_price == 150.0

    # 2. Scale into another 100 shares @ $160
    # Average = (100 * 150 + 100 * 160) / 200 = $155
    res2 = tracker.apply_fill(
        symbol="AAPL",
        side=OrderSide.BUY,
        fill_price=Decimal("160.0"),
        fill_quantity=Decimal("100.0"),
        spec=equity_spec,
        accounting_model=AccountingClassification.CASH_EQUITY,
        timestamp=t0,
    )
    assert res2.new_average_entry_price == Decimal("155.0")
    assert res2.remaining_quantity == Decimal("200.0")

    # Authoritative core domain Position representation check
    pos = tracker.get_position("AAPL")
    assert pos is not None
    assert isinstance(pos, Position)
    assert pos.symbol == "AAPL"
    assert pos.quantity == 200.0
    assert pos.average_entry_price == 155.0
    assert pos.side == OrderSide.BUY


def test_partial_close_preserves_average_entry_price(equity_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Buy 200 shares @ $155 average
    tracker.apply_fill(
        "AAPL",
        OrderSide.BUY,
        Decimal("150.0"),
        Decimal("100.0"),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
        t0,
    )
    tracker.apply_fill(
        "AAPL",
        OrderSide.BUY,
        Decimal("160.0"),
        Decimal("100.0"),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
        t0,
    )

    # Partial sell: 50 shares @ $170
    # Realized PnL = (170 - 155) * 50 * 1.0 = $750
    res_sell = tracker.apply_fill(
        "AAPL",
        OrderSide.SELL,
        Decimal("170.0"),
        Decimal("50.0"),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
        t0,
    )
    assert res_sell.closed_quantity == Decimal("50.0")
    assert res_sell.realized_pnl == Decimal("750.0")
    assert res_sell.remaining_quantity == Decimal("150.0")
    # Average entry price remains $155
    assert res_sell.new_average_entry_price == Decimal("155.0")
    assert not res_sell.is_closed

    pos = tracker.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == 150.0
    assert pos.average_entry_price == 155.0


def test_full_close_and_cumulative_realized_pnl(equity_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    tracker.apply_fill(
        "AAPL",
        OrderSide.BUY,
        Decimal("150.0"),
        Decimal("100.0"),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
        t0,
    )
    res_close = tracker.apply_fill(
        "AAPL",
        OrderSide.SELL,
        Decimal("165.0"),
        Decimal("100.0"),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
        t0,
    )

    # Realized PnL = (165 - 150) * 100 = $1500
    assert res_close.is_closed
    assert res_close.remaining_quantity == Decimal("0.0")
    assert res_close.realized_pnl == Decimal("1500.0")

    # Cumulative realized PnL persists after position closure
    assert tracker.get_cumulative_realized_pnl("USD") == Decimal("1500.0")
    # Position tracker returns None for closed position
    pos = tracker.get_position("AAPL")
    assert pos is None

    # PositionUpdatedEvent can still be generated for closing state
    evt = tracker.get_position_event("AAPL", t0)
    assert isinstance(evt, PositionUpdatedEvent)
    assert evt.position.quantity == 0.0
    assert evt.position.realized_pnl == 1500.0


def test_cash_equity_overselling_rejected(equity_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    tracker.apply_fill(
        "AAPL",
        OrderSide.BUY,
        Decimal("150.0"),
        Decimal("100.0"),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
        t0,
    )

    with pytest.raises(OversellingError, match="exceeds open position quantity"):
        tracker.apply_fill(
            "AAPL",
            OrderSide.SELL,
            Decimal("155.0"),
            Decimal("101.0"),
            equity_spec,
            AccountingClassification.CASH_EQUITY,
            t0,
        )


def test_cash_equity_short_selling_prohibited(equity_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    with pytest.raises(ValueError, match="Short cash equities are prohibited"):
        tracker.apply_fill(
            "AAPL",
            OrderSide.SELL,
            Decimal("150.0"),
            Decimal("50.0"),
            equity_spec,
            AccountingClassification.CASH_EQUITY,
            t0,
        )


def test_cfd_long_and_short_lifecycle_and_pnl(cfd_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # 1. Open Long CFD: 10 contracts @ 18000 EUR
    tracker.apply_fill(
        "GER40",
        OrderSide.BUY,
        Decimal("18000.0"),
        Decimal("10.0"),
        cfd_spec,
        AccountingClassification.LINEAR_CFD,
        t0,
    )
    pos = tracker.get_position("GER40")
    assert pos is not None
    assert pos.quantity == 10.0
    assert pos.side == OrderSide.BUY

    # Mark update to 18100 EUR -> Unrealized PnL = (18100 - 18000) * 10 = +1000 EUR
    tracker.update_mark("GER40", Decimal("18100.0"), t0)
    pos_marked = tracker.get_position("GER40")
    assert pos_marked is not None
    assert pos_marked.unrealized_pnl == 1000.0

    # Close Long CFD @ 18150 EUR -> Realized PnL = (18150 - 18000) * 10 = +1500 EUR
    res_close = tracker.apply_fill(
        "GER40",
        OrderSide.SELL,
        Decimal("18150.0"),
        Decimal("10.0"),
        cfd_spec,
        AccountingClassification.LINEAR_CFD,
        t0,
    )
    assert res_close.realized_pnl == Decimal("1500.0")
    assert tracker.get_cumulative_realized_pnl("EUR") == Decimal("1500.0")

    # 2. Open Short CFD: 5 contracts @ 18200 EUR
    tracker.apply_fill(
        "GER40",
        OrderSide.SELL,
        Decimal("18200.0"),
        Decimal("5.0"),
        cfd_spec,
        AccountingClassification.LINEAR_CFD,
        t0,
    )
    pos_short = tracker.get_position("GER40")
    assert pos_short is not None
    assert pos_short.side == OrderSide.SELL
    assert pos_short.quantity == 5.0

    # Mark update to 18150 EUR (price drop favorable for short)
    # Unrealized PnL = (18200 - 18150) * 5 = +250 EUR
    tracker.update_mark("GER40", Decimal("18150.0"), t0)
    pos_short_marked = tracker.get_position("GER40")
    assert pos_short_marked is not None
    assert pos_short_marked.unrealized_pnl == 250.0

    # Close Short CFD @ 18100 EUR -> Realized PnL = (18200 - 18100) * 5 = +500 EUR
    res_short_close = tracker.apply_fill(
        "GER40",
        OrderSide.BUY,
        Decimal("18100.0"),
        Decimal("5.0"),
        cfd_spec,
        AccountingClassification.LINEAR_CFD,
        t0,
    )
    assert res_short_close.realized_pnl == Decimal("500.0")
    assert tracker.get_cumulative_realized_pnl("EUR") == Decimal("2000.0")


def test_cfd_opposing_fill_exceeding_position_rejected(cfd_spec: InstrumentSpecification) -> None:
    tracker = PositionTracker()
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Open Long 5 contracts
    tracker.apply_fill(
        "GER40",
        OrderSide.BUY,
        Decimal("18000.0"),
        Decimal("5.0"),
        cfd_spec,
        AccountingClassification.LINEAR_CFD,
        t0,
    )

    # Selling 6 contracts (attempted auto-reversal) must fail closed with PositionReversalError
    with pytest.raises(PositionReversalError, match="Auto-reversal is prohibited"):
        tracker.apply_fill(
            "GER40",
            OrderSide.SELL,
            Decimal("18050.0"),
            Decimal("6.0"),
            cfd_spec,
            AccountingClassification.LINEAR_CFD,
            t0,
        )


# ==============================================================================
# 2. Portfolio Coordinator Integration Tests (Reusing OrderFilledEvent)
# ==============================================================================


def test_portfolio_cash_equity_buy_and_sell_reconciliation(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # 1. Opening balance: $20,000 USD
    portfolio.initialize_cash("USD", Decimal("20000.00"), t0)
    assert portfolio.cashbook.get_balance("USD") == Decimal("20000.00")
    assert portfolio.get_total_equity(as_of=t0) == Decimal("20000.00")

    # 2. Buy 100 AAPL @ $150 with $5 commission
    # Notional = $15,000, Commission = $5
    # Cash becomes 20000 - 15000 - 5 = $4,995
    order_buy = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        id="ord-buy-001",
        created_at=t0,
        updated_at=t0,
    )
    fill_buy_event = OrderFilledEvent(
        order=order_buy,
        fill_price=150.0,
        fill_quantity=100.0,
        fee=5.0,
        timestamp=t0,
    )

    evt1 = portfolio.apply_fill(
        event=fill_buy_event,
        spec=equity_spec,
        accounting_type=AccountingClassification.CASH_EQUITY,
        commission_currency="USD",
    )
    assert isinstance(evt1, PositionUpdatedEvent)
    assert isinstance(evt1.position, Position)
    assert evt1.position.quantity == 100.0
    assert evt1.position.average_entry_price == 150.0
    assert portfolio.cashbook.get_balance("USD") == Decimal("4995.00")

    # Share Market value = 100 * $150 = $15,000
    # Equity = Cash ($4,995) + Market Value ($15,000) = $19,995 ($5 paid in commission)
    assert portfolio.get_total_equity(as_of=t0) == Decimal("19995.00")

    # 3. Update mark to $160
    # Share Market Value = 100 * $160 = $16,000
    # Equity = $4,995 + $16,000 = $20,995
    portfolio.update_market_price("AAPL", Decimal("160.0"), t0)
    assert portfolio.get_total_equity(as_of=t0) == Decimal("20995.00")

    # 4. Sell 100 AAPL @ $160 with $5 commission
    # Sale proceeds = 100 * $160 = $16,000 credited to cash
    # Cash becomes 4995 + 16000 - 5 = $20,990
    order_sell = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100.0,
        id="ord-sell-001",
        created_at=t0,
        updated_at=t0,
    )
    fill_sell_event = OrderFilledEvent(
        order=order_sell,
        fill_price=160.0,
        fill_quantity=100.0,
        fee=5.0,
        timestamp=t0,
    )
    evt2 = portfolio.apply_fill(
        event=fill_sell_event,
        spec=equity_spec,
        accounting_type=AccountingClassification.CASH_EQUITY,
        commission_currency="USD",
    )
    assert isinstance(evt2, PositionUpdatedEvent)
    assert evt2.position.quantity == 0.0
    assert portfolio.cashbook.get_balance("USD") == Decimal("20990.00")

    # Open equity positions are 0
    # Equity = Cash = $20,990 (Initial 20000 + 1000 gross PnL - 10 total commissions)
    assert portfolio.get_total_equity(as_of=t0) == Decimal("20990.00")

    # Reconcile ledger cash matches cashbook exactly
    assert portfolio.ledger.reconcile_cash("USD") == portfolio.cashbook.get_balance("USD")


def test_portfolio_unfunded_cash_equity_buy_rejected(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Account has $10,000
    portfolio.initialize_cash("USD", Decimal("10000.00"), t0)

    # Attempting to buy $15,000 of AAPL must fail closed with InsufficientFundsError
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        id="ord-unfunded",
        created_at=t0,
        updated_at=t0,
    )
    fill_event = OrderFilledEvent(
        order=order,
        fill_price=150.0,
        fill_quantity=100.0,
        fee=0.0,
        timestamp=t0,
    )
    with pytest.raises(InsufficientFundsError, match="Insufficient funds"):
        portfolio.apply_fill(
            event=fill_event,
            spec=equity_spec,
            accounting_type=AccountingClassification.CASH_EQUITY,
        )

    # Cash and ledger must remain completely unmutated
    assert portfolio.cashbook.get_balance("USD") == Decimal("10000.00")
    assert portfolio.ledger.entry_count == 1  # Only initial opening balance


def test_portfolio_cfd_margin_and_realized_pnl_reconciliation(
    cfd_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    # Account base is USD, but CFD trades in EUR
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Initialize 10,000 EUR
    portfolio.initialize_cash("EUR", Decimal("10000.00"), t0)

    # 1. Open Long 10 GER40 @ 18,000 EUR with 10 EUR commission
    # Crucial CFD Invariant: Opening notional (180,000 EUR) is NOT debited from cash!
    order_buy = Order(
        symbol="GER40",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="cfd-buy-001",
        created_at=t0,
        updated_at=t0,
    )
    fill_buy_event = OrderFilledEvent(
        order=order_buy,
        fill_price=18000.0,
        fill_quantity=10.0,
        fee=10.0,
        timestamp=t0,
    )
    evt1 = portfolio.apply_fill(
        event=fill_buy_event,
        spec=cfd_spec,
        accounting_type=AccountingClassification.LINEAR_CFD,
        commission_currency="EUR",
    )
    assert isinstance(evt1, PositionUpdatedEvent)
    assert evt1.position.quantity == 10.0

    # Cash is debited ONLY for the commission: 10000 - 10 = 9,990 EUR
    assert portfolio.cashbook.get_balance("EUR") == Decimal("9990.00")

    # Mark price at 18,100 EUR -> Unrealized PnL = (18100 - 18000) * 10 = +1,000 EUR
    portfolio.update_market_price("GER40", Decimal("18100.0"), t0)

    # Equity in EUR = Cash (9,990 EUR) + Unrealized PnL (1,000 EUR) = 10,990 EUR
    # Converted to USD @ EUR/USD = 1.1000 -> 10,990 * 1.10 = 12,089.00 USD
    assert portfolio.get_total_equity(as_of=t0) == Decimal("12089.00")

    # 2. Close CFD @ 18,100 EUR with 10 EUR commission
    # Realized PnL (+1,000 EUR) is posted to Cash
    # Cash becomes 9990 + 1000 - 10 = 10,980 EUR
    order_close = Order(
        symbol="GER40",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="cfd-close-001",
        created_at=t0,
        updated_at=t0,
    )
    fill_close_event = OrderFilledEvent(
        order=order_close,
        fill_price=18100.0,
        fill_quantity=10.0,
        fee=10.0,
        timestamp=t0,
    )
    evt2 = portfolio.apply_fill(
        event=fill_close_event,
        spec=cfd_spec,
        accounting_type=AccountingClassification.LINEAR_CFD,
        commission_currency="EUR",
    )
    assert isinstance(evt2, PositionUpdatedEvent)
    assert evt2.position.quantity == 0.0
    assert portfolio.cashbook.get_balance("EUR") == Decimal("10980.00")

    # Position is closed, unrealized PnL is 0
    # Equity = Cash = 10,980 EUR * 1.10 = 12,078.00 USD
    assert portfolio.get_total_equity(as_of=t0) == Decimal("12078.00")
    assert portfolio.ledger.reconcile_cash("EUR") == portfolio.cashbook.get_balance("EUR")


def test_portfolio_actual_fx_cash_conversion(converter: TimestampedCurrencyConverter) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    portfolio.initialize_cash("EUR", Decimal("1000.00"), t0)

    # Convert 500 EUR -> USD @ EUR/USD = 1.1000 -> 550 USD
    res = portfolio.convert_cash(
        "EUR", "USD", Decimal("500.00"), timestamp=t0, reference_id="fx-001"
    )
    assert res.converted_amount == Decimal("550.00")

    # Balances updated
    assert portfolio.cashbook.get_balance("EUR") == Decimal("500.00")
    assert portfolio.cashbook.get_balance("USD") == Decimal("550.00")

    # Paired entries in ledger
    fx_entries = portfolio.ledger.get_entries_by_reference("fx-001")
    assert len(fx_entries) == 2
    src_entry = next(e for e in fx_entries if e.entry_type == LedgerEntryType.FX_SOURCE)
    dst_entry = next(e for e in fx_entries if e.entry_type == LedgerEntryType.FX_DESTINATION)

    assert src_entry.currency == "EUR"
    assert src_entry.amount == Decimal("-500.00")
    assert dst_entry.currency == "USD"
    assert dst_entry.amount == Decimal("550.00")

    # Cash reconciled
    assert portfolio.ledger.reconcile_cash("EUR") == Decimal("500.00")
    assert portfolio.ledger.reconcile_cash("USD") == Decimal("550.00")


def test_portfolio_valuation_only_translation_creates_no_ledger_entries(
    converter: TimestampedCurrencyConverter,
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    portfolio.initialize_cash("EUR", Decimal("1000.00"), t0)
    entries_before = portfolio.ledger.entry_count

    # Valuation-only calculation
    equity = portfolio.get_total_equity(as_of=t0)
    assert equity == Decimal("1100.00")

    # Ledger entry count must NOT change
    assert portfolio.ledger.entry_count == entries_before


def test_portfolio_stale_or_missing_fx_fails_closed(
    converter: TimestampedCurrencyConverter,
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # 1. Missing FX pair (e.g. holding CHF without CHF/USD rate)
    portfolio.initialize_cash("CHF", Decimal("1000.00"), t0)
    with pytest.raises(MissingFXRateError):
        portfolio.get_total_equity(as_of=t0)

    # 2. Stale FX rate: 2 days later without updated rate
    t_stale = t0 + timedelta(days=2)
    portfolio2 = Portfolio(account_currency="USD", currency_converter=converter)
    portfolio2.initialize_cash("EUR", Decimal("1000.00"), t0)
    with pytest.raises(StaleFXRateError):
        portfolio2.get_total_equity(as_of=t_stale)


def test_portfolio_marked_event_generation(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # Deposit 10,000 USD
    portfolio.initialize_cash("USD", Decimal("10000.00"), t0)

    # Buy 20 AAPL @ 150
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=20.0,
        id="ord-01",
        created_at=t0,
        updated_at=t0,
    )
    fill = OrderPartiallyFilledEvent(
        order=order,
        fill_price=150.0,
        fill_quantity=20.0,
        fee=2.0,
        timestamp=t0,
    )
    portfolio.apply_fill(fill, equity_spec, AccountingClassification.CASH_EQUITY, "USD")

    # Mark portfolio to market at price $160 with peak equity $12,000
    marked_evt = portfolio.mark_portfolio(
        timestamp=t0,
        current_prices={"AAPL": Decimal("160.0")},
        peak_equity=Decimal("12000.0"),
    )

    assert isinstance(marked_evt, PortfolioMarkedEvent)
    # Cash = 10000 - 3000 - 2 = 6998
    assert marked_evt.cash == 6998.0
    # Market value = 20 * 160 = 3200; Equity = 6998 + 3200 = 10198
    assert marked_evt.equity == 10198.0
    # Peak = 12000; Drawdown = (12000 - 10198) / 12000 * 100 = 15.01666...%
    assert marked_evt.drawdown_pct == pytest.approx(15.01666, rel=1e-3)


# ==============================================================================
# 3. Correctness Hardening: Atomicity, Consistency, Causality & Reconciliation
# ==============================================================================


def test_portfolio_atomic_rollback_on_failed_fill(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    # 1. Establish opening balance: $10,000 USD
    portfolio.initialize_cash("USD", Decimal("10000.00"), t0)
    assert portfolio.cashbook.get_balance("USD") == Decimal("10000.00")
    assert portfolio.ledger.entry_count == 1
    assert portfolio.ledger.last_sequence_id == 1

    # Buy 50 AAPL @ $100 -> successful
    order1 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        id="ord-buy-50",
        created_at=t0,
        updated_at=t0,
    )
    fill1 = OrderFilledEvent(
        order=order1, fill_price=100.0, fill_quantity=50.0, fee=0.0, timestamp=t0
    )
    portfolio.apply_fill(fill1, equity_spec, AccountingClassification.CASH_EQUITY)

    # State after valid buy
    assert portfolio.cashbook.get_balance("USD") == Decimal("5000.00")
    assert portfolio.ledger.entry_count == 2
    assert portfolio.ledger.last_sequence_id == 2
    pos = portfolio.positions.get_position("AAPL")
    assert pos is not None and pos.quantity == 50.0

    # 2. Attempt invalid sell exceeding position (sell 60 shares when holding 50)
    order_oversell = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=60.0,
        id="ord-oversell-60",
        created_at=t0,
        updated_at=t0,
    )
    fill_oversell = OrderFilledEvent(
        order=order_oversell, fill_price=110.0, fill_quantity=60.0, fee=5.0, timestamp=t0
    )

    with pytest.raises(OversellingError, match="exceeds open position quantity"):
        portfolio.apply_fill(
            fill_oversell, equity_spec, AccountingClassification.CASH_EQUITY, "USD"
        )

    # Verify atomic rollback: CashBook, Ledger, and PositionTracker remain 100% unchanged
    assert portfolio.cashbook.get_balance("USD") == Decimal("5000.00")
    assert portfolio.ledger.entry_count == 2
    assert portfolio.ledger.last_sequence_id == 2
    pos_after = portfolio.positions.get_position("AAPL")
    assert pos_after is not None and pos_after.quantity == 50.0
    assert portfolio.positions.get_cumulative_realized_pnl("USD") == Decimal("0.0")


def test_symbol_cannot_switch_accounting_classification_while_position_open(
    equity_spec: InstrumentSpecification,
    cfd_spec: InstrumentSpecification,
    converter: TimestampedCurrencyConverter,
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("10000.00"), t0)

    # Open CASH_EQUITY on AAPL
    order1 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        id="ord-cash-buy",
        created_at=t0,
        updated_at=t0,
    )
    fill1 = OrderFilledEvent(
        order=order1, fill_price=100.0, fill_quantity=50.0, fee=0.0, timestamp=t0
    )
    portfolio.apply_fill(fill1, equity_spec, AccountingClassification.CASH_EQUITY)

    # Attempt to apply LINEAR_CFD fill to AAPL while CASH_EQUITY position is open
    order_cfd = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="ord-cfd-buy",
        created_at=t0,
        updated_at=t0,
    )
    fill_cfd = OrderFilledEvent(
        order=order_cfd, fill_price=100.0, fill_quantity=10.0, fee=0.0, timestamp=t0
    )

    with pytest.raises(ValueError, match="cannot accept fills with 'LINEAR_CFD'"):
        portfolio.apply_fill(fill_cfd, equity_spec, AccountingClassification.LINEAR_CFD)

    # State remains unmutated
    assert portfolio.cashbook.get_balance("USD") == Decimal("5000.00")
    assert portfolio.ledger.entry_count == 2
    pos = portfolio.positions.get_position("AAPL")
    assert pos is not None and pos.quantity == 50.0


def test_symbol_cannot_switch_currency_while_position_open(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("10000.00"), t0)

    # Open AAPL with USD quote currency
    order1 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        id="ord-usd",
        created_at=t0,
        updated_at=t0,
    )
    fill1 = OrderFilledEvent(
        order=order1, fill_price=100.0, fill_quantity=50.0, fee=0.0, timestamp=t0
    )
    portfolio.apply_fill(fill1, equity_spec, AccountingClassification.CASH_EQUITY)

    # Create inconsistent spec where AAPL quote currency is suddenly EUR
    inconsistent_spec = InstrumentSpecification(
        symbol="AAPL",
        tick_size=0.01,
        contract_size=1.0,
        min_volume=1.0,
        volume_step=1.0,
        price_precision=2,
        profit_currency="USD",
        quote_currency="EUR",  # Inconsistent!
        base_currency="USD",
        valuation_model=ValuationModel.LINEAR,
    )
    order2 = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="ord-eur",
        created_at=t0,
        updated_at=t0,
    )
    fill2 = OrderFilledEvent(
        order=order2, fill_price=100.0, fill_quantity=10.0, fee=0.0, timestamp=t0
    )

    with pytest.raises(ValueError, match="quote currency mismatch"):
        portfolio.apply_fill(fill2, inconsistent_spec, AccountingClassification.CASH_EQUITY)

    assert portfolio.cashbook.get_balance("USD") == Decimal("5000.00")
    assert portfolio.ledger.entry_count == 2


def test_historical_mark_causality_rejections(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("10000.00"), t0)

    # Buy 10 AAPL @ 150 at t0
    order = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="ord-mark-test",
        created_at=t0,
        updated_at=t0,
    )
    fill = OrderFilledEvent(
        order=order, fill_price=150.0, fill_quantity=10.0, fee=0.0, timestamp=t0
    )
    portfolio.apply_fill(fill, equity_spec, AccountingClassification.CASH_EQUITY)

    # Mark position at t2 (12:00)
    portfolio.update_market_price("AAPL", Decimal("160.0"), t2)

    # 1. Backward mark rejection: Attempting to update mark at t1 (11:00 < 12:00) must fail closed
    with pytest.raises(TemporalOrderViolationError, match="Backward mark timestamp"):
        portfolio.update_market_price("AAPL", Decimal("155.0"), t1)

    # 2. Future mark rejection: Requesting portfolio equity at t1 (11:00) when position mark is from t2 (12:00)
    # must fail closed to prevent consuming future prices in historical valuation
    with pytest.raises(TemporalOrderViolationError, match="future mark rejection"):
        portfolio.get_total_equity(as_of=t1)

    with pytest.raises(TemporalOrderViolationError, match="future mark rejection"):
        portfolio.mark_portfolio(timestamp=t1)

    # Valuation at or after t2 succeeds
    assert portfolio.get_total_equity(as_of=t2) == Decimal("10100.00")


def test_cfd_realized_pnl_reconciliation_and_no_double_counting(
    cfd_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    # Portfolio in USD, instrument trades in EUR @ EUR/USD = 1.1000
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    # 1. Deposit 10,000 EUR
    portfolio.initialize_cash("EUR", Decimal("10000.00"), t0)
    assert portfolio.cashbook.get_balance("EUR") == Decimal("10000.00")
    # Equity in USD = 10,000 * 1.10 = 11,000.00 USD
    assert portfolio.get_total_equity(as_of=t0) == Decimal("11000.00")

    # 2. Open Long 10 GER40 CFD @ 18,000 EUR with 10 EUR commission
    order_open = Order(
        symbol="GER40",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="cfd-open",
        created_at=t0,
        updated_at=t0,
    )
    fill_open = OrderFilledEvent(
        order=order_open, fill_price=18000.0, fill_quantity=10.0, fee=10.0, timestamp=t0
    )
    portfolio.apply_fill(fill_open, cfd_spec, AccountingClassification.LINEAR_CFD, "EUR")

    # Cash balance reflects only commission deduction: 10,000 - 10 = 9,990 EUR
    assert portfolio.cashbook.get_balance("EUR") == Decimal("9990.00")

    # 3. Mark moves to 18,100 EUR at t1
    portfolio.update_market_price("GER40", Decimal("18100.0"), t1)
    # Unrealized PnL = (18100 - 18000) * 10 = +1000 EUR
    # Equity = Cash (9990 EUR) + Unrealized (1000 EUR) = 10,990 EUR * 1.10 = 12,089.00 USD
    assert portfolio.get_total_equity(as_of=t1) == Decimal("12089.00")

    # 4. Close CFD @ 18,100 EUR at t2 with 10 EUR commission
    order_close = Order(
        symbol="GER40",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="cfd-close",
        created_at=t2,
        updated_at=t2,
    )
    fill_close = OrderFilledEvent(
        order=order_close, fill_price=18100.0, fill_quantity=10.0, fee=10.0, timestamp=t2
    )
    portfolio.apply_fill(fill_close, cfd_spec, AccountingClassification.LINEAR_CFD, "EUR")

    # Final Cash in EUR = 9990 + 1000 (realized PnL) - 10 (commission) = 10,980 EUR
    assert portfolio.cashbook.get_balance("EUR") == Decimal("10980.00")
    # PositionTracker cumulative realized trading P&L = +1,000 EUR (gross trading P&L, commissions tracked in ledger)
    assert portfolio.positions.get_cumulative_realized_pnl("EUR") == Decimal("1000.00")

    # 5. Portfolio valuation at t2:
    # Final Equity = Cash (10,980 EUR * 1.10) + Unrealized (0.00) = 12,078.00 USD.
    # CRITICAL: Realized P&L is NOT added again to Equity!
    assert portfolio.get_total_equity(as_of=t2) == Decimal("12078.00")

    # 6. PortfolioMarkedEvent reconciliation
    marked_evt = portfolio.mark_portfolio(timestamp=t2)
    assert marked_evt.cash == 12078.0
    assert marked_evt.equity == 12078.0
    assert marked_evt.unrealized_pnl == 0.0
    # Reporting field displays cumulative realized PnL converted to USD: 1000 EUR * 1.10 = 1100.0 USD
    assert marked_evt.realized_pnl == 1100.0


# ==============================================================================
# 5. Stable Instrument Accounting Identity & Atomic Marking Tests
# ==============================================================================


def test_closed_position_retains_accounting_identity_and_rejects_incompatible_reopen(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("50000.00"), t0)

    # 1. Buy 100 AAPL as CASH_EQUITY
    order_buy = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=100.0,
        id="buy-1",
        created_at=t0,
        updated_at=t0,
    )
    fill_buy = OrderFilledEvent(
        order=order_buy, fill_price=150.0, fill_quantity=100.0, fee=0.0, timestamp=t0
    )
    portfolio.apply_fill(fill_buy, equity_spec, AccountingClassification.CASH_EQUITY)
    assert portfolio.positions.get_position("AAPL") is not None

    # Check identity established
    identity = portfolio.get_instrument_identity("AAPL")
    assert identity is not None
    assert identity.accounting_classification == AccountingClassification.CASH_EQUITY

    # 2. Sell 100 AAPL (fully close position)
    order_sell = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=100.0,
        id="sell-1",
        created_at=t1,
        updated_at=t1,
    )
    fill_sell = OrderFilledEvent(
        order=order_sell, fill_price=155.0, fill_quantity=100.0, fee=0.0, timestamp=t1
    )
    portfolio.apply_fill(fill_sell, equity_spec, AccountingClassification.CASH_EQUITY)
    assert portfolio.positions.get_position("AAPL") is None  # Closed

    # 3. Attempt to reopen closed AAPL as LINEAR_CFD -> MUST be rejected
    order_cfd = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        id="cfd-reopen",
        created_at=t2,
        updated_at=t2,
    )
    fill_cfd = OrderFilledEvent(
        order=order_cfd, fill_price=155.0, fill_quantity=50.0, fee=0.0, timestamp=t2
    )
    with pytest.raises(ValueError, match="established accounting classification"):
        portfolio.apply_fill(fill_cfd, equity_spec, AccountingClassification.LINEAR_CFD)

    # 4. Reopen as CASH_EQUITY -> MUST succeed
    order_reopen = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        id="equity-reopen",
        created_at=t2,
        updated_at=t2,
    )
    fill_reopen = OrderFilledEvent(
        order=order_reopen, fill_price=155.0, fill_quantity=50.0, fee=0.0, timestamp=t2
    )
    evt = portfolio.apply_fill(fill_reopen, equity_spec, AccountingClassification.CASH_EQUITY)
    assert evt.position.quantity == 50.0
    assert portfolio.positions.get_position("AAPL") is not None


def test_closed_position_retains_currency_identity_and_rejects_currency_change(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)
    t2 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("50000.00"), t0)
    portfolio.initialize_cash("EUR", Decimal("50000.00"), t0)

    # 1. Buy and sell AAPL to close it
    order_buy = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="buy-1",
        created_at=t0,
        updated_at=t0,
    )
    portfolio.apply_fill(
        OrderFilledEvent(
            order=order_buy, fill_price=150.0, fill_quantity=10.0, fee=0.0, timestamp=t0
        ),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
    )
    order_sell = Order(
        symbol="AAPL",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="sell-1",
        created_at=t1,
        updated_at=t1,
    )
    portfolio.apply_fill(
        OrderFilledEvent(
            order=order_sell, fill_price=155.0, fill_quantity=10.0, fee=0.0, timestamp=t1
        ),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
    )
    assert portfolio.positions.get_position("AAPL") is None

    # 2. Incompatible quote currency spec
    eur_quote_spec = InstrumentSpecification(
        symbol="AAPL",
        tick_size=0.01,
        contract_size=1.0,
        min_volume=1.0,
        volume_step=1.0,
        price_precision=2,
        profit_currency="USD",
        quote_currency="EUR",
        base_currency="USD",
        valuation_model=ValuationModel.LINEAR,
    )
    order_reopen = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="reopen-eur-quote",
        created_at=t2,
        updated_at=t2,
    )
    with pytest.raises(ValueError, match="quote currency mismatch"):
        portfolio.apply_fill(
            OrderFilledEvent(
                order=order_reopen, fill_price=150.0, fill_quantity=10.0, fee=0.0, timestamp=t2
            ),
            eur_quote_spec,
            AccountingClassification.CASH_EQUITY,
        )

    # 3. Incompatible profit currency spec
    eur_profit_spec = InstrumentSpecification(
        symbol="AAPL",
        tick_size=0.01,
        contract_size=1.0,
        min_volume=1.0,
        volume_step=1.0,
        price_precision=2,
        profit_currency="EUR",
        quote_currency="USD",
        base_currency="USD",
        valuation_model=ValuationModel.LINEAR,
    )
    with pytest.raises(ValueError, match="profit currency mismatch"):
        portfolio.apply_fill(
            OrderFilledEvent(
                order=order_reopen, fill_price=150.0, fill_quantity=10.0, fee=0.0, timestamp=t2
            ),
            eur_profit_spec,
            AccountingClassification.CASH_EQUITY,
        )


def test_explicitly_configured_separate_instrument_identities(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("50000.00"), t0)

    # AAPL is CASH_EQUITY
    aapl_id = portfolio.register_instrument(
        symbol="AAPL",
        accounting_classification=AccountingClassification.CASH_EQUITY,
        quote_currency="USD",
        profit_currency="USD",
    )
    assert aapl_id.accounting_classification == AccountingClassification.CASH_EQUITY

    # AAPL_CFD is LINEAR_CFD
    aapl_cfd_id = portfolio.register_instrument(
        symbol="AAPL_CFD",
        accounting_classification=AccountingClassification.LINEAR_CFD,
        quote_currency="USD",
        profit_currency="USD",
    )
    assert aapl_cfd_id.accounting_classification == AccountingClassification.LINEAR_CFD

    # Verify both can be retrieved independently
    assert portfolio.get_instrument_identity("AAPL") == aapl_id
    assert portfolio.get_instrument_identity("AAPL_CFD") == aapl_cfd_id


def test_mark_portfolio_atomic_rollback_on_partial_failure(
    equity_spec: InstrumentSpecification, converter: TimestampedCurrencyConverter
) -> None:
    portfolio = Portfolio(account_currency="USD", currency_converter=converter)
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 11, 0, tzinfo=UTC)

    portfolio.initialize_cash("USD", Decimal("50000.00"), t0)

    # Open AAPL @ $150
    order_aapl = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="buy-aapl",
        created_at=t0,
        updated_at=t0,
    )
    portfolio.apply_fill(
        OrderFilledEvent(
            order=order_aapl, fill_price=150.0, fill_quantity=10.0, fee=0.0, timestamp=t0
        ),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
    )

    # Open MSFT @ $300
    msft_spec = InstrumentSpecification(
        symbol="MSFT",
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
    order_msft = Order(
        symbol="MSFT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="buy-msft",
        created_at=t0,
        updated_at=t0,
    )
    portfolio.apply_fill(
        OrderFilledEvent(
            order=order_msft, fill_price=300.0, fill_quantity=10.0, fee=0.0, timestamp=t0
        ),
        msft_spec,
        AccountingClassification.CASH_EQUITY,
    )

    aapl_rec_before = portfolio.positions.get_record("AAPL")
    msft_rec_before = portfolio.positions.get_record("MSFT")
    assert aapl_rec_before is not None and aapl_rec_before.current_price == Decimal("150.0")
    assert aapl_rec_before.last_updated == t0
    assert msft_rec_before is not None and msft_rec_before.current_price == Decimal("300.0")
    assert msft_rec_before.last_updated == t0

    cb_balances_before = dict(portfolio.cashbook.get_all_balances())
    ledger_count_before = len(portfolio.ledger.get_entries())

    # Attempt mark_portfolio with valid AAPL price (160.0) but INVALID MSFT price (-10.0)
    with pytest.raises(ValueError, match="Mark price must be positive"):
        portfolio.mark_portfolio(
            timestamp=t1,
            current_prices={"AAPL": 160.0, "MSFT": -10.0},
        )

    # Verify atomic rollback: AAPL mark was NOT permanently changed to 160.0 or t1
    aapl_rec_after = portfolio.positions.get_record("AAPL")
    msft_rec_after = portfolio.positions.get_record("MSFT")
    assert aapl_rec_after is not None
    assert aapl_rec_after.current_price == Decimal("150.0")
    assert aapl_rec_after.last_updated == t0
    assert msft_rec_after is not None
    assert msft_rec_after.current_price == Decimal("300.0")
    assert msft_rec_after.last_updated == t0

    # Cashbook and ledger unchanged
    assert portfolio.cashbook.get_all_balances() == cb_balances_before
    assert len(portfolio.ledger.get_entries()) == ledger_count_before


def test_mark_portfolio_atomic_rollback_on_valuation_failure(
    equity_spec: InstrumentSpecification, cfd_spec: InstrumentSpecification
) -> None:
    # Converter with short 1-hour max_age
    conv = TimestampedCurrencyConverter(default_max_age=timedelta(hours=1), pivot_currency="USD")
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
    conv.add_rate(FXRateObservation("EUR", "USD", Decimal("1.1000"), t0))

    portfolio = Portfolio(account_currency="USD", currency_converter=conv)
    portfolio.initialize_cash("USD", Decimal("50000.00"), t0)

    # Open AAPL @ $150
    order_aapl = Order(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=10.0,
        id="buy-aapl",
        created_at=t0,
        updated_at=t0,
    )
    portfolio.apply_fill(
        OrderFilledEvent(
            order=order_aapl, fill_price=150.0, fill_quantity=10.0, fee=0.0, timestamp=t0
        ),
        equity_spec,
        AccountingClassification.CASH_EQUITY,
    )

    # Open GER40 CFD @ 18000 EUR
    order_ger = Order(
        symbol="GER40",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=1.0,
        id="buy-ger",
        created_at=t0,
        updated_at=t0,
    )
    portfolio.apply_fill(
        OrderFilledEvent(
            order=order_ger, fill_price=18000.0, fill_quantity=1.0, fee=0.0, timestamp=t0
        ),
        cfd_spec,
        AccountingClassification.LINEAR_CFD,
    )

    # Mark at t1 = t0 + 2 days (EUR rate is now stale -> StaleFXRateError on valuation)
    t1 = t0 + timedelta(days=2)
    with pytest.raises(StaleFXRateError):
        portfolio.mark_portfolio(
            timestamp=t1,
            current_prices={"AAPL": 160.0, "GER40": 18200.0},
        )

    # Verify atomic rollback: AAPL and GER40 marks and timestamps restored to t0 state
    aapl_rec = portfolio.positions.get_record("AAPL")
    ger_rec = portfolio.positions.get_record("GER40")
    assert aapl_rec is not None
    assert aapl_rec.current_price == Decimal("150.0")
    assert aapl_rec.last_updated == t0
    assert ger_rec is not None
    assert ger_rec.current_price == Decimal("18000.0")
    assert ger_rec.last_updated == t0
