"""Unit tests for PositionTracker, cost basis accounting, and Portfolio coordinator."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import OrderSide, OrderType, ValuationModel
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
    res = portfolio.convert_cash("EUR", "USD", Decimal("500.00"), timestamp=t0, reference_id="fx-001")
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


def test_portfolio_stale_or_missing_fx_fails_closed(converter: TimestampedCurrencyConverter) -> None:
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
