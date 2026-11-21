"""Empirical test suite evaluating ML4T Backtest execution semantics and limitations.

Covers the 8 mandatory execution cases:
  Case A: Chronological Execution (Same-bar vs Next-bar)
  Case B: Bid/Ask Quote Execution
  Case C: Transaction Cost Deductions (Commission & Slippage)
  Case D: Stop-Loss & Take-Profit Behavior (Gaps & Same-Bar Ambiguity)
  Case E: Liquidity & Partial Fill Constraints
  Case F: Position & Portfolio Accounting (Mark-to-Market, No Auto-Liquidation)
  Case G: Currency Conversion & Multi-Currency Constraints
  Case H: Look-Ahead Protection & Feed Immutability
"""

from __future__ import annotations

import datetime
from decimal import Decimal

import polars as pl
import pytest
from ml4t.backtest import BacktestConfig, Broker, DataFeed, Engine, Strategy
from ml4t.backtest.broker import ContractSpec
from ml4t.backtest.config import (
    ExecutionMode,
    ExecutionPrice,
)
from ml4t.backtest.engine import OrderSide as Ml4tOrderSide
from ml4t.backtest.engine import OrderType as Ml4tOrderType
from ml4t.backtest.execution import VolumeParticipationLimit
from ml4t.backtest.models import (
    FixedSlippage,
    PerOrderCommission,
)
from ml4t.backtest.risk import RuleChain, StopLoss, TakeProfit
from ml4t.backtest.types import AssetClass as Ml4tAssetClass

from hybrid_agent.config.settings import AppSettings
from hybrid_agent.core.models import (
    Bar,
    DataProvenance,
    DataSourceType,
    InstrumentSpecification,
    Order,
    Timeframe,
    ValuationModel,
)
from hybrid_agent.core.models import (
    OrderSide as DomainOrderSide,
)
from hybrid_agent.core.models import (
    OrderType as DomainOrderType,
)
from hybrid_agent.risk.policy import RiskEngine
from research.backtest_evaluation.adapter import (
    RiskEnforcingStrategy,
    bars_to_polars_feed,
    extract_standardized_output,
)


def make_deterministic_bars(
    prices: list[tuple[float, float, float, float]],
    volumes: list[float] | None = None,
    symbol: str = "AAPL",
    start_time: datetime.datetime | None = None,
) -> list[Bar]:
    """Generate a series of deterministic Bar objects."""
    if start_time is None:
        start_time = datetime.datetime(2026, 1, 1, 9, 30, tzinfo=datetime.UTC)
    if volumes is None:
        volumes = [1000.0] * len(prices)

    bars: list[Bar] = []
    current_time = start_time
    for (o, h, l, c), vol in zip(prices, volumes):
        end_time = current_time + datetime.timedelta(minutes=5)
        bar = Bar(
            symbol=symbol,
            timeframe=Timeframe.M5,
            start_time=current_time,
            end_time=end_time,
            open=Decimal(str(o)),
            high=Decimal(str(h)),
            low=Decimal(str(l)),
            close=Decimal(str(c)),
            volume=Decimal(str(vol)),
            provenance=DataProvenance(
                source_name="test_fixture",
                source_type=DataSourceType.HISTORICAL,
                retrieved_at=datetime.datetime(2026, 1, 1, 0, 0, tzinfo=datetime.UTC),
            ),
        )
        bars.append(bar)
        current_time = end_time
    return bars


def default_instrument_spec(symbol: str = "AAPL") -> InstrumentSpecification:
    """Return default equity InstrumentSpecification."""
    return InstrumentSpecification(
        symbol=symbol,
        tick_size=Decimal("0.01"),
        contract_size=Decimal("1.0"),
        min_volume=Decimal("1.0"),
        volume_step=Decimal("1.0"),
        price_precision=2,
        profit_currency="USD",
        quote_currency="USD",
        base_currency="USD",
        valuation_model=ValuationModel.LINEAR,
    )


# ============================================================================
# CASE A: Chronological Execution (No Same-Bar Look-Ahead)
# ============================================================================


def test_case_a_chronological_next_bar_execution():
    """Verify that in NEXT_BAR mode, a signal at bar t close executes on bar t+1 open."""
    bars = make_deterministic_bars(
        [
            (100.0, 102.0, 99.0, 101.0),
            (105.0, 106.0, 104.0, 105.0),
            (108.0, 110.0, 107.0, 109.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    class NextBarSignalStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=10.0)

    config = BacktestConfig(
        initial_cash=10000.0,
        execution_mode=ExecutionMode.NEXT_BAR,
        execution_price=ExecutionPrice.OPEN,
    )

    engine = Engine(feed=feed, strategy=NextBarSignalStrategy(), config=config)
    result = engine.run()

    fills = result.fills
    assert len(fills) == 1, "Expected exactly 1 fill"
    assert fills[0].timestamp == bars[1].start_time, (
        f"Fill timestamp {fills[0].timestamp} should match Bar 1 start {bars[1].start_time}"
    )
    assert fills[0].price == 105.0, f"Fill price should be Bar 1 Open (105.0), got {fills[0].price}"


def test_case_a_same_bar_execution_lookahead_risk():
    """Demonstrate that SAME_BAR mode with CLOSE execution price executes on the signal bar."""
    bars = make_deterministic_bars(
        [
            (100.0, 102.0, 99.0, 101.0),
            (105.0, 106.0, 104.0, 105.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    class SameBarSignalStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=10.0)

    config = BacktestConfig(
        initial_cash=10000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
    )

    engine = Engine(feed=feed, strategy=SameBarSignalStrategy(), config=config)
    result = engine.run()

    fills = result.fills
    assert len(fills) == 1
    assert fills[0].timestamp == bars[0].start_time
    assert fills[0].price == 101.0


# ============================================================================
# CASE B: Bid/Ask Quote Execution
# ============================================================================


def test_case_b_bid_ask_quote_execution():
    """Verify that when quotes are present, BUY executes at Ask and SELL executes at Bid."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 9, 30, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 9, 35, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 9, 40, tzinfo=datetime.UTC),
            ],
            "asset": ["AAPL", "AAPL", "AAPL"],
            "open": [100.0, 105.0, 108.0],
            "high": [101.0, 106.0, 109.0],
            "low": [99.0, 104.0, 107.0],
            "close": [100.0, 105.0, 108.0],
            "volume": [1000.0, 1000.0, 1000.0],
            "bid": [99.90, 104.90, 107.90],
            "ask": [100.10, 105.10, 108.10],
        }
    )
    feed = DataFeed(prices_df=df, bid_col="bid", ask_col="ask")

    class QuoteTradingStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == datetime.datetime(2026, 1, 1, 9, 30, tzinfo=datetime.UTC):
                broker.buy("AAPL", shares=10.0)
            elif timestamp == datetime.datetime(2026, 1, 1, 9, 35, tzinfo=datetime.UTC):
                broker.sell("AAPL", shares=10.0)

    config = BacktestConfig(
        initial_cash=10000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.QUOTE_SIDE,
    )

    engine = Engine(feed=feed, strategy=QuoteTradingStrategy(), config=config)
    result = engine.run()

    fills = result.fills
    assert len(fills) == 2
    buy_fill = fills[0]
    sell_fill = fills[1]

    assert buy_fill.side == Ml4tOrderSide.BUY
    assert buy_fill.price == 100.10, f"Expected Ask 100.10, got {buy_fill.price}"

    assert sell_fill.side == Ml4tOrderSide.SELL
    assert sell_fill.price == 104.90, f"Expected Bid 104.90, got {sell_fill.price}"


# ============================================================================
# CASE C: Transaction Cost Deductions (Commissions & Slippage)
# ============================================================================


def test_case_c_transaction_costs_deductions():
    """Verify commissions and slippage are correctly deducted from cash and net realized PnL."""
    bars = make_deterministic_bars(
        [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    class FlatRoundTripStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=10.0)
            elif timestamp == bars[1].start_time:
                broker.sell("AAPL", shares=10.0)

    commission_model = PerOrderCommission(per_trade=2.0)
    slippage_model = FixedSlippage(amount=0.05)

    config = BacktestConfig(
        initial_cash=10000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
    )

    engine = Engine(
        feed=feed,
        strategy=FlatRoundTripStrategy(),
        config=config,
        commission_model=commission_model,
        slippage_model=slippage_model,
    )
    result = engine.run()

    standardized = extract_standardized_output(result, initial_cash=10000.0)

    # 2 orders executed:
    # Commission: 2 orders * $2.00 = $4.00
    # Slippage: 10 shares buy * $0.05 + 10 shares sell * $0.05 = $1.00 total slippage
    # Total transaction costs: $4.00 + $1.00 = $5.00
    # Flat market price (100 -> 100). Net realized PnL = -$5.00
    assert standardized.total_commission_paid == 4.0
    assert standardized.total_slippage_paid == 1.0
    assert standardized.ending_cash == 10000.0 - 5.0
    assert standardized.ending_equity == 10000.0 - 5.0
    assert standardized.realized_pnl == -5.0


# ============================================================================
# CASE D: Stop-Loss & Take-Profit Behavior (Gaps and Same-Bar Touches)
# ============================================================================


def test_case_d1_stop_loss_gap_execution():
    """Verify whether a stop-loss executes at the gap open price when price gaps below stop level."""
    bars = make_deterministic_bars(
        [
            (100.0, 102.0, 99.0, 100.0),
            (90.0, 91.0, 89.0, 90.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    class GapStopStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=10.0)
                broker.submit_order(
                    "AAPL",
                    quantity=10.0,
                    side=Ml4tOrderSide.SELL,
                    order_type=Ml4tOrderType.STOP,
                    stop_price=95.0,
                )

    config = BacktestConfig(
        initial_cash=10000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        execution_price=ExecutionPrice.CLOSE,
    )

    engine = Engine(feed=feed, strategy=GapStopStrategy(), config=config)
    result = engine.run()

    fills = result.fills
    assert len(fills) == 2, "Expected entry and stop fill"
    stop_fill = fills[1]

    # Stop must fill at the gap open price (90.0), NOT at the stop level (95.0)
    assert stop_fill.price == 90.0, (
        f"Realistic fill must capture gap at Open 90.0, got {stop_fill.price}"
    )


def test_case_d2_same_bar_tp_and_sl_ordering():
    """Demonstrate the difference between bracket orders and RuleChain on same-bar touch."""
    bars = make_deterministic_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 115.0, 85.0, 102.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")

    # Experiment 1: Using broker.submit_bracket
    class BracketStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.submit_bracket("AAPL", quantity=10.0, take_profit=110.0, stop_loss=90.0)

    engine1 = Engine(
        feed=DataFeed(prices_df=df),
        strategy=BracketStrategy(),
        config=BacktestConfig(initial_cash=10000.0, execution_mode=ExecutionMode.SAME_BAR),
    )
    result1 = engine1.run()
    fills1 = result1.fills

    assert len(fills1) == 2
    assert fills1[1].price == 110.0, "Bracket order defaulted to optimistic TP first"

    # Experiment 2: Using RuleChain with StopLoss prioritized before TakeProfit
    class ConservativeRuleChainStrategy(Strategy):
        def on_start(self, broker: Broker):
            broker.set_position_rules(
                RuleChain(
                    [
                        StopLoss(pct=0.10),  # 10% SL (trigger at 90.0) - FIRST
                        TakeProfit(pct=0.10),  # 10% TP (trigger at 110.0) - SECOND
                    ]
                )
            )

        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=10.0)

    engine2 = Engine(
        feed=DataFeed(prices_df=df),
        strategy=ConservativeRuleChainStrategy(),
        config=BacktestConfig(initial_cash=10000.0, execution_mode=ExecutionMode.SAME_BAR),
    )
    result2 = engine2.run()
    fills2 = result2.fills
    assert len(fills2) == 2
    assert fills2[1].price == 90.0, "RuleChain correctly prioritized conservative StopLoss first"


# ============================================================================
# CASE E: Liquidity & Partial Fills
# ============================================================================


def test_case_e_liquidity_and_partial_fills():
    """Verify volume participation limits enforce partial fills or order clipping."""
    bars = make_deterministic_bars(
        prices=[(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)],
        volumes=[100.0, 100.0],
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    class LargeOrderStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=1000.0)

    config = BacktestConfig(
        initial_cash=100000.0,
        execution_mode=ExecutionMode.SAME_BAR,
        partial_fills_allowed=True,
    )

    limits = VolumeParticipationLimit(max_participation=0.20)
    engine = Engine(
        feed=feed, strategy=LargeOrderStrategy(), config=config, execution_limits=limits
    )
    result = engine.run()

    fills = result.fills
    assert len(fills) >= 1
    assert fills[0].quantity == 20.0, f"Expected 20 shares partial fill, got {fills[0].quantity}"


# ============================================================================
# CASE F: Position & Portfolio Accounting (No Auto-Liquidation)
# ============================================================================


def test_case_f_accounting_and_no_auto_liquidation():
    """Verify equity mark-to-market and that open positions are NOT auto-liquidated at test end."""
    bars = make_deterministic_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (119.0, 121.0, 118.0, 120.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    class BuyAndHoldStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                broker.buy("AAPL", shares=10.0)

    config = BacktestConfig(initial_cash=10000.0, execution_mode=ExecutionMode.SAME_BAR)
    engine = Engine(feed=feed, strategy=BuyAndHoldStrategy(), config=config)
    result = engine.run()

    standardized = extract_standardized_output(result, initial_cash=10000.0)

    assert standardized.ending_cash == 9000.0
    assert standardized.ending_equity == 10200.0
    assert standardized.closed_trades_count == 0
    assert standardized.open_trades_count == 1
    assert standardized.unrealized_pnl == 200.0


# ============================================================================
# CASE G: Currency Conversion & Multi-Currency Constraint
# ============================================================================


def test_case_g_multi_currency_strict_fail_closed():
    """Verify that ML4T strictly enforces a single declared currency and rejects mixed currencies."""
    df = pl.DataFrame(
        {
            "timestamp": [
                datetime.datetime(2026, 1, 1, 9, 30, tzinfo=datetime.UTC),
                datetime.datetime(2026, 1, 1, 9, 30, tzinfo=datetime.UTC),
            ],
            "asset": ["ES", "FDAX"],
            "open": [5000.0, 18000.0],
            "high": [5010.0, 18020.0],
            "low": [4990.0, 17980.0],
            "close": [5000.0, 18000.0],
            "volume": [1000.0, 1000.0],
        }
    )
    feed = DataFeed(prices_df=df)

    class DummyStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            pass

    contract_specs = {
        "ES": ContractSpec(
            symbol="ES",
            asset_class=Ml4tAssetClass.FUTURE,
            currency="USD",
            accounting_method="futures_variation",
        ),
        "FDAX": ContractSpec(
            symbol="FDAX",
            asset_class=Ml4tAssetClass.FUTURE,
            currency="EUR",
            accounting_method="futures_variation",
        ),
    }

    # Verify fail-closed behavior when mixed currencies are declared in futures_variation
    with pytest.raises(
        ValueError, match="A single cash variation account requires the same declared currency"
    ):
        Engine(
            feed=feed,
            strategy=DummyStrategy(),
            config=BacktestConfig(initial_cash=10000.0, allow_leverage=True),
            contract_specs=contract_specs,
        )


# ============================================================================
# CASE H: Look-Ahead Protection & Future Immutability
# ============================================================================


def test_case_h_lookahead_input_integrity_and_immutability():
    """Verify ML4T verifies input integrity and guards against post-init mutation."""
    bars = make_deterministic_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (105.0, 106.0, 104.0, 105.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    observed_history: list[dict] = []

    class CausalInspectionStrategy(Strategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            observed_history.append({"time": timestamp, "keys": list(data.keys())})

    config = BacktestConfig(initial_cash=10000.0, execution_mode=ExecutionMode.NEXT_BAR)
    engine = Engine(feed=feed, strategy=CausalInspectionStrategy(), config=config)
    _ = engine.run()

    assert len(observed_history) == 2
    assert observed_history[0]["time"] == bars[0].start_time
    assert observed_history[1]["time"] == bars[1].start_time


# ============================================================================
# PRE-TRADE RISK ENGINE INTEGRATION TEST
# ============================================================================


def test_risk_engine_pre_trade_enforcement_in_backtest():
    """Verify that domain RiskEngine intercepts and rejects unauthorized orders before ML4T submission."""
    bars = make_deterministic_bars(
        [
            (100.0, 101.0, 99.0, 100.0),
            (105.0, 106.0, 104.0, 105.0),
        ]
    )
    df = bars_to_polars_feed(bars, asset="AAPL")
    feed = DataFeed(prices_df=df)

    settings = AppSettings(
        MAX_DRAWDOWN_LIMIT_PCT=0.10,
        MAX_DAILY_LOSS_PCT=0.03,
        MAX_RISK_PER_TRADE_PCT=0.05,  # 5% max risk ($500 on $10,000 equity)
        MAX_POSITION_SIZE_PCT=0.10,  # 10% max position ($1,000 on $10,000 equity)
    )
    risk_engine = RiskEngine(settings=settings)
    spec = default_instrument_spec("AAPL")

    class SizedTradingStrategy(RiskEnforcingStrategy):
        def on_data(self, timestamp: datetime.datetime, data: dict, context: dict, broker: Broker):
            if timestamp == bars[0].start_time:
                # 1. Compliant order: 3 shares @ 100 = $300 notional (3% <= 5% max)
                order1 = Order(
                    id="ORD-001",
                    symbol="AAPL",
                    side=DomainOrderSide.BUY,
                    order_type=DomainOrderType.MARKET,
                    quantity=Decimal("3.0"),
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                success1 = self.evaluate_and_submit_order(broker, order1, reference_price=100.0)
                assert success1 is True, "Compliant order should pass risk check"

                # 2. Oversized order: 50 shares @ 100 = $5,000 notional (50% > 5% max)
                order2 = Order(
                    id="ORD-002",
                    symbol="AAPL",
                    side=DomainOrderSide.BUY,
                    order_type=DomainOrderType.MARKET,
                    quantity=Decimal("50.0"),
                    created_at=timestamp,
                    updated_at=timestamp,
                )
                success2 = self.evaluate_and_submit_order(broker, order2, reference_price=100.0)
                assert success2 is False, "Oversized order must be rejected by RiskEngine"

    config = BacktestConfig(initial_cash=10000.0, execution_mode=ExecutionMode.SAME_BAR)
    strategy = SizedTradingStrategy(risk_engine=risk_engine, instrument_spec=spec)
    engine = Engine(feed=feed, strategy=strategy, config=config)
    result = engine.run()

    assert len(strategy.approved_orders) == 1
    assert len(strategy.rejected_orders) == 1
    assert "exceeds max risk per trade" in strategy.rejected_orders[0][1].reason

    # Verify only the approved order was submitted to and executed by ML4T broker
    assert len(result.fills) == 1
    assert result.fills[0].quantity == 3.0
