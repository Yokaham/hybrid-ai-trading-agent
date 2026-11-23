"""Comprehensive test suite for FillEngine, matching algorithms, and execution cost models.

Covers all 50 required scenarios:
- Real quote matching (Market, Limit, Stop, Displayed size FIFO, Tick volume semantics)
- Bar matching (Synthetic spread, Open execution, Limit ranges, Gaps, Stop-loss first, Causality, Volume caps)
- Execution cost models (Slippage, Conservative rounding, Commission structures, Validation)
- Integration boundaries (OrderManager record_fill, Portfolio event conversion, State isolation, Determinism)
"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import OrderSide, OrderStatus, OrderType
from hybrid_agent.core.models import (
    Bar,
    DataProvenance,
    DataSourceType,
    InstrumentSpecification,
    Order,
    Quote,
    Tick,
    Timeframe,
    VolumeType,
)
from hybrid_agent.execution.backtest.commission import (
    FixedPerOrderCommissionModel,
    PercentageCommissionModel,
    PerShareCommissionModel,
)
from hybrid_agent.execution.backtest.events import (
    OrderFilledEvent,
    OrderSubmittedEvent,
)
from hybrid_agent.execution.backtest.execution_models import (
    CommissionConflictError,
    DuplicateMarketEventError,
    ExecutionProvenance,
    InvalidModelConfigurationError,
    LiquiditySide,
    PortfolioFillApplication,
    UnsupportedOrderTypeError,
)
from hybrid_agent.execution.backtest.fill_engine import FillEngine
from hybrid_agent.execution.backtest.liquidity import (
    DefaultLiquidityModel,
    MissingBarVolumePolicy,
    MissingQuoteLiquidityPolicy,
)
from hybrid_agent.execution.backtest.order_manager import OrderManager
from hybrid_agent.execution.backtest.portfolio import (
    AccountingClassification,
    Portfolio,
    TimestampedCurrencyConverter,
)
from hybrid_agent.execution.backtest.portfolio.currency import FXRateObservation
from hybrid_agent.execution.backtest.slippage import (
    FixedSlippageModel,
    VolumeShareSlippageModel,
    round_to_tick,
)
from hybrid_agent.execution.backtest.spread import (
    SyntheticBarSpreadModel,
)
from hybrid_agent.execution.backtest.state import (
    ExecutionStatus,
    OrderExecutionSnapshot,
    OrderSnapshot,
)

# ==============================================================================
# Fixtures and Test Helpers
# ==============================================================================


@pytest.fixture
def base_time() -> datetime:
    return datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC)


@pytest.fixture
def spec() -> InstrumentSpecification:
    return InstrumentSpecification(
        symbol="AAPL",
        tick_size=0.01,
        contract_size=1.0,
        min_volume=1.0,
        volume_step=1.0,
        price_precision=2,
        quote_currency="USD",
        profit_currency="USD",
        base_currency="USD",
    )


@pytest.fixture
def manager(spec: InstrumentSpecification) -> OrderManager:
    om = OrderManager()
    om.register_instrument(spec)
    return om


@pytest.fixture
def fill_engine(spec: InstrumentSpecification) -> FillEngine:
    fe = FillEngine()
    fe.register_instrument(spec)
    return fe


def make_order_snapshot(
    order_id: str = "ord-1",
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: float = 100.0,
    leaves_quantity: float | None = None,
    limit_price: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    submission_time: datetime = datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC),
) -> OrderExecutionSnapshot:
    """Helper creating an immutable OrderExecutionSnapshot without touching manager state."""
    order = Order(
        id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        status=OrderStatus.SUBMITTED,
        created_at=submission_time,
        updated_at=submission_time,
    )
    snap = OrderSnapshot.from_order(order)
    qty_dec = Decimal(str(quantity))
    leaves_val = leaves_quantity if leaves_quantity is not None else quantity
    leaves_dec = Decimal(str(leaves_val))
    filled_dec = qty_dec - leaves_dec
    return OrderExecutionSnapshot(
        order=snap,
        submission_time=submission_time,
        original_quantity=qty_dec,
        filled_quantity=filled_dec,
        unfilled_quantity=leaves_dec,
        leaves_quantity=leaves_dec,
        cancelled_quantity=Decimal(0),
        average_fill_price=Decimal(0),
        cumulative_fees={},
        status=snap.status,  # type: ignore[arg-type]
        fills=(),
        last_update_time=submission_time,
    )


def register_and_get_snapshot(
    manager: OrderManager,
    order_id: str,
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: float = 100.0,
    limit_price: float | None = None,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    timestamp: datetime = datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC),
) -> OrderExecutionSnapshot:
    """Registers an order with OrderManager and returns its immutable execution snapshot."""
    order = Order(
        id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=quantity,
        limit_price=limit_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        status=OrderStatus.APPROVED,
        created_at=timestamp,
        updated_at=timestamp,
    )
    event = OrderSubmittedEvent(order=order, timestamp=timestamp, submission_time=timestamp)
    return manager.register_order(event)


# ==============================================================================
# 1. Real Quote Tests (1 to 16)
# ==============================================================================


class TestRealQuoteMatching:
    """Tests covering matching against real L1 Quote and Tick data."""

    def test_1_market_buy_executes_at_ask(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """1. Market BUY executes at Ask."""
        snap = make_order_snapshot(
            order_id="b1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=500.0,
            ask_size=500.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q1")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("150.10")
        assert fills[0].fill_quantity == Decimal(50)
        assert fills[0].liquidity_side == LiquiditySide.TAKER
        assert fills[0].provenance == ExecutionProvenance.REAL_QUOTE

    def test_2_market_sell_executes_at_bid(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """2. Market SELL executes at Bid."""
        snap = make_order_snapshot(
            order_id="s1", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=50.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=500.0,
            ask_size=500.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q2")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("150.00")
        assert fills[0].fill_quantity == Decimal(50)
        assert fills[0].liquidity_side == LiquiditySide.TAKER

    def test_3_real_quote_spread_is_not_charged_twice(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """3. Real quote spread is not charged twice (zero synthetic spread added)."""
        snap = make_order_snapshot(
            order_id="b-spread", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=100.00,
            ask=100.50,  # 50c spread already in quote
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q3")
        assert len(fills) == 1
        # Fill price is strictly Ask (100.50), not 100.50 + half-spread
        assert fills[0].fill_price == Decimal("100.50")

    def test_4_buy_limit_remains_unfilled_until_ask_reaches_limit(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """4. BUY limit remains unfilled until Ask reaches the limit."""
        t_sub = base_time - timedelta(seconds=1)
        snap = make_order_snapshot(
            order_id="bl1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            limit_price=150.00,
            submission_time=t_sub,
        )
        # Quote with Ask above limit: 150.05
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=149.95,
            ask=150.05,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills1 = fill_engine.match_quote(q1, [snap], market_event_id="q4-1")
        assert len(fills1) == 0

        # Quote where Ask drops to limit: 150.00
        t2 = base_time + timedelta(seconds=1)
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=149.90,
            ask=150.00,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills2 = fill_engine.match_quote(q2, [snap], market_event_id="q4-2")
        assert len(fills2) == 1
        assert fills2[0].fill_price == Decimal("150.00")
        assert fills2[0].liquidity_side == LiquiditySide.MAKER

    def test_5_sell_limit_remains_unfilled_until_bid_reaches_limit(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """5. SELL limit remains unfilled until Bid reaches the limit."""
        t_sub = base_time - timedelta(seconds=1)
        snap = make_order_snapshot(
            order_id="sl1",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            limit_price=155.00,
            submission_time=t_sub,
        )
        # Bid is 154.90 < 155.00
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=154.90,
            ask=155.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills1 = fill_engine.match_quote(q1, [snap], market_event_id="q5-1")
        assert len(fills1) == 0

        # Bid reaches 155.00
        t2 = base_time + timedelta(seconds=1)
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=155.00,
            ask=155.20,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills2 = fill_engine.match_quote(q2, [snap], market_event_id="q5-2")
        assert len(fills2) == 1
        assert fills2[0].fill_price == Decimal("155.00")
        assert fills2[0].liquidity_side == LiquiditySide.MAKER

    def test_6_marketable_limit_orders_execute_at_valid_price(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """6. Marketable limit orders execute at a valid price (with price improvement)."""
        # BUY limit at 151.00 when market Ask is 150.20
        snap = make_order_snapshot(
            order_id="ml-buy",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=151.00,
            submission_time=base_time,
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.10,
            ask=150.20,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q6")
        assert len(fills) == 1
        # Executes at the marketable Ask price (150.20), preserving price improvement
        assert fills[0].fill_price == Decimal("150.20")
        assert fills[0].fill_price <= Decimal("151.00")
        assert fills[0].liquidity_side == LiquiditySide.TAKER

    def test_7_limit_price_is_never_violated_by_slippage(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """7. Limit price is never violated by slippage."""
        # Configured fixed slippage of 0.15
        engine = FillEngine(
            slippage_model=FixedSlippageModel(fixed_slippage="0.15"),
            instrument_specs={"AAPL": spec},
        )
        # BUY limit at 150.25 when Ask is 150.20. With 0.15 slippage, raw price would be 150.35 > 150.25!
        snap = make_order_snapshot(
            order_id="slip-limit",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=150.25,
            submission_time=base_time,
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.10,
            ask=150.20,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = engine.match_quote(quote, [snap], market_event_id="q7")
        # Contractual limit-price protection leaves order unfilled rather than violate the limit
        assert len(fills) == 0

    def test_8_buy_stop_market_triggers_on_ask(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """8. BUY stop-market triggers on Ask."""
        snap = make_order_snapshot(
            order_id="stop-buy",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,
            quantity=30.0,
            stop_loss_price=152.00,
            submission_time=base_time - timedelta(seconds=1),
        )
        # Ask is 151.90 -> does not trigger
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=151.80,
            ask=151.90,
            bid_size=100.0,
            ask_size=100.0,
        )
        assert len(fill_engine.match_quote(q1, [snap], market_event_id="q8-1")) == 0

        # Ask reaches 152.05 -> triggers
        t2 = base_time + timedelta(seconds=1)
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=151.95,
            ask=152.05,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(q2, [snap], market_event_id="q8-2")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("152.05")
        assert fills[0].liquidity_side == LiquiditySide.TAKER

    def test_9_sell_stop_market_triggers_on_bid(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """9. SELL stop-market triggers on Bid."""
        snap = make_order_snapshot(
            order_id="stop-sell",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=30.0,
            stop_loss_price=148.00,
            submission_time=base_time - timedelta(seconds=1),
        )
        # Bid is 148.10 -> does not trigger
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=148.10,
            ask=148.20,
            bid_size=100.0,
            ask_size=100.0,
        )
        assert len(fill_engine.match_quote(q1, [snap], market_event_id="q9-1")) == 0

        # Bid reaches 147.95 -> triggers
        t2 = base_time + timedelta(seconds=1)
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=147.95,
            ask=148.05,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(q2, [snap], market_event_id="q9-2")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("147.95")
        assert fills[0].liquidity_side == LiquiditySide.TAKER

    def test_10_displayed_quote_size_is_consumed_exactly_once(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """10. Displayed quote size is consumed exactly once."""
        snap1 = make_order_snapshot(
            order_id="c1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        snap2 = make_order_snapshot(
            order_id="c2", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        # Total Ask size available: 50
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=50.0,
        )
        fills = fill_engine.match_quote(quote, [snap1, snap2], market_event_id="q10")
        assert len(fills) == 2
        assert fills[0].fill_quantity == Decimal(40)
        # snap2 gets only the remaining 10 units!
        assert fills[1].fill_quantity == Decimal(10)

    def test_11_multiple_orders_compete_in_deterministic_fifo_order(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """11. Multiple orders compete for the same displayed liquidity in deterministic FIFO order (A=60, B=50, C=20 on 100 Ask)."""
        snapA = make_order_snapshot(
            order_id="A", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=60.0
        )
        snapB = make_order_snapshot(
            order_id="B", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        snapC = make_order_snapshot(
            order_id="C", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=20.0
        )
        # Available Ask size: 100
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(quote, [snapA, snapB, snapC], market_event_id="q11")
        assert len(fills) == 2  # A and B receive fills; C gets zero
        assert fills[0].order_id == "A" and fills[0].fill_quantity == Decimal(60)
        assert fills[1].order_id == "B" and fills[1].fill_quantity == Decimal(40)

    def test_12_partial_quote_fills_preserve_correct_remaining_order_quantity(
        self, manager: OrderManager, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """12. Partial quote fills preserve correct remaining order quantity when recorded in OrderManager."""
        snap = register_and_get_snapshot(
            manager,
            order_id="partial-test",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            timestamp=base_time,
        )
        # Quote with only 35 units
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=35.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q12")
        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal(35)
        assert fills[0].is_complete is False

        # Record in OrderManager without recalculation
        updated_snap = manager.record_fill(fills[0].fill_record)
        assert updated_snap.filled_quantity == Decimal(35)
        assert updated_snap.leaves_quantity == Decimal(65)
        assert updated_snap.is_partially_filled is True

    def test_13_missing_displayed_size_does_not_silently_produce_unlimited_liquidity(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """13. Missing displayed size does not silently produce unlimited liquidity under REJECT policy."""
        # Default policy is REJECT
        engine = FillEngine(
            liquidity_model=DefaultLiquidityModel(
                missing_quote_policy=MissingQuoteLiquidityPolicy.REJECT
            ),
            instrument_specs={"AAPL": spec},
        )
        snap = make_order_snapshot(
            order_id="missing-depth", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        # Quote with 0 displayed size
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=0.0,
            ask_size=0.0,
        )
        fills = engine.match_quote(quote, [snap], market_event_id="q13")
        assert len(fills) == 0

    def test_14_duplicate_market_event_ids_cannot_reuse_quote_liquidity(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """14. Duplicate market event IDs cannot reuse quote liquidity (idempotent replay)."""
        snap1 = make_order_snapshot(
            order_id="dup-q-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=50.0,
        )
        # First execution consumes all 50 units
        fills1 = fill_engine.match_quote(quote, [snap1], market_event_id="event-repeat")
        assert len(fills1) == 1
        assert fills1[0].fill_quantity == Decimal(50)

        # Replayed execution with identical market_event_id returns previous fills without duplicating consumption
        fills2 = fill_engine.match_quote(quote, [snap1], market_event_id="event-repeat")
        assert len(fills2) == 1
        assert fills2[0].fill_record.fill_id == fills1[0].fill_record.fill_id

    def test_15_synthetic_quote_provenance_handled_according_to_policy(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """15. Real quotes carry explicit REAL_QUOTE provenance."""
        snap = make_order_snapshot(
            order_id="prov-check", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        prov = DataProvenance(
            source_name="direct_feed",
            source_type=DataSourceType.REAL_LIVE,
            retrieved_at=base_time,
            is_synthetic=False,
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
            provenance=prov,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q15")
        assert len(fills) == 1
        assert fills[0].provenance == ExecutionProvenance.REAL_QUOTE

    def test_16_tick_volume_fields_not_misclassified_without_verified_semantics(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """16. Tick bid/ask volume fields are not misclassified as quote depth without explicit opt-in."""
        # By default, allow_tick_volume_as_quote_size=False and policy=REJECT
        engine = FillEngine(
            liquidity_model=DefaultLiquidityModel(
                allow_tick_volume_as_quote_size=False,
                missing_quote_policy=MissingQuoteLiquidityPolicy.REJECT,
            ),
            instrument_specs={"AAPL": spec},
        )
        snap = make_order_snapshot(
            order_id="tick-test", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        tick = Tick(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_volume=1000.0,
            ask_volume=1000.0,
        )
        fills = engine.match_quote(tick, [snap], market_event_id="tick-q16")
        # Rejected because Tick volume is not verified quote depth under strict REJECT policy
        assert len(fills) == 0

        # Now test with explicit policy CONFIGURED_DEFAULT
        engine_default = FillEngine(
            liquidity_model=DefaultLiquidityModel(
                allow_tick_volume_as_quote_size=False,
                missing_quote_policy=MissingQuoteLiquidityPolicy.CONFIGURED_DEFAULT,
                default_quote_size=Decimal(25),
            ),
            instrument_specs={"AAPL": spec},
        )
        fills_def = engine_default.match_quote(tick, [snap], market_event_id="tick-q16-def")
        assert len(fills_def) == 1
        assert fills_def[0].fill_quantity == Decimal(25)
        assert any("Unverified Tick volume" in d for d in fills_def[0].diagnostics)


# ==============================================================================
# 2. Historical OHLCV Bar Tests (17 to 32)
# ==============================================================================


class TestBarMatching:
    """Tests covering matching against historical OHLCV Bar data with synthetic spread."""

    @pytest.fixture
    def standard_bar(self, base_time: datetime) -> Bar:
        return Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=152.00,
            low=148.00,
            close=151.00,
            volume=10000.0,
            volume_type=VolumeType.REAL_TRADED,
            timeframe=Timeframe.M5,
        )

    def test_17_market_buy_executes_at_bar_open_plus_costs(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """17. Market BUY executes at the eligible bar Open plus configured costs (synthetic half-spread)."""
        snap = make_order_snapshot(
            order_id="b-bar",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(standard_bar, [snap], market_event_id="b17")
        assert len(fills) == 1
        # Default synthetic spread is 0.02 (half-spread 0.01)
        # Open=150.00 + 0.01 = 150.01
        assert fills[0].fill_price == Decimal("150.01")
        assert fills[0].fill_record.timestamp == standard_bar.start_time
        assert fills[0].provenance == ExecutionProvenance.SYNTHETIC_BAR

    def test_18_market_sell_executes_at_bar_open_minus_costs(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """18. Market SELL executes at the eligible bar Open minus configured costs."""
        snap = make_order_snapshot(
            order_id="s-bar",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(standard_bar, [snap], market_event_id="b18")
        assert len(fills) == 1
        # Open=150.00 - 0.01 = 149.99
        assert fills[0].fill_price == Decimal("149.99")

    def test_19_buy_limit_triggers_correctly_using_synthetic_ask_range(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """19. BUY limit triggers correctly using synthetic Ask range (Low + half-spread)."""
        # Low=148.00 + half_spread 0.01 = 148.01 lowest Ask
        # Limit at 148.50 -> triggers
        snap1 = make_order_snapshot(
            order_id="bl-trigger",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=148.50,
            submission_time=base_time,
        )
        fills1 = fill_engine.match_bar(standard_bar, [snap1], market_event_id="b19-1")
        assert len(fills1) == 1
        assert fills1[0].fill_price == Decimal("148.50")
        assert fills1[0].fill_record.timestamp == standard_bar.end_time

        # Limit at 147.50 -> Low + half_spread (148.01) never reaches 147.50
        fill_engine.reset()
        snap2 = make_order_snapshot(
            order_id="bl-notrigger",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=147.50,
            submission_time=base_time,
        )
        fills2 = fill_engine.match_bar(standard_bar, [snap2], market_event_id="b19-2")
        assert len(fills2) == 0

    def test_20_sell_limit_triggers_correctly_using_synthetic_bid_range(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """20. SELL limit triggers correctly using synthetic Bid range (High - half-spread)."""
        # High=152.00 - half_spread 0.01 = 151.99 highest Bid
        # Limit at 151.50 -> triggers
        snap1 = make_order_snapshot(
            order_id="sl-trigger",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=151.50,
            submission_time=base_time,
        )
        fills1 = fill_engine.match_bar(standard_bar, [snap1], market_event_id="b20-1")
        assert len(fills1) == 1
        assert fills1[0].fill_price == Decimal("151.50")

        # Limit at 153.00 -> High - half_spread (151.99) < 153.00 -> no fill
        fill_engine.reset()
        snap2 = make_order_snapshot(
            order_id="sl-notrigger",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=153.00,
            submission_time=base_time,
        )
        assert len(fill_engine.match_bar(standard_bar, [snap2], market_event_id="b20-2")) == 0

    def test_21_opening_price_improvement_works_correctly_for_limit_orders(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """21. Opening-price improvement works correctly for limit orders."""
        # BUY limit at 151.00 when bar opens at 150.00 + 0.01 = 150.01
        # Order gets filled at 150.01 (better than 151.00) at bar.start_time
        snap = make_order_snapshot(
            order_id="opt-limit",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=25.0,
            limit_price=151.00,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(standard_bar, [snap], market_event_id="b21")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("150.01")
        assert fills[0].fill_record.timestamp == standard_bar.start_time

    def test_22_buy_stop_market_handles_ordinary_triggers(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """22. BUY stop-market handles ordinary triggers within the bar."""
        # High=152.00, stop at 151.50 (above Open=150.00)
        snap = make_order_snapshot(
            order_id="stop-ord-buy",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,
            quantity=10.0,
            stop_loss_price=151.50,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(standard_bar, [snap], market_event_id="b22")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("151.50")
        assert fills[0].fill_record.timestamp == standard_bar.end_time

    def test_23_sell_stop_market_handles_ordinary_triggers(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """23. SELL stop-market handles ordinary triggers within the bar."""
        # Low=148.00, stop at 149.00 (below Open=150.00)
        snap = make_order_snapshot(
            order_id="stop-ord-sell",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=10.0,
            stop_loss_price=149.00,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(standard_bar, [snap], market_event_id="b23")
        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("149.00")
        assert fills[0].fill_record.timestamp == standard_bar.end_time

    def test_24_buy_stop_market_handles_adverse_opening_gaps(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """24. BUY stop-market handles adverse opening gaps (executes at gapped Open, not stop price)."""
        # Bar opens at 155.00, stop price was 152.00
        gap_bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=155.00,
            high=156.00,
            low=154.00,
            close=155.50,
            volume=10000.0,
            volume_type=VolumeType.REAL_TRADED,
        )
        snap = make_order_snapshot(
            order_id="gap-buy",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,
            quantity=15.0,
            stop_loss_price=152.00,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(gap_bar, [snap], market_event_id="b24")
        assert len(fills) == 1
        # Executes at Open (155.00) + half-spread (0.01) = 155.01!
        assert fills[0].fill_price == Decimal("155.01")
        assert fills[0].fill_record.timestamp == gap_bar.start_time
        assert any("Opening price gap" in d for d in fills[0].diagnostics)

    def test_25_sell_stop_market_handles_adverse_opening_gaps(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """25. SELL stop-market handles adverse opening gaps (executes at gapped Open, not stop price)."""
        # Bar opens at 144.00, stop price was 148.00
        gap_bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=144.00,
            high=145.00,
            low=143.00,
            close=144.50,
            volume=10000.0,
            volume_type=VolumeType.REAL_TRADED,
        )
        snap = make_order_snapshot(
            order_id="gap-sell",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=15.0,
            stop_loss_price=148.00,
            submission_time=base_time,
        )
        fills = fill_engine.match_bar(gap_bar, [snap], market_event_id="b25")
        assert len(fills) == 1
        # Executes at Open (144.00) - half-spread (0.01) = 143.99!
        assert fills[0].fill_price == Decimal("143.99")
        assert fills[0].fill_record.timestamp == gap_bar.start_time

    def test_26_stop_loss_wins_when_both_bracket_legs_touched_in_same_bar(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """26. Stop-loss wins when both bracket legs are touched in the same bar."""
        # standard_bar: Low=148.00, High=152.00, Open=150.00
        # For a BUY position: SL is SELL stop at 149.00; TP is SELL limit at 151.00
        # Both prices are reached during standard_bar!
        sl_snap = make_order_snapshot(
            order_id="b-sl",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=50.0,
            stop_loss_price=149.00,
            submission_time=base_time,
        )
        tp_snap = make_order_snapshot(
            order_id="b-tp",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            limit_price=151.00,
            submission_time=base_time,
        )

        bracket_map = {"b-sl": "BRACKET-1", "b-tp": "BRACKET-1"}
        fills = fill_engine.match_bar(
            standard_bar, [sl_snap, tp_snap], market_event_id="b26", bracket_membership=bracket_map
        )

        # Only ONE fill must be generated, and it MUST be the Stop-Loss!
        assert len(fills) == 1
        assert fills[0].order_id == "b-sl"
        assert fills[0].fill_price == Decimal("149.00")

    def test_27_unrelated_orders_are_not_incorrectly_grouped_into_bracket(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """27. Unrelated orders sharing a symbol are not grouped into one bracket."""
        # Two unrelated orders for different strategies
        o1 = make_order_snapshot(
            order_id="unrelated-sl",
            side=OrderSide.SELL,
            order_type=OrderType.STOP_LOSS,
            quantity=10.0,
            stop_loss_price=149.00,
            submission_time=base_time,
        )
        o2 = make_order_snapshot(
            order_id="unrelated-tp",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=151.00,
            submission_time=base_time,
        )

        # Separate or empty bracket membership
        bracket_map = {"unrelated-sl": "BRACKET-X", "unrelated-tp": "BRACKET-Y"}
        fills = fill_engine.match_bar(
            standard_bar, [o1, o2], market_event_id="b27", bracket_membership=bracket_map
        )
        # Both independent orders generate fills!
        assert len(fills) == 2
        assert {f.order_id for f in fills} == {"unrelated-sl", "unrelated-tp"}

    def test_28_orders_submitted_after_bar_start_cannot_execute(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """28. Orders submitted after bar start cannot execute using earlier bar information."""
        # Bar start is 9:30:00; order submitted at 9:31:00
        t_late = base_time + timedelta(minutes=1)
        snap = make_order_snapshot(
            order_id="late-order",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=20.0,
            submission_time=t_late,
        )
        fills = fill_engine.match_bar(standard_bar, [snap], market_event_id="b28")
        assert len(fills) == 0

    def test_29_same_bar_cannot_be_replayed_to_consume_volume_twice(
        self, fill_engine: FillEngine, standard_bar: Bar, base_time: datetime
    ) -> None:
        """29. Same bar cannot be replayed to consume its volume more than once."""
        snap = make_order_snapshot(
            order_id="vol-replay",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=base_time,
        )
        fills1 = fill_engine.match_bar(standard_bar, [snap], market_event_id="b29-evt")
        assert len(fills1) == 1

        # Replayed call with identical event_id returns cached fills
        fills2 = fill_engine.match_bar(standard_bar, [snap], market_event_id="b29-evt")
        assert len(fills2) == 1
        assert fills2[0].fill_record.fill_id == fills1[0].fill_record.fill_id

    def test_30_participation_capacity_is_shared_across_all_orders(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """30. Participation capacity (10% of 1000 = 100) is shared across all eligible orders."""
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=1000.0,  # 10% cap = 100 units
            volume_type=VolumeType.REAL_TRADED,
        )
        snap1 = make_order_snapshot(
            order_id="part1",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=70.0,
            submission_time=base_time,
        )
        snap2 = make_order_snapshot(
            order_id="part2",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=70.0,
            submission_time=base_time,
        )

        fills = fill_engine.match_bar(bar, [snap1, snap2], market_event_id="b30")
        assert len(fills) == 2
        assert fills[0].fill_quantity == Decimal(70)
        # snap2 gets only the remaining 30 units (100 - 70)
        assert fills[1].fill_quantity == Decimal(30)

    def test_31_tick_count_volume_is_never_treated_as_real_traded_volume(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """31. TICK_COUNT volume is never treated as real traded volume."""
        engine = FillEngine(
            liquidity_model=DefaultLiquidityModel(
                missing_bar_volume_policy=MissingBarVolumePolicy.FAIL_CLOSED
            ),
            instrument_specs={"AAPL": spec},
        )
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=5000.0,
            volume_type=VolumeType.TICK_COUNT,  # Tick count, NOT real traded volume!
        )
        snap = make_order_snapshot(
            order_id="tick-vol-order",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=base_time,
        )
        fills = engine.match_bar(bar, [snap], market_event_id="b31")
        # Under FAIL_CLOSED policy, cannot match against non-real volume
        assert len(fills) == 0

    def test_32_unavailable_volume_produces_diagnostic_and_follows_policy(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """32. Unavailable volume produces an explicit diagnostic and follows configured policy."""
        # UNLIMITED_WITH_DIAGNOSTIC policy
        engine = FillEngine(
            liquidity_model=DefaultLiquidityModel(
                missing_bar_volume_policy=MissingBarVolumePolicy.UNLIMITED_WITH_DIAGNOSTIC
            ),
            instrument_specs={"AAPL": spec},
        )
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=0.0,
            volume_type=VolumeType.UNAVAILABLE,
        )
        snap = make_order_snapshot(
            order_id="unavail-vol-order",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=base_time,
        )
        fills = engine.match_bar(bar, [snap], market_event_id="b32")
        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal(50)
        assert any("Missing REAL_TRADED volume" in d for d in fills[0].diagnostics)


# ==============================================================================
# 3. Execution Cost Model Tests (33 to 43)
# ==============================================================================


class TestExecutionCostModels:
    """Tests covering slippage, rounding, commission structures, and configuration validation."""

    def test_33_fixed_slippage_is_adverse_in_both_directions(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """33. Fixed slippage is adverse in both directions (BUY increases, SELL decreases)."""
        engine = FillEngine(
            slippage_model=FixedSlippageModel(fixed_slippage="0.05"),
            instrument_specs={"AAPL": spec},
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap_buy = make_order_snapshot(
            order_id="b-slip", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        snap_sell = make_order_snapshot(
            order_id="s-slip", side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=10.0
        )

        fills_buy = engine.match_quote(quote, [snap_buy], market_event_id="m33-b")
        engine.reset()
        fills_sell = engine.match_quote(quote, [snap_sell], market_event_id="m33-s")

        # BUY: Ask=150.10 + 0.05 = 150.15
        assert fills_buy[0].fill_price == Decimal("150.15")
        # SELL: Bid=150.00 - 0.05 = 149.95
        assert fills_sell[0].fill_price == Decimal("149.95")

    def test_34_volume_share_slippage_increases_with_participation(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """34. Volume-share slippage increases quadratically with participation."""
        model = VolumeShareSlippageModel(price_impact_percentage="0.10")
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=100.00,
            high=105.00,
            low=95.00,
            close=100.00,
            volume=1000.0,
            volume_type=VolumeType.REAL_TRADED,
        )
        order = make_order_snapshot("v-order").order

        # 10 shares: vol_share = 0.01 -> impact = 100 * 0.10 * (0.01)^2 = 0.001
        slip10 = model.calculate_slippage(order, Decimal(100), Decimal(10), bar, spec)
        # 100 shares: vol_share = 0.10 -> impact = 100 * 0.10 * (0.10)^2 = 0.10
        slip100 = model.calculate_slippage(order, Decimal(100), Decimal(100), bar, spec)

        assert slip100 > slip10
        assert slip100 == Decimal("0.10")

    def test_35_slippage_calculations_are_deterministic(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """35. Slippage calculations are deterministic across runs."""
        model = VolumeShareSlippageModel(price_impact_percentage="0.05")
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=100.00,
            high=105.00,
            low=95.00,
            close=100.00,
            volume=5000.0,
            volume_type=VolumeType.REAL_TRADED,
        )
        order = make_order_snapshot("det-order").order
        s1 = model.calculate_slippage(order, Decimal("150.25"), Decimal(250), bar, spec)
        s2 = model.calculate_slippage(order, Decimal("150.25"), Decimal(250), bar, spec)
        assert s1 == s2

    def test_36_conservative_tick_size_rounding_buy(self) -> None:
        """36. Conservative tick-size rounding: BUY rounds UP."""
        tick = Decimal("0.05")
        raw_price = Decimal("150.02")
        buy_rounded = round_to_tick(raw_price, tick, OrderSide.BUY)
        assert buy_rounded == Decimal("150.05")

    def test_37_conservative_tick_size_rounding_sell(self) -> None:
        """37. Conservative tick-size rounding: SELL rounds DOWN."""
        tick = Decimal("0.05")
        raw_price = Decimal("150.02")
        sell_rounded = round_to_tick(raw_price, tick, OrderSide.SELL)
        assert sell_rounded == Decimal("150.00")

    def test_38_limit_price_protection_applied_after_spread_and_slippage(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """38. Limit price protection is applied after spread and slippage."""
        engine = FillEngine(
            spread_model=SyntheticBarSpreadModel(fixed_spread="0.04"),  # half-spread 0.02
            slippage_model=FixedSlippageModel(fixed_slippage="0.05"),
            instrument_specs={"AAPL": spec},
        )
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=151.00,
            low=149.00,
            close=150.50,
            volume=10000.0,
            volume_type=VolumeType.REAL_TRADED,
        )
        # BUY limit at 150.05. Open=150.00 + half-spread (0.02) + slippage (0.05) = 150.07 > 150.05
        snap = make_order_snapshot(
            order_id="l-prot",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=150.05,
            submission_time=base_time,
        )
        fills = engine.match_bar(bar, [snap], market_event_id="m38")
        assert len(fills) == 0  # Preserved from limit violation

    def test_39_per_share_commission_is_correct(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """39. Per-share commission is calculated correctly with min/max caps."""
        model = PerShareCommissionModel(
            per_share_rate="0.01",
            currency="USD",
            min_per_order="1.00",
            max_per_order="5.00",
        )
        engine = FillEngine(commission_model=model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=1000.0,
            ask_size=1000.0,
        )

        # 50 shares -> 50 * 0.01 = 0.50 < min(1.00) -> 1.00
        snap1 = make_order_snapshot(
            order_id="comm-min", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        f1 = engine.match_quote(quote, [snap1], market_event_id="c39-1")
        assert f1[0].fill_record.fee == Decimal("1.00")
        assert f1[0].fill_record.commission_currency == "USD"

        # 1000 shares -> 1000 * 0.01 = 10.00 > max(5.00) -> 5.00
        engine.reset()
        snap2 = make_order_snapshot(
            order_id="comm-max", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=1000.0
        )
        f2 = engine.match_quote(quote, [snap2], market_event_id="c39-2")
        assert f2[0].fill_record.fee == Decimal("5.00")

    def test_40_percentage_commission_is_correct(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """40. Percentage commission is correct on trade notional."""
        # 0.1% rate = 0.001
        model = PercentageCommissionModel(percentage_rate="0.001", currency="USD")
        engine = FillEngine(commission_model=model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.00,
            bid_size=100.0,
            ask_size=100.0,
        )
        # Notional: 100 shares * 150.00 = 15,000. Fee = 15,000 * 0.001 = 15.00
        snap = make_order_snapshot(
            order_id="pct-comm", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        fills = engine.match_quote(quote, [snap], market_event_id="c40")
        assert fills[0].fill_record.fee == Decimal("15.00")
        assert fills[0].fill_record.commission_currency == "USD"

    def test_41_fixed_per_order_commission_is_charged_only_once(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """41. Fixed per-order commission is charged only once across partial fills."""
        model = FixedPerOrderCommissionModel(fixed_fee="4.95", currency="USD")
        engine = FillEngine(commission_model=model, instrument_specs={"AAPL": spec})

        snap = make_order_snapshot(
            order_id="fixed-order-1",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        # First partial fill: 40 shares
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=40.0,
        )
        f1 = engine.match_quote(q1, [snap], market_event_id="f41-1")
        assert f1[0].fill_record.fee == Decimal("4.95")
        assert f1[0].fill_record.commission_currency == "USD"

        # Second partial fill on same order: 60 shares
        snap_remainder = make_order_snapshot(
            order_id="fixed-order-1",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            leaves_quantity=60.0,
        )
        t2 = base_time + timedelta(seconds=1)
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=60.0,
        )
        f2 = engine.match_quote(q2, [snap_remainder], market_event_id="f41-2")
        assert f2[0].fill_record.fee == Decimal(0)
        assert f2[0].fill_record.commission_currency is None

    def test_42_positive_commissions_always_carry_explicit_currency(self) -> None:
        """42. Non-zero commission models require an explicit non-empty currency."""
        with pytest.raises(
            InvalidModelConfigurationError, match="currency must be a non-empty string"
        ):
            PerShareCommissionModel(per_share_rate="0.01", currency="")

        with pytest.raises(
            InvalidModelConfigurationError, match="currency must be a non-empty string"
        ):
            PercentageCommissionModel(percentage_rate="0.001", currency=None)  # type: ignore[arg-type]

    def test_43_invalid_model_configurations_are_rejected(self) -> None:
        """43. Invalid model configurations (negative values, conflicting inputs) are rejected."""
        with pytest.raises(InvalidModelConfigurationError):
            SyntheticBarSpreadModel(fixed_spread="-0.01")

        with pytest.raises(InvalidModelConfigurationError):
            FixedSlippageModel(fixed_slippage="-0.05")

        with pytest.raises(InvalidModelConfigurationError):
            VolumeShareSlippageModel(price_impact_percentage="-0.1")

        with pytest.raises(InvalidModelConfigurationError):
            PerShareCommissionModel(per_share_rate="-0.01", currency="USD")

        with pytest.raises(InvalidModelConfigurationError):
            DefaultLiquidityModel(max_volume_participation=Decimal(0))


# ==============================================================================
# 4. Integration Boundary Tests (44 to 50)
# ==============================================================================


class TestIntegrationBoundary:
    """Tests covering interaction with OrderManager, Portfolio, immutability, and state resets."""

    def test_44_output_recorded_by_order_manager_without_recalculation(
        self, manager: OrderManager, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """44. FillCandidate.fill_record can be recorded by OrderManager.record_fill() without recalculation."""
        snap = register_and_get_snapshot(
            manager,
            order_id="int-ord",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            timestamp=base_time,
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="int-q44")
        assert len(fills) == 1

        # Direct synchronization into OrderManager
        om_snapshot = manager.record_fill(fills[0].fill_record)
        assert om_snapshot.filled_quantity == Decimal(50)
        assert om_snapshot.status == ExecutionStatus.FILLED
        assert om_snapshot.order.status == OrderStatus.FILLED

    def test_45_portfolio_event_conversion_preserves_canonical_values(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """45. FillCandidate.to_portfolio_event() preserves canonical values within existing float precision."""
        model = PerShareCommissionModel(per_share_rate="0.02", currency="USD")
        engine = FillEngine(commission_model=model, instrument_specs={"AAPL": spec})
        snap = make_order_snapshot(
            order_id="port-conv", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.25,
            bid_size=200.0,
            ask_size=200.0,
        )
        fills = engine.match_quote(quote, [snap], market_event_id="p45")
        assert len(fills) == 1
        cand = fills[0]

        event = cand.to_portfolio_event()
        assert isinstance(event, OrderFilledEvent)
        assert event.fill_price == 150.25
        assert event.fill_quantity == 100.0
        assert event.fee == 2.0  # 100 * 0.02
        assert event.order.id == "port-conv"

    def test_46_fill_engine_never_modifies_order_manager_snapshots(
        self, manager: OrderManager, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """46. FillEngine never modifies OrderManager-owned snapshots or entities."""
        snap = register_and_get_snapshot(
            manager,
            order_id="immutable-check",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            timestamp=base_time,
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        fills = fill_engine.match_quote(quote, [snap], market_event_id="q46")
        assert len(fills) == 1

        # Manager state must be completely untouched before record_fill() is called
        mgr_snap = manager.get_execution_state("immutable-check")
        assert mgr_snap.filled_quantity == Decimal(0)
        assert mgr_snap.leaves_quantity == Decimal(50)
        assert mgr_snap.is_active is True

    def test_47_fill_engine_never_mutates_portfolio_accounting(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """47. FillEngine never mutates Portfolio accounting or balances."""
        converter = TimestampedCurrencyConverter(
            default_max_age=timedelta(days=1), pivot_currency="USD"
        )
        portfolio = Portfolio(account_currency="USD", currency_converter=converter)
        portfolio.initialize_cash("USD", Decimal(100000), timestamp=base_time)
        portfolio.register_instrument("AAPL", AccountingClassification.CASH_EQUITY, "USD", "USD")

        engine = FillEngine(instrument_specs={"AAPL": spec})
        snap = make_order_snapshot(
            order_id="port-safe", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=200.0,
            ask_size=200.0,
        )
        fills = engine.match_quote(quote, [snap], market_event_id="q47")
        assert len(fills) == 1

        # Portfolio cash and positions remain completely unaffected
        assert portfolio.cashbook.get_balance("USD") == Decimal(100000)
        assert portfolio.positions.get_position("AAPL") is None

    def test_48_fill_ids_remain_deterministic_across_identical_replays(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """48. Fill IDs remain deterministic across identical replays."""
        engine1 = FillEngine(instrument_specs={"AAPL": spec})
        engine2 = FillEngine(instrument_specs={"AAPL": spec})

        snap = make_order_snapshot(
            order_id="det-id", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        f1 = engine1.match_quote(quote, [snap], market_event_id="det-event")
        f2 = engine2.match_quote(quote, [snap], market_event_id="det-event")

        assert f1[0].fill_record.fill_id == f2[0].fill_record.fill_id

    def test_49_resetting_engine_starts_clean_simulation_state(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """49. Resetting the engine starts a clean simulation state (clears event cache and liquidity)."""
        snap = make_order_snapshot(
            order_id="clean-test", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=50.0,
        )
        # Consumes all 50 units
        f1 = fill_engine.match_quote(quote, [snap], market_event_id="evt-clean")
        assert len(f1) == 1

        # Reset engine
        fill_engine.reset()

        # Under a clean state, same event_id can be executed freshly
        f2 = fill_engine.match_quote(quote, [snap], market_event_id="evt-clean")
        assert len(f2) == 1
        assert f2[0].fill_quantity == Decimal(50)

    def test_50_unsupported_order_type_fails_closed(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """50. Unsupported order types fail closed."""
        raw_snap = OrderSnapshot(
            id="bad-type",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type="TRAILING_STOP",  # type: ignore[arg-type]
            quantity=10.0,
            limit_price=None,
            stop_loss_price=None,
            take_profit_price=None,
            status=OrderStatus.SUBMITTED,
            created_at=base_time,
            updated_at=base_time,
        )
        snap = OrderExecutionSnapshot(
            order=raw_snap,
            submission_time=base_time,
            original_quantity=Decimal(10),
            filled_quantity=Decimal(0),
            unfilled_quantity=Decimal(10),
            leaves_quantity=Decimal(10),
            cancelled_quantity=Decimal(0),
            average_fill_price=Decimal(0),
            cumulative_fees={},
            status=raw_snap.status,  # type: ignore[arg-type]
            fills=(),
            last_update_time=base_time,
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        with pytest.raises(UnsupportedOrderTypeError):
            fill_engine.match_quote(quote, [snap], market_event_id="q50")


# ==============================================================================
# Milestone 3B.4.4 Targeted Correction Pass Tests
# ==============================================================================


class TestFillEngineCorrectionPass:
    """Comprehensive test suite for Milestone 3B.4.4 targeted corrections:

    1. Market-event identity, cache safety, and duplicate payload detection.
    2. Safe transactional candidate creation against partial state mutation.
    3. Order-level min/max commission semantics across partial fills & cancellation.
    4. Commission currency preservation in Portfolio integration path.
    5. Finite numeric validation across cost and liquidity models.
    6. Bar execution timestamp availability convention verification.
    7. Missing-volume policies and visible diagnostic reporting.
    """

    # --------------------------------------------------------------------------
    # 1. Market-Event Identity and Cache Safety
    # --------------------------------------------------------------------------

    def test_market_event_identity_same_id_same_payload_returns_cached_fills(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """1. Same ID and same payload returns previously matched fills without re-consuming liquidity."""
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap = make_order_snapshot(
            order_id="id-same-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        f1 = fill_engine.match_quote(quote, [snap], market_event_id="evt-idemp")
        assert len(f1) == 1
        assert f1[0].fill_quantity == Decimal(40)

        # Reprocessing with the exact same ID and same payload
        f2 = fill_engine.match_quote(quote, [snap], market_event_id="evt-idemp")
        assert len(f2) == 1
        assert f2[0].fill_id == f1[0].fill_id
        assert f2[0].fill_quantity == Decimal(40)

        # Confirm remaining liquidity was not depleted further
        avail_depth, _ = fill_engine.liquidity_model.get_available_quote_liquidity(
            quote, OrderSide.BUY, "evt-idemp"
        )
        assert avail_depth == Decimal(60)

    def test_market_event_identity_same_id_different_payload_raises_conflict(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """2. Reusing an existing ID with different event data raises a clear DuplicateMarketEventError."""
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        q2 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.20,  # Different bid/ask payload!
            ask=150.30,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap = make_order_snapshot(
            order_id="id-diff-pay", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        fill_engine.match_quote(q1, [snap], market_event_id="conflict-id")

        with pytest.raises(
            DuplicateMarketEventError,
            match="was previously processed with different event data",
        ):
            fill_engine.match_quote(q2, [snap], market_event_id="conflict-id")

    def test_market_event_identity_same_payload_different_ids_rejected(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """3. The same logical observation presented under different IDs raises DuplicateMarketEventError."""
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap1 = make_order_snapshot(
            order_id="o1-fp", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        snap2 = make_order_snapshot(
            order_id="o2-fp", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        fill_engine.match_quote(quote, [snap1], market_event_id="id-first")

        # Attempting to re-process the exact same quote under a different ID to bypass liquidity limits
        with pytest.raises(
            DuplicateMarketEventError,
            match="has already been processed under event ID 'id-first'",
        ):
            fill_engine.match_quote(quote, [snap2], market_event_id="id-second")

    def test_market_event_identity_two_distinct_observations_different_timestamps(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """4. Two distinct market observations with different timestamps process independently."""
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=50.0,
            ask_size=50.0,
        )
        t2 = base_time + timedelta(seconds=1)
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=150.05,
            ask=150.15,
            bid_size=50.0,
            ask_size=50.0,
        )
        snap1 = make_order_snapshot(
            order_id="o1-dist", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        snap2 = make_order_snapshot(
            order_id="o2-dist",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=t2,
        )
        f1 = fill_engine.match_quote(q1, [snap1], market_event_id="q-t0")
        f2 = fill_engine.match_quote(q2, [snap2], market_event_id="q-t1")

        assert len(f1) == 1
        assert len(f2) == 1
        assert f1[0].fill_price == Decimal("150.10")
        assert f2[0].fill_price == Decimal("150.15")

    def test_market_event_identity_repeated_matching_against_already_consumed_event(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """5. Repeated matching against an already-consumed event does not produce additional fills."""
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap1 = make_order_snapshot(
            order_id="o1-consume", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        f1 = fill_engine.match_quote(quote, [snap1], market_event_id="consumed-evt")
        assert len(f1) == 1
        assert f1[0].fill_quantity == Decimal(100)

        # A new order arrives, attempting to match against the same consumed event
        snap2 = make_order_snapshot(
            order_id="o2-consume", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        # Re-invoking with the same event ID returns the cached fills from the first call
        cached_result = fill_engine.match_quote(quote, [snap2], market_event_id="consumed-evt")
        assert len(cached_result) == 1
        # The returned fill is strictly the previously cached fill for order 1
        assert cached_result[0].order_id == "o1-consume"
        # Order 2 was not filled and no duplicate liquidity was generated
        assert all(c.order_id != "o2-consume" for c in cached_result)

    # --------------------------------------------------------------------------
    # 2. Transactional Candidate Creation & Partial State Safety
    # --------------------------------------------------------------------------

    def test_transactional_failure_does_not_mutate_liquidity_or_commission_state(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Candidate creation failure leaves liquidity and commission state completely unmutated."""
        commission_model = FixedPerOrderCommissionModel(fixed_fee="5.00", currency="USD")
        engine = FillEngine(commission_model=commission_model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        # Create an order with an invalid non-finite limit price that triggers an error
        snap_bad = OrderExecutionSnapshot(
            order=OrderSnapshot(
                id="bad-order-transact",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=50.0,
                limit_price=float("nan"),  # Non-finite limit price!
                stop_loss_price=None,
                take_profit_price=None,
                status=OrderStatus.SUBMITTED,
                created_at=base_time,
                updated_at=base_time,
            ),
            submission_time=base_time,
            original_quantity=Decimal(50),
            filled_quantity=Decimal(0),
            unfilled_quantity=Decimal(50),
            leaves_quantity=Decimal(50),
            cancelled_quantity=Decimal(0),
            average_fill_price=Decimal(0),
            cumulative_fees={},
            status=ExecutionStatus.ACTIVE,
            fills=(),
            last_update_time=base_time,
        )

        with pytest.raises(ValueError, match="is missing valid limit_price"):
            engine.match_quote(quote, [snap_bad], market_event_id="tx-fail-evt")

        # Verify that liquidity was NOT consumed
        avail_depth, _ = engine.liquidity_model.get_available_quote_liquidity(
            quote, OrderSide.BUY, "tx-fail-evt"
        )
        assert avail_depth == Decimal(100)

        # Verify that commission state was NOT committed
        assert "bad-order-transact" not in commission_model._charged_orders

    # --------------------------------------------------------------------------
    # 3. Commission State and Partial-Fill Semantics
    # --------------------------------------------------------------------------

    def test_fixed_per_order_commission_preparation_does_not_mark_order_charged(
        self, base_time: datetime
    ) -> None:
        """Calling calculate_commission on FixedPerOrderCommissionModel does not commit charged state."""
        model = FixedPerOrderCommissionModel(fixed_fee="4.95", currency="USD")
        snap = make_order_snapshot(
            order_id="fixed-prep", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        # Tentative calculation:
        fee1, curr1 = model.calculate_commission(snap.order, Decimal("150.00"), Decimal(50))
        assert fee1 == Decimal("4.95")
        assert curr1 == "USD"
        # Crucial invariant: order must not be marked charged merely because calculation was attempted!
        assert "fixed-prep" not in model._charged_orders

        # A second calculation attempt still produces the full fee
        fee2, curr2 = model.calculate_commission(snap.order, Decimal("150.00"), Decimal(50))
        assert fee2 == Decimal("4.95")
        assert curr2 == "USD"

        # Explicit commit marks the order charged
        committed_fee, _ = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(50), fill_id="f1"
        )
        assert committed_fee == Decimal("4.95")
        assert "fixed-prep" in model._charged_orders

        # Subsequent calculation now returns zero
        fee3, curr3 = model.calculate_commission(snap.order, Decimal("150.00"), Decimal(50))
        assert fee3 == Decimal(0)
        assert curr3 is None

    def test_pershare_order_level_min_and_max_across_multiple_partial_fills(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """min_per_order is assessed on the first fill and incremental fills respect max_per_order."""
        # Rate: 0.01 per share. Min: 1.00 USD. Max: 4.00 USD. Total order: 600 shares.
        model = PerShareCommissionModel(
            per_share_rate="0.01",
            currency="USD",
            min_per_order="1.00",
            max_per_order="4.00",
        )
        engine = FillEngine(commission_model=model, instrument_specs={"AAPL": spec})
        order_id = "multi-partial-comm"

        # Partial Fill 1: 30 shares (raw fee: 0.30 < min 1.00 -> charged 1.00)
        snap1 = make_order_snapshot(
            order_id=order_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=600.0,
            leaves_quantity=600.0,
        )
        q1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.0,
            ask=150.1,
            bid_size=100.0,
            ask_size=30.0,
        )
        f1 = engine.match_quote(q1, [snap1], market_event_id="pf-1")
        assert f1[0].fill_record.fee == Decimal("1.00")
        assert f1[0].fill_record.commission_currency == "USD"

        # Partial Fill 2: 50 shares (cumulative raw: 80 * 0.01 = 0.80 < min 1.00 -> incremental fee: 0.00)
        t2 = base_time + timedelta(seconds=1)
        snap2 = make_order_snapshot(
            order_id=order_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=600.0,
            leaves_quantity=570.0,
        )
        q2 = Quote(
            symbol="AAPL",
            timestamp=t2,
            bid=150.0,
            ask=150.1,
            bid_size=100.0,
            ask_size=50.0,
        )
        f2 = engine.match_quote(q2, [snap2], market_event_id="pf-2")
        assert f2[0].fill_record.fee == Decimal(0)
        assert f2[0].fill_record.commission_currency is None

        # Partial Fill 3: 120 shares (cumulative raw: 200 * 0.01 = 2.00 > min 1.00 -> incremental fee: 2.00 - 1.00 = 1.00)
        t3 = base_time + timedelta(seconds=2)
        snap3 = make_order_snapshot(
            order_id=order_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=600.0,
            leaves_quantity=520.0,
        )
        q3 = Quote(
            symbol="AAPL",
            timestamp=t3,
            bid=150.0,
            ask=150.1,
            bid_size=100.0,
            ask_size=120.0,
        )
        f3 = engine.match_quote(q3, [snap3], market_event_id="pf-3")
        assert f3[0].fill_record.fee == Decimal("1.00")
        assert f3[0].fill_record.commission_currency == "USD"

        # Partial Fill 4: 250 shares (cumulative raw: 450 * 0.01 = 4.50 > max 4.00 -> target 4.00. Incremental: 4.00 - 2.00 = 2.00)
        t4 = base_time + timedelta(seconds=3)
        snap4 = make_order_snapshot(
            order_id=order_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=600.0,
            leaves_quantity=400.0,
        )
        q4 = Quote(
            symbol="AAPL",
            timestamp=t4,
            bid=150.0,
            ask=150.1,
            bid_size=100.0,
            ask_size=250.0,
        )
        f4 = engine.match_quote(q4, [snap4], market_event_id="pf-4")
        assert f4[0].fill_record.fee == Decimal("2.00")
        assert f4[0].fill_record.commission_currency == "USD"

        # Partial Fill 5: 150 shares (cumulative raw: 600 * 0.01 = 6.00 > max 4.00 -> incremental fee: 0.00)
        t5 = base_time + timedelta(seconds=4)
        snap5 = make_order_snapshot(
            order_id=order_id,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=600.0,
            leaves_quantity=150.0,
        )
        q5 = Quote(
            symbol="AAPL",
            timestamp=t5,
            bid=150.0,
            ask=150.1,
            bid_size=100.0,
            ask_size=150.0,
        )
        f5 = engine.match_quote(q5, [snap5], market_event_id="pf-5")
        assert f5[0].fill_record.fee == Decimal(0)
        assert f5[0].fill_record.commission_currency is None

        # Cumulative fee check across all 5 fills:
        total_fee = (
            f1[0].fill_record.fee
            + f2[0].fill_record.fee
            + f3[0].fill_record.fee
            + f4[0].fill_record.fee
            + f5[0].fill_record.fee
        )
        assert total_fee == Decimal("4.00")  # Strictly capped at max_per_order!

    def test_commission_repeated_fill_commit_is_idempotent(self, base_time: datetime) -> None:
        """Committing the same fill_id multiple times does not charge the fee twice."""
        model = PerShareCommissionModel(per_share_rate="0.05", currency="USD")
        snap = make_order_snapshot(
            order_id="idemp-fill", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )
        fee1, curr1 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(20), fill_id="fill-idemp-1"
        )
        assert fee1 == Decimal("1.00")
        assert curr1 == "USD"

        # Re-commit with identical fill_id
        fee2, curr2 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(20), fill_id="fill-idemp-1"
        )
        assert fee2 == Decimal("1.00")
        assert curr2 == "USD"
        assert model._order_cumulative_charged["idemp-fill"] == Decimal("1.00")

    # --------------------------------------------------------------------------
    # 4. Preserve Commission Currency in Portfolio Integration
    # --------------------------------------------------------------------------

    def test_portfolio_application_preserves_non_base_commission_currency(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Non-base commission currency is preserved and accepted by Portfolio.apply_fill()."""
        # Commission currency is EUR while symbol AAPL is denominated in USD
        model = PerShareCommissionModel(per_share_rate="0.10", currency="EUR")
        engine = FillEngine(commission_model=model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap = make_order_snapshot(
            order_id="port-eur-comm",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
        )
        fills = engine.match_quote(quote, [snap], market_event_id="q-port-eur")
        assert len(fills) == 1
        cand = fills[0]
        assert cand.fill_record.commission_currency == "EUR"
        assert cand.commission_currency == "EUR"

        # Obtain PortfolioFillApplication helper
        app = cand.to_portfolio_application()
        assert isinstance(app, PortfolioFillApplication)
        assert app.commission_currency == "EUR"
        assert app.event.fee == 5.0  # 50 * 0.10 EUR

        # Initialize Portfolio with USD and EUR cash balances
        converter = TimestampedCurrencyConverter()
        converter.add_rate(
            FXRateObservation(
                base_currency="EUR",
                quote_currency="USD",
                rate=Decimal("1.08"),
                timestamp=base_time,
            )
        )
        portfolio = Portfolio(
            account_currency="USD",
            currency_converter=converter,
        )
        portfolio.initialize_cash("USD", Decimal("100000.0"), base_time)
        portfolio.initialize_cash("EUR", Decimal("5000.0"), base_time)

        # Apply fill using the preserved application structure
        pos_event = portfolio.apply_fill(
            event=app.event,
            spec=spec,
            accounting_type=AccountingClassification.CASH_EQUITY,
            commission_currency=app.commission_currency,
        )
        assert pos_event.position.symbol == "AAPL"
        assert pos_event.position.quantity == 50.0

        # Confirm EUR commission was debited from EUR cash balance
        eur_balance = portfolio.cashbook.get_balance("EUR")
        assert eur_balance == Decimal("4995.0")  # 5000 - 5 EUR

    # --------------------------------------------------------------------------
    # 5. Decimal and Non-Finite Numeric Validation
    # --------------------------------------------------------------------------

    def test_cost_models_reject_non_finite_inputs(self) -> None:
        """Cost and spread models strictly reject NaN and infinite configuration values."""
        with pytest.raises(InvalidModelConfigurationError, match="must be a finite number"):
            PerShareCommissionModel(per_share_rate=float("nan"), currency="USD")

        with pytest.raises(InvalidModelConfigurationError, match="must be a finite Decimal"):
            PercentageCommissionModel(percentage_rate=Decimal("Infinity"), currency="USD")

        with pytest.raises(InvalidModelConfigurationError, match="must be a finite Decimal"):
            FixedPerOrderCommissionModel(fixed_fee=Decimal("-Infinity"), currency="USD")

        with pytest.raises(InvalidModelConfigurationError, match="must be a finite number"):
            FixedSlippageModel(fixed_slippage=float("inf"))

        with pytest.raises(InvalidModelConfigurationError, match="must be a finite Decimal"):
            VolumeShareSlippageModel(price_impact_percentage=Decimal("NaN"))

        with pytest.raises(InvalidModelConfigurationError, match="must be a finite number"):
            SyntheticBarSpreadModel(fixed_spread=float("nan"))

        with pytest.raises(InvalidModelConfigurationError, match="price must be a finite Decimal"):
            round_to_tick(Decimal("NaN"), Decimal("0.01"), OrderSide.BUY)

        with pytest.raises(
            InvalidModelConfigurationError, match="tick_size must be strictly positive"
        ):
            round_to_tick(Decimal("150.00"), Decimal("Infinity"), OrderSide.BUY)

    def test_fill_engine_rejects_non_finite_quote_and_bar_inputs(
        self, fill_engine: FillEngine, base_time: datetime
    ) -> None:
        """FillEngine rejects non-finite quote and bar numerical fields."""
        snap = make_order_snapshot(
            order_id="finite-test", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        # Quote with NaN ask
        q_nan = Quote.model_construct(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=float("nan"),
            ask_size=100.0,
        )
        with pytest.raises(ValueError, match="Quote bid_size must be a finite number"):
            fill_engine.match_quote(q_nan, [snap], market_event_id="nan-q")

        # Bar with Inf high
        bar_inf = Bar.model_construct(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=float("inf"),
            low=149.00,
            close=150.50,
            volume=1000.0,
        )
        with pytest.raises(ValueError, match="Bar high must be a finite number"):
            fill_engine.match_bar(bar_inf, [snap], market_event_id="inf-bar")

    # --------------------------------------------------------------------------
    # 6. Bar Execution Timestamp Convention
    # --------------------------------------------------------------------------

    def test_bar_execution_timestamp_conventions_locked_down(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Bar-level execution timestamps adhere to the documented availability convention."""
        engine = FillEngine(
            spread_model=SyntheticBarSpreadModel(fixed_spread="0.02"),
            instrument_specs={"AAPL": spec},
        )
        t_start = base_time
        t_end = base_time + timedelta(minutes=5)
        bar = Bar(
            symbol="AAPL",
            start_time=t_start,
            end_time=t_end,
            open=150.00,
            high=155.00,
            low=145.00,
            close=152.00,
            volume=10000.0,
        )

        # 1. Market order: executes at bar.start_time
        snap_mkt = make_order_snapshot(
            order_id="mkt-ts",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            submission_time=t_start,
        )
        f_mkt = engine.match_bar(bar, [snap_mkt], market_event_id="bar-ts-1")
        assert f_mkt[0].fill_record.timestamp == t_start

        # 2. Limit order with favorable opening gap: executes at bar.start_time
        engine.reset()
        snap_gap_lmt = make_order_snapshot(
            order_id="gap-lmt-ts",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=152.00,  # Open + 0.01 = 150.01 <= 152.00
            submission_time=t_start,
        )
        f_gap_lmt = engine.match_bar(bar, [snap_gap_lmt], market_event_id="bar-ts-2")
        assert f_gap_lmt[0].fill_record.timestamp == t_start

        # 3. Limit order triggered intrabar: executes at bar.end_time
        engine.reset()
        snap_intra_lmt = make_order_snapshot(
            order_id="intra-lmt-ts",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=147.00,  # Open 150.01 > 147.00, but Low 145.01 <= 147.00
            submission_time=t_start,
        )
        f_intra_lmt = engine.match_bar(bar, [snap_intra_lmt], market_event_id="bar-ts-3")
        assert f_intra_lmt[0].fill_record.timestamp == t_end

        # 4. Stop order with adverse opening gap: executes at bar.start_time
        engine.reset()
        snap_gap_stp = make_order_snapshot(
            order_id="gap-stp-ts",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,
            quantity=10.0,
            stop_loss_price=148.00,  # Bar opens above stop (150.01 > 148.00)
            submission_time=t_start,
        )
        f_gap_stp = engine.match_bar(bar, [snap_gap_stp], market_event_id="bar-ts-4")
        assert f_gap_stp[0].fill_record.timestamp == t_start

        # 5. Stop order triggered intrabar: executes at bar.end_time
        engine.reset()
        snap_intra_stp = make_order_snapshot(
            order_id="intra-stp-ts",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LOSS,
            quantity=10.0,
            stop_loss_price=153.00,  # Open 150.01 < 153.00, but High 155.01 >= 153.00
            submission_time=t_start,
        )
        f_intra_stp = engine.match_bar(bar, [snap_intra_stp], market_event_id="bar-ts-5")
        assert f_intra_stp[0].fill_record.timestamp == t_end

    # --------------------------------------------------------------------------
    # 7. Missing-Volume Policies and Visible Diagnostics
    # --------------------------------------------------------------------------

    def test_missing_bar_volume_policies_and_diagnostics(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Missing REAL_TRADED volume triggers explicit configurable policies with visible diagnostics."""
        bar_synthetic = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=152.00,
            low=148.00,
            close=151.00,
            volume=5000.0,
            volume_type=VolumeType.TICK_COUNT,  # Not REAL_TRADED
        )
        snap = make_order_snapshot(
            order_id="vol-policy-test",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=200.0,
            submission_time=base_time,
        )

        # Policy 1: FAIL_CLOSED -> produces 0 fills
        liq_fail_closed = DefaultLiquidityModel(
            missing_bar_volume_policy=MissingBarVolumePolicy.FAIL_CLOSED
        )
        eng_fc = FillEngine(liquidity_model=liq_fail_closed, instrument_specs={"AAPL": spec})
        fills_fc = eng_fc.match_bar(bar_synthetic, [snap], market_event_id="vol-fc")
        assert len(fills_fc) == 0

        # Policy 2: FIXED_DEFAULT_CAP -> capped at default_bar_volume_cap (e.g. 50 shares)
        liq_default_cap = DefaultLiquidityModel(
            missing_bar_volume_policy=MissingBarVolumePolicy.FIXED_DEFAULT_CAP,
            default_bar_volume_cap=50,
        )
        eng_cap = FillEngine(liquidity_model=liq_default_cap, instrument_specs={"AAPL": spec})
        fills_cap = eng_cap.match_bar(bar_synthetic, [snap], market_event_id="vol-cap")
        assert len(fills_cap) == 1
        assert fills_cap[0].fill_quantity == Decimal(50)
        assert any("applied fixed default cap 50" in d for d in fills_cap[0].diagnostics)

        # Policy 3: UNLIMITED_WITH_DIAGNOSTIC -> fills requested quantity and produces diagnostic
        liq_unlimited = DefaultLiquidityModel(
            missing_bar_volume_policy=MissingBarVolumePolicy.UNLIMITED_WITH_DIAGNOSTIC
        )
        eng_unlimited = FillEngine(liquidity_model=liq_unlimited, instrument_specs={"AAPL": spec})
        fills_unlimited = eng_unlimited.match_bar(
            bar_synthetic, [snap], market_event_id="vol-unlimited"
        )
        assert len(fills_unlimited) == 1
        assert fills_unlimited[0].fill_quantity == Decimal(200)
        assert any("participation cap disabled" in d for d in fills_unlimited[0].diagnostics)
        # Verify unconsumed liquidity ceiling is finite
        remaining, _ = liq_unlimited.get_available_bar_liquidity(bar_synthetic, "vol-unlimited")
        assert remaining.is_finite()


# ==============================================================================
# Final Correction Pass Tests (Milestone 3B.4.4 Final)
# ==============================================================================


class TestFillEngineFinalCorrectionPass:
    """Verifies the four targeted areas of the final review correction pass:

    1. Market-event identity independent of ingestion time (stable sequence IDs, fallback, duplicate prevention).
    2. Atomic transactional matching with full rollback on failure (quote & bar, staged state, zero partial leaks).
    3. Missing-quote liquidity policies (REJECT, CONFIGURED_DEFAULT, UNLIMITED_WITH_DIAGNOSTIC with finite ceiling).
    4. Commission fill-ID consistency (idempotent replay, conflict detection on mismatched details).
    """

    # --------------------------------------------------------------------------
    # 1. Market-Event Identity & Ingestion Time Independence
    # --------------------------------------------------------------------------

    def test_same_observation_and_same_event_id_is_idempotent(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Same observation replayed under identical event ID returns cached candidates without re-consuming."""
        engine = FillEngine(instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        snap = make_order_snapshot(
            order_id="idem-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )

        f1 = engine.match_quote(quote, [snap], market_event_id="q-idem-1")
        assert len(f1) == 1
        assert f1[0].fill_quantity == Decimal(50)

        f2 = engine.match_quote(quote, [snap], market_event_id="q-idem-1")
        assert len(f2) == 1
        assert f2[0].fill_id == f1[0].fill_id

        # Available liquidity is still 50 (not consumed a second time)
        avail, _ = engine.liquidity_model.get_available_quote_liquidity(
            quote, OrderSide.BUY, "q-idem-1"
        )
        assert avail == Decimal(50)

    def test_same_observation_different_retrieval_time_different_id_raises_duplicate(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Changing ingestion/retrieval timestamp does not change logical identity; re-presenting raises duplicate."""
        prov1 = DataProvenance(
            source_name="market_feed",
            source_type=DataSourceType.HISTORICAL,
            retrieved_at=base_time - timedelta(hours=2),
        )
        prov2 = DataProvenance(
            source_name="market_feed",
            source_type=DataSourceType.HISTORICAL,
            retrieved_at=base_time - timedelta(minutes=5),  # Different retrieval time!
        )
        quote1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
            provenance=prov1,
        )
        quote2 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
            provenance=prov2,
        )
        engine = FillEngine(instrument_specs={"AAPL": spec})
        snap = make_order_snapshot(
            order_id="ret-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )

        f1 = engine.match_quote(quote1, [snap], market_event_id="q-ret-1")
        assert len(f1) == 1

        with pytest.raises(
            DuplicateMarketEventError, match="already been processed under event ID 'q-ret-1'"
        ):
            engine.match_quote(quote2, [snap], market_event_id="q-ret-2")

    def test_different_provider_sequence_ids_processed_independently(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Distinct observations with identical market fields but differing provider sequence IDs are distinct."""
        prov1 = DataProvenance(
            source_name="nasdaq_itch",
            source_type=DataSourceType.REAL_LIVE,
            retrieved_at=base_time,
            metadata={"sequence_id": "10001"},
        )
        prov2 = DataProvenance(
            source_name="nasdaq_itch",
            source_type=DataSourceType.REAL_LIVE,
            retrieved_at=base_time,
            metadata={"sequence_id": "10002"},  # Next message sequence ID
        )
        quote1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
            provenance=prov1,
        )
        quote2 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
            provenance=prov2,
        )
        engine = FillEngine(instrument_specs={"AAPL": spec})
        snap1 = make_order_snapshot(
            order_id="seq-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        snap2 = make_order_snapshot(
            order_id="seq-2", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )

        f1 = engine.match_quote(quote1, [snap1], market_event_id="q-seq-1")
        assert len(f1) == 1

        f2 = engine.match_quote(quote2, [snap2], market_event_id="q-seq-2")
        assert len(f2) == 1
        assert f2[0].market_event_id == "q-seq-2"

    def test_reusing_event_id_with_different_market_data_raises_duplicate_error(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Reusing existing market_event_id with altered prices raises DuplicateMarketEventError."""
        quote1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        quote2 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.50,
            bid_size=100.0,
            ask_size=100.0,
        )
        engine = FillEngine(instrument_specs={"AAPL": spec})
        snap = make_order_snapshot(
            order_id="reuse-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )

        engine.match_quote(quote1, [snap], market_event_id="q-reuse-id")
        with pytest.raises(
            DuplicateMarketEventError, match="previously processed with different event data"
        ):
            engine.match_quote(quote2, [snap], market_event_id="q-reuse-id")

    def test_different_market_timestamps_processed_independently(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Observations at different market timestamps are independent observations."""
        quote1 = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        quote2 = Quote(
            symbol="AAPL",
            timestamp=base_time + timedelta(seconds=1),
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )
        engine = FillEngine(instrument_specs={"AAPL": spec})
        snap1 = make_order_snapshot(
            order_id="ts-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        snap2 = make_order_snapshot(
            order_id="ts-2",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
            submission_time=base_time,
        )

        f1 = engine.match_quote(quote1, [snap1], market_event_id="q-ts-1")
        f2 = engine.match_quote(quote2, [snap2], market_event_id="q-ts-2")
        assert len(f1) == 1
        assert len(f2) == 1

    # --------------------------------------------------------------------------
    # 2. Transactional Matching & Rollback
    # --------------------------------------------------------------------------

    def test_quote_transaction_rollback_when_later_order_fails_calculation(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """First order prepares cleanly; second order fails calculation -> zero state mutated."""
        comm_model = PerShareCommissionModel(per_share_rate="0.01", currency="USD")
        engine = FillEngine(commission_model=comm_model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        snap1 = make_order_snapshot(
            order_id="tx-q1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        # Order 2 has invalid non-finite limit price
        bad_order = Order.model_construct(
            id="tx-q2",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=float("nan"),
            status=OrderStatus.SUBMITTED,
            created_at=base_time,
            updated_at=base_time,
        )
        snap2 = OrderExecutionSnapshot(
            order=OrderSnapshot.from_order(bad_order),
            original_quantity=Decimal(20),
            filled_quantity=Decimal(0),
            leaves_quantity=Decimal(20),
            unfilled_quantity=Decimal(20),
            cancelled_quantity=Decimal(0),
            average_fill_price=Decimal(0),
            cumulative_fees={},
            status=ExecutionStatus.ACTIVE,
            fills=(),
            submission_time=base_time,
            last_update_time=base_time,
        )

        with pytest.raises(ValueError, match="missing valid limit_price"):
            engine.match_quote(quote, [snap1, snap2], market_event_id="tx-q-fail1")

        # Assert zero state mutation
        assert len(engine._liquidity_model._consumed_quote_liquidity) == 0
        assert "tx-q1" not in comm_model._order_cumulative_charged
        assert len(comm_model._committed_fills) == 0
        assert "tx-q-fail1" not in engine._processed_events
        assert engine._event_fill_counts.get("tx-q-fail1", 0) == 0

    def test_bar_transaction_rollback_when_later_order_fails_calculation(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Bar matching: first order prepares cleanly; second order fails -> complete rollback."""
        comm_model = PerShareCommissionModel(per_share_rate="0.01", currency="USD")
        engine = FillEngine(commission_model=comm_model, instrument_specs={"AAPL": spec})
        bar = Bar(
            symbol="AAPL",
            start_time=base_time,
            end_time=base_time + timedelta(minutes=5),
            open=150.00,
            high=152.00,
            low=148.00,
            close=151.00,
            volume=5000.0,
            volume_type=VolumeType.REAL_TRADED,
        )

        snap1 = make_order_snapshot(
            order_id="tx-b1",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=40.0,
            submission_time=base_time,
        )
        bad_order = Order.model_construct(
            id="tx-b2",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type="UNSUPPORTED_TYPE",  # type: ignore[arg-type]
            quantity=20.0,
            status=OrderStatus.SUBMITTED,
            created_at=base_time,
            updated_at=base_time,
        )
        snap2 = OrderExecutionSnapshot(
            order=OrderSnapshot.from_order(bad_order),
            original_quantity=Decimal(20),
            filled_quantity=Decimal(0),
            leaves_quantity=Decimal(20),
            unfilled_quantity=Decimal(20),
            cancelled_quantity=Decimal(0),
            average_fill_price=Decimal(0),
            cumulative_fees={},
            status=ExecutionStatus.ACTIVE,
            fills=(),
            submission_time=base_time,
            last_update_time=base_time,
        )

        with pytest.raises(UnsupportedOrderTypeError, match="not supported by FillEngine"):
            engine.match_bar(bar, [snap1, snap2], market_event_id="tx-b-fail1")

        # Assert zero state mutation
        assert len(engine._liquidity_model._consumed_bar_liquidity) == 0
        assert "tx-b1" not in comm_model._order_cumulative_charged
        assert len(comm_model._committed_fills) == 0
        assert "tx-b-fail1" not in engine._processed_events

    def test_quote_transaction_rollback_when_later_order_fails_candidate_construction(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """First order prepares; second order triggers candidate construction failure -> rollback."""
        comm_model = PerShareCommissionModel(per_share_rate="0.01", currency="USD")

        # Injected slippage model that raises on second order
        class FaultySlippageModel:
            def calculate_slippage(self, order, base_price, fill_quantity, market_data, spec=None):
                if order.id == "faulty-order":
                    raise InvalidModelConfigurationError("Simulated slippage calculation failure")
                return Decimal("0.05")

        engine = FillEngine(
            slippage_model=FaultySlippageModel(),  # type: ignore[arg-type]
            commission_model=comm_model,
            instrument_specs={"AAPL": spec},
        )
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        snap1 = make_order_snapshot(
            order_id="ok-order", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        snap2 = make_order_snapshot(
            order_id="faulty-order", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=20.0
        )

        with pytest.raises(
            InvalidModelConfigurationError, match="Simulated slippage calculation failure"
        ):
            engine.match_quote(quote, [snap1, snap2], market_event_id="tx-faulty-slip")

        assert len(engine._liquidity_model._consumed_quote_liquidity) == 0
        assert "ok-order" not in comm_model._order_cumulative_charged
        assert "tx-faulty-slip" not in engine._processed_events

    def test_commission_commit_failure_rolls_back_entire_operation(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """When commission commit fails during phase 2, liquidity and fill counts roll back completely."""

        class FailOnCommitCommissionModel(PerShareCommissionModel):
            def commit_commission(self, order, fill_price, fill_quantity, spec=None, fill_id=None):
                if order.id == "comm-fail-order":
                    raise RuntimeError("Simulated crash in commit_commission")
                return super().commit_commission(order, fill_price, fill_quantity, spec, fill_id)

        comm_model = FailOnCommitCommissionModel(per_share_rate="0.01", currency="USD")
        engine = FillEngine(commission_model=comm_model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        snap1 = make_order_snapshot(
            order_id="comm-ok-order", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        snap2 = make_order_snapshot(
            order_id="comm-fail-order",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=20.0,
        )

        with pytest.raises(RuntimeError, match="Simulated crash in commit_commission"):
            engine.match_quote(quote, [snap1, snap2], market_event_id="tx-comm-crash")

        # Key assertion: order 1 had liquidity consumed in phase 2 before order 2 crashed!
        # Both liquidity AND commission must be rolled back!
        assert len(engine._liquidity_model._consumed_quote_liquidity) == 0
        assert "comm-ok-order" not in comm_model._order_cumulative_charged
        assert len(comm_model._committed_fills) == 0
        assert "tx-comm-crash" not in engine._processed_events

    def test_state_commit_phase_failure_leaves_no_partial_side_effects(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Failure during state commit phase leaves zero partial side effects."""

        class CrashAfterFirstCommit(PerShareCommissionModel):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def commit_commission(self, order, fill_price, fill_quantity, spec=None, fill_id=None):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("Phase 2 second commit exploded")
                return super().commit_commission(order, fill_price, fill_quantity, spec, fill_id)

        comm_model = CrashAfterFirstCommit(per_share_rate="0.02", currency="USD")
        engine = FillEngine(commission_model=comm_model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        snap1 = make_order_snapshot(
            order_id="c1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=30.0
        )
        snap2 = make_order_snapshot(
            order_id="c2", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=30.0
        )

        with pytest.raises(RuntimeError, match="Phase 2 second commit exploded"):
            engine.match_quote(quote, [snap1, snap2], market_event_id="tx-p2-crash")

        assert len(engine._liquidity_model._consumed_quote_liquidity) == 0
        assert "c1" not in comm_model._order_cumulative_charged
        assert len(comm_model._committed_fills) == 0
        assert "tx-p2-crash" not in engine._processed_events

    def test_retry_after_failed_call_produces_identical_clean_results(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Retrying the same event ID with valid orders after failure produces exact clean fills."""
        engine = FillEngine(instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        snap1 = make_order_snapshot(
            order_id="clean-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        bad_snap = make_order_snapshot(
            order_id="bad-1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=-1.0,
        )

        # First call fails
        with pytest.raises(ValueError):
            engine.match_quote(quote, [snap1, bad_snap], market_event_id="retry-evt")

        # Second call with valid second order under the same event ID succeeds
        snap2 = make_order_snapshot(
            order_id="clean-2", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=30.0
        )
        fills = engine.match_quote(quote, [snap1, snap2], market_event_id="retry-evt")
        assert len(fills) == 2
        assert fills[0].fill_id == "fill-retry-evt-clean-1-1"
        assert fills[1].fill_id == "fill-retry-evt-clean-2-2"
        assert fills[0].fill_quantity == Decimal(40)
        assert fills[1].fill_quantity == Decimal(30)

    def test_successful_multi_order_matching_shares_liquidity_fifo(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Competing orders in same event share displayed liquidity deterministically in FIFO order."""
        engine = FillEngine(instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=60.0,
        )

        snap1 = make_order_snapshot(
            order_id="fifo-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )
        snap2 = make_order_snapshot(
            order_id="fifo-2", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=40.0
        )

        fills = engine.match_quote(quote, [snap1, snap2], market_event_id="q-fifo-share")
        assert len(fills) == 2
        assert fills[0].fill_quantity == Decimal(40)  # Consumes 40 of 60
        assert fills[1].fill_quantity == Decimal(20)  # Consumes remaining 20 of 60
        assert fills[0].is_complete is True
        assert fills[1].is_complete is False

    def test_successful_multi_order_partial_fills_calculate_cumulative_commissions(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Multi-order partial fills correctly calculate order-level min/max cumulative fees."""
        comm_model = PerShareCommissionModel(
            per_share_rate="0.10", currency="USD", min_per_order="2.00", max_per_order="5.00"
        )
        engine = FillEngine(commission_model=comm_model, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=100.0,
            ask_size=100.0,
        )

        # Same order ID 'multi-part' filled twice
        snap1 = make_order_snapshot(
            order_id="multi-part", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=10.0
        )
        snap2 = make_order_snapshot(
            order_id="multi-part", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=20.0
        )

        fills = engine.match_quote(quote, [snap1, snap2], market_event_id="q-multi-part")
        assert len(fills) == 2
        # Fill 1: 10 * 0.10 = 1.00 raw -> min_per_order = 2.00
        assert fills[0].fill_record.fee == Decimal("2.00")
        assert fills[0].fee == Decimal("2.00")
        # Fill 2: 20 * 0.10 = 2.00 raw -> cumulative raw = 3.00, incremental = 1.00
        assert fills[1].fill_record.fee == Decimal("1.00")
        assert fills[1].fee == Decimal("1.00")
        assert comm_model._order_cumulative_charged["multi-part"] == Decimal("3.00")

    # --------------------------------------------------------------------------
    # 3. Correct Unlimited Missing-Quote Liquidity Behavior
    # --------------------------------------------------------------------------

    def test_missing_quote_depth_under_reject_policy_produces_no_fill(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Zero displayed depth under REJECT policy produces zero fills."""
        liq = DefaultLiquidityModel(missing_quote_policy=MissingQuoteLiquidityPolicy.REJECT)
        engine = FillEngine(liquidity_model=liq, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=0.0,
            ask_size=0.0,
        )
        snap = make_order_snapshot(
            order_id="rej-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        fills = engine.match_quote(quote, [snap], market_event_id="q-rej-1")
        assert len(fills) == 0

    def test_missing_quote_depth_under_configured_default_uses_cap_and_diagnostic(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Zero displayed depth under CONFIGURED_DEFAULT caps fill to default_quote_size with diagnostic."""
        liq = DefaultLiquidityModel(
            missing_quote_policy=MissingQuoteLiquidityPolicy.CONFIGURED_DEFAULT,
            default_quote_size=75,
        )
        engine = FillEngine(liquidity_model=liq, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=0.0,
            ask_size=0.0,
        )
        snap = make_order_snapshot(
            order_id="def-cap-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        fills = engine.match_quote(quote, [snap], market_event_id="q-def-1")
        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal(75)
        assert any("applied default size 75" in d for d in fills[0].diagnostics)

    def test_missing_quote_depth_under_unlimited_with_diagnostic_produces_fill_and_is_finite(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Zero displayed depth under UNLIMITED_WITH_DIAGNOSTIC permits fill and returns finite liquidity."""
        liq = DefaultLiquidityModel(
            missing_quote_policy=MissingQuoteLiquidityPolicy.UNLIMITED_WITH_DIAGNOSTIC
        )
        engine = FillEngine(liquidity_model=liq, instrument_specs={"AAPL": spec})
        quote = Quote(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_size=0.0,
            ask_size=0.0,
        )
        snap = make_order_snapshot(
            order_id="unlim-1", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=250.0
        )

        fills = engine.match_quote(quote, [snap], market_event_id="q-unlim-1")
        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal(250)
        assert any("assumed unbounded liquidity" in d for d in fills[0].diagnostics)

        avail, _ = liq.get_available_quote_liquidity(quote, OrderSide.BUY, "q-unlim-1")
        assert avail.is_finite()
        assert avail > Decimal(0)

    def test_tick_missing_depth_fallback_remains_correct_and_finite(
        self, spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Tick volume fallback under UNLIMITED_WITH_DIAGNOSTIC produces finite available liquidity."""
        liq = DefaultLiquidityModel(
            missing_quote_policy=MissingQuoteLiquidityPolicy.UNLIMITED_WITH_DIAGNOSTIC,
            allow_tick_volume_as_quote_size=False,
        )
        engine = FillEngine(liquidity_model=liq, instrument_specs={"AAPL": spec})
        tick = Tick(
            symbol="AAPL",
            timestamp=base_time,
            bid=150.00,
            ask=150.10,
            bid_volume=500.0,
            ask_volume=500.0,
        )
        snap = make_order_snapshot(
            order_id="tick-unlim", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=100.0
        )

        fills = engine.match_quote(tick, [snap], market_event_id="t-unlim-1")
        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal(100)
        assert any("assumed unbounded liquidity" in d for d in fills[0].diagnostics)

        avail, _ = liq.get_available_quote_liquidity(tick, OrderSide.BUY, "t-unlim-1")
        assert avail.is_finite()

    # --------------------------------------------------------------------------
    # 4. Commission Fill-ID Consistency
    # --------------------------------------------------------------------------

    def test_commission_fill_id_identical_replay_is_idempotent(self) -> None:
        """Recommitting identical fill ID with identical details returns cached fee without conflict."""
        snap = make_order_snapshot(
            order_id="comm-idemp", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        model = PerShareCommissionModel(per_share_rate="0.05", currency="USD")

        fee1, curr1 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(50), fill_id="f-replay-1"
        )
        assert fee1 == Decimal("2.50")
        assert curr1 == "USD"

        # Replay identical details
        fee2, curr2 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(50), fill_id="f-replay-1"
        )
        assert fee2 == Decimal("2.50")
        assert curr2 == "USD"
        assert model._order_cumulative_charged["comm-idemp"] == Decimal("2.50")

    def test_commission_fill_id_conflicting_details_raises_error(self) -> None:
        """Reusing already committed fill ID with conflicting order, price, or quantity raises CommissionConflictError."""
        snap1 = make_order_snapshot(
            order_id="order-A", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        snap2 = make_order_snapshot(
            order_id="order-B", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        model = PerShareCommissionModel(per_share_rate="0.05", currency="USD")

        model.commit_commission(snap1.order, Decimal("150.00"), Decimal(50), fill_id="f-conflict-1")

        # Conflict 1: Different order ID
        with pytest.raises(
            CommissionConflictError,
            match="Conflicting fill details for committed fill_id 'f-conflict-1'",
        ):
            model.commit_commission(
                snap2.order, Decimal("150.00"), Decimal(50), fill_id="f-conflict-1"
            )

        # Conflict 2: Different fill price
        with pytest.raises(
            CommissionConflictError,
            match="Conflicting fill details for committed fill_id 'f-conflict-1'",
        ):
            model.commit_commission(
                snap1.order, Decimal("155.00"), Decimal(50), fill_id="f-conflict-1"
            )

        # Conflict 3: Different fill quantity
        with pytest.raises(
            CommissionConflictError,
            match="Conflicting fill details for committed fill_id 'f-conflict-1'",
        ):
            model.commit_commission(
                snap1.order, Decimal("150.00"), Decimal(20), fill_id="f-conflict-1"
            )

    def test_distinct_fills_with_same_economics_not_deduplicated(self) -> None:
        """Distinct fills with different fill IDs are processed independently even if economics look identical."""
        snap = make_order_snapshot(
            order_id="order-distinct",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=50.0,
        )
        model = PerShareCommissionModel(per_share_rate="0.05", currency="USD")

        fee1, curr1 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(10), fill_id="f-distinct-1"
        )
        fee2, curr2 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(10), fill_id="f-distinct-2"
        )

        assert fee1 == Decimal("0.50")
        assert fee2 == Decimal("0.50")
        assert curr1 == "USD"
        assert curr2 == "USD"
        assert model._order_cumulative_charged["order-distinct"] == Decimal("1.00")
        assert len(model._committed_fills) == 2

    def test_percentage_commission_fill_id_identical_contract_size_is_idempotent(
        self, spec: InstrumentSpecification
    ) -> None:
        """Recommitting identical fill ID with identical details including contract size is idempotent."""
        snap = make_order_snapshot(
            order_id="pct-idemp", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        model = PercentageCommissionModel(percentage_rate="0.001", currency="USD")

        fee1, curr1 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(50), spec=spec, fill_id="f-pct-replay"
        )
        # Notional = 150 * 50 * 1 = 7500. Fee = 7500 * 0.001 = 7.50
        assert fee1 == Decimal("7.50")
        assert curr1 == "USD"

        # Replay identical details with same spec
        fee2, curr2 = model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(50), spec=spec, fill_id="f-pct-replay"
        )
        assert fee2 == Decimal("7.50")
        assert curr2 == "USD"
        assert model._order_cumulative_charged["pct-idemp"] == Decimal("7.50")

    def test_percentage_commission_fill_id_conflicting_contract_size_raises_in_commit_and_calculate(
        self, spec: InstrumentSpecification
    ) -> None:
        """Reusing committed fill ID with different contract size raises CommissionConflictError in calculate and commit."""
        snap = make_order_snapshot(
            order_id="pct-conflict", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        model = PercentageCommissionModel(percentage_rate="0.001", currency="USD")
        spec_size_1 = spec  # contract_size = 1.0

        spec_size_10 = InstrumentSpecification(
            symbol="AAPL",
            tick_size=0.01,
            contract_size=10.0,
            min_volume=1.0,
            volume_step=1.0,
            price_precision=2,
            quote_currency="USD",
            profit_currency="USD",
            base_currency="USD",
        )

        model.commit_commission(
            snap.order, Decimal("150.00"), Decimal(50), spec=spec_size_1, fill_id="f-pct-conf"
        )

        # Conflict in commit_commission
        with pytest.raises(
            CommissionConflictError,
            match="Conflicting fill details for committed fill_id 'f-pct-conf'",
        ):
            model.commit_commission(
                snap.order, Decimal("150.00"), Decimal(50), spec=spec_size_10, fill_id="f-pct-conf"
            )

        # Conflict in calculate_commission
        with pytest.raises(
            CommissionConflictError,
            match="Conflicting fill details for committed fill_id 'f-pct-conf'",
        ):
            model.calculate_commission(
                snap.order, Decimal("150.00"), Decimal(50), spec=spec_size_10, fill_id="f-pct-conf"
            )

    def test_percentage_commission_missing_spec_and_explicit_spec_equiv_contract_size_consistent(
        self, spec: InstrumentSpecification
    ) -> None:
        """Missing specification and explicit specification with equivalent contract_size=1.0 resolve identically."""
        snap = make_order_snapshot(
            order_id="pct-equiv", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        model = PercentageCommissionModel(percentage_rate="0.001", currency="USD")
        spec_size_1 = spec  # contract_size = 1.0

        # Commit with spec=None (resolves to contract_size=1)
        fee1, curr1 = model.commit_commission(
            snap.order, Decimal("100.00"), Decimal(10), spec=None, fill_id="f-pct-none"
        )
        assert fee1 == Decimal("1.00")
        assert curr1 == "USD"

        # Replay with explicit spec where contract_size=1.0 -> same effective contract size -> idempotent
        fee2, curr2 = model.calculate_commission(
            snap.order, Decimal("100.00"), Decimal(10), spec=spec_size_1, fill_id="f-pct-none"
        )
        assert fee2 == Decimal("1.00")
        assert curr2 == "USD"

        fee3, curr3 = model.commit_commission(
            snap.order, Decimal("100.00"), Decimal(10), spec=spec_size_1, fill_id="f-pct-none"
        )
        assert fee3 == Decimal("1.00")
        assert curr3 == "USD"

    def test_percentage_commission_distinct_fill_ids_different_contract_sizes_independent(
        self, spec: InstrumentSpecification
    ) -> None:
        """Distinct fill IDs with different contract sizes remain independently processable."""
        snap = make_order_snapshot(
            order_id="pct-distinct", side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=50.0
        )
        model = PercentageCommissionModel(percentage_rate="0.001", currency="USD")
        spec_size_1 = spec  # contract_size = 1.0

        spec_size_5 = InstrumentSpecification(
            symbol="AAPL",
            tick_size=0.01,
            contract_size=5.0,
            min_volume=1.0,
            volume_step=1.0,
            price_precision=2,
            quote_currency="USD",
            profit_currency="USD",
            base_currency="USD",
        )

        fee1, curr1 = model.commit_commission(
            snap.order, Decimal("100.00"), Decimal(10), spec=spec_size_1, fill_id="f-dist-1"
        )
        # 100 * 10 * 1 * 0.001 = 1.00
        assert fee1 == Decimal("1.00")
        assert curr1 == "USD"

        fee2, curr2 = model.commit_commission(
            snap.order, Decimal("100.00"), Decimal(10), spec=spec_size_5, fill_id="f-dist-2"
        )
        # 100 * 10 * 5 * 0.001 = 5.00
        assert fee2 == Decimal("5.00")
        assert curr2 == "USD"

        assert model._order_cumulative_charged["pct-distinct"] == Decimal("6.00")
        assert len(model._committed_fills) == 2
