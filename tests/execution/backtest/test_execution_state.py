"""Comprehensive unit test suite for OrderExecutionState and FillRecord."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, Order
from hybrid_agent.execution.backtest.state import (
    ExecutionStatus,
    FillRecord,
    OrderExecutionState,
)


@pytest.fixture
def sample_spec() -> InstrumentSpecification:
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
def fractional_spec() -> InstrumentSpecification:
    return InstrumentSpecification(
        symbol="BTCUSD",
        tick_size=0.1,
        contract_size=1.0,
        min_volume=0.001,
        volume_step=0.001,
        price_precision=1,
        profit_currency="USD",
        quote_currency="USD",
        base_currency="BTC",
        valuation_model=ValuationModel.LINEAR,
    )


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_order(base_time: datetime) -> Order:
    order = Order(
        id="ord-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        limit_price=150.00,
        created_at=base_time,
        updated_at=base_time,
    )
    order.transition_to(OrderStatus.APPROVED, timestamp=base_time)
    order.transition_to(OrderStatus.SUBMITTED, timestamp=base_time)
    return order


# ==============================================================================
# 1. FillRecord Invariants & Construction Tests
# ==============================================================================


class TestFillRecord:
    def test_valid_fill_record(self, base_time: datetime) -> None:
        fill = FillRecord(
            fill_id="f-1",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal("150.25"),
            fill_quantity=Decimal(50),
            fee=Decimal("1.50"),
            commission_currency="USD",
            slippage=Decimal("0.05"),
            timestamp=base_time,
        )
        assert fill.fill_id == "f-1"
        assert fill.fill_price == Decimal("150.25")
        assert fill.fill_quantity == Decimal(50)
        assert fill.fee == Decimal("1.50")
        assert fill.commission_currency == "USD"
        assert fill.slippage == Decimal("0.05")
        assert fill.timestamp == base_time

    def test_immutable_fill_record(self, base_time: datetime) -> None:
        fill = FillRecord(
            fill_id="f-1",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time,
        )
        with pytest.raises((AttributeError, TypeError)):
            fill.fill_quantity = Decimal(20)  # type: ignore

    def test_zero_fee_without_commission_currency_allowed(self, base_time: datetime) -> None:
        fill = FillRecord(
            fill_id="f-1",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            fee=Decimal(0),
            commission_currency=None,
            timestamp=base_time,
        )
        assert fill.fee == Decimal(0)
        assert fill.commission_currency is None

    def test_positive_fee_requires_commission_currency(self, base_time: datetime) -> None:
        with pytest.raises(ValueError, match="commission_currency must be provided"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                fee=Decimal("1.00"),
                commission_currency=None,
                timestamp=base_time,
            )

        with pytest.raises(ValueError, match="commission_currency must be provided"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                fee=Decimal("1.00"),
                commission_currency="   ",
                timestamp=base_time,
            )

    def test_negative_or_zero_price_rejected(self, base_time: datetime) -> None:
        with pytest.raises(ValueError, match="fill_price must be strictly positive"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(0),
                fill_quantity=Decimal(10),
                timestamp=base_time,
            )
        with pytest.raises(ValueError, match="fill_price must be strictly positive"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(-10),
                fill_quantity=Decimal(10),
                timestamp=base_time,
            )

    def test_negative_or_zero_quantity_rejected(self, base_time: datetime) -> None:
        with pytest.raises(ValueError, match="fill_quantity must be strictly positive"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(0),
                timestamp=base_time,
            )

    def test_negative_fee_rejected(self, base_time: datetime) -> None:
        with pytest.raises(ValueError, match="fee must be non-negative"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                fee=Decimal("-0.50"),
                commission_currency="USD",
                timestamp=base_time,
            )

    def test_non_finite_decimal_rejected(self, base_time: datetime) -> None:
        with pytest.raises(ValueError, match="finite Decimal"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal("NaN"),
                fill_quantity=Decimal(10),
                timestamp=base_time,
            )

    def test_naive_timestamp_rejected(self) -> None:
        naive_time = datetime(2026, 1, 1, 10, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                timestamp=naive_time,
            )

    def test_empty_identifiers_rejected(self, base_time: datetime) -> None:
        with pytest.raises(ValueError, match="fill_id must be a non-empty string"):
            FillRecord(
                fill_id="",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                timestamp=base_time,
            )
        with pytest.raises(ValueError, match="order_id must be a non-empty string"):
            FillRecord(
                fill_id="f-1",
                order_id="",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                timestamp=base_time,
            )
        with pytest.raises(ValueError, match="symbol must be a non-empty string"):
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                timestamp=base_time,
            )


# ==============================================================================
# 2. OrderExecutionState Initialization Tests
# ==============================================================================


class TestOrderExecutionStateInit:
    def test_successful_initialization(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        assert state.order == sample_order
        assert state.order is not sample_order
        # Verify defensive copy isolation
        order_copy = state.order
        order_copy.quantity = 999.0
        assert state.order.quantity == 100.0
        assert state.original_quantity == Decimal("100.0")
        assert state.submission_time == base_time
        assert state.original_quantity == Decimal("100.0")
        assert state.filled_quantity == Decimal(0)
        assert state.unfilled_quantity == Decimal("100.0")
        assert state.leaves_quantity == Decimal("100.0")
        assert state.average_fill_price == Decimal(0)
        assert state.cumulative_fees == {}
        assert state.status == ExecutionStatus.ACTIVE
        assert state.is_active is True
        assert state.is_partially_filled is False
        assert state.is_filled is False
        assert state.is_cancelled is False
        assert state.is_terminal is False
        assert state.fills == ()
        assert state.last_update_time == base_time

    def test_initialization_with_approved_order(self, base_time: datetime) -> None:
        order = Order(
            id="ord-appr",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            created_at=base_time,
            updated_at=base_time,
        )
        order.transition_to(OrderStatus.APPROVED, timestamp=base_time)
        state = OrderExecutionState(order, submission_time=base_time)
        assert state.status == ExecutionStatus.ACTIVE

    def test_unapproved_order_rejected(self, base_time: datetime) -> None:
        order = Order(
            id="ord-pending",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.0,
            created_at=base_time,
            updated_at=base_time,
        )
        assert order.status == OrderStatus.PENDING_RISK
        with pytest.raises(ValueError, match="APPROVED or SUBMITTED"):
            OrderExecutionState(order, submission_time=base_time)

    def test_naive_submission_time_rejected(self, sample_order: Order) -> None:
        naive = datetime(2026, 1, 1, 10, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            OrderExecutionState(sample_order, submission_time=naive)

    def test_submission_time_prior_to_created_at_rejected(
        self, sample_order: Order, base_time: datetime
    ) -> None:
        early = base_time - timedelta(seconds=1)
        with pytest.raises(ValueError, match="cannot be earlier than order.created_at"):
            OrderExecutionState(sample_order, submission_time=early)

    def test_instrument_symbol_mismatch_rejected(
        self, sample_order: Order, base_time: datetime
    ) -> None:
        other_spec = InstrumentSpecification(
            symbol="MSFT",
            tick_size=0.01,
            contract_size=1.0,
            min_volume=1.0,
            volume_step=1.0,
            price_precision=2,
            valuation_model=ValuationModel.LINEAR,
        )
        with pytest.raises(ValueError, match="does not match order symbol"):
            OrderExecutionState(sample_order, submission_time=base_time, instrument_spec=other_spec)

    def test_volume_step_mismatch_rejected(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        order = Order(
            id="ord-bad-step",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10.5,  # volume_step is 1.0
            created_at=base_time,
            updated_at=base_time,
        )
        order.transition_to(OrderStatus.APPROVED, timestamp=base_time)
        with pytest.raises(OrderValidationError, match="does not align with volume_step"):
            OrderExecutionState(order, submission_time=base_time, instrument_spec=sample_spec)

    def test_min_volume_violation_rejected(self, base_time: datetime) -> None:
        spec = InstrumentSpecification(
            symbol="AAPL",
            tick_size=0.01,
            contract_size=1.0,
            min_volume=10.0,
            volume_step=1.0,
            price_precision=2,
            valuation_model=ValuationModel.LINEAR,
        )
        order = Order(
            id="ord-small",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=5.0,  # Below min_volume 10.0
            created_at=base_time,
            updated_at=base_time,
        )
        order.transition_to(OrderStatus.APPROVED, timestamp=base_time)
        with pytest.raises(OrderValidationError, match="is below minimum allowed volume"):
            OrderExecutionState(order, submission_time=base_time, instrument_spec=spec)


# ==============================================================================
# 3. Fill Recording & Atomic State Updates Tests
# ==============================================================================


class TestRecordFill:
    def test_single_complete_fill(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t1 = base_time + timedelta(seconds=1)
        fill = FillRecord(
            fill_id="fill-1",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal("150.00"),
            fill_quantity=Decimal(100),
            fee=Decimal("1.00"),
            commission_currency="USD",
            timestamp=t1,
        )
        state.record_fill(fill)

        assert state.status == ExecutionStatus.FILLED
        assert state.is_filled is True
        assert state.is_terminal is True
        assert state.filled_quantity == Decimal(100)
        assert state.unfilled_quantity == Decimal(0)
        assert state.leaves_quantity == Decimal(0)
        assert state.average_fill_price == Decimal("150.00")
        assert state.cumulative_fees == {"USD": Decimal("1.00")}
        assert len(state.fills) == 1
        assert state.fills[0] == fill
        assert state.last_update_time == t1

    def test_single_partial_fill(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t1 = base_time + timedelta(seconds=1)
        fill = FillRecord(
            fill_id="fill-1",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal("150.00"),
            fill_quantity=Decimal(40),
            fee=Decimal("0.50"),
            commission_currency="USD",
            timestamp=t1,
        )
        state.record_fill(fill)

        assert state.status == ExecutionStatus.PARTIALLY_FILLED
        assert state.is_partially_filled is True
        assert state.is_terminal is False
        assert state.filled_quantity == Decimal(40)
        assert state.unfilled_quantity == Decimal(60)
        assert state.leaves_quantity == Decimal(60)
        assert state.average_fill_price == Decimal("150.00")
        assert state.cumulative_fees == {"USD": Decimal("0.50")}

    def test_multiple_partial_fills_and_weighted_average_price(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t1 = base_time + timedelta(seconds=1)
        t2 = base_time + timedelta(seconds=2)
        t3 = base_time + timedelta(seconds=3)

        # Fill 1: 30 shares @ 150.00 with $1 USD fee
        state.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal("150.00"),
                fill_quantity=Decimal(30),
                fee=Decimal("1.00"),
                commission_currency="USD",
                timestamp=t1,
            )
        )
        assert state.filled_quantity == Decimal(30)
        assert state.leaves_quantity == Decimal(70)
        assert state.average_fill_price == Decimal("150.00")

        # Fill 2: 30 shares @ 152.00 with $0.80 EUR fee
        state.record_fill(
            FillRecord(
                fill_id="f2",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal("152.00"),
                fill_quantity=Decimal(30),
                fee=Decimal("0.80"),
                commission_currency="EUR",
                timestamp=t2,
            )
        )
        assert state.filled_quantity == Decimal(60)
        assert state.leaves_quantity == Decimal(40)
        # Avg price = (30*150 + 30*152)/60 = 151.00
        assert state.average_fill_price == Decimal("151.00")
        assert state.cumulative_fees == {"USD": Decimal("1.00"), "EUR": Decimal("0.80")}

        # Fill 3: final 40 shares @ 155.00 with $1.50 USD fee
        state.record_fill(
            FillRecord(
                fill_id="f3",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal("155.00"),
                fill_quantity=Decimal(40),
                fee=Decimal("1.50"),
                commission_currency="USD",
                timestamp=t3,
            )
        )
        assert state.status == ExecutionStatus.FILLED
        assert state.filled_quantity == Decimal(100)
        assert state.leaves_quantity == Decimal(0)
        # Avg price = (60*151 + 40*155)/100 = (9060 + 6200)/100 = 152.60
        assert state.average_fill_price == Decimal("152.60")
        assert state.cumulative_fees == {"USD": Decimal("2.50"), "EUR": Decimal("0.80")}
        assert len(state.fills) == 3

    def test_duplicate_fill_id_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t1 = base_time + timedelta(seconds=1)
        fill = FillRecord(
            fill_id="dup-fill",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(20),
            timestamp=t1,
        )
        state.record_fill(fill)
        with pytest.raises(ValueError, match="Duplicate fill_id"):
            state.record_fill(fill)

    def test_mismatched_order_id_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        fill = FillRecord(
            fill_id="f-1",
            order_id="wrong-order",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(20),
            timestamp=base_time,
        )
        with pytest.raises(ValueError, match="does not match order id"):
            state.record_fill(fill)

    def test_mismatched_symbol_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        fill = FillRecord(
            fill_id="f-1",
            order_id="ord-001",
            symbol="MSFT",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(20),
            timestamp=base_time,
        )
        with pytest.raises(ValueError, match="does not match order symbol"):
            state.record_fill(fill)

    def test_mismatched_side_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        fill = FillRecord(
            fill_id="f-1",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_price=Decimal(150),
            fill_quantity=Decimal(20),
            timestamp=base_time,
        )
        with pytest.raises(ValueError, match="does not match order side"):
            state.record_fill(fill)

    def test_fill_exceeding_leaves_quantity_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        # Order is 100 shares. Fill 105 shares.
        fill = FillRecord(
            fill_id="f-oversize",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(105),
            timestamp=base_time,
        )
        with pytest.raises(ValueError, match="exceeds executable leaves quantity"):
            state.record_fill(fill)

    def test_backward_timestamp_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t_early = base_time - timedelta(seconds=1)
        fill_early = FillRecord(
            fill_id="f-early",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=t_early,
        )
        with pytest.raises(ValueError, match="cannot be earlier than submission_time"):
            state.record_fill(fill_early)

        # Record valid fill at t=5
        t5 = base_time + timedelta(seconds=5)
        state.record_fill(
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(10),
                timestamp=t5,
            )
        )
        # Attempt fill at t=4 (earlier than last update)
        t4 = base_time + timedelta(seconds=4)
        fill_back = FillRecord(
            fill_id="f-back",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=t4,
        )
        with pytest.raises(ValueError, match="cannot be earlier than previous update"):
            state.record_fill(fill_back)

    def test_fill_after_full_execution_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        state.record_fill(
            FillRecord(
                fill_id="f-all",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(100),
                timestamp=base_time,
            )
        )
        assert state.is_filled is True

        with pytest.raises(ValueError, match="terminal execution state 'FILLED'"):
            state.record_fill(
                FillRecord(
                    fill_id="f-extra",
                    order_id="ord-001",
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    fill_price=Decimal(150),
                    fill_quantity=Decimal(10),
                    timestamp=base_time + timedelta(seconds=1),
                )
            )

    def test_fill_after_cancellation_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        state.cancel_remaining(timestamp=base_time, reason="Market closed")
        assert state.is_cancelled is True

        with pytest.raises(ValueError, match="terminal execution state 'CANCELLED'"):
            state.record_fill(
                FillRecord(
                    fill_id="f-late",
                    order_id="ord-001",
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    fill_price=Decimal(150),
                    fill_quantity=Decimal(10),
                    timestamp=base_time + timedelta(seconds=1),
                )
            )

    def test_failed_fill_leaves_state_completely_unchanged(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        """Verifies transactional rollback: state is unaltered when record_fill validation fails."""
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t1 = base_time + timedelta(seconds=1)

        # Apply initial partial fill
        state.record_fill(
            FillRecord(
                fill_id="f-valid",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(30),
                fee=Decimal("1.00"),
                commission_currency="USD",
                timestamp=t1,
            )
        )

        # Snapshot state attributes
        snap_filled = state.filled_quantity
        snap_leaves = state.leaves_quantity
        snap_avg_px = state.average_fill_price
        snap_fees = deepcopy(state.cumulative_fees)
        snap_status = state.status
        snap_fills_count = len(state.fills)
        snap_last_update = state.last_update_time

        # Attempt invalid fill (quantity exceeds remaining 70)
        with pytest.raises(ValueError):
            state.record_fill(
                FillRecord(
                    fill_id="f-invalid",
                    order_id="ord-001",
                    symbol="AAPL",
                    side=OrderSide.BUY,
                    fill_price=Decimal(150),
                    fill_quantity=Decimal(80),  # Only 70 remaining!
                    timestamp=t1 + timedelta(seconds=1),
                )
            )

        # Verify every attribute is bit-for-bit unchanged
        assert state.filled_quantity == snap_filled
        assert state.leaves_quantity == snap_leaves
        assert state.average_fill_price == snap_avg_px
        assert state.cumulative_fees == snap_fees
        assert state.status == snap_status
        assert len(state.fills) == snap_fills_count
        assert state.last_update_time == snap_last_update


# ==============================================================================
# 4. Cancellation Semantics Tests
# ==============================================================================


class TestCancellation:
    def test_cancellation_before_any_fill(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t_cancel = base_time + timedelta(seconds=10)
        state.cancel_remaining(timestamp=t_cancel, reason="Strategy timeout")

        assert state.status == ExecutionStatus.CANCELLED
        assert state.is_cancelled is True
        assert state.is_terminal is True
        assert state.filled_quantity == Decimal(0)
        assert state.unfilled_quantity == Decimal("100.0")
        assert state.leaves_quantity == Decimal(0)
        assert state.cancelled_quantity == Decimal("100.0")
        assert state.cancellation_timestamp == t_cancel
        assert state.cancellation_reason == "Strategy timeout"
        assert state.last_update_time == t_cancel

    def test_cancellation_after_partial_fill(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t1 = base_time + timedelta(seconds=1)
        state.record_fill(
            FillRecord(
                fill_id="f-part",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal("150.50"),
                fill_quantity=Decimal(40),
                fee=Decimal("1.00"),
                commission_currency="USD",
                timestamp=t1,
            )
        )

        t_cancel = base_time + timedelta(seconds=5)
        state.cancel_remaining(timestamp=t_cancel, reason="Risk circuit breaker")

        assert state.status == ExecutionStatus.CANCELLED
        assert state.is_cancelled is True
        assert state.is_terminal is True
        # Executed fill is preserved
        assert state.filled_quantity == Decimal(40)
        assert state.average_fill_price == Decimal("150.50")
        assert len(state.fills) == 1
        # Remainder is cancelled
        assert state.unfilled_quantity == Decimal("60.0")
        assert state.leaves_quantity == Decimal(0)
        assert state.cancelled_quantity == Decimal("60.0")
        assert state.cancellation_timestamp == t_cancel
        assert state.cancellation_reason == "Risk circuit breaker"

    def test_empty_cancellation_reason_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        with pytest.raises(ValueError, match="non-empty string"):
            state.cancel_remaining(timestamp=base_time, reason="")
        with pytest.raises(ValueError, match="non-empty string"):
            state.cancel_remaining(timestamp=base_time, reason="   ")

    def test_backward_cancellation_timestamp_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        t_early = base_time - timedelta(seconds=1)
        with pytest.raises(ValueError, match="cannot be earlier than submission_time"):
            state.cancel_remaining(timestamp=t_early, reason="Cancelled")

    def test_repeated_cancellation_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        state.cancel_remaining(timestamp=base_time, reason="First cancel")
        with pytest.raises(
            ValueError, match="Cannot cancel order with execution status 'CANCELLED'"
        ):
            state.cancel_remaining(
                timestamp=base_time + timedelta(seconds=1), reason="Second cancel"
            )

    def test_cancellation_after_full_fill_rejected(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        state.record_fill(
            FillRecord(
                fill_id="f-all",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(100),
                timestamp=base_time,
            )
        )
        assert state.is_filled is True
        with pytest.raises(ValueError, match="Cannot cancel order with execution status 'FILLED'"):
            state.cancel_remaining(
                timestamp=base_time + timedelta(seconds=1), reason="Cancel after fill"
            )


# ==============================================================================
# 5. Precision & Invariants Tests
# ==============================================================================


class TestPrecisionAndInvariants:
    def test_fractional_crypto_quantities(
        self, fractional_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        order = Order(
            id="ord-btc-1",
            symbol="BTCUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.025,
            limit_price=60000.0,
            created_at=base_time,
            updated_at=base_time,
        )
        order.transition_to(OrderStatus.APPROVED, timestamp=base_time)
        state = OrderExecutionState(
            order, submission_time=base_time, instrument_spec=fractional_spec
        )

        assert state.original_quantity == Decimal("0.025")
        assert state.leaves_quantity == Decimal("0.025")

        # Fill 0.010 BTC @ $60,000
        state.record_fill(
            FillRecord(
                fill_id="f-btc-1",
                order_id="ord-btc-1",
                symbol="BTCUSD",
                side=OrderSide.BUY,
                fill_price=Decimal("60000.0"),
                fill_quantity=Decimal("0.010"),
                fee=Decimal("0.60"),
                commission_currency="USD",
                timestamp=base_time + timedelta(seconds=1),
            )
        )
        assert state.filled_quantity == Decimal("0.010")
        assert state.leaves_quantity == Decimal("0.015")
        assert state.unfilled_quantity == Decimal("0.015")

        # Fill 0.015 BTC @ $61,000
        state.record_fill(
            FillRecord(
                fill_id="f-btc-2",
                order_id="ord-btc-1",
                symbol="BTCUSD",
                side=OrderSide.BUY,
                fill_price=Decimal("61000.0"),
                fill_quantity=Decimal("0.015"),
                fee=Decimal("0.90"),
                commission_currency="USD",
                timestamp=base_time + timedelta(seconds=2),
            )
        )
        assert state.status == ExecutionStatus.FILLED
        assert state.filled_quantity == Decimal("0.025")
        assert state.leaves_quantity == Decimal(0)
        # Weighted avg = (0.010 * 60000 + 0.015 * 61000) / 0.025 = (600 + 915) / 0.025 = 1515 / 0.025 = 60600.0
        assert state.average_fill_price == Decimal("60600.0")

    def test_original_equals_filled_plus_unfilled_invariant(
        self, sample_order: Order, sample_spec: InstrumentSpecification, base_time: datetime
    ) -> None:
        state = OrderExecutionState(
            sample_order, submission_time=base_time, instrument_spec=sample_spec
        )
        assert state.original_quantity == state.filled_quantity + state.unfilled_quantity

        # Partial fill
        state.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(35),
                timestamp=base_time + timedelta(seconds=1),
            )
        )
        assert state.original_quantity == state.filled_quantity + state.unfilled_quantity

        # Cancel remainder
        state.cancel_remaining(timestamp=base_time + timedelta(seconds=2), reason="User abort")
        assert state.original_quantity == state.filled_quantity + state.unfilled_quantity
        assert state.cancelled_quantity == state.unfilled_quantity
        assert state.leaves_quantity == Decimal(0)
