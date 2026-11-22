"""Comprehensive unit test suite for OrderManager core lifecycle and registry."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    ValuationModel,
)
from hybrid_agent.core.models import InstrumentSpecification, Order
from hybrid_agent.execution.backtest.events import OrderCancelledEvent, OrderSubmittedEvent
from hybrid_agent.execution.backtest.order_manager import (
    DuplicateFillError,
    DuplicateOrderError,
    InvalidOrderStateError,
    OrderManager,
    OrderNotFoundError,
)
from hybrid_agent.execution.backtest.state import (
    ExecutionStatus,
    FillRecord,
)


@pytest.fixture
def base_time() -> datetime:
    """Standard base simulation timestamp in UTC."""
    return datetime(2026, 1, 15, 9, 30, tzinfo=UTC)


@pytest.fixture
def sample_spec() -> InstrumentSpecification:
    """Standard equity instrument specification."""
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
def sample_approved_order(base_time: datetime) -> Order:
    """Sample order in APPROVED status."""
    return Order(
        id="ord-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        limit_price=150.0,
        status=OrderStatus.APPROVED,
        created_at=base_time - timedelta(seconds=5),
        updated_at=base_time - timedelta(seconds=5),
    )


@pytest.fixture
def sample_submitted_order(base_time: datetime) -> Order:
    """Sample order in SUBMITTED status."""
    return Order(
        id="ord-002",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=50.0,
        limit_price=150.0,
        status=OrderStatus.SUBMITTED,
        created_at=base_time - timedelta(seconds=5),
        updated_at=base_time - timedelta(seconds=5),
    )


# ==============================================================================
# 1. Registration Tests
# ==============================================================================


class TestOrderRegistration:
    def test_register_approved_order(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )

        state = manager.register_order(event, instrument_spec=sample_spec)

        assert state.order.id == "ord-001"
        assert state.order.status == OrderStatus.SUBMITTED
        assert state.status == ExecutionStatus.ACTIVE
        assert state.leaves_quantity == Decimal(100)
        assert manager.has_order("ord-001")
        assert manager.active_order_count == 1
        assert manager.total_order_count == 1

    def test_register_submitted_order(
        self,
        base_time: datetime,
        sample_submitted_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_submitted_order,
            submission_time=base_time,
        )

        state = manager.register_order(event, instrument_spec=sample_spec)

        assert state.order.id == "ord-002"
        assert state.order.status == OrderStatus.SUBMITTED
        assert state.status == ExecutionStatus.ACTIVE
        assert state.leaves_quantity == Decimal(50)

    def test_reject_pending_risk_order(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        manager = OrderManager()
        order = Order(
            id="ord-pending",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
            status=OrderStatus.PENDING_RISK,
            created_at=base_time,
            updated_at=base_time,
        )
        event = OrderSubmittedEvent(timestamp=base_time, order=order, submission_time=base_time)

        with pytest.raises(InvalidOrderStateError, match="APPROVED or SUBMITTED"):
            manager.register_order(event, instrument_spec=sample_spec)

    def test_reject_rejected_order(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        manager = OrderManager()
        order = Order(
            id="ord-rej",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
            status=OrderStatus.REJECTED,
            created_at=base_time,
            updated_at=base_time,
        )
        event = OrderSubmittedEvent(timestamp=base_time, order=order, submission_time=base_time)

        with pytest.raises(InvalidOrderStateError, match="APPROVED or SUBMITTED"):
            manager.register_order(event, instrument_spec=sample_spec)

    def test_reject_filled_or_cancelled_order(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        manager = OrderManager()
        order_filled = Order(
            id="ord-filled",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
            status=OrderStatus.FILLED,
            created_at=base_time,
            updated_at=base_time,
        )
        event_filled = OrderSubmittedEvent(
            timestamp=base_time, order=order_filled, submission_time=base_time
        )
        with pytest.raises(InvalidOrderStateError, match="APPROVED or SUBMITTED"):
            manager.register_order(event_filled, instrument_spec=sample_spec)

        order_cancelled = Order(
            id="ord-cancelled",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
            status=OrderStatus.CANCELLED,
            created_at=base_time,
            updated_at=base_time,
        )
        event_cancelled = OrderSubmittedEvent(
            timestamp=base_time, order=order_cancelled, submission_time=base_time
        )
        with pytest.raises(InvalidOrderStateError, match="APPROVED or SUBMITTED"):
            manager.register_order(event_cancelled, instrument_spec=sample_spec)

    def test_reject_mismatched_instrument_specification(
        self, base_time: datetime, sample_approved_order: Order
    ) -> None:
        manager = OrderManager()
        wrong_spec = InstrumentSpecification(
            symbol="MSFT",
            tick_size=0.01,
            contract_size=1.0,
            min_volume=1.0,
            volume_step=1.0,
            profit_currency="USD",
            quote_currency="USD",
            base_currency="USD",
        )
        event = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )

        with pytest.raises(ValueError, match="does not match order symbol"):
            manager.register_order(event, instrument_spec=wrong_spec)

    def test_reject_conflicting_instrument_specification_registration(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        manager.register_instrument(sample_spec)

        conflicting_spec = InstrumentSpecification(
            symbol="AAPL",
            tick_size=0.05,
            contract_size=1.0,
            min_volume=1.0,
            volume_step=1.0,
            profit_currency="USD",
            quote_currency="USD",
            base_currency="USD",
        )
        event = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )

        with pytest.raises(ValueError, match="Conflicting instrument specification"):
            manager.register_order(event, instrument_spec=conflicting_spec)

    def test_reject_timestamp_mismatch(
        self, base_time: datetime, sample_approved_order: Order
    ) -> None:
        manager = OrderManager()
        t1 = base_time
        t2 = base_time + timedelta(seconds=1)

        event = OrderSubmittedEvent(
            timestamp=t1,
            order=sample_approved_order,
            submission_time=t2,
        )
        with pytest.raises(ValueError, match="must match event.submission_time"):
            manager.register_order(event)

    def test_original_input_order_is_not_mutated(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )

        state = manager.register_order(event, instrument_spec=sample_spec)

        # Manager's copy must be SUBMITTED
        assert state.order.status == OrderStatus.SUBMITTED
        # Original input order remains APPROVED
        assert sample_approved_order.status == OrderStatus.APPROVED

        # Subsequent mutation of input order must not alter manager's state
        sample_approved_order.quantity = 999.0
        assert state.original_quantity == Decimal(100)
        assert state.order.quantity == 100.0

    def test_duplicate_registration_idempotency(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event1 = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )
        state1 = manager.register_order(event1, instrument_spec=sample_spec)

        # Re-register identical order and submission timestamp
        event2 = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )
        state2 = manager.register_order(event2, instrument_spec=sample_spec)

        assert state1 is state2
        assert manager.active_order_count == 1
        assert manager.total_order_count == 1

    def test_reject_duplicate_order_id_with_conflicting_parameters(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event1 = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )
        manager.register_order(event1, instrument_spec=sample_spec)

        # Same order ID but different quantity
        conflicting_order = Order(
            id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=200.0,
            limit_price=150.0,
            status=OrderStatus.APPROVED,
            created_at=sample_approved_order.created_at,
            updated_at=sample_approved_order.updated_at,
        )
        event2 = OrderSubmittedEvent(
            timestamp=base_time,
            order=conflicting_order,
            submission_time=base_time,
        )

        with pytest.raises(DuplicateOrderError, match="conflicting order parameters"):
            manager.register_order(event2, instrument_spec=sample_spec)


# ==============================================================================
# 2. Retrieval Tests
# ==============================================================================


class TestOrderRetrieval:
    def test_get_execution_state_and_has_order(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        assert not manager.has_order("ord-001")
        with pytest.raises(OrderNotFoundError, match="not found in registry"):
            manager.get_execution_state("ord-001")

        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        registered = manager.register_order(event, instrument_spec=sample_spec)

        assert manager.has_order("ord-001")
        retrieved = manager.get_execution_state("ord-001")
        assert retrieved is registered

    def test_get_active_orders_and_all_orders_with_symbol_filter(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        manager = OrderManager()
        manager.register_instrument(sample_spec)

        spec_msft = InstrumentSpecification(
            symbol="MSFT",
            tick_size=0.01,
            contract_size=1.0,
            min_volume=1.0,
            volume_step=1.0,
            profit_currency="USD",
            quote_currency="USD",
            base_currency="USD",
        )
        manager.register_instrument(spec_msft)

        o1 = Order(
            id="aapl-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=10.0,
            limit_price=150.0,
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )
        o2 = Order(
            id="msft-1",
            symbol="MSFT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=20.0,
            limit_price=400.0,
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )
        o3 = Order(
            id="aapl-2",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=30.0,
            limit_price=151.0,
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )

        manager.register_order(
            OrderSubmittedEvent(timestamp=base_time, order=o1, submission_time=base_time)
        )
        manager.register_order(
            OrderSubmittedEvent(timestamp=base_time, order=o2, submission_time=base_time)
        )
        manager.register_order(
            OrderSubmittedEvent(timestamp=base_time, order=o3, submission_time=base_time)
        )

        assert manager.active_order_count == 3
        assert manager.total_order_count == 3

        # Filter active
        active_aapl = manager.get_active_orders(symbol="AAPL")
        assert len(active_aapl) == 2
        assert [s.order.id for s in active_aapl] == ["aapl-1", "aapl-2"]

        active_msft = manager.get_active_orders(symbol="MSFT")
        assert len(active_msft) == 1
        assert active_msft[0].order.id == "msft-1"

        all_orders = manager.get_all_orders()
        assert [s.order.id for s in all_orders] == ["aapl-1", "msft-1", "aapl-2"]

    def test_returned_collections_are_defensive_copies(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        active = manager.get_active_orders()
        active.clear()
        assert manager.active_order_count == 1
        assert len(manager.get_active_orders()) == 1

        all_orders = manager.get_all_orders()
        all_orders.clear()
        assert manager.total_order_count == 1
        assert len(manager.get_all_orders()) == 1


# ==============================================================================
# 3. Fill Recording Tests
# ==============================================================================


class TestOrderFillRecording:
    def test_single_partial_fill(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        t_fill = base_time + timedelta(seconds=1)
        fill = FillRecord(
            fill_id="fill-001",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal("150.25"),
            fill_quantity=Decimal(40),
            timestamp=t_fill,
            fee=Decimal("1.50"),
            commission_currency="USD",
        )

        state = manager.record_fill(fill)

        assert state.status == ExecutionStatus.PARTIALLY_FILLED
        assert state.order.status == OrderStatus.SUBMITTED
        assert state.filled_quantity == Decimal(40)
        assert state.leaves_quantity == Decimal(60)
        assert state.unfilled_quantity == Decimal(60)
        assert state.average_fill_price == Decimal("150.25")
        assert manager.active_order_count == 1
        assert manager.get_order_by_fill_id("fill-001") is state

    def test_multiple_partial_fills_and_full_fill_completion(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        # Fill 1: 40 shares @ 150.0
        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(40),
                timestamp=t1,
            )
        )

        # Fill 2: 30 shares @ 152.0
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-2",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(152),
                fill_quantity=Decimal(30),
                timestamp=t2,
            )
        )

        state = manager.get_execution_state("ord-001")
        assert state.status == ExecutionStatus.PARTIALLY_FILLED
        assert state.filled_quantity == Decimal(70)
        assert state.leaves_quantity == Decimal(30)
        # Avg price = (40*150 + 30*152) / 70 = (6000 + 4560) / 70 = 10560 / 70 = 150.857142857...
        expected_avg = Decimal(10560) / Decimal(70)
        assert state.average_fill_price == expected_avg
        assert manager.active_order_count == 1

        # Fill 3: final 30 shares @ 151.0 (completing order)
        t3 = base_time + timedelta(seconds=3)
        completed_state = manager.record_fill(
            FillRecord(
                fill_id="f-3",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(151),
                fill_quantity=Decimal(30),
                timestamp=t3,
            )
        )

        assert completed_state.status == ExecutionStatus.FILLED
        assert completed_state.order.status == OrderStatus.FILLED
        assert completed_state.order.updated_at == t3
        assert completed_state.filled_quantity == Decimal(100)
        assert completed_state.leaves_quantity == Decimal(0)
        assert manager.active_order_count == 0
        assert manager.total_order_count == 1
        assert len(manager.get_active_orders()) == 0
        assert len(manager.get_all_orders()) == 1

    def test_reject_duplicate_fill_id_on_same_order(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        fill1 = FillRecord(
            fill_id="f-dup",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(20),
            timestamp=base_time + timedelta(seconds=1),
        )
        manager.record_fill(fill1)

        fill2 = FillRecord(
            fill_id="f-dup",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(20),
            timestamp=base_time + timedelta(seconds=2),
        )

        with pytest.raises(DuplicateFillError, match="already been recorded"):
            manager.record_fill(fill2)

    def test_reject_duplicate_fill_id_across_different_orders(
        self,
        base_time: datetime,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        o1 = Order(
            id="ord-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )
        o2 = Order(
            id="ord-2",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=150.0,
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )
        manager.register_order(
            OrderSubmittedEvent(timestamp=base_time, order=o1, submission_time=base_time),
            instrument_spec=sample_spec,
        )
        manager.register_order(
            OrderSubmittedEvent(timestamp=base_time, order=o2, submission_time=base_time),
            instrument_spec=sample_spec,
        )

        fill1 = FillRecord(
            fill_id="global-fill-id",
            order_id="ord-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time + timedelta(seconds=1),
        )
        manager.record_fill(fill1)

        fill2 = FillRecord(
            fill_id="global-fill-id",
            order_id="ord-2",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time + timedelta(seconds=1),
        )

        with pytest.raises(DuplicateFillError, match="already been recorded for order 'ord-1'"):
            manager.record_fill(fill2)

    def test_reject_fill_for_unknown_order(self, base_time: datetime) -> None:
        manager = OrderManager()
        fill = FillRecord(
            fill_id="f-unknown",
            order_id="ord-999",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time,
        )

        with pytest.raises(OrderNotFoundError, match="not found in registry"):
            manager.record_fill(fill)

    def test_reject_fill_with_mismatched_symbol_or_side(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        # Mismatched symbol
        fill_wrong_symbol = FillRecord(
            fill_id="f-sym",
            order_id="ord-001",
            symbol="MSFT",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time + timedelta(seconds=1),
        )
        with pytest.raises(ValueError, match="does not match order symbol"):
            manager.record_fill(fill_wrong_symbol)

        # Mismatched side
        fill_wrong_side = FillRecord(
            fill_id="f-side",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time + timedelta(seconds=1),
        )
        with pytest.raises(ValueError, match="does not match order side"):
            manager.record_fill(fill_wrong_side)

    def test_reject_fill_after_order_completed(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        # Complete fill of 100
        manager.record_fill(
            FillRecord(
                fill_id="f-complete",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(100),
                timestamp=base_time + timedelta(seconds=1),
            )
        )

        # Subsequent fill attempt
        fill_after = FillRecord(
            fill_id="f-after",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            timestamp=base_time + timedelta(seconds=2),
        )
        with pytest.raises(InvalidOrderStateError, match="Cannot record fill on terminal order"):
            manager.record_fill(fill_after)

    def test_failed_fill_guarantees_atomicity(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        # Valid initial partial fill of 40
        manager.record_fill(
            FillRecord(
                fill_id="f-good",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(40),
                timestamp=base_time + timedelta(seconds=1),
            )
        )

        state_before = manager.get_execution_state("ord-001")
        assert state_before.filled_quantity == Decimal(40)
        assert state_before.leaves_quantity == Decimal(60)

        # Overfill attempt: remaining is 60, attempt 70
        bad_fill = FillRecord(
            fill_id="f-bad-overfill",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(70),
            timestamp=base_time + timedelta(seconds=2),
        )

        with pytest.raises(ValueError, match="exceeds executable leaves quantity"):
            manager.record_fill(bad_fill)

        # Verify manager indexes and execution state are completely untouched
        state_after = manager.get_execution_state("ord-001")
        assert state_after.filled_quantity == Decimal(40)
        assert state_after.leaves_quantity == Decimal(60)
        assert state_after.status == ExecutionStatus.PARTIALLY_FILLED
        assert manager.get_order_by_fill_id("f-bad-overfill") is None
        assert manager.active_order_count == 1


# ==============================================================================
# 4. Cancellation Tests
# ==============================================================================


class TestOrderCancellation:
    def test_cancel_order_before_any_fill(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        t_cancel = base_time + timedelta(seconds=10)
        cancelled_event = manager.cancel_order(
            order_id="ord-001",
            timestamp=t_cancel,
            reason="User cancelled resting order",
        )

        assert isinstance(cancelled_event, OrderCancelledEvent)
        assert cancelled_event.timestamp == t_cancel
        assert cancelled_event.order.id == "ord-001"
        assert cancelled_event.order.status == OrderStatus.CANCELLED
        assert cancelled_event.reason == "User cancelled resting order"

        state = manager.get_execution_state("ord-001")
        assert state.status == ExecutionStatus.CANCELLED
        assert state.order.status == OrderStatus.CANCELLED
        assert state.filled_quantity == Decimal(0)
        assert state.leaves_quantity == Decimal(0)
        assert state.cancelled_quantity == Decimal(100)
        assert state.unfilled_quantity == Decimal(100)
        assert state.cancellation_timestamp == t_cancel
        assert state.cancellation_reason == "User cancelled resting order"

        # Active vs total accounting
        assert manager.active_order_count == 0
        assert manager.total_order_count == 1
        assert len(manager.get_active_orders()) == 0
        assert len(manager.get_all_orders()) == 1

    def test_cancel_order_after_partial_fill(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        # 30 shares filled
        manager.record_fill(
            FillRecord(
                fill_id="f-partial",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal("150.50"),
                fill_quantity=Decimal(30),
                timestamp=base_time + timedelta(seconds=1),
            )
        )

        # Cancel remaining 70 shares
        t_cancel = base_time + timedelta(seconds=5)
        manager.cancel_order("ord-001", timestamp=t_cancel, reason="Partial remainder cancelled")

        state = manager.get_execution_state("ord-001")
        assert state.status == ExecutionStatus.CANCELLED
        assert state.order.status == OrderStatus.CANCELLED
        # Executed fill preserved
        assert state.filled_quantity == Decimal(30)
        assert len(state.fills) == 1
        assert state.average_fill_price == Decimal("150.50")
        # Leaves zeroed out
        assert state.leaves_quantity == Decimal(0)
        # Cancelled remainder recorded
        assert state.cancelled_quantity == Decimal(70)
        assert state.unfilled_quantity == Decimal(70)
        assert state.cancellation_timestamp == t_cancel

        assert manager.active_order_count == 0
        assert manager.total_order_count == 1

    def test_reject_cancel_for_unknown_order(self, base_time: datetime) -> None:
        manager = OrderManager()
        with pytest.raises(OrderNotFoundError, match="not found in registry"):
            manager.cancel_order("ord-missing", timestamp=base_time, reason="cancel")

    def test_reject_repeated_cancellation(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        manager.cancel_order(
            "ord-001", timestamp=base_time + timedelta(seconds=1), reason="First cancel"
        )

        with pytest.raises(InvalidOrderStateError, match="terminal execution status 'CANCELLED'"):
            manager.cancel_order(
                "ord-001", timestamp=base_time + timedelta(seconds=2), reason="Second cancel"
            )

    def test_reject_cancellation_after_full_fill(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        manager.record_fill(
            FillRecord(
                fill_id="f-all",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(100),
                timestamp=base_time + timedelta(seconds=1),
            )
        )

        with pytest.raises(InvalidOrderStateError, match="terminal execution status 'FILLED'"):
            manager.cancel_order(
                "ord-001", timestamp=base_time + timedelta(seconds=2), reason="Cancel after filled"
            )

    def test_reject_cancellation_with_empty_reason(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        with pytest.raises(ValueError, match="Cancellation reason must be a non-empty string"):
            manager.cancel_order(
                "ord-001", timestamp=base_time + timedelta(seconds=1), reason="   "
            )

    def test_reject_cancellation_with_backward_timestamp(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        earlier = base_time - timedelta(seconds=1)
        with pytest.raises(ValueError, match="cannot be earlier than submission_time"):
            manager.cancel_order("ord-001", timestamp=earlier, reason="time travel")


# ==============================================================================
# 5. Determinism & Lifecycle Rules Tests
# ==============================================================================


class TestOrderManagerDeterminism:
    def test_deterministic_active_order_sorting(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        manager = OrderManager()
        manager.register_instrument(sample_spec)

        # Register 3 orders at identical submission time
        for i in [3, 1, 2]:
            order = Order(
                id=f"ord-{i}",
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=10.0,
                limit_price=150.0,
                status=OrderStatus.APPROVED,
                created_at=base_time,
                updated_at=base_time,
            )
            manager.register_order(
                OrderSubmittedEvent(timestamp=base_time, order=order, submission_time=base_time)
            )

        active = manager.get_active_orders()
        # Insertion order was ord-3, ord-1, ord-2. Registration sequence breaks ties deterministically.
        assert [s.order.id for s in active] == ["ord-3", "ord-1", "ord-2"]

    def test_terminal_orders_never_reenter_active_registry(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        manager.register_order(event, instrument_spec=sample_spec)

        # Cancel order
        manager.cancel_order(
            "ord-001", timestamp=base_time + timedelta(seconds=1), reason="cancelled"
        )
        assert manager.active_order_count == 0

        # Attempt to re-register the same order idempotently
        reregistered = manager.register_order(event, instrument_spec=sample_spec)
        assert reregistered.status == ExecutionStatus.CANCELLED
        # It must NOT re-enter active orders!
        assert manager.active_order_count == 0
        assert len(manager.get_active_orders()) == 0
        assert manager.total_order_count == 1
