"""Comprehensive unit test suite for OrderManager core lifecycle, registry, and snapshot isolation."""

from dataclasses import FrozenInstanceError
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
    OrderExecutionSnapshot,
    OrderSnapshot,
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

        snapshot = manager.register_order(event, instrument_spec=sample_spec)

        assert isinstance(snapshot, OrderExecutionSnapshot)
        assert isinstance(snapshot.order, OrderSnapshot)
        assert snapshot.order.id == "ord-001"
        assert snapshot.order.status == OrderStatus.SUBMITTED
        assert snapshot.status == ExecutionStatus.ACTIVE
        assert snapshot.leaves_quantity == Decimal(100)
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

        snapshot = manager.register_order(event, instrument_spec=sample_spec)

        assert snapshot.order.id == "ord-002"
        assert snapshot.order.status == OrderStatus.SUBMITTED
        assert snapshot.status == ExecutionStatus.ACTIVE
        assert snapshot.leaves_quantity == Decimal(50)

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

    def test_reject_backward_lifecycle_timestamps(
        self, base_time: datetime, sample_approved_order: Order
    ) -> None:
        manager = OrderManager()
        # created_at in future relative to submission_time
        order_future_created = sample_approved_order.model_copy(deep=True)
        object.__setattr__(order_future_created, "created_at", base_time + timedelta(seconds=5))
        with pytest.raises(
            ValueError, match="order.created_at .* cannot be later than submission_time"
        ):
            manager.register_order(
                OrderSubmittedEvent(
                    timestamp=base_time,
                    order=order_future_created,
                    submission_time=base_time,
                )
            )

        # updated_at in future relative to submission_time
        order_future_updated = sample_approved_order.model_copy(deep=True)
        object.__setattr__(order_future_updated, "updated_at", base_time + timedelta(seconds=5))
        with pytest.raises(
            ValueError, match="order.updated_at .* cannot be later than submission_time"
        ):
            manager.register_order(
                OrderSubmittedEvent(
                    timestamp=base_time,
                    order=order_future_updated,
                    submission_time=base_time,
                )
            )

    def test_reject_whitespace_in_order_id(
        self, base_time: datetime, sample_approved_order: Order
    ) -> None:
        manager = OrderManager()
        order_space = sample_approved_order.model_copy(deep=True)
        object.__setattr__(order_space, "id", " ord-001 ")
        with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
            manager.register_order(
                OrderSubmittedEvent(
                    timestamp=base_time,
                    order=order_space,
                    submission_time=base_time,
                )
            )

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

        snapshot = manager.register_order(event, instrument_spec=sample_spec)

        # Manager's copy must be SUBMITTED
        assert snapshot.order.status == OrderStatus.SUBMITTED
        # Original input order remains APPROVED
        assert sample_approved_order.status == OrderStatus.APPROVED

        # Subsequent mutation of input order must not alter manager's state
        sample_approved_order.quantity = 999.0
        assert snapshot.original_quantity == Decimal(100)
        assert snapshot.order.quantity == 100.0

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
        snap1 = manager.register_order(event1, instrument_spec=sample_spec)

        # Re-register identical order and submission timestamp
        event2 = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )
        snap2 = manager.register_order(event2, instrument_spec=sample_spec)

        assert snap1 == snap2
        assert snap1.order.id == snap2.order.id
        assert manager.active_order_count == 1
        assert manager.total_order_count == 1

    def test_duplicate_delivery_with_invalid_status_rejected_before_idempotency(
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

        # Re-registering with invalid status CANCELLED must fail before duplicate handling
        order_invalid = sample_approved_order.model_copy(deep=True)
        object.__setattr__(order_invalid, "status", OrderStatus.CANCELLED)
        event_invalid = OrderSubmittedEvent(
            timestamp=base_time,
            order=order_invalid,
            submission_time=base_time,
        )
        with pytest.raises(InvalidOrderStateError, match="APPROVED or SUBMITTED"):
            manager.register_order(event_invalid, instrument_spec=sample_spec)

    def test_duplicate_order_id_with_different_creation_time_rejected(
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

        # Same order ID but different created_at
        diff_order = sample_approved_order.model_copy(deep=True)
        object.__setattr__(diff_order, "created_at", base_time - timedelta(seconds=20))
        event2 = OrderSubmittedEvent(
            timestamp=base_time,
            order=diff_order,
            submission_time=base_time,
        )
        with pytest.raises(DuplicateOrderError, match="conflicting order parameters"):
            manager.register_order(event2, instrument_spec=sample_spec)

    def test_late_instrument_specification_attachment(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        # Initial registration without instrument spec
        event1 = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )
        snap1 = manager.register_order(event1, instrument_spec=None)
        assert snap1.order.id == "ord-001"

        # Subsequent idempotent registration with spec attaches spec safely
        event2 = OrderSubmittedEvent(
            timestamp=base_time,
            order=sample_approved_order,
            submission_time=base_time,
        )
        snap2 = manager.register_order(event2, instrument_spec=sample_spec)
        assert snap2.order.id == "ord-001"

        # Conflicting late spec must be rejected
        conflicting_spec = InstrumentSpecification(
            symbol="AAPL",
            tick_size=0.10,
            contract_size=1.0,
            min_volume=1.0,
            volume_step=1.0,
            profit_currency="USD",
            quote_currency="USD",
            base_currency="USD",
        )
        with pytest.raises(DuplicateOrderError, match="conflicting instrument specification"):
            manager.register_order(event2, instrument_spec=conflicting_spec)

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
        assert retrieved == registered
        assert isinstance(retrieved, OrderExecutionSnapshot)

    def test_whitespace_order_id_lookup_rejected(
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

        with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
            manager.get_execution_state(" ord-001 ")

        assert not manager.has_order(" ord-001 ")

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

        snapshot = manager.record_fill(fill)

        assert isinstance(snapshot, OrderExecutionSnapshot)
        assert snapshot.status == ExecutionStatus.PARTIALLY_FILLED
        assert snapshot.order.status == OrderStatus.SUBMITTED
        assert snapshot.filled_quantity == Decimal(40)
        assert snapshot.leaves_quantity == Decimal(60)
        assert snapshot.unfilled_quantity == Decimal(60)
        assert snapshot.average_fill_price == Decimal("150.25")
        assert manager.active_order_count == 1
        assert manager.get_order_by_fill_id("fill-001") == snapshot

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

        snapshot = manager.get_execution_state("ord-001")
        assert snapshot.status == ExecutionStatus.PARTIALLY_FILLED
        assert snapshot.filled_quantity == Decimal(70)
        assert snapshot.leaves_quantity == Decimal(30)
        expected_avg = Decimal(10560) / Decimal(70)
        assert snapshot.average_fill_price == expected_avg
        assert manager.active_order_count == 1

        # Fill 3: final 30 shares @ 151.0 (completing order)
        t3 = base_time + timedelta(seconds=3)
        completed_snapshot = manager.record_fill(
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

        assert completed_snapshot.status == ExecutionStatus.FILLED
        assert completed_snapshot.order.status == OrderStatus.FILLED
        assert completed_snapshot.order.updated_at == t3
        assert completed_snapshot.filled_quantity == Decimal(100)
        assert completed_snapshot.leaves_quantity == Decimal(0)
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

        snap_before = manager.get_execution_state("ord-001")
        assert snap_before.filled_quantity == Decimal(40)
        assert snap_before.leaves_quantity == Decimal(60)

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

        snap_after = manager.get_execution_state("ord-001")
        assert snap_after.filled_quantity == Decimal(40)
        assert snap_after.leaves_quantity == Decimal(60)
        assert snap_after.status == ExecutionStatus.PARTIALLY_FILLED
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

        snapshot = manager.get_execution_state("ord-001")
        assert snapshot.status == ExecutionStatus.CANCELLED
        assert snapshot.order.status == OrderStatus.CANCELLED
        assert snapshot.filled_quantity == Decimal(0)
        assert snapshot.leaves_quantity == Decimal(0)
        assert snapshot.cancelled_quantity == Decimal(100)
        assert snapshot.unfilled_quantity == Decimal(100)
        assert snapshot.cancellation_timestamp == t_cancel
        assert snapshot.cancellation_reason == "User cancelled resting order"

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

        t_cancel = base_time + timedelta(seconds=5)
        manager.cancel_order("ord-001", timestamp=t_cancel, reason="Partial remainder cancelled")

        snapshot = manager.get_execution_state("ord-001")
        assert snapshot.status == ExecutionStatus.CANCELLED
        assert snapshot.order.status == OrderStatus.CANCELLED
        assert snapshot.filled_quantity == Decimal(30)
        assert len(snapshot.fills) == 1
        assert snapshot.average_fill_price == Decimal("150.50")
        assert snapshot.leaves_quantity == Decimal(0)
        assert snapshot.cancelled_quantity == Decimal(70)
        assert snapshot.unfilled_quantity == Decimal(70)
        assert snapshot.cancellation_timestamp == t_cancel

        assert manager.active_order_count == 0
        assert manager.total_order_count == 1

    def test_reject_cancel_for_unknown_order(self, base_time: datetime) -> None:
        manager = OrderManager()
        with pytest.raises(OrderNotFoundError, match="not found in registry"):
            manager.cancel_order("ord-missing", timestamp=base_time, reason="cancel")

    def test_reject_cancel_with_whitespace_order_id(self, base_time: datetime) -> None:
        manager = OrderManager()
        with pytest.raises(ValueError, match="cannot contain leading or trailing whitespace"):
            manager.cancel_order(" ord-001 ", timestamp=base_time, reason="cancel")

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
# 5. Adversarial Encapsulation & Snapshot Isolation Tests
# ==============================================================================


class TestAdversarialEncapsulation:
    def test_snapshot_mutation_is_strictly_impossible(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        snapshot = manager.register_order(event, instrument_spec=sample_spec)

        # Mutating frozen snapshot fields directly must raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            snapshot.leaves_quantity = Decimal(0)  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            snapshot.filled_quantity = Decimal(100)  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            snapshot.status = ExecutionStatus.FILLED  # type: ignore[misc]

        # Mutating nested OrderSnapshot must raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            snapshot.order.quantity = 999.0  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            snapshot.order.status = OrderStatus.CANCELLED  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            snapshot.order.limit_price = 50.0  # type: ignore[misc]

    def test_snapshot_fees_and_fills_cannot_be_mutated(
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

        fill = FillRecord(
            fill_id="f-fee",
            order_id="ord-001",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_price=Decimal(150),
            fill_quantity=Decimal(10),
            fee=Decimal(2),
            commission_currency="USD",
            timestamp=base_time + timedelta(seconds=1),
        )
        snapshot = manager.record_fill(fill)

        # Cumulative fees proxy rejects mutation
        with pytest.raises(TypeError):
            snapshot.cumulative_fees["USD"] = Decimal(999)  # type: ignore[index]

        # Fills tuple rejects mutation
        with pytest.raises(AttributeError):
            snapshot.fills.append(fill)  # type: ignore[attr-defined]

    def test_direct_execution_state_mutation_methods_do_not_exist_on_snapshot(
        self,
        base_time: datetime,
        sample_approved_order: Order,
        sample_spec: InstrumentSpecification,
    ) -> None:
        manager = OrderManager()
        event = OrderSubmittedEvent(
            timestamp=base_time, order=sample_approved_order, submission_time=base_time
        )
        snapshot = manager.register_order(event, instrument_spec=sample_spec)

        # record_fill does not exist on snapshot
        assert not hasattr(snapshot, "record_fill")

        # cancel_remaining does not exist on snapshot
        assert not hasattr(snapshot, "cancel_remaining")

        # transition_order_status does not exist on snapshot
        assert not hasattr(snapshot, "transition_order_status")

    def test_mutating_cancelled_event_order_does_not_modify_manager(
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

        cancelled_event = manager.cancel_order(
            "ord-001", timestamp=base_time + timedelta(seconds=1), reason="cancel"
        )

        # Mutate the order entity returned in the event
        cancelled_event.order.quantity = 999.0
        cancelled_event.order.limit_price = 1.0

        # Manager's internal state must remain completely unaltered
        snapshot = manager.get_execution_state("ord-001")
        assert snapshot.original_quantity == Decimal(100)
        assert snapshot.order.quantity == 100.0
        assert snapshot.order.limit_price == 150.0

    def test_mutating_items_from_returned_lists_is_impossible(
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

        active_orders = manager.get_active_orders()
        assert len(active_orders) == 1
        with pytest.raises(FrozenInstanceError):
            active_orders[0].leaves_quantity = Decimal(0)  # type: ignore[misc]

        all_orders = manager.get_all_orders()
        assert len(all_orders) == 1
        with pytest.raises(FrozenInstanceError):
            all_orders[0].order.quantity = 0.0  # type: ignore[misc]

    def test_active_order_count_and_state_consistency(
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

        assert manager.active_order_count == len(manager.get_active_orders()) == 1

        # Partial fill
        manager.record_fill(
            FillRecord(
                fill_id="f-1",
                order_id="ord-001",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_price=Decimal(150),
                fill_quantity=Decimal(50),
                timestamp=base_time + timedelta(seconds=1),
            )
        )
        assert manager.active_order_count == len(manager.get_active_orders()) == 1

        # Cancellation of remainder
        manager.cancel_order("ord-001", timestamp=base_time + timedelta(seconds=2), reason="done")
        assert manager.active_order_count == len(manager.get_active_orders()) == 0
        assert manager.total_order_count == len(manager.get_all_orders()) == 1


# ==============================================================================
# 6. Determinism Tests
# ==============================================================================


class TestOrderManagerDeterminism:
    def test_deterministic_active_order_sorting(
        self, base_time: datetime, sample_spec: InstrumentSpecification
    ) -> None:
        manager = OrderManager()
        manager.register_instrument(sample_spec)

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

        manager.cancel_order(
            "ord-001", timestamp=base_time + timedelta(seconds=1), reason="cancelled"
        )
        assert manager.active_order_count == 0

        reregistered = manager.register_order(event, instrument_spec=sample_spec)
        assert reregistered.status == ExecutionStatus.CANCELLED
        assert manager.active_order_count == 0
        assert len(manager.get_active_orders()) == 0
        assert manager.total_order_count == 1
