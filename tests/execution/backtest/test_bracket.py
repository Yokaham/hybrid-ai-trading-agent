"""Comprehensive unit tests for OCO brackets and contingent order lifecycle management."""

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from hybrid_agent.core.constants import OrderSide, OrderStatus, OrderType
from hybrid_agent.core.models import InstrumentSpecification, Order
from hybrid_agent.execution.backtest.bracket import (
    BracketExposureBreachError,
    BracketNotFoundError,
    BracketOrderRole,
    BracketSnapshot,
    BracketStatus,
    DuplicateBracketError,
    InvalidBracketStateError,
)
from hybrid_agent.execution.backtest.events import OrderFilledEvent, OrderSubmittedEvent
from hybrid_agent.execution.backtest.order_manager import (
    DuplicateFillError,
    OrderManager,
)
from hybrid_agent.execution.backtest.portfolio import (
    AccountingClassification,
    Portfolio,
    TimestampedCurrencyConverter,
)
from hybrid_agent.execution.backtest.state import ExecutionStatus, FillRecord


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
    )


@pytest.fixture
def manager(spec: InstrumentSpecification) -> OrderManager:
    om = OrderManager()
    om.register_instrument(spec)
    return om


def make_parent_order(
    order_id: str = "ord-parent",
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    quantity: float = 100.0,
    limit_price: float = 150.0,
    stop_loss_price: float | None = 140.0,
    take_profit_price: float | None = 165.0,
    timestamp: datetime = datetime(2025, 1, 15, 9, 30, 0, tzinfo=UTC),
) -> Order:
    return Order(
        id=order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.LIMIT,
        quantity=quantity,
        limit_price=limit_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        status=OrderStatus.APPROVED,
        created_at=timestamp,
        updated_at=timestamp,
    )


class TestOCOBracketLifecycle:
    """Tests covering the 20 required OCO bracket and contingent order scenarios."""

    def test_no_protective_child_exists_before_entry_fill(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """1. No protective child exists before an entry fill."""
        order = make_parent_order()
        event = OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        manager.register_order(event)

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.is_pending_entry is True
        assert bracket.status == BracketStatus.PENDING_ENTRY
        assert bracket.allocated_quantity == Decimal(0)
        assert bracket.residual_quantity == Decimal(0)
        assert bracket.active_stop_loss_order_id is None
        assert bracket.active_take_profit_order_id is None

        # Only parent exists in active orders
        active_orders = manager.get_active_orders()
        assert len(active_orders) == 1
        assert active_orders[0].order.id == "ord-parent"
        assert manager.active_bracket_count == 0

    def test_first_partial_entry_fill_creates_protection_for_actual_filled_quantity(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """2. First partial entry fill creates protection for the actual filled quantity."""
        order = make_parent_order(quantity=100.0)
        event = OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        manager.register_order(event)

        t1 = base_time + timedelta(seconds=1)
        fill1 = FillRecord(
            fill_id="f1",
            order_id="ord-parent",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_quantity=Decimal(30),
            fill_price=Decimal("150.00"),
            fee=Decimal(0),
            commission_currency=None,
            timestamp=t1,
        )
        manager.record_fill(fill1)

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.is_active is True
        assert bracket.status == BracketStatus.ACTIVE
        assert bracket.allocated_quantity == Decimal(30)
        assert bracket.residual_quantity == Decimal(30)
        assert manager.active_bracket_count == 1

        sl_id = bracket.active_stop_loss_order_id
        tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None
        assert tp_id is not None

        sl_state = manager.get_execution_state(sl_id)
        tp_state = manager.get_execution_state(tp_id)

        assert sl_state.original_quantity == Decimal(30)
        assert sl_state.order.side == OrderSide.SELL
        assert sl_state.order.order_type == OrderType.STOP_LOSS
        assert sl_state.order.stop_loss_price == 140.0

        assert tp_state.original_quantity == Decimal(30)
        assert tp_state.order.side == OrderSide.SELL
        assert tp_state.order.order_type == OrderType.LIMIT
        assert tp_state.order.limit_price == 165.0

    def test_subsequent_partial_entry_fills_increase_protected_exposure_correctly(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """3. Subsequent partial entry fills increase protected exposure correctly via cancel-and-replace."""
        order = make_parent_order(quantity=100.0)
        event = OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        manager.register_order(event)

        # First partial fill: 30
        t1 = base_time + timedelta(seconds=1)
        fill1 = FillRecord(
            fill_id="f1",
            order_id="ord-parent",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_quantity=Decimal(30),
            fill_price=Decimal("150.00"),
            fee=Decimal(0),
            timestamp=t1,
        )
        manager.record_fill(fill1)

        b1 = manager.get_bracket_for_order("ord-parent")
        assert b1 is not None
        old_sl_id = b1.active_stop_loss_order_id
        old_tp_id = b1.active_take_profit_order_id
        assert old_sl_id is not None and old_tp_id is not None

        # Second partial fill: 20 -> protected exposure grows to 50
        t2 = base_time + timedelta(seconds=2)
        fill2 = FillRecord(
            fill_id="f2",
            order_id="ord-parent",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_quantity=Decimal(20),
            fill_price=Decimal("150.00"),
            fee=Decimal(0),
            timestamp=t2,
        )
        manager.record_fill(fill2)

        b2 = manager.get_bracket_for_order("ord-parent")
        assert b2 is not None
        assert b2.allocated_quantity == Decimal(50)
        assert b2.residual_quantity == Decimal(50)

        # Verify old children were cancelled cleanly without silent mutation
        old_sl_state = manager.get_execution_state(old_sl_id)
        old_tp_state = manager.get_execution_state(old_tp_id)
        assert old_sl_state.is_cancelled is True
        assert old_tp_state.is_cancelled is True
        assert old_sl_state.original_quantity == Decimal(30)
        assert "scale-up" in (old_sl_state.cancellation_reason or "")

        # Verify new active children sized at 50
        new_sl_id = b2.active_stop_loss_order_id
        new_tp_id = b2.active_take_profit_order_id
        assert new_sl_id is not None and new_tp_id is not None
        assert new_sl_id != old_sl_id
        assert new_tp_id != old_tp_id

        new_sl_state = manager.get_execution_state(new_sl_id)
        new_tp_state = manager.get_execution_state(new_tp_id)
        assert new_sl_state.original_quantity == Decimal(50)
        assert new_tp_state.original_quantity == Decimal(50)
        assert new_sl_state.is_active is True
        assert new_tp_state.is_active is True

    def test_parent_cancellation_without_fills_leaves_no_active_children(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """4. Parent cancellation without fills leaves no active children."""
        order = make_parent_order()
        event = OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        manager.register_order(event)

        t1 = base_time + timedelta(seconds=1)
        manager.cancel_order("ord-parent", timestamp=t1, reason="User cancelled")

        parent_state = manager.get_execution_state("ord-parent")
        assert parent_state.is_cancelled is True

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.is_cancelled is True
        assert bracket.status == BracketStatus.CANCELLED
        assert bracket.active_stop_loss_order_id is None
        assert bracket.active_take_profit_order_id is None
        assert manager.active_order_count == 0

    def test_parent_cancellation_after_partial_fills_preserves_protection_for_filled_exposure(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """5. Parent cancellation after partial fills preserves protection for filled exposure."""
        order = make_parent_order(quantity=100.0)
        event = OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        manager.register_order(event)

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(40),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        # Cancel remaining 60 of parent order
        t2 = base_time + timedelta(seconds=2)
        manager.cancel_order("ord-parent", timestamp=t2, reason="Cancel entry remainder")

        parent_state = manager.get_execution_state("ord-parent")
        assert parent_state.is_cancelled is True
        assert parent_state.filled_quantity == Decimal(40)
        assert parent_state.cancelled_quantity == Decimal(60)

        # Protective orders MUST remain active for the 40 filled exposure
        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.is_active is True
        assert bracket.allocated_quantity == Decimal(40)
        assert bracket.residual_quantity == Decimal(40)

        sl_id = bracket.active_stop_loss_order_id
        tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None and tp_id is not None
        assert manager.get_execution_state(sl_id).is_active is True
        assert manager.get_execution_state(tp_id).is_active is True
        assert manager.get_execution_state(sl_id).leaves_quantity == Decimal(40)

    def test_full_parent_fill_produces_protection_for_full_executed_quantity(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """6. Full parent fill produces protection for the full executed quantity."""
        order = make_parent_order(quantity=100.0)
        event = OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        manager.register_order(event)

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(100),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.is_active is True
        assert bracket.allocated_quantity == Decimal(100)
        assert bracket.residual_quantity == Decimal(100)

        sl_state = manager.get_execution_state(bracket.active_stop_loss_order_id or "")
        tp_state = manager.get_execution_state(bracket.active_take_profit_order_id or "")
        assert sl_state.original_quantity == Decimal(100)
        assert tp_state.original_quantity == Decimal(100)

    def test_stop_loss_partial_fill_adjusts_sibling_correctly(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """7. Stop-loss partial fill adjusts the sibling correctly via cancel-and-replace."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        sl_id = bracket.active_stop_loss_order_id
        old_tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None and old_tp_id is not None

        # Stop loss partially fills 20 of 50
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-sl-1",
                order_id=sl_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(20),
                fill_price=Decimal("140.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        updated_bracket = manager.get_bracket_for_order("ord-parent")
        assert updated_bracket is not None
        assert updated_bracket.closed_quantity == Decimal(20)
        assert updated_bracket.residual_quantity == Decimal(30)

        # Sibling TP was cancelled and replaced with quantity 30
        old_tp_state = manager.get_execution_state(old_tp_id)
        assert old_tp_state.is_cancelled is True

        new_tp_id = updated_bracket.active_take_profit_order_id
        assert new_tp_id != old_tp_id
        assert new_tp_id is not None
        new_tp_state = manager.get_execution_state(new_tp_id)
        assert new_tp_state.original_quantity == Decimal(30)
        assert new_tp_state.leaves_quantity == Decimal(30)

        # SL order itself has 30 leaves remaining
        sl_state = manager.get_execution_state(sl_id)
        assert sl_state.leaves_quantity == Decimal(30)

    def test_take_profit_partial_fill_adjusts_sibling_correctly(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """8. Take-profit partial fill adjusts the sibling correctly via cancel-and-replace."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        old_sl_id = bracket.active_stop_loss_order_id
        tp_id = bracket.active_take_profit_order_id
        assert old_sl_id is not None and tp_id is not None

        # Take profit partially fills 20 of 50
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-tp-1",
                order_id=tp_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(20),
                fill_price=Decimal("165.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        updated_bracket = manager.get_bracket_for_order("ord-parent")
        assert updated_bracket is not None
        assert updated_bracket.closed_quantity == Decimal(20)
        assert updated_bracket.residual_quantity == Decimal(30)

        # Sibling SL was cancelled and replaced with quantity 30
        old_sl_state = manager.get_execution_state(old_sl_id)
        assert old_sl_state.is_cancelled is True

        new_sl_id = updated_bracket.active_stop_loss_order_id
        assert new_sl_id != old_sl_id
        assert new_sl_id is not None
        new_sl_state = manager.get_execution_state(new_sl_id)
        assert new_sl_state.original_quantity == Decimal(30)
        assert new_sl_state.leaves_quantity == Decimal(30)

        # TP order itself has 30 leaves remaining
        tp_state = manager.get_execution_state(tp_id)
        assert tp_state.leaves_quantity == Decimal(30)

    def test_full_stop_loss_fill_cancels_take_profit_leg(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """9. Full stop-loss fill cancels the take-profit leg and closes the bracket."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        sl_id = bracket.active_stop_loss_order_id
        tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None and tp_id is not None

        # Stop loss fills completely (50 of 50)
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-sl-full",
                order_id=sl_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(50),
                fill_price=Decimal("140.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        updated_bracket = manager.get_bracket_for_order("ord-parent")
        assert updated_bracket is not None
        assert updated_bracket.is_closed is True
        assert updated_bracket.status == BracketStatus.CLOSED
        assert updated_bracket.residual_quantity == Decimal(0)
        assert updated_bracket.closed_quantity == Decimal(50)
        assert updated_bracket.active_stop_loss_order_id is None
        assert updated_bracket.active_take_profit_order_id is None

        # Sibling TP was cancelled
        tp_state = manager.get_execution_state(tp_id)
        assert tp_state.is_cancelled is True
        assert "fully closed" in (tp_state.cancellation_reason or "")
        assert manager.active_order_count == 0

    def test_full_take_profit_fill_cancels_stop_loss_leg(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """10. Full take-profit fill cancels the stop-loss leg and closes the bracket."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        sl_id = bracket.active_stop_loss_order_id
        tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None and tp_id is not None

        # Take profit fills completely (50 of 50)
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-tp-full",
                order_id=tp_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(50),
                fill_price=Decimal("165.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        updated_bracket = manager.get_bracket_for_order("ord-parent")
        assert updated_bracket is not None
        assert updated_bracket.is_closed is True
        assert updated_bracket.status == BracketStatus.CLOSED
        assert updated_bracket.residual_quantity == Decimal(0)
        assert updated_bracket.active_stop_loss_order_id is None
        assert updated_bracket.active_take_profit_order_id is None

        # Sibling SL was cancelled
        sl_state = manager.get_execution_state(sl_id)
        assert sl_state.is_cancelled is True
        assert manager.active_order_count == 0

    def test_duplicate_parent_fill_events_do_not_duplicate_child_protection(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """11. Duplicate parent fill events do not duplicate child protection."""
        order = make_parent_order(quantity=100.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        fill1 = FillRecord(
            fill_id="f-dup-parent",
            order_id="ord-parent",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_quantity=Decimal(30),
            fill_price=Decimal("150.00"),
            fee=Decimal(0),
            timestamp=t1,
        )
        manager.record_fill(fill1)

        # Attempt duplicate fill delivery
        with pytest.raises(DuplicateFillError, match="already been recorded"):
            manager.record_fill(fill1)

        # Idempotent ignore option verification
        snap = manager.record_fill(fill1, ignore_duplicate=True)
        assert snap.filled_quantity == Decimal(30)

        # Verify bracket state is unchanged and not double-counted
        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.allocated_quantity == Decimal(30)
        assert bracket.residual_quantity == Decimal(30)

        sl_state = manager.get_execution_state(bracket.active_stop_loss_order_id or "")
        assert sl_state.original_quantity == Decimal(30)

    def test_duplicate_exit_fill_events_do_not_double_count_exits(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """12. Duplicate exit fill events do not double-count exits."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        sl_id = bracket.active_stop_loss_order_id
        assert sl_id is not None

        t2 = base_time + timedelta(seconds=2)
        fill_exit = FillRecord(
            fill_id="f-exit-dup",
            order_id=sl_id,
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_quantity=Decimal(20),
            fill_price=Decimal("140.00"),
            fee=Decimal(0),
            timestamp=t2,
        )
        manager.record_fill(fill_exit)

        # Attempt duplicate
        with pytest.raises(DuplicateFillError):
            manager.record_fill(fill_exit)

        # Idempotent ignore option
        manager.record_fill(fill_exit, ignore_duplicate=True)

        updated_bracket = manager.get_bracket_for_order("ord-parent")
        assert updated_bracket is not None
        assert updated_bracket.closed_quantity == Decimal(20)
        assert updated_bracket.residual_quantity == Decimal(30)

    def test_late_sibling_fill_after_cancellation_handled_and_audited(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """13. A late sibling fill after cancellation is handled and audited correctly without breach if within limits."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        sl_id = bracket.active_stop_loss_order_id
        old_tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None and old_tp_id is not None

        # SL fills 20 -> TP is cancelled (and replaced with new TP for 30)
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-sl-1",
                order_id=sl_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(20),
                fill_price=Decimal("140.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        # Sibling old_tp_id was cancelled:
        assert manager.get_execution_state(old_tp_id).is_cancelled is True

        # Now suppose an in-flight fill of 15 arrives late on the cancelled old_tp_id:
        # Total closed = 20 + 15 = 35 <= 50 allocated. This is valid and within allocated exposure!
        t3 = base_time + timedelta(seconds=3)
        late_fill = FillRecord(
            fill_id="f-tp-late",
            order_id=old_tp_id,
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_quantity=Decimal(15),
            fill_price=Decimal("165.00"),
            fee=Decimal(0),
            timestamp=t3,
        )
        snap = manager.record_fill(late_fill, allow_late_fill=True)

        assert snap.status == ExecutionStatus.CANCELLED
        assert snap.filled_quantity == Decimal(15)
        assert len(snap.fills) == 1

        updated_bracket = manager.get_bracket_for_order("ord-parent")
        assert updated_bracket is not None
        assert updated_bracket.closed_quantity == Decimal(35)
        assert updated_bracket.residual_quantity == Decimal(15)
        assert updated_bracket.has_exposure_breach is False

    def test_exposure_overrun_produces_explicit_breach(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """14. Exposure overrun produces an explicit breach rather than silently corrupting state."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-entry",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        sl_id = bracket.active_stop_loss_order_id
        tp_id = bracket.active_take_profit_order_id
        assert sl_id is not None and tp_id is not None

        # SL fills full 50 -> bracket closes, TP cancelled
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-sl-full",
                order_id=sl_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(50),
                fill_price=Decimal("140.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        # TP is cancelled
        assert manager.get_execution_state(tp_id).is_cancelled is True

        # Now late fill of 20 arrives for cancelled TP! Total closed would be 50 + 20 = 70 > 50!
        t3 = base_time + timedelta(seconds=3)
        late_fill = FillRecord(
            fill_id="f-tp-overrun",
            order_id=tp_id,
            symbol="AAPL",
            side=OrderSide.SELL,
            fill_quantity=Decimal(20),
            fill_price=Decimal("165.00"),
            fee=Decimal(0),
            timestamp=t3,
        )

        with pytest.raises(BracketExposureBreachError, match="exposure breach"):
            manager.record_fill(late_fill, allow_late_fill=True)

        breached_bracket = manager.get_bracket_for_order("ord-parent")
        assert breached_bracket is not None
        assert breached_bracket.is_breached is True
        assert breached_bracket.status == BracketStatus.BREACHED
        assert breached_bracket.has_exposure_breach is True
        assert breached_bracket.breach_details is not None
        assert "exceeding allocated entry quantity" in breached_bracket.breach_details

        # Late fill was recorded into order history
        tp_state = manager.get_execution_state(tp_id)
        assert tp_state.filled_quantity == Decimal(20)

    def test_separate_bracket_groups_for_same_instrument_isolated(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """15. Separate bracket groups for the same instrument do not interfere with one another."""
        o1 = make_parent_order("ord-1", quantity=50.0)
        o2 = make_parent_order("ord-2", quantity=100.0)

        manager.register_order(
            OrderSubmittedEvent(order=o1, timestamp=base_time, submission_time=base_time)
        )
        manager.register_order(
            OrderSubmittedEvent(order=o2, timestamp=base_time, submission_time=base_time)
        )

        b1 = manager.get_bracket_for_order("ord-1")
        b2 = manager.get_bracket_for_order("ord-2")
        assert b1 is not None and b2 is not None
        assert b1.bracket_id != b2.bracket_id

        # Fill ord-1
        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-1",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        b1_updated = manager.get_bracket("BRACKET-ord-1")
        b2_updated = manager.get_bracket("BRACKET-ord-2")

        assert b1_updated.status == BracketStatus.ACTIVE
        assert b1_updated.allocated_quantity == Decimal(50)

        # b2 is completely unaffected
        assert b2_updated.status == BracketStatus.PENDING_ENTRY
        assert b2_updated.allocated_quantity == Decimal(0)
        assert b2_updated.active_stop_loss_order_id is None

    def test_invalid_child_direction_symbol_price_quantity_rejected(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """16. Invalid child direction, symbol, price, or quantity is rejected."""
        # Validation on Order model instantiation
        with pytest.raises(ValueError):
            make_parent_order(
                "bad-buy",
                limit_price=150.0,
                stop_loss_price=155.0,  # Invalid: SL > entry
                take_profit_price=160.0,
            )

        with pytest.raises(ValueError):
            make_parent_order(
                "bad-sell",
                side=OrderSide.SELL,
                limit_price=150.0,
                stop_loss_price=140.0,  # Invalid: SL < entry for SELL
                take_profit_price=130.0,
            )

        # Register market parent order without limit price
        market_order = Order(
            id="ord-market",
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )
        manager.register_order(
            OrderSubmittedEvent(order=market_order, timestamp=base_time, submission_time=base_time)
        )

        # Invalid direction when registering bracket: BUY stop_loss >= take_profit
        with pytest.raises(ValueError, match="strictly below take_profit_price"):
            manager.register_bracket("ord-market", stop_loss_price=160.0, take_profit_price=150.0)

        # Non-positive prices
        with pytest.raises(ValueError, match="stop_loss_price must be positive"):
            manager.register_bracket("ord-market", stop_loss_price=-10.0, take_profit_price=160.0)

        with pytest.raises(ValueError, match="take_profit_price must be positive"):
            manager.register_bracket("ord-market", stop_loss_price=140.0, take_profit_price=0.0)

    def test_public_snapshots_cannot_mutate_internal_bracket_state(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """17. Public snapshots cannot mutate internal bracket or order state."""
        order = make_parent_order()
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        snap = manager.get_bracket_for_order("ord-parent")
        assert snap is not None
        assert isinstance(snap, BracketSnapshot)

        with pytest.raises(FrozenInstanceError):
            snap.status = BracketStatus.CLOSED  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            snap.allocated_quantity = Decimal(999)  # type: ignore[misc]

        assert isinstance(snap.stop_loss_order_ids, tuple)
        assert isinstance(snap.take_profit_order_ids, tuple)

    def test_event_timestamps_remain_chronological_and_deterministic(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """18. Event timestamps remain chronological and deterministic without wall-clock dependence."""
        order = make_parent_order()
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=10)
        manager.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.created_at == base_time
        assert bracket.updated_at == t1

        sl_snap = manager.get_execution_state(bracket.active_stop_loss_order_id or "")
        assert sl_snap.submission_time == t1

    def test_closed_bracket_cannot_create_or_reactivate_children(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """19. Closed brackets cannot create or reactivate children."""
        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="ord-parent",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        tp_id = bracket.active_take_profit_order_id
        assert tp_id is not None

        # Full take profit fill closes bracket
        t2 = base_time + timedelta(seconds=2)
        manager.record_fill(
            FillRecord(
                fill_id="f-tp-full",
                order_id=tp_id,
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(50),
                fill_price=Decimal("165.00"),
                fee=Decimal(0),
                timestamp=t2,
            )
        )

        closed_bracket = manager.get_bracket_for_order("ord-parent")
        assert closed_bracket is not None
        assert closed_bracket.is_closed is True
        assert closed_bracket.active_stop_loss_order_id is None
        assert closed_bracket.active_take_profit_order_id is None

        # Re-registering bracket on same parent returns existing closed snapshot
        rereg = manager.register_bracket(
            "ord-parent",
            stop_loss_price=order.stop_loss_price or 140.0,
            take_profit_price=order.take_profit_price or 165.0,
        )
        assert rereg.is_closed is True
        assert rereg.active_stop_loss_order_id is None

    def test_existing_execution_and_accounting_behaviour_remains_unchanged(
        self, manager: OrderManager, base_time: datetime, spec: InstrumentSpecification
    ) -> None:
        """20. Existing execution and accounting behaviour remains unchanged (Portfolio isolation)."""
        converter = TimestampedCurrencyConverter(
            default_max_age=timedelta(days=1), pivot_currency="USD"
        )
        portfolio = Portfolio(account_currency="USD", currency_converter=converter)
        portfolio.initialize_cash("USD", Decimal(100000), timestamp=base_time)
        portfolio.register_instrument(
            symbol="AAPL",
            accounting_classification=AccountingClassification.CASH_EQUITY,
            quote_currency="USD",
            profit_currency="USD",
        )

        order = make_parent_order(quantity=50.0)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        # Before fill
        assert portfolio.cashbook.get_balance("USD") == Decimal(100000)
        assert portfolio.positions.get_position("AAPL") is None

        # Fill parent entry
        t1 = base_time + timedelta(seconds=1)
        fill = FillRecord(
            fill_id="f1",
            order_id="ord-parent",
            symbol="AAPL",
            side=OrderSide.BUY,
            fill_quantity=Decimal(50),
            fill_price=Decimal("150.00"),
            fee=Decimal(5),
            commission_currency="USD",
            timestamp=t1,
        )
        manager.record_fill(fill)

        # Authoritative Portfolio applies fill independently
        order_filled_evt = OrderFilledEvent(
            order=order.model_copy(),
            fill_price=150.0,
            fill_quantity=50.0,
            fee=5.0,
            timestamp=t1,
        )
        portfolio.apply_fill(
            event=order_filled_evt,
            spec=spec,
            accounting_type=AccountingClassification.CASH_EQUITY,
            commission_currency="USD",
        )

        # Portfolio accounting is updated
        pos = portfolio.positions.get_position("AAPL")
        assert pos is not None
        assert pos.quantity == 50.0
        assert portfolio.cashbook.get_balance("USD") == Decimal(100000) - Decimal(7500) - Decimal(5)

        # OrderManager and Bracket maintain only group-level quantity allocation, zero cash/position data
        bracket = manager.get_bracket_for_order("ord-parent")
        assert bracket is not None
        assert bracket.allocated_quantity == Decimal(50)
        assert bracket.residual_quantity == Decimal(50)
        assert not hasattr(bracket, "cash")
        assert not hasattr(bracket, "position")
        assert not hasattr(bracket, "realized_pnl")

    def test_short_sell_parent_bracket_lifecycle(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """Validates short/SELL parent entry order with inverse protective direction (SL > entry > TP)."""
        sell_order = Order(
            id="ord-sell",
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50.0,
            limit_price=150.0,
            stop_loss_price=160.0,  # Above entry
            take_profit_price=135.0,  # Below entry
            status=OrderStatus.APPROVED,
            created_at=base_time,
            updated_at=base_time,
        )
        manager.register_order(
            OrderSubmittedEvent(order=sell_order, timestamp=base_time, submission_time=base_time)
        )

        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f-sell-1",
                order_id="ord-sell",
                symbol="AAPL",
                side=OrderSide.SELL,
                fill_quantity=Decimal(50),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        bracket = manager.get_bracket_for_order("ord-sell")
        assert bracket is not None
        assert bracket.entry_side == OrderSide.SELL
        assert bracket.is_active is True

        sl_snap = manager.get_execution_state(bracket.active_stop_loss_order_id or "")
        tp_snap = manager.get_execution_state(bracket.active_take_profit_order_id or "")

        # Child orders MUST be BUY
        assert sl_snap.order.side == OrderSide.BUY
        assert sl_snap.order.order_type == OrderType.STOP_LOSS
        assert sl_snap.order.stop_loss_price == 160.0

        assert tp_snap.order.side == OrderSide.BUY
        assert tp_snap.order.order_type == OrderType.LIMIT
        assert tp_snap.order.limit_price == 135.0

    def test_bracket_query_apis_and_counts(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """Validates bracket registry queries, roles, and collection filters."""
        o1 = make_parent_order("o1")
        o2 = make_parent_order("o2", symbol="AAPL")
        manager.register_order(
            OrderSubmittedEvent(order=o1, timestamp=base_time, submission_time=base_time)
        )
        manager.register_order(
            OrderSubmittedEvent(order=o2, timestamp=base_time, submission_time=base_time)
        )

        assert manager.total_bracket_count == 2
        assert manager.active_bracket_count == 0  # Both pending entry

        assert manager.has_bracket("BRACKET-o1") is True
        assert manager.has_bracket("BRACKET-o999") is False

        with pytest.raises(BracketNotFoundError):
            manager.get_bracket("BRACKET-o999")

        assert manager.get_order_role("o1") == BracketOrderRole.PARENT_ENTRY

        all_b = manager.get_all_brackets(symbol="AAPL")
        assert len(all_b) == 2

        # Activate o1
        t1 = base_time + timedelta(seconds=1)
        manager.record_fill(
            FillRecord(
                fill_id="f1",
                order_id="o1",
                symbol="AAPL",
                side=OrderSide.BUY,
                fill_quantity=Decimal(10),
                fill_price=Decimal("150.00"),
                fee=Decimal(0),
                timestamp=t1,
            )
        )

        assert manager.active_bracket_count == 1
        active_b = manager.get_active_brackets(symbol="AAPL")
        assert len(active_b) == 1
        assert active_b[0].bracket_id == "BRACKET-o1"

        sl_id = active_b[0].active_stop_loss_order_id or ""
        assert manager.get_order_role(sl_id) == BracketOrderRole.STOP_LOSS

    def test_duplicate_and_invalid_bracket_registration(
        self, manager: OrderManager, base_time: datetime
    ) -> None:
        """Validates DuplicateBracketError and InvalidBracketStateError."""
        order = make_parent_order("o-dup-test", stop_loss_price=None, take_profit_price=None)
        manager.register_order(
            OrderSubmittedEvent(order=order, timestamp=base_time, submission_time=base_time)
        )

        b1 = manager.register_bracket("o-dup-test", stop_loss_price=140.0, take_profit_price=160.0)
        assert b1.bracket_id == "BRACKET-o-dup-test"

        # Conflicting registration raises DuplicateBracketError
        with pytest.raises(DuplicateBracketError, match="already linked to bracket"):
            manager.register_bracket("o-dup-test", stop_loss_price=145.0, take_profit_price=160.0)

        # Cancel unfilled parent
        t1 = base_time + timedelta(seconds=1)
        manager.cancel_order("o-dup-test", timestamp=t1, reason="cancel")

        # Attaching to cancelled unfilled parent raises InvalidBracketStateError
        o_cancelled = make_parent_order("o-cancel", stop_loss_price=None, take_profit_price=None)
        manager.register_order(
            OrderSubmittedEvent(order=o_cancelled, timestamp=base_time, submission_time=base_time)
        )
        manager.cancel_order("o-cancel", timestamp=t1, reason="cancel")
        with pytest.raises(InvalidBracketStateError, match="Cannot attach bracket to cancelled"):
            manager.register_bracket("o-cancel", stop_loss_price=140.0, take_profit_price=160.0)
