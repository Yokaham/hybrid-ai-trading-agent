"""Unit tests for deterministic EventQueue and causal boundary ordering."""

from datetime import UTC, datetime, timedelta

import pytest

from hybrid_agent.core.constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    TemporalOrderViolationError,
)
from hybrid_agent.core.events import BaseEvent, SignalEvent
from hybrid_agent.core.models import Bar, Order
from hybrid_agent.execution.backtest.clock import SimulationClock
from hybrid_agent.execution.backtest.event_queue import EventQueue
from hybrid_agent.execution.backtest.events import (
    EventPriority,
    MarketBarClosedEvent,
    MarketBarOpenedEvent,
    OrderFilledEvent,
    OrderSubmittedEvent,
    PortfolioMarkedEvent,
)
from hybrid_agent.execution.backtest.exceptions import (
    EmptyQueueError,
    InvalidEventPriorityError,
    RetroactiveEventError,
)


class DummyCustomEvent(BaseEvent, frozen=True):
    name: str


@pytest.fixture
def clock() -> SimulationClock:
    return SimulationClock()


@pytest.fixture
def queue(clock: SimulationClock) -> EventQueue:
    return EventQueue(clock=clock)


def test_empty_queue_behavior(queue: EventQueue) -> None:
    assert queue.is_empty()
    assert len(queue) == 0
    assert queue.peek() is None

    with pytest.raises(EmptyQueueError, match="Cannot pop from an empty EventQueue"):
        queue.pop()


def test_single_event_enqueue_and_pop(queue: EventQueue) -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    evt = DummyCustomEvent(timestamp=t0, name="e1")

    item = queue.enqueue(evt, priority=EventPriority.MARKET_BAR_CLOSED)
    assert not queue.is_empty()
    assert len(queue) == 1
    assert queue.peek() == item

    popped = queue.pop()
    assert popped == item
    assert popped.event == evt
    assert queue.clock.now() == t0
    assert queue.is_empty()


def test_deterministic_timestamp_ordering(queue: EventQueue) -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=1)
    t2 = t0 + timedelta(minutes=2)

    # Enqueue in reverse chronological order
    queue.enqueue(DummyCustomEvent(timestamp=t2, name="t2"), priority=1)
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="t0"), priority=1)
    queue.enqueue(DummyCustomEvent(timestamp=t1, name="t1"), priority=1)

    p0 = queue.pop()
    p1 = queue.pop()
    p2 = queue.pop()
    assert isinstance(p0.event, DummyCustomEvent) and p0.event.name == "t0"
    assert isinstance(p1.event, DummyCustomEvent) and p1.event.name == "t1"
    assert isinstance(p2.event, DummyCustomEvent) and p2.event.name == "t2"


def test_deterministic_priority_ordering_at_same_timestamp(queue: EventQueue) -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)

    # Enqueue events at the same timestamp in arbitrary priority order
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="p8"), priority=EventPriority.PORTFOLIO_MARK_TO_MARKET)
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="p1"), priority=EventPriority.MARKET_BAR_CLOSED)
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="p5"), priority=EventPriority.MARKET_BAR_OPENED)
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="p3"), priority=EventPriority.ORDER_PROPOSAL_RISK)

    # Must pop strictly in ascending priority order (1, 3, 5, 8)
    assert queue.pop().priority == 1
    assert queue.pop().priority == 3
    assert queue.pop().priority == 5
    assert queue.pop().priority == 8


def test_fifo_ordering_at_identical_timestamp_and_priority(queue: EventQueue) -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)

    e1 = queue.enqueue(DummyCustomEvent(timestamp=t0, name="first"), priority=EventPriority.STRATEGY_EVALUATION)
    e2 = queue.enqueue(DummyCustomEvent(timestamp=t0, name="second"), priority=EventPriority.STRATEGY_EVALUATION)
    e3 = queue.enqueue(DummyCustomEvent(timestamp=t0, name="third"), priority=EventPriority.STRATEGY_EVALUATION)

    # Monotonic sequence IDs
    assert e1.sequence_id < e2.sequence_id < e3.sequence_id

    # Strict FIFO popping
    p1 = queue.pop()
    p2 = queue.pop()
    p3 = queue.pop()
    assert isinstance(p1.event, DummyCustomEvent) and p1.event.name == "first"
    assert isinstance(p2.event, DummyCustomEvent) and p2.event.name == "second"
    assert isinstance(p3.event, DummyCustomEvent) and p3.event.name == "third"


def test_rejection_of_past_timestamp_events(queue: EventQueue) -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    t1 = t0 + timedelta(minutes=5)

    queue.enqueue(DummyCustomEvent(timestamp=t1, name="t1"), priority=1)
    queue.pop()  # Clock is now at t1

    # Attempting to enqueue an event at t0 < clock.now() must be rejected
    past_event = DummyCustomEvent(timestamp=t0, name="past")
    with pytest.raises(TemporalOrderViolationError, match="Temporal causality violation"):
        queue.enqueue(past_event, priority=1)


def test_invalid_priority_rejected(queue: EventQueue) -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    evt = DummyCustomEvent(timestamp=t0, name="e")

    with pytest.raises(InvalidEventPriorityError, match="Priority must be between 1 and 8"):
        queue.enqueue(evt, priority=0)

    with pytest.raises(InvalidEventPriorityError, match="Priority must be between 1 and 8"):
        queue.enqueue(evt, priority=9)


def test_non_base_event_rejected(queue: EventQueue) -> None:
    with pytest.raises(TypeError, match="Expected BaseEvent"):
        queue.enqueue("not an event", priority=1)  # type: ignore


def test_event_generation_during_dispatch_higher_or_equal_priority(queue: EventQueue) -> None:
    """Demonstrates that events generated during dispatch at timestamp T are scheduled deterministically."""
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)

    # Initial event at priority 2 (Strategy Evaluation)
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="strategy_eval"), priority=EventPriority.STRATEGY_EVALUATION)

    dispatched = queue.pop()
    assert dispatched.priority == EventPriority.STRATEGY_EVALUATION

    # During strategy evaluation at t0, strategy generates an Order Proposal at Priority 3
    # Same timestamp, higher priority number (3 >= 2) -> Allowed!
    p3 = queue.enqueue(DummyCustomEvent(timestamp=t0, name="order_proposal"), priority=EventPriority.ORDER_PROPOSAL_RISK)
    assert p3.priority == EventPriority.ORDER_PROPOSAL_RISK

    # Pop Priority 3
    dispatched_p3 = queue.pop()
    assert isinstance(dispatched_p3.event, DummyCustomEvent)
    assert dispatched_p3.event.name == "order_proposal"

    # During Priority 3, order is approved and submitted at Priority 4 -> Allowed!
    p4 = queue.enqueue(DummyCustomEvent(timestamp=t0, name="order_submit"), priority=EventPriority.ORDER_SUBMISSION_RESERVATION)
    assert p4.priority == EventPriority.ORDER_SUBMISSION_RESERVATION

    dispatched_p4 = queue.pop()
    assert isinstance(dispatched_p4.event, DummyCustomEvent)
    assert dispatched_p4.event.name == "order_submit"


def test_rejection_of_retroactive_lower_priority_events(queue: EventQueue) -> None:
    """Verifies that an event cannot be injected retroactively at an earlier priority stage at the current timestamp."""
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)

    queue.enqueue(DummyCustomEvent(timestamp=t0, name="opened"), priority=EventPriority.MARKET_BAR_OPENED)

    # Pop priority 5
    popped = queue.pop()
    assert popped.priority == EventPriority.MARKET_BAR_OPENED

    # Attempting to enqueue an order proposal (Priority 3) at timestamp t0 after Priority 5 has already executed
    # must fail closed with RetroactiveEventError!
    retroactive_evt = DummyCustomEvent(timestamp=t0, name="late_proposal")
    with pytest.raises(RetroactiveEventError, match="Retroactive event violation"):
        queue.enqueue(retroactive_evt, priority=EventPriority.ORDER_PROPOSAL_RISK)

    # But an event at Priority 5 or Priority 6 at t0 is accepted:
    p6 = queue.enqueue(DummyCustomEvent(timestamp=t0, name="fill_matching"), priority=EventPriority.ORDER_MATCHING_FILL)
    assert p6.priority == EventPriority.ORDER_MATCHING_FILL


def test_dispatch_advances_to_next_timestamp_resets_priority_check(queue: EventQueue) -> None:
    """When the queue moves to timestamp T+1, Priority 1 is allowed again for T+1."""
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    t1 = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)

    # Priority 8 at t0
    queue.enqueue(DummyCustomEvent(timestamp=t0, name="p8_t0"), priority=EventPriority.PORTFOLIO_MARK_TO_MARKET)
    # Priority 1 at t1
    queue.enqueue(DummyCustomEvent(timestamp=t1, name="p1_t1"), priority=EventPriority.MARKET_BAR_CLOSED)

    # Pop t0
    item0 = queue.pop()
    assert item0.timestamp == t0
    assert item0.priority == EventPriority.PORTFOLIO_MARK_TO_MARKET

    # Pop t1
    item1 = queue.pop()
    assert item1.timestamp == t1
    assert item1.priority == EventPriority.MARKET_BAR_CLOSED


# ==============================================================================
# Critical Boundary-Ordering & Zero-Latency Execution Sequence Tests
# ==============================================================================

def test_critical_zero_latency_boundary_sequence(queue: EventQueue) -> None:
    """Exact demonstration of the zero-latency boundary sequence at timestamp T.

    At boundary timestamp T:
    1. MarketBarClosedEvent for bar t completes.
    2. Strategy evaluation callback processes bar t.
    3. Order proposal and pre-trade risk evaluation.
    4. Order submission (records T_submission = T, T_eligible = T).
    5. MarketBarOpenedEvent for bar t+1 opens at T.
    6. Order matching and fill processing (order matches against bar t+1 Open).
    7. Ledger reconciliation.
    8. Portfolio mark-to-market.
    """
    t_start_t = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    t_boundary = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    t_end_t1 = datetime(2026, 1, 1, 9, 40, tzinfo=UTC)

    bar_t = Bar(
        symbol="AAPL",
        start_time=t_start_t,
        end_time=t_boundary,
        open=150.0,
        high=152.0,
        low=149.0,
        close=151.0,
        volume=1000.0,
    )
    bar_t1 = Bar(
        symbol="AAPL",
        start_time=t_boundary,
        end_time=t_end_t1,
        open=151.2,
        high=153.0,
        low=150.5,
        close=152.5,
        volume=1200.0,
    )

    execution_log: list[str] = []

    # 1. Enqueue Bar t Closed Event at boundary T
    queue.enqueue(
        MarketBarClosedEvent(timestamp=t_boundary, bar=bar_t),
        priority=EventPriority.MARKET_BAR_CLOSED,
    )

    # 5. Enqueue Bar t+1 Opened Event at boundary T
    queue.enqueue(
        MarketBarOpenedEvent(timestamp=t_boundary, bar=bar_t1),
        priority=EventPriority.MARKET_BAR_OPENED,
    )

    # Step 1: Pop MarketBarClosedEvent (Priority 1)
    evt1 = queue.pop()
    assert evt1.priority == EventPriority.MARKET_BAR_CLOSED
    execution_log.append("1: MarketBarClosedEvent(bar_t)")

    # Step 2: Dispatch triggers Strategy Evaluation (Priority 2)
    queue.enqueue(
        SignalEvent(timestamp=t_boundary, symbol="AAPL", score=1.0, confidence=0.9, direction=1),
        priority=EventPriority.STRATEGY_EVALUATION,
    )
    evt2 = queue.pop()
    assert evt2.priority == EventPriority.STRATEGY_EVALUATION
    execution_log.append("2: StrategyEvaluation")

    # Step 3: Strategy proposes order to pre-trade risk engine (Priority 3)
    order = Order(
        id="ord-next-open",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        status=OrderStatus.SUBMITTED,
        created_at=t_boundary,
        updated_at=t_boundary,
    )
    queue.enqueue(
        DummyCustomEvent(timestamp=t_boundary, name="RiskApproved"),
        priority=EventPriority.ORDER_PROPOSAL_RISK,
    )
    evt3 = queue.pop()
    assert evt3.priority == EventPriority.ORDER_PROPOSAL_RISK
    execution_log.append("3: RiskApproved")

    # Step 4: Order submitted and registered before MarketBarOpenedEvent executes (Priority 4)
    queue.enqueue(
        OrderSubmittedEvent(timestamp=t_boundary, order=order, submission_time=t_boundary),
        priority=EventPriority.ORDER_SUBMISSION_RESERVATION,
    )
    evt4 = queue.pop()
    assert evt4.priority == EventPriority.ORDER_SUBMISSION_RESERVATION
    execution_log.append("4: OrderSubmitted")

    # Step 5: MarketBarOpenedEvent pops next (Priority 5)
    evt5 = queue.pop()
    assert evt5.priority == EventPriority.MARKET_BAR_OPENED
    execution_log.append("5: MarketBarOpenedEvent(bar_t1)")

    # Step 6: Order matching executes order against bar_t1 Open (Priority 6)
    # T_execution (t_boundary) >= T_submission (t_boundary)
    assert isinstance(evt4.event, OrderSubmittedEvent)
    assert evt4.event.submission_time <= t_boundary
    queue.enqueue(
        OrderFilledEvent(
            timestamp=t_boundary,
            order=order,
            fill_price=bar_t1.open,
            fill_quantity=50.0,
            fee=1.0,
            slippage=0.0,
        ),
        priority=EventPriority.ORDER_MATCHING_FILL,
    )
    evt6 = queue.pop()
    assert evt6.priority == EventPriority.ORDER_MATCHING_FILL
    assert isinstance(evt6.event, OrderFilledEvent)
    execution_log.append(f"6: OrderFilled at {evt6.event.fill_price}")

    # Step 7: Ledger and position reconciliation (Priority 7)
    queue.enqueue(
        DummyCustomEvent(timestamp=t_boundary, name="LedgerReconciled"),
        priority=EventPriority.LEDGER_RECONCILIATION,
    )
    evt7 = queue.pop()
    assert evt7.priority == EventPriority.LEDGER_RECONCILIATION
    execution_log.append("7: LedgerReconciled")

    # Step 8: Portfolio mark-to-market snapshot (Priority 8)
    queue.enqueue(
        PortfolioMarkedEvent(
            timestamp=t_boundary,
            equity=10000.0,
            cash=2440.0,
            unrealized_pnl=0.0,
            realized_pnl=0.0,
            drawdown_pct=0.0,
        ),
        priority=EventPriority.PORTFOLIO_MARK_TO_MARKET,
    )
    evt8 = queue.pop()
    assert evt8.priority == EventPriority.PORTFOLIO_MARK_TO_MARKET
    execution_log.append("8: PortfolioMarked")

    assert execution_log == [
        "1: MarketBarClosedEvent(bar_t)",
        "2: StrategyEvaluation",
        "3: RiskApproved",
        "4: OrderSubmitted",
        "5: MarketBarOpenedEvent(bar_t1)",
        "6: OrderFilled at 151.2",
        "7: LedgerReconciled",
        "8: PortfolioMarked",
    ]


def test_order_submitted_after_open_event_cannot_execute_retrospectively(queue: EventQueue) -> None:
    """If an order is submitted after the bar opening event has already processed,

    it cannot be backdated or filled at that opening price.
    """
    t_boundary = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    t_next_tick = t_boundary + timedelta(seconds=10)

    bar_t1 = Bar(
        symbol="AAPL",
        start_time=t_boundary,
        end_time=t_boundary + timedelta(minutes=5),
        open=151.2,
        high=153.0,
        low=150.5,
        close=152.5,
        volume=1200.0,
    )

    # 1. MarketBarOpenedEvent is enqueued and dispatched first (Priority 5)
    queue.enqueue(
        MarketBarOpenedEvent(timestamp=t_boundary, bar=bar_t1),
        priority=EventPriority.MARKET_BAR_OPENED,
    )
    open_evt = queue.pop()
    assert open_evt.priority == EventPriority.MARKET_BAR_OPENED

    # 2. Attempting to submit an order retrospectively at Priority 4 at t_boundary fails closed!
    order = Order(
        id="ord-late",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=50.0,
        status=OrderStatus.SUBMITTED,
        created_at=t_boundary,
        updated_at=t_boundary,
    )
    late_submission = OrderSubmittedEvent(timestamp=t_boundary, order=order, submission_time=t_boundary)
    with pytest.raises(RetroactiveEventError, match="Retroactive event violation"):
        queue.enqueue(late_submission, priority=EventPriority.ORDER_SUBMISSION_RESERVATION)

    # 3. Instead, the late order must be submitted with a timestamp after the open (e.g. at next tick)
    valid_delayed_submission = OrderSubmittedEvent(
        timestamp=t_next_tick, order=order, submission_time=t_next_tick
    )
    queue.enqueue(valid_delayed_submission, priority=EventPriority.ORDER_SUBMISSION_RESERVATION)

    # Clock cannot fill order at t_boundary Open; order executes only at or after t_next_tick
    next_evt = queue.pop()
    assert next_evt.timestamp == t_next_tick
    assert next_evt.timestamp >= valid_delayed_submission.submission_time
