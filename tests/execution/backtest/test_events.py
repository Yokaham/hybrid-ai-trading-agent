"""Unit tests for backtesting event DTOs and priority contracts."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from hybrid_agent.core.constants import OrderSide, OrderStatus, OrderType
from hybrid_agent.core.models import Bar, Order, Position
from hybrid_agent.execution.backtest.events import (
    BacktestCompletedEvent,
    EventPriority,
    MarketBarClosedEvent,
    MarketBarOpenedEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderPartiallyFilledEvent,
    OrderSubmittedEvent,
    PortfolioMarkedEvent,
    PositionUpdatedEvent,
)


@pytest.fixture
def sample_bar() -> Bar:
    t_start = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    t_end = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    return Bar(
        symbol="AAPL",
        start_time=t_start,
        end_time=t_end,
        open=150.0,
        high=152.0,
        low=149.5,
        close=151.0,
        volume=1000.0,
    )


@pytest.fixture
def sample_order() -> Order:
    now = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    return Order(
        id="ord-001",
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=100.0,
        limit_price=150.0,
        status=OrderStatus.SUBMITTED,
        created_at=now,
        updated_at=now,
    )


def test_event_priority_ordering() -> None:
    assert int(EventPriority.MARKET_BAR_CLOSED) == 1
    assert int(EventPriority.STRATEGY_EVALUATION) == 2
    assert int(EventPriority.ORDER_PROPOSAL_RISK) == 3
    assert int(EventPriority.ORDER_SUBMISSION_RESERVATION) == 4
    assert int(EventPriority.MARKET_BAR_OPENED) == 5
    assert int(EventPriority.ORDER_MATCHING_FILL) == 6
    assert int(EventPriority.LEDGER_RECONCILIATION) == 7
    assert int(EventPriority.PORTFOLIO_MARK_TO_MARKET) == 8

    # Strictly monotonic order
    priorities = list(EventPriority)
    assert sorted(priorities) == priorities


def test_market_bar_closed_event_valid(sample_bar: Bar) -> None:
    # Availability exactly at bar close
    evt1 = MarketBarClosedEvent(timestamp=sample_bar.end_time, bar=sample_bar)
    assert evt1.timestamp == sample_bar.end_time
    assert evt1.bar == sample_bar

    # Availability with positive latency (e.g. +500ms network latency)
    t_avail = sample_bar.end_time + timedelta(milliseconds=500)
    evt2 = MarketBarClosedEvent(timestamp=t_avail, bar=sample_bar)
    assert evt2.timestamp == t_avail


def test_market_bar_closed_event_before_end_time_rejected(sample_bar: Bar) -> None:
    # Cannot be available before the bar ends (zero lookahead constraint)
    t_premature = sample_bar.end_time - timedelta(seconds=1)
    with pytest.raises(ValidationError, match="Data causality violation"):
        MarketBarClosedEvent(timestamp=t_premature, bar=sample_bar)


def test_market_bar_opened_event_valid(sample_bar: Bar) -> None:
    evt = MarketBarOpenedEvent(timestamp=sample_bar.start_time, bar=sample_bar)
    assert evt.timestamp == sample_bar.start_time
    assert evt.bar == sample_bar


def test_market_bar_opened_event_mismatch_rejected(sample_bar: Bar) -> None:
    t_mismatch = sample_bar.start_time + timedelta(seconds=1)
    with pytest.raises(ValidationError, match="Temporal boundary mismatch"):
        MarketBarOpenedEvent(timestamp=t_mismatch, bar=sample_bar)


def test_order_submitted_event(sample_order: Order) -> None:
    t_sub = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    evt = OrderSubmittedEvent(timestamp=t_sub, order=sample_order)
    assert evt.timestamp == t_sub
    assert evt.submission_time == t_sub
    assert evt.order.id == "ord-001"


def test_order_partially_filled_event(sample_order: Order) -> None:
    t_fill = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    evt = OrderPartiallyFilledEvent(
        timestamp=t_fill,
        order=sample_order,
        fill_price=149.8,
        fill_quantity=40.0,
        fee=1.5,
        slippage=0.05,
    )
    assert evt.fill_quantity == 40.0
    assert evt.fill_price == 149.8
    assert evt.fee == 1.5
    assert evt.slippage == 0.05


def test_order_filled_event(sample_order: Order) -> None:
    t_fill = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    evt = OrderFilledEvent(
        timestamp=t_fill,
        order=sample_order,
        fill_price=149.8,
        fill_quantity=100.0,
        fee=2.5,
        slippage=0.05,
    )
    assert evt.fill_quantity == 100.0
    assert evt.fill_price == 149.8
    assert evt.fee == 2.5


def test_order_cancelled_event(sample_order: Order) -> None:
    t_cancel = datetime(2026, 1, 1, 9, 36, tzinfo=UTC)
    evt = OrderCancelledEvent(
        timestamp=t_cancel,
        order=sample_order,
        reason="User requested cancel",
    )
    assert evt.reason == "User requested cancel"
    assert evt.order == sample_order


def test_position_updated_event() -> None:
    t_upd = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    pos = Position(
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=100.0,
        average_entry_price=150.0,
        current_price=151.0,
    )
    evt = PositionUpdatedEvent(timestamp=t_upd, position=pos)
    assert evt.position == pos


def test_portfolio_marked_event() -> None:
    t_mark = datetime(2026, 1, 1, 9, 35, tzinfo=UTC)
    evt = PortfolioMarkedEvent(
        timestamp=t_mark,
        equity=10500.0,
        cash=9000.0,
        unrealized_pnl=1500.0,
        realized_pnl=0.0,
        drawdown_pct=0.02,
    )
    assert evt.equity == 10500.0
    assert evt.drawdown_pct == 0.02


def test_backtest_completed_event() -> None:
    t_end = datetime(2026, 1, 1, 16, 0, tzinfo=UTC)
    evt = BacktestCompletedEvent(
        timestamp=t_end,
        total_bars=390,
        total_trades=12,
        final_equity=11200.5,
        completed_at=t_end,
    )
    assert evt.total_bars == 390
    assert evt.final_equity == 11200.5
