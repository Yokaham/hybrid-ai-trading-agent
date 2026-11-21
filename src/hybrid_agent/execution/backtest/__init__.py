"""Native backtest execution engine infrastructure."""

from hybrid_agent.core.constants import TemporalOrderViolationError
from hybrid_agent.execution.backtest.clock import SimulationClock
from hybrid_agent.execution.backtest.event_queue import EventQueue, PrioritizedEvent
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
from hybrid_agent.execution.backtest.exceptions import (
    ClockNotInitializedError,
    EmptyQueueError,
    InvalidEventPriorityError,
    RetroactiveEventError,
)

__all__ = [
    "BacktestCompletedEvent",
    "ClockNotInitializedError",
    "EmptyQueueError",
    "EventPriority",
    "EventQueue",
    "InvalidEventPriorityError",
    "MarketBarClosedEvent",
    "MarketBarOpenedEvent",
    "OrderCancelledEvent",
    "OrderFilledEvent",
    "OrderPartiallyFilledEvent",
    "OrderSubmittedEvent",
    "PortfolioMarkedEvent",
    "PositionUpdatedEvent",
    "PrioritizedEvent",
    "RetroactiveEventError",
    "SimulationClock",
    "TemporalOrderViolationError",
]
