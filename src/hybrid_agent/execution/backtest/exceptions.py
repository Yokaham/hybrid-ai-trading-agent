"""Exception types for the backtest execution engine."""

from hybrid_agent.core.constants import TemporalOrderViolationError


class ClockNotInitializedError(RuntimeError):
    """Raised when an operation requires an initialized SimulationClock."""


class EmptyQueueError(IndexError):
    """Raised when an attempt is made to pop from an empty EventQueue."""


class RetroactiveEventError(TemporalOrderViolationError):
    """Raised when an event is enqueued retroactively at or before an already executed stage."""


class InvalidEventPriorityError(ValueError):
    """Raised when an event is enqueued with an unrecognized or out-of-range priority."""


class SequenceIdOverflowError(OverflowError):
    """Raised when the EventQueue sequence ID counter reaches the maximum unsigned 64-bit integer limit (2**64 - 1)."""
