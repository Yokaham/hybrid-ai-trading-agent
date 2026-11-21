"""Unit tests for the backtest SimulationClock."""

from datetime import UTC, datetime, timedelta

import pytest

from hybrid_agent.core.constants import TemporalOrderViolationError
from hybrid_agent.execution.backtest.clock import SimulationClock
from hybrid_agent.execution.backtest.exceptions import ClockNotInitializedError


def test_clock_uninitialized_state() -> None:
    clock = SimulationClock()
    assert not clock.is_initialized
    with pytest.raises(ClockNotInitializedError, match="has not been initialized"):
        clock.now()


def test_clock_initialization() -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock = SimulationClock(initial_time=t0)
    assert clock.is_initialized
    assert clock.now() == t0

    # Explicit initialization method
    clock2 = SimulationClock()
    clock2.initialize(t0)
    assert clock2.is_initialized
    assert clock2.now() == t0


def test_clock_naive_datetime_rejected() -> None:
    clock = SimulationClock()
    naive_dt = datetime(2026, 1, 1, 9, 30)
    with pytest.raises(ValueError, match="timezone-aware"):
        clock.initialize(naive_dt)

    with pytest.raises(ValueError, match="timezone-aware"):
        clock.advance_to(naive_dt)


def test_clock_monotonic_advancement() -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock = SimulationClock(t0)

    t1 = t0 + timedelta(minutes=5)
    advanced = clock.advance_to(t1)
    assert advanced == t1
    assert clock.now() == t1

    t2 = t1 + timedelta(seconds=1)
    clock.advance_to(t2)
    assert clock.now() == t2


def test_clock_advance_to_same_timestamp_allowed() -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock = SimulationClock(t0)

    # Advancing to the exact same timestamp is permitted (no-op)
    res = clock.advance_to(t0)
    assert res == t0
    assert clock.now() == t0


def test_clock_backward_advancement_rejected() -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock = SimulationClock(t0)

    earlier = t0 - timedelta(seconds=1)
    with pytest.raises(TemporalOrderViolationError, match="Cannot advance SimulationClock backward"):
        clock.advance_to(earlier)

    # Verify time was not corrupted
    assert clock.now() == t0


def test_clock_reinitialization_backward_rejected() -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock = SimulationClock(t0)

    earlier = t0 - timedelta(minutes=1)
    with pytest.raises(TemporalOrderViolationError, match="Cannot re-initialize SimulationClock backward"):
        clock.initialize(earlier)


def test_clock_reset() -> None:
    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock = SimulationClock(t0)
    assert clock.is_initialized

    clock.reset()
    assert not clock.is_initialized
    with pytest.raises(ClockNotInitializedError):
        clock.now()

    # Can be re-initialized after reset
    t1 = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
    clock.initialize(t1)
    assert clock.now() == t1


def test_clock_repr() -> None:
    clock = SimulationClock()
    assert "UNINITIALIZED" in repr(clock)

    t0 = datetime(2026, 1, 1, 9, 30, tzinfo=UTC)
    clock.initialize(t0)
    assert "2026-01-01T09:30:00+00:00" in repr(clock)
