"""Strictly monotonic deterministic simulation clock for backtesting."""

from datetime import datetime

from hybrid_agent.core.constants import TemporalOrderViolationError
from hybrid_agent.core.models import validate_utc_timestamp
from hybrid_agent.execution.backtest.exceptions import ClockNotInitializedError


class SimulationClock:
    """Strictly monotonic, deterministic simulation clock operating exclusively in UTC.

    Maintains a simulated timeline that:
    1. Starts in an explicitly uninitialized state until initialized or advanced.
    2. Enforces strictly non-decreasing time advancement (t_next >= t_current).
    3. Allows multiple events at the exact same timestamp without state corruption.
    4. Prohibits any backward time movement by raising TemporalOrderViolationError.
    5. Operates independently from machine wall-clock time.
    """

    def __init__(self, initial_time: datetime | None = None) -> None:
        """Instantiate a SimulationClock, optionally initializing it to a specific UTC timestamp."""
        self._current_time: datetime | None = None
        if initial_time is not None:
            self.initialize(initial_time)

    @property
    def is_initialized(self) -> bool:
        """Return True if the clock has been initialized with a valid UTC timestamp."""
        return self._current_time is not None

    def initialize(self, timestamp: datetime) -> None:
        """Initialize the clock with a starting UTC timestamp.

        Raises:
            ValueError: If the timestamp is not timezone-aware UTC.
            TemporalOrderViolationError: If already initialized and timestamp is earlier than current time.
        """
        validated = validate_utc_timestamp(timestamp)
        if self._current_time is not None and validated < self._current_time:
            raise TemporalOrderViolationError(
                f"Cannot re-initialize SimulationClock backward: "
                f"target {validated.isoformat()} is earlier than current {self._current_time.isoformat()}"
            )
        self._current_time = validated

    def now(self) -> datetime:
        """Return the current simulated UTC time.

        Raises:
            ClockNotInitializedError: If the clock has not been initialized.
        """
        if self._current_time is None:
            raise ClockNotInitializedError(
                "SimulationClock has not been initialized. "
                "Call initialize() or advance_to() with a valid UTC timestamp."
            )
        return self._current_time

    def advance_to(self, target_time: datetime) -> datetime:
        """Advance the simulated clock to the target UTC timestamp.

        Advancing to the exact current timestamp is permitted (no-op).
        Advancing backward raises TemporalOrderViolationError.

        Args:
            target_time: Timezone-aware UTC timestamp to advance to.

        Returns:
            The updated current simulation timestamp.

        Raises:
            ValueError: If target_time is not timezone-aware UTC.
            TemporalOrderViolationError: If target_time < current_time.
        """
        validated = validate_utc_timestamp(target_time)

        if self._current_time is None:
            self._current_time = validated
            return self._current_time

        if validated < self._current_time:
            raise TemporalOrderViolationError(
                f"Temporal violation: Cannot advance SimulationClock backward from "
                f"{self._current_time.isoformat()} to {validated.isoformat()}"
            )

        self._current_time = validated
        return self._current_time

    def reset(self) -> None:
        """Reset the clock back to the uninitialized state."""
        self._current_time = None

    def __repr__(self) -> str:
        current_str = self._current_time.isoformat() if self._current_time else "UNINITIALIZED"
        return f"SimulationClock(current_time={current_str})"
