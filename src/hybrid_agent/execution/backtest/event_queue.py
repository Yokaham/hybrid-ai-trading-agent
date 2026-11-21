"""Deterministic priority event queue for historical backtesting."""

import heapq
from dataclasses import dataclass, field
from datetime import datetime

from hybrid_agent.core.constants import TemporalOrderViolationError
from hybrid_agent.core.events import BaseEvent
from hybrid_agent.core.models import validate_utc_timestamp
from hybrid_agent.execution.backtest.clock import SimulationClock
from hybrid_agent.execution.backtest.events import EventPriority
from hybrid_agent.execution.backtest.exceptions import (
    EmptyQueueError,
    InvalidEventPriorityError,
    RetroactiveEventError,
)


@dataclass(order=True, frozen=True)
class PrioritizedEvent:
    """Heap-sortable wrapper prioritizing events by (timestamp, priority, sequence_id).

    The underlying event is excluded from comparisons to guarantee 100% deterministic
    ordering based strictly on temporal milestones and monotonic sequence IDs.
    """

    timestamp: datetime
    priority: int
    sequence_id: int
    event: BaseEvent = field(compare=False)


class EventQueue:
    """Deterministic, priority-ordered event queue for historical market simulations.

    Invariants:
    1. Events are ordered strictly by (timestamp, priority, sequence_id).
    2. Lower priority numbers dispatch before higher priority numbers within identical timestamps.
    3. Events with identical timestamps and priorities follow strict FIFO ordering via monotonic sequence IDs.
    4. Past-dated events (timestamp < clock.now()) are strictly rejected.
    5. Retroactive events (timestamp == current_timestamp and priority < current_priority) are rejected.
    6. Downstream events generated during dispatch at the current timestamp with priority >= current_priority
       are accepted and scheduled deterministically with incremented sequence IDs.
    """

    def __init__(self, clock: SimulationClock | None = None) -> None:
        """Initialize the EventQueue with an optional external SimulationClock."""
        self._clock: SimulationClock = clock if clock is not None else SimulationClock()
        self._heap: list[PrioritizedEvent] = []
        self._sequence_counter: int = 0
        self._current_dispatch_timestamp: datetime | None = None
        self._current_dispatch_priority: int | None = None

    @property
    def clock(self) -> SimulationClock:
        """Return the simulation clock associated with this event queue."""
        return self._clock

    def __len__(self) -> int:
        """Return the number of events currently waiting in the queue."""
        return len(self._heap)

    def is_empty(self) -> bool:
        """Return True if no events remain in the queue."""
        return len(self._heap) == 0

    def enqueue(self, event: BaseEvent, priority: EventPriority | int) -> PrioritizedEvent:
        """Enqueue an event with a validated priority.

        Args:
            event: A valid BaseEvent instance with a timezone-aware UTC timestamp.
            priority: An EventPriority enum value or integer between 1 and 8.

        Returns:
            The PrioritizedEvent instance inserted into the queue.

        Raises:
            TypeError: If event is not an instance of BaseEvent.
            InvalidEventPriorityError: If priority is outside the range 1..8.
            ValueError: If event.timestamp is not timezone-aware UTC.
            TemporalOrderViolationError: If event.timestamp is earlier than current clock time.
            RetroactiveEventError: If event.timestamp matches current dispatch time but priority is lower
                than the currently executing priority stage.
        """
        if not isinstance(event, BaseEvent):
            raise TypeError(f"Expected BaseEvent instance, got {type(event).__name__}")

        # Validate and normalize priority
        try:
            p_val = int(priority)
        except (ValueError, TypeError) as e:
            raise InvalidEventPriorityError(f"Invalid priority value: {priority}") from e

        if p_val < 1 or p_val > 8:
            raise InvalidEventPriorityError(
                f"Priority must be between 1 and 8 (got {p_val})"
            )

        # Validate UTC timestamp
        event_time = validate_utc_timestamp(event.timestamp)

        # Enforce causality against the simulation clock
        if self._clock.is_initialized:
            clock_time = self._clock.now()
            if event_time < clock_time:
                raise TemporalOrderViolationError(
                    f"Temporal causality violation: Cannot enqueue event at past timestamp "
                    f"{event_time.isoformat()} when simulation clock is at {clock_time.isoformat()}"
                )

        # Enforce non-retroactive dispatch constraint at the current timestamp
        if (
            self._current_dispatch_timestamp is not None
            and event_time == self._current_dispatch_timestamp
            and self._current_dispatch_priority is not None
            and p_val < self._current_dispatch_priority
        ):
            raise RetroactiveEventError(
                f"Retroactive event violation: Cannot enqueue priority {p_val} event at timestamp "
                f"{event_time.isoformat()} because priority {self._current_dispatch_priority} "
                f"has already been dispatched or is currently executing."
            )

        # Generate monotonically increasing sequence ID
        seq_id = self._sequence_counter
        self._sequence_counter += 1

        prioritized = PrioritizedEvent(
            timestamp=event_time,
            priority=p_val,
            sequence_id=seq_id,
            event=event,
        )
        heapq.heappush(self._heap, prioritized)
        return prioritized

    def peek(self) -> PrioritizedEvent | None:
        """Inspect the next event to be dispatched without popping it.

        Returns:
            The highest-priority PrioritizedEvent, or None if the queue is empty.
        """
        if not self._heap:
            return None
        return self._heap[0]

    def pop(self) -> PrioritizedEvent:
        """Pop and return the highest-priority event from the queue.

        Advances the SimulationClock to the event's timestamp and updates
        the active dispatch stage tracking.

        Returns:
            The PrioritizedEvent with the lowest (timestamp, priority, sequence_id).

        Raises:
            EmptyQueueError: If the queue is empty.
        """
        if not self._heap:
            raise EmptyQueueError("Cannot pop from an empty EventQueue")

        item = heapq.heappop(self._heap)

        # Advance clock monotonically
        self._clock.advance_to(item.timestamp)

        # Track active dispatch milestone
        if self._current_dispatch_timestamp != item.timestamp:
            self._current_dispatch_timestamp = item.timestamp
            self._current_dispatch_priority = item.priority
        else:
            self._current_dispatch_priority = item.priority

        return item

    def clear(self) -> None:
        """Clear all pending events from the queue."""
        self._heap.clear()
