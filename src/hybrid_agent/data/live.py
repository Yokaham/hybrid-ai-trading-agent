"""Live market data infrastructure, streaming aggregator, and connection lifecycle."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum

from hybrid_agent.core.models import (
    Bar,
    DataProvenance,
    DataSourceType,
    Quote,
    Tick,
    Timeframe,
    validate_utc_timestamp,
)
from hybrid_agent.data.base import LiveMarketDataProvider

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """Lifecycle status of a live market data connection."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    FAILED = "FAILED"


class LiveFreshnessTracker:
    """Tracks recency of incoming market telemetry to detect stale data."""

    def __init__(self, default_max_staleness_seconds: float = 30.0) -> None:
        self.default_max_staleness_seconds = default_max_staleness_seconds
        self._last_seen: dict[str, datetime] = {}

    def update(self, symbol: str, timestamp: datetime) -> None:
        """Records arrival of valid quote/tick for a symbol."""
        utc_ts = validate_utc_timestamp(timestamp)
        self._last_seen[symbol.upper()] = utc_ts

    def get_last_seen(self, symbol: str) -> datetime | None:
        return self._last_seen.get(symbol.upper())

    def is_stale(
        self,
        symbol: str,
        current_time: datetime | None = None,
        max_staleness_seconds: float | None = None,
    ) -> bool:
        """Evaluates whether market quotes for the symbol exceed staleness thresholds."""
        last = self.get_last_seen(symbol)
        if last is None:
            return True
        now = validate_utc_timestamp(current_time) if current_time else datetime.now(UTC)
        threshold = max_staleness_seconds or self.default_max_staleness_seconds
        return (now - last).total_seconds() > threshold


class TickDeduplicator:
    """Filters duplicate or out-of-order streaming tick events."""

    def __init__(self) -> None:
        self._last_timestamps: dict[str, datetime] = {}

    def is_valid_sequence(self, tick: Tick) -> bool:
        """Returns True if the tick is strictly later than the last processed tick for the symbol."""
        last_ts = self._last_timestamps.get(tick.symbol)
        if last_ts is not None and tick.timestamp <= last_ts:
            return False
        self._last_timestamps[tick.symbol] = tick.timestamp
        return True


class LiveCandleAggregator:
    """
    Incrementally aggregates incoming raw ticks or quotes into discrete time-bucketed OHLCV bars.
    When a tick crosses a timeframe boundary, seals the completed bar and dispatches it.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: Timeframe,
        on_bar_completed: Callable[[Bar], None] | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.on_bar_completed = on_bar_completed

        self._current_bar_start: datetime | None = None
        self._current_bar_end: datetime | None = None
        self._open: float | None = None
        self._high: float | None = None
        self._low: float | None = None
        self._close: float | None = None
        self._volume: float = 0.0

    def _align_interval_start(self, ts: datetime) -> datetime:
        """Aligns a timestamp to the floor boundary of the timeframe."""
        step = self.timeframe.seconds
        epoch_seconds = int(ts.timestamp())
        floored_seconds = (epoch_seconds // step) * step
        return datetime.fromtimestamp(floored_seconds, tz=UTC)

    def process_tick(self, tick: Tick) -> Bar | None:
        """
        Ingests a streaming tick.
        If the tick spans past the current bar end time, seals and returns the completed bar.
        """
        if tick.symbol.upper() != self.symbol:
            raise ValueError(
                f"Tick symbol '{tick.symbol}' does not match aggregator '{self.symbol}'"
            )

        price = (tick.bid + tick.ask) / 2.0
        volume = (tick.bid_volume + tick.ask_volume) / 2.0
        tick_time = tick.timestamp

        completed_bar: Bar | None = None

        if self._current_bar_start is None or self._current_bar_end is None:
            # Initialize first candle boundary
            self._current_bar_start = self._align_interval_start(tick_time)
            self._current_bar_end = self._current_bar_start + self.timeframe.timedelta
            self._open = price
            self._high = price
            self._low = price
            self._close = price
            self._volume = volume
        elif tick_time >= self._current_bar_end:
            # Seal completed candle
            assert self._current_bar_start is not None
            assert self._current_bar_end is not None
            assert self._open is not None
            assert self._high is not None
            assert self._low is not None
            assert self._close is not None

            completed_bar = Bar(
                symbol=self.symbol,
                start_time=self._current_bar_start,
                end_time=self._current_bar_end,
                open=self._open,
                high=self._high,
                low=self._low,
                close=self._close,
                volume=self._volume,
                timeframe=self.timeframe,
                is_synthetic=False,
                provenance=tick.provenance,
            )

            if self.on_bar_completed:
                self.on_bar_completed(completed_bar)

            # Roll to new candle
            self._current_bar_start = self._align_interval_start(tick_time)
            self._current_bar_end = self._current_bar_start + self.timeframe.timedelta
            self._open = price
            self._high = price
            self._low = price
            self._close = price
            self._volume = volume
        else:
            # Update developing candle
            assert self._high is not None
            assert self._low is not None
            self._high = max(self._high, price)
            self._low = min(self._low, price)
            self._close = price
            self._volume += volume

        return completed_bar


class MockLiveMarketDataProvider(LiveMarketDataProvider):
    """Deterministic simulated live data provider for testing without external networks."""

    def __init__(self, initial_quotes: dict[str, tuple[float, float]] | None = None) -> None:
        self._state = ConnectionState.DISCONNECTED
        self._subscribed: set[str] = set()
        self._quotes: dict[str, Quote] = {}
        self._ticks: dict[str, Tick] = {}
        self._reconnect_attempts = 0

        # Seed with initial quotes if provided (e.g. {'EURUSD': (1.0850, 1.0852)})
        if initial_quotes:
            now = datetime.now(UTC)
            for sym, (bid, ask) in initial_quotes.items():
                self._quotes[sym.upper()] = Quote(
                    symbol=sym.upper(),
                    timestamp=now,
                    bid=bid,
                    ask=ask,
                    provenance=DataProvenance(
                        source_name="mock_live",
                        source_type=DataSourceType.SIMULATED,
                        retrieved_at=now,
                    ),
                )

    @property
    def name(self) -> str:
        return "mock_live"

    @property
    def state(self) -> ConnectionState:
        return self._state

    def connect(self) -> None:
        self._state = ConnectionState.CONNECTED
        self._reconnect_attempts = 0

    def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    def simulate_disconnect(self) -> None:
        """Simulates an abrupt network disconnection."""
        self._state = ConnectionState.DISCONNECTED

    def simulate_reconnection(self) -> None:
        """Simulates exponential backoff reconnection."""
        self._state = ConnectionState.RECONNECTING
        self._reconnect_attempts += 1
        self._state = ConnectionState.CONNECTED

    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    def subscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed.add(s.upper())

    def unsubscribe(self, symbols: list[str]) -> None:
        for s in symbols:
            self._subscribed.discard(s.upper())

    def emit_quote(self, quote: Quote) -> None:
        """Simulates incoming quote broadcast."""
        self._quotes[quote.symbol.upper()] = quote

    def emit_tick(self, tick: Tick) -> None:
        """Simulates incoming tick broadcast."""
        self._ticks[tick.symbol.upper()] = tick

    def get_latest_quote(self, symbol: str) -> Quote | None:
        if not self.is_connected():
            return None
        return self._quotes.get(symbol.upper())

    def get_latest_tick(self, symbol: str) -> Tick | None:
        if not self.is_connected():
            return None
        return self._ticks.get(symbol.upper())
