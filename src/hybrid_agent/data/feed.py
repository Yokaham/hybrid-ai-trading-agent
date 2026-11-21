"""Unified thread-safe market data feed consuming historical and live data sources."""

import bisect
import threading
from collections import defaultdict

import pandas as pd

from hybrid_agent.core.models import Bar, Timeframe


class MarketDataFeed:
    """
    Unified downstream data buffer.
    Serves as the common normalized interface consumed by feature engineering and research,
    regardless of whether data originated from Parquet storage, a live stream, or synthetic tests.

    Thread Safety & Concurrency:
    - Thread-Safe: All read and write operations are synchronized using an internal re-entrant lock (RLock).
    - Multi-Timeframe Isolation: Buffers are strictly partitioned by both (symbol, timeframe).
    - Chronological Ordering: Incoming bars are maintained in strictly ascending chronological order.
      Out-of-order bars are inserted into their correct sorted position rather than appended out of sequence.
    - Explicit Revision Handling: If a bar arrives with an existing start_time, the existing bar is updated.
    - Buffer Bounds: Enforces max_buffer_size per (symbol, timeframe) partition.
    """

    def __init__(self, max_buffer_size: int = 5000) -> None:
        self.max_buffer_size = max_buffer_size
        self._bars: dict[tuple[str, Timeframe], list[Bar]] = defaultdict(list)
        self._lock = threading.RLock()

    def load_historical_bars(self, bars: list[Bar]) -> None:
        """Loads a historical sequence of bars into the feed buffer under thread safety."""
        with self._lock:
            for bar in bars:
                self.append_bar(bar)

    def append_bar(self, bar: Bar) -> None:
        """
        Appends or inserts a new sealed bar, maintaining temporal order and buffer bounds.
        Handles revisions in-place and places out-of-order bars in chronological order.
        """
        with self._lock:
            key = (bar.symbol.upper(), bar.timeframe)
            buf = self._bars[key]

            if not buf:
                buf.append(bar)
                return

            # Fast path: strictly newer bar
            if bar.start_time > buf[-1].start_time:
                buf.append(bar)
            elif bar.start_time == buf[-1].start_time:
                # Latest revision for current bar
                buf[-1] = bar
            else:
                # Out-of-order bar: binary search for sorted placement
                start_times = [b.start_time for b in buf]
                idx = bisect.bisect_left(start_times, bar.start_time)
                if idx < len(buf) and buf[idx].start_time == bar.start_time:
                    # In-place revision
                    buf[idx] = bar
                else:
                    buf.insert(idx, bar)

            # Enforce max buffer size
            if len(buf) > self.max_buffer_size:
                self._bars[key] = buf[-self.max_buffer_size :]

    def get_bars(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.D1,
        count: int | None = None,
    ) -> list[Bar]:
        """Returns the most recent N bars for an instrument and timeframe."""
        with self._lock:
            key = (symbol.upper(), timeframe)
            buf = self._bars.get(key, [])
            if count is None:
                return list(buf)
            return list(buf[-count:])

    def to_dataframe(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.D1,
        count: int | None = None,
    ) -> pd.DataFrame:
        """
        Converts buffered bars for a (symbol, timeframe) to a standard pandas DataFrame.
        Columns: ['start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume', 'volume_type']
        Index: DatetimeIndex in UTC.
        """
        with self._lock:
            bars = self.get_bars(symbol, timeframe, count)
            if not bars:
                return pd.DataFrame(
                    columns=[
                        "start_time",
                        "end_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "volume_type",
                    ]
                )

            records = [
                {
                    "start_time": b.start_time,
                    "end_time": b.end_time,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "volume_type": b.volume_type.value
                    if hasattr(b, "volume_type")
                    else "REAL_TRADED",
                }
                for b in bars
            ]
            df = pd.DataFrame(records)
            df = df.set_index("start_time", drop=False)
            return df

    def clear(self) -> None:
        """Clears all buffered bars across all instruments and timeframes."""
        with self._lock:
            self._bars.clear()
