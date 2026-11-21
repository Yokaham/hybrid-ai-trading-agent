"""Unified market data feed consuming historical and live data sources."""

from collections import defaultdict

import pandas as pd

from hybrid_agent.core.models import Bar


class MarketDataFeed:
    """
    Unified downstream data buffer.
    Serves as the common normalized interface consumed by feature engineering and research,
    regardless of whether data originated from Parquet storage, a live stream, or synthetic tests.
    """

    def __init__(self, max_buffer_size: int = 5000) -> None:
        self.max_buffer_size = max_buffer_size
        self._bars: dict[str, list[Bar]] = defaultdict(list)

    def load_historical_bars(self, bars: list[Bar]) -> None:
        """Loads a historical sequence of bars into the feed buffer."""
        for bar in bars:
            self.append_bar(bar)

    def append_bar(self, bar: Bar) -> None:
        """Appends a new sealed bar, maintaining temporal order and buffer bounds."""
        symbol = bar.symbol.upper()
        buf = self._bars[symbol]

        # Ignore exact duplicate timestamps
        if buf and buf[-1].start_time == bar.start_time:
            buf[-1] = bar  # Update with latest revision
            return

        buf.append(bar)
        if len(buf) > self.max_buffer_size:
            self._bars[symbol] = buf[-self.max_buffer_size :]

    def get_bars(self, symbol: str, count: int | None = None) -> list[Bar]:
        """Returns the most recent N bars for an instrument."""
        buf = self._bars.get(symbol.upper(), [])
        if count is None:
            return list(buf)
        return list(buf[-count:])

    def to_dataframe(self, symbol: str, count: int | None = None) -> pd.DataFrame:
        """
        Converts buffered bars for a symbol to a standard pandas DataFrame.
        Columns: ['start_time', 'end_time', 'open', 'high', 'low', 'close', 'volume']
        Index: DatetimeIndex in UTC.
        """
        bars = self.get_bars(symbol, count)
        if not bars:
            return pd.DataFrame(
                columns=["start_time", "end_time", "open", "high", "low", "close", "volume"]
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
            }
            for b in bars
        ]
        df = pd.DataFrame(records)
        df.set_index("start_time", inplace=True, drop=False)
        return df
