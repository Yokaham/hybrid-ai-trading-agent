"""Yahoo Finance historical market data provider adapter."""

import logging
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

from hybrid_agent.core.models import (
    Bar,
    DataProvenance,
    DataSourceType,
    Timeframe,
    validate_utc_timestamp,
)
from hybrid_agent.data.base import HistoricalMarketDataProvider

logger = logging.getLogger(__name__)


class YahooFinanceHistoricalProvider(HistoricalMarketDataProvider):
    """
    Retrieves historical market data from Yahoo Finance via yfinance.

    Operational Characteristics & Limitations:
    - Freely accessible without an API key; suitable for prototyping and research.
    - Not institutional-grade; rate limits apply and underlying Yahoo endpoints are unofficial.
    - Intraday data limited to 60 days (for 1m/5m/15m/1h). Daily/weekly data available across long horizons.
    - Symbols: Equities use standard tickers ('AAPL', 'MSFT'). Forex pairs use '=X' suffix ('EURUSD=X').
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "yahoo"

    def is_available(self) -> bool:
        """Tests connectivity with a minimal query."""
        try:
            ticker = yf.Ticker("AAPL")
            info = ticker.history(period="1d")
            return not info.empty
        except Exception:  # noqa: BLE001
            return False

    def format_symbol(self, symbol: str) -> str:
        """Maps internal symbols to Yahoo Finance conventions."""
        cleaned = symbol.strip().upper()
        # If 6-letter FX pair like EURUSD, map to EURUSD=X
        if len(cleaned) == 6 and cleaned.isalpha() and not cleaned.endswith("=X"):
            # Common currency abbreviations
            common_fx = {"EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"}
            if cleaned[:3] in common_fx and cleaned[3:] in common_fx:
                return f"{cleaned}=X"
        return cleaned

    def map_timeframe(self, timeframe: Timeframe) -> str:
        """Maps internal Timeframe enum to Yahoo Finance interval string."""
        mapping = {
            Timeframe.M1: "1m",
            Timeframe.M5: "5m",
            Timeframe.M15: "15m",
            Timeframe.H1: "1h",
            Timeframe.D1: "1d",
            Timeframe.W1: "1wk",
        }
        if timeframe not in mapping:
            raise ValueError(f"Timeframe {timeframe.value} is not supported by Yahoo Finance")
        return mapping[timeframe]

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Fetches and normalizes historical bars from Yahoo Finance."""
        start_utc = validate_utc_timestamp(start)
        end_utc = validate_utc_timestamp(end)

        if end_utc <= start_utc:
            raise ValueError(f"end ({end_utc}) must be later than start ({start_utc})")

        yf_symbol = self.format_symbol(symbol)
        yf_interval = self.map_timeframe(timeframe)

        try:
            ticker = yf.Ticker(yf_symbol)
            # yfinance history start/end can be ISO strings or datetime
            df: pd.DataFrame = ticker.history(
                interval=yf_interval,
                start=start_utc.strftime("%Y-%m-%d"),
                end=end_utc.strftime("%Y-%m-%d"),
                auto_adjust=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to fetch historical data from Yahoo Finance for %s: %s", yf_symbol, exc
            )
            return []

        if df.empty or "Open" not in df.columns:
            return []

        # Convert index to UTC datetime if timezone aware, or localize if naive
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        bars: list[Bar] = []
        step_delta = timeframe.timedelta
        prov = DataProvenance(
            source_name=self.name,
            source_type=DataSourceType.HISTORICAL,
            retrieved_at=datetime.now(UTC),
            metadata={"raw_symbol": yf_symbol, "interval": yf_interval},
        )

        for ts, row in df.iterrows():
            bar_start = ts.to_pydatetime()
            bar_end = bar_start + step_delta

            open_p = float(row["Open"])
            high_p = float(row["High"])
            low_p = float(row["Low"])
            close_p = float(row["Close"])
            vol = float(row.get("Volume", 0.0))

            # Filter out corrupt zero-price rows
            if open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0:
                continue

            # Ensure high is >= low and >= open, close
            high_p = max(high_p, open_p, close_p)
            low_p = min(low_p, open_p, close_p)

            bars.append(
                Bar(
                    symbol=symbol.upper(),
                    start_time=bar_start,
                    end_time=bar_end,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=max(0.0, vol),
                    timeframe=timeframe,
                    is_synthetic=False,
                    provenance=prov,
                )
            )

        return bars
