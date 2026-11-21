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
    VolumeType,
    validate_utc_timestamp,
)
from hybrid_agent.data.base import HistoricalMarketDataProvider, ProviderConnectionError

logger = logging.getLogger(__name__)


class YahooFinanceHistoricalProvider(HistoricalMarketDataProvider):
    """
    Retrieves historical market data from Yahoo Finance via yfinance.

    Operational Characteristics & Limitations:
    - Freely accessible without an API key; suitable exclusively for exploratory prototyping and research.
    - Not institutional-grade; rate limits apply and underlying Yahoo endpoints are unofficial scraping APIs.
    - Intraday data retention: 1m data is restricted to the last 7 calendar days; 5m, 15m, and 1h intervals
      are restricted to the last 60 calendar days. Daily (1d) and weekly (1w) bars span multi-decade horizons.
    - End boundary semantics: For daily/weekly intervals, Yahoo Finance interprets the 'end' date parameter
      as strictly exclusive (e.g. end='2025-01-05' returns bars up to 2025-01-04). For intraday intervals,
      start and end datetimes can be passed with minute resolution.
    - Symbols: Equities use standard ticker symbols ('AAPL', 'MSFT'). Forex pairs use '=X' suffix ('EURUSD=X').
      Forex feeds from Yahoo Finance do not provide executed volume (volume=0.0).
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
        """
        Fetches and normalizes historical bars from Yahoo Finance.
        Preserves original provider OHLCV values without inventing or clamping numbers.
        Propagates provider failures through ProviderConnectionError.
        """
        start_utc = validate_utc_timestamp(start)
        end_utc = validate_utc_timestamp(end)

        if end_utc <= start_utc:
            raise ValueError(f"end ({end_utc}) must be later than start ({start_utc})")

        yf_symbol = self.format_symbol(symbol)
        yf_interval = self.map_timeframe(timeframe)

        # For intraday timeframes, pass full ISO datetime strings; for daily/weekly pass date strings
        is_intraday = timeframe in (Timeframe.M1, Timeframe.M5, Timeframe.M15, Timeframe.H1)
        start_param: str = (
            start_utc.strftime("%Y-%m-%d %H:%M:%S")
            if is_intraday
            else start_utc.strftime("%Y-%m-%d")
        )
        end_param: str = (
            end_utc.strftime("%Y-%m-%d %H:%M:%S") if is_intraday else end_utc.strftime("%Y-%m-%d")
        )

        try:
            ticker = yf.Ticker(yf_symbol)
            df: pd.DataFrame = ticker.history(
                interval=yf_interval,
                start=start_param,
                end=end_param,
                auto_adjust=False,
            )
        except Exception as exc:
            logger.error(
                "Failed to fetch historical data from Yahoo Finance for %s: %s", yf_symbol, exc
            )
            raise ProviderConnectionError(
                f"Yahoo Finance query failure for {yf_symbol}: {exc}"
            ) from exc

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
            is_synthetic=False,
            metadata={"raw_symbol": yf_symbol, "interval": yf_interval},
        )

        for ts, row in df.iterrows():
            bar_start = ts.to_pydatetime()
            bar_end = bar_start + step_delta

            # Strictly filter to requested time interval
            if bar_start < start_utc or bar_start > end_utc:
                continue

            open_p = float(row["Open"])
            high_p = float(row["High"])
            low_p = float(row["Low"])
            close_p = float(row["Close"])
            vol = float(row.get("Volume", 0.0))

            # Reject corrupt or non-positive price records without modifying or manufacturing prices
            if open_p <= 0 or high_p <= 0 or low_p <= 0 or close_p <= 0:
                logger.warning(
                    "Discarding non-positive price row from Yahoo for %s @ %s (O:%f, H:%f, L:%f, C:%f)",
                    yf_symbol,
                    bar_start,
                    open_p,
                    high_p,
                    low_p,
                    close_p,
                )
                continue

            # Reject negative volume records without silent clamping
            if vol < 0:
                logger.warning(
                    "Discarding negative volume row from Yahoo for %s @ %s (vol:%f)",
                    yf_symbol,
                    bar_start,
                    vol,
                )
                continue

            # Reject geometric price inversions without manufacturing fake high/low
            if (
                high_p < low_p
                or high_p < open_p
                or high_p < close_p
                or low_p > open_p
                or low_p > close_p
            ):
                logger.warning(
                    "Discarding geometrically invalid row from Yahoo for %s @ %s (O:%f, H:%f, L:%f, C:%f)",
                    yf_symbol,
                    bar_start,
                    open_p,
                    high_p,
                    low_p,
                    close_p,
                )
                continue

            vol_type = (
                VolumeType.UNAVAILABLE
                if (yf_symbol.endswith("=X") and vol == 0.0)
                else VolumeType.REAL_TRADED
            )

            bars.append(
                Bar(
                    symbol=symbol.upper(),
                    start_time=bar_start,
                    end_time=bar_end,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=vol,
                    volume_type=vol_type,
                    timeframe=timeframe,
                    is_synthetic=False,
                    provenance=prov,
                )
            )

        return bars
