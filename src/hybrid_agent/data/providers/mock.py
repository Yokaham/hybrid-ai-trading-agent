"""Deterministic mock historical data provider for offline testing and simulations."""

import random
from datetime import UTC, datetime, timedelta

from hybrid_agent.core.models import (
    Bar,
    DataProvenance,
    DataSourceType,
    Timeframe,
    validate_utc_timestamp,
)
from hybrid_agent.data.base import HistoricalMarketDataProvider


class MockHistoricalDataProvider(HistoricalMarketDataProvider):
    """Generates deterministic, mathematically consistent synthetic OHLCV bars."""

    def __init__(
        self,
        seed: int = 42,
        initial_price: float = 100.0,
        volatility: float = 0.01,
        drift: float = 0.0001,
    ) -> None:
        self._seed = seed
        self._initial_price = initial_price
        self._volatility = volatility
        self._drift = drift

    @property
    def name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Generates deterministic synthetic bars between start and end timestamps."""
        start_utc = validate_utc_timestamp(start)
        end_utc = validate_utc_timestamp(end)

        if end_utc <= start_utc:
            raise ValueError(f"end ({end_utc}) must be later than start ({start_utc})")

        rng = random.Random(self._seed + hash(symbol) + hash(timeframe.value))
        step_seconds = timeframe.seconds

        bars: list[Bar] = []
        curr_time = start_utc
        price = self._initial_price

        prov = DataProvenance(
            source_name=self.name,
            source_type=DataSourceType.SIMULATED,
            retrieved_at=datetime.now(UTC),
            is_synthetic=True,
            metadata={"seed": self._seed, "synthetic": True},
        )

        while curr_time < end_utc:
            bar_end = curr_time + timedelta(seconds=step_seconds)
            # Simulate random walk price movements
            ret = rng.gauss(self._drift, self._volatility)
            open_p = round(max(0.01, price), 5)
            close_p = round(max(0.01, price * (1.0 + ret)), 5)

            # High and low bounds
            wick_high = abs(rng.gauss(0, self._volatility * 0.5))
            wick_low = abs(rng.gauss(0, self._volatility * 0.5))
            high_p = round(max(open_p, close_p) * (1.0 + wick_high), 5)
            low_p = round(min(open_p, close_p) * (1.0 - wick_low), 5)

            # Ensure low is positive
            low_p = max(0.0001, low_p)

            # Volume
            vol = round(abs(rng.gauss(1000.0, 300.0)), 2)

            bars.append(
                Bar(
                    symbol=symbol.upper(),
                    start_time=curr_time,
                    end_time=bar_end,
                    open=open_p,
                    high=high_p,
                    low=low_p,
                    close=close_p,
                    volume=vol,
                    timeframe=timeframe,
                    is_synthetic=True,
                    provenance=prov,
                )
            )

            price = close_p
            curr_time = bar_end

        return bars
