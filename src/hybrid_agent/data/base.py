"""Market data provider interfaces and abstractions."""

from abc import ABC, abstractmethod
from datetime import datetime

from hybrid_agent.core.models import Bar, Quote, Tick, Timeframe


class MarketDataProvider(ABC):
    """Abstract base class for all market data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns provider identification name."""
        raise NotImplementedError


class HistoricalMarketDataProvider(MarketDataProvider):
    """Abstract interface defining required capabilities for historical market data retrieval."""

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """
        Retrieves historical OHLCV bars for the specified instrument and timeframe.
        start and end must be timezone-aware UTC datetimes with end > start.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Checks whether the historical data service is reachable and responsive."""
        raise NotImplementedError


class LiveMarketDataProvider(MarketDataProvider):
    """Abstract interface for streaming live or simulated current market data."""

    @abstractmethod
    def connect(self) -> None:
        """Establishes connection to the data feed."""
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Closes connection to the data feed cleanly."""
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the feed is active and connected."""
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, symbols: list[str]) -> None:
        """Subscribes to live quotes/ticks for the given symbols."""
        raise NotImplementedError

    @abstractmethod
    def unsubscribe(self, symbols: list[str]) -> None:
        """Unsubscribes from live quotes/ticks for the given symbols."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Quote | None:
        """Returns the most recent Quote snapshot for the symbol, or None if unavailable."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_tick(self, symbol: str) -> Tick | None:
        """Returns the most recent Tick for the symbol, or None if unavailable."""
        raise NotImplementedError
