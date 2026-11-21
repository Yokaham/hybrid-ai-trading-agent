"""Deterministic unit tests for YahooFinanceHistoricalProvider using mocked provider responses."""

import unittest
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from hybrid_agent.core.models import Timeframe, VolumeType
from hybrid_agent.data.base import ProviderConnectionError
from hybrid_agent.data.providers.yahoo import YahooFinanceHistoricalProvider


class TestYahooFinanceHistoricalProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = YahooFinanceHistoricalProvider()
        self.start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
        self.end = datetime(2025, 1, 5, 0, 0, 0, tzinfo=UTC)

    @patch("yfinance.Ticker")
    def test_preserves_exact_start_and_end_boundaries(self, mock_ticker_cls: MagicMock) -> None:
        """Proves that returned bars strictly fall within the requested [start, end) half-open interval."""
        dates = [
            datetime(2024, 12, 31, 0, 0, 0, tzinfo=UTC),  # Before start (should be filtered out)
            datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),  # Exactly at start (included)
            datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC),  # In range (included)
            datetime(2025, 1, 4, 23, 59, 0, tzinfo=UTC),  # Just before end (included)
            datetime(
                2025, 1, 5, 0, 0, 0, tzinfo=UTC
            ),  # Exactly at end (filtered out under [start, end))
            datetime(2025, 1, 6, 0, 0, 0, tzinfo=UTC),  # After end (filtered out)
        ]
        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 102.5, 103.0, 104.0],
                "High": [105.0, 106.0, 107.0, 107.5, 108.0, 109.0],
                "Low": [99.0, 100.0, 101.0, 101.5, 102.0, 103.0],
                "Close": [104.0, 105.0, 106.0, 106.5, 107.0, 108.0],
                "Volume": [1000.0, 1200.0, 1300.0, 1350.0, 1400.0, 1500.0],
            },
            index=pd.DatetimeIndex(dates),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        bars = self.provider.get_historical_bars("AAPL", Timeframe.D1, self.start, self.end)

        self.assertEqual(len(bars), 3)
        self.assertEqual(bars[0].start_time, datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC))
        self.assertEqual(bars[1].start_time, datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC))
        self.assertEqual(bars[2].start_time, datetime(2025, 1, 4, 23, 59, 0, tzinfo=UTC))
        # Ensure bar exactly at end (2025-01-05 00:00:00) is strictly excluded
        self.assertTrue(all(b.start_time < self.end for b in bars))

    @patch("yfinance.Ticker")
    def test_preserves_original_ohlcv_without_manufacturing_prices(
        self, mock_ticker_cls: MagicMock
    ) -> None:
        """Proves that prices and volumes are preserved exactly as returned by provider."""
        dates = [datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)]
        mock_df = pd.DataFrame(
            {
                "Open": [150.123],
                "High": [155.456],
                "Low": [149.789],
                "Close": [153.321],
                "Volume": [54321.0],
            },
            index=pd.DatetimeIndex(dates),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        bars = self.provider.get_historical_bars("AAPL", Timeframe.D1, self.start, self.end)

        self.assertEqual(len(bars), 1)
        b = bars[0]
        self.assertEqual(b.open, 150.123)
        self.assertEqual(b.high, 155.456)
        self.assertEqual(b.low, 149.789)
        self.assertEqual(b.close, 153.321)
        self.assertEqual(b.volume, 54321.0)
        self.assertEqual(b.volume_type, VolumeType.REAL_TRADED)

    @patch("yfinance.Ticker")
    def test_discards_invalid_or_negative_rows_without_modifying_data(
        self, mock_ticker_cls: MagicMock
    ) -> None:
        """Proves that invalid prices (H < L) or negative volume are discarded, never silently clamped."""
        dates = [
            datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),  # Valid
            datetime(2025, 1, 2, 0, 0, 0, tzinfo=UTC),  # Invalid: High < Low
            datetime(2025, 1, 3, 0, 0, 0, tzinfo=UTC),  # Invalid: negative volume
            datetime(2025, 1, 4, 0, 0, 0, tzinfo=UTC),  # Invalid: non-positive price
        ]
        mock_df = pd.DataFrame(
            {
                "Open": [100.0, 100.0, 100.0, 0.0],
                "High": [105.0, 90.0, 105.0, 105.0],  # row 1: H(90) < L(95)
                "Low": [95.0, 95.0, 95.0, 95.0],
                "Close": [102.0, 98.0, 102.0, 102.0],
                "Volume": [1000.0, 1000.0, -500.0, 1000.0],  # row 2: negative vol
            },
            index=pd.DatetimeIndex(dates),
        )
        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        bars = self.provider.get_historical_bars("AAPL", Timeframe.D1, self.start, self.end)

        # Only row 0 is valid and returned; rows 1, 2, 3 were rejected without manufacturing fake numbers
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].start_time, datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC))

    @patch("yfinance.Ticker")
    def test_propagates_provider_failure_as_typed_error(self, mock_ticker_cls: MagicMock) -> None:
        """Proves that yfinance network/timeout exceptions raise ProviderConnectionError."""
        mock_instance = MagicMock()
        mock_instance.history.side_effect = ConnectionError("HTTP 500 Connection reset by peer")
        mock_ticker_cls.return_value = mock_instance

        with self.assertRaises(ProviderConnectionError) as ctx:
            self.provider.get_historical_bars("AAPL", Timeframe.D1, self.start, self.end)

        self.assertIn("Yahoo Finance query failure", str(ctx.exception))

    @patch("yfinance.Ticker")
    def test_distinguishes_empty_valid_response(self, mock_ticker_cls: MagicMock) -> None:
        """Proves that a valid empty response returns an empty list without raising an error."""
        mock_instance = MagicMock()
        mock_instance.history.return_value = pd.DataFrame()  # Empty dataframe
        mock_ticker_cls.return_value = mock_instance

        bars = self.provider.get_historical_bars("AAPL", Timeframe.D1, self.start, self.end)
        self.assertEqual(bars, [])


if __name__ == "__main__":
    unittest.main()
