"""Unit tests for historical market data ingestion pipeline."""

import shutil
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from hybrid_agent.core.models import Bar, Timeframe
from hybrid_agent.data.base import HistoricalMarketDataProvider
from hybrid_agent.data.pipeline import HistoricalDataPipeline
from hybrid_agent.data.providers.mock import MockHistoricalDataProvider
from hybrid_agent.data.storage import ParquetStorageEngine


class FailingHistoricalProvider(HistoricalMarketDataProvider):
    """Simulates network or provider timeout errors."""

    @property
    def name(self) -> str:
        return "failing_mock"

    def is_available(self) -> bool:
        return False

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        raise ConnectionError("Remote historical API timeout: connection refused.")


class TestHistoricalIngestionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ParquetStorageEngine(base_dir=self.temp_dir)
        self.provider = MockHistoricalDataProvider(seed=123, initial_price=150.0)
        self.pipeline = HistoricalDataPipeline(provider=self.provider, storage=self.storage)
        self.base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_successful_ingestion(self) -> None:
        end_time = self.base_time + timedelta(days=10)
        result = self.pipeline.ingest(
            symbol="MSFT",
            timeframe=Timeframe.D1,
            start=self.base_time,
            end=end_time,
        )
        self.assertTrue(result.success)
        self.assertEqual(result.bars_retrieved, 10)
        self.assertEqual(result.bars_stored, 10)
        self.assertIsNotNone(result.storage_path)

        # Read back from storage to verify
        stored_bars = self.storage.read_bars("MSFT", Timeframe.D1, source_name="mock")
        self.assertEqual(len(stored_bars), 10)

    def test_incremental_ingestion_without_duplicates(self) -> None:
        # Step 1: Ingest Days 0 to 5
        t5 = self.base_time + timedelta(days=5)
        res1 = self.pipeline.ingest("MSFT", Timeframe.D1, self.base_time, t5)
        self.assertTrue(res1.success)
        self.assertEqual(res1.bars_stored, 5)

        # Step 2: Ingest Days 3 to 10 (overlap from day 3 to 5)
        t10 = self.base_time + timedelta(days=10)
        t3 = self.base_time + timedelta(days=3)
        res2 = self.pipeline.ingest("MSFT", Timeframe.D1, t3, t10)
        self.assertTrue(res2.success)

        # Verify storage: Exactly 10 total bars, zero duplicates
        all_bars = self.storage.read_bars("MSFT", Timeframe.D1, source_name="mock")
        self.assertEqual(len(all_bars), 10)

    def test_provider_failure_handled_gracefully(self) -> None:
        failing_provider = FailingHistoricalProvider()
        pipeline = HistoricalDataPipeline(provider=failing_provider, storage=self.storage)
        result = pipeline.ingest(
            symbol="MSFT",
            timeframe=Timeframe.D1,
            start=self.base_time,
            end=self.base_time + timedelta(days=5),
        )
        self.assertFalse(result.success)
        self.assertEqual(result.bars_stored, 0)
        self.assertIn("Remote historical API timeout", str(result.error_message))

    def test_invalid_data_never_persisted(self) -> None:
        """Proves that datasets with hard integrity violations are never written to disk."""
        corrupt_bar = Bar(
            symbol="MSFT",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(days=1),
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=100.0,
            timeframe=Timeframe.D1,
        )

        class CorruptProvider(HistoricalMarketDataProvider):
            @property
            def name(self) -> str:
                return "corrupt"

            def is_available(self) -> bool:
                return True

            def get_historical_bars(
                self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
            ) -> list[Bar]:
                # Force an invalid bar via object.__setattr__ to bypass Pydantic model validator
                bad_bar = Bar(
                    symbol="MSFT",
                    start_time=start + timedelta(days=1),
                    end_time=start + timedelta(days=2),
                    open=100.0,
                    high=110.0,
                    low=90.0,
                    close=95.0,
                    volume=50.0,
                    timeframe=Timeframe.D1,
                )
                object.__setattr__(bad_bar, "high", 80.0)  # Corrupt: high < low
                return [corrupt_bar, bad_bar]

        pipeline = HistoricalDataPipeline(provider=CorruptProvider(), storage=self.storage)
        result = pipeline.ingest(
            "MSFT", Timeframe.D1, self.base_time, self.base_time + timedelta(days=2)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.bars_stored, 0)
        self.assertIn("Data quality validation failed", str(result.error_message))
        self.assertIsNone(result.storage_path)

        # Confirm storage file was NEVER created
        stored = self.storage.read_bars("MSFT", Timeframe.D1, source_name="corrupt")
        self.assertEqual(len(stored), 0)

    def test_wrong_symbols_and_timeframes_rejected(self) -> None:
        """Proves that bars with wrong symbol or timeframe are rejected before storage."""

        class MismatchedProvider(HistoricalMarketDataProvider):
            @property
            def name(self) -> str:
                return "mismatch"

            def is_available(self) -> bool:
                return True

            def get_historical_bars(
                self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
            ) -> list[Bar]:
                return [
                    Bar(
                        symbol="WRONG_SYMBOL",
                        start_time=start,
                        end_time=start + timedelta(days=1),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=100.0,
                        volume=100.0,
                        timeframe=Timeframe.D1,
                    )
                ]

        pipeline = HistoricalDataPipeline(provider=MismatchedProvider(), storage=self.storage)
        result = pipeline.ingest(
            "MSFT", Timeframe.D1, self.base_time, self.base_time + timedelta(days=1)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.bars_stored, 0)
        self.assertIn("symbol mismatch", str(result.error_message))

    def test_out_of_range_records_rejected(self) -> None:
        """Proves that bars prior to requested start or after requested end are rejected."""

        class OutOfRangeProvider(HistoricalMarketDataProvider):
            @property
            def name(self) -> str:
                return "out_of_range"

            def is_available(self) -> bool:
                return True

            def get_historical_bars(
                self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
            ) -> list[Bar]:
                return [
                    Bar(
                        symbol="MSFT",
                        start_time=start - timedelta(days=10),  # Prior to start
                        end_time=start - timedelta(days=9),
                        open=100.0,
                        high=105.0,
                        low=95.0,
                        close=100.0,
                        volume=100.0,
                        timeframe=Timeframe.D1,
                    )
                ]

        pipeline = HistoricalDataPipeline(provider=OutOfRangeProvider(), storage=self.storage)
        result = pipeline.ingest(
            "MSFT", Timeframe.D1, self.base_time, self.base_time + timedelta(days=1)
        )

        self.assertFalse(result.success)
        self.assertEqual(result.bars_stored, 0)
        self.assertIn("earlier than requested start", str(result.error_message))

    def test_empty_provider_response_handled_distinctly(self) -> None:
        """Proves that empty provider responses are flagged distinctly without reporting false success."""

        class EmptyProvider(HistoricalMarketDataProvider):
            @property
            def name(self) -> str:
                return "empty"

            def is_available(self) -> bool:
                return True

            def get_historical_bars(
                self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
            ) -> list[Bar]:
                return []

        pipeline = HistoricalDataPipeline(provider=EmptyProvider(), storage=self.storage)
        result = pipeline.ingest(
            "MSFT", Timeframe.D1, self.base_time, self.base_time + timedelta(days=1)
        )

        self.assertFalse(result.success)
        self.assertTrue(result.is_empty)
        self.assertEqual(result.bars_stored, 0)
        self.assertEqual(result.bars_retrieved, 0)
        self.assertIn("Empty provider response", str(result.error_message))

    def test_market_closure_gaps_do_not_fail_valid_dataset(self) -> None:
        """Proves that expected weekend or holiday gaps are tracked as suspected gaps without rejecting valid data."""
        t_friday = datetime(2026, 1, 2, 0, 0, 0, tzinfo=UTC)  # Friday
        t_monday = datetime(2026, 1, 5, 0, 0, 0, tzinfo=UTC)  # Monday (3-day jump)

        bars = [
            Bar(
                symbol="MSFT",
                start_time=t_friday,
                end_time=t_friday + timedelta(days=1),
                open=100.0,
                high=105.0,
                low=95.0,
                close=102.0,
                volume=1000.0,
                timeframe=Timeframe.D1,
            ),
            Bar(
                symbol="MSFT",
                start_time=t_monday,
                end_time=t_monday + timedelta(days=1),
                open=102.0,
                high=107.0,
                low=101.0,
                close=105.0,
                volume=1200.0,
                timeframe=Timeframe.D1,
            ),
        ]

        class WeekendProvider(HistoricalMarketDataProvider):
            @property
            def name(self) -> str:
                return "weekend"

            def is_available(self) -> bool:
                return True

            def get_historical_bars(
                self, symbol: str, timeframe: Timeframe, start: datetime, end: datetime
            ) -> list[Bar]:
                return bars

        pipeline = HistoricalDataPipeline(provider=WeekendProvider(), storage=self.storage)
        result = pipeline.ingest("MSFT", Timeframe.D1, t_friday, t_monday + timedelta(days=1))

        self.assertTrue(result.success)
        self.assertEqual(result.bars_stored, 2)
        self.assertEqual(len(result.missing_intervals), 1)
        self.assertIn("suspected_gaps", result.diagnostics)


if __name__ == "__main__":
    unittest.main()
