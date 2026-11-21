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


if __name__ == "__main__":
    unittest.main()
