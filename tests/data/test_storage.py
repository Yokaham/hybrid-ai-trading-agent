"""Unit tests for Parquet partitioned storage and DuckDB analytical queries."""

import shutil
import tempfile
import unittest
from datetime import UTC, datetime, timedelta

from hybrid_agent.core.models import Bar, DataProvenance, DataSourceType, Timeframe
from hybrid_agent.data.storage import DuckDBAnalyticalEngine, ParquetStorageEngine


def create_bars_sequence(symbol: str, count: int, start: datetime) -> list[Bar]:
    bars: list[Bar] = []
    for i in range(count):
        t0 = start + timedelta(days=i)
        t1 = t0 + timedelta(days=1)
        bars.append(
            Bar(
                symbol=symbol,
                start_time=t0,
                end_time=t1,
                open=100.0 + i,
                high=105.0 + i,
                low=95.0 + i,
                close=102.0 + i,
                volume=1000.0 * (i + 1),
                timeframe=Timeframe.D1,
                provenance=DataProvenance(
                    source_name="test_source",
                    source_type=DataSourceType.HISTORICAL,
                    retrieved_at=datetime.now(UTC),
                ),
            )
        )
    return bars


class TestStorageAndAnalytics(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.storage = ParquetStorageEngine(base_dir=self.temp_dir)
        self.analytics = DuckDBAnalyticalEngine(db_path=":memory:")
        self.base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def tearDown(self) -> None:
        self.analytics.close()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parquet_write_and_read_consistency(self) -> None:
        bars = create_bars_sequence("AAPL", 5, self.base_time)
        file_path = self.storage.write_bars(bars, source_name="test_source")
        self.assertTrue(file_path.exists())

        # Read back all bars
        read_bars = self.storage.read_bars("AAPL", Timeframe.D1, source_name="test_source")
        self.assertEqual(len(read_bars), 5)
        self.assertEqual(read_bars[0].symbol, "AAPL")
        self.assertEqual(read_bars[0].close, 102.0)
        self.assertEqual(read_bars[-1].close, 106.0)

    def test_parquet_read_date_filtering(self) -> None:
        bars = create_bars_sequence("AAPL", 10, self.base_time)
        self.storage.write_bars(bars, source_name="test_source")

        filter_start = self.base_time + timedelta(days=3)
        filter_end = self.base_time + timedelta(days=6)
        filtered = self.storage.read_bars(
            "AAPL",
            Timeframe.D1,
            source_name="test_source",
            start=filter_start,
            end=filter_end,
        )
        self.assertEqual(len(filtered), 4)  # days 3, 4, 5, 6
        self.assertEqual(filtered[0].start_time, filter_start)

    def test_repeated_write_idempotency_and_deduplication(self) -> None:
        bars = create_bars_sequence("AAPL", 5, self.base_time)
        # Write first time
        self.storage.write_bars(bars, source_name="test_source")
        # Write exact same bars second time
        self.storage.write_bars(bars, source_name="test_source")

        read_bars = self.storage.read_bars("AAPL", Timeframe.D1, source_name="test_source")
        self.assertEqual(len(read_bars), 5)  # Must still be exactly 5, zero duplicates

    def test_duckdb_analytical_query(self) -> None:
        bars = create_bars_sequence("AAPL", 10, self.base_time)
        file_path = self.storage.write_bars(bars, source_name="test_source")

        # Query through DuckDB engine
        df = self.analytics.query_bars_from_parquet(file_path)
        self.assertEqual(len(df), 10)
        self.assertEqual(df["symbol"].iloc[0], "AAPL")

        # Summary statistics
        stats = self.analytics.compute_summary_statistics(file_path)
        self.assertEqual(stats["bar_count"].iloc[0], 10)
        self.assertEqual(stats["lowest_price"].iloc[0], 95.0)


if __name__ == "__main__":
    unittest.main()
