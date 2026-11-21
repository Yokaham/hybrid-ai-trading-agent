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
        self.assertEqual(len(filtered), 3)  # days 3, 4, 5 under exclusive end [start, end)
        self.assertEqual(filtered[0].start_time, filter_start)
        self.assertTrue(all(b.start_time < filter_end for b in filtered))

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

    def test_provenance_preservation_across_all_source_types(self) -> None:
        """Proves that simulated data remains simulated, original timestamps and metadata are preserved."""
        retrieval_timestamp = datetime(2025, 6, 15, 14, 30, 0, tzinfo=UTC)
        prov = DataProvenance(
            source_name="sim_engine",
            source_type=DataSourceType.SIMULATED,
            retrieved_at=retrieval_timestamp,
            is_synthetic=True,
            metadata={"seed": 42, "generator": "gbm"},
        )
        bar = Bar(
            symbol="BTCUSD",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(days=1),
            open=50000.0,
            high=51000.0,
            low=49000.0,
            close=50500.0,
            volume=10.0,
            timeframe=Timeframe.D1,
            is_synthetic=True,
            provenance=prov,
        )

        file_path = self.storage.write_bars([bar], source_name="sim_engine")
        self.assertTrue(file_path.exists())

        # Read back and inspect provenance
        loaded_bars = self.storage.read_bars("BTCUSD", Timeframe.D1, source_name="sim_engine")
        self.assertEqual(len(loaded_bars), 1)
        loaded = loaded_bars[0]

        self.assertTrue(loaded.is_synthetic)
        self.assertIsNotNone(loaded.provenance)
        assert loaded.provenance is not None

        # Provenance source_type must remain strictly SIMULATED, NOT HISTORICAL
        self.assertEqual(loaded.provenance.source_type, DataSourceType.SIMULATED)
        self.assertTrue(loaded.provenance.is_synthetic)
        self.assertEqual(loaded.provenance.source_name, "sim_engine")
        self.assertEqual(loaded.provenance.retrieved_at, retrieval_timestamp)
        self.assertEqual(loaded.provenance.metadata.get("seed"), 42)
        self.assertEqual(loaded.provenance.metadata.get("generator"), "gbm")

    def test_duckdb_parquet_query_with_exclusive_end(self) -> None:
        """Proves DuckDB query_bars_from_parquet enforces the half-open interval [start, end)."""
        bars = create_bars_sequence("MSFT", 10, self.base_time)
        file_path = self.storage.write_bars(bars, source_name="test_source")

        filter_start = self.base_time + timedelta(days=2)
        filter_end = self.base_time + timedelta(days=5)
        df = self.analytics.query_bars_from_parquet(file_path, start=filter_start, end=filter_end)

        # Days 2, 3, 4 -> Exactly 3 bars (Day 5 is strictly excluded)
        self.assertEqual(len(df), 3)
        self.assertTrue((df["start_time"] >= filter_start).all())
        self.assertTrue((df["start_time"] < filter_end).all())

    def test_mixed_symbol_write_rejected(self) -> None:
        """Proves that attempting to write bars of different symbols in one batch raises ValueError."""
        bar1 = Bar(
            symbol="AAPL",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(days=1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=152.0,
            volume=100.0,
            timeframe=Timeframe.D1,
        )
        bar2 = Bar(
            symbol="MSFT",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(days=1),
            open=300.0,
            high=305.0,
            low=298.0,
            close=302.0,
            volume=100.0,
            timeframe=Timeframe.D1,
        )
        with self.assertRaises(ValueError) as ctx:
            self.storage.write_bars([bar1, bar2], source_name="test_source")
        self.assertIn("Mixed-symbol batch rejected", str(ctx.exception))

    def test_mixed_timeframe_write_rejected(self) -> None:
        """Proves that attempting to write bars of different timeframes in one batch raises ValueError."""
        bar1 = Bar(
            symbol="AAPL",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(days=1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=152.0,
            volume=100.0,
            timeframe=Timeframe.D1,
        )
        bar2 = Bar(
            symbol="AAPL",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(hours=1),
            open=150.0,
            high=151.0,
            low=149.5,
            close=150.5,
            volume=100.0,
            timeframe=Timeframe.H1,
        )
        with self.assertRaises(ValueError) as ctx:
            self.storage.write_bars([bar1, bar2], source_name="test_source")
        self.assertIn("Mixed-timeframe batch rejected", str(ctx.exception))

    def test_duckdb_path_with_spaces_and_quotes_handled_safely(self) -> None:
        """Proves that file paths with spaces are queried safely without SQL syntax errors."""
        special_dir = tempfile.mkdtemp(prefix="market data space ")
        try:
            special_storage = ParquetStorageEngine(base_dir=special_dir)
            bars = create_bars_sequence("GOOG", 3, self.base_time)
            file_path = special_storage.write_bars(bars, source_name="test_source")

            df = self.analytics.query_bars_from_parquet(file_path)
            self.assertEqual(len(df), 3)
            self.assertEqual(df["symbol"].iloc[0], "GOOG")
        finally:
            shutil.rmtree(special_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
