"""Unit tests for market data quality checks, gap detection, and quote validation."""

import unittest
from datetime import UTC, datetime, timedelta

from hybrid_agent.core.models import Bar, Quote, Timeframe
from hybrid_agent.data.quality import DataQualityValidator


def create_test_bar(
    timestamp: datetime,
    open_p: float = 100.0,
    high_p: float = 105.0,
    low_p: float = 95.0,
    close_p: float = 102.0,
    volume: float = 1000.0,
    symbol: str = "EURUSD",
    timeframe: Timeframe = Timeframe.D1,
) -> Bar:
    return Bar(
        symbol=symbol,
        start_time=timestamp,
        end_time=timestamp + timeframe.timedelta,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        volume=volume,
        timeframe=timeframe,
    )


class TestDataQualityValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_valid_bar_sequence_passes(self) -> None:
        bars = [create_test_bar(self.base_time + timedelta(days=i)) for i in range(5)]
        report = DataQualityValidator.validate_bars(bars, expected_timeframe=Timeframe.D1)
        self.assertTrue(report.is_valid)
        self.assertFalse(report.has_issues)
        self.assertEqual(report.total_bars, 5)
        self.assertEqual(report.duplicate_count, 0)
        self.assertEqual(len(report.missing_intervals), 0)

    def test_duplicate_bar_timestamps_detected(self) -> None:
        bars = [
            create_test_bar(self.base_time),
            create_test_bar(self.base_time),  # Duplicate
            create_test_bar(self.base_time + timedelta(days=1)),
        ]
        report = DataQualityValidator.validate_bars(bars, expected_timeframe=Timeframe.D1)
        self.assertFalse(report.is_valid)
        self.assertTrue(report.has_issues)
        self.assertEqual(report.duplicate_count, 1)

    def test_out_of_order_bars_detected(self) -> None:
        bars = [
            create_test_bar(self.base_time + timedelta(days=2)),
            create_test_bar(self.base_time + timedelta(days=1)),  # Out of order
        ]
        report = DataQualityValidator.validate_bars(bars, expected_timeframe=Timeframe.D1)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("Out-of-order" in v for v in report.violations))

    def test_missing_intervals_detected(self) -> None:
        # Day 0, Day 1, then jump to Day 4 (gap of 2 days)
        bars = [
            create_test_bar(self.base_time),
            create_test_bar(self.base_time + timedelta(days=1)),
            create_test_bar(self.base_time + timedelta(days=4)),
        ]
        report = DataQualityValidator.validate_bars(bars, expected_timeframe=Timeframe.D1)
        self.assertTrue(report.is_valid)  # Bars themselves are valid
        self.assertTrue(report.has_issues)  # But there is a missing interval
        self.assertEqual(len(report.missing_intervals), 1)

    def test_quote_validation_checks(self) -> None:
        valid_quote = Quote(
            symbol="EURUSD",
            timestamp=self.base_time,
            bid=1.0850,
            ask=1.0852,
            bid_size=10.0,
            ask_size=15.0,
        )
        self.assertEqual(len(DataQualityValidator.validate_quotes([valid_quote])), 0)

    def test_empty_bars_handling(self) -> None:
        report = DataQualityValidator.validate_bars([])
        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_bars, 0)


if __name__ == "__main__":
    unittest.main()
