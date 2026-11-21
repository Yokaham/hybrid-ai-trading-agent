"""Unit tests for thread-safe MarketDataFeed multi-timeframe isolation and concurrency."""

import threading
import unittest
from datetime import UTC, datetime, timedelta

from hybrid_agent.core.models import Bar, Timeframe
from hybrid_agent.data.feed import MarketDataFeed


class TestMarketDataFeed(unittest.TestCase):
    def setUp(self) -> None:
        self.feed = MarketDataFeed(max_buffer_size=10)
        self.base_time = datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC)

    def test_multi_timeframe_isolation_for_same_symbol(self) -> None:
        """Proves that bars for different timeframes of the same symbol are stored in isolated buffers."""
        bar_d1 = Bar(
            symbol="EURUSD",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(days=1),
            open=1.0850,
            high=1.0900,
            low=1.0800,
            close=1.0870,
            volume=5000.0,
            timeframe=Timeframe.D1,
        )
        bar_h1 = Bar(
            symbol="EURUSD",
            start_time=self.base_time,
            end_time=self.base_time + timedelta(hours=1),
            open=1.0850,
            high=1.0860,
            low=1.0840,
            close=1.0855,
            volume=500.0,
            timeframe=Timeframe.H1,
        )

        self.feed.append_bar(bar_d1)
        self.feed.append_bar(bar_h1)

        d1_bars = self.feed.get_bars("EURUSD", timeframe=Timeframe.D1)
        h1_bars = self.feed.get_bars("EURUSD", timeframe=Timeframe.H1)

        self.assertEqual(len(d1_bars), 1)
        self.assertEqual(len(h1_bars), 1)
        self.assertEqual(d1_bars[0].timeframe, Timeframe.D1)
        self.assertEqual(h1_bars[0].timeframe, Timeframe.H1)
        self.assertEqual(d1_bars[0].close, 1.0870)
        self.assertEqual(h1_bars[0].close, 1.0855)

    def test_chronological_ordering_of_out_of_order_bars(self) -> None:
        """Proves that out-of-order bars are placed in strictly sorted chronological order."""
        t0 = self.base_time
        t1 = t0 + timedelta(hours=1)
        t2 = t0 + timedelta(hours=2)

        bar0 = Bar(
            symbol="EURUSD",
            start_time=t0,
            end_time=t0 + timedelta(hours=1),
            open=1.085,
            high=1.086,
            low=1.084,
            close=1.085,
            volume=100.0,
            timeframe=Timeframe.H1,
        )
        bar1 = Bar(
            symbol="EURUSD",
            start_time=t1,
            end_time=t1 + timedelta(hours=1),
            open=1.085,
            high=1.086,
            low=1.084,
            close=1.085,
            volume=100.0,
            timeframe=Timeframe.H1,
        )
        bar2 = Bar(
            symbol="EURUSD",
            start_time=t2,
            end_time=t2 + timedelta(hours=1),
            open=1.085,
            high=1.086,
            low=1.084,
            close=1.085,
            volume=100.0,
            timeframe=Timeframe.H1,
        )

        # Append in non-chronological order: bar0, then bar2, then late-arriving bar1
        self.feed.append_bar(bar0)
        self.feed.append_bar(bar2)
        self.feed.append_bar(bar1)

        bars = self.feed.get_bars("EURUSD", timeframe=Timeframe.H1)
        self.assertEqual(len(bars), 3)
        self.assertEqual(bars[0].start_time, t0)
        self.assertEqual(bars[1].start_time, t1)
        self.assertEqual(bars[2].start_time, t2)

    def test_duplicate_and_revision_handling(self) -> None:
        """Proves that an updated bar with the same timestamp replaces the existing bar in place."""
        t0 = self.base_time
        bar_initial = Bar(
            symbol="EURUSD",
            start_time=t0,
            end_time=t0 + timedelta(hours=1),
            open=1.085,
            high=1.086,
            low=1.084,
            close=1.0850,
            volume=100.0,
            timeframe=Timeframe.H1,
        )
        bar_revised = Bar(
            symbol="EURUSD",
            start_time=t0,
            end_time=t0 + timedelta(hours=1),
            open=1.085,
            high=1.088,
            low=1.084,
            close=1.0875,  # Revised price
            volume=150.0,
            timeframe=Timeframe.H1,
        )

        self.feed.append_bar(bar_initial)
        self.assertEqual(len(self.feed.get_bars("EURUSD", Timeframe.H1)), 1)

        # Revision with identical start_time
        self.feed.append_bar(bar_revised)
        bars = self.feed.get_bars("EURUSD", Timeframe.H1)
        self.assertEqual(len(bars), 1)
        self.assertEqual(bars[0].close, 1.0875)
        self.assertEqual(bars[0].high, 1.088)

    def test_buffer_size_limit_enforced(self) -> None:
        """Proves that buffer length never exceeds max_buffer_size."""
        feed = MarketDataFeed(max_buffer_size=5)
        for i in range(12):
            t = self.base_time + timedelta(hours=i)
            c = 1.085 + (i * 0.001)
            b = Bar(
                symbol="EURUSD",
                start_time=t,
                end_time=t + timedelta(hours=1),
                open=1.085,
                high=c + 0.005,
                low=1.080,
                close=c,
                volume=100.0,
                timeframe=Timeframe.H1,
            )
            feed.append_bar(b)

        bars = feed.get_bars("EURUSD", Timeframe.H1)
        self.assertEqual(len(bars), 5)
        self.assertEqual(bars[-1].start_time, self.base_time + timedelta(hours=11))

    def test_thread_safe_concurrent_access(self) -> None:
        """Proves thread-safety under concurrent reader and writer threads."""
        feed = MarketDataFeed(max_buffer_size=100)
        errors: list[Exception] = []

        def writer(offset: int) -> None:
            try:
                for i in range(50):
                    t = self.base_time + timedelta(minutes=offset * 50 + i)
                    b = Bar(
                        symbol="EURUSD",
                        start_time=t,
                        end_time=t + timedelta(minutes=1),
                        open=1.085,
                        high=1.086,
                        low=1.084,
                        close=1.085,
                        volume=10.0,
                        timeframe=Timeframe.M1,
                    )
                    feed.append_bar(b)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader() -> None:
            try:
                for _ in range(50):
                    _ = feed.get_bars("EURUSD", Timeframe.M1, count=10)
                    _ = feed.to_dataframe("EURUSD", Timeframe.M1, count=10)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(1,)),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
