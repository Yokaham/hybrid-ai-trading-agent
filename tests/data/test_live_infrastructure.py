"""Unit tests for live market data infrastructure, staleness, deduplication, and aggregation."""

import unittest
from datetime import UTC, datetime, timedelta

from hybrid_agent.core.models import (
    DataSourceType,
    Tick,
    Timeframe,
)
from hybrid_agent.data.live import (
    ConnectionState,
    LiveCandleAggregator,
    LiveFreshnessTracker,
    MockLiveMarketDataProvider,
    TickDeduplicator,
)


class TestLiveInfrastructure(unittest.TestCase):
    def setUp(self) -> None:
        self.base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    def test_mock_live_provider_lifecycle_and_provenance(self) -> None:
        provider = MockLiveMarketDataProvider(initial_quotes={"EURUSD": (1.0850, 1.0852)})
        self.assertFalse(provider.is_connected())
        self.assertEqual(provider.state, ConnectionState.DISCONNECTED)

        # Before connecting, quotes are None
        self.assertIsNone(provider.get_latest_quote("EURUSD"))

        # Connect
        provider.connect()
        self.assertTrue(provider.is_connected())
        quote = provider.get_latest_quote("EURUSD")
        self.assertIsNotNone(quote)
        assert quote is not None
        self.assertEqual(quote.bid, 1.0850)
        self.assertEqual(quote.provenance.source_type, DataSourceType.SIMULATED)  # type: ignore[union-attr]

        # Simulate disconnect and reconnect
        provider.simulate_disconnect()
        self.assertFalse(provider.is_connected())
        provider.simulate_reconnection()
        self.assertTrue(provider.is_connected())

        provider.disconnect()
        self.assertFalse(provider.is_connected())

    def test_live_freshness_tracker(self) -> None:
        tracker = LiveFreshnessTracker(default_max_staleness_seconds=10.0)
        self.assertTrue(tracker.is_stale("AAPL"))  # Never seen is stale

        t0 = self.base_time
        tracker.update("AAPL", t0)

        # 5 seconds later -> Fresh
        self.assertFalse(tracker.is_stale("AAPL", current_time=t0 + timedelta(seconds=5)))

        # 15 seconds later -> Stale
        self.assertTrue(tracker.is_stale("AAPL", current_time=t0 + timedelta(seconds=15)))

    def test_tick_deduplicator(self) -> None:
        dedup = TickDeduplicator()
        t0 = self.base_time
        t1 = t0 + timedelta(seconds=1)

        tick1 = Tick(symbol="EURUSD", timestamp=t0, bid=1.0850, ask=1.0852)
        tick_dup = Tick(symbol="EURUSD", timestamp=t0, bid=1.0850, ask=1.0852)
        tick_old = Tick(
            symbol="EURUSD", timestamp=t0 - timedelta(seconds=1), bid=1.0849, ask=1.0851
        )
        tick2 = Tick(symbol="EURUSD", timestamp=t1, bid=1.0851, ask=1.0853)

        self.assertTrue(dedup.is_valid_sequence(tick1))
        self.assertFalse(dedup.is_valid_sequence(tick_dup))  # Exact duplicate timestamp rejected
        self.assertFalse(dedup.is_valid_sequence(tick_old))  # Out-of-order tick rejected
        self.assertTrue(dedup.is_valid_sequence(tick2))  # Later timestamp accepted

    def test_live_candle_aggregator(self) -> None:
        completed_bars = []
        aggregator = LiveCandleAggregator(
            symbol="EURUSD",
            timeframe=Timeframe.M1,
            on_bar_completed=lambda b: completed_bars.append(b),
        )

        t0 = datetime(2026, 1, 1, 12, 0, 10, tzinfo=UTC)
        t1 = datetime(2026, 1, 1, 12, 0, 30, tzinfo=UTC)
        t2 = datetime(2026, 1, 1, 12, 0, 50, tzinfo=UTC)
        t_next = datetime(2026, 1, 1, 12, 1, 5, tzinfo=UTC)  # Crosses 1-minute boundary

        # Feed ticks in minute 12:00:00 -> 12:01:00
        aggregator.process_tick(
            Tick(symbol="EURUSD", timestamp=t0, bid=1.0850, ask=1.0852, bid_volume=1, ask_volume=1)
        )
        aggregator.process_tick(
            Tick(symbol="EURUSD", timestamp=t1, bid=1.0860, ask=1.0862, bid_volume=2, ask_volume=2)
        )
        aggregator.process_tick(
            Tick(symbol="EURUSD", timestamp=t2, bid=1.0840, ask=1.0842, bid_volume=1, ask_volume=1)
        )
        self.assertEqual(len(completed_bars), 0)  # Candle still forming

        # Tick in next minute -> Seals previous candle
        sealed = aggregator.process_tick(
            Tick(
                symbol="EURUSD",
                timestamp=t_next,
                bid=1.0855,
                ask=1.0857,
                bid_volume=1,
                ask_volume=1,
            )
        )
        self.assertIsNotNone(sealed)
        self.assertEqual(len(completed_bars), 1)

        bar = completed_bars[0]
        self.assertEqual(bar.symbol, "EURUSD")
        self.assertEqual(bar.timeframe, Timeframe.M1)
        self.assertAlmostEqual(bar.open, 1.0851)  # Mid price of tick1
        self.assertAlmostEqual(bar.high, 1.0861)  # Mid price of tick2
        self.assertAlmostEqual(bar.low, 1.0841)  # Mid price of tick3
        self.assertAlmostEqual(bar.close, 1.0841)  # Mid price of tick3


if __name__ == "__main__":
    unittest.main()
