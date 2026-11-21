"""Unit tests for FeaturePipeline, incremental consistency, no lookahead, and registry."""

import unittest
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from hybrid_agent.core.models import Bar, DataProvenance, DataSourceType, Timeframe
from hybrid_agent.features.pipeline import FeaturePipeline
from hybrid_agent.features.registry import FeatureRegistry


def generate_test_bars(n: int) -> list[Bar]:
    np.random.seed(42)
    base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    bars: list[Bar] = []
    price = 100.0

    for i in range(n):
        t0 = base_time + timedelta(days=i)
        t1 = t0 + timedelta(days=1)
        ret = np.random.normal(0.0005, 0.01)
        close_p = round(price * (1 + ret), 2)
        open_p = price
        high_p = round(max(open_p, close_p) * 1.005, 2)
        low_p = round(min(open_p, close_p) * 0.995, 2)
        vol = float(np.random.randint(1000, 5000))

        bars.append(
            Bar(
                symbol="EURUSD",
                start_time=t0,
                end_time=t1,
                open=open_p,
                high=high_p,
                low=low_p,
                close=close_p,
                volume=vol,
                timeframe=Timeframe.D1,
                provenance=DataProvenance(
                    source_name="test",
                    source_type=DataSourceType.SIMULATED,
                    retrieved_at=datetime.now(UTC),
                ),
            )
        )
        price = close_p
    return bars


class TestFeaturePipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = FeaturePipeline()
        self.bars = generate_test_bars(80)

    def test_batch_transform_dataframe(self) -> None:
        records = [
            {"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
            for b in self.bars
        ]
        df = pd.DataFrame(records)
        transformed = self.pipeline.transform(df)

        # Expected indicator columns are appended
        expected_cols = [
            "sma_20",
            "sma_50",
            "ema_20",
            "adx_14",
            "rsi_14",
            "macd_line",
            "atr_14",
            "bb_upper_20",
            "vol_sma_20",
            "rvol_20",
        ]
        for col in expected_cols:
            self.assertIn(col, transformed.columns)

    def test_incremental_versus_batch_consistency(self) -> None:
        """Verifies that streaming incremental calculations equal batch dataframe outputs."""
        history = self.bars[:60]
        latest_bar = self.bars[60]

        # 1. Incremental calculation
        inc_features = self.pipeline.transform_incremental(latest_bar, history)

        # 2. Batch calculation
        records = [
            {"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
            for b in (history + [latest_bar])
        ]
        df = pd.DataFrame(records)
        batch_df = self.pipeline.transform(df)
        batch_latest = batch_df.iloc[-1]

        # Compare common indicators
        for k in ["sma_20", "ema_20", "rsi_14", "atr_14", "bb_middle_20", "vol_sma_20"]:
            self.assertAlmostEqual(inc_features[k], batch_latest[k], places=5)

    def test_no_lookahead_leakage(self) -> None:
        """Proves that mutating future bars does not alter historical feature outputs."""
        bars_copy = list(self.bars)
        df_original = pd.DataFrame(
            [
                {"open": b.open, "high": b.high, "low": b.low, "close": b.close, "volume": b.volume}
                for b in bars_copy
            ]
        )
        res_original = self.pipeline.transform(df_original)

        # Drastically mutate future bars (index 70 onwards)
        df_mutated = df_original.copy()
        df_mutated.loc[70:, "close"] = df_mutated.loc[70:, "close"] * 5.0
        df_mutated.loc[70:, "high"] = df_mutated.loc[70:, "high"] * 5.0
        res_mutated = self.pipeline.transform(df_mutated)

        # Features before index 70 must be exactly identical
        for col in ["sma_20", "rsi_14", "macd_line", "atr_14"]:
            orig_vals = res_original[col].iloc[:65].dropna().values
            mut_vals = res_mutated[col].iloc[:65].dropna().values
            self.assertTrue(np.allclose(orig_vals, mut_vals, atol=1e-8))

    def test_feature_registry_catalog(self) -> None:
        reg = FeatureRegistry()
        FeaturePipeline(registry=reg)
        all_features = reg.list_all()
        self.assertGreater(len(all_features), 5)
        markdown = reg.generate_catalog_markdown()
        self.assertIn("# Feature Catalog", markdown)
        self.assertIn("RSI", markdown)


if __name__ == "__main__":
    unittest.main()
