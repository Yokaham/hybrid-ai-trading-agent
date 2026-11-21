"""Unit tests for technical indicators, warm-up validation, and missing values."""

import unittest

import numpy as np
import pandas as pd

from hybrid_agent.features.indicators import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_historical_volatility,
    compute_macd,
    compute_relative_volume,
    compute_roc,
    compute_rsi,
    compute_sma,
)


class TestIndicators(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        n = 100
        # Deterministic price series
        returns = np.random.normal(0.0005, 0.01, size=n)
        price = 100.0 * np.cumprod(1 + returns)
        self.close = pd.Series(price)
        self.high = self.close * (1 + np.abs(np.random.normal(0, 0.005, size=n)))
        self.low = self.close * (1 - np.abs(np.random.normal(0, 0.005, size=n)))
        self.volume = pd.Series(np.random.uniform(500, 2000, size=n))

    def test_sma_calculation_and_warmup(self) -> None:
        sma = compute_sma(self.close, period=10)
        # First 9 elements should be NaN (warmup)
        self.assertTrue(sma.iloc[:9].isna().all())
        self.assertFalse(sma.iloc[9:].isna().any())
        self.assertAlmostEqual(sma.iloc[9], self.close.iloc[:10].mean())

    def test_ema_calculation_and_warmup(self) -> None:
        ema = compute_ema(self.close, period=10)
        self.assertTrue(ema.iloc[:9].isna().all())
        self.assertFalse(ema.iloc[9:].isna().any())

    def test_rsi_bounds_and_warmup(self) -> None:
        rsi = compute_rsi(self.close, period=14)
        # Warmup period is 14
        self.assertTrue(rsi.iloc[:14].isna().all())
        valid_rsi = rsi.iloc[14:]
        self.assertFalse(valid_rsi.isna().any())
        self.assertTrue((valid_rsi >= 0.0).all())
        self.assertTrue((valid_rsi <= 100.0).all())

    def test_macd_relationship_and_warmup(self) -> None:
        macd_line, signal_line, hist = compute_macd(self.close, fast=12, slow=26, signal=9)
        # Histogram must equal macd_line - signal_line
        diff = (macd_line - signal_line) - hist
        self.assertTrue(np.allclose(diff.dropna(), 0.0, atol=1e-7))

    def test_adx_bounds_and_components(self) -> None:
        adx, _plus_di, _minus_di = compute_adx(self.high, self.low, self.close, period=14)
        valid_adx = adx.dropna()
        self.assertTrue((valid_adx >= 0.0).all())
        self.assertTrue((valid_adx <= 100.0).all())

    def test_atr_positive(self) -> None:
        atr = compute_atr(self.high, self.low, self.close, period=14)
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0.0).all())

    def test_bollinger_bands_ordering(self) -> None:
        upper, middle, lower, _pct_b, bandwidth = compute_bollinger_bands(
            self.close, period=20, num_std=2.0
        )
        valid_idx = upper.dropna().index
        # Upper >= Middle >= Lower
        self.assertTrue((upper.loc[valid_idx] >= middle.loc[valid_idx]).all())
        self.assertTrue((middle.loc[valid_idx] >= lower.loc[valid_idx]).all())
        self.assertTrue((bandwidth.loc[valid_idx] >= 0.0).all())

    def test_roc_calculation(self) -> None:
        roc = compute_roc(self.close, period=5)
        self.assertTrue(roc.iloc[:5].isna().all())
        expected = ((self.close.iloc[5] - self.close.iloc[0]) / self.close.iloc[0]) * 100.0
        self.assertAlmostEqual(roc.iloc[5], expected)

    def test_historical_volatility_positive(self) -> None:
        hv = compute_historical_volatility(self.close, period=20, trading_periods=252)
        valid_hv = hv.dropna()
        self.assertTrue((valid_hv > 0.0).all())

    def test_relative_volume(self) -> None:
        rvol = compute_relative_volume(self.volume, period=20)
        valid_rvol = rvol.dropna()
        self.assertTrue((valid_rvol > 0.0).all())

    def test_invalid_parameters_raise_error(self) -> None:
        with self.assertRaises(ValueError):
            compute_sma(self.close, period=0)
        with self.assertRaises(ValueError):
            compute_macd(self.close, fast=26, slow=12)  # fast >= slow


if __name__ == "__main__":
    unittest.main()
