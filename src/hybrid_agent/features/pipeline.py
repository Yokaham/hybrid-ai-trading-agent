"""Feature engineering pipeline orchestrating technical indicator transformations."""

import collections
import math

import numpy as np
import pandas as pd

from hybrid_agent.core.models import Bar, VolumeType
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
    compute_volume_sma,
)
from hybrid_agent.features.registry import (
    FeatureMetadata,
    FeatureRegistry,
    default_feature_registry,
)


class StatefulIncrementalFeatureEngine:
    """
    Maintains incremental mathematical state for real-time streaming feature extraction.
    Updates rolling and recursive indicators in O(1) time per bar without recalculating full history.
    Supports state invalidation and rebuilding when historical bars are revised.
    """

    def __init__(self, annualization_factor: float | None = 252.0) -> None:
        if annualization_factor is not None and annualization_factor < 0:
            raise ValueError(
                f"annualization_factor must be non-negative, got {annualization_factor}"
            )
        self.annualization_factor = annualization_factor
        self.reset()

    def reset(self) -> None:
        """Clears all internal rolling and recursive indicator states."""
        self._count = 0
        self._prev_close: float | None = None
        self._prev_high: float | None = None
        self._prev_low: float | None = None

        # SMA
        self._sma_20_window: collections.deque[float] = collections.deque(maxlen=20)
        self._sma_50_window: collections.deque[float] = collections.deque(maxlen=50)

        # EMA
        self._ema_20: float | None = None

        # MACD (fast=12, slow=26, signal=9)
        self._ema_fast: float | None = None
        self._ema_slow: float | None = None
        self._macd_signal: float | None = None

        # RSI (Wilder 14)
        self._rsi_avg_gain: float | None = None
        self._rsi_avg_loss: float | None = None

        # ROC (12)
        self._roc_window: collections.deque[float] = collections.deque(maxlen=13)

        # ATR (14)
        self._atr_avg_tr: float | None = None

        # ADX (14)
        self._adx_tr_smooth: float | None = None
        self._adx_plus_dm_smooth: float | None = None
        self._adx_minus_dm_smooth: float | None = None
        self._adx_value: float | None = None

        # Bollinger Bands (20, 2.0)
        self._bb_window: collections.deque[float] = collections.deque(maxlen=20)

        # Historical Volatility (20)
        self._hv_returns: collections.deque[float] = collections.deque(maxlen=20)

        # Volume SMA & RVOL (20)
        self._vol_window: collections.deque[float | None] = collections.deque(maxlen=20)

    def rebuild(self, bars: list[Bar]) -> None:
        """Rebuilds state from a historical sequence of bars (e.g. after bar revisions)."""
        self.reset()
        for b in bars:
            self.update(b)

    def update(self, bar: Bar) -> dict[str, float]:
        """
        Ingests a newly completed bar and incrementally updates all mathematical indicator states.
        Executes in O(1) time without recomputing over historical series.
        """
        self._count += 1
        close = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        vol = float(bar.volume)

        features: dict[str, float] = {}

        # 1. SMA (20 & 50)
        self._sma_20_window.append(close)
        if len(self._sma_20_window) == 20:
            features["sma_20"] = sum(self._sma_20_window) / 20.0

        self._sma_50_window.append(close)
        if len(self._sma_50_window) == 50:
            features["sma_50"] = sum(self._sma_50_window) / 50.0

        # 2. EMA (20)
        alpha_ema20 = 2.0 / (20.0 + 1.0)
        if self._ema_20 is None:
            self._ema_20 = close
        else:
            self._ema_20 = alpha_ema20 * close + (1.0 - alpha_ema20) * self._ema_20
        if self._count >= 20:
            features["ema_20"] = self._ema_20

        # 3. MACD (fast=12, slow=26, signal=9)
        alpha_fast = 2.0 / (12.0 + 1.0)
        alpha_slow = 2.0 / (26.0 + 1.0)
        alpha_signal = 2.0 / (9.0 + 1.0)

        if self._ema_fast is None:
            self._ema_fast = close
        else:
            self._ema_fast = alpha_fast * close + (1.0 - alpha_fast) * self._ema_fast

        if self._ema_slow is None:
            self._ema_slow = close
        else:
            self._ema_slow = alpha_slow * close + (1.0 - alpha_slow) * self._ema_slow

        macd_line = self._ema_fast - self._ema_slow
        if self._count >= 26:
            features["macd_line"] = macd_line

            if self._macd_signal is None:
                self._macd_signal = macd_line
            else:
                self._macd_signal = (
                    alpha_signal * macd_line + (1.0 - alpha_signal) * self._macd_signal
                )

            if self._count >= 34:
                features["macd_signal"] = self._macd_signal
                features["macd_hist"] = macd_line - self._macd_signal

        # 4. RSI (14) - Wilder smoothing (alpha = 1 / 14)
        alpha_wilder = 1.0 / 14.0
        if self._prev_close is not None:
            delta = close - self._prev_close
            gain = max(0.0, delta)
            loss = max(0.0, -delta)

            if self._rsi_avg_gain is None or self._rsi_avg_loss is None:
                self._rsi_avg_gain = gain
                self._rsi_avg_loss = loss
            else:
                self._rsi_avg_gain = alpha_wilder * gain + (1.0 - alpha_wilder) * self._rsi_avg_gain
                self._rsi_avg_loss = alpha_wilder * loss + (1.0 - alpha_wilder) * self._rsi_avg_loss

            if self._count > 14:
                if self._rsi_avg_gain == 0.0 and self._rsi_avg_loss == 0.0:
                    features["rsi_14"] = 50.0
                elif self._rsi_avg_loss == 0.0:
                    features["rsi_14"] = 100.0
                elif self._rsi_avg_gain == 0.0:
                    features["rsi_14"] = 0.0
                else:
                    rs = self._rsi_avg_gain / self._rsi_avg_loss
                    features["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # 5. ROC (12)
        self._roc_window.append(close)
        if len(self._roc_window) == 13:
            old_close = self._roc_window[0]
            if old_close != 0.0:
                features["roc_12"] = ((close - old_close) / old_close) * 100.0
            else:
                features["roc_12"] = 0.0

        # 6. ATR (14)
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))

        if self._atr_avg_tr is None:
            self._atr_avg_tr = tr
        else:
            self._atr_avg_tr = alpha_wilder * tr + (1.0 - alpha_wilder) * self._atr_avg_tr

        if self._count >= 14:
            features["atr_14"] = self._atr_avg_tr

        # 7. ADX (14)
        if self._prev_high is None or self._prev_low is None:
            # First bar (count=1): initialize smoothers
            self._adx_tr_smooth = tr
            self._adx_plus_dm_smooth = 0.0
            self._adx_minus_dm_smooth = 0.0
        else:
            up_move = high - self._prev_high
            down_move = self._prev_low - low
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0

            if (
                self._adx_tr_smooth is None
                or self._adx_plus_dm_smooth is None
                or self._adx_minus_dm_smooth is None
            ):
                self._adx_tr_smooth = tr
                self._adx_plus_dm_smooth = plus_dm
                self._adx_minus_dm_smooth = minus_dm
            else:
                self._adx_tr_smooth = alpha_wilder * tr + (1.0 - alpha_wilder) * self._adx_tr_smooth
                self._adx_plus_dm_smooth = (
                    alpha_wilder * plus_dm + (1.0 - alpha_wilder) * self._adx_plus_dm_smooth
                )
                self._adx_minus_dm_smooth = (
                    alpha_wilder * minus_dm + (1.0 - alpha_wilder) * self._adx_minus_dm_smooth
                )

        if self._count >= 14 and self._adx_tr_smooth is not None and self._adx_tr_smooth > 0:
            p_di = 100.0 * (self._adx_plus_dm_smooth / self._adx_tr_smooth)
            m_di = 100.0 * (self._adx_minus_dm_smooth / self._adx_tr_smooth)
            features["plus_di_14"] = p_di
            features["minus_di_14"] = m_di

            di_sum = p_di + m_di
            dx = 100.0 * (abs(p_di - m_di) / di_sum) if di_sum > 0 else 0.0

            if self._adx_value is None:
                self._adx_value = dx
            else:
                self._adx_value = alpha_wilder * dx + (1.0 - alpha_wilder) * self._adx_value

            if self._count >= 27:
                features["adx_14"] = self._adx_value

        # 8. Bollinger Bands (20, 2.0)
        self._bb_window.append(close)
        if len(self._bb_window) == 20:
            mean = sum(self._bb_window) / 20.0
            variance = sum((x - mean) ** 2 for x in self._bb_window) / 19.0
            std = math.sqrt(variance)
            upper = mean + 2.0 * std
            lower = mean - 2.0 * std
            width = upper - lower

            features["bb_middle_20"] = mean
            features["bb_upper_20"] = upper
            features["bb_lower_20"] = lower
            features["bb_pct_b_20"] = (close - lower) / width if width > 0 else 0.5
            features["bb_bandwidth_20"] = width / mean if mean > 0 else 0.0

        # 9. Historical Volatility (20)
        if self._prev_close is not None and self._prev_close > 0 and close > 0:
            log_ret = math.log(close / self._prev_close)
            self._hv_returns.append(log_ret)
            if len(self._hv_returns) == 20:
                mean_ret = sum(self._hv_returns) / 20.0
                var_ret = sum((r - mean_ret) ** 2 for r in self._hv_returns) / 19.0
                hv_std = math.sqrt(var_ret)
                ann = (
                    math.sqrt(self.annualization_factor)
                    if (self.annualization_factor is not None and self.annualization_factor > 0)
                    else 1.0
                )
                features["hv_20"] = hv_std * ann

        # 10. Volume SMA & RVOL (20) - per-observation availability
        is_vol_unavailable = (
            (hasattr(bar, "volume_type") and bar.volume_type == VolumeType.UNAVAILABLE)
            or (bar.volume is None)
            or math.isnan(bar.volume)
        )
        if is_vol_unavailable:
            self._vol_window.append(None)
        else:
            self._vol_window.append(vol)

        if len(self._vol_window) == 20 and None not in self._vol_window:
            valid_sum = sum(v for v in self._vol_window if v is not None)
            v_mean = valid_sum / 20.0
            if v_mean > 0:
                features["vol_sma_20"] = v_mean
                if not is_vol_unavailable:
                    features["rvol_20"] = vol / v_mean

        self._prev_close = close
        self._prev_high = high
        self._prev_low = low

        return features


class FeaturePipeline:
    """
    Transforms normalized market bars into quantitative features.
    Guarantees deterministic output, 100% feature registry coverage, and strict lookahead-free calculations.
    """

    def __init__(
        self,
        registry: FeatureRegistry | None = None,
        annualization_factor: float | None = 252.0,
    ) -> None:
        if annualization_factor is not None and annualization_factor < 0:
            raise ValueError(
                f"annualization_factor must be non-negative, got {annualization_factor}"
            )
        self.annualization_factor = annualization_factor
        self.registry = registry or default_feature_registry
        self._register_default_features()
        self._default_engine: StatefulIncrementalFeatureEngine | None = None

    def _register_default_features(self) -> None:
        """Populates the feature registry with default mathematical specifications."""
        features = [
            FeatureMetadata(
                name="SMA_20",
                description="20-period Simple Moving Average of closing prices",
                category="TREND",
                required_inputs=["close"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["sma_20"],
            ),
            FeatureMetadata(
                name="SMA_50",
                description="50-period Simple Moving Average of closing prices",
                category="TREND",
                required_inputs=["close"],
                parameters={"period": 50},
                warmup_period=50,
                output_columns=["sma_50"],
            ),
            FeatureMetadata(
                name="EMA_20",
                description="20-period Exponential Moving Average of closing prices",
                category="TREND",
                required_inputs=["close"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["ema_20"],
            ),
            FeatureMetadata(
                name="ADX_14",
                description="Average Directional Index with +DI and -DI (Wilder smoothing)",
                category="TREND",
                required_inputs=["high", "low", "close"],
                parameters={"period": 14},
                warmup_period=28,
                output_columns=["adx_14", "plus_di_14", "minus_di_14"],
            ),
            FeatureMetadata(
                name="RSI_14",
                description="Relative Strength Index using Wilder smoothing",
                category="MOMENTUM",
                required_inputs=["close"],
                parameters={"period": 14},
                warmup_period=14,
                output_columns=["rsi_14"],
            ),
            FeatureMetadata(
                name="MACD",
                description="Moving Average Convergence Divergence",
                category="MOMENTUM",
                required_inputs=["close"],
                parameters={"fast": 12, "slow": 26, "signal": 9},
                warmup_period=35,
                output_columns=["macd_line", "macd_signal", "macd_hist"],
            ),
            FeatureMetadata(
                name="ROC_12",
                description="12-period Price Rate of Change percentage",
                category="MOMENTUM",
                required_inputs=["close"],
                parameters={"period": 12},
                warmup_period=12,
                output_columns=["roc_12"],
            ),
            FeatureMetadata(
                name="ATR_14",
                description="14-period Average True Range volatility indicator",
                category="VOLATILITY",
                required_inputs=["high", "low", "close"],
                parameters={"period": 14},
                warmup_period=14,
                output_columns=["atr_14"],
            ),
            FeatureMetadata(
                name="BollingerBands_20",
                description="Bollinger Bands (Upper, Middle, Lower, %B, Bandwidth)",
                category="VOLATILITY",
                required_inputs=["close"],
                parameters={"period": 20, "num_std": 2.0},
                warmup_period=20,
                output_columns=[
                    "bb_upper_20",
                    "bb_middle_20",
                    "bb_lower_20",
                    "bb_pct_b_20",
                    "bb_bandwidth_20",
                ],
            ),
            FeatureMetadata(
                name="HistoricalVolatility_20",
                description="Annualized historical volatility based on log returns",
                category="VOLATILITY",
                required_inputs=["close"],
                parameters={"period": 20, "annualization_factor": self.annualization_factor},
                warmup_period=21,
                output_columns=["hv_20"],
            ),
            FeatureMetadata(
                name="VolumeSMA_20",
                description="20-period Simple Moving Average of traded volume",
                category="VOLUME",
                required_inputs=["volume"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["vol_sma_20"],
            ),
            FeatureMetadata(
                name="RelativeVolume_20",
                description="Ratio of current volume to rolling volume SMA",
                category="VOLUME",
                required_inputs=["volume"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["rvol_20"],
            ),
        ]
        for f in features:
            try:
                self.registry.register(f)
            except ValueError:
                pass  # Already registered

    def create_incremental_engine(
        self, annualization_factor: float | None = None
    ) -> StatefulIncrementalFeatureEngine:
        """Instantiates a dedicated stateful incremental calculation engine."""
        factor = self.annualization_factor if annualization_factor is None else annualization_factor
        return StatefulIncrementalFeatureEngine(annualization_factor=factor)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all standard quantitative indicators across historical batch dataframe.
        Input dataframe must contain: ['open', 'high', 'low', 'close', 'volume'].
        Returns dataframe with original prices and all calculated features appended.
        """
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise ValueError(f"DataFrame missing required price/volume columns: {missing}")

        res = df.copy()

        # Trend
        res["sma_20"] = compute_sma(res["close"], period=20)
        res["sma_50"] = compute_sma(res["close"], period=50)
        res["ema_20"] = compute_ema(res["close"], period=20)
        adx, plus_di, minus_di = compute_adx(res["high"], res["low"], res["close"], period=14)
        res["adx_14"] = adx
        res["plus_di_14"] = plus_di
        res["minus_di_14"] = minus_di

        # Momentum
        res["rsi_14"] = compute_rsi(res["close"], period=14)
        macd_line, macd_signal, macd_hist = compute_macd(res["close"], fast=12, slow=26, signal=9)
        res["macd_line"] = macd_line
        res["macd_signal"] = macd_signal
        res["macd_hist"] = macd_hist
        res["roc_12"] = compute_roc(res["close"], period=12)

        # Volatility
        res["atr_14"] = compute_atr(res["high"], res["low"], res["close"], period=14)
        upper, middle, lower, pct_b, bandwidth = compute_bollinger_bands(
            res["close"], period=20, num_std=2.0
        )
        res["bb_upper_20"] = upper
        res["bb_middle_20"] = middle
        res["bb_lower_20"] = lower
        res["bb_pct_b_20"] = pct_b
        res["bb_bandwidth_20"] = bandwidth
        res["hv_20"] = compute_historical_volatility(
            res["close"], period=20, annualization_factor=self.annualization_factor
        )

        # Volume (Per-observation availability handling)
        vol_series = res["volume"].astype(float).copy()
        if "volume_type" in res.columns:
            unavail_mask = (res["volume_type"] == VolumeType.UNAVAILABLE.value) | (
                res["volume_type"] == VolumeType.UNAVAILABLE
            )
            vol_series[unavail_mask] = np.nan

        valid_vol = vol_series.dropna()
        if valid_vol.empty or (valid_vol == 0.0).all():
            res["vol_sma_20"] = np.nan
            res["rvol_20"] = np.nan
        else:
            res["vol_sma_20"] = compute_volume_sma(vol_series, period=20)
            res["rvol_20"] = compute_relative_volume(vol_series, period=20)

        return res

    def transform_incremental(
        self,
        current_bar: Bar,
        history_bars: list[Bar] | None = None,
        engine: StatefulIncrementalFeatureEngine | None = None,
    ) -> dict[str, float]:
        """
        Calculates latest scalar feature values for a newly sealed bar using stateful incremental computation.
        Accepts newly completed bars and updates features in O(1) time without recalculating full history.
        """
        if engine is not None:
            return engine.update(current_bar)

        if self._default_engine is None:
            self._default_engine = self.create_incremental_engine()
            if history_bars:
                self._default_engine.rebuild(history_bars)

        return self._default_engine.update(current_bar)
