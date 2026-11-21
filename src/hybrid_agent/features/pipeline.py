"""Feature engineering pipeline orchestrating technical indicator transformations."""

import pandas as pd

from hybrid_agent.core.models import Bar
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


class FeaturePipeline:
    """
    Transforms normalized market bars into quantitative features.
    Guarantees deterministic output and strict lookahead-free calculations.
    """

    def __init__(self, registry: FeatureRegistry | None = None) -> None:
        self.registry = registry or default_feature_registry
        self._register_default_features()

    def _register_default_features(self) -> None:
        """Populates the feature registry with default mathematical specifications."""
        features = [
            FeatureMetadata(
                name="SMA",
                description="Simple Moving Average of closing prices",
                category="TREND",
                required_inputs=["close"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["sma_20"],
            ),
            FeatureMetadata(
                name="EMA",
                description="Exponential Moving Average of closing prices",
                category="TREND",
                required_inputs=["close"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["ema_20"],
            ),
            FeatureMetadata(
                name="ADX",
                description="Average Directional Index with +DI and -DI",
                category="TREND",
                required_inputs=["high", "low", "close"],
                parameters={"period": 14},
                warmup_period=28,
                output_columns=["adx_14", "plus_di_14", "minus_di_14"],
            ),
            FeatureMetadata(
                name="RSI",
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
                name="ROC",
                description="Price Rate of Change percentage",
                category="MOMENTUM",
                required_inputs=["close"],
                parameters={"period": 12},
                warmup_period=12,
                output_columns=["roc_12"],
            ),
            FeatureMetadata(
                name="ATR",
                description="Average True Range volatility indicator",
                category="VOLATILITY",
                required_inputs=["high", "low", "close"],
                parameters={"period": 14},
                warmup_period=14,
                output_columns=["atr_14"],
            ),
            FeatureMetadata(
                name="BollingerBands",
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
                name="HistoricalVolatility",
                description="Annualized historical volatility based on log returns",
                category="VOLATILITY",
                required_inputs=["close"],
                parameters={"period": 20, "trading_periods": 252},
                warmup_period=21,
                output_columns=["hv_20"],
            ),
            FeatureMetadata(
                name="VolumeSMA",
                description="Simple Moving Average of traded volume",
                category="VOLUME",
                required_inputs=["volume"],
                parameters={"period": 20},
                warmup_period=20,
                output_columns=["vol_sma_20"],
            ),
            FeatureMetadata(
                name="RelativeVolume",
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
        res["hv_20"] = compute_historical_volatility(res["close"], period=20, trading_periods=252)

        # Volume
        res["vol_sma_20"] = compute_volume_sma(res["volume"], period=20)
        res["rvol_20"] = compute_relative_volume(res["volume"], period=20)

        return res

    def transform_incremental(
        self,
        current_bar: Bar,
        history_bars: list[Bar],
    ) -> dict[str, float]:
        """
        Calculates latest scalar feature values for a newly sealed bar given its preceding history.
        Ensures 100% equivalence between streaming updates and batch dataframe calculations.
        """
        all_bars = history_bars + [current_bar]
        records = [
            {
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
            }
            for b in all_bars
        ]
        df = pd.DataFrame(records)
        transformed = self.transform(df)
        latest_row = transformed.iloc[-1].to_dict()

        # Return only calculated feature columns
        feature_keys = [
            "sma_20",
            "sma_50",
            "ema_20",
            "adx_14",
            "plus_di_14",
            "minus_di_14",
            "rsi_14",
            "macd_line",
            "macd_signal",
            "macd_hist",
            "roc_12",
            "atr_14",
            "bb_upper_20",
            "bb_middle_20",
            "bb_lower_20",
            "bb_pct_b_20",
            "bb_bandwidth_20",
            "hv_20",
            "vol_sma_20",
            "rvol_20",
        ]
        return {k: float(latest_row[k]) for k in feature_keys if pd.notna(latest_row.get(k))}
