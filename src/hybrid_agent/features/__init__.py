"""Feature extraction, indicators, and feature registry package."""

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
from hybrid_agent.features.pipeline import FeaturePipeline, StatefulIncrementalFeatureEngine
from hybrid_agent.features.registry import (
    FeatureMetadata,
    FeatureRegistry,
    default_feature_registry,
)

__all__ = [
    "FeatureMetadata",
    "FeaturePipeline",
    "FeatureRegistry",
    "StatefulIncrementalFeatureEngine",
    "compute_adx",
    "compute_atr",
    "compute_bollinger_bands",
    "compute_ema",
    "compute_historical_volatility",
    "compute_macd",
    "compute_relative_volume",
    "compute_roc",
    "compute_rsi",
    "compute_sma",
    "compute_volume_sma",
    "default_feature_registry",
]
