"""Deterministic mathematical indicators for trend, momentum, volatility, and volume."""

import numpy as np
import pandas as pd

# ==========================================
# 1. TREND INDICATORS
# ==========================================


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Computes Simple Moving Average with strict warm-up validation."""
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")
    return series.rolling(window=period, min_periods=period).mean()


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Computes Exponential Moving Average with strict warm-up validation."""
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")
    # Compute using standard alpha = 2 / (period + 1) with min_periods requirement
    ema = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return ema


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Average Directional Index (ADX), Plus Directional Indicator (+DI),
    and Minus Directional Indicator (-DI) using standard Wilder smoothing.
    Returns: (ADX, Plus_DI, Minus_DI)
    """
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=low.index)

    plus_mask = (up_move > down_move) & (up_move > 0)
    minus_mask = (down_move > up_move) & (down_move > 0)

    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]

    # Wilder smoothing: alpha = 1 / period
    tr_smooth = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * (di_diff / di_sum.replace(0, np.nan))

    adx = dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    return adx, plus_di, minus_di


# ==========================================
# 2. MOMENTUM INDICATORS
# ==========================================


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Computes Relative Strength Index using Wilder exponential smoothing."""
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing: alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Boundary cleanups where loss is 0 (RSI=100) or gain is 0 (RSI=0)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)

    # Re-apply NaN for initial warm-up period
    rsi.iloc[:period] = np.nan

    return rsi


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes Moving Average Convergence Divergence (MACD).
    Returns: (macd_line, signal_line, histogram)
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("fast, slow, and signal periods must be strictly positive")
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be strictly less than slow ({slow})")

    fast_ema = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_ema = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_roc(close: pd.Series, period: int = 12) -> pd.Series:
    """Computes Price Rate of Change (% change over N periods)."""
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")
    shifted = close.shift(period)
    return ((close - shifted) / shifted.replace(0, np.nan)) * 100.0


# ==========================================
# 3. VOLATILITY INDICATORS
# ==========================================


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Computes Average True Range (ATR) with Wilder smoothing."""
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def compute_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Computes Bollinger Bands.
    Returns: (upper_band, middle_band, lower_band, percent_b, bandwidth)
    """
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")
    if num_std <= 0:
        raise ValueError(f"num_std must be strictly positive, got {num_std}")

    middle = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)

    width = upper - lower
    percent_b = (close - lower) / width.replace(0, np.nan)
    bandwidth = width / middle.replace(0, np.nan)

    return upper, middle, lower, percent_b, bandwidth


def compute_historical_volatility(
    close: pd.Series,
    period: int = 20,
    trading_periods: int = 252,
) -> pd.Series:
    """Computes annualized Historical Volatility based on log returns."""
    if period <= 1:
        raise ValueError(f"period must be greater than 1, got {period}")
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=period, min_periods=period).std()
    return rolling_std * np.sqrt(trading_periods)


# ==========================================
# 4. VOLUME INDICATORS
# ==========================================


def compute_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Computes Volume Simple Moving Average."""
    if period <= 0:
        raise ValueError(f"period must be strictly positive, got {period}")
    return volume.rolling(window=period, min_periods=period).mean()


def compute_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Computes Relative Volume (Current Volume / Rolling Volume SMA)."""
    vol_sma = compute_volume_sma(volume, period)
    return volume / vol_sma.replace(0, np.nan)
