from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    frames = []
    for _, group in df.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        frames.append(_add_for_ticker(group.copy(), config))
    return pd.concat(frames, ignore_index=True)


def _add_for_ticker(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_cfg = config.get("features", {})
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    for window in feature_cfg.get("sma_windows", [20, 50, 200]):
        df[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()
    for window in feature_cfg.get("ema_windows", [12, 26]):
        df[f"ema_{window}"] = close.ewm(span=window, adjust=False, min_periods=window).mean()

    macd_cfg = feature_cfg.get("macd", {})
    fast = int(macd_cfg.get("fast", 12))
    slow = int(macd_cfg.get("slow", 26))
    signal = int(macd_cfg.get("signal", 9))
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    df["macd_line"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd_line"].ewm(span=signal, adjust=False, min_periods=signal).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    rsi_window = int(feature_cfg.get("rsi_window", 14))
    df["rsi_14"] = _rsi(close, rsi_window)

    boll_cfg = feature_cfg.get("bollinger", {})
    boll_window = int(boll_cfg.get("window", 20))
    boll_std = float(boll_cfg.get("std_dev", 2))
    middle = close.rolling(window=boll_window, min_periods=boll_window).mean()
    rolling_std = close.rolling(window=boll_window, min_periods=boll_window).std()
    df["bb_middle"] = middle
    df["bb_upper"] = middle + (rolling_std * boll_std)
    df["bb_lower"] = middle - (rolling_std * boll_std)
    df["bb_percent_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    atr_window = int(feature_cfg.get("atr_window", 14))
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(window=atr_window, min_periods=atr_window).mean()

    stoch_cfg = feature_cfg.get("stochastic", {})
    stoch_window = int(stoch_cfg.get("window", 14))
    smooth_window = int(stoch_cfg.get("smooth_window", 3))
    rolling_low = low.rolling(window=stoch_window, min_periods=stoch_window).min()
    rolling_high = high.rolling(window=stoch_window, min_periods=stoch_window).max()
    df["stoch_k"] = 100 * (close - rolling_low) / (rolling_high - rolling_low)
    df["stoch_d"] = df["stoch_k"].rolling(window=smooth_window, min_periods=smooth_window).mean()

    df["obv"] = (np.sign(close.diff()).fillna(0) * volume).cumsum()
    df["daily_return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))
    volatility_window = int(feature_cfg.get("volatility_window", 20))
    df["volatility_20"] = df["daily_return"].rolling(window=volatility_window, min_periods=volatility_window).std()
    df["pct_above_sma_50"] = (close / df.get("sma_50") - 1.0) * 100
    df["pct_above_sma_200"] = (close / df.get("sma_200") - 1.0) * 100

    sma_50 = df.get("sma_50")
    sma_200 = df.get("sma_200")
    if sma_50 is not None and sma_200 is not None:
        previous_relation = sma_50.shift(1) - sma_200.shift(1)
        current_relation = sma_50 - sma_200
        df["golden_cross"] = ((previous_relation <= 0) & (current_relation > 0)).astype(int)
        df["death_cross"] = ((previous_relation >= 0) & (current_relation < 0)).astype(int)
    else:
        df["golden_cross"] = 0
        df["death_cross"] = 0

    return df


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
