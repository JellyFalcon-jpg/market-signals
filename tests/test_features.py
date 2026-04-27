from __future__ import annotations

import numpy as np
import pandas as pd

from config_loader import load_config
from features.pipeline import build_features


def sample_ohlcv(rows: int = 260) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    close = pd.Series(np.linspace(100, 160, rows))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "adj_close": close,
            "volume": 1_000_000,
        }
    )


def test_feature_pipeline_adds_expected_columns() -> None:
    config = load_config()
    features = build_features(sample_ohlcv(), config)
    expected = {
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "macd_line",
        "macd_signal",
        "rsi_14",
        "bb_upper",
        "bb_lower",
        "atr_14",
        "stoch_k",
        "stoch_d",
        "daily_return",
        "log_return",
        "volatility_20",
        "pct_above_sma_50",
        "pct_above_sma_200",
    }
    assert expected.issubset(features.columns)
    assert features["sma_20"].iloc[19] == features["close"].iloc[:20].mean()


def test_features_are_sorted_by_ticker_and_date() -> None:
    config = load_config()
    raw = sample_ohlcv(30).sample(frac=1, random_state=1)
    features = build_features(raw, config)
    assert features["date"].is_monotonic_increasing
