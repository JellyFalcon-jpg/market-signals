from __future__ import annotations

import pandas as pd

from config_loader import load_config
from signals.ensemble import generate_signals


def feature_row(**overrides: object) -> dict:
    base = {
        "date": pd.Timestamp("2024-01-01"),
        "ticker": "AAPL",
        "open": 100,
        "high": 105,
        "low": 99,
        "close": 110,
        "volume": 1000,
        "sma_20": 105,
        "sma_50": 100,
        "sma_200": 95,
        "ema_12": 108,
        "ema_26": 104,
        "macd_line": 1,
        "macd_signal": 0,
        "macd_hist": 1,
        "rsi_14": 50,
        "bb_upper": 120,
        "bb_lower": 90,
        "bb_percent_b": 0.66,
        "atr_14": 3,
        "stoch_k": 80,
        "stoch_d": 70,
        "obv": 1000,
        "daily_return": 0.01,
        "log_return": 0.01,
        "volatility_20": 0.02,
        "pct_above_sma_50": 10,
        "pct_above_sma_200": 15,
        "golden_cross": 0,
        "death_cross": 0,
    }
    base.update(overrides)
    return base


def test_ensemble_buy_threshold_without_ml() -> None:
    config = load_config()
    df = pd.DataFrame(
        [
            feature_row(date=pd.Timestamp("2024-01-01"), sma_20=90, sma_50=100, macd_line=-1, macd_signal=0),
            feature_row(date=pd.Timestamp("2024-01-02"), sma_20=105, sma_50=100, macd_line=1, macd_signal=0, golden_cross=1),
        ]
    )
    signals = generate_signals(df, config, use_ml=False)
    assert signals.iloc[-1]["signal"] == "BUY"


def test_ensemble_sell_threshold_without_ml() -> None:
    config = load_config()
    rows = [
        feature_row(date=pd.Timestamp("2024-01-01"), close=90, sma_20=105, sma_50=100, sma_200=105, macd_line=1, macd_signal=0),
        feature_row(date=pd.Timestamp("2024-01-02"), close=80, sma_20=90, sma_50=100, sma_200=110, macd_line=-1, macd_signal=0, rsi_14=80, death_cross=1),
    ]
    signals = generate_signals(pd.DataFrame(rows), config, use_ml=False)
    assert signals.iloc[-1]["signal"] == "SELL"
