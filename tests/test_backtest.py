from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from config_loader import load_config
from features.pipeline import build_features


def test_backtest_uses_prior_signal_for_current_return() -> None:
    config = load_config()
    dates = pd.date_range("2023-01-01", periods=260, freq="D")
    close = pd.Series(np.linspace(100, 180, 260))
    raw = pd.DataFrame(
        {
            "date": dates,
            "ticker": "AAPL",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "adj_close": close,
            "volume": 1000,
        }
    )
    features = build_features(raw, config)
    results, metrics = run_backtest(features, config, "AAPL", use_ml=False)
    assert results["position"].iloc[0] == 0
    assert "total_return" in metrics
    assert "max_drawdown" in metrics
