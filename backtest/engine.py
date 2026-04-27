from __future__ import annotations

import numpy as np
import pandas as pd

from signals.ensemble import generate_signals


def run_backtest(features: pd.DataFrame, config: dict, ticker: str, use_ml: bool = False) -> tuple[pd.DataFrame, dict]:
    ticker = ticker.upper()
    frame = features[features["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No feature data found for {ticker}")

    signals = generate_signals(frame, config, use_ml=use_ml)
    signal_map = {"BUY": 1, "HOLD": np.nan, "SELL": 0}
    results = signals[["date", "ticker", "close", "signal", "composite_score", "technical_score", "ml_score"]].copy()
    results["target_position"] = results["signal"].map(signal_map).ffill().fillna(0)
    results["position"] = results["target_position"].shift(1).fillna(0)
    results["market_return"] = results["close"].pct_change().fillna(0)
    trading_cost = float(config.get("backtest", {}).get("trading_cost_bps", 5)) / 10000
    results["trade"] = results["position"].diff().abs().fillna(results["position"].abs())
    results["strategy_return"] = (results["position"] * results["market_return"]) - (results["trade"] * trading_cost)
    initial_cash = float(config.get("backtest", {}).get("initial_cash", 10000))
    results["equity"] = initial_cash * (1 + results["strategy_return"]).cumprod()
    results["buy_hold_equity"] = initial_cash * (1 + results["market_return"]).cumprod()
    metrics = calculate_metrics(results)
    return results, metrics


def calculate_metrics(results: pd.DataFrame) -> dict:
    strategy_returns = results["strategy_return"].fillna(0)
    total_return = (results["equity"].iloc[-1] / results["equity"].iloc[0]) - 1 if len(results) > 1 else 0.0
    sharpe = 0.0
    if strategy_returns.std(ddof=0) > 0:
        sharpe = float((strategy_returns.mean() / strategy_returns.std(ddof=0)) * np.sqrt(252))
    rolling_peak = results["equity"].cummax()
    drawdown = (results["equity"] / rolling_peak) - 1
    winning = strategy_returns[strategy_returns > 0].sum()
    losing = strategy_returns[strategy_returns < 0].sum()
    profit_factor = float(winning / abs(losing)) if losing < 0 else float("inf")
    active_days = strategy_returns[strategy_returns != 0]
    win_rate = float((active_days > 0).mean()) if not active_days.empty else 0.0
    buy_hold_return = (results["buy_hold_equity"].iloc[-1] / results["buy_hold_equity"].iloc[0]) - 1 if len(results) > 1 else 0.0
    return {
        "total_return": float(total_return),
        "buy_hold_return": float(buy_hold_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }
