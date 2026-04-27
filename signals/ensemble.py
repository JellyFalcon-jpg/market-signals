from __future__ import annotations

import json

import pandas as pd

from signals.ml_model import predict_scores
from signals.rule_based import score_rule_based


def generate_signals(features: pd.DataFrame, config: dict, use_ml: bool = True) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    technical = score_rule_based(features, config)
    if use_ml:
        ml_scores = predict_scores(features, config)
    else:
        ml_scores = technical[["date", "ticker"]].copy()
        ml_scores["ml_score"] = float(config.get("signals", {}).get("ml_missing_score", 0.0))
        ml_scores["ml_probability_up"] = pd.NA
        ml_scores["ml_reason"] = "disabled"

    merged = technical.merge(ml_scores, on=["date", "ticker"], how="left")
    weights = config.get("signals", {}).get("ensemble_weights", {})
    technical_weight = float(weights.get("technical", 0.6))
    ml_weight = float(weights.get("ml", 0.4))
    ml_unavailable = merged["ml_reason"].isin(["model_missing", "disabled"])
    effective_ml_weight = pd.Series(ml_weight, index=merged.index, dtype=float)
    effective_ml_weight.loc[ml_unavailable] = 0.0
    total_weight = (abs(technical_weight) + effective_ml_weight.abs()).replace(0, 1.0)
    merged["ml_score"] = merged["ml_score"].fillna(float(config.get("signals", {}).get("ml_missing_score", 0.0)))
    merged["composite_score"] = (
        (merged["technical_score"].fillna(0) * technical_weight)
        + (merged["ml_score"].fillna(0) * effective_ml_weight)
    ) / total_weight
    thresholds = config.get("signals", {}).get("thresholds", {})
    buy_threshold = float(thresholds.get("buy", 0.3))
    sell_threshold = float(thresholds.get("sell", -0.3))
    merged["signal"] = merged["composite_score"].apply(lambda score: _signal(score, buy_threshold, sell_threshold))
    merged["reason"] = merged.apply(lambda row: _reason(row), axis=1)
    return merged.sort_values(["ticker", "date"]).reset_index(drop=True)


def latest_signals(signal_frame: pd.DataFrame) -> pd.DataFrame:
    if signal_frame.empty:
        return signal_frame
    return signal_frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1).reset_index(drop=True)


def _signal(score: float, buy_threshold: float, sell_threshold: float) -> str:
    if score > buy_threshold:
        return "BUY"
    if score < sell_threshold:
        return "SELL"
    return "HOLD"


def _reason(row: pd.Series) -> str:
    payload = {
        "technical_score": round(float(row.get("technical_score", 0.0)), 4),
        "ml_score": round(float(row.get("ml_score", 0.0)), 4),
        "technical_reason": row.get("technical_reason", "{}"),
        "ml_reason": row.get("ml_reason", "unknown"),
    }
    return json.dumps(payload, sort_keys=True)
