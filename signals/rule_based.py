from __future__ import annotations

import json

import numpy as np
import pandas as pd


TECHNICAL_RULE_COLUMNS = [
    "rule_sma_trend",
    "rule_sma_cross",
    "rule_macd",
    "rule_rsi",
    "rule_bollinger",
    "rule_golden_death_cross",
]


def score_rule_based(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    frames = []
    for _, group in df.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        frames.append(_score_ticker(group.copy(), config))
    return pd.concat(frames, ignore_index=True)


def _score_ticker(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    signal_cfg = config.get("signals", {})
    rsi_cfg = signal_cfg.get("rsi", {})
    oversold = float(rsi_cfg.get("oversold", 30))
    overbought = float(rsi_cfg.get("overbought", 70))

    df["rule_sma_trend"] = np.select(
        [
            (df["close"] > df["sma_50"]) & (df["sma_50"] > df["sma_200"]),
            (df["close"] < df["sma_50"]) & (df["sma_50"] < df["sma_200"]),
        ],
        [1, -1],
        default=0,
    )

    sma_relation = df["sma_20"] - df["sma_50"]
    df["rule_sma_cross"] = np.select(
        [(sma_relation.shift(1) <= 0) & (sma_relation > 0), (sma_relation.shift(1) >= 0) & (sma_relation < 0)],
        [1, -1],
        default=0,
    )

    macd_relation = df["macd_line"] - df["macd_signal"]
    df["rule_macd"] = np.select(
        [(macd_relation.shift(1) <= 0) & (macd_relation > 0), (macd_relation.shift(1) >= 0) & (macd_relation < 0)],
        [1, -1],
        default=0,
    )

    df["rule_rsi"] = np.select([df["rsi_14"] <= oversold, df["rsi_14"] >= overbought], [1, -1], default=0)
    df["rule_bollinger"] = np.select([df["close"] > df["bb_upper"], df["close"] < df["bb_lower"]], [1, -1], default=0)
    df["rule_golden_death_cross"] = np.select([df["golden_cross"] == 1, df["death_cross"] == 1], [1, -1], default=0)

    weights = signal_cfg.get("technical_weights", {})
    weighted_columns = {
        "rule_sma_trend": float(weights.get("sma_trend", 1.0)),
        "rule_sma_cross": float(weights.get("sma_cross", 1.0)),
        "rule_macd": float(weights.get("macd", 1.0)),
        "rule_rsi": float(weights.get("rsi", 0.8)),
        "rule_bollinger": float(weights.get("bollinger", 0.8)),
        "rule_golden_death_cross": float(weights.get("golden_death_cross", 1.2)),
    }
    total_weight = sum(abs(weight) for weight in weighted_columns.values()) or 1.0
    weighted_score = sum(df[column].fillna(0) * weight for column, weight in weighted_columns.items())
    df["technical_score"] = (weighted_score / total_weight).clip(-1, 1)
    df["technical_reason"] = df.apply(lambda row: _reason(row), axis=1)
    return df


def _reason(row: pd.Series) -> str:
    components = {column.replace("rule_", ""): int(row.get(column, 0) or 0) for column in TECHNICAL_RULE_COLUMNS}
    return json.dumps(components, sort_keys=True)
