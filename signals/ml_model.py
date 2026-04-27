from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from config_loader import project_path


FEATURE_COLUMNS = [
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "bb_percent_b",
    "atr_14",
    "stoch_k",
    "stoch_d",
    "obv",
    "daily_return",
    "log_return",
    "volatility_20",
    "pct_above_sma_50",
    "pct_above_sma_200",
    "golden_cross",
    "death_cross",
]


def model_dir(config: dict) -> Path:
    path = project_path(config, config.get("ml", {}).get("model_dir", "models"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(config: dict, ticker: str) -> Path:
    return model_dir(config) / f"{ticker.upper()}_ml_model.joblib"


def build_training_frame(features: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    frames = []
    for _, group in features.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        out = group.copy()
        out["future_close"] = out["close"].shift(-horizon_days)
        out["target_up"] = (out["future_close"] > out["close"]).astype(int)
        out = out.iloc[:-horizon_days] if horizon_days > 0 else out
        frames.append(out)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_models(features: pd.DataFrame, config: dict, tickers: Iterable[str] | None = None) -> dict[str, dict]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Training requires joblib. Install project requirements first.") from exc

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
    except ImportError:
        RandomForestClassifier = None
        accuracy_score = None
        roc_auc_score = None

    ml_cfg = config.get("ml", {})
    horizon_days = int(ml_cfg.get("horizon_days", 1))
    training = build_training_frame(features, horizon_days)
    selected = [ticker.upper() for ticker in (tickers or sorted(training["ticker"].unique()))]
    results: dict[str, dict] = {}

    for ticker in selected:
        ticker_frame = training[training["ticker"] == ticker].dropna(subset=FEATURE_COLUMNS + ["target_up"])
        if ticker_frame.empty or ticker_frame["target_up"].nunique() < 2:
            results[ticker] = {"trained": False, "reason": "not enough labeled class variety"}
            continue

        x = ticker_frame[FEATURE_COLUMNS]
        y = ticker_frame["target_up"].astype(int)
        random_state = int(ml_cfg.get("random_state", 42))
        split_at = max(1, int(len(ticker_frame) * (1 - float(ml_cfg.get("test_size", 0.2)))))
        split_at = min(split_at, len(ticker_frame) - 1)
        x_train, x_test = x.iloc[:split_at], x.iloc[split_at:]
        y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
        if y_train.nunique() < 2:
            results[ticker] = {"trained": False, "reason": "training window has one target class"}
            continue
        if RandomForestClassifier is not None:
            rf_cfg = ml_cfg.get("random_forest", {})
            model = RandomForestClassifier(
                n_estimators=int(rf_cfg.get("n_estimators", 300)),
                max_depth=rf_cfg.get("max_depth", 8),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 5)),
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(x_train, y_train)
            probabilities = model.predict_proba(x_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            metrics = {"accuracy": float(accuracy_score(y_test, predictions))}
            if y_test.nunique() > 1:
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
            model_type = "random_forest"
        else:
            model = _fit_centroid_model(x_train, y_train)
            probabilities = _predict_centroid_probability(model, x_test)
            predictions = (probabilities >= 0.5).astype(int)
            metrics = {"accuracy": float((predictions == y_test.to_numpy()).mean())}
            model_type = "centroid_fallback"

        artifact = {
            "ticker": ticker,
            "model_type": model_type,
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "horizon_days": horizon_days,
            "metrics": metrics,
        }
        joblib.dump(artifact, model_path(config, ticker))
        results[ticker] = {"trained": True, **metrics}
    return results


def predict_scores(features: pd.DataFrame, config: dict) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(columns=["date", "ticker", "ml_score", "ml_probability_up", "ml_reason"])

    rows = []
    missing_score = float(config.get("signals", {}).get("ml_missing_score", 0.0))
    for ticker, group in features.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        ticker = str(ticker).upper()
        path = model_path(config, ticker)
        scored = group[["date", "ticker"]].copy()
        if not path.exists():
            scored["ml_score"] = missing_score
            scored["ml_probability_up"] = np.nan
            scored["ml_reason"] = "model_missing"
            rows.append(scored)
            continue

        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Loading trained models requires joblib. Install project requirements first.") from exc

        artifact = joblib.load(path)
        columns = artifact.get("feature_columns", FEATURE_COLUMNS)
        valid_mask = group[columns].notna().all(axis=1)
        scores = pd.Series(missing_score, index=group.index, dtype=float)
        probabilities = pd.Series(np.nan, index=group.index, dtype=float)
        if valid_mask.any():
            if artifact.get("model_type") == "centroid_fallback":
                probability_values = _predict_centroid_probability(artifact["model"], group.loc[valid_mask, columns])
            else:
                probability_values = artifact["model"].predict_proba(group.loc[valid_mask, columns])[:, 1]
            probabilities.loc[valid_mask] = probability_values
            scores.loc[valid_mask] = (probability_values * 2.0) - 1.0
        scored["ml_score"] = scores.clip(-1, 1).to_numpy()
        scored["ml_probability_up"] = probabilities.to_numpy()
        scored["ml_reason"] = np.where(valid_mask, "model_prediction", "insufficient_features")
        rows.append(scored)
    return pd.concat(rows, ignore_index=True)


def _fit_centroid_model(x: pd.DataFrame, y: pd.Series) -> dict:
    mean = x.mean()
    std = x.std(ddof=0).replace(0, 1)
    scaled = (x - mean) / std
    positive = scaled[y == 1]
    negative = scaled[y == 0]
    return {
        "mean": mean.to_dict(),
        "std": std.to_dict(),
        "positive_centroid": positive.mean().fillna(0).to_dict(),
        "negative_centroid": negative.mean().fillna(0).to_dict(),
        "base_rate": float(y.mean()),
    }


def _predict_centroid_probability(model: dict, x: pd.DataFrame) -> np.ndarray:
    columns = list(x.columns)
    mean = pd.Series(model["mean"], dtype=float).reindex(columns).fillna(0)
    std = pd.Series(model["std"], dtype=float).reindex(columns).replace(0, 1).fillna(1)
    positive = pd.Series(model["positive_centroid"], dtype=float).reindex(columns).fillna(0)
    negative = pd.Series(model["negative_centroid"], dtype=float).reindex(columns).fillna(0)
    scaled = (x - mean) / std
    distance_positive = np.sqrt(((scaled - positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((scaled - negative) ** 2).sum(axis=1))
    probability = distance_negative / (distance_positive + distance_negative + 1e-12)
    base_rate = float(model.get("base_rate", 0.5))
    return ((probability * 0.8) + (base_rate * 0.2)).clip(0, 1).to_numpy()
