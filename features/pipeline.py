from __future__ import annotations

import pandas as pd

from features.technical import add_technical_indicators


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Return an enriched DataFrame sorted by ticker/date."""
    if df.empty:
        return df.copy()
    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required feature input columns: {sorted(missing)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return add_technical_indicators(out, config)
