from __future__ import annotations

from pathlib import Path

import pandas as pd

from config_loader import project_path


def store_dir(config: dict) -> Path:
    configured = config.get("data", {}).get("store_dir", "data/store/ohlcv")
    path = project_path(config, configured)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ticker_path(config: dict, ticker: str) -> Path:
    return store_dir(config) / f"{ticker.upper()}.parquet"


def read_ticker(config: dict, ticker: str) -> pd.DataFrame:
    path = ticker_path(config, ticker)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.sort_values("date").reset_index(drop=True)


def write_ticker(config: dict, ticker: str, df: pd.DataFrame) -> Path:
    path = ticker_path(config, ticker)
    normalized = normalize_ohlcv(df)
    normalized = normalized.drop_duplicates(["ticker", "date"], keep="last")
    normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
    normalized.to_parquet(path, index=False)
    return path


def append_ticker(config: dict, ticker: str, new_rows: pd.DataFrame) -> Path:
    existing = read_ticker(config, ticker)
    combined = pd.concat([existing, new_rows], ignore_index=True)
    return write_ticker(config, ticker, combined)


def load_watchlist_data(config: dict, tickers: list[str] | None = None) -> pd.DataFrame:
    watchlist = tickers or config.get("watchlist", [])
    frames = [read_ticker(config, ticker) for ticker in watchlist]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out.columns = [str(column).lower().replace(" ", "_") for column in out.columns]
    rename_map = {"adj_close": "adj_close", "adjclose": "adj_close"}
    out = out.rename(columns=rename_map)
    required = ["date", "ticker", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in out.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")
    if "adj_close" not in out.columns:
        out["adj_close"] = out["close"]
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
