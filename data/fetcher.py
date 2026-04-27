from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import pandas as pd


def fetch_daily_ohlcv(
    tickers: Iterable[str],
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    auto_adjust: bool = False,
) -> pd.DataFrame:
    """Fetch daily OHLCV data from yfinance and normalize it for storage."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance to fetch market data: python -m pip install yfinance") from exc

    ticker_list = [ticker.upper() for ticker in tickers]
    if not ticker_list:
        return pd.DataFrame()

    raw = yf.download(
        tickers=ticker_list,
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in ticker_list:
            if ticker not in raw.columns.get_level_values(0):
                continue
            frame = raw[ticker].reset_index()
            frame["ticker"] = ticker
            frames.append(_normalize_yfinance_frame(frame))
    else:
        frame = raw.reset_index()
        frame["ticker"] = ticker_list[0]
        frames.append(_normalize_yfinance_frame(frame))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def _normalize_yfinance_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(column).lower().replace(" ", "_") for column in out.columns]
    if "adj_close" not in out.columns and "close" in out.columns:
        out["adj_close"] = out["close"]
    out = out.rename(columns={"datetime": "date"})
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    return out[["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]].dropna(subset=["close"])
