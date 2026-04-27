from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from config_loader import load_config
from data.fetcher import fetch_daily_ohlcv
from data.storage import append_ticker, read_ticker
from features.pipeline import build_features
from signals.ensemble import generate_signals, latest_signals


KNOWN_COMPANIES = {
    "apple": "AAPL",
    "apple inc": "AAPL",
    "tesla": "TSLA",
    "tesla inc": "TSLA",
    "microsoft": "MSFT",
    "microsoft corp": "MSFT",
    "nvidia": "NVDA",
    "nvidia corp": "NVDA",
    "spy": "SPY",
    "s&p 500": "SPY",
}


def signal_for_query(query: str, config: dict | None = None) -> dict:
    cfg = config or load_config()
    ticker = resolve_ticker(query)
    update_ticker_data(ticker, cfg)
    raw = read_ticker(cfg, ticker)
    if raw.empty:
        raise RuntimeError(f"No market data found for {ticker}")
    features = build_features(raw, cfg)
    signals = latest_signals(generate_signals(features, cfg, use_ml=True))
    if signals.empty:
        raise RuntimeError(f"Could not generate signal for {ticker}")
    row = signals.iloc[-1]
    return {
        "ticker": ticker,
        "date": pd.to_datetime(row["date"]).date().isoformat(),
        "signal": row["signal"],
        "composite_score": float(row["composite_score"]),
        "technical_score": float(row["technical_score"]),
        "ml_score": float(row["ml_score"]),
        "reason": row.get("reason", ""),
    }


def update_ticker_data(ticker: str, config: dict) -> int:
    existing = read_ticker(config, ticker)
    if existing.empty:
        start = config.get("data", {}).get("start_date", "2018-01-01")
    else:
        start = (existing["date"].max() + timedelta(days=1)).date().isoformat()

    if datetime.fromisoformat(str(start)).date() > datetime.now(ZoneInfo("America/New_York")).date():
        return 0

    fetched = fetch_daily_ohlcv(
        [ticker],
        start=start,
        auto_adjust=bool(config.get("data", {}).get("auto_adjust", False)),
    )
    if fetched.empty:
        return 0
    append_ticker(config, ticker, fetched)
    return len(fetched)


def resolve_ticker(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        raise ValueError("Please provide a ticker or company name, for example: /signal AAPL")

    known = KNOWN_COMPANIES.get(cleaned.lower())
    if known:
        return known

    if re.fullmatch(r"[A-Za-z.]{1,8}", cleaned):
        return cleaned.upper()

    try:
        import yfinance as yf

        search = yf.Search(cleaned, max_results=1)
        quotes = getattr(search, "quotes", None) or []
        if quotes:
            symbol = quotes[0].get("symbol")
            if symbol:
                return str(symbol).upper()
    except Exception:
        pass

    raise ValueError(f"Could not resolve '{query}' to a ticker. Try the stock symbol, e.g. AAPL.")


def format_signal_reply(signal: dict) -> str:
    return "\n".join(
        [
            f"{signal['ticker']} signal: {signal['signal']}",
            f"Date: {signal['date']}",
            f"Composite: {signal['composite_score']:+.2f}",
            f"Technical: {signal['technical_score']:+.2f}",
            f"ML Score: {signal['ml_score']:+.2f}",
            "",
            "For informational and educational purposes only. Not financial advice.",
        ]
    )
