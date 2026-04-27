from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config_loader import load_config
from data.fetcher import fetch_daily_ohlcv
from data.storage import append_ticker, read_ticker


def update_data(config: dict | None = None) -> dict[str, int]:
    cfg = config or load_config()
    watchlist = [ticker.upper() for ticker in cfg.get("watchlist", [])]
    if not watchlist:
        raise ValueError("No tickers configured in watchlist")

    counts: dict[str, int] = {}
    latest_fetch_date = _latest_fetch_date()
    for ticker in watchlist:
        existing = read_ticker(cfg, ticker)
        if existing.empty:
            start = cfg.get("data", {}).get("start_date", "2018-01-01")
        else:
            start = (existing["date"].max() + timedelta(days=1)).date().isoformat()

        if datetime.fromisoformat(str(start)).date() > latest_fetch_date:
            counts[ticker] = 0
            continue

        fetched = fetch_daily_ohlcv(
            [ticker],
            start=start,
            auto_adjust=bool(cfg.get("data", {}).get("auto_adjust", False)),
        )
        if fetched.empty:
            counts[ticker] = 0
            continue
        append_ticker(cfg, ticker, fetched)
        counts[ticker] = len(fetched)
    return counts


def _latest_fetch_date() -> datetime.date:
    """Use New York date so Asia time zones do not request a future US market day."""
    return datetime.now(ZoneInfo("America/New_York")).date()


def main() -> None:
    counts = update_data()
    for ticker, count in counts.items():
        print(f"{ticker}: appended {count} rows")


if __name__ == "__main__":
    main()
