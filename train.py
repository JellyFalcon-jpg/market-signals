from __future__ import annotations

import argparse

from config_loader import load_config
from data.storage import load_watchlist_data
from features.pipeline import build_features
from signals.ml_model import train_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train market signal ML models")
    parser.add_argument("--ticker", action="append", help="Train one or more tickers; defaults to config watchlist")
    args = parser.parse_args()
    config = load_config()
    raw = load_watchlist_data(config, args.ticker)
    if raw.empty:
        raise RuntimeError("No stored OHLCV data found. Run python data/data_update.py first.")
    features = build_features(raw, config)
    results = train_models(features, config, args.ticker)
    for ticker, result in results.items():
        print(f"{ticker}: {result}")


if __name__ == "__main__":
    main()
