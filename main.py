from __future__ import annotations

import asyncio
from warnings import warn

from config_loader import load_config
from data.data_update import update_data
from data.storage import load_watchlist_data
from features.pipeline import build_features
from output.cli_display import append_signal_logs, display_signal_table
from output.telegram_notify import send_telegram_summary, telegram_enabled
from signals.ensemble import generate_signals, latest_signals


def run_daily(config: dict | None = None) -> None:
    cfg = config or load_config()
    update_data(cfg)
    raw = load_watchlist_data(cfg)
    if raw.empty:
        raise RuntimeError("No stored market data available after update")
    features = build_features(raw, cfg)
    all_signals = generate_signals(features, cfg, use_ml=True)
    daily = latest_signals(all_signals)
    display_signal_table(daily, cfg)
    append_signal_logs(daily, cfg)
    if telegram_enabled(cfg):
        try:
            asyncio.run(send_telegram_summary(daily, cfg))
        except Exception as exc:
            warn(f"Telegram notification failed: {exc}", RuntimeWarning)


if __name__ == "__main__":
    run_daily()
