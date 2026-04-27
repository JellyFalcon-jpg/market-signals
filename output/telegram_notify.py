from __future__ import annotations

import os

import pandas as pd


def telegram_enabled(config: dict) -> bool:
    return bool(config.get("notifications", {}).get("telegram", {}).get("enabled", False))


async def send_telegram_summary(signals: pd.DataFrame, config: dict) -> None:
    telegram_cfg = config.get("notifications", {}).get("telegram", {})
    if not telegram_cfg.get("enabled", False):
        return
    token = (os.environ.get(telegram_cfg.get("bot_token_env", "TELEGRAM_BOT_TOKEN")) or "").strip()
    chat_id = (os.environ.get(telegram_cfg.get("chat_id_env", "TELEGRAM_CHAT_ID")) or "").strip()
    if not token or not chat_id:
        raise RuntimeError("Telegram is enabled, but token or chat id environment variables are missing")

    from telegram import Bot

    lines = ["Daily market signals"]
    for _, row in signals.sort_values("ticker").iterrows():
        lines.append(
            f"{row['ticker']}: {row['signal']} "
            f"(composite {float(row['composite_score']):+.2f}, "
            f"technical {float(row['technical_score']):+.2f}, ml {float(row['ml_score']):+.2f})"
        )
    disclaimer = config.get("output", {}).get("disclaimer")
    if disclaimer:
        lines.append("")
        lines.append(disclaimer)

    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text="\n".join(lines))
