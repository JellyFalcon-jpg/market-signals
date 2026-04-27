from __future__ import annotations

import os

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config_loader import load_config
from signals.query import format_signal_reply, signal_for_query


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(help_text())


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(help_text())


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("pong")


async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    await reply_with_signal(update, query)


async def text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await reply_with_signal(update, update.message.text.strip())


async def reply_with_signal(update: Update, query: str) -> None:
    if not query:
        await update.message.reply_text("Send a ticker or company name, for example: /signal AAPL")
        return

    await update.message.reply_text(f"Checking {query}...")
    try:
        signal = signal_for_query(query, load_config())
    except Exception as exc:
        await update.message.reply_text(f"Could not generate a signal: {exc}")
        return
    await update.message.reply_text(format_signal_reply(signal))


def help_text() -> str:
    return (
        "Send a ticker or company name and I will return BUY / SELL / HOLD.\n"
        "Examples:\n"
        "/signal AAPL\n"
        "/signal Apple\n"
        "Tesla\n\n"
        "Health check: /ping"
    )


def main() -> None:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    if not token:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before running telegram_bot.py")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_message))
    print("Telegram signal bot is running. Open Telegram and send /ping or /signal AAPL.")
    app.run_polling()


if __name__ == "__main__":
    main()
