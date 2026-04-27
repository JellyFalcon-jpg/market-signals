# Market Signals

Daily market analysis system for educational BUY / SELL / HOLD decision-support signals.

The project ingests daily OHLCV data, computes technical features, runs rule-based and optional ML strategies, and prints a traceable signal table suitable for a daily cron job or on-demand CLI.

## Disclaimer

This system is for informational and educational purposes only. Signals are decision-support inputs, not financial advice. Always apply independent judgment before acting. Past performance does not guarantee future results.

## Quick Start

```sh
python -m pip install -r requirements.txt
python data/data_update.py
python train.py
python main.py
```

Run a backtest:

```sh
python -m backtest.report --ticker AAPL
```

Run the interactive Telegram bot:

```sh
export TELEGRAM_BOT_TOKEN="your_bot_token"
python telegram_bot.py
```

Then message the bot:

```text
/signal AAPL
/signal Apple
Tesla
```

## Configuration

Edit `config/settings.yaml` to change the watchlist, thresholds, model settings, storage paths, and notification settings.

Defaults:

- Data source: `yfinance`
- Storage: Parquet under `data/store/ohlcv`
- Watchlist: `AAPL`, `TSLA`, `MSFT`, `NVDA`, `SPY`
- Final thresholds: `BUY > 0.3`, `SELL < -0.3`, otherwise `HOLD`
- Ensemble weights: technical `0.6`, ML `0.4`

## Daily Flow

1. `data/data_update.py` fetches and appends missing daily OHLCV rows.
2. `features/pipeline.py` computes indicators and derived features.
3. `signals/rule_based.py` scores technical strategies.
4. `signals/ml_model.py` loads trained models when available.
5. `signals/ensemble.py` produces final signals and reasoning.
6. `output/cli_display.py` prints a Rich table and appends audit logs.

## Cron Example

```cron
30 6 * * 1-5 cd /path/to/market-signals && python main.py >> output/logs/cron.log 2>&1
```

Schedule this after your market data provider has daily candles available.
