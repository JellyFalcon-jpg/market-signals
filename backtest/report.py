from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from backtest.engine import run_backtest
from config_loader import load_config, project_path
from data.storage import load_watchlist_data
from features.pipeline import build_features


def write_backtest_report(config: dict, ticker: str, use_ml: bool = False) -> Path:
    raw = load_watchlist_data(config, [ticker])
    features = build_features(raw, config)
    results, metrics = run_backtest(features, config, ticker, use_ml=use_ml)
    reports_dir = project_path(config, config.get("output", {}).get("reports_dir", "output/reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_path = reports_dir / f"{ticker.upper()}_equity_curve.png"
    ax = results.plot(x="date", y=["equity", "buy_hold_equity"], figsize=(10, 5), title=f"{ticker.upper()} Backtest")
    ax.set_ylabel("Equity")
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)

    report_path = reports_dir / f"{ticker.upper()}_backtest.md"
    lines = [
        f"# {ticker.upper()} Backtest",
        "",
        config.get("output", {}).get("disclaimer", ""),
        "",
        "## Metrics",
        "",
        f"- Total return: {_pct(metrics['total_return'])}",
        f"- Buy and hold return: {_pct(metrics['buy_hold_return'])}",
        f"- Sharpe ratio: {metrics['sharpe_ratio']:.2f}",
        f"- Max drawdown: {_pct(metrics['max_drawdown'])}",
        f"- Win rate: {_pct(metrics['win_rate'])}",
        f"- Profit factor: {_format_profit_factor(metrics['profit_factor'])}",
        "",
        f"Equity curve: `{plot_path.name}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _format_profit_factor(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a backtest report")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--use-ml", action="store_true")
    args = parser.parse_args()
    config = load_config()
    report_path = write_backtest_report(config, args.ticker, use_ml=args.use_ml)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
