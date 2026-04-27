from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from config_loader import project_path


def display_signal_table(signals: pd.DataFrame, config: dict) -> None:
    console = Console()
    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker")
    table.add_column("Signal", justify="center")
    table.add_column("Composite", justify="right")
    table.add_column("Technical", justify="right")
    table.add_column("ML Score", justify="right")

    for _, row in signals.sort_values("ticker").iterrows():
        table.add_row(
            str(row["ticker"]),
            str(row["signal"]),
            _format_score(row.get("composite_score")),
            _format_score(row.get("technical_score")),
            _format_score(row.get("ml_score")),
        )
    console.print(table)
    disclaimer = config.get("output", {}).get("disclaimer")
    if disclaimer:
        console.print(disclaimer)


def append_signal_logs(signals: pd.DataFrame, config: dict) -> None:
    if signals.empty:
        return
    output_cfg = config.get("output", {})
    csv_path = project_path(config, output_cfg.get("signal_log_csv", "output/logs/signals.csv"))
    jsonl_path = project_path(config, output_cfg.get("signal_log_jsonl", "output/logs/signals.jsonl"))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    log_columns = [
        "date",
        "ticker",
        "signal",
        "composite_score",
        "technical_score",
        "ml_score",
        "ml_probability_up",
        "reason",
    ]
    out = signals[[column for column in log_columns if column in signals.columns]].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)
    out.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for _, row in out.iterrows():
            payload = row.where(pd.notna(row), None).to_dict()
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _format_score(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    score = float(value)
    return f"{score:+.2f}"
