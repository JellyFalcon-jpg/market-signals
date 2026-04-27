from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config and return a plain dictionary."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data["_config_path"] = str(config_path)
    data["_project_root"] = str(config_path.resolve().parents[1])
    return data


def project_path(config: dict[str, Any], *parts: str) -> Path:
    root = Path(config.get("_project_root", PROJECT_ROOT))
    return root.joinpath(*parts)
