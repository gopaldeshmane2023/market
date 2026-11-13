"""Shared helpers for TradeSentinel scripts (paths, YAML, JSON, run-status tracking)."""
from __future__ import annotations

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
IST = ZoneInfo("Asia/Kolkata")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def today_ist() -> date:
    return datetime.now(IST).date()


def add_dir_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-dir", default=str(REPO_ROOT / "config"))
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"))


def dirs_from_args(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    config_dir, data_dir, reports_dir = (Path(args.config_dir), Path(args.data_dir), Path(args.reports_dir))
    reports_dir.mkdir(parents=True, exist_ok=True)
    return config_dir, data_dir, reports_dir


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required config file missing: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def json_safe(obj: Any) -> Any:
    """Convert numpy/pandas/datetime values (and NaN/inf) into JSON-safe values."""
    if obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    return obj


def write_json(path: Path | str, obj: Any) -> None:
    Path(path).write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_status(reports_dir: Path, phase: str, status: str, message: str = "") -> None:
    """Record a phase outcome (OK / DEGRADED / SKIPPED / FAILED) in reports/run_status.json."""
    path = Path(reports_dir) / "run_status.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[phase] = {"status": status, "message": message, "updated_utc": now_utc().isoformat(timespec="seconds")}
    write_json(path, data)


def gh_warning(msg: str) -> None:
    print(f"::warning::{msg}")


def gh_error(msg: str) -> None:
    print(f"::error::{msg}")


def fmt(value: Any, digits: int = 2, dash: str = "n/a") -> str:
    """Format a number for reports; None/NaN become a visible 'n/a' (never silently 0)."""
    if value is None:
        return dash
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return dash if not math.isfinite(v) else f"{v:,.{digits}f}"
