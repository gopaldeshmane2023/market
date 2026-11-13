"""Helpers to build a throwaway repo layout (config/data/reports) for tests."""
from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from common import REPO_ROOT

SEC_HEADER = ("SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, "
              "TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER")


def sec_line(sym, series, date, prev, o, h, l, c, qty, deliv_qty="500000", deliv_per="45.5"):
    turnover_lacs = c * qty / 1e5
    return f"{sym}, {series}, {date}, {prev}, {o}, {h}, {l}, {c}, {c}, {c}, {qty}, {turnover_lacs:.2f}, 1000, {deliv_qty}, {deliv_per}"


def make_sandbox(tmp: Path, min_rows: int = 1) -> tuple[Path, Path, Path]:
    cfg, data, rep = tmp / "config", tmp / "data", tmp / "reports"
    for d in (cfg, data, rep):
        d.mkdir(parents=True, exist_ok=True)
    y = yaml.safe_load((REPO_ROOT / "config" / "screening.yml").read_text())
    y["nse"]["min_rows_full_market"] = min_rows
    (cfg / "screening.yml").write_text(yaml.safe_dump(y))
    shutil.copy(REPO_ROOT / "config" / "market_impact.yml", cfg / "market_impact.yml")
    return cfg, data, rep
