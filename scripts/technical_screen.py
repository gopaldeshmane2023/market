#!/usr/bin/env python3
"""Phase 3 — Technical screening (pandas/numpy only; no TA library).

Universe:  your watchlist (config/watchlist.csv) and/or the NSE universe from reports/nse_eod_processed.csv.
History:   data/history/<SYMBOL>.csv if present (Date,Open,High,Low,Close,Volume), otherwise Yahoo Finance (<SYMBOL>.NS).
Outputs:   reports/technical_screen_full.csv, reports/technical_shortlist.csv, reports/technical_screen.json

An indicator combination is a filter, not a forecast. Passing it does not mean a stock will rise.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import (add_dir_args, dirs_from_args, gh_error, gh_warning, load_yaml, now_utc, today_ist, update_status,
                    write_json)

PHASE = "technical_screen"
HistoryProvider = Callable[[str, str], "tuple[pd.DataFrame | None, str]"]


# --------------------------------------------------------------------------- indicators
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. All-gain => 100, all-loss => 0, no movement => 50."""
    delta = close.diff()
    gain, loss = delta.clip(lower=0.0), (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    out = out.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def clean_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise column names, index and invalid rows. Returns None if unusable."""
    if df is None or len(df) == 0:
        return None
    d = df.copy()
    d.columns = [str(c).strip().title() for c in d.columns]
    need = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in d.columns for c in need):
        return None
    d = d[need].apply(pd.to_numeric, errors="coerce")
    idx = pd.to_datetime(d.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d = d[d.index.notna()]
    d = d.dropna(subset=["Close", "High", "Low"])
    d = d[(d["Close"] > 0) & (d["High"] >= d["Low"])]
    d["Volume"] = d["Volume"].fillna(0.0)
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d if len(d) else None


def compute_indicators(df: pd.DataFrame, vol_window: int = 20) -> pd.DataFrame:
    """Return df with indicator columns. Average volume/value use the PRIOR `vol_window` sessions so that
    today's own spike does not inflate its own baseline."""
    o = df.copy()
    c = o["Close"]
    o["ema20"], o["ema50"], o["ema200"] = ema(c, 20), ema(c, 50), ema(c, 200)
    o["rsi14"] = rsi(c, 14)
    o = o.join(macd(c, 12, 26, 9))
    o["atr14"] = atr(o["High"], o["Low"], c, 14)
    o["avg_vol20"] = o["Volume"].shift(1).rolling(vol_window, min_periods=vol_window).mean()
    o["rel_volume"] = o["Volume"] / o["avg_vol20"].replace(0.0, np.nan)
    o["avg_traded_value_cr20"] = (c * o["Volume"]).shift(1).rolling(vol_window, min_periods=vol_window).mean() / 1e7
    o["daily_pct"] = c.pct_change() * 100.0
    o["high20"] = o["High"].rolling(20, min_periods=20).max()
    o["high50"] = o["High"].rolling(50, min_periods=50).max()
    o["dist_high20_pct"] = (c / o["high20"] - 1.0) * 100.0
    o["dist_high50_pct"] = (c / o["high50"] - 1.0) * 100.0
    return o


def count_circuit_days(df: pd.DataFrame, band_pct: float, lookback: int, tol: float) -> tuple[float, float]:
    """Approximate count of upper/lower-circuit-like sessions in the last `lookback` bars.
    Uses TODAY's band for all days (bands can change) — an approximation, documented in the README."""
    if not np.isfinite(band_pct):
        return np.nan, np.nan
    ret = df["Close"].pct_change() * 100.0
    w = df.tail(lookback)
    r = ret.loc[w.index]
    up = ((r >= band_pct - tol) & (w["Close"] >= w["High"] * (1 - 1e-4))).sum()
    dn = ((r <= -(band_pct - tol)) & (w["Close"] <= w["Low"] * (1 + 1e-4))).sum()
    return float(up), float(dn)


# --------------------------------------------------------------------------- evaluation
def _num(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def evaluate_symbol(symbol: str, df: pd.DataFrame | None, nse: dict | None, cfg: dict, ref_date: date,
                    history_source: str = "") -> dict:
    """Evaluate one symbol. `cfg` is the 'technical' section of screening.yml."""
    nse = nse or {}
    series = str(nse.get("series")) if nse.get("series") not in (None, "", "nan") and not pd.isna(nse.get("series")) else "UNKNOWN"
    row: dict = {"symbol": symbol, "series": series, "history_source": history_source,
                 "delivery_pct": _num(nse.get("delivery_pct")), "nse_close": _num(nse.get("close")),
                 "nse_traded_value_cr": _num(nse.get("traded_value_cr")),
                 "price_band_pct": _num(nse.get("price_band_pct")),
                 "circuit_status": nse.get("circuit_status", "UNKNOWN") if not pd.isna(nse.get("circuit_status", np.nan)) else "UNKNOWN",
                 "security_status": nse.get("security_status", "UNVERIFIED") if not pd.isna(nse.get("security_status", np.nan)) else "UNVERIFIED",
                 "master_series": nse.get("master_series") if not pd.isna(nse.get("master_series", np.nan)) else "",
                 "surveillance": nse.get("surveillance") if not pd.isna(nse.get("surveillance", np.nan)) else "NO_DATA",
                 "nse_source_date": nse.get("source_date", ""), "nse_data_quality": nse.get("data_quality", "")}
    fails: list[str] = []
    passes: list[str] = []

    d = clean_ohlcv(df)
    if d is None:
        row.update(data_status="NO_DATA", bars=0, technical_pass=False, shortlisted=False, series_ok=False,
                   reasons_pass="", reasons_fail="No usable price history (download failed, symbol not on Yahoo, or empty).")
        return row

    ind = compute_indicators(d)
    last = ind.iloc[-1]
    bars = int(len(d))
    last_date = d.index[-1].date()
    age = (ref_date - last_date).days
    stale = age > int(cfg["max_stale_days"])
    row.update(bars=bars, data_date=last_date.isoformat(), data_age_days=age, last_close=float(last["Close"]),
               daily_pct=_num(last["daily_pct"]), ema20=_num(last["ema20"]), ema50=_num(last["ema50"]),
               ema200=_num(last["ema200"]), rsi14=_num(last["rsi14"]), macd=_num(last["macd"]),
               macd_signal=_num(last["macd_signal"]), atr14=_num(last["atr14"]), volume=float(last["Volume"]),
               avg_vol20=_num(last["avg_vol20"]), rel_volume=_num(last["rel_volume"]),
               avg_traded_value_cr20=_num(last["avg_traded_value_cr20"]),
               dist_high20_pct=_num(last["dist_high20_pct"]), dist_high50_pct=_num(last["dist_high50_pct"]))
    row["macd_status"] = ("ABOVE_SIGNAL" if row["macd"] > row["macd_signal"] else "BELOW_SIGNAL") \
        if np.isfinite(row["macd"]) and np.isfinite(row["macd_signal"]) else "N/A"
    row["return_lookback_pct"] = _num((d["Close"].iloc[-1] / d["Close"].iloc[-1 - int(cfg["momentum"]["lookback_days"])] - 1) * 100) \
        if bars > int(cfg["momentum"]["lookback_days"]) else float("nan")
    up, dn = count_circuit_days(d, row["price_band_pct"], int(cfg["circuit_lookback_days"]), 0.15)
    row["circuit_up_days"], row["circuit_down_days"] = up, dn

    data_status = "OK"
    if stale:
        data_status = "STALE"
        fails.append(f"Price data is stale: last bar {last_date} is {age} days before reference date {ref_date}.")
    if np.isfinite(row["nse_close"]) and str(nse.get("source_date", "")) == last_date.isoformat():
        diff = abs(row["last_close"] / row["nse_close"] - 1) * 100
        row["price_mismatch"] = bool(diff > float(cfg["price_mismatch_pct"]))
        if row["price_mismatch"]:
            fails.append(f"Yahoo close {row['last_close']:.2f} differs from NSE close {row['nse_close']:.2f} by {diff:.1f}% "
                         "(possible corporate action or bad data).")
    else:
        row["price_mismatch"] = False
    row["data_status"] = data_status

    # --- criteria
    min_hist = int(cfg["min_history_bars"])
    needed = ["ema20", "ema50", "ema200", "rsi14", "macd", "macd_signal", "rel_volume"]
    nan_ind = [k for k in needed if not np.isfinite(row[k])]
    history_ok = bars >= min_hist and not nan_ind
    row["history_ok"] = history_ok
    row["series_ok"] = row["series"] in [s.upper() for s in cfg["allowed_series"]]
    flags = dict(trend_ok=False, rsi_ok=False, macd_ok=False, volume_ok=False, momentum_ok=False)
    if not history_ok:
        why = f"{bars} bars < {min_hist} required" if bars < min_hist else f"indicators not computable ({', '.join(nan_ind)})"
        fails.append(f"Insufficient history: {why}.")
    else:
        flags["trend_ok"] = (row["ema20"] > row["ema50"] > row["ema200"]) if cfg["trend_required"] else True
        (passes if flags["trend_ok"] else fails).append(
            f"EMA20 {row['ema20']:.2f} {'>' if row['ema20'] > row['ema50'] else '<='} EMA50 {row['ema50']:.2f} "
            f"{'>' if row['ema50'] > row['ema200'] else '<='} EMA200 {row['ema200']:.2f}"
            + (" (trend aligned)" if flags["trend_ok"] else " (trend NOT aligned)"))
        lo, hi = float(cfg["rsi"]["min"]), float(cfg["rsi"]["max"])
        flags["rsi_ok"] = lo <= row["rsi14"] <= hi
        (passes if flags["rsi_ok"] else fails).append(
            f"RSI14 {row['rsi14']:.1f} {'within' if flags['rsi_ok'] else 'outside'} {lo:g}-{hi:g}")
        flags["macd_ok"] = (row["macd"] > row["macd_signal"]) if cfg["macd_above_signal"] else True
        (passes if flags["macd_ok"] else fails).append(
            f"MACD {row['macd']:.3f} {'above' if row['macd'] > row['macd_signal'] else 'not above'} signal {row['macd_signal']:.3f}")
        mrv = float(cfg["min_relative_volume"])
        flags["volume_ok"] = row["rel_volume"] >= mrv
        (passes if flags["volume_ok"] else fails).append(
            f"Relative volume {row['rel_volume']:.2f}x {'>=' if flags['volume_ok'] else '<'} {mrv:g}x (prior 20-session average)")
        m = cfg["momentum"]
        mom_a = np.isfinite(row["daily_pct"]) and row["daily_pct"] > float(m["min_daily_change_pct"])
        mom_b = np.isfinite(row["return_lookback_pct"]) and row["return_lookback_pct"] > float(m["min_lookback_return_pct"])
        flags["momentum_ok"] = bool(mom_a and mom_b)
        (passes if flags["momentum_ok"] else fails).append(
            f"Momentum: daily {row['daily_pct']:+.2f}% (need > {m['min_daily_change_pct']:g}%), "
            f"{m['lookback_days']}-day {row['return_lookback_pct']:+.2f}% (need > {m['min_lookback_return_pct']:g}%)"
            + ("" if flags["momentum_ok"] else " — not met"))
    row.update(flags)
    if row["series_ok"]:
        passes.append(f"Series {row['series']} is eligible for the normal delivery shortlist")
    elif row["series"] == "UNKNOWN":
        fails.append("Series unknown (no NSE data for this symbol) — cannot be shortlisted until verified.")
    else:
        fails.append(f"Series {row['series']} is not in the allowed list {cfg['allowed_series']}.")

    row["technical_pass"] = bool(history_ok and all(flags.values()))
    row["shortlisted"] = bool(row["technical_pass"] and row["series_ok"] and data_status == "OK" and not row["price_mismatch"])
    row["reasons_pass"] = " | ".join(passes)
    row["reasons_fail"] = " | ".join(fails)
    return row


# --------------------------------------------------------------------------- universe & history
def load_watchlist(path: Path) -> tuple[list[dict], list[str]]:
    """Read symbol,exchange,notes. Lines starting with # are comments. Returns (entries, problems)."""
    if not path.exists():
        return [], [f"{path} not found."]
    try:
        df = pd.read_csv(path, dtype=str, comment="#", skipinitialspace=True).fillna("")
    except pd.errors.EmptyDataError:
        return [], ["watchlist.csv is empty."]
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError(f"watchlist.csv must have a 'symbol' column (expected: symbol,exchange,notes); found {list(df.columns)}")
    entries, problems, seen = [], [], set()
    for _, r in df.iterrows():
        sym = r["symbol"].strip().upper()
        if not sym:
            continue
        exch = (r.get("exchange", "") or "NSE").strip().upper()
        if sym in seen:
            problems.append(f"{sym}: duplicate watchlist row ignored.")
            continue
        seen.add(sym)
        if exch != "NSE":
            problems.append(f"{sym}: exchange '{exch}' is not supported (NSE only); skipped.")
            continue
        entries.append({"symbol": sym, "exchange": exch, "notes": r.get("notes", "")})
    return entries, problems


def default_history_provider(data_dir: Path) -> HistoryProvider:
    def provider(symbol: str, period: str) -> tuple[pd.DataFrame | None, str]:
        local = data_dir / "history" / f"{symbol}.csv"
        if local.exists():
            df = pd.read_csv(local, index_col=0, parse_dates=True)
            return df, f"local:{local.name}"
        import yfinance as yf
        last: Exception | None = None
        for attempt in range(2):
            try:
                df = yf.Ticker(f"{symbol}.NS").history(period=period, interval="1d", auto_adjust=False)
                if df is not None and not df.empty:
                    return df, "yfinance"
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(1.0)
        if last:
            raise last
        return None, "yfinance"
    return provider


# --------------------------------------------------------------------------- main
FULL_COLUMNS = ["symbol", "series", "data_status", "data_date", "data_age_days", "bars", "last_close", "daily_pct",
                "ema20", "ema50", "ema200", "rsi14", "macd", "macd_signal", "macd_status", "rel_volume", "volume",
                "avg_vol20", "avg_traded_value_cr20", "atr14", "dist_high20_pct", "dist_high50_pct",
                "return_lookback_pct", "delivery_pct", "nse_close", "nse_traded_value_cr", "price_band_pct",
                "circuit_status", "circuit_up_days", "circuit_down_days", "security_status", "master_series",
                "surveillance", "price_mismatch", "history_ok", "trend_ok", "rsi_ok", "macd_ok", "volume_ok",
                "momentum_ok", "series_ok", "technical_pass", "shortlisted", "reasons_pass", "reasons_fail",
                "history_source", "in_watchlist", "watchlist_notes", "nse_source_date", "nse_data_quality"]


def run(args: argparse.Namespace, provider: HistoryProvider | None = None) -> int:
    config_dir, data_dir, reports_dir = dirs_from_args(args)
    cfg_all = load_yaml(config_dir / "screening.yml")
    cfg = cfg_all["technical"]
    provider = provider or default_history_provider(data_dir)

    nse_path = reports_dir / "nse_eod_processed.csv"
    nse = pd.read_csv(nse_path, dtype={"symbol": str}) if nse_path.exists() else None
    if nse is not None:
        nse = nse.drop_duplicates("symbol", keep="first")      # processed file keeps EQ first (sorted by series)
        eq = nse[nse["series"] == "EQ"]
        nse = pd.concat([eq, nse[~nse["symbol"].isin(eq["symbol"])]]).drop_duplicates("symbol")
    nse_by_symbol = {r["symbol"]: r for r in nse.to_dict("records")} if nse is not None else {}

    wl_entries, problems = ([], [])
    if args.universe in ("watchlist", "both"):
        wl_entries, problems = load_watchlist(config_dir / "watchlist.csv")
    universe: dict[str, dict] = {e["symbol"]: {"in_watchlist": True, "notes": e["notes"]} for e in wl_entries}
    if args.universe in ("nse", "both"):
        if nse is None:
            msg = f"Universe '{args.universe}' requested but {nse_path.name} is missing (run nse_data.py with NSE files first)."
            if args.universe == "nse":
                gh_error(f"technical_screen: {msg}")
                update_status(reports_dir, PHASE, "FAILED", msg)
                return 1
            gh_warning(f"technical_screen: {msg} Using watchlist only.")
            problems.append(msg)
        else:
            u = cfg["universe"]
            pool = nse[(nse["series"] == "EQ") & (nse["traded_value_cr"] >= float(u["min_traded_value_cr"]))]
            pool = pool.sort_values("traded_value_cr", ascending=False).head(args.max_symbols or int(u["max_symbols"]))
            for s in pool["symbol"]:
                universe.setdefault(s, {"in_watchlist": False, "notes": ""})
    if not universe:
        msg = "Universe is empty (watchlist has no NSE symbols and no NSE universe was loaded). Nothing to screen."
        gh_warning(f"technical_screen: {msg}")
        update_status(reports_dir, PHASE, "SKIPPED", msg)
        return 0 if args.optional else 1

    ref_date = today_ist()  # staleness is judged against today (IST), not against the NSE file date

    rows = []
    for i, (sym, meta) in enumerate(sorted(universe.items()), 1):
        try:
            df, src = provider(sym, cfg["history_period"])
        except Exception as exc:  # noqa: BLE001 — one failing symbol must not stop the run
            df, src = None, f"error: {type(exc).__name__}: {exc}"
        row = evaluate_symbol(sym, df, nse_by_symbol.get(sym), cfg, ref_date, src)
        row["in_watchlist"], row["watchlist_notes"] = meta["in_watchlist"], meta["notes"]
        rows.append(row)
        if i % 25 == 0:
            print(f"[technical_screen] {i}/{len(universe)} processed")

    full = pd.DataFrame(rows).reindex(columns=FULL_COLUMNS)
    full.to_csv(reports_dir / "technical_screen_full.csv", index=False)
    short = full[full["shortlisted"] == True].sort_values("rel_volume", ascending=False)   # noqa: E712
    short.to_csv(reports_dir / "technical_shortlist.csv", index=False)

    n_nodata = int((full["data_status"] == "NO_DATA").sum())
    summary = {"generated_at_utc": now_utc().isoformat(timespec="seconds"), "universe": args.universe,
               "symbols_screened": len(full), "no_data": n_nodata,
               "stale": int((full["data_status"] == "STALE").sum()),
               "technical_pass": int(full["technical_pass"].sum()), "shortlisted": int(len(short)),
               "nse_data_used": nse is not None, "reference_date": ref_date.isoformat(),
               "watchlist_problems": problems, "thresholds": cfg,
               "caveat": "Indicator filters are not predictions. A shortlist is a list to study, not to buy."}
    write_json(reports_dir / "technical_screen.json", summary)
    status = "OK"
    notes = []
    if n_nodata:
        status = "DEGRADED"
        notes.append(f"{n_nodata} symbol(s) without price data")
    if nse is None:
        status = "DEGRADED"
        notes.append("no NSE data: series unknown, so nothing can be shortlisted")
    if problems:
        notes.append(f"{len(problems)} watchlist problem(s)")
    update_status(reports_dir, PHASE, status, f"{len(full)} screened, {summary['technical_pass']} indicator-pass, "
                  f"{len(short)} shortlisted" + (f"; {'; '.join(notes)}" if notes else ""))
    if n_nodata == len(full):
        gh_error("technical_screen: no symbol returned price history — network/Yahoo problem?")
        return 1
    if notes:
        gh_warning("technical_screen: " + "; ".join(notes))
    print(f"[technical_screen] screened={len(full)} indicator_pass={summary['technical_pass']} shortlisted={len(short)}")
    return 0


def main(argv: list[str] | None = None, provider: HistoryProvider | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dir_args(ap)
    ap.add_argument("--universe", choices=["watchlist", "nse", "both"], default="watchlist")
    ap.add_argument("--max-symbols", type=int, default=0, help="cap for the NSE universe (default: config)")
    ap.add_argument("--optional", action="store_true", help="exit 0 (SKIPPED) when the universe is empty")
    args = ap.parse_args(argv)
    try:
        return run(args, provider)
    except Exception as exc:  # noqa: BLE001
        gh_error(f"technical_screen failed: {type(exc).__name__}: {exc}")
        update_status(Path(args.reports_dir), PHASE, "FAILED", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
