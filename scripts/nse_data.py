#!/usr/bin/env python3
"""Phase 2 — NSE data collection and validation.

Reads OFFICIAL NSE downloadable files that YOU place in data/ (or, optionally, tries the public
archive URLs with --try-download), validates their columns, normalises them and writes:

  reports/nse_eod_processed.csv   normalised end-of-day table (one row per symbol/series)
  reports/nse_data_report.json/md source date, download timestamp, validation results

Supported inputs (any one price file is enough; the first one found wins, in this order):
  1. sec_bhavdata_full*.csv   NSE "Full Bhavcopy and Security-wise Delivery Positions" (includes delivery %)
  2. BhavCopy_NSE_CM_*.csv    NSE CM UDiFF bhavcopy (no delivery data)
  3. cm*bhav.csv              legacy CM bhavcopy (no delivery data)
Optional companions:
  nse_security_master.csv     NSE EQUITY_L.csv (listed securities)
  nse_price_band.csv          symbol, price_band_pct      (manually maintained)
  nse_surveillance.csv        symbol, list [, stage]      (ASM/GSM — manually maintained)

This script never bypasses CAPTCHAs, logins or access controls.
A failed or empty download is NEVER treated as an empty market: it is an error.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from common import (add_dir_args, dirs_from_args, fmt, gh_error, gh_warning, load_yaml, now_utc, today_ist,
                    update_status, write_json)

PHASE = "nse_data"


class ValidationError(Exception):
    """Raised when an input file is missing required columns or fails sanity checks."""


# column signatures after upper-casing and stripping header names
FORMATS: dict[str, dict] = {
    "sec_bhavdata_full": {
        "glob": "sec_bhavdata_full*",
        "required": {"SYMBOL", "SERIES", "DATE1", "PREV_CLOSE", "OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE", "CLOSE_PRICE",
                     "TTL_TRD_QNTY", "TURNOVER_LACS", "DELIV_QTY", "DELIV_PER"},
        "map": {"SYMBOL": "symbol", "SERIES": "series", "DATE1": "date", "PREV_CLOSE": "prev_close",
                "OPEN_PRICE": "open", "HIGH_PRICE": "high", "LOW_PRICE": "low", "CLOSE_PRICE": "close",
                "TTL_TRD_QNTY": "volume", "TURNOVER_LACS": "turnover_lacs", "DELIV_QTY": "delivery_qty",
                "DELIV_PER": "delivery_pct"},
        "has_delivery": True,
    },
    "udiff": {
        "glob": "BhavCopy_NSE_CM_*",
        "required": {"TCKRSYMB", "SCTYSRS", "TRADDT", "OPNPRIC", "HGHPRIC", "LWPRIC", "CLSPRIC", "PRVSCLSGPRIC",
                     "TTLTRADGVOL", "TTLTRFVAL"},
        "map": {"TCKRSYMB": "symbol", "SCTYSRS": "series", "TRADDT": "date", "OPNPRIC": "open", "HGHPRIC": "high",
                "LWPRIC": "low", "CLSPRIC": "close", "PRVSCLSGPRIC": "prev_close", "TTLTRADGVOL": "volume",
                "TTLTRFVAL": "turnover_inr", "ISIN": "isin"},
        "has_delivery": False,
    },
    "legacy": {
        "glob": "cm*bhav*",
        "required": {"SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE", "PREVCLOSE", "TOTTRDQTY", "TOTTRDVAL",
                     "TIMESTAMP"},
        "map": {"SYMBOL": "symbol", "SERIES": "series", "TIMESTAMP": "date", "OPEN": "open", "HIGH": "high",
                "LOW": "low", "CLOSE": "close", "PREVCLOSE": "prev_close", "TOTTRDQTY": "volume",
                "TOTTRDVAL": "turnover_inr", "ISIN": "isin"},
        "has_delivery": False,
    },
}
MASTER_COLUMNS_REQUIRED = {"SYMBOL", "SERIES"}
DATE_FORMATS = ["%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d%b%Y"]
OUT_COLUMNS = ["symbol", "series", "open", "high", "low", "close", "prev_close", "volume", "traded_value_cr",
               "delivery_qty", "delivery_pct", "price_band_pct", "circuit_status", "security_status", "master_series",
               "surveillance", "isin", "data_quality", "source_date", "source_file", "source_format", "download_ts"]


# --------------------------------------------------------------------------- reading helpers
def read_csv_normalised(path: Path) -> pd.DataFrame:
    """Read a CSV (or single-file zip) as strings with upper-cased/stripped headers and stripped values."""
    try:
        df = pd.read_csv(path, dtype=str, skipinitialspace=True, comment=None)
    except pd.errors.EmptyDataError as exc:
        raise ValidationError(f"{path.name}: file is empty") from exc
    except Exception as exc:  # noqa: BLE001
        raise ValidationError(f"{path.name}: cannot be parsed as CSV ({type(exc).__name__}: {exc})") from exc
    df.columns = [str(c).strip().upper() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype("string").str.strip()
    return df


def parse_dates(values: pd.Series, source: str) -> pd.Series:
    last_bad = len(values)
    for f in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=f, errors="coerce")
        bad = int(parsed.isna().sum())
        if bad == 0:
            return parsed
        last_bad = min(last_bad, bad)
    raise ValidationError(f"{source}: could not parse the date column (tried {DATE_FORMATS}); "
                          f"best attempt left {last_bad} unparsable values. NSE may have changed the date format.")


def to_num(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(s.astype("string").replace({"-": pd.NA, "": pd.NA}), errors="coerce")
    return out.astype("float64")      # plain NaN (not pd.NA) so downstream numpy logic behaves


def detect_format(columns: set[str]) -> str | None:
    for name, spec in FORMATS.items():
        if spec["required"] <= columns:
            return name
    return None


def parse_price_file(path: Path, fmt_name: str | None = None, min_rows: int = 500) -> tuple[pd.DataFrame, dict]:
    """Parse and validate one NSE price file. Raises ValidationError with exact reasons."""
    raw = read_csv_normalised(path)
    cols = set(raw.columns)
    name = fmt_name or detect_format(cols)
    if name is None:
        diffs = {n: sorted(s["required"] - cols) for n, s in FORMATS.items()}
        raise ValidationError(f"{path.name}: unrecognised NSE format. Columns found: {sorted(cols)}. "
                              f"Missing columns per known format: {diffs}. NSE may have changed the file layout.")
    spec = FORMATS[name]
    missing = sorted(spec["required"] - cols)
    if missing:
        raise ValidationError(f"{path.name}: missing required columns for format '{name}': {missing}")
    if len(raw) < min_rows:
        raise ValidationError(f"{path.name}: only {len(raw)} rows (< {min_rows}). A full-market bhavcopy is much larger — "
                              "the file looks truncated, partial or from a failed download.")

    df = pd.DataFrame({new: raw[old] for old, new in spec["map"].items() if old in raw.columns})
    df["symbol"] = df["symbol"].astype("string").str.upper()
    df["series"] = df["series"].astype("string").str.upper()
    df["date"] = parse_dates(df["date"], path.name)
    for c in ("open", "high", "low", "close", "prev_close", "volume", "delivery_qty", "delivery_pct",
              "turnover_lacs", "turnover_inr"):
        if c in df.columns:
            df[c] = to_num(df[c])
    if "turnover_lacs" in df.columns:
        df["traded_value_cr"] = df["turnover_lacs"] / 100.0           # 1 crore = 100 lakh
    else:
        df["traded_value_cr"] = df["turnover_inr"] / 1e7              # 1 crore = 1e7 INR
    for c in ("delivery_qty", "delivery_pct", "isin"):
        if c not in df.columns:
            df[c] = pd.NA if c == "isin" else np.nan

    warnings: list[str] = []
    dup = df.duplicated(["symbol", "series"], keep="first")
    if dup.any():
        warnings.append(f"{int(dup.sum())} duplicate symbol/series rows dropped (first kept).")
        df = df[~dup].copy()

    dq = pd.Series("", index=df.index, dtype="object")

    def flag(mask: pd.Series, text: str) -> None:
        nonlocal dq
        dq = dq.where(~mask, dq + text + ";")

    flag(df[["open", "high", "low", "close"]].isna().any(axis=1), "missing_ohlc")
    flag(df["prev_close"].isna(), "missing_prev_close")
    flag((df["close"] <= 0) | (df["open"] <= 0), "non_positive_price")
    flag(df["high"] < df["low"], "high_below_low")
    tol = 1e-6
    flag((df["close"] > df["high"] * (1 + tol)) | (df["close"] < df["low"] * (1 - tol)), "close_outside_range")
    flag(df["volume"].isna() | (df["volume"] < 0), "bad_volume")
    bad_deliv = df["delivery_pct"].notna() & ((df["delivery_pct"] < 0) | (df["delivery_pct"] > 100))
    df.loc[bad_deliv, "delivery_pct"] = np.nan
    flag(bad_deliv, "bad_delivery_pct")
    df["data_quality"] = dq.str.rstrip(";")

    dates = sorted(df["date"].dropna().unique())
    if not dates:
        raise ValidationError(f"{path.name}: no valid dates after parsing.")
    if len(dates) > 1:
        warnings.append(f"File contains {len(dates)} different dates; using the latest ({pd.Timestamp(dates[-1]).date()}).")
        df = df[df["date"] == dates[-1]].copy()
    src_date = pd.Timestamp(dates[-1]).date()
    df["source_date"] = src_date.isoformat()
    meta = {"file": path.name, "format": name, "rows": int(len(df)), "source_date": src_date.isoformat(),
            "has_delivery": spec["has_delivery"], "warnings": warnings,
            "rows_with_quality_flags": int((df["data_quality"] != "").sum()),
            "delivery_rows": int(df["delivery_pct"].notna().sum())}
    df["source_format"] = name
    return df, meta


def discover_price_files(data_dir: Path) -> list[Path]:
    found: list[Path] = []
    for spec in FORMATS.values():
        found += sorted(p for p in data_dir.glob(spec["glob"]) if p.suffix.lower() in (".csv", ".zip"))
    return found


def load_best_price_file(data_dir: Path, min_rows: int) -> tuple[pd.DataFrame | None, dict]:
    """Parse all candidate files; prefer the latest source date, then the format with delivery data."""
    candidates = discover_price_files(data_dir)
    info: dict = {"candidates": [p.name for p in candidates], "ignored": []}
    if not candidates:
        return None, info
    best: tuple | None = None
    for p in candidates:
        df, meta = parse_price_file(p, min_rows=min_rows)   # any invalid candidate fails loudly
        key = (meta["source_date"], meta["has_delivery"])
        if best is None or key > best[0]:
            best = (key, df, meta, p)
    for p in candidates:
        if p != best[3]:
            info["ignored"].append(p.name)
    df, meta = best[1], best[2]
    df["source_file"] = best[3].name
    info.update(meta)
    return df, info


# --------------------------------------------------------------------------- companions
def load_security_master(path: Path) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"{path.name} not found — security status will be UNVERIFIED."
    raw = read_csv_normalised(path) if path.stat().st_size > 0 else pd.DataFrame()
    if raw.empty and len(raw.columns) == 0:
        return None, f"{path.name} is empty — security status will be UNVERIFIED."
    cols = set(raw.columns)
    if not MASTER_COLUMNS_REQUIRED <= cols:
        raise ValidationError(f"{path.name}: missing required columns {sorted(MASTER_COLUMNS_REQUIRED - cols)}; "
                              f"found {sorted(cols)}. Expected NSE's EQUITY_L.csv layout.")
    if raw.empty:
        return None, f"{path.name} has headers but no rows (placeholder) — security status will be UNVERIFIED. " \
                     "Replace it with NSE's EQUITY_L.csv."
    m = pd.DataFrame({"symbol": raw["SYMBOL"].str.upper(), "master_series": raw["SERIES"].str.upper()})
    return m.drop_duplicates("symbol"), f"{len(m)} listed securities loaded from {path.name}."


def load_symbol_csv(path: Path, value_aliases: dict[str, list[str]]) -> tuple[pd.DataFrame | None, str]:
    if not path.exists():
        return None, f"{path.name} not provided."
    raw = read_csv_normalised(path)
    if raw.empty:
        return None, f"{path.name} has no rows."
    cols = list(raw.columns)
    sym_col = next((c for c in ("SYMBOL", "TCKRSYMB") if c in cols), None)
    if sym_col is None:
        raise ValidationError(f"{path.name}: needs a SYMBOL column; found {cols}.")
    out = pd.DataFrame({"symbol": raw[sym_col].str.upper()})
    for target, aliases in value_aliases.items():
        col = next((a for a in aliases if a in cols), None)
        if col is None:
            raise ValidationError(f"{path.name}: needs one of {aliases} for '{target}'; found {cols}.")
        out[target] = raw[col]
    if "STAGE" in cols:
        out["stage"] = raw["STAGE"]
    return out.drop_duplicates("symbol"), f"{len(out)} rows loaded from {path.name}."


# --------------------------------------------------------------------------- enrichment
def classify_circuit(close: float, prev: float, high: float, low: float, band: float, tol: float) -> str:
    """Approximate circuit detection: move within `tol` points of the band AND close at the day's extreme."""
    if not np.isfinite(band) or not all(np.isfinite(v) for v in (close, prev, high, low)) or prev <= 0:
        return "UNKNOWN"
    chg = (close - prev) / prev * 100.0
    if chg >= band - tol and close >= high * (1 - 1e-4):
        return "UPPER_CIRCUIT"
    if chg <= -(band - tol) and close <= low * (1 + 1e-4):
        return "LOWER_CIRCUIT"
    return "NONE"


def enrich(df: pd.DataFrame, master: pd.DataFrame | None, band: pd.DataFrame | None, surv: pd.DataFrame | None,
           tol: float) -> pd.DataFrame:
    df = df.copy()
    if master is not None:
        df = df.merge(master, on="symbol", how="left")
        df["security_status"] = np.where(df["master_series"].notna(), "LISTED_IN_MASTER", "NOT_IN_MASTER")
    else:
        df["master_series"] = pd.NA
        df["security_status"] = "UNVERIFIED"
    if band is not None:
        b = band.rename(columns={"band": "price_band_pct"}).copy()
        b["price_band_pct"] = pd.to_numeric(b["price_band_pct"].astype("string").str.replace("%", "", regex=False),
                                            errors="coerce").astype("float64")
        df = df.merge(b[["symbol", "price_band_pct"]], on="symbol", how="left")
    else:
        df["price_band_pct"] = np.nan
    df["price_band_pct"] = df["price_band_pct"].astype("float64")
    df["circuit_status"] = [classify_circuit(c, p, h, l, b, tol) for c, p, h, l, b in
                            zip(df["close"], df["prev_close"], df["high"], df["low"], df["price_band_pct"])]
    if surv is not None:
        s = surv.copy()
        s["surveillance"] = s["list"].astype("string").str.upper() + \
            s.get("stage", pd.Series("", index=s.index)).fillna("").astype("string").map(lambda x: f" {x}" if x else "")
        df = df.merge(s[["symbol", "surveillance"]], on="symbol", how="left")
        df["surveillance"] = df["surveillance"].fillna("NONE")   # NONE = checked against the supplied list, not on it
    else:
        df["surveillance"] = "NO_DATA"                            # NO_DATA = no surveillance list was supplied
    return df


# --------------------------------------------------------------------------- optional download
NSE_ARCHIVE_FULL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
NSE_ARCHIVE_MASTER = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"


def try_download(data_dir: Path, today: date, lookback: int, min_rows: int) -> list[str]:
    """Best-effort fetch of NSE's public static archive files. No cookies, no tricks, no retries on 403.

    These URLs are NSE's documented download locations, but NSE may block cloud IPs (e.g. GitHub runners) or
    change paths at any time — so every response is validated and failures are reported, never hidden.
    """
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (TradeSentinel personal research; contact: repo owner)",
               "Accept": "text/csv,*/*"}
    msgs: list[str] = []
    got = False
    d = today
    tried = 0
    while tried < lookback:
        if d.weekday() < 5:
            tried += 1
            url = NSE_ARCHIVE_FULL.format(ddmmyyyy=d.strftime("%d%m%Y"))
            try:
                r = requests.get(url, headers=headers, timeout=25)
                if r.status_code == 200 and b"SYMBOL" in r.content[:400].upper():
                    tmp = data_dir / f".dl_{d:%d%m%Y}.csv"
                    tmp.write_bytes(r.content)
                    try:
                        parse_price_file(tmp, min_rows=min_rows)
                    except ValidationError as exc:
                        msgs.append(f"{url}: downloaded but failed validation ({exc}); discarded.")
                        tmp.unlink(missing_ok=True)
                    else:
                        final = data_dir / f"sec_bhavdata_full_{d:%d%m%Y}.csv"
                        tmp.replace(final)
                        write_json(data_dir / f"{final.name}.download.json",
                                   {"url": url, "downloaded_utc": now_utc().isoformat(timespec="seconds"),
                                    "http_status": 200, "bytes": len(r.content)})
                        msgs.append(f"Downloaded {final.name} from {url}.")
                        got = True
                        break
                else:
                    msgs.append(f"{url}: HTTP {r.status_code} (no usable file).")
            except Exception as exc:  # noqa: BLE001
                msgs.append(f"{url}: {type(exc).__name__}: {exc}")
        d -= timedelta(days=1)
    if not got:
        msgs.append("NSE download FAILED for all tried dates; place the file in data/ manually (see README).")
    try:
        r = requests.get(NSE_ARCHIVE_MASTER, headers=headers, timeout=25)
        if r.status_code == 200 and b"SYMBOL" in r.content[:200].upper() and r.content.count(b"\n") > 500:
            (data_dir / "nse_security_master.csv").write_bytes(r.content)
            msgs.append("Security master refreshed from NSE EQUITY_L.csv.")
        else:
            msgs.append(f"Security master download not usable (HTTP {r.status_code}); existing file kept.")
    except Exception as exc:  # noqa: BLE001
        msgs.append(f"Security master download failed: {type(exc).__name__}: {exc}; existing file kept.")
    return msgs


# --------------------------------------------------------------------------- reporting
def download_timestamp(data_dir: Path, filename: str) -> str:
    sidecar = data_dir / f"{filename}.download.json"
    if sidecar.exists():
        try:
            import json
            return json.loads(sidecar.read_text())["downloaded_utc"] + " (recorded at download)"
        except Exception:  # noqa: BLE001
            pass
    p = data_dir / filename
    return datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds") + " (file modified time)"


def render_report(rep: dict) -> str:
    L = ["# TradeSentinel — NSE Data Report", "", f"Generated: {rep['generated_at_utc']} UTC", ""]
    L += [f"- **Price file:** `{rep['file']}` (format `{rep['format']}`)",
          f"- **Source (trade) date:** {rep['source_date']}  ({rep['source_age_days']} day(s) old; limit {rep['max_stale_days']})",
          f"- **Download/obtained:** {rep['download_ts']}",
          f"- **Rows processed:** {rep['rows']}  (EQ series: {rep['eq_rows']})",
          f"- **Delivery data:** {'available' if rep['has_delivery'] else 'NOT in this file — delivery % will be blank'} "
          f"({rep['delivery_rows']} rows)",
          f"- **Security master:** {rep['master_note']}",
          f"- **Price-band data:** {rep['band_note']}",
          f"- **Surveillance (ASM/GSM) data:** {rep['surv_note']}",
          f"- **Rows with data-quality flags:** {rep['rows_with_quality_flags']}", ""]
    if rep["warnings"]:
        L += ["## Warnings", ""] + [f"- {w}" for w in rep["warnings"]] + [""]
    if rep.get("download_messages"):
        L += ["## Download attempt", ""] + [f"- {m}" for m in rep["download_messages"]] + [""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dir_args(ap)
    ap.add_argument("--optional", action="store_true", help="exit 0 (SKIPPED) if no NSE price file is present")
    ap.add_argument("--try-download", action="store_true", help="try NSE public archive URLs first (best effort)")
    args = ap.parse_args(argv)
    config_dir, data_dir, reports_dir = dirs_from_args(args)
    try:
        cfg = load_yaml(config_dir / "screening.yml")["nse"]
    except Exception as exc:  # noqa: BLE001
        gh_error(f"nse_data: invalid configuration: {exc}")
        update_status(reports_dir, PHASE, "FAILED", str(exc))
        return 1

    dl_msgs: list[str] = []
    if args.try_download:
        dl_msgs = try_download(data_dir, today_ist(), int(cfg["download_lookback_days"]), int(cfg["min_rows_full_market"]))
        for m in dl_msgs:
            print(f"[nse_data] {m}")

    try:
        df, info = load_best_price_file(data_dir, int(cfg["min_rows_full_market"]))
    except ValidationError as exc:
        gh_error(f"nse_data: INVALID INPUT — {exc}")
        update_status(reports_dir, PHASE, "FAILED", str(exc))
        return 1
    if df is None:
        msg = ("No NSE price file found in data/ (expected sec_bhavdata_full*.csv, BhavCopy_NSE_CM_*.csv or cm*bhav.csv). "
               "Phase 2 skipped; NSE-dependent checks (series, delivery %, circuits) will be UNKNOWN.")
        if args.optional:
            print(f"[nse_data] SKIPPED: {msg}")
            gh_warning(f"nse_data SKIPPED: {msg}")
            update_status(reports_dir, PHASE, "SKIPPED", msg)
            return 0
        gh_error(f"nse_data: {msg}")
        update_status(reports_dir, PHASE, "FAILED", msg)
        return 1

    try:
        master, master_note = load_security_master(data_dir / "nse_security_master.csv")
        band, band_note = load_symbol_csv(data_dir / "nse_price_band.csv", {"band": ["PRICE_BAND_PCT", "PRICE_BAND", "BAND"]})
        surv, surv_note = load_symbol_csv(data_dir / "nse_surveillance.csv", {"list": ["LIST", "SURVEILLANCE", "TYPE"]})
    except ValidationError as exc:
        gh_error(f"nse_data: INVALID INPUT — {exc}")
        update_status(reports_dir, PHASE, "FAILED", str(exc))
        return 1

    out = enrich(df, master, band, surv, float(cfg["circuit_tolerance_pct"]))
    out["download_ts"] = download_timestamp(data_dir, info["file"])
    out = out[OUT_COLUMNS].sort_values(["symbol", "series"]).reset_index(drop=True)
    out.to_csv(reports_dir / "nse_eod_processed.csv", index=False)

    src_date = date.fromisoformat(info["source_date"])
    age = (today_ist() - src_date).days
    warnings = list(info.get("warnings", []))
    if info.get("ignored"):
        warnings.append(f"Other price files ignored (older or without delivery data): {', '.join(info['ignored'])}")
    if age > int(cfg["max_stale_days"]):
        warnings.append(f"STALE SOURCE: NSE file is dated {src_date} ({age} days old; limit {cfg['max_stale_days']}). "
                        "Download the latest bhavcopy.")
    if not info["has_delivery"]:
        warnings.append("This file format has no delivery data; use sec_bhavdata_full for delivery %.")
    if master is None:
        warnings.append(master_note)
    rep = {"generated_at_utc": now_utc().isoformat(timespec="seconds"), "file": info["file"], "format": info["format"],
           "source_date": info["source_date"], "source_age_days": age, "max_stale_days": cfg["max_stale_days"],
           "download_ts": out["download_ts"].iloc[0] if len(out) else "n/a", "rows": int(len(out)),
           "eq_rows": int((out["series"] == "EQ").sum()), "has_delivery": info["has_delivery"],
           "delivery_rows": info["delivery_rows"], "master_note": master_note, "band_note": band_note,
           "surv_note": surv_note, "rows_with_quality_flags": info["rows_with_quality_flags"], "warnings": warnings,
           "download_messages": dl_msgs, "candidates": info["candidates"]}
    write_json(reports_dir / "nse_data_report.json", rep)
    (reports_dir / "nse_data_report.md").write_text(render_report(rep), encoding="utf-8")
    status = "DEGRADED" if warnings else "OK"
    update_status(reports_dir, PHASE, status, f"{info['file']} dated {src_date}, {len(out)} rows" +
                  (f"; {len(warnings)} warning(s)" if warnings else ""))
    for w in warnings:
        gh_warning(f"nse_data: {w}")
    print(f"[nse_data] {info['file']} ({info['format']}) dated {src_date}: {len(out)} rows, {rep['eq_rows']} EQ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
