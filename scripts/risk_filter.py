#!/usr/bin/env python3
"""Phase 4 — Risk and exit-eligibility filter.

Reads reports/technical_screen_full.csv and assigns every symbol PASS / WARNING / REJECT with exact reasons.

PASS means "no rule below was violated on the data available". It is NOT a safety guarantee:
EQ series does not guarantee same-day exit or adequate liquidity. Circuits, a missing bid side, thin order-book
depth and exchange restrictions can still prevent you from selling. T+1 settlement and trade-to-trade
classification are different concepts and must not be confused.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (add_dir_args, dirs_from_args, fmt, gh_error, gh_warning, load_yaml, now_utc, update_status,
                    write_json)

PHASE = "risk_filter"
MISSING = ("", "nan", "none", "unknown", "<na>")


def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and not np.isfinite(v)) or str(v).strip().lower() in MISSING


def _f(v) -> float:
    try:
        x = float(v)
        return x if np.isfinite(x) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def evaluate_risk(row: dict, cfg: dict) -> dict:
    """Pure function: row dict + 'risk' config -> {status, liquidity_status, reject, warn, notes}."""
    reject: list[str] = []
    warn: list[str] = []
    notes: list[str] = []

    def escalate(action: str, msg: str) -> None:
        (reject if action == "reject" else warn).append(msg)

    # ---- trading series
    series = "" if _is_missing(row.get("series")) else str(row["series"]).strip().upper()
    eligible = [s.upper() for s in cfg["eligible_series"]]
    restricted = {k.upper(): v for k, v in cfg["restricted_series"].items()}
    if not series:
        reject.append("Series unknown/unverified: rejected until verified against NSE data (no NSE file or symbol missing).")
    elif series in eligible:
        pass
    elif series in restricted:
        reject.append(f"Series {series} is {restricted[series]}: excluded from the normal delivery shortlist.")
    else:
        reject.append(f"Series '{series}' is not recognised: rejected until verified.")
    mseries = "" if _is_missing(row.get("master_series")) else str(row["master_series"]).strip().upper()
    if series in eligible and mseries and mseries != series:
        warn.append(f"Series conflict: bhavcopy says {series} but the security master lists {mseries}. Verify on NSE.")

    # ---- data availability / freshness
    status = str(row.get("data_status", "NO_DATA"))
    if status == "NO_DATA" or _is_missing(row.get("last_close")):
        reject.append("No usable price data: cannot assess liquidity or risk.")
    if status == "STALE":
        escalate(cfg["stale_action"], f"Stale price data (last bar {row.get('data_date')}, {row.get('data_age_days')} days old).")
    if str(row.get("price_mismatch", "")).lower() == "true":
        warn.append("Price mismatch between Yahoo and NSE closes: possible corporate action or bad data.")
    if not _is_missing(row.get("nse_data_quality")):
        warn.append(f"NSE row has data-quality flags: {row['nse_data_quality']}.")

    # ---- price
    close = _f(row.get("last_close"))
    if np.isfinite(close) and close < float(cfg["min_price"]):
        reject.append(f"Price ₹{close:.2f} is below the minimum ₹{float(cfg['min_price']):g}.")

    # ---- liquidity
    min_tv = float(cfg["min_avg_traded_value_cr"])
    tv, vol = _f(row.get("avg_traded_value_cr20")), _f(row.get("avg_vol20"))
    liquidity = "UNKNOWN"
    if not np.isfinite(tv):
        reject.append("Average traded value unavailable: liquidity cannot be verified.")
    elif tv < min_tv:
        liquidity = "LOW"
        reject.append(f"Low liquidity: 20-day average traded value ₹{tv:.2f} cr < minimum ₹{min_tv:g} cr.")
    elif tv < min_tv * float(cfg["thin_liquidity_multiple"]):
        liquidity = "THIN"
        warn.append(f"Thin liquidity: ₹{tv:.2f} cr average traded value is under {cfg['thin_liquidity_multiple']:g}x the minimum.")
    else:
        liquidity = "OK" if tv < min_tv * 5 else "HIGH"
    if not np.isfinite(vol):
        reject.append("Average volume unavailable: liquidity cannot be verified.")
    elif vol < float(cfg["min_avg_volume"]):
        liquidity = "LOW"
        reject.append(f"Low volume: 20-day average {vol:,.0f} shares < minimum {float(cfg['min_avg_volume']):,.0f}.")

    # ---- spread (only if data exists)
    bid, ask = _f(row.get("bid")), _f(row.get("ask"))
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask >= bid:
        spread = (ask - bid) / ((ask + bid) / 2) * 100
        if spread > float(cfg["max_spread_pct"]):
            warn.append(f"Wide bid/ask spread {spread:.2f}% > {float(cfg['max_spread_pct']):g}%.")
    else:
        notes.append("Bid/ask spread not assessed (no reliable bid/ask data in the free files).")

    # ---- delivery
    dp = _f(row.get("delivery_pct"))
    if not np.isfinite(dp):
        warn.append("Delivery % unavailable (use NSE sec_bhavdata_full): insufficient delivery data.")
    elif dp < float(cfg["min_delivery_pct"]):
        warn.append(f"Low delivery {dp:.1f}% < {float(cfg['min_delivery_pct']):g}%: move may be speculative/intraday-driven.")

    # ---- circuits
    cs = str(row.get("circuit_status", "UNKNOWN")).upper()
    if cs == "LOWER_CIRCUIT":
        reject.append("Hit LOWER circuit on the latest session: exit may be impossible (sellers only).")
    elif cs == "UPPER_CIRCUIT":
        escalate(cfg["upper_circuit_action"], "Hit UPPER circuit on the latest session: you may not be able to buy, "
                                              "and the order book can flip violently.")
    elif cs == "UNKNOWN" or _is_missing(row.get("price_band_pct")):
        notes.append("Circuit status not assessed (no price-band data supplied in data/nse_price_band.csv).")
    ups, dns = _f(row.get("circuit_up_days")), _f(row.get("circuit_down_days"))
    reps = float(cfg["repeated_circuit_days"])
    if np.isfinite(ups) and np.isfinite(dns) and ups + dns >= reps:
        warn.append(f"Repeated circuit-like sessions: {int(ups)} upper / {int(dns)} lower in the lookback window (approximate).")

    # ---- surveillance
    surv = "" if _is_missing(row.get("surveillance")) else str(row["surveillance"]).strip()
    if surv.upper() == "NO_DATA" or not surv:
        notes.append("ASM/GSM surveillance not assessed (no data/nse_surveillance.csv supplied) — check NSE manually.")
    elif surv.upper() != "NONE":
        escalate(cfg["surveillance_action"], f"Surveillance classification: {surv}. Manual review required.")

    # ---- security master
    ss = str(row.get("security_status", "UNVERIFIED")).upper()
    if ss == "NOT_IN_MASTER":
        warn.append("Symbol not found in the NSE security master: listing status unverified (suspended/delisted/renamed?).")
    elif ss in ("UNVERIFIED", "NAN", ""):
        notes.append("Security status unverified (security master not loaded).")

    final = "REJECT" if reject else "WARNING" if warn else "PASS"
    return {"status": final, "liquidity_status": liquidity, "reject_reasons": reject, "warning_reasons": warn, "notes": notes}


def render_markdown(rep: dict, df: pd.DataFrame) -> str:
    c = rep["counts"]
    L = ["# TradeSentinel — Risk & Exit-Eligibility Report", "", f"Generated: {rep['generated_at_utc']} UTC", "",
         f"**PASS {c['PASS']} · WARNING {c['WARNING']} · REJECT {c['REJECT']}** (of {rep['symbols']} screened)", "",
         "> **PASS does not mean safe.** It means no rule below was violated on the data available. EQ series does not "
         "guarantee same-day exit or adequate liquidity: circuits, missing buyers, thin order-book depth and exchange "
         "restrictions can still trap you. T+1 settlement and trade-to-trade classification are different concepts.", "",
         "## Rules applied", "",
         "- **REJECT:** series not EQ (BE/BZ/other restricted = excluded; unknown = rejected until verified), no/stale-unusable data, "
         "price below minimum, average traded value or volume below minimum (or unverifiable), lower circuit on the latest session.",
         "- **WARNING:** thin liquidity, low or missing delivery %, upper circuit, repeated circuit-like sessions, ASM/GSM listing, "
         "stale data, Yahoo-vs-NSE price mismatch, symbol not in security master, wide spread (if data exists).",
         "- **Not assessed unless you supply data:** circuits/price bands (`data/nse_price_band.csv`), ASM/GSM (`data/nse_surveillance.csv`), bid/ask spreads.", ""]
    for name in ("REJECT", "WARNING", "PASS"):
        sub = df[df["risk_status"] == name]
        L += [f"## {name} ({len(sub)})", ""]
        if sub.empty:
            L += ["_None._", ""]
            continue
        for _, r in sub.iterrows():
            L.append(f"### {r['symbol']} — series {r['series'] if not pd.isna(r['series']) else 'UNKNOWN'} — liquidity {r['liquidity_status']}")
            for k, label in (("reject_reasons", "Reject"), ("warning_reasons", "Warning"), ("notes", "Note")):
                for item in [x for x in str(r[k]).split(" | ") if x and x != "nan"]:
                    L.append(f"- **{label}:** {item}")
            if not any(str(r[k]) not in ("", "nan") for k in ("reject_reasons", "warning_reasons")):
                L.append("- No rule violated on available data (not a safety guarantee).")
            L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dir_args(ap)
    args = ap.parse_args(argv)
    config_dir, _, reports_dir = dirs_from_args(args)
    src = reports_dir / "technical_screen_full.csv"
    try:
        cfg = load_yaml(config_dir / "screening.yml")["risk"]
    except Exception as exc:  # noqa: BLE001
        gh_error(f"risk_filter: invalid configuration: {exc}")
        update_status(reports_dir, PHASE, "FAILED", str(exc))
        return 1
    if not src.exists():
        msg = f"{src.name} not found — technical screening did not produce output, so risk filtering cannot run."
        gh_error(f"risk_filter: {msg}")
        update_status(reports_dir, PHASE, "FAILED", msg)
        return 1
    tech = pd.read_csv(src, dtype={"symbol": str, "series": str, "surveillance": str, "master_series": str})
    if tech.empty:
        gh_error("risk_filter: technical_screen_full.csv has no rows.")
        update_status(reports_dir, PHASE, "FAILED", "technical output empty")
        return 1

    results = []
    for rec in tech.to_dict("records"):
        r = evaluate_risk(rec, cfg)
        results.append({"risk_status": r["status"], "liquidity_status": r["liquidity_status"],
                        "reject_reasons": " | ".join(r["reject_reasons"]), "warning_reasons": " | ".join(r["warning_reasons"]),
                        "notes": " | ".join(r["notes"])})
    res = pd.DataFrame(results)
    out = pd.concat([tech[["symbol", "series", "last_close", "avg_traded_value_cr20", "avg_vol20", "delivery_pct",
                           "circuit_status", "surveillance", "data_status", "technical_pass", "shortlisted"]], res], axis=1)
    out.to_csv(reports_dir / "risk_report.csv", index=False)

    counts = {k: int((out["risk_status"] == k).sum()) for k in ("PASS", "WARNING", "REJECT")}
    rep = {"generated_at_utc": now_utc().isoformat(timespec="seconds"), "symbols": len(out), "counts": counts,
           "config": cfg, "stocks": out.to_dict("records"),
           "disclaimer": "PASS is not a safety guarantee; EQ series does not guarantee an exit."}
    write_json(reports_dir / "risk_report.json", rep)
    (reports_dir / "risk_report.md").write_text(render_markdown(rep, out), encoding="utf-8")

    # final shortlist = indicator shortlist that is not REJECTed
    merged = tech.merge(res.assign(symbol=tech["symbol"]), on="symbol")
    final = merged[(merged["shortlisted"] == True) & (merged["risk_status"] != "REJECT")]   # noqa: E712
    final.sort_values("rel_volume", ascending=False).to_csv(reports_dir / "final_shortlist.csv", index=False)
    update_status(reports_dir, PHASE, "OK", f"PASS {counts['PASS']}, WARNING {counts['WARNING']}, REJECT {counts['REJECT']}; "
                  f"final shortlist {len(final)}")
    print(f"[risk_filter] {counts} final_shortlist={len(final)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
