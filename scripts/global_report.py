#!/usr/bin/env python3
"""Phase 1 — Global market intelligence.

Fetches daily data from Yahoo Finance (via yfinance), computes per-factor changes,
validates the ^TNX quote convention, builds a bounded (-100..+100) heuristic score
and writes reports/global_report.md and reports/global_report.json.
"""
from __future__ import annotations

import argparse
import operator
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common import (IST, add_dir_args, dirs_from_args, fmt, gh_error, gh_warning, load_yaml, now_utc,
                    update_status, write_json)

PHASE = "global_report"
OPS = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt}


# --------------------------------------------------------------------------- config
def validate_config(cfg: dict) -> None:
    errors: list[str] = []
    for key in ("settings", "factors", "sectors"):
        if key not in cfg:
            errors.append(f"missing top-level key '{key}'")
    factors = cfg.get("factors", {}) or {}
    for key, f in factors.items():
        for req in ("name", "ticker", "weight", "direction", "kind"):
            if req not in f:
                errors.append(f"factor '{key}' missing '{req}'")
        if f.get("kind") not in ("pct", "yield_bp"):
            errors.append(f"factor '{key}': kind must be 'pct' or 'yield_bp'")
        if f.get("direction") not in (1, -1):
            errors.append(f"factor '{key}': direction must be 1 or -1")
        w = f.get("weight")
        if not isinstance(w, (int, float)) or isinstance(w, bool) or w < 0:
            errors.append(f"factor '{key}': weight must be a number >= 0")
        full = f.get("full_move_bp") if f.get("kind") == "yield_bp" else f.get("full_move_pct")
        if not isinstance(full, (int, float)) or full <= 0:
            errors.append(f"factor '{key}': full_move_{'bp' if f.get('kind') == 'yield_bp' else 'pct'} must be > 0")
    for sector, s in (cfg.get("sectors", {}) or {}).items():
        for fk in (s.get("drivers") or {}):
            if fk not in factors:
                errors.append(f"sector '{sector}' references unknown factor '{fk}'")
    for i, rf in enumerate(cfg.get("risk_flags", []) or []):
        if rf.get("factor") not in factors or rf.get("op") not in OPS or rf.get("metric") not in (
                "latest_close", "pct_change", "change_bp"):
            errors.append(f"risk_flags[{i}] is invalid: {rf}")
    if errors:
        raise ValueError("Invalid market_impact.yml:\n - " + "\n - ".join(errors))


# --------------------------------------------------------------------------- data fetch
def default_fetcher(ticker: str, period: str) -> pd.DataFrame:
    """Fetch daily bars from Yahoo Finance with simple retries. Raises on persistent errors."""
    import yfinance as yf  # imported lazily so unit tests do not need network libs

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # noqa: BLE001 - network/library errors vary
            last_err = exc
        time.sleep(1.5 * (attempt + 1))
    if last_err:
        raise last_err
    return pd.DataFrame()


# --------------------------------------------------------------------------- ^TNX convention
def resolve_yield_convention(latest_raw: float, setting: str = "auto") -> tuple[str, list[str]]:
    """Decide whether ^TNX is quoted in percent (4.25) or tenths of a percent (42.5).

    Yahoo historically quoted ^TNX as yield x 10 and now generally quotes the yield in percent.
    Instead of silently assuming either, we auto-detect from magnitude (a 10-year yield above 15%
    is implausible, so raw >= 15 implies tenths), record the decision, and flag ambiguous ranges.
    """
    if setting == "percent":
        return "percent", [f"Convention forced to 'percent' by config (raw {latest_raw:.3f} => {latest_raw:.3f}%)."]
    if setting == "tenths":
        return "tenths", [f"Convention forced to 'tenths' by config (raw {latest_raw:.3f} => {latest_raw / 10:.3f}%)."]
    if setting != "auto":
        raise ValueError(f"tnx_convention must be auto|percent|tenths, got {setting!r}")
    notes: list[str] = []
    if latest_raw >= 15:
        conv = "tenths"
        notes.append(f"Auto-detected tenths-of-percent quoting: raw {latest_raw:.2f} => yield {latest_raw / 10:.3f}%.")
    else:
        conv = "percent"
        notes.append(f"Auto-detected percent quoting: raw {latest_raw:.3f} => yield {latest_raw:.3f}%.")
    if 8 <= latest_raw < 15:
        notes.append("AMBIGUOUS: raw value is in the 8-15 range, which could be a 8-15% yield or a 0.8-1.5% yield "
                     "(tenths). Set settings.tnx_convention explicitly in market_impact.yml.")
    return conv, notes


# --------------------------------------------------------------------------- per-factor summary
def summarize_factor(key: str, spec: dict, df: pd.DataFrame | None, now: datetime, settings: dict) -> dict:
    out: dict = {
        "key": key, "name": spec["name"], "ticker": spec["ticker"], "kind": spec["kind"],
        "weight": float(spec["weight"]), "direction": int(spec["direction"]), "status": "OK", "notes": [],
        "latest_close": None, "previous_close": None, "abs_change": None, "pct_change": None, "change_bp": None,
        "move": None, "move_unit": "bp" if spec["kind"] == "yield_bp" else "%",
        "data_timestamp": None, "data_date": None, "age_days": None, "stale": None,
    }
    if df is None or len(df) == 0 or "Close" not in df.columns:
        out["status"] = "MISSING"
        out["notes"].append("No data returned by the source.")
        return out
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 2:
        out["status"] = "INSUFFICIENT_HISTORY"
        out["notes"].append(f"Only {len(close)} valid close(s); need at least 2 to compute a change.")
        return out
    latest, prev = float(close.iloc[-1]), float(close.iloc[-2])
    ts = pd.Timestamp(close.index[-1])
    out["data_timestamp"] = ts.isoformat()
    out["data_date"] = ts.date().isoformat()
    if latest <= 0 or prev <= 0:
        out["status"] = "INVALID"
        out["notes"].append(f"Non-positive close in data (latest={latest}, previous={prev}).")
        return out

    if spec["kind"] == "yield_bp":
        conv, notes = resolve_yield_convention(latest, settings.get("tnx_convention", "auto"))
        out["notes"].extend(notes)
        out["tnx_convention"] = conv
        div = 10.0 if conv == "tenths" else 1.0
        y_latest, y_prev = latest / div, prev / div
        ratio = prev / latest
        if not (0 < y_latest < 15 and 0 < y_prev < 15) or not (0.2 < ratio < 5):
            out["status"] = "INVALID"
            out["notes"].append(f"Converted yields look implausible (latest={y_latest:.3f}%, previous={y_prev:.3f}%). "
                                "Quote convention could not be validated; factor excluded.")
            return out
        latest, prev = y_latest, y_prev
        out["change_bp"] = (latest - prev) * 100.0
        out["move"] = out["change_bp"]
        if abs(out["change_bp"]) > float(settings.get("tnx_max_plausible_daily_move_bp", 60)):
            out["status"] = "SUSPECT"
            out["notes"].append(f"Daily yield move of {out['change_bp']:.1f}bp exceeds the plausibility limit; "
                                "excluded from the score — verify the data.")

    out["latest_close"], out["previous_close"] = latest, prev
    out["abs_change"] = latest - prev
    out["pct_change"] = (latest - prev) / prev * 100.0
    if spec["kind"] != "yield_bp":
        out["move"] = out["pct_change"]

    age = (now.date() - ts.date()).days
    out["age_days"] = age
    out["stale"] = age > int(settings.get("max_age_days", 4))
    if out["stale"] and out["status"] == "OK":
        out["status"] = "STALE"
        out["notes"].append(f"Latest bar is {age} calendar days old (limit {settings.get('max_age_days', 4)}).")
    return out


# --------------------------------------------------------------------------- scoring
def score_factors(factors: list[dict], cfg: dict) -> dict:
    """Weighted, normalised score in [-100, +100]. Mutates factors with signal/contribution info."""
    settings = cfg["settings"]
    policy = settings.get("stale_policy", "exclude")
    min_cov = float(settings.get("min_coverage_pct", 60))
    thr = settings.get("assessment_thresholds", {"positive": 25, "negative": -25})

    included: list[dict] = []
    total_weight = sum(f["weight"] for f in factors if f["weight"] > 0)
    for f in factors:
        spec = cfg["factors"][f["key"]]
        f.update(raw_signal=None, contribution_points=None, max_points=None, included_in_score=False,
                 exclusion_reason=None)
        usable = f["status"] == "OK" or (f["status"] == "STALE" and policy == "include")
        if usable and f["move"] is not None:
            full = float(spec["full_move_bp"] if spec["kind"] == "yield_bp" else spec["full_move_pct"])
            f["raw_signal"] = float(np.clip(f["move"] / full, -1.0, 1.0))
        if f["weight"] <= 0:
            f["exclusion_reason"] = "weight 0 (tracked only)"
        elif f["raw_signal"] is None:
            f["exclusion_reason"] = f"status {f['status']}"
        else:
            f["included_in_score"] = True
            included.append(f)

    used_weight = sum(f["weight"] for f in included)
    coverage = (used_weight / total_weight * 100.0) if total_weight > 0 else 0.0
    result: dict = {"score": None, "assessment": "INSUFFICIENT_DATA", "coverage_pct": round(coverage, 1),
                    "used_weight": used_weight, "total_weight": total_weight, "factors_used": len(included),
                    "min_coverage_pct": min_cov}
    if used_weight <= 0 or coverage < min_cov:
        for f in included:   # nothing is scored when coverage is insufficient
            f["included_in_score"] = False
            f["exclusion_reason"] = "insufficient overall data coverage; no score produced"
        return result
    points = {f["key"]: f["direction"] * f["weight"] * f["raw_signal"] for f in included}
    score = float(np.clip(100.0 * sum(points.values()) / used_weight, -100.0, 100.0))
    for f in included:
        f["contribution_points"] = 100.0 * points[f["key"]] / used_weight
        f["max_points"] = 100.0 * f["weight"] / used_weight
    result["score"] = round(score, 1)
    result["assessment"] = ("POSITIVE" if score >= thr["positive"] else "NEGATIVE" if score <= thr["negative"]
                            else "MIXED")
    return result


def sector_outlook(cfg: dict, factors: list[dict]) -> list[dict]:
    """Heuristic macro-sensitivity tilt per sector (NOT observed sector performance)."""
    by_key = {f["key"]: f for f in factors}
    thr = float(cfg["settings"].get("sector_threshold", 15))
    rows = []
    for name, s in cfg["sectors"].items():
        num = den = 0.0
        used, contribs = 0, []
        drivers = s.get("drivers", {}) or {}
        for fk, sens in drivers.items():
            f = by_key.get(fk)
            if f and f.get("raw_signal") is not None:
                num += sens * f["raw_signal"]
                den += abs(sens)
                used += 1
                contribs.append((fk, sens * f["raw_signal"]))
        tilt = 100.0 * num / den if den > 0 else None
        label = "NO DATA" if tilt is None else "TAILWIND" if tilt >= thr else "HEADWIND" if tilt <= -thr else "NEUTRAL/MIXED"
        top = sorted(contribs, key=lambda kv: abs(kv[1]), reverse=True)[:2]
        rows.append({"sector": name, "tilt": None if tilt is None else round(tilt, 1), "label": label,
                     "drivers_used": used, "drivers_total": len(drivers), "ambiguous": bool(s.get("ambiguous", False)),
                     "factors": ", ".join(f"{by_key[k]['name']} ({'+' if v >= 0 else ''}{v:.2f})" for k, v in top),
                     "note": s.get("note", ""), "driver_map": drivers})
    return rows


def evaluate_risk_flags(cfg: dict, factors: list[dict]) -> list[str]:
    by_key = {f["key"]: f for f in factors}
    msgs = []
    for rf in cfg.get("risk_flags", []) or []:
        f = by_key.get(rf["factor"])
        if not f or f["status"] != "OK":
            continue
        val = f.get(rf["metric"])
        if val is not None and OPS[rf["op"]](val, rf["value"]):
            msgs.append(f"{rf['message']} ({f['name']}: {rf['metric']}={val:.2f})")
    return msgs


# --------------------------------------------------------------------------- rendering
def render_markdown(rep: dict) -> str:
    sc = rep["score"]
    lines = ["# TradeSentinel — Global Market Report", "",
             f"Generated: {rep['generated_at_ist']} IST ({rep['generated_at_utc']} UTC)  ",
             "Source: Yahoo Finance via `yfinance` (free, unofficial, may be delayed or occasionally wrong).", ""]
    lines += ["## Global impact score", ""]
    if sc["score"] is None:
        lines += [f"**Score: not produced — {sc['assessment']}.** Only {sc['coverage_pct']}% of the configured weight has "
                  f"usable data (minimum {sc['min_coverage_pct']}%). Do not read anything into this report's direction.", ""]
    else:
        lines += [f"**Score: {sc['score']:+.1f} / 100 → {sc['assessment']}**  ",
                  f"Data coverage: {sc['coverage_pct']}% of configured weight ({sc['factors_used']} factors scored).", ""]
    lines += ["> A high positive score is **not** a buy signal. It only says overnight global cues were mostly supportive "
              "for Indian risk appetite. It says nothing about any individual stock's liquidity, trend, valuation or "
              "exit-ability. Use it as context for caution, never as a trigger.", ""]

    lines += ["## Factor-wise contribution", "",
              "| Factor | Ticker | Latest | Previous | Change | Move | Data date | Age (d) | Status | Dir | Weight | Signal | Contribution (pts) |",
              "|---|---|---:|---:|---:|---:|---|---:|---|:-:|---:|---:|---:|"]
    for f in rep["factors"]:
        is_y = f["kind"] == "yield_bp"
        latest = fmt(f["latest_close"], 3) + ("%" if is_y and f["latest_close"] is not None else "")
        prev = fmt(f["previous_close"], 3) + ("%" if is_y and f["previous_close"] is not None else "")
        chg = fmt(f["abs_change"], 3)
        move = (fmt(f["change_bp"], 1) + " bp") if is_y else (fmt(f["pct_change"], 2) + " %")
        if f["move"] is None:
            move = "n/a"
        contrib = "excluded" if not f["included_in_score"] else f"{f['contribution_points']:+.1f}"
        sig = fmt(f["raw_signal"], 2)
        lines.append(f"| {f['name']} | `{f['ticker']}` | {latest} | {prev} | {chg} | {move} | {f['data_date'] or 'n/a'} | "
                     f"{f['age_days'] if f['age_days'] is not None else 'n/a'} | {f['status']} | {'+' if f['direction'] > 0 else '-'} | "
                     f"{f['weight']:g} | {sig} | {contrib} |")
    lines.append("")
    if sc["score"] is not None:
        lines.append(f"Sum of contributions = {sum(f['contribution_points'] or 0 for f in rep['factors']):+.1f} (= score).")
        lines.append("")

    problems = [f for f in rep["factors"] if f["status"] != "OK" or f["notes"]]
    if problems:
        lines += ["## Data quality notes", ""]
        for f in problems:
            lines.append(f"- **{f['name']}** (`{f['ticker']}`) — {f['status']}: " + " ".join(f["notes"] or ["see status"]))
        lines.append("")

    lines += ["## Important macro risks", ""]
    lines += [f"- {m}" for m in rep["macro_risks"]] or ["- No configured risk threshold was breached on usable data."]
    lines.append("")

    lines += ["## Sector impact (heuristic macro-sensitivity, NOT observed sector performance)", "",
              "This table only translates today's global factor moves through the sensitivities in `config/market_impact.yml`. "
              "No sector index data is used, so nothing here describes how a sector actually traded.", "",
              "| Sector | Tilt | Outlook | Main factor drivers today | Why these factors matter |", "|---|---:|---|---|---|"]
    for s in rep["sectors"]:
        tilt = "n/a" if s["tilt"] is None else f"{s['tilt']:+.0f}"
        label = s["label"] + (" (ambiguous)" if s["ambiguous"] and s["label"] != "NO DATA" else "")
        lines.append(f"| {s['sector']} | {tilt} | {label} | {s['factors'] or 'n/a'} | {s['note']} |")
    lines.append("")

    lines += ["## Method", "",
              "For each scored factor *i* with weight *w_i*, direction *d_i* (+1 or -1) and daily move *m_i* "
              "(% change, or basis points for the US 10-year yield):", "",
              "```", "signal_i       = clip(m_i / full_move_i, -1, +1)",
              "points_i       = d_i * w_i * signal_i",
              "score          = 100 * sum(points_i) / sum(w_i)      over factors with usable data",
              "contribution_i = 100 * points_i / sum(w_i)           (contributions add up to the score)", "```", "",
              "- **Bounds:** because |signal| <= 1, the score always lies in [-100, +100].",
              "- **Configuration:** weights, directions and saturation moves are in `config/market_impact.yml`.",
              "- **Missing data:** factors with no usable data are excluded and the remaining weights are re-normalised. "
              f"If usable weight falls below {sc['min_coverage_pct']}% of the configured total, no score is produced.",
              f"- **Stale data:** a latest bar older than {rep['settings']['max_age_days']} calendar days is flagged STALE and "
              f"{'excluded' if rep['settings']['stale_policy'] == 'exclude' else 'included with a warning'} (policy: "
              f"`{rep['settings']['stale_policy']}`).",
              "- **US 10-year yield:** `^TNX` has been quoted both as yield x10 and directly in percent. The script detects the "
              "convention from magnitude, records the decision in the data-quality notes, validates the converted yield (0-15%) "
              "and the day-over-day plausibility, and reports the change in basis points.", "",
              "## Limitations", "",
              "- The directions and weights are **heuristics**, not universal or stable correlations; they can flip in some regimes "
              "(e.g. gold, crude during supply shocks, DXY vs. INR during domestic-driven moves).",
              "- Yahoo Finance data is free but unofficial: it can lag, have gaps, or return partial sessions. Futures tickers "
              "(`BZ=F`, `GC=F`, `SI=F`, `HG=F`) roll contracts and can show roll artefacts.",
              "- Daily closes ignore overnight/early-Asia moves that happen after the US close and before this report runs.",
              "- Sector tilts are macro-sensitivity guesses and ignore company-specific and domestic drivers.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def build_report(cfg: dict, fetcher: Callable[[str, str], pd.DataFrame], now: datetime | None = None) -> dict:
    now = now or now_utc()
    settings = cfg["settings"]
    factors: list[dict] = []
    for key, spec in cfg["factors"].items():
        try:
            df = fetcher(spec["ticker"], settings.get("history_period", "20d"))
            f = summarize_factor(key, spec, df, now, settings)
        except Exception as exc:  # noqa: BLE001 — one failing ticker must not kill the report
            f = summarize_factor(key, spec, None, now, settings)
            f["status"] = "FETCH_ERROR"
            f["notes"] = [f"{type(exc).__name__}: {exc}"]
        factors.append(f)
    score = score_factors(factors, cfg)
    return {
        "generated_at_utc": now.isoformat(timespec="seconds"),
        "generated_at_ist": now.astimezone(IST).strftime("%Y-%m-%d %H:%M"),
        "source": "Yahoo Finance via yfinance",
        "settings": settings, "score": score, "factors": factors,
        "sectors": sector_outlook(cfg, factors), "macro_risks": evaluate_risk_flags(cfg, factors),
        "disclaimer": "Heuristic context only. Not investment advice and not a buy/sell signal.",
    }


def main(argv: list[str] | None = None, fetcher: Callable[[str, str], pd.DataFrame] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_dir_args(ap)
    args = ap.parse_args(argv)
    config_dir, _, reports_dir = dirs_from_args(args)
    try:
        cfg = load_yaml(config_dir / "market_impact.yml")
        validate_config(cfg)
    except Exception as exc:  # noqa: BLE001
        gh_error(f"global_report: invalid configuration: {exc}")
        update_status(reports_dir, PHASE, "FAILED", str(exc))
        return 1

    rep = build_report(cfg, fetcher or default_fetcher)
    write_json(reports_dir / "global_report.json", rep)
    (reports_dir / "global_report.md").write_text(render_markdown(rep), encoding="utf-8")

    usable = [f for f in rep["factors"] if f["status"] in ("OK", "STALE", "SUSPECT")]
    sc = rep["score"]
    if not usable:
        msg = "No factor returned usable data — network or Yahoo outage? Report written but is empty."
        gh_error(f"global_report: {msg}")
        update_status(reports_dir, PHASE, "FAILED", msg)
        return 1
    if sc["score"] is None:
        msg = f"Coverage {sc['coverage_pct']}% < {sc['min_coverage_pct']}%: score not produced."
        gh_warning(f"global_report: {msg}")
        update_status(reports_dir, PHASE, "DEGRADED", msg)
    else:
        bad = [f["name"] for f in rep["factors"] if f["status"] != "OK"]
        update_status(reports_dir, PHASE, "OK" if not bad else "DEGRADED",
                      f"score {sc['score']:+.1f} ({sc['assessment']})" + (f"; problems: {', '.join(bad)}" if bad else ""))
        if bad:
            gh_warning(f"global_report: factors with data problems: {', '.join(bad)}")
    print(f"[global_report] score={sc['score']} assessment={sc['assessment']} coverage={sc['coverage_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
