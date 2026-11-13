#!/usr/bin/env python3
"""Assemble reports/premarket_report.md from the outputs of the other phases.

Never invents data: any section whose inputs are missing says so explicitly. No entry prices, targets or stop losses
are generated anywhere in TradeSentinel.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from common import add_dir_args, dirs_from_args, fmt, gh_warning, now_utc, update_status

PHASE = "final_report"
STATUS_ICON = {"OK": "✅", "DEGRADED": "⚠️", "SKIPPED": "⏭️", "FAILED": "❌"}


def _load_json(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    return None


def _clean(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) == "nan" else str(v)


def candidate_block(r: pd.Series) -> list[str]:
    liquidity = _clean(r.get("liquidity_status")) or "UNKNOWN"
    L = [f"### {r['symbol']}  —  series {_clean(r.get('series')) or 'UNKNOWN'} · risk **{r['risk_status']}** · liquidity {liquidity}", "",
         f"- **Last close:** ₹{fmt(r.get('last_close'))} on {_clean(r.get('data_date')) or 'n/a'} · **Daily change:** {fmt(r.get('daily_pct'))}%",
         f"- **EMA20 / 50 / 200:** {fmt(r.get('ema20'))} / {fmt(r.get('ema50'))} / {fmt(r.get('ema200'))}",
         f"- **RSI14:** {fmt(r.get('rsi14'), 1)} · **MACD:** {_clean(r.get('macd_status'))} · **Relative volume:** {fmt(r.get('rel_volume'))}x",
         f"- **Delivery %:** {fmt(r.get('delivery_pct'), 1)}" + ("" if not pd.isna(r.get('delivery_pct')) else " (not in the NSE file supplied)"),
         f"- **Avg traded value (20d):** ₹{fmt(r.get('avg_traded_value_cr20'))} cr · **Distance from 20d/50d high:** "
         f"{fmt(r.get('dist_high20_pct'))}% / {fmt(r.get('dist_high50_pct'))}%",
         "", "**Reasons for selection**"]
    L += [f"- {x}" for x in _clean(r.get("reasons_pass")).split(" | ") if x]
    L += ["", "**Reasons for caution**"]
    cautions = [x for x in _clean(r.get("warning_reasons")).split(" | ") if x] + \
               [x for x in _clean(r.get("reasons_fail")).split(" | ") if x] + \
               [x for x in _clean(r.get("notes")).split(" | ") if x]
    cautions.append("Indicator agreement is not a forecast; the move may already be extended. Check the chart, news and order book yourself.")
    L += [f"- {x}" for x in cautions] + [""]
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_dir_args(ap)
    args = ap.parse_args(argv)
    _, _, reports = dirs_from_args(args)

    g = _load_json(reports / "global_report.json")
    status = _load_json(reports / "run_status.json") or {}
    L = ["# TradeSentinel — Pre-market Report", "", f"Generated: {now_utc().isoformat(timespec='seconds')} UTC", "",
         "**WAIT. CONFIRM. PROTECT.**", "",
         "> Research output only. No orders are placed. No entry prices, targets or stop losses are generated. "
         "Nothing here is investment advice.", ""]

    L += ["## 1. Run status & data provenance", "", "| Phase | Status | Detail |", "|---|---|---|"]
    for ph in ("global_report", "nse_data", "technical_screen", "risk_filter"):
        s = status.get(ph)
        L.append(f"| {ph} | {STATUS_ICON.get(s['status'], '?')} {s['status']} | {s['message']} |" if s
                 else f"| {ph} | ⏭️ NOT RUN | no status recorded |")
    nse = _load_json(reports / "nse_data_report.json")
    if nse:
        L += ["", f"NSE source: `{nse['file']}` dated **{nse['source_date']}** ({nse['source_age_days']} day(s) old), "
                  f"obtained {nse['download_ts']}."]
    L.append("")

    L += ["## 2. Global market summary", ""]
    if not g:
        L += ["_Global report not available._", ""]
    else:
        sc = g["score"]
        L += [f"**Score: {'n/a' if sc['score'] is None else format(sc['score'], '+.1f')} / 100 — {sc['assessment']}** "
              f"(coverage {sc['coverage_pct']}%). Not a buy signal.", "",
              "| Factor | Move | Status | Contribution (pts) |", "|---|---:|---|---:|"]
        for f in g["factors"]:
            mv = "n/a" if f["move"] is None else (f"{f['change_bp']:+.1f} bp" if f["kind"] == "yield_bp" else f"{f['pct_change']:+.2f}%")
            ct = "excluded" if not f["included_in_score"] else f"{f['contribution_points']:+.1f}"
            L.append(f"| {f['name']} | {mv} | {f['status']} | {ct} |")
        L += ["", "**Important macro risks**", ""]
        L += [f"- {m}" for m in g["macro_risks"]] or ["- None of the configured thresholds was breached."]
        L += ["", "## 3. Sector impact (heuristic macro-sensitivity — not observed performance)", "",
              "| Sector | Tilt | Outlook | Main drivers today |", "|---|---:|---|---|"]
        for s in g["sectors"]:
            L.append(f"| {s['sector']} | {'n/a' if s['tilt'] is None else format(s['tilt'], '+.0f')} | {s['label']}"
                     f"{' (ambiguous)' if s['ambiguous'] else ''} | {s['factors'] or 'n/a'} |")
        L.append("")

    L += ["## 4. Stock shortlist", ""]
    full_p, risk_p, final_p = reports / "technical_screen_full.csv", reports / "risk_report.csv", reports / "final_shortlist.csv"
    if not (full_p.exists() and risk_p.exists() and final_p.exists()):
        L += ["_Technical screening / risk filtering did not run, so there is no shortlist._", ""]
    else:
        final = pd.read_csv(final_p)
        tech = pd.read_csv(full_p)
        risk = pd.read_csv(risk_p)
        L += [f"Screened **{len(tech)}** symbols · indicator pass **{int(tech['technical_pass'].sum())}** · "
              f"final shortlist (indicator pass + EQ + not REJECT) **{len(final)}**.", ""]
        if final.empty:
            L += ["**No candidates today.** That is a valid outcome — waiting is a position. Likely reasons: indicators not aligned, "
                  "series/liquidity rules, or missing NSE data (without it every symbol's series is unknown and cannot be shortlisted).", ""]
        if tech["series"].isna().all() or (tech["series"] == "UNKNOWN").all():
            L += ["⚠️ **No NSE series data was available**, so no stock can be verified as EQ. Provide NSE files (see README).", ""]
        for _, r in final.iterrows():
            r = r.copy()
            L += candidate_block(r)
        rej = risk[(risk["technical_pass"] == True) & (risk["risk_status"] == "REJECT")]   # noqa: E712
        if len(rej):
            L += ["### Passed indicators but REJECTED by the risk filter", ""]
            L += [f"- **{r['symbol']}**: {_clean(r['reject_reasons'])}" for _, r in rej.iterrows()] + [""]

    L += ["## 5. Limitations", "",
          "- Free data (Yahoo, manually supplied NSE files) can be late, wrong or incomplete; every phase reports what it could not verify.",
          "- Indicator filters do not predict returns. The global score and sector tilts are heuristics.",
          "- Circuits, order-book depth and surveillance status are only assessed when you supply the data; otherwise they are marked not assessed.",
          "- EQ series ≠ guaranteed exit. Position sizing and stops are your responsibility.", ""]
    (reports / "premarket_report.md").write_text("\n".join(L), encoding="utf-8")
    update_status(reports, PHASE, "OK", "premarket_report.md written")
    print("[final_report] wrote premarket_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
