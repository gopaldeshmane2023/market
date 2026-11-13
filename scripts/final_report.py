#!/usr/bin/env python3
"""Assemble the files you actually read:

  reports/premarket_report.md          plain-English daily report (also shown on the GitHub run Summary page)
  reports/TradeSentinel_results.xlsx   one Excel file: Top_Candidates, Why_Filtered, All_Screened, Rejected

Never invents data. No entry prices, targets or stop losses are generated anywhere in TradeSentinel.
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from common import add_dir_args, dirs_from_args, fmt, gh_warning, now_utc, update_status

PHASE = "final_report"
ICON = {"OK": "✅", "DEGRADED": "⚠️", "SKIPPED": "⏭️", "FAILED": "❌"}
TOP_COLS = ["rank", "symbol", "tier", "rank_score", "risk_status", "liquidity_status", "series", "last_close", "daily_pct",
            "ema20", "ema50", "ema200", "rsi14", "macd_status", "rel_volume", "delivery_pct", "avg_traded_value_cr20",
            "dist_high20_pct", "criteria_missing", "reasons_pass", "reasons_fail", "warning_reasons", "notes"]


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except json.JSONDecodeError:
        return None


def _c(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) or str(v) == "nan" else str(v)


def block(r: pd.Series) -> list[str]:
    tier = "✅ STRICT PASS — all bullish rules met" if r["tier"] == "STRICT_PASS" else \
        f"🟡 NEAR MISS — missing: {_c(r.get('criteria_missing'))}"
    L = [f"### #{int(r['rank'])} {r['symbol']} — {tier}", "",
         f"- Score **{fmt(r['rank_score'], 1)}/100** · risk **{r['risk_status']}** · liquidity {_c(r.get('liquidity_status')) or 'UNKNOWN'} · series {_c(r.get('series')) or 'UNKNOWN'}",
         f"- Close ₹{fmt(r.get('last_close'))} ({_c(r.get('data_date'))}) · day {fmt(r.get('daily_pct'))}% · "
         f"EMA20/50/200 {fmt(r.get('ema20'))} / {fmt(r.get('ema50'))} / {fmt(r.get('ema200'))}",
         f"- RSI {fmt(r.get('rsi14'), 1)} · MACD {_c(r.get('macd_status'))} · relative volume {fmt(r.get('rel_volume'))}x · "
         f"delivery {fmt(r.get('delivery_pct'), 1)}% · avg traded value ₹{fmt(r.get('avg_traded_value_cr20'))} cr · "
         f"{fmt(r.get('dist_high20_pct'))}% from 20-day high", "", "Why it ranks here:"]
    L += [f"- {x}" for x in _c(r.get("reasons_pass")).split(" | ") if x] or ["- (no rule fully met)"]
    caut = [x for x in _c(r.get("reasons_fail")).split(" | ") if x] + [x for x in _c(r.get("warning_reasons")).split(" | ") if x]
    L += ["", "Cautions:"] + [f"- {x}" for x in caut]
    L += ["- Not a buy instruction: confirm the chart, news and order book yourself before acting.", ""]
    return L


def write_excel(reports, top, tech, risk, funnel) -> bool:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        gh_warning("openpyxl not installed: Excel workbook skipped (CSV/MD reports are still produced).")
        return False
    with pd.ExcelWriter(reports / "TradeSentinel_results.xlsx", engine="openpyxl") as xw:
        (top[[c for c in TOP_COLS if c in top.columns]] if len(top) else pd.DataFrame({"info": ["No candidates survived the hard safety rules today."]})
         ).to_excel(xw, sheet_name="Top_Candidates", index=False)
        fr = [("Stocks screened", funnel["screened"]), ("No price data", funnel["no_price_data"]),
              ("Stale data", funnel["stale_data"]), ("Not enough history for indicators", funnel["insufficient_history"]),
              ("Failed: EMA20>EMA50>EMA200 trend", funnel["failed_trend_ema20_50_200"]),
              ("Failed: RSI 50-70", funnel["failed_rsi_50_70"]), ("Failed: MACD above signal", funnel["failed_macd"]),
              ("Failed: relative volume >= 1.5x", funnel["failed_relative_volume"]),
              ("Failed: positive momentum", funnel["failed_momentum"]),
              ("Passed ALL indicator rules", funnel["passed_all_indicators"]),
              ("Rejected by risk filter (total)", funnel["risk_rejected"])] + \
             [(f"  Risk reject: {k}", v) for k, v in funnel["risk_rejected_by_reason"].items()] + \
             [("Strict shortlist (all rules + not rejected)", funnel["strict_final_shortlist"])]
        pd.DataFrame(fr, columns=["Step", "Number of stocks"]).to_excel(xw, sheet_name="Why_Filtered", index=False)
        tech.merge(risk[["symbol", "risk_status", "reject_reasons", "warning_reasons"]], on="symbol", how="left") \
            .sort_values("rank_score", ascending=False).to_excel(xw, sheet_name="All_Screened", index=False)
        risk[risk["risk_status"] == "REJECT"][["symbol", "series", "reject_reasons"]].to_excel(xw, sheet_name="Rejected", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in col[:50]) + 2))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    add_dir_args(ap)
    args = ap.parse_args(argv)
    _, _, reports = dirs_from_args(args)

    g = _json(reports / "global_report.json")
    status = _json(reports / "run_status.json") or {}
    nse = _json(reports / "nse_data_report.json")
    funnel = _json(reports / "funnel.json")
    have = all((reports / f).exists() for f in ("technical_screen_full.csv", "risk_report.csv", "top_candidates.csv"))
    tech = risk = top = None
    if have:
        tech, risk, top = (pd.read_csv(reports / f, keep_default_na=False, na_values=[""]) for f in
                           ("technical_screen_full.csv", "risk_report.csv", "top_candidates.csv"))

    L = ["# TradeSentinel — Pre-market Report", "", f"Generated: {now_utc().isoformat(timespec='seconds')} UTC · **WAIT. CONFIRM. PROTECT.**", "",
         "> Research output only — no orders, and no entry prices, targets or stop losses are generated. Nothing here is investment advice.", ""]

    L += ["## How to read this report (30 seconds)", "",
          "1. **Section 1** = today's best ranked stocks. Open `TradeSentinel_results.xlsx` → sheet **Top_Candidates** for the same list in Excel.",
          "2. **STRICT PASS** = met every bullish rule. **NEAR MISS** = best remaining stock that passed the hard safety rules but is missing the listed rule(s); "
          "it is shown so the list is never empty, and it needs *your* confirmation before any action.",
          "3. Hard safety rules are never relaxed to fill the list: non-EQ series, illiquid stocks, missing data, lower circuit are always excluded.",
          "4. Sheet **Why_Filtered** (and Section 2 below) shows exactly how many stocks fell out at each step.", ""]

    L += ["## 1. Top candidates today", ""]
    if top is None:
        L += ["_Screening did not run — see run status in Section 4._", ""]
    elif top.empty:
        L += ["**No stock survived the hard safety rules today** (see Section 2 for why). Nothing is forced into the list.", ""]
    else:
        strict = int((top["tier"] == "STRICT_PASS").sum())
        L += [f"**{len(top)} candidates** ({strict} strict pass, {len(top) - strict} near miss), ranked by score. "
              "The score is for sorting only — it is not a probability of profit.", "",
              "| # | Symbol | Tier | Score | Risk | Close ₹ | Day % | RSI | Rel.vol | Missing rule(s) |", "|--:|---|---|--:|---|--:|--:|--:|--:|---|"]
        for _, r in top.iterrows():
            L.append(f"| {int(r['rank'])} | **{r['symbol']}** | {r['tier']} | {fmt(r['rank_score'], 1)} | {r['risk_status']} | "
                     f"{fmt(r['last_close'])} | {fmt(r['daily_pct'])} | {fmt(r['rsi14'], 1)} | {fmt(r['rel_volume'])} | {_c(r['criteria_missing'])} |")
        L.append("")
        for _, r in top.iterrows():
            L += block(r)
        if len(top) < 5:
            L += [f"⚠️ Only {len(top)} stock(s) passed the hard safety rules, so fewer than 5 are shown. Reasons are in Section 2."
                  "A short list is correct output when few stocks are both liquid and eligible.", ""]

    L += ["## 2. Why stocks were filtered out", ""]
    if funnel:
        L += ["| Step | Stocks |", "|---|--:|", f"| Screened | {funnel['screened']} |",
              f"| No price data / stale / too little history | {funnel['no_price_data']} / {funnel['stale_data']} / {funnel['insufficient_history']} |",
              f"| Failed trend (EMA20>50>200) | {funnel['failed_trend_ema20_50_200']} |",
              f"| Failed RSI 50–70 | {funnel['failed_rsi_50_70']} |", f"| Failed MACD | {funnel['failed_macd']} |",
              f"| Failed relative volume ≥1.5x | {funnel['failed_relative_volume']} |",
              f"| Failed momentum | {funnel['failed_momentum']} |",
              f"| Passed **all** indicator rules | {funnel['passed_all_indicators']} |",
              f"| Rejected by risk filter | {funnel['risk_rejected']} |"]
        L += [f"| &nbsp;&nbsp;↳ {k} | {v} |" for k, v in funnel["risk_rejected_by_reason"].items()]
        L += [f"| **Strict shortlist** | **{funnel['strict_final_shortlist']}** |", ""]
        if funnel["risk_rejected_by_reason"].get("series not EQ / unverified", 0) >= max(1, funnel["screened"] // 2):
            L += ["⚠️ Most stocks were rejected because their **series is unknown**. Put the latest NSE `sec_bhavdata_full_*.csv` in `data/` "
                  "(or enable `NSE_AUTO_DOWNLOAD`) so series can be verified.", ""]
        if funnel["no_price_data"] >= max(1, funnel["screened"] // 2):
            L += ["⚠️ Most stocks returned **no price history** — likely a Yahoo Finance/network problem on this run. Re-run the workflow.", ""]
    else:
        L += ["_Not available._", ""]

    L += ["## 3. Global market & sectors", ""]
    if not g:
        L += ["_Global report not available._", ""]
    else:
        sc = g["score"]
        L += [f"**Global score: {'n/a' if sc['score'] is None else format(sc['score'], '+.1f')} / 100 — {sc['assessment']}** "
              f"(data coverage {sc['coverage_pct']}%). Context only, not a buy signal.", "",
              "| Factor | Move | Status | Contribution (pts) |", "|---|--:|---|--:|"]
        for f in g["factors"]:
            mv = "n/a" if f["move"] is None else (f"{f['change_bp']:+.1f} bp" if f["kind"] == "yield_bp" else f"{f['pct_change']:+.2f}%")
            L.append(f"| {f['name']} | {mv} | {f['status']} | {'excluded' if not f['included_in_score'] else format(f['contribution_points'], '+.1f')} |")
        L += ["", "**Macro risks:** " + ("; ".join(g["macro_risks"]) if g["macro_risks"] else "none of the configured thresholds breached."), "",
              "Sector tilts are a heuristic macro-sensitivity guess, **not** observed sector performance:", "",
              "| Sector | Tilt | Outlook |", "|---|--:|---|"]
        L += [f"| {s['sector']} | {'n/a' if s['tilt'] is None else format(s['tilt'], '+.0f')} | {s['label']}{' (ambiguous)' if s['ambiguous'] else ''} |"
              for s in g["sectors"]] + [""]

    L += ["## 4. Run status & data provenance", "", "| Phase | Status | Detail |", "|---|---|---|"]
    for ph in ("global_report", "nse_data", "technical_screen", "risk_filter"):
        s = status.get(ph)
        L.append(f"| {ph} | {ICON.get(s['status'], '?')} {s['status']} | {s['message']} |" if s else f"| {ph} | ⏭️ NOT RUN | |")
    if nse:
        L += ["", f"NSE source: `{nse['file']}` dated **{nse['source_date']}** ({nse['source_age_days']} day(s) old); obtained {nse['download_ts']}."]
    L += ["", "## 5. Limitations", "",
          "- Free data (Yahoo, manually supplied NSE files) can be late, wrong or incomplete; each phase reports what it could not verify.",
          "- Indicator filters and the rank score do not predict returns. Near-miss stocks are weaker setups than strict passes.",
          "- Circuits, order-book depth and ASM/GSM are only assessed when you supply the data. EQ series ≠ guaranteed exit.",
          "- Position size and exit plan are your responsibility.", ""]
    (reports / "premarket_report.md").write_text("\n".join(L), encoding="utf-8")

    try:
        xl = write_excel(reports, top, tech, risk, funnel) if have and funnel else False
    except Exception as exc:  # noqa: BLE001 — the Excel file is a convenience; never fail the run over it
        gh_warning(f"Excel workbook could not be written: {type(exc).__name__}: {exc}")
        (reports / "TradeSentinel_results.xlsx").unlink(missing_ok=True)
        xl = False
    update_status(reports, PHASE, "OK", "premarket_report.md" + (" + TradeSentinel_results.xlsx" if xl else "") + " written")
    print("[final_report] wrote premarket_report.md" + (" and TradeSentinel_results.xlsx" if xl else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
