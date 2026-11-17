#!/usr/bin/env python3
"""Weekly Swing Strategy scanner (independent of the daily screen).

Price action first: candle patterns, support/resistance, breakout CLOSE confirmation, volume/delivery, trend, then
sector and global context. Hard risk gates override any score. Only COMPLETED daily candles are classified; a candle
that is still forming can at most be reported as BREAKOUT_PENDING_CLOSE.

Outputs: reports/swing_report.md, swing_candidates.csv, swing_rejected.csv, swing_report.json
Reuses: technical_screen (indicators, history loader, watchlist), risk_filter.evaluate_risk (series/liquidity/circuit/
surveillance gates), global_report.json (macro context), nse_eod_processed.csv (series, delivery, bands).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import swing_engine as se
import swing_levels as lv
import swing_patterns as sp
import technical_screen as ts
from common import (IST, add_dir_args, dirs_from_args, fmt, gh_error, gh_warning, load_yaml, update_status, write_json)
from risk_filter import evaluate_risk

PHASE = "swing_scan"


# --------------------------------------------------------------------------- data completeness
def split_complete(df: pd.DataFrame, now_ist: datetime, cfg: dict) -> tuple[pd.DataFrame, pd.Series | None, str]:
    """Return (complete_bars, forming_bar_or_None, note). A bar dated today is complete only after close + buffer.
    Yahoo has no explicit 'complete' flag, so completeness is time-based and cross-checked against NSE's EOD file."""
    if df is None or df.empty:
        return df, None, "no data"
    h, mnt = [int(x) for x in cfg["session"]["close_time_ist"].split(":")]
    done_at = datetime.combine(now_ist.date(), dtime(h, mnt), tzinfo=now_ist.tzinfo) + timedelta(minutes=int(cfg["session"]["completion_buffer_min"]))
    last_d = df.index[-1].date()
    if last_d > now_ist.date():
        return df.iloc[:-1], None, f"dropped a bar dated in the future ({last_d})"
    if last_d == now_ist.date() and now_ist < done_at and now_ist.weekday() < 5:
        return df.iloc[:-1], df.iloc[-1], f"today's bar is still forming (session ends {cfg['session']['close_time_ist']} IST +{cfg['session']['completion_buffer_min']}min): excluded"
    return df, None, "last bar complete"


# --------------------------------------------------------------------------- context
def load_sector_map(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "company", "industry", "sector", "macro_sector", "sector_index"])
    d = pd.read_csv(path, dtype=str, comment="#", skipinitialspace=True).fillna("")
    d.columns = [c.strip().lower() for c in d.columns]
    d["symbol"] = d["symbol"].str.strip().str.upper()
    return d.drop_duplicates("symbol")


def index_state(df: pd.DataFrame | None) -> dict:
    """Direction of an index/benchmark from completed daily bars."""
    d = ts.clean_ohlcv(df)
    if d is None or len(d) < 55:
        return {"status": "UNKNOWN", "ret20": None, "ret5": None}
    c = d["Close"]
    e20, e50 = ts.ema(c, 20).iloc[-1], ts.ema(c, 50).iloc[-1]
    st = "UP" if (c.iloc[-1] > e20 > e50) else "DOWN" if (c.iloc[-1] < e20 and e20 < e50) else "SIDEWAYS"
    return {"status": st, "ret20": float(c.iloc[-1] / c.iloc[-21] - 1) * 100, "ret5": float(c.iloc[-1] / c.iloc[-6] - 1) * 100,
            "close": float(c.iloc[-1]), "date": d.index[-1].date().isoformat()}


def global_alignment(gjson: dict | None, macro_sector: str) -> dict:
    if not gjson or gjson.get("score", {}).get("score") is None:
        return {"available": False, "note": "global score unavailable"}
    sc = gjson["score"]["score"]
    s = (sc + 100) / 200.0
    tilt = None
    for sec in gjson.get("sectors", []):
        if macro_sector and sec["sector"].lower() == macro_sector.lower() and sec["tilt"] is not None:
            tilt = sec["tilt"]
            label = sec["label"]
    if tilt is not None:
        s = 0.5 * s + 0.5 * (tilt + 100) / 200.0
    return {"available": True, "score": sc, "assessment": gjson["score"]["assessment"], "sector_tilt": tilt,
            "value": float(np.clip(s, 0, 1)), "risks": gjson.get("macro_risks", []),
            "note": f"global score {sc:+.0f} ({gjson['score']['assessment']})" + (f", sector macro tilt {tilt:+.0f}" if tilt is not None else "")}


# --------------------------------------------------------------------------- liquidity
def liquidity_metrics(d: pd.DataFrame, nse: dict | None, cfg: dict) -> dict:
    c, v = d["Close"], d["Volume"]
    tv = c * v
    adv20 = tv.shift(1).rolling(20, min_periods=20).mean().iloc[-1] / 1e7
    adv5 = tv.shift(1).rolling(5, min_periods=5).mean().iloc[-1] / 1e7
    cur = tv.iloc[-1] / 1e7
    capital = float(cfg["capital"])
    cap_pct = float(cfg["liquidity"]["max_participation_pct"])
    out = {"adv20_cr": float(adv20), "adv5_cr": float(adv5), "current_tv_cr": float(cur),
           "avg_vol20": float(v.shift(1).rolling(20, min_periods=20).mean().iloc[-1])}
    if np.isfinite(adv20) and adv20 > 0:
        out["participation_8L_pct"] = capital / (adv20 * 1e7) * 100.0
        out["max_order_value"] = adv20 * 1e7 * cap_pct / 100.0
    else:
        out["participation_8L_pct"], out["max_order_value"] = None, None
    return out


def liquidity_score(liq: dict, plan: dict, cfg: dict) -> float:
    min_adv = float(cfg["risk"]["min_avg_traded_value_cr"])
    adv_s = float(np.clip((liq["adv20_cr"] if np.isfinite(liq["adv20_cr"]) else 0) / (5 * min_adv), 0, 1))
    cap = float(cfg["liquidity"]["max_participation_pct"])
    p = liq.get("participation_8L_pct")
    part_s = 0.0 if p is None else float(np.clip(cap / p, 0, 1)) if p > 0 else 1.0
    return 0.6 * adv_s + 0.4 * part_s


# --------------------------------------------------------------------------- scoring
def score_candidate(ev: dict, plan: dict, liq: dict, delivery: float | None, sector: dict, glob: dict, weekly_pat: float, cfg: dict) -> dict:
    W = cfg["scoring"]["weights"]
    s = ev["primary"]
    comps: dict[str, dict] = {}
    pats = ev["patterns"]
    best = max([p["strength"] for p in pats if p["confirmed"]] + [0.5 * p["strength"] for p in pats if not p["confirmed"]] + [0.0])
    if s and s["code"] in ("A", "B", "D", "E", "F") and s["confirmed"]:
        best = max(best, 0.9 if s.get("quality") in ("STRONG", "STRONG_DISPLACEMENT") or s["code"] in ("B", "D") else 0.6)
    med = sum(1 for w in ev["warnings"] if w["severity"] == "MEDIUM")
    pat_s = float(np.clip(0.8 * best + 0.2 * weekly_pat - min(0.4, 0.15 * med), 0, 1))
    comps["patterns"] = {"value": pat_s, "detail": f"best bullish pattern {best:.2f}, weekly {weekly_pat:.2f}, medium warnings {med}"}

    if s:
        q = 0.4 if not s["confirmed"] else {"STRONG": 1.0, "STRONG_DISPLACEMENT": 1.0, "WEAK": 0.6, "EARLY_REVERSAL": 0.5}.get(s.get("quality"), 0.8)
        zone = ev["breakout"].get("zone") if s["code"] == "A" else None
        loc = 1.0 if (zone and zone["strength"] >= 2) else 0.7 if (zone or any(p["at_support"] for p in pats)) else 0.4
        rr = plan.get("rr")
        room = 0.5 if rr is None else float(np.clip(rr / (2 * float(cfg["trade_plan"]["min_rr"])), 0, 1))
        st = 0.45 * q + 0.30 * loc + 0.25 * room
        comps["structure"] = {"value": float(st), "detail": f"setup quality {q:.2f}, location {loc:.2f}, room to next resistance {room:.2f}" + (" (no resistance identified)" if rr is None else "")}
    else:
        comps["structure"] = {"value": 0.0, "detail": "no setup"}

    r = ev["rel_vol"]
    vol_s = float(np.clip((r if np.isfinite(r) else 0) / (2 * float(cfg["breakout"]["min_rel_vol"])), 0, 1))
    if not ev["bull_bar"]:
        vol_s *= 0.3
    if delivery is not None and np.isfinite(delivery):
        vd = 0.6 * vol_s + 0.4 * float(np.clip(delivery / float(cfg["liquidity"]["good_delivery_pct"]), 0, 1))
        dn = f"relative volume {r:.2f}x, delivery {delivery:.1f}%"
    else:
        vd, dn = vol_s, f"relative volume {r:.2f}x; delivery % unavailable (volume only)"
    comps["volume_delivery"] = {"value": vd, "detail": dn}

    ls = liquidity_score(liq, plan, cfg)
    comps["liquidity"] = {"value": ls, "detail": f"20d avg traded value Rs{liq['adv20_cr']:.1f} cr, 8L order = {fmt(liq.get('participation_8L_pct'), 2)}% of it; order-book depth NOT verified"}

    t = ev["trend"]
    rsi = t["rsi"]
    rsi_s = 1.0 if 50 <= rsi <= 70 else max(0.0, 1 - min(abs(rsi - 50), abs(rsi - 70)) / 20) if np.isfinite(rsi) else 0.0
    tm = 0.3 * t["weekly_ok"] + 0.2 * t["hh_hl"] + 0.2 * (t["close_above_ema20"] and t["close_above_ema50"] and t["ema20_above_ema50"]) + 0.15 * rsi_s + 0.15 * t["macd_above"]
    comps["trend_momentum"] = {"value": float(tm), "detail": f"weekly {t['weekly_status']}, HH/HL {t['hh_hl']}, EMA stack {t['ema20_above_ema50']}, RSI {fmt(rsi, 1)}, MACD>signal {t['macd_above']}"}

    if sector.get("available"):
        sv = {"UP": 0.5, "SIDEWAYS": 0.25, "DOWN": 0.0}.get(sector["status"], 0.0) + (0.5 if (sector.get("rs") or 0) > 0 else 0.15 if (sector.get("rs") or 0) > -2 else 0.0)
        comps["sector"] = {"value": sv, "detail": f"sector {sector['status']}, stock vs sector {fmt(sector.get('rs'), 1)} pts"}
    else:
        comps["sector"] = {"value": None, "detail": "UNKNOWN: " + sector.get("note", "no sector data")}
    comps["global"] = {"value": glob.get("value") if glob.get("available") else None, "detail": glob.get("note", "unavailable")}

    avail = {k: v for k, v in comps.items() if v["value"] is not None}
    tw = sum(W[k] for k in avail)
    total = sum(W[k] * v["value"] for k, v in avail.items()) / tw * 100.0 if tw else 0.0
    for k, v in comps.items():
        v["weight"] = W[k]
        v["points"] = None if v["value"] is None else W[k] * v["value"] / tw * 100.0 if tw else 0.0
        v["max_points"] = None if v["value"] is None or not tw else W[k] / tw * 100.0
    return {"total": float(total), "components": comps, "coverage_weight": float(tw), "unverified": [k for k, v in comps.items() if v["value"] is None]}


def grade(total: float, primary: dict | None, plan: dict, cfg: dict, warnings: list, delivery: float | None) -> str:
    g = cfg["scoring"]
    if total < g["grade_c"]:
        return "Rejected"
    confirmed = bool(primary and primary["confirmed"])
    early = bool(primary and primary.get("quality") == "EARLY_REVERSAL")
    medium = any(w["severity"] == "MEDIUM" for w in warnings)
    if confirmed and not early:
        if total >= g["grade_a_plus"] and plan.get("rr") is not None and not medium and (delivery is None or delivery >= float(cfg["risk"]["min_delivery_pct"])) \
                and primary.get("quality") not in ("WEAK",):
            return "A+"
        if total >= g["grade_a"]:
            return "A"
    if total >= g["grade_b"]:
        return "B"
    return "C"


# --------------------------------------------------------------------------- per-symbol analysis
def analyse_symbol(symbol: str, df_raw, nse: dict | None, meta: dict, cfg: dict, risk_cfg: dict, ctx: dict, now_ist: datetime, src: str = "") -> dict:
    tcfg = {"circuit_lookback_days": 10}
    row = {"symbol": symbol, "status": "REJECTED", "reasons": [], "history_source": src}
    d = ts.clean_ohlcv(df_raw)
    if d is None:
        row["reasons"].append("No usable price history (download failed or symbol not on Yahoo)")
        return row
    d, forming, note = split_complete(d, now_ist, cfg)
    row["bar_note"] = note
    if len(d) < int(cfg["data"]["min_bars"]):
        row["reasons"].append(f"Insufficient history: {len(d)} complete bars < {cfg['data']['min_bars']}")
        return row
    last_date = d.index[-1].date()
    age = (now_ist.date() - last_date).days
    stale = age > int(cfg["data"]["max_stale_days"])
    prep = se.prepare(d, cfg)
    i = len(d) - 1
    ev = se.evaluate_bar(prep, i)
    m = prep.m
    r = m.iloc[i]
    ev["rel_vol"], ev["bull_bar"] = float(r["rel_vol"]) if np.isfinite(r["rel_vol"]) else float("nan"), bool(r["bull"])

    # pending (forming) bar: can never be a confirmed breakout
    pending = None
    if forming is not None:
        z = [zz for zz in prep.zones(i) if zz["high"] >= d["Close"].iloc[-1] * 0.999]
        if z:
            zz = min(z, key=lambda q: q["high"])
            trig = zz["high"] + lv._buffer(prep, i, zz["high"])
            if float(forming["High"]) > trig:
                pending = {"level": zz["high"], "trigger": trig, "partial_high": float(forming["High"])}
                ev["breakout_status"] = "BREAKOUT_PENDING_CLOSE"

    nse = nse or {}
    delivery = None
    nse_date = str(nse.get("source_date", ""))
    if nse and nse_date == last_date.isoformat() and pd.notna(nse.get("delivery_pct")):
        delivery = float(nse["delivery_pct"])
    band = float(nse["price_band_pct"]) if nse.get("price_band_pct") is not None and pd.notna(nse.get("price_band_pct")) else float("nan")
    up, dn = ts.count_circuit_days(d, band, 10, 0.15)
    liq = liquidity_metrics(d, nse, cfg)
    close = float(d["Close"].iloc[-1])
    series = nse.get("series") if nse.get("series") is not None and not pd.isna(nse.get("series")) else None
    rrow = {"series": series, "master_series": nse.get("master_series"), "data_status": "STALE" if stale else "OK", "data_date": last_date.isoformat(),
            "data_age_days": age, "last_close": close, "avg_traded_value_cr20": liq["adv20_cr"], "avg_vol20": liq["avg_vol20"],
            "delivery_pct": delivery if delivery is not None else np.nan, "circuit_status": nse.get("circuit_status", "UNKNOWN"),
            "price_band_pct": band, "circuit_up_days": up, "circuit_down_days": dn,
            "surveillance": nse.get("surveillance", "NO_DATA"), "security_status": nse.get("security_status", "UNVERIFIED"),
            "price_mismatch": False, "nse_data_quality": nse.get("data_quality", "")}
    if np.isfinite(float(nse.get("close", np.nan) if nse.get("close") is not None else np.nan)) and nse_date == last_date.isoformat():
        diff = abs(close / float(nse["close"]) - 1) * 100
        rrow["price_mismatch"] = bool(diff > 2.0)
    rk = evaluate_risk(rrow, risk_cfg)

    plan = se.trade_plan(ev, cfg, float(cfg["capital"]), liq["adv20_cr"])
    meta_sector = ctx["sector_for"](symbol, d)
    glob = global_alignment(ctx["global"], meta.get("macro_sector", ""))
    wk = prep.wk
    weekly_pat = 0.0
    if len(wk) > 60:
        wp = sp.detect_patterns(wk[["Open", "High", "Low", "Close", "Volume"]], cfg["patterns"])
        bulls = [float(wp[f"geo_{n}"].iloc[-1]) * 0.8 for n in sp.BULL_REVERSAL + sp.BULL_CONTINUATION_SIMPLE if wp[f"pat_{n}"].iloc[-1]]
        weekly_pat = max(bulls) if bulls else 0.0
    sc = score_candidate(ev, plan, liq, delivery, meta_sector, glob, weekly_pat, cfg)

    reasons = list(rk["reject_reasons"])
    if ev["invalidated"]:
        reasons.append("Setup invalidated: " + "; ".join(f"{w['name']} ({w['detail'][:80]})" for w in ev["warnings"] if w["severity"] == "HIGH"))
    if not ev["primary"]:
        reasons.append("No actionable setup (no breakout/retest/reversal/pullback/consolidation/FVG on the last completed candle)")
    reasons += [f"Trade plan: {x}" for x in plan["issues"]]
    if plan.get("qty") is not None and plan["qty"] <= 0 and ev["primary"] and not plan["issues"]:
        reasons.append("Position size computes to 0 shares under the risk/participation limits")
    mo = liq.get("max_order_value")
    if mo is not None and mo < float(cfg["liquidity"]["min_order_value"]):
        reasons.append(f"Participation cap allows only Rs{mo:,.0f} per order (< Rs{cfg['liquidity']['min_order_value']:,})")
    g_ = grade(sc["total"], ev["primary"], plan, cfg, ev["warnings"], delivery)
    if g_ == "Rejected" and not reasons:
        reasons.append(f"Score {sc['total']:.0f} below the minimum for a C grade ({cfg['scoring']['grade_c']})")
    unusual = []
    if np.isfinite(ev["rel_vol"]) and ev["rel_vol"] >= float(cfg["liquidity"]["unusual_rel_vol"]):
        unusual.append(f"unusual volume {ev['rel_vol']:.1f}x")
    dp = d["Close"].pct_change().iloc[-1] * 100
    if abs(dp) >= float(cfg["liquidity"]["unusual_move_pct"]):
        unusual.append(f"unusual one-day move {dp:+.1f}%: chasing risk")
    ind_ = {k: float(r[k]) for k in ("rsi", "ema20", "atr")}
    if np.isfinite(ind_["rsi"]) and ind_["rsi"] > 70:
        unusual.append(f"RSI {ind_['rsi']:.0f} > 70: momentum is already stretched")
    if np.isfinite(ind_["atr"]) and ind_["atr"] > 0 and (close - ind_["ema20"]) / ind_["atr"] >= 2.5:
        unusual.append(f"close is {(close - ind_['ema20']) / ind_['atr']:.1f} ATR above EMA20: extended, chasing risk")

    row.update(status="REJECTED" if (reasons or g_ == "Rejected") else "CANDIDATE", grade="Rejected" if reasons else g_, reasons=reasons,
               ev=ev, plan=plan, liq=liq, score=sc, risk=rk, delivery=delivery, sector=meta_sector, glob=glob, meta=meta, close=close,
               data_date=last_date.isoformat(), unusual=unusual, pending=pending, weekly_pattern=weekly_pat,
               nse_series=series, daily_pct=float(dp), indicators={k: float(r[k]) if np.isfinite(r[k]) else float("nan") for k in ("ema20", "ema50", "ema200", "rsi", "macd", "macd_signal", "atr")})
    return row


# --------------------------------------------------------------------------- reporting
FLAT_COLUMNS = ["symbol", "company", "grade", "score", "setup", "setup_confirmed", "setup_class", "industry", "sector", "last_close", "data_date",
                "pattern", "pattern_strength", "pattern_confirmation", "support_zone", "resistance_zone", "breakout_status", "upper_wick_body",
                "rel_volume", "delivery_pct", "ema20", "ema50", "ema200", "rsi", "macd_vs_signal", "weekly_trend", "fvg_status", "sector_status",
                "sector_rel_strength_pts", "global_alignment", "liquidity_score", "adv20_cr", "participation_8L_pct", "max_order_value", "risk_status",
                "entry_ref", "entry_basis", "invalidation", "stop_pct", "qty", "exposure", "risk_amount", "next_resistance", "reward_risk",
                "unverified_factors", "warnings", "reasons"]


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    """Always return a frame with every column so an empty result still has a header row."""
    return pd.DataFrame(rows).reindex(columns=FLAT_COLUMNS)


def flat_row(r: dict) -> dict:
    ev, plan, liq, sc = r.get("ev"), r.get("plan"), r.get("liq"), r.get("score")
    base = {"symbol": r["symbol"], "company": (r.get("meta") or {}).get("company", ""), "grade": r.get("grade", "Rejected"),
            "reasons": " | ".join(r.get("reasons", []))}
    if ev is None:
        return base
    pr = ev["primary"]
    pat = ev["patterns"][0] if ev["patterns"] else None
    base.update({
        "score": round(sc["total"], 1), "setup": f"{pr['code']}: {pr['name']}" if pr else "", "setup_confirmed": pr["confirmed"] if pr else "",
        "setup_class": (pr or {}).get("quality") or "", "industry": (r["meta"] or {}).get("industry", ""), "sector": (r["meta"] or {}).get("sector", ""),
        "last_close": round(r["close"], 2), "data_date": r["data_date"], "pattern": pat["name"] if pat else "", "pattern_strength": round(pat["strength"], 2) if pat else "",
        "pattern_confirmation": pat["confirmation"] if pat else "", "support_zone": _z(ev["support_zone"]), "resistance_zone": _z(ev["resistance_zone"]),
        "breakout_status": ev["breakout_status"], "upper_wick_body": round(ev["patterns"][0]["upper_wick_body"], 2) if pat else round(float(ev.get("uw_body", np.nan)), 2) if "uw_body" in ev else "",
        "rel_volume": round(ev["rel_vol"], 2), "delivery_pct": r["delivery"] if r["delivery"] is not None else "",
        "ema20": round(r["indicators"]["ema20"], 2), "ema50": round(r["indicators"]["ema50"], 2), "ema200": round(r["indicators"]["ema200"], 2),
        "rsi": round(r["indicators"]["rsi"], 1), "macd_vs_signal": "above" if ev["trend"]["macd_above"] else "below", "weekly_trend": ev["trend"]["weekly_status"],
        "fvg_status": ev["fvg"]["status"], "sector_status": r["sector"].get("status", "UNKNOWN"), "sector_rel_strength_pts": r["sector"].get("rs", ""),
        "global_alignment": r["glob"].get("note", ""), "liquidity_score": round(sc["components"]["liquidity"]["value"], 2),
        "adv20_cr": round(liq["adv20_cr"], 1), "participation_8L_pct": round(liq["participation_8L_pct"], 2) if liq.get("participation_8L_pct") is not None else "",
        "max_order_value": round(liq["max_order_value"]) if liq.get("max_order_value") else "", "risk_status": r["risk"]["status"],
        "entry_ref": round(plan["entry_ref"], 2) if plan["entry_ref"] else "", "entry_basis": plan["entry_basis"] or "", "invalidation": round(plan["invalidation"], 2) if plan["invalidation"] else "",
        "stop_pct": round(plan["stop_pct"], 2) if plan["stop_pct"] else "", "qty": plan["qty"] if plan["qty"] is not None else "", "exposure": round(plan["exposure"]) if plan["exposure"] else "",
        "risk_amount": round(plan["risk_amount"]) if plan["risk_amount"] else "", "next_resistance": _z(plan["next_resistance"]), "reward_risk": round(plan["rr"], 2) if plan["rr"] is not None else "n/a",
        "unverified_factors": ",".join(sc["unverified"]), "warnings": " | ".join(w["name"] + ":" + w["severity"] for w in ev["warnings"])})
    return base


def _z(z):
    return "" if not z else f"{z[0]:.2f}-{z[1]:.2f}"


def block(r: dict, cfg: dict) -> list[str]:
    ev, plan, liq, sc, pr = r["ev"], r["plan"], r["liq"], r["score"], r["ev"]["primary"]
    meta = r["meta"]
    name = meta.get("company") or r["symbol"]
    kind = "✅ CONFIRMED" if pr["confirmed"] else "⏳ EARLY — waiting for confirmation"
    L = [f"### {r['symbol']} — {name} · grade **{r['grade']}** · score {sc['total']:.0f}/100", "",
         f"- **Setup:** {pr['code']} — {pr['name']} · {kind}" + (f" · class {pr.get('quality')}" if pr.get("quality") else ""),
         f"- **Industry / sector:** {meta.get('industry') or 'UNKNOWN'} / {meta.get('sector') or 'UNKNOWN'}",
         f"- **Last close:** ₹{r['close']:.2f} ({r['data_date']}) · day {r['daily_pct']:+.2f}%",
         f"- **Breakout status:** {ev['breakout_status']} · **Support zone:** {_z(ev['support_zone']) or 'none identified'} · **Resistance zone:** {_z(ev['resistance_zone']) or 'none identified (open air)'}"]
    pat = ev["patterns"][0] if ev["patterns"] else None
    if pat:
        L += [f"- **Candle pattern:** {pat['name'].replace('_', ' ')} ({pat['date']}) · strength {pat['strength']:.2f} · {pat['confirmation']}",
              f"  - *Formula:* {pat['explanation']}"]
    else:
        L.append("- **Candle pattern:** none that meets the thresholds (setup is structure-based)")
    ind = r["indicators"]
    L += [f"- **Upper wick/body:** {ev['patterns'][0]['upper_wick_body']:.2f}" if pat else "- **Upper wick/body:** n/a",
          f"- **Relative volume:** {ev['rel_vol']:.2f}x · **Delivery %:** {fmt(r['delivery'], 1)}" + ("" if r["delivery"] is not None else " (not available for this date)"),
          f"- **EMA20/50/200:** {ind['ema20']:.2f} / {ind['ema50']:.2f} / {ind['ema200']:.2f} · **RSI:** {ind['rsi']:.1f} · **MACD:** {'above' if ev['trend']['macd_above'] else 'below'} signal",
          f"- **Weekly trend:** {ev['trend']['weekly_status']} · **FVG:** {ev['fvg']['status']}" + (f" (bullish gap {ev['fvg']['bullish']['low']:.2f}-{ev['fvg']['bullish']['high']:.2f}, {ev['fvg']['bullish']['state']})" if ev['fvg']['bullish'] else ""),
          f"- **Sector:** {r['sector'].get('status', 'UNKNOWN')}" + (f", stock vs sector {r['sector']['rs']:+.1f} pts over {cfg['sector']['outperform_lookback']}d" if r['sector'].get('rs') is not None else "") + f" · **Global:** {r['glob'].get('note')}",
          f"- **Liquidity score:** {sc['components']['liquidity']['value']:.2f} · 20d avg traded value ₹{liq['adv20_cr']:.1f} cr · ₹8L order = {fmt(liq.get('participation_8L_pct'), 2)}% of it (cap {cfg['liquidity']['max_participation_pct']}%) · max order at cap ₹{fmt(liq.get('max_order_value'), 0)} · order-book depth **not verified**",
          f"- **Risk classification:** {r['risk']['status']}" + (f" — {'; '.join(r['risk']['warning_reasons'])}" if r['risk']['warning_reasons'] else "")]
    L += ["", "**Trade plan (formula-derived reference levels — not targets, not advice):**",
          f"- Entry reference ₹{plan['entry_ref']:.2f} ({plan['entry_basis']}) · technical invalidation ₹{plan['invalidation']:.2f} ({plan['stop_pct']:.1f}% below) — {pr['name'].split(':')[0]} structure low minus {cfg['trade_plan']['stop_buffer_atr']}×ATR",
          f"- Position by risk budget ({cfg['risk_per_trade_pct']}% of ₹{cfg['capital']:,.0f}): **{plan['qty']} shares**, exposure ≈ ₹{plan['exposure']:,.0f}, planned loss ≈ ₹{plan['risk_amount']:,.0f}",
          f"- Next resistance (reference only): {_z(plan['next_resistance']) or 'none identified — reward/risk not estimable'} · reward/risk to it: {fmt(plan['rr'], 2) if plan['rr'] is not None else 'n/a'}", ""]
    L += ["**Score breakdown:** " + " · ".join(f"{k} {v['points']:.1f}/{v['max_points']:.1f}" if v["points"] is not None else f"{k} unverified" for k, v in sc["components"].items()) +
          (f" _(score renormalised over {sc['coverage_weight']:.0f}% of weight; unverified: {', '.join(sc['unverified'])})_" if sc["unverified"] else ""), ""]
    why = [f"{k}: {v['detail']}" for k, v in sc["components"].items() if v["value"] is not None and v["value"] >= 0.6]
    L += ["**Why selected**"] + [f"- {x}" for x in why] + [""]
    fail = [f"{k}: {v['detail']}" for k, v in sc["components"].items() if v["value"] is not None and v["value"] < 0.5]
    fail += [f"warning — {w['name']} ({w['severity']}): {w['detail'][:110]}" for w in ev["warnings"]]
    fail += r["unusual"] + [f"risk filter: {x}" for x in r["risk"]["warning_reasons"]]
    if not pr["confirmed"]:
        fail.append("Setup is NOT confirmed: it needs a daily close above its trigger (₹%.2f) before it can be treated as a breakout/reversal." % pr["trigger"])
    fail += [f"{n}" for n in r["risk"]["notes"]][:2]
    fail.append(f"Invalidation: a close below ₹{plan['invalidation']:.2f} voids the setup. Overnight gaps can exceed the stop distance. Order-book depth is not verified.")
    L += ["**Why it could fail**"] + [f"- {x}" for x in fail] + [""]
    return L


def render_md(rep: dict, cands: list[dict], rej: list[dict], cfg: dict) -> str:
    L = ["# TradeSentinel — Weekly Swing Report", "", f"Generated {rep['generated_ist']} IST · mode **{rep['mode']}** · data as of last COMPLETE daily candle", "",
         "**WAIT. CONFIRM. PROTECT.** Research output, not advice, no orders placed. A stock near resistance is **not** a confirmed breakout until a daily candle CLOSES above resistance + buffer.", ""]
    L += ["## How to read this", "",
          "- **A+ / A** = confirmed setup, strongest/strong. **B** = early setup or confirmed-but-weaker: *waiting for confirmation* where marked. **C** = weak. **Rejected** = failed a hard gate (series, liquidity, data, invalidation, reward/risk, bearish distribution).",
          "- A high score never overrides a hard-gate rejection. Early reversals (EMA trend not yet up) are labelled EARLY_REVERSAL and capped at B.",
          "- Reference levels (invalidation, next resistance) come from formulas on the chart; they are not predictions or targets. Backtest evidence for each pattern is in `swing_backtest.md` — read it before trusting any grade.", ""]
    L += ["## Market context", ""]
    n = rep["context"]["nifty"]
    L += [f"- **Nifty 50 ({cfg['sector']['benchmark']}):** {n['status']}" + (f", 5d {n['ret5']:+.1f}%, 20d {n['ret20']:+.1f}%" if n.get("ret20") is not None else " (data unavailable)")]
    g = rep["context"]["global"]
    if g:
        L += [f"- **Global score:** {g['score']['score'] if g['score']['score'] is None else format(g['score']['score'], '+.0f')} ({g['score']['assessment']}) from the existing global report ({g['generated_at_ist']} IST)"]
        L += ["", "| Macro factor | Move | Status |", "|---|--:|---|"]
        for f in g["factors"]:
            if f["key"] in ("brent", "dxy", "usdinr", "us10y", "gold", "copper", "vix", "sp500", "nasdaq"):
                mv = "n/a" if f["move"] is None else (f"{f['change_bp']:+.1f} bp" if f["kind"] == "yield_bp" else f"{f['pct_change']:+.2f}%")
                L.append(f"| {f['name']} | {mv} | {f['status']} |")
        if g["macro_risks"]:
            L += ["", "Macro headwinds: " + "; ".join(g["macro_risks"])]
    else:
        L += ["- **Global report not available** (run `global_report.py`); the 5% global factor is unverified."]
    L += [""]
    if rep["notes"]:
        L += ["## Data notes", ""] + [f"- {x}" for x in rep["notes"]] + [""]
    counts = {k: sum(1 for c in cands if c["grade"] == k) for k in ("A+", "A", "B", "C")}
    L += [f"## Summary: A+ {counts['A+']} · A {counts['A']} · B {counts['B']} · C {counts['C']} · Rejected {len(rej)} (of {rep['screened']} screened)", ""]
    for gname, title in (("A+", "A+ — Confirmed setups"), ("A", "A — Strong setups"), ("B", "B — Early setups / waiting for confirmation")):
        sub = [c for c in cands if c["grade"] == gname]
        L += [f"## {title}", ""]
        if not sub:
            L += ["_None today._", ""]
        for c in sub:
            L += block(c, cfg)
    cs = [c for c in cands if c["grade"] == "C"]
    if cs:
        L += ["## C — Weak setups (listed, not recommended)", "", "| Symbol | Setup | Score | Why weak |", "|---|---|--:|---|"]
        L += [f"| {c['symbol']} | {c['ev']['primary']['name'] if c['ev']['primary'] else ''} | {c['score']['total']:.0f} | " + "; ".join(k for k, v in c['score']['components'].items() if v['value'] is not None and v['value'] < 0.5) + " |" for c in cs] + [""]
    L += ["## Rejections by reason", ""]
    cnt: dict[str, int] = {}
    for r in rej:
        key = (r["reasons"][0] if r.get("reasons") else "other").split(":")[0][:90]
        cnt[key] = cnt.get(key, 0) + 1
    L += [f"- {v} × {k}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])] + ["", "Full list with every reason: `swing_rejected.csv`.", ""]
    L += ["## Liquidity primer", "",
          "Turnover (₹) = price × shares traded. A ₹20 stock trading 10 lakh shares turns over only ₹2 crore, while a ₹2,000 stock with the same 10 lakh shares turns over ₹200 crore: **share volume alone says nothing about how much capital you can deploy.** "
          f"Participation = order value ÷ average daily traded value; the cap here is {cfg['liquidity']['max_participation_pct']}%. Turnover is not order-book depth: depth and bid/ask spreads are not available from free data, so **execution capacity is not verified** for any stock.", "",
          "## Limitations", "",
          "- Free, unofficial Yahoo data: gaps, delays, split/bonus artefacts. Only completed daily candles are classified; completeness is time-based (Yahoo has no completeness flag) and cross-checked against NSE's EOD file when supplied.",
          "- Support/resistance are zones, not exact prices. Patterns are formulas: a pattern is evidence, not a forecast. See the backtest for measured value.",
          "- Delivery % exists only for the NSE file's date; no history of delivery is used. Sector data is UNKNOWN unless `config/sector_map.csv` and sector index data are available.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main
def default_index_provider(ticker: str, period: str):
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
        return (df if df is not None and not df.empty else None), "yfinance"
    except Exception:  # noqa: BLE001
        return None, "yfinance-error"


def run(args, provider=None, index_provider=None, now_ist: datetime | None = None) -> int:
    config_dir, data_dir, reports = dirs_from_args(args)
    cfg = load_yaml(config_dir / "swing.yml")
    scr = load_yaml(config_dir / "screening.yml")
    risk_cfg = {**scr["risk"], **cfg["risk"]}
    now_ist = now_ist or datetime.now(IST)
    provider = provider or ts.default_history_provider(data_dir)
    index_provider = index_provider or default_index_provider
    notes: list[str] = []

    nse_path = reports / "nse_eod_processed.csv"
    nse = pd.read_csv(nse_path, dtype={"symbol": str}) if nse_path.exists() else None
    if nse is not None:
        eq = nse[nse["series"] == "EQ"]
        nse = pd.concat([eq, nse[~nse["symbol"].isin(eq["symbol"])]]).drop_duplicates("symbol")
        nse_by = {r["symbol"]: r for r in nse.to_dict("records")}
    else:
        nse_by = {}
        notes.append("No NSE EOD file processed: series, delivery %, circuits and surveillance are unverified, so every stock is rejected (series unknown).")
    gjson = None
    gp = reports / "global_report.json"
    if gp.exists():
        import json
        gjson = json.loads(gp.read_text(encoding="utf-8"))
    else:
        notes.append("global_report.json missing: global factor unverified.")
    smap = load_sector_map(config_dir / cfg["sector"]["map_file"])
    meta_by = {r["symbol"]: r for r in smap.to_dict("records")}

    entries, problems = ([], [])
    if args.universe in ("watchlist", "both"):
        entries, problems = ts.load_watchlist(config_dir / "watchlist.csv")
    universe = {e["symbol"]: {"company": "", **meta_by.get(e["symbol"], {})} for e in entries}
    if args.universe in ("nse", "both") and nse is not None:
        pool = nse[(nse["series"] == "EQ") & (nse["traded_value_cr"] >= float(cfg["risk"]["min_avg_traded_value_cr"]) / 4)]
        for s in pool.sort_values("traded_value_cr", ascending=False).head(args.max_symbols or int(cfg["data"]["max_symbols"]))["symbol"]:
            universe.setdefault(s, meta_by.get(s, {}))
    elif args.universe == "nse":
        gh_error("swing_scan: --universe nse needs NSE data (run nse_data.py).")
        update_status(reports, PHASE, "FAILED", "universe nse without NSE data")
        return 1
    if not universe:
        update_status(reports, PHASE, "SKIPPED", "empty universe")
        gh_warning("swing_scan: empty universe; nothing to scan.")
        return 0 if args.optional else 1
    for s, m in universe.items():
        if nse_by.get(s) is not None and not m.get("company") and pd.notna(nse_by[s].get("company_name")):
            m["company"] = nse_by[s]["company_name"]

    # benchmark + sector indices (unique tickers fetched once)
    idx_cache: dict[str, pd.DataFrame | None] = {}

    def idx(t: str):
        if t and t not in idx_cache:
            df, _ = index_provider(t, "2y")
            idx_cache[t] = ts.clean_ohlcv(df)
        return idx_cache.get(t)
    nifty = index_state(idx(cfg["sector"]["benchmark"]))
    if nifty["status"] == "UNKNOWN":
        notes.append(f"Benchmark {cfg['sector']['benchmark']} unavailable: Nifty direction unverified.")
    lb = int(cfg["sector"]["outperform_lookback"])

    def sector_for(symbol: str, d: pd.DataFrame) -> dict:
        t = (universe.get(symbol) or {}).get("sector_index", "")
        if not t:
            return {"available": False, "status": "UNKNOWN", "note": "no sector_index mapped for this symbol in config/sector_map.csv"}
        st = index_state(idx(t))
        if st["status"] == "UNKNOWN":
            return {"available": False, "status": "UNKNOWN", "note": f"sector index {t} data unavailable"}
        rs = float(d["Close"].iloc[-1] / d["Close"].iloc[-1 - lb] - 1) * 100 - st["ret20"] if len(d) > lb + 1 else None
        return {"available": True, "status": st["status"], "rs": rs, "sector_ret20": st["ret20"], "index": t}
    ctx = {"sector_for": sector_for, "global": gjson}

    results = []
    for k, (sym, meta) in enumerate(sorted(universe.items()), 1):
        try:
            df, src = provider(sym, cfg["data"]["history_period"])
        except Exception as exc:  # noqa: BLE001
            df, src = None, f"error: {type(exc).__name__}: {exc}"
        try:
            results.append(analyse_symbol(sym, df, nse_by.get(sym), meta, cfg, risk_cfg, ctx, now_ist, src))
        except Exception as exc:  # noqa: BLE001 — one bad symbol must not stop the scan
            results.append({"symbol": sym, "status": "REJECTED", "grade": "Rejected", "reasons": [f"Analysis error: {type(exc).__name__}: {exc}"]})
        if k % 25 == 0:
            print(f"[swing_scan] {k}/{len(universe)}")
    cands = [r for r in results if r.get("grade") in ("A+", "A", "B", "C") and not r["reasons"]]
    rej = [r for r in results if r not in cands]
    order = {"A+": 0, "A": 1, "B": 2, "C": 3}
    cands.sort(key=lambda r: (order[r["grade"]], -r["score"]["total"]))

    _to_frame([flat_row(r) for r in cands]).to_csv(reports / "swing_candidates.csv", index=False)
    _to_frame([flat_row(r) for r in rej]).to_csv(reports / "swing_rejected.csv", index=False)
    no_hist = sum(1 for r in results if any("No usable price history" in x for x in r.get("reasons", [])))
    if no_hist:
        notes.append(f"{no_hist} symbol(s) returned no price history.")
    notes += [f"Bar completeness: {r['bar_note']} ({r['symbol']})" for r in results if "forming" in r.get("bar_note", "")][:3]
    notes += problems
    rep = {"generated_ist": now_ist.strftime("%Y-%m-%d %H:%M"), "mode": args.mode, "screened": len(results), "notes": notes,
           "context": {"nifty": nifty, "global": gjson}, "counts": {g: sum(1 for c in cands if c["grade"] == g) for g in ("A+", "A", "B", "C")} | {"Rejected": len(rej)}}
    (reports / "swing_report.md").write_text(render_md(rep, cands, rej, cfg), encoding="utf-8")
    write_json(reports / "swing_report.json", {**rep, "candidates": [_slim(r) for r in cands], "rejected": [{"symbol": r["symbol"], "reasons": r.get("reasons", [])} for r in rej]})
    status = "OK" if not notes else "DEGRADED"
    update_status(reports, PHASE, status, f"A+ {rep['counts']['A+']}, A {rep['counts']['A']}, B {rep['counts']['B']}, C {rep['counts']['C']}, rejected {len(rej)}")
    for n_ in notes[:5]:
        gh_warning(f"swing_scan: {n_}")
    if no_hist == len(results):
        gh_error("swing_scan: no symbol returned price history.")
        return 1
    print(f"[swing_scan] {rep['counts']}")
    return 0


def _slim(r: dict) -> dict:
    f = flat_row(r)
    f["score_components"] = {k: {"value": v["value"], "points": v["points"], "max_points": v["max_points"], "weight": v["weight"], "detail": v["detail"]} for k, v in r["score"]["components"].items()}
    f["pattern_explanation"] = r["ev"]["patterns"][0]["explanation"] if r["ev"]["patterns"] else ""
    return f


def main(argv=None, provider=None, index_provider=None, now_ist=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dir_args(ap)
    ap.add_argument("--universe", choices=["watchlist", "nse", "both"], default="watchlist")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--mode", choices=["daily", "weekend"], default="daily")
    ap.add_argument("--optional", action="store_true")
    args = ap.parse_args(argv)
    try:
        return run(args, provider, index_provider, now_ist)
    except Exception as exc:  # noqa: BLE001
        gh_error(f"swing_scan failed: {type(exc).__name__}: {exc}")
        update_status(Path(args.reports_dir), PHASE, "FAILED", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
