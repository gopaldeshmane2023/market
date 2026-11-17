"""Swing engine: evaluates ONE completed daily bar using only data up to that bar (no look-ahead).

`prepare()` builds per-bar indicators/patterns once; `evaluate_bar(prep, i)` classifies the setups on bar i:
  A confirmed breakout (close above resistance+buffer)   B breakout retest      C support reversal pattern
  D bullish pullback in an uptrend                       E consolidation/flag/triangle/cup breakout
  F FVG continuation
Setups are kept separate. Unconfirmed variants (waiting for a close / next candle) are flagged confirmed=False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import swing_levels as lv
import swing_patterns as sp
from technical_screen import macd as _macd

HIGH_WARN = {"bearish_engulfing", "evening_star", "dark_cloud", "distribution_candle"}


class Prep:
    def __init__(self, df: pd.DataFrame, cfg: dict):
        self.cfg, self.df = cfg, df
        self.m = sp.detect_patterns(df, cfg["patterns"], cfg["breakout"]["buffer_pct"])
        mm = _macd(df["Close"])
        self.m["macd"], self.m["macd_signal"] = mm["macd"], mm["macd_signal"]
        lcfg = cfg["levels"]
        self.ph, self.pl = lv.pivot_arrays(df["High"].values, df["Low"].values, int(lcfg["pivot_left"]), int(lcfg["pivot_right"]))
        self.wk, self.wmap = lv.weekly_features(df)
        self.bo_memo: dict = {}
        self.zone_memo: dict = {}

    def zones(self, i: int) -> list[dict]:
        if i not in self.zone_memo:
            self.zone_memo[i] = lv.build_zones(self, i)
        return self.zone_memo[i]


def prepare(df: pd.DataFrame, cfg: dict) -> Prep:
    return Prep(df, cfg)


def uptrend_daily(m: pd.DataFrame, i: int) -> bool:
    r = m.iloc[i]
    return bool(r["Close_above_ema50"]) if "Close_above_ema50" in m else False


def trend_flags(prep: Prep, i: int) -> dict:
    m, df = prep.m, prep.df
    r, c = m.iloc[i], df["Close"].iloc[i]
    w = prep.wmap.iloc[i]
    ema_stack = bool(r["ema20"] > r["ema50"]) if np.isfinite(r["ema50"]) else False
    est = bool(ema_stack and c > r["ema50"] and (np.isfinite(r["ema200"]) and (r["ema50"] > r["ema200"] or m["ema50"].iloc[max(0, i - 10)] < r["ema50"])))
    return {"close_above_ema20": bool(c > r["ema20"]), "close_above_ema50": bool(c > r["ema50"]), "ema20_above_ema50": ema_stack,
            "established_uptrend": est, "weekly_status": str(w["w_status"]) if pd.notna(w["w_status"]) else "UNKNOWN",
            "weekly_ok": str(w["w_status"]) == "UP", "hh_hl": lv.hh_hl(prep, i),
            "rsi": float(r["rsi"]) if np.isfinite(r["rsi"]) else float("nan"),
            "macd_above": bool(r["macd"] > r["macd_signal"]) if np.isfinite(r["macd_signal"]) else False}


def _pattern_record(prep: Prep, name: str, j: int, i: int) -> dict | None:
    cfg, m, df = prep.cfg, prep.m, prep.df
    p = cfg["patterns"]
    if not m[f"pat_{name}"].iloc[j]:
        return None
    nb = sp.PATTERN_BARS.get(name, 1)
    lowest = float(df["Low"].iloc[max(0, j - nb + 1):j + 1].min())
    zones_j = prep.zones(j - 1)
    sup = lv.at_support(zones_j, lowest, cfg["levels"])
    rv = m["rel_vol"].iloc[j]
    vol_s = float(np.clip((rv if np.isfinite(rv) else 0) / 1.5, 0, 1))
    loc_s = float(min(1.0, 0.6 + 0.1 * sup["strength"])) if sup else 0.2
    reversal = name in sp.BULL_REVERSAL
    ctx_s = 1.0 if (m["prior_down"].iloc[j] if reversal else df["Close"].iloc[j] > m["ema20"].iloc[j]) else 0.5
    w = p["strength_weights"]
    shape = float(m[f"geo_{name}"].iloc[j])
    strength = w["shape"] * shape + w["volume"] * vol_s + w["location"] * loc_s + w["context"] * ctx_s
    self_conf = name in p["self_confirming"]
    confirmed_later = j < i and df["Close"].iloc[i] > df["High"].iloc[j]
    confirmed = bool(self_conf or confirmed_later)
    if confirmed_later or self_conf:
        strength += float(p["confirmation_bonus"])
    strength = float(min(strength, 1.0))
    expl = (f"{sp.describe_pattern(name, m, j)} | strength = {w['shape']}*shape {shape:.2f} + {w['volume']}*volume {vol_s:.2f} "
            f"+ {w['location']}*location {loc_s:.2f} + {w['context']}*context {ctx_s:.2f}"
            + (f" + {p['confirmation_bonus']} confirmation" if (confirmed_later or self_conf) else "") + f" = {strength:.2f}")
    return {"name": name, "bar_index": j, "date": df.index[j].date().isoformat(), "kind": "reversal" if reversal else "continuation",
            "strength": strength, "confirmed": confirmed,
            "confirmation": "built into the pattern" if self_conf else ("confirmed: later close above pattern high" if confirmed_later else "UNCONFIRMED: needs next candle to close above the pattern high"),
            "at_support": bool(sup), "support_zone": (sup["low"], sup["high"]) if sup else None,
            "upper_wick_body": float(m["uw_body"].iloc[j]), "lower_wick_body": float(m["lw_body"].iloc[j]),
            "body_range": float(m["body_ratio"].iloc[j]) if np.isfinite(m["body_ratio"].iloc[j]) else float("nan"),
            "rel_vol": float(rv) if np.isfinite(rv) else float("nan"), "lowest_low": lowest, "high": float(df["High"].iloc[j]),
            "explanation": expl}


def failed_breakdown_recovery(prep: Prep, i: int) -> dict | None:
    """Within 10 bars a close fell below a support zone, then price recovered back above the zone (bull trap for sellers)."""
    df = prep.df
    c = df["Close"].values
    zones = prep.zones(i - 10) if i > 40 else []
    for z in zones:
        if z["mid"] >= c[i]:
            continue
        window = c[i - 10:i]
        if (window < z["low"] - z["tol"]).any() and c[i] > z["high"]:
            return z
    return None


def evaluate_bar(prep: Prep, i: int) -> dict:
    cfg, m, df = prep.cfg, prep.m, prep.df
    r = m.iloc[i]
    close = float(df["Close"].iloc[i])
    atr = float(r["atr"]) if np.isfinite(r["atr"]) else float("nan")
    zones_prev = prep.zones(i - 1)          # PRIOR structure: today's own high/low must not become "resistance"/"support"
    zones_now = zones_prev
    support, resistance = lv.nearest_support(zones_prev, close), lv.nearest_resistance(zones_prev, close)
    tr = trend_flags(prep, i)
    ev: dict = {"bar_index": i, "date": df.index[i].date().isoformat(), "close": close, "atr": atr, "trend": tr,
                "support_zone": (support["low"], support["high"]) if support else None,
                "resistance_zone": (resistance["low"], resistance["high"]) if resistance else None,
                "resistance_zone_obj": resistance, "zones_now": zones_now}

    # ---- candle patterns (today and yesterday; yesterday's pattern can be confirmed by today's close)
    pats = []
    for j in (i, i - 1):
        for name in sp.BULL_REVERSAL + sp.BULL_CONTINUATION_SIMPLE:
            rec = _pattern_record(prep, name, j, i)
            if rec and rec["strength"] >= float(cfg["patterns"]["min_pattern_strength"]):
                pats.append(rec)
    pats.sort(key=lambda x: (-int(x["confirmed"]), -x["strength"]))
    ev["patterns"] = pats

    # ---- breakout / retest / failed
    bo = lv.classify_breakout(prep, i, zones_prev)
    prep.bo_memo[i] = bo
    rt = lv.retest_or_failed(prep, i)
    ev["breakout"], ev["retest"] = bo, rt
    if rt["status"] == "FAILED_BREAKOUT":
        ev["breakout_status"] = "FAILED_BREAKOUT"
    elif rt["status"] == "BREAKOUT_RETEST":
        ev["breakout_status"] = "BREAKOUT_RETEST"
    else:
        ev["breakout_status"] = bo["status"]

    # ---- bearish warnings (active: today, or yesterday if today did not negate it)
    warns = []
    res_obj = lv.nearest_resistance(zones_prev, float(df["Close"].iloc[i - 1])) if zones_prev else None
    for j in (i, i - 1):
        for name in sp.BEARISH:
            if not m[f"pat_{name}"].iloc[j]:
                continue
            if j < i and df["Close"].iloc[i] > df["High"].iloc[j]:
                continue
            near_res = bool(res_obj and df["High"].iloc[j] >= res_obj["low"] - res_obj["tol"])
            sev = "HIGH" if name in HIGH_WARN or (name in ("shooting_star", "long_upper_wick_rejection") and near_res) else "MEDIUM"
            warns.append({"name": name, "date": df.index[j].date().isoformat(), "severity": sev, "near_resistance": near_res,
                          "detail": sp.describe_pattern(name, m, j)})
    if rt["status"] == "FAILED_BREAKOUT":
        warns.append({"name": "failed_breakout", "date": ev["date"], "severity": "HIGH", "near_resistance": True,
                      "detail": f"closed back below the breakout level {rt['level']:.2f}"})
    if bo["status"] == "RESISTANCE_REJECTION":
        warns.append({"name": "resistance_rejection", "date": ev["date"], "severity": "HIGH", "near_resistance": True,
                      "detail": "; ".join(bo["quality_issues"]) or "rejected at resistance"})
    ev["warnings"] = warns

    # ---- FVG, chart patterns
    ev["fvg"] = lv.fvg_scan(prep, i)
    ev["consolidation"] = lv.consolidation_breakout(prep, i)
    ev["flag"] = lv.flag_breakout(prep, i)
    ev["triangle"] = lv.ascending_triangle_breakout(prep, i)
    ev["cup"] = lv.cup_handle_breakout(prep, i)
    ev["failed_breakdown"] = failed_breakdown_recovery(prep, i) is not None

    # ---- setups
    buf = float(cfg["trade_plan"]["stop_buffer_atr"]) * (atr if np.isfinite(atr) else 0.0)
    setups = []
    if bo["status"] == "CONFIRMED_BREAKOUT":
        setups.append({"code": "A", "name": "Confirmed breakout", "confirmed": True, "entry_basis": "close", "trigger": bo["trigger"],
                       "structure_low": min(float(df["Low"].iloc[i]), bo["level"]), "level": bo["level"], "quality": bo["quality"],
                       "notes": bo["quality_issues"]})
    elif bo["status"] == "POTENTIAL_BREAKOUT":
        setups.append({"code": "A", "name": "Potential breakout (needs a daily close above resistance)", "confirmed": False, "entry_basis": "trigger",
                       "trigger": bo["trigger"], "structure_low": float(df["Low"].iloc[i]), "level": bo["level"], "quality": None, "notes": bo["quality_issues"]})
    if rt["status"] == "BREAKOUT_RETEST":
        setups.append({"code": "B", "name": "Breakout retest", "confirmed": True, "entry_basis": "close", "trigger": rt["level"],
                       "structure_low": min(rt["retest_low"], rt["level"]), "level": rt["level"], "quality": rt["breakout_quality"], "notes": []})
    for rec in pats:
        if rec["kind"] != "reversal" or not (rec["at_support"] or ev["failed_breakdown"]) or rec["strength"] < float(cfg["patterns"]["min_pattern_strength"]):
            continue
        setups.append({"code": "C", "name": f"Support reversal: {rec['name'].replace('_', ' ')}", "confirmed": rec["confirmed"],
                       "entry_basis": "close" if rec["confirmed"] else "trigger", "trigger": rec["high"], "structure_low": rec["lowest_low"],
                       "level": rec["support_zone"][1] if rec["support_zone"] else None,
                       "quality": "EARLY_REVERSAL" if not tr["established_uptrend"] else "REVERSAL_IN_UPTREND", "notes": [rec["confirmation"]],
                       "pattern": rec["name"]})
        break
    pb = cfg["pullback"]
    lb = int(pb["lookback"])
    if tr["established_uptrend"] and i > 20:
        lo_win = float(df["Low"].iloc[i - lb:i + 1].min())
        touch = lo_win <= max(r["ema20"], r["ema50"]) * (1 + float(pb["ema_tol_pct"]) / 100.0) or bool(support and lo_win <= support["high"] * 1.01)
        recent_high = float(df["High"].iloc[i - 15:i].max())
        depth = (recent_high - lo_win) / recent_high * 100.0
        bull_rej = bool(r["bull"] and r["close_pos"] >= 0.6 and close > r["ema20"] and close > df["High"].iloc[i - 1] * 0.998 and
                        (r["lower_ratio"] >= 0.2 or any(p["name"] in ("hammer", "bullish_engulfing", "piercing") for p in pats)))
        if touch and float(pb["min_depth_pct"]) <= depth <= float(pb["max_depth_pct"]) and bull_rej and close >= r["ema50"]:
            setups.append({"code": "D", "name": "Bullish pullback to EMA/support in an uptrend", "confirmed": True, "entry_basis": "close", "trigger": float(df["High"].iloc[i]),
                           "structure_low": lo_win, "level": float(r["ema20"]), "quality": f"pullback depth {depth:.1f}%", "notes": []})
    for key, label in (("consolidation", "Consolidation breakout"), ("flag", None), ("triangle", "Ascending-triangle breakout"), ("cup", "Cup-and-handle breakout")):
        d = ev[key]
        if d:
            nm = label or f"Bull {d['kind']} breakout"
            low = d.get("range_low") or d.get("flag_low") or d.get("last_swing_low") or float(df["Low"].iloc[i])
            trig = d.get("trigger") or d.get("resistance") or close
            setups.append({"code": "E", "name": nm, "confirmed": True, "entry_basis": "close", "trigger": float(trig), "structure_low": float(min(low, df["Low"].iloc[i])),
                           "level": float(trig), "quality": "STRONG", "notes": [], "detail": d, "subtype": key})
    fv = ev["fvg"]
    if fv["continuation"] and tr["close_above_ema50"]:
        b = fv["bullish"]
        setups.append({"code": "F", "name": "FVG continuation (retrace into bullish FVG and rejection)", "confirmed": True, "entry_basis": "close", "trigger": b["high"],
                       "structure_low": min(b["low"], float(df["Low"].iloc[i])), "level": b["high"],
                       "quality": "STRONG_DISPLACEMENT" if b["strong_displacement"] else "NORMAL", "notes": []})
    order = {c: k for k, c in enumerate(cfg["setups"]["priority"])}
    setups.sort(key=lambda s: (not s["confirmed"], order.get(s["code"], 99)))
    for s in setups:
        s["stop_buffer"] = buf
    ev["setups"] = setups
    ev["primary"] = setups[0] if setups else None
    ev["invalidated"] = any(w["severity"] == "HIGH" for w in warns)
    return ev


def trade_plan(ev: dict, cfg: dict, capital: float, adv_cr: float | None) -> dict:
    """Formula-derived reference levels. NOT targets: next resistance is shown only as a distance for reward/risk."""
    tp, setup = cfg["trade_plan"], ev["primary"]
    out = {"entry_ref": None, "entry_basis": None, "invalidation": None, "stop_pct": None, "risk_per_share": None, "qty": None,
           "exposure": None, "risk_amount": None, "next_resistance": None, "reward_ref": None, "rr": None, "issues": []}
    if not setup:
        out["issues"].append("no actionable setup")
        return out
    close = ev["close"]
    entry = close if (setup["confirmed"] or tp["entry_unconfirmed"] == "close") else float(setup["trigger"])
    entry = max(entry, close) if not setup["confirmed"] else entry
    invalid = setup["structure_low"] - setup["stop_buffer"]
    out.update(entry_ref=entry, entry_basis="last close (confirmed setup)" if setup["confirmed"] else "trigger level (NOT confirmed yet)",
               invalidation=invalid)
    if not np.isfinite(invalid) or invalid >= entry:
        out["issues"].append("no valid technical invalidation level below entry")
        return out
    rps = entry - invalid
    out["risk_per_share"], out["stop_pct"] = rps, rps / entry * 100.0
    if out["stop_pct"] > float(tp["max_stop_pct"]):
        out["issues"].append(f"invalidation is {out['stop_pct']:.1f}% below entry (> {tp['max_stop_pct']}%): no reasonable invalidation")
    risk_budget = capital * float(cfg["risk_per_trade_pct"]) / 100.0
    qty = int(risk_budget // rps)
    max_pos = capital * float(cfg["max_position_pct_of_capital"]) / 100.0
    qty = min(qty, int(max_pos // entry))
    if adv_cr and np.isfinite(adv_cr):
        cap_value = adv_cr * 1e7 * float(cfg["liquidity"]["max_participation_pct"]) / 100.0
        qty = min(qty, int(cap_value // entry))
    out.update(qty=qty, exposure=qty * entry, risk_amount=qty * rps)
    nr = lv.nearest_resistance(ev["zones_now"], entry)
    if nr is not None and nr["low"] > entry * 1.002:
        out["next_resistance"] = (nr["low"], nr["high"])
        out["reward_ref"] = nr["low"] - entry
        out["rr"] = out["reward_ref"] / rps
    elif nr is not None:
        out["next_resistance"] = (nr["low"], nr["high"])
        out["reward_ref"], out["rr"] = 0.0, 0.0
    if out["rr"] is not None and out["rr"] < float(tp["min_rr"]):
        out["issues"].append(f"next resistance too close: reward/risk {out['rr']:.2f} < {tp['min_rr']}")
    return out
