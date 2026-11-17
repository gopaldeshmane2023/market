"""Support/resistance zones, breakout classification, retests, FVGs, chart patterns and weekly structure.

NO LOOK-AHEAD: a pivot at bar j is only used once bar j+pivot_right has closed; resistance for a breakout on bar i is
built from bars <= i-1 only; weekly values at bar i use COMPLETED weeks only.

Breakout statuses (never mix them):
  POTENTIAL_BREAKOUT    price within `potential_pct` below resistance (or closed above the level but inside the buffer)
  BREAKOUT_PENDING_CLOSE today's bar is still forming and has crossed the trigger: NOT confirmed
  CONFIRMED_BREAKOUT    daily CLOSE > resistance + buffer (quality STRONG/WEAK from body, close position, wick, volume)
  BREAKOUT_RETEST       a recent confirmed breakout was revisited, held, and printed a bullish candle
  FAILED_BREAKOUT       after a confirmed breakout a close fell back below the level (minus tolerance)
  RESISTANCE_REJECTION  crossed the level intraday but closed below it (or long upper wick with a weak close)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- pivots
def pivot_arrays(h: np.ndarray, l: np.ndarray, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """Boolean arrays marking pivot highs/lows at index j. A pivot at j is only KNOWN at bar j+right."""
    n = len(h)
    ph, pl = np.zeros(n, bool), np.zeros(n, bool)
    for j in range(left, n - right):
        if h[j] > h[j - left:j].max() and h[j] >= h[j + 1:j + right + 1].max():
            ph[j] = True
        if l[j] < l[j - left:j].min() and l[j] <= l[j + 1:j + right + 1].min():
            pl[j] = True
    return ph, pl


# --------------------------------------------------------------------------- zones
def build_zones(prep, i: int) -> list[dict]:
    """Support/resistance zones from data up to and including bar i (call with i-1 to get a breakout's prior resistance)."""
    cfg = prep.cfg["levels"]
    df, m = prep.df, prep.m
    if i < 25:
        return []
    h, l, c, v = df["High"].values, df["Low"].values, df["Close"].values, df["Volume"].values
    atr = m["atr"].iloc[i] if np.isfinite(m["atr"].iloc[i]) else c[i] * 0.02
    tol = max(float(cfg["zone_tol_pct"]) / 100.0 * c[i], float(cfg["zone_tol_atr"]) * atr)
    lo_i = max(0, i - int(cfg["lookback"]) + 1)
    right = int(cfg["pivot_right"])
    cands: list[tuple[float, float, str, float]] = []   # (low, high, kind, weight)
    known = i - right
    for j in range(lo_i, known + 1):
        if prep.ph[j]:
            cands.append((h[j], h[j], "pivot_high", 1.0))
        if prep.pl[j]:
            cands.append((l[j], l[j], "pivot_low", 1.0))
    for n, kind_h, kind_l in ((20, "high20", "low20"), (50, "high50", "low50")):
        a = max(0, i - n + 1)
        cands.append((h[a:i + 1].max(),) * 2 + (kind_h, 1.0))
        cands.append((l[a:i + 1].min(),) * 2 + (kind_l, 1.0))
    vb = int(cfg["volume_zone_bars"])
    a = max(0, i - vb + 1)
    if i - a >= 20:
        cl = c[a:i + 1]
        vols = v[a:i + 1]
        width = max(tol, 1e-9)
        bins = np.floor((cl - cl.min()) / width).astype(int)
        sums = np.bincount(bins, weights=vols)
        mean_v = sums[sums > 0].mean() if (sums > 0).any() else 0
        for b in np.where(sums >= float(cfg["volume_zone_mult"]) * mean_v)[0]:
            if sums[b] > 0:
                cands.append((cl.min() + b * width, cl.min() + (b + 1) * width, "volume_zone", 1.5))
    cands.sort(key=lambda x: (x[0] + x[1]) / 2)
    zones: list[dict] = []
    for lo, hi, kind, w in cands:
        if zones and (lo - zones[-1]["high"]) <= tol:
            z = zones[-1]
            z["high"], z["low"] = max(z["high"], hi), min(z["low"], lo)
            z["kinds"].append(kind)
            z["strength"] += w
            if kind.startswith("pivot"):
                z["touches"] += 1
        else:
            zones.append({"low": lo, "high": hi, "kinds": [kind], "strength": w, "touches": 1 if kind.startswith("pivot") else 0})
    out = []
    for z in zones:
        z["mid"] = (z["low"] + z["high"]) / 2
        z["tol"] = tol
        z["rejections"] = sum(1 for k in z["kinds"] if k == "pivot_high")
        if z["strength"] >= float(cfg["min_zone_strength"]):
            out.append(z)
    return out


def nearest_support(zones: list[dict], price: float) -> dict | None:
    below = [z for z in zones if z["mid"] <= price]
    return max(below, key=lambda z: z["high"]) if below else None


def nearest_resistance(zones: list[dict], price: float) -> dict | None:
    above = [z for z in zones if z["mid"] > price and z["low"] > price * 0.999]
    return min(above, key=lambda z: z["low"]) if above else None


def at_support(zones: list[dict], low: float, cfg_levels: dict) -> dict | None:
    """Zone (support-capable) that a candle's low reached: low within zone +/- tolerance."""
    prox = float(cfg_levels["support_proximity_pct"]) / 100.0
    best = None
    for z in zones:
        t = max(z["tol"], prox * z["mid"])
        if z["low"] - t <= low <= z["high"] + t:
            if best is None or z["strength"] > best["strength"]:
                best = z
    return best


# --------------------------------------------------------------------------- breakout
def _buffer(prep, i: int, level: float) -> float:
    b = prep.cfg["breakout"]
    atr = prep.m["atr"].iloc[i]
    return max(level * float(b["buffer_pct"]) / 100.0, float(b["buffer_atr"]) * (atr if np.isfinite(atr) else 0.0))


def classify_breakout(prep, i: int, zones_prev: list[dict] | None = None) -> dict:
    """Classify bar i against resistance built from bars <= i-1. Uses only the CLOSED candle i."""
    cfg = prep.cfg["breakout"]
    df, m = prep.df, prep.m
    out = {"status": "NONE", "level": None, "trigger": None, "zone": None, "quality": None, "quality_issues": [],
           "possible_failed_breakout": False}
    if i < 30:
        return out
    zones_prev = zones_prev if zones_prev is not None else build_zones(prep, i - 1)
    c_prev = df["Close"].iloc[i - 1]
    cands = [z for z in zones_prev if z["high"] >= c_prev * 0.999]
    if not cands:
        return out
    z = min(cands, key=lambda z: z["high"])
    level = z["high"]
    trig = level + _buffer(prep, i, level)
    o, h, l, c = (df[k].iloc[i] for k in ("Open", "High", "Low", "Close"))
    r = m.iloc[i]
    out.update(level=level, trigger=trig, zone=z)
    upper_ratio = r["upper_ratio"] if np.isfinite(r["upper_ratio"]) else 0.0
    close_pos = r["close_pos"] if np.isfinite(r["close_pos"]) else 0.5
    if c > trig:
        issues = []
        if not (r["bull"] and r["body_ratio"] >= cfg["min_body_ratio"]):
            issues.append(f"weak body (body/range {r['body_ratio']:.2f} < {cfg['min_body_ratio']})")
        if close_pos < cfg["min_close_pos"]:
            issues.append(f"close not near the high ({close_pos * 100:.0f}% of range < {cfg['min_close_pos'] * 100:.0f}%)")
        if upper_ratio > cfg["max_upper_ratio"]:
            issues.append(f"upper wick {upper_ratio * 100:.0f}% of range > {cfg['max_upper_ratio'] * 100:.0f}%")
        if not (np.isfinite(r["rel_vol"]) and r["rel_vol"] >= cfg["min_rel_vol"]):
            issues.append(f"volume {r['rel_vol']:.2f}x < {cfg['min_rel_vol']}x")
        if upper_ratio >= cfg["failed_upper_ratio"] and close_pos <= cfg["failed_close_pos"]:
            out.update(status="RESISTANCE_REJECTION", possible_failed_breakout=True, quality="REJECTED", quality_issues=issues)
        else:
            out.update(status="CONFIRMED_BREAKOUT", quality="STRONG" if not issues else "WEAK", quality_issues=issues)
    elif h > trig and c <= level:
        out.update(status="RESISTANCE_REJECTION", possible_failed_breakout=upper_ratio >= cfg["failed_upper_ratio"],
                   quality="REJECTED", quality_issues=["crossed resistance intraday but closed below it"])
    elif c > level:
        out.update(status="POTENTIAL_BREAKOUT", quality_issues=["closed above the level but inside the buffer: not confirmed"])
    elif (level - c) / level * 100.0 <= float(cfg["potential_pct"]):
        out.update(status="POTENTIAL_BREAKOUT")
    return out


def get_breakout(prep, i: int) -> dict:
    """Memoised classify_breakout with a cheap prefilter so full zone building only runs on plausible bars."""
    if i in prep.bo_memo:
        return prep.bo_memo[i]
    df, m = prep.df, prep.m
    res = {"status": "NONE", "level": None, "trigger": None, "zone": None, "quality": None, "quality_issues": [],
           "possible_failed_breakout": False}
    if i >= 30:
        c = df["Close"].values
        near_high = c[i] >= c[max(0, i - 20):i].max() * 0.96
        if near_high and c[i] >= c[i - 1] * 0.995:
            res = classify_breakout(prep, i)
    prep.bo_memo[i] = res
    return res


def retest_or_failed(prep, i: int) -> dict:
    """Look back at confirmed breakouts within `retest_window` bars: BREAKOUT_RETEST, FAILED_BREAKOUT or nothing."""
    cfg = prep.cfg["breakout"]
    df, m = prep.df, prep.m
    c, l, h = df["Close"].values, df["Low"].values, df["High"].values
    res = {"status": "NONE"}
    for j in range(i - 1, max(i - int(cfg["retest_window"]) - 1, 29), -1):
        ev = get_breakout(prep, j)
        if ev["status"] != "CONFIRMED_BREAKOUT":
            continue
        level = ev["level"]
        failed_at = [k for k in range(j + 1, i + 1) if c[k] < level * (1 - cfg["fail_tol_pct"] / 100.0)]
        if failed_at:
            if failed_at[-1] == i or failed_at[0] == i:
                return {"status": "FAILED_BREAKOUT", "level": level, "breakout_bar": j, "failed_bar": failed_at[0],
                        "failed_today": failed_at[0] == i}
            continue
        touched = l[i] <= level * (1 + cfg["retest_tol_pct"] / 100.0)
        held = c[i] >= level and l[i] >= level * (1 - cfg["retest_hold_tol_pct"] / 100.0)
        r = m.iloc[i]
        bullish = bool(r["bull"]) and np.isfinite(r["close_pos"]) and r["close_pos"] >= 0.6
        if touched and held and bullish and i - j >= 1:
            return {"status": "BREAKOUT_RETEST", "level": level, "breakout_bar": j, "retest_low": float(l[i]),
                    "breakout_quality": ev["quality"], "zone": ev["zone"]}
    return res


# --------------------------------------------------------------------------- fair value gaps
def fvg_scan(prep, i: int) -> dict:
    """Three-candle FVGs in the last `fvg.lookback` bars. Bullish: L[k] > H[k-2]. Bearish: H[k] < L[k-2]."""
    cfg = prep.cfg["fvg"]
    df, m = prep.df, prep.m
    h, l, c, o = df["High"].values, df["Low"].values, df["Close"].values, df["Open"].values
    res = {"bullish": None, "bearish": None, "continuation": False, "status": "NONE"}
    start = max(2, i - int(cfg["lookback"]))
    best, bear = None, None
    for k in range(start, i + 1):
        if l[k] > h[k - 2]:
            lo, hi = h[k - 2], l[k]
            pct = (hi - lo) / c[k] * 100.0
            if pct < float(cfg["min_gap_pct"]):
                continue
            mid_body = abs(c[k - 1] - o[k - 1])
            atr = m["atr"].iloc[k - 1]
            strong = bool(c[k - 1] > o[k - 1] and np.isfinite(atr) and mid_body >= float(cfg["displacement_atr"]) * atr
                          and np.isfinite(m["rel_vol"].iloc[k - 1]) and max(m["rel_vol"].iloc[k - 1], m["rel_vol"].iloc[k]) >= float(cfg["displacement_rel_vol"]))
            after_low = l[k + 1:i + 1]
            touched = bool(len(after_low) and after_low.min() <= hi)
            filled = bool(len(after_low) and after_low.min() <= lo)
            state = "MITIGATED" if filled else ("PARTIALLY_MITIGATED" if touched else "FRESH")
            rec = {"created_bar": k, "date": df.index[k].date().isoformat(), "low": float(lo), "high": float(hi), "size": float(hi - lo),
                   "pct": float(pct), "state": state, "strong_displacement": strong, "volume_confirmed": bool(strong)}
            if state != "MITIGATED" and (best is None or (rec["strong_displacement"], k) > (best["strong_displacement"], best["created_bar"])):
                best = rec
        if h[k] < l[k - 2]:
            lo, hi = h[k], l[k - 2]
            pct = (hi - lo) / c[k] * 100.0
            if pct >= float(cfg["min_gap_pct"]) and not (len(h[k + 1:i + 1]) and h[k + 1:i + 1].max() >= hi):
                bear = {"created_bar": k, "date": df.index[k].date().isoformat(), "low": float(lo), "high": float(hi), "pct": float(pct)}
    res["bullish"], res["bearish"] = best, bear
    if best:
        in_gap = l[i] <= best["high"] and l[i] >= best["low"] - m["atr"].iloc[i] * 0.25
        holds_above = c[i] >= best["high"] and c[i] > o[i]
        lower_ratio = m["lower_ratio"].iloc[i]
        first_touch_here = best["created_bar"] < i - 0 and (l[best["created_bar"] + 1:i].min() > best["high"] if i - best["created_bar"] > 1 else True)
        if in_gap and holds_above and np.isfinite(lower_ratio) and lower_ratio >= 0.25 and best["created_bar"] < i:
            res["continuation"] = True
            res["status"] = "RETRACE_REJECTION"
        elif best["state"] == "FRESH":
            res["status"] = "FRESH"
        elif best["state"] == "PARTIALLY_MITIGATED":
            res["status"] = "PARTIALLY_MITIGATED"
    elif bear:
        res["status"] = "BEARISH_FVG_OVERHEAD"
    return res


# --------------------------------------------------------------------------- weekly structure (completed weeks only)
def weekly_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (weekly_ohlcv, per-day mapping of the last COMPLETED week's trend). Week ends Friday."""
    wk = df.resample("W-FRI").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(subset=["Close"])
    wk["w_ema20"] = wk["Close"].ewm(span=20, adjust=False, min_periods=20).mean()
    wk["w_ema50"] = wk["Close"].ewm(span=50, adjust=False, min_periods=50).mean()
    wk["w_ema20_rising"] = wk["w_ema20"] > wk["w_ema20"].shift(3)
    ok = (wk["Close"] > wk["w_ema20"]) & (wk["w_ema20"] > wk["w_ema50"])
    down = (wk["Close"] < wk["w_ema20"]) & (wk["w_ema20"] < wk["w_ema50"])
    wk["w_status"] = np.where(wk["w_ema50"].isna(), "UNKNOWN", np.where(ok, "UP", np.where(down, "DOWN", "MIXED")))
    idx = df.index
    prev_friday = idx - pd.to_timedelta((idx.weekday - 4) % 7, unit="D")
    # a daily bar on Friday sees that Friday's weekly candle as completed; earlier days see the previous Friday
    cols = ["Close", "w_ema20", "w_ema50", "w_ema20_rising", "w_status"]
    uniq = prev_friday.unique()
    table = wk[cols].reindex(wk.index.union(uniq)).ffill().reindex(uniq)
    mapped = table.loc[prev_friday].copy()
    mapped.index = idx
    mapped = mapped.rename(columns={"Close": "w_close"})
    return wk, mapped


def strong_close(prep, i: int, min_rel_vol: float | None = None) -> bool:
    """Shared candle-quality gate for any breakout: green body, close near the high, small upper wick, volume expansion."""
    bo = prep.cfg["breakout"]
    r = prep.m.iloc[i]
    rv = bo["min_rel_vol"] if min_rel_vol is None else min_rel_vol
    return bool(r["bull"] and r["body_ratio"] >= bo["min_body_ratio"] and r["close_pos"] >= bo["min_close_pos"]
                and r["upper_ratio"] <= bo["max_upper_ratio"] and np.isfinite(r["rel_vol"]) and r["rel_vol"] >= rv)


# --------------------------------------------------------------------------- chart patterns
def consolidation_breakout(prep, i: int) -> dict | None:
    cfg, bo = prep.cfg["consolidation"], prep.cfg["breakout"]
    df, m = prep.df, prep.m
    n = int(cfg["bars"])
    if i < n + 1:
        return None
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    hi, lo = h[i - n:i].max(), l[i - n:i].min()
    atr = m["atr"].iloc[i - 1]
    width_pct = (hi - lo) / c[i - n:i].mean() * 100.0
    tight = width_pct <= float(cfg["max_width_pct"]) and np.isfinite(atr) and (hi - lo) <= float(cfg["max_width_atr"]) * atr * np.sqrt(n)
    trig = hi + _buffer(prep, i, hi)
    if tight and c[i] > trig and strong_close(prep, i, cfg["min_rel_vol"]):
        return {"range_high": float(hi), "range_low": float(lo), "width_pct": float(width_pct), "trigger": float(trig)}
    return None


def flag_breakout(prep, i: int) -> dict | None:
    """Bullish flag / pennant: strong pole, shallow tight consolidation, close above the flag high (causal)."""
    cfg, bo = prep.cfg["flag"], prep.cfg["breakout"]
    df, m = prep.df, prep.m
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    r = m.iloc[i]
    if i < 30 or not (c[i] > c[i - 1] and strong_close(prep, i)):
        return None
    for L in range(int(cfg["min_flag_bars"]), int(cfg["max_flag_bars"]) + 1):
        fs, fe = i - L, i - 1
        ps = fs - int(cfg["pole_bars"])
        if ps < 1:
            continue
        pole_gain = (c[fs - 1] / c[ps - 1] - 1.0) * 100.0
        if pole_gain < float(cfg["pole_min_pct"]):
            continue
        fh, fl = h[fs:fe + 1].max(), l[fs:fe + 1].min()
        pole_top, pole_bot = max(h[ps:fs].max(), fh), l[ps:fs].min()
        retrace = (pole_top - fl) / max(pole_top - pole_bot, 1e-9)
        width_pct = (fh - fl) / c[fs - 1] * 100.0
        if retrace > float(cfg["max_retrace"]) or width_pct > float(cfg["max_width_ratio"]) * pole_gain:
            continue
        if c[i] > fh + _buffer(prep, i, fh):
            xs = np.arange(L)
            sh, sl = np.polyfit(xs, h[fs:fe + 1], 1)[0], np.polyfit(xs, l[fs:fe + 1], 1)[0]
            kind = "pennant" if (sh < 0 < sl) else "flag"   # converging highs/lows => pennant
            return {"kind": kind, "pole_gain_pct": float(pole_gain), "flag_bars": L, "flag_high": float(fh), "flag_low": float(fl),
                    "retrace": float(retrace), "trigger": float(fh)}
    return None


def ascending_triangle_breakout(prep, i: int) -> dict | None:
    cfg = prep.cfg["triangle"]
    df, m = prep.df, prep.m
    right = int(prep.cfg["levels"]["pivot_right"])
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    a, known = max(0, i - int(cfg["lookback"])), i - 1 - right
    hi_idx = [j for j in range(a, known + 1) if prep.ph[j]]
    lo_idx = [j for j in range(a, known + 1) if prep.pl[j]]
    if len(hi_idx) < int(cfg["min_touches"]) or len(lo_idx) < 2:
        return None
    top = max(h[j] for j in hi_idx)
    flat = [j for j in hi_idx if abs(h[j] - top) / top * 100.0 <= float(cfg["flat_tol_pct"])]
    rising = l[lo_idx[-1]] > l[lo_idx[0]] * 1.005
    if len(flat) >= int(cfg["min_touches"]) and rising and c[i - 1] <= top + _buffer(prep, i, top) and c[i] > top + _buffer(prep, i, top) \
            and strong_close(prep, i):
        return {"resistance": float(top), "touches": len(flat), "last_swing_low": float(l[lo_idx[-1]])}
    return None


def cup_handle_breakout(prep, i: int) -> dict | None:
    cfg = prep.cfg["cup"]
    df, m = prep.df, prep.m
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    if i < int(cfg["min_bars"]) + int(cfg["handle_min_bars"]):
        return None
    for hl in range(int(cfg["handle_min_bars"]), int(cfg["handle_max_bars"]) + 1):
        he = i - 1
        hs = he - hl + 1
        win_start = max(0, hs - int(cfg["max_bars"]))
        seg = h[win_start:hs]
        if len(seg) < int(cfg["min_bars"]):
            continue
        left_i = win_start + int(np.argmax(seg))
        left_rim = h[left_i]
        cup = l[left_i:hs]
        if len(cup) < int(cfg["min_bars"]) - 5:
            continue
        low = cup.min()
        depth = (left_rim - low) / left_rim * 100.0
        if not (float(cfg["min_depth_pct"]) <= depth <= float(cfg["max_depth_pct"])):
            continue
        right_rim = h[hs:he + 1].max()
        if abs(right_rim - left_rim) / left_rim * 100.0 > float(cfg["rim_tol_pct"]):
            continue
        hdepth = (right_rim - l[hs:he + 1].min()) / right_rim * 100.0
        if hdepth > float(cfg["handle_max_depth_pct"]) or hdepth > depth / 2:
            continue
        # cup bottom should be roughly mid-cup (rounded, not a V): lowest point in the middle third
        low_pos = (int(np.argmin(cup))) / max(len(cup) - 1, 1)
        if not 0.2 <= low_pos <= 0.8:
            continue
        trig = max(h[hs:he + 1].max(), 0) + _buffer(prep, i, right_rim)
        if c[i] > trig and strong_close(prep, i):
            return {"left_rim": float(left_rim), "cup_depth_pct": float(depth), "handle_bars": hl, "handle_depth_pct": float(hdepth), "trigger": float(trig)}
    return None


# --------------------------------------------------------------------------- structure helpers
def hh_hl(prep, i: int) -> bool:
    """Last two CONFIRMED pivot highs ascending and last two CONFIRMED pivot lows ascending."""
    right = int(prep.cfg["levels"]["pivot_right"])
    known = i - right
    a = max(0, i - int(prep.cfg["levels"]["lookback"]))
    hi = [j for j in range(a, known + 1) if prep.ph[j]]
    lo = [j for j in range(a, known + 1) if prep.pl[j]]
    h, l = prep.df["High"].values, prep.df["Low"].values
    return bool(len(hi) >= 2 and len(lo) >= 2 and h[hi[-1]] > h[hi[-2]] and l[lo[-1]] > l[lo[-2]])
