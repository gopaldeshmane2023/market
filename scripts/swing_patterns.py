"""Daily/weekly candle-pattern engine for the swing strategy.

Everything here is vectorised and CAUSAL: the value at bar i only uses bars <= i, so a pattern detected on bar i is
known at the close of bar i (no look-ahead). Each pattern has an explicit mathematical definition and thresholds from
config/swing.yml -> patterns. Geometry scores (0..1) say how clearly the candle satisfies its definition.

Definitions (O,H,L,C = open/high/low/close; body=|C-O|; range=H-L; upper=H-max(O,C); lower=min(O,C)-L):
  hammer           prior down-move; lower/body >= 2; upper/range <= 0.25; body in top 40% of range; body/range >= 0.10
  inverted_hammer  prior down-move; upper/body >= 2; lower/range <= 0.15; body in bottom 40% of range
  bullish_engulfing prior down-move; red candle then green candle with O <= prev C and C >= prev O and a larger body
  morning_star     long red; small-body candle at/below the red close; green closing above the red body midpoint
  piercing         long red; green opens at/below prev close and closes above the red midpoint but below its open
  tweezer_bottom   red then green with lows equal within 0.1 ATR after a down-move
  bullish_harami   long red; green body fully inside the red body and <= 50% of its size
  dragonfly_doji   body/range <= 0.10; lower wick >= 65% of range; upper wick <= 10% of range; after a down-move
  three_inside_up  bullish harami followed by a green close above the first candle's open
  three_white_soldiers three green candles, rising closes, each opening inside the prior body, closing near the high
  bullish_marubozu green; body/range >= 0.90 and body >= 0.8 ATR
  inside_bar_breakout inside bar (within the mother bar's range) then a green close above the mother bar's high
Bearish warnings mirror these (shooting_star, hanging_man, bearish_engulfing, evening_star, bearish_harami,
dark_cloud) plus long_upper_wick_rejection, distribution_candle and bullish_trend_exhaustion.
A candle with a tiny body is NOT called a doji/hammer unless it satisfies the definition above.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from technical_screen import atr as _atr, ema as _ema, rsi as _rsi

BULL_REVERSAL = ["hammer", "inverted_hammer", "bullish_engulfing", "morning_star", "piercing", "tweezer_bottom",
                 "bullish_harami", "dragonfly_doji", "three_inside_up", "three_white_soldiers"]
BULL_CONTINUATION_SIMPLE = ["bullish_marubozu", "inside_bar_breakout"]
BEARISH = ["shooting_star", "hanging_man", "bearish_engulfing", "evening_star", "bearish_harami", "dark_cloud",
           "long_upper_wick_rejection", "distribution_candle", "bullish_trend_exhaustion"]
ALL_PATTERNS = BULL_REVERSAL + BULL_CONTINUATION_SIMPLE + BEARISH
PATTERN_BARS = {"hammer": 1, "inverted_hammer": 1, "bullish_engulfing": 2, "morning_star": 3, "piercing": 2,
                "tweezer_bottom": 2, "bullish_harami": 2, "dragonfly_doji": 1, "three_inside_up": 3,
                "three_white_soldiers": 3, "bullish_marubozu": 1, "inside_bar_breakout": 3}


def candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar candle geometry plus ATR14, EMA20/50/200, RSI14, 20-bar relative volume and prior-trend flags."""
    o, h, l, c, v = (df[k] for k in ("Open", "High", "Low", "Close", "Volume"))
    m = pd.DataFrame(index=df.index)
    m["body"] = (c - o).abs()
    m["range"] = h - l
    m["upper"] = h - np.maximum(o, c)
    m["lower"] = np.minimum(o, c) - l
    rng = m["range"].replace(0, np.nan)
    body = m["body"].replace(0, np.nan)
    m["body_ratio"] = m["body"] / rng
    m["upper_ratio"] = m["upper"] / rng
    m["lower_ratio"] = m["lower"] / rng
    zero_body = (m["body"] == 0) & (m["range"] > 0)
    m["uw_body"] = (m["upper"] / body).where(~zero_body, 99.0)
    m["lw_body"] = (m["lower"] / body).where(~zero_body, 99.0)
    m["close_pos"] = (c - l) / rng
    m["bull"] = c > o
    m["bear"] = c < o
    m["atr"] = _atr(h, l, c, 14)
    m["ema20"], m["ema50"], m["ema200"] = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    m["rsi"] = _rsi(c, 14)
    avg_vol = v.shift(1).rolling(20, min_periods=20).mean()
    m["rel_vol"] = v / avg_vol.replace(0, np.nan)
    return m


def _context(m: pd.DataFrame, c: pd.Series, p: dict) -> tuple[pd.Series, pd.Series]:
    n, mv = int(p["trend_lookback"]), float(p["min_trend_move_pct"]) / 100.0
    prev = c.shift(1)
    ret = prev / c.shift(1 + n) - 1.0
    prior_down = (prev < m["ema20"].shift(1)) & (ret < -mv)
    prior_up = (prev > m["ema20"].shift(1)) & (ret > mv)
    return prior_down.fillna(False), prior_up.fillna(False)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def detect_patterns(df: pd.DataFrame, p: dict, buffer_pct: float = 0.25) -> pd.DataFrame:
    """Return metrics + boolean `pat_<name>` and float `geo_<name>` (shape quality 0..1) columns for every bar."""
    m = candle_metrics(df)
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    prior_down, prior_up = _context(m, c, p)
    m["prior_down"], m["prior_up"] = prior_down, prior_up
    atr, body, rng = m["atr"], m["body"], m["range"]
    big_enough = rng >= float(p["min_range_atr"]) * atr
    S = lambda x, k=1: x.shift(k)  # noqa: E731
    strong = lambda k=0: (S(body, k) >= float(p["strong_body_atr"]) * S(atr, k)) if k else (body >= float(p["strong_body_atr"]) * atr)  # noqa: E731

    pat: dict[str, tuple[pd.Series, pd.Series]] = {}

    hammer_shape = (m["lw_body"] >= p["hammer_lower_body_min"]) & (m["upper_ratio"] <= p["hammer_upper_range_max"]) & \
                   (m["close_pos"] >= p["hammer_close_pos_min"]) & (m["body_ratio"] >= p["min_body_ratio"]) & big_enough
    hammer_geo = _clip01(0.5 * m["lw_body"] / (2 * p["hammer_lower_body_min"]) + 0.5 * m["close_pos"])
    pat["hammer"] = (hammer_shape & prior_down, hammer_geo)
    pat["hanging_man"] = (hammer_shape & prior_up, hammer_geo)

    inv_shape = (m["uw_body"] >= p["inverted_upper_body_min"]) & (m["lower_ratio"] <= p["inverted_lower_range_max"]) & \
                (m["close_pos"] <= p["inverted_close_pos_max"]) & (m["body_ratio"] >= p["min_body_ratio"]) & big_enough
    inv_geo = _clip01(0.5 * m["uw_body"] / (2 * p["inverted_upper_body_min"]) + 0.5 * (1 - m["close_pos"]))
    pat["inverted_hammer"] = (inv_shape & prior_down, inv_geo)
    pat["shooting_star"] = (inv_shape & prior_up, inv_geo)

    bear1, bull1 = S(m["bear"]).fillna(False).astype(bool), S(m["bull"]).fillna(False).astype(bool)
    be_cond = bear1 & m["bull"] & (o <= S(c)) & (c >= S(o)) & (body > S(body)) & (m["body_ratio"] >= p["min_body_ratio"])
    ratio = (body / S(body).replace(0, np.nan)).fillna(0)
    pat["bullish_engulfing"] = (be_cond & prior_down, _clip01(ratio / 2.5))
    bear_eng = bull1 & m["bear"] & (o >= S(c)) & (c <= S(o)) & (body > S(body)) & (m["body_ratio"] >= p["min_body_ratio"])
    pat["bearish_engulfing"] = (bear_eng & prior_up, _clip01(ratio / 2.5))

    mid1 = (S(o, 2) + S(c, 2)) / 2
    star_small = S(body) <= p["star_small_body_ratio"] * S(body, 2)
    ms = S(m["bear"], 2).fillna(False).astype(bool) & (S(body, 2) >= p["strong_body_atr"] * S(atr, 2)) & star_small & \
        (np.maximum(S(o), S(c)) <= S(c, 2) + 0.1 * S(atr)) & m["bull"] & (c >= mid1) & S(prior_down, 2)
    pat["morning_star"] = (ms, _clip01((c - mid1) / (S(body, 2).replace(0, np.nan) / 2 + 1e-9)))
    mid1u = mid1
    es = S(m["bull"], 2).fillna(False).astype(bool) & (S(body, 2) >= p["strong_body_atr"] * S(atr, 2)) & star_small & \
        (np.minimum(S(o), S(c)) >= S(c, 2) - 0.1 * S(atr)) & m["bear"] & (c <= mid1u) & S(prior_up, 2)
    pat["evening_star"] = (es, _clip01((mid1u - c) / (S(body, 2).replace(0, np.nan) / 2 + 1e-9)))

    mid_prev = (S(o) + S(c)) / 2
    pierc = bear1 & (S(body) >= p["strong_body_atr"] * S(atr)) & m["bull"] & (o <= S(c)) & (c > mid_prev) & (c < S(o))
    pat["piercing"] = (pierc & prior_down, _clip01((c - mid_prev) / (S(body).replace(0, np.nan) / 2 + 1e-9)))
    dark = bull1 & (S(body) >= p["strong_body_atr"] * S(atr)) & m["bear"] & (o >= S(c)) & (c < mid_prev) & (c > S(o))
    pat["dark_cloud"] = (dark & prior_up, _clip01((mid_prev - c) / (S(body).replace(0, np.nan) / 2 + 1e-9)))

    tol = float(p["tweezer_tol_atr"]) * atr
    tw = bear1 & m["bull"] & ((l - S(l)).abs() <= tol) & prior_down & big_enough
    pat["tweezer_bottom"] = (tw, _clip01(1 - (l - S(l)).abs() / (tol + 1e-9)))

    harami_core = bear1 & (S(body) >= p["strong_body_atr"] * S(atr)) & (o >= S(c)) & (c <= S(o)) & \
        (body <= p["harami_body_ratio"] * S(body))
    pat["bullish_harami"] = (harami_core & m["bull"] & prior_down, _clip01(1 - body / (S(body).replace(0, np.nan) * p["harami_body_ratio"] + 1e-9)))
    bear_h = bull1 & (S(body) >= p["strong_body_atr"] * S(atr)) & (o <= S(c)) & (c >= S(o)) & \
        (body <= p["harami_body_ratio"] * S(body)) & m["bear"] & prior_up
    pat["bearish_harami"] = (bear_h, _clip01(1 - body / (S(body).replace(0, np.nan) * p["harami_body_ratio"] + 1e-9)))

    dfly = (m["body_ratio"] <= p["doji_body_ratio_max"]) & (m["lower_ratio"] >= p["dragonfly_lower_range_min"]) & \
           (m["upper_ratio"] <= p["dragonfly_upper_range_max"]) & big_enough & prior_down
    pat["dragonfly_doji"] = (dfly, _clip01(m["lower_ratio"]))

    tiu = S(pat["bullish_harami"][0]).fillna(False).astype(bool) & m["bull"] & (c > S(o, 2))
    pat["three_inside_up"] = (tiu, _clip01((c - S(o, 2)) / (S(atr) + 1e-9)))

    def soldier(k):
        return S(m["bull"], k).fillna(False).astype(bool) & (S(m["body_ratio"], k) >= p["soldiers_body_ratio"]) & \
            (S(m["upper_ratio"], k) <= p["soldiers_upper_range_max"])
    tws = (m["bull"] & (m["body_ratio"] >= p["soldiers_body_ratio"]) & (m["upper_ratio"] <= p["soldiers_upper_range_max"]))
    tws = tws & soldier(1) & soldier(2) & (c > S(c)) & (S(c) > S(c, 2)) & (o >= S(o)) & (o <= S(c)) & \
        (S(o) >= S(o, 2)) & (S(o) <= S(c, 2)) & (S(c, 3) <= S(m["ema20"], 3) * 1.02)
    pat["three_white_soldiers"] = (tws, _clip01((m["body_ratio"] + S(m["body_ratio"]) + S(m["body_ratio"], 2)) / 3))

    maru = m["bull"] & (m["body_ratio"] >= p["marubozu_body_ratio"]) & (body >= p["marubozu_body_atr"] * atr)
    pat["bullish_marubozu"] = (maru, _clip01(m["body_ratio"]))
    inside = (S(h) <= S(h, 2)) & (S(l) >= S(l, 2))
    ibb = inside & (c > S(h, 2) * (1 + buffer_pct / 100.0)) & m["bull"] & (m["body_ratio"] >= 0.4)
    pat["inside_bar_breakout"] = (ibb, _clip01(m["body_ratio"]))

    rej = (m["upper_ratio"] >= p["rejection_upper_range"]) & (rng >= 0.7 * atr) & (m["close_pos"] <= 0.4)
    pat["long_upper_wick_rejection"] = (rej, _clip01(m["upper_ratio"]))
    dist = (m["rel_vol"] >= p["distribution_rel_vol"]) & ((m["close_pos"] <= p["distribution_close_pos_max"]) | m["bear"]) & \
           prior_up & (rng >= 1.0 * atr)
    pat["distribution_candle"] = (dist, _clip01(m["rel_vol"] / (2 * p["distribution_rel_vol"])))
    ext = ((c - m["ema20"]) >= p["exhaustion_atr"] * atr) | (m["rsi"] >= p["exhaustion_rsi"])
    exh = ext & ((m["upper_ratio"] >= 0.3) | m["bear"])
    pat["bullish_trend_exhaustion"] = (exh, _clip01((c - m["ema20"]) / (2 * p["exhaustion_atr"] * atr + 1e-9)))

    cols = {}
    for name, (flag, geo) in pat.items():
        cols[f"pat_{name}"] = flag.fillna(False).astype(bool)
        cols[f"geo_{name}"] = pd.to_numeric(geo, errors="coerce").fillna(0.0)
    return pd.concat([m, pd.DataFrame(cols, index=df.index)], axis=1)


def describe_pattern(name: str, m: pd.DataFrame, i: int) -> str:
    """Formula-based explanation with the actual numbers of bar i."""
    r = m.iloc[i]
    base = (f"body/range {r['body_ratio']:.2f}, upper wick/body {r['uw_body']:.2f}, lower wick/body {r['lw_body']:.2f}, "
            f"close at {r['close_pos'] * 100:.0f}% of range, relative volume {r['rel_vol']:.2f}x" if np.isfinite(r["range"]) and r["range"] > 0 else "no range")
    rules = {"hammer": "lower wick >= 2x body, small upper wick, body in top 40% of range, after a down-move",
             "inverted_hammer": "upper wick >= 2x body, small lower wick, body in bottom 40%, after a down-move",
             "bullish_engulfing": "green body engulfs the prior red body after a down-move",
             "morning_star": "long red, small star at/below its close, green closing above the red midpoint",
             "piercing": "green opens at/below prior close, closes above the red midpoint but below its open",
             "tweezer_bottom": "two candles with equal lows (within 0.1 ATR) after a down-move",
             "bullish_harami": "small green body inside a long red body after a down-move",
             "dragonfly_doji": "tiny body at the top, long lower wick, no upper wick, after a down-move",
             "three_inside_up": "bullish harami confirmed by a green close above the first candle's open",
             "three_white_soldiers": "three rising green candles, each opening inside the prior body, closing near highs",
             "bullish_marubozu": "green candle with body >= 90% of range and >= 0.8 ATR",
             "inside_bar_breakout": "green close above the mother bar's high after an inside bar"}
    return f"{name.replace('_', ' ')}: {rules.get(name, '')} | {base}"
