"""Synthetic data builders used only by the tests (no network, fully deterministic)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(seed: int = 0, n: int = 300, drift: float = 0.0015, vol: float = 0.01, base: float = 100.0,
               volume: float = 2_000_000, end: str = "2026-11-11", last_move: float | None = None,
               last_vol_mult: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    if last_move is not None:
        rets[-1] = last_move
    close = base * np.cumprod(1 + rets)
    open_ = np.r_[base, close[:-1]] * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    vols = volume * (1 + rng.normal(0, 0.15, n)).clip(0.3)
    vols[-1] *= last_vol_mult
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vols}, index=idx)


# seed/last_move/last_vol_mult below were found offline (dev time) to satisfy every bullish criterion;
# test_passing_series_is_really_passing guards that this stays true.
PASSING = dict(seed=6, drift=0.0012, vol=0.01, last_move=0.012, last_vol_mult=2.2)


def passing_ohlcv(**overrides) -> pd.DataFrame:
    return make_ohlcv(**{**PASSING, **overrides})


def breakout_ohlcv(variant: str = "confirmed", seed: int = 1, n_trend: int = 290, n_base: int = 30, end: str = "2026-11-11",
                   volume: float = 2_000_000, base_price: float = 100.0) -> pd.DataFrame:
    """Uptrend -> 30-bar base capped by resistance R (touched several times) -> a final candle whose shape depends on `variant`:
      confirmed      strong green body closing well above R on 3x volume
      near           closes just BELOW R (approaching resistance, no breakout)
      wick_reject    trades above R intraday, long upper wick, closes below R
      weak_close     closes above R+buffer but with a long upper wick and weak close
      inside_buffer  closes above R but inside the buffer
    """
    rng = np.random.default_rng(seed)
    trend = base_price * np.cumprod(1 + rng.normal(0.003, 0.005, n_trend))
    top = float(trend.max())              # the base sits at the highs: nothing above R in the history
    R = top * 1.03
    phase = np.linspace(0, 5 * np.pi, n_base)
    bclose = R * (0.975 + 0.02 * np.sin(phase))     # base closes stay below R; its top forms the resistance zone
    closes = np.r_[trend, bclose]
    n = len(closes)
    opens = np.r_[trend[0], closes[:-1]] * (1 + rng.normal(0, 0.0008, n))
    highs = np.maximum(opens, closes) * 1.004
    lows = np.minimum(opens, closes) * 0.996
    for k in range(n_trend, n):                                           # base highs capped at R, with repeated touches of R
        highs[k] = min(highs[k], R)
        if (k - n_trend) % 7 == 3:
            highs[k] = R
    vols = volume * (1 + rng.normal(0, 0.12, n)).clip(0.4)
    last_o = R * 0.995
    spec = {"confirmed": (R * 1.030, R * 1.032, last_o * 0.998, 3.0),
            "near": (R * 0.992, R * 0.994, last_o * 0.99, 2.0),
            "wick_reject": (R * 0.993, R * 1.040, last_o * 0.99, 3.0),
            "weak_close": (R * 1.006, R * 1.045, last_o * 0.99, 3.0),
            "inside_buffer": (R * 1.001, R * 1.004, last_o * 0.99, 2.0)}[variant]
    c, h, l, vm = spec
    closes[-1], highs[-1], lows[-1], opens[-1] = c, h, l, last_o
    highs[-1] = max(highs[-1], c, last_o)
    lows[-1] = min(lows[-1], c, last_o)
    vols[-1] = volume * vm
    idx = pd.bdate_range(end=end, periods=n)
    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols}, index=idx)
    df.attrs["resistance"] = float(R)
    return df


def from_rows(rows, start_price=100.0, end="2026-11-11", vol=1_000_000, lead=0):
    """Build an OHLCV frame from explicit (o,h,l,c[,v]) rows, with `lead` flat bars before them."""
    rows = [tuple(r) + ((vol,) if len(r) == 4 else ()) for r in rows]
    flat = [(start_price, start_price + 0.5, start_price - 0.5, start_price, vol)] * lead
    allr = flat + rows
    idx = pd.bdate_range(end=end, periods=len(allr))
    return pd.DataFrame(allr, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)


def trend_rows(n=60, start=100.0, end_price=70.0, vol=1_000_000, wick=0.4):
    """Smooth trend (down if end<start) of ordinary candles; used as context for pattern tests."""
    closes = np.linspace(start, end_price, n)
    rows, prev = [], start
    for c in closes:
        rows.append((prev, max(prev, c) + wick, min(prev, c) - wick, c, vol))
        prev = c
    return rows


def append_bars(df, rows, vol=None):
    """Append explicit (o,h,l,c[,v]) rows on following business days."""
    v = vol if vol is not None else float(df["Volume"].tail(20).mean())
    rows = [tuple(r) + ((v,) if len(r) == 4 else ()) for r in rows]
    idx = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=len(rows))
    extra = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close", "Volume"], index=idx)
    out = pd.concat([df, extra])
    out.attrs = df.attrs
    return out
