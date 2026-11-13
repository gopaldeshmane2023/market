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
