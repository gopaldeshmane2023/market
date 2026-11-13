import numpy as np
import pandas as pd
import pytest

from synth import make_ohlcv
import technical_screen as ts


def test_ema_matches_manual_recursion():
    s = pd.Series(np.arange(1.0, 41.0) ** 1.1)
    span = 10
    alpha = 2 / (span + 1)
    manual = [s.iloc[0]]
    for v in s.iloc[1:]:
        manual.append(alpha * v + (1 - alpha) * manual[-1])
    got = ts.ema(s, span)
    assert got.iloc[:span - 1].isna().all()                      # min_periods respected
    assert np.allclose(got.iloc[span - 1:].values, np.array(manual)[span - 1:])


def test_rsi_extremes_and_flat():
    up = pd.Series(np.arange(1.0, 60.0))
    down = pd.Series(np.arange(60.0, 1.0, -1.0))
    flat = pd.Series([10.0] * 60)
    assert ts.rsi(up).iloc[-1] == pytest.approx(100.0)
    assert ts.rsi(down).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert ts.rsi(flat).iloc[-1] == pytest.approx(50.0)


def test_rsi_matches_wilder_reference():
    rng = np.random.default_rng(3)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 120)))
    delta = close.diff().dropna()
    gains, losses = delta.clip(lower=0).values, (-delta.clip(upper=0)).values
    ag, al = gains[:14].mean(), losses[:14].mean()          # Wilder seed = simple mean of first 14
    for g, l in zip(gains[14:], losses[14:]):
        ag, al = (ag * 13 + g) / 14, (al * 13 + l) / 14
    ref = 100 - 100 / (1 + ag / al)
    # ewm(adjust=False) seeds differently from the simple-mean seed; after >100 bars the difference is negligible
    assert ts.rsi(close, 14).iloc[-1] == pytest.approx(ref, abs=0.5)


def test_macd_components_consistent():
    df = make_ohlcv(1, 120)
    m = ts.macd(df["Close"])
    assert np.allclose((ts.ema(df["Close"], 12) - ts.ema(df["Close"], 26)).dropna(), m["macd"].dropna())
    assert np.allclose(m["macd_hist"].dropna(), (m["macd"] - m["macd_signal"]).dropna())


def test_atr_constant_range():
    n = 60
    close = pd.Series([100.0] * n)
    high, low = close + 1, close - 1
    assert ts.atr(high, low, close).iloc[-1] == pytest.approx(2.0)


def test_relative_volume_uses_prior_sessions_only():
    df = make_ohlcv(2, 80, volume=1_000_000)
    df.iloc[-1, df.columns.get_loc("Volume")] = 3_000_000
    ind = ts.compute_indicators(df)
    expected = 3_000_000 / df["Volume"].iloc[-21:-1].mean()
    assert ind["rel_volume"].iloc[-1] == pytest.approx(expected)
    assert ind["rel_volume"].iloc[-1] > 2.5          # today's spike is not diluted by itself


def test_distance_from_highs_never_positive():
    ind = ts.compute_indicators(make_ohlcv(4, 120))
    assert (ind["dist_high20_pct"].dropna() <= 1e-9).all()
    assert (ind["dist_high50_pct"].dropna() <= 1e-9).all()


def test_clean_ohlcv_drops_bad_rows():
    df = make_ohlcv(5, 30)
    df.iloc[3, df.columns.get_loc("Close")] = np.nan
    df.iloc[4, df.columns.get_loc("Close")] = -1
    dup = pd.concat([df, df.iloc[[10]]])
    out = ts.clean_ohlcv(dup)
    assert len(out) == 28 and out.index.is_unique and out.index.is_monotonic_increasing
    assert ts.clean_ohlcv(pd.DataFrame({"x": [1]})) is None
    assert ts.clean_ohlcv(None) is None
