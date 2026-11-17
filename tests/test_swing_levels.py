import numpy as np
import pandas as pd
import pytest

from common import REPO_ROOT, load_yaml
from synth import append_bars, breakout_ohlcv, from_rows, make_ohlcv
import swing_engine as se
import swing_levels as lv


@pytest.fixture()
def cfg():
    return load_yaml(REPO_ROOT / "config" / "swing.yml")


def ev_last(df, cfg):
    prep = se.prepare(df, cfg)
    return prep, se.evaluate_bar(prep, len(df) - 1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_breakout_statuses_never_mixed(cfg, seed):
    expected = {"confirmed": "CONFIRMED_BREAKOUT", "near": "POTENTIAL_BREAKOUT", "wick_reject": "RESISTANCE_REJECTION",
                "weak_close": "RESISTANCE_REJECTION", "inside_buffer": "POTENTIAL_BREAKOUT"}
    for variant, status in expected.items():
        df = breakout_ohlcv(variant, seed=seed)
        _, ev = ev_last(df, cfg)
        assert ev["breakout_status"] == status, variant
        assert ev["breakout"]["level"] == pytest.approx(df.attrs["resistance"], rel=1e-3)


def test_near_resistance_is_never_a_confirmed_breakout(cfg):
    for variant in ("near", "inside_buffer", "wick_reject", "weak_close"):
        _, ev = ev_last(breakout_ohlcv(variant, seed=4), cfg)
        assert not any(s["code"] == "A" and s["confirmed"] for s in ev["setups"]), variant
        assert ev["breakout_status"] != "CONFIRMED_BREAKOUT"


def test_intraday_cross_but_close_below_is_not_confirmed(cfg):
    df = breakout_ohlcv("wick_reject", seed=5)
    R = df.attrs["resistance"]
    assert df["High"].iloc[-1] > R * 1.0025 and df["Close"].iloc[-1] < R       # crossed intraday, closed below
    _, ev = ev_last(df, cfg)
    assert ev["breakout_status"] == "RESISTANCE_REJECTION" and ev["invalidated"]


def test_close_must_exceed_level_plus_buffer(cfg):
    df = breakout_ohlcv("inside_buffer", seed=6)
    R = df.attrs["resistance"]
    prep, ev = ev_last(df, cfg)
    assert df["Close"].iloc[-1] > R and df["Close"].iloc[-1] < ev["breakout"]["trigger"]
    assert ev["breakout_status"] == "POTENTIAL_BREAKOUT"


def test_engine_has_no_lookahead(cfg):
    """Evaluating bar i on the full history must equal evaluating it on data truncated at i."""
    df = pd.concat([make_ohlcv(7, 420, drift=0.001, vol=0.012)])
    full = se.prepare(df, cfg)
    for i in (330, 360, 395):
        part = se.prepare(df.iloc[:i + 1], cfg)
        a, b = se.evaluate_bar(full, i), se.evaluate_bar(part, i)
        sig = lambda e: (e["breakout_status"], e["breakout"]["level"], sorted((s["code"], s["confirmed"]) for s in e["setups"]),  # noqa: E731
                         [p["name"] for p in e["patterns"]], [w["name"] for w in e["warnings"]], e["trend"]["weekly_status"],
                         e["fvg"]["status"], len(e["zones_now"]), e["trend"]["hh_hl"])
        assert sig(a) == sig(b), f"look-ahead leak at bar {i}"


def test_pivot_is_only_used_after_confirmation(cfg):
    rows = [(100, 100.5, 99.5, 100)] * 70
    df = from_rows(rows)
    h = df["High"].values.copy()
    j = 50
    h[j] = 107.77                                        # unique sharp peak
    df["High"] = h
    prep = se.prepare(df, cfg)
    right = cfg["levels"]["pivot_right"]
    assert prep.ph[j]
    before = prep.zones(j + right - 1)                   # pivot not yet confirmed (needs `right` later bars)
    after = prep.zones(j + right)
    peak_kinds = lambda zs: [k for z in zs if abs(z["high"] - 107.77) < 1e-6 for k in z["kinds"]]  # noqa: E731
    assert "pivot_high" not in peak_kinds(before)        # the peak is already the 20-day high, but not yet a KNOWN pivot
    assert "pivot_high" in peak_kinds(after)


def test_breakout_retest_and_failed_breakout(cfg):
    base = breakout_ohlcv("confirmed", seed=8)
    R = base.attrs["resistance"]
    pull = [(R * 1.025, R * 1.03, R * 1.01, R * 1.015), (R * 1.012, R * 1.015, R * 1.0, R * 1.006)]
    retest = append_bars(base, pull + [(R * 1.002, R * 1.022, R * 0.998, R * 1.02)])
    _, ev = ev_last(retest, cfg)
    assert ev["breakout_status"] == "BREAKOUT_RETEST"
    assert any(s["code"] == "B" and s["confirmed"] for s in ev["setups"])
    failed = append_bars(base, [(R * 1.025, R * 1.03, R * 1.01, R * 1.015), (R * 1.0, R * 1.01, R * 0.95, R * 0.965)])
    _, ev2 = ev_last(failed, cfg)
    assert ev2["breakout_status"] == "FAILED_BREAKOUT" and ev2["invalidated"]
    assert any(w["name"] == "failed_breakout" for w in ev2["warnings"])


def fvg_frame(extra=()):
    rows = [(100.0, 100.5, 99.5, 100.0)] * 60
    rows += [(100.0, 100.5, 99.5, 100.2),             # k0: high 100.5
             (100.3, 103.5, 100.2, 103.3),            # k1: strong displacement
             (103.4, 104.5, 102.0, 104.2)]            # k2: low 102.0 > k0 high 100.5 -> bullish FVG [100.5, 102.0]
    rows += list(extra)
    df = from_rows(rows)
    v = df["Volume"].values.copy()
    v[60 + 1] = 3_000_000
    df["Volume"] = v
    return df


def test_bullish_fvg_boundaries_and_states(cfg):
    prep = se.prepare(fvg_frame(), cfg)
    f = lv.fvg_scan(prep, len(prep.df) - 1)["bullish"]
    assert f["low"] == pytest.approx(100.5) and f["high"] == pytest.approx(102.0) and f["size"] == pytest.approx(1.5)
    assert f["pct"] == pytest.approx(1.5 / 104.2 * 100, rel=1e-3) and f["state"] == "FRESH" and f["strong_displacement"]
    part = se.prepare(fvg_frame([(104.0, 104.3, 101.8, 103.5)]), cfg)
    assert lv.fvg_scan(part, len(part.df) - 1)["bullish"]["state"] == "PARTIALLY_MITIGATED"
    filled = se.prepare(fvg_frame([(104.0, 104.3, 100.3, 102.0)]), cfg)
    assert lv.fvg_scan(filled, len(filled.df) - 1)["bullish"] is None            # fully mitigated gaps are dropped


def test_fvg_continuation_requires_retrace_and_rejection(cfg):
    ok = se.prepare(fvg_frame([(104.0, 104.3, 103.0, 103.9), (102.0, 103.2, 101.4, 102.9)]), cfg)
    assert lv.fvg_scan(ok, len(ok.df) - 1)["continuation"]
    no_touch = se.prepare(fvg_frame([(104.0, 104.5, 103.5, 104.3), (104.3, 105.0, 103.9, 104.8)]), cfg)
    assert not lv.fvg_scan(no_touch, len(no_touch.df) - 1)["continuation"]


def test_bearish_fvg_detected(cfg):
    rows = [(100.0, 100.5, 99.5, 100.0)] * 60 + [(100.0, 100.5, 99.5, 99.8), (99.7, 99.8, 96.0, 96.2), (96.0, 97.0, 95.0, 95.5)]
    prep = se.prepare(from_rows(rows), cfg)
    b = lv.fvg_scan(prep, len(prep.df) - 1)["bearish"]
    assert b and b["high"] == pytest.approx(99.5) and b["low"] == pytest.approx(97.0)


def test_weekly_mapping_uses_completed_weeks_only(cfg):
    df = make_ohlcv(3, 400, end="2026-11-13")                     # ends on a Friday
    wk, wmap = lv.weekly_features(df)
    friday = df.index[-1]
    assert friday.weekday() == 4
    assert wmap["w_close"].iloc[-1] == pytest.approx(df["Close"].iloc[-1])             # Friday sees its own completed week
    monday = friday + pd.Timedelta(days=3)
    df2 = append_bars(df, [(df["Close"].iloc[-1],) * 4])
    wk2, wmap2 = lv.weekly_features(df2)
    assert df2.index[-1].weekday() == 0
    assert wmap2["w_close"].iloc[-1] == pytest.approx(df["Close"].iloc[-1])             # Monday still sees LAST Friday
    wed = append_bars(df, [(df["Close"].iloc[-1],) * 4] * 3)
    _, wmap3 = lv.weekly_features(wed)
    assert wmap3["w_close"].iloc[-1] == pytest.approx(df["Close"].iloc[-1])             # mid-week values never leak in


def test_flag_breakout_detected_with_pole_and_tight_flag(cfg):
    rows = [(100.0, 100.5, 99.5, 100.0)] * 60
    px = 100.0
    for _ in range(5):                                            # pole: +2% per day
        rows.append((px, px * 1.022, px * 0.999, px * 1.02))
        px *= 1.02
    for k in range(5):                                            # flag: tight, slightly down
        rows.append((px, px * 1.002, px * 0.992, px * (0.998 if k % 2 else 1.0)))
    top = max(r[1] for r in rows[-5:])
    rows.append((px * 0.999, top * 1.03, px * 0.998, top * 1.028))
    df = from_rows(rows)
    v = df["Volume"].values.copy()
    v[-1] = 3_000_000
    df["Volume"] = v
    prep = se.prepare(df, cfg)
    fl = lv.flag_breakout(prep, len(df) - 1)
    assert fl and fl["pole_gain_pct"] >= cfg["flag"]["pole_min_pct"] and fl["retrace"] <= cfg["flag"]["max_retrace"]


def test_breakout_candle_quality_gate_applies_to_chart_patterns(cfg):
    df = breakout_ohlcv("weak_close", seed=2)
    prep, ev = ev_last(df, cfg)
    assert not lv.strong_close(prep, len(df) - 1)
    assert ev["flag"] is None and ev["triangle"] is None and ev["cup"] is None and ev["consolidation"] is None


def test_zones_are_zones_not_exact_prices(cfg):
    df = breakout_ohlcv("near", seed=2)
    prep = se.prepare(df, cfg)
    zs = prep.zones(len(df) - 2)
    assert zs and all(z["tol"] > 0 for z in zs)
    res = [z for z in zs if abs(z["high"] - df.attrs["resistance"]) < 1e-6][0]
    assert res["touches"] >= 2 and res["rejections"] >= 2          # repeated rejection zone
