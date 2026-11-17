import numpy as np
import pandas as pd
import pytest

from common import REPO_ROOT, load_yaml
from synth import from_rows, make_ohlcv, trend_rows
import swing_patterns as sp


@pytest.fixture()
def cfg():
    return load_yaml(REPO_ROOT / "config" / "swing.yml")


def flags(rows, cfg, up=False):
    base = trend_rows(60, 70, 100, ) if up else trend_rows(60, 100, 70)
    df = from_rows(base + rows)
    return sp.detect_patterns(df, cfg["patterns"]).iloc[-1]


def test_hammer_requires_the_definition(cfg):
    hammer = flags([(69.0, 69.45, 67.2, 69.4)], cfg)
    assert hammer["pat_hammer"] and not hammer["pat_hanging_man"]
    assert hammer["lw_body"] >= cfg["patterns"]["hammer_lower_body_min"] and hammer["close_pos"] >= 0.6


def test_tiny_body_with_long_lower_wick_is_dragonfly_not_hammer(cfg):
    r = flags([(69.0, 69.05, 67.2, 69.02)], cfg)
    assert not r["pat_hammer"] and r["pat_dragonfly_doji"]


def test_small_candle_is_not_labelled_a_hammer(cfg):
    r = flags([(69.0, 69.12, 68.7, 69.1)], cfg)                      # lower wick > 2x body but range far below 0.5 ATR
    assert not r["pat_hammer"] and not r["pat_dragonfly_doji"]


def test_same_shape_after_uptrend_is_hanging_man(cfg):
    r = flags([(99.0, 99.45, 97.2, 99.4)], cfg, up=True)
    assert r["pat_hanging_man"] and not r["pat_hammer"]


def test_bullish_and_bearish_engulfing(cfg):
    bull = flags([(70.0, 70.1, 69.1, 69.2), (69.0, 70.7, 68.9, 70.6)], cfg)
    assert bull["pat_bullish_engulfing"] and not bull["pat_bearish_engulfing"]
    bear = flags([(100.0, 100.9, 99.95, 100.8), (101.0, 101.1, 99.6, 99.7)], cfg, up=True)
    assert bear["pat_bearish_engulfing"] and not bear["pat_bullish_engulfing"]


def test_morning_star(cfg):
    r = flags([(70.5, 70.6, 68.4, 68.5), (68.3, 68.6, 68.0, 68.4), (68.6, 70.0, 68.5, 69.8)], cfg)
    assert r["pat_morning_star"]


def test_marubozu_and_shooting_star(cfg):
    assert flags([(70.0, 72.02, 69.98, 72.0)], cfg)["pat_bullish_marubozu"]
    ss = flags([(100.0, 102.4, 99.95, 100.5)], cfg, up=True)
    assert ss["pat_shooting_star"] and not ss["pat_inverted_hammer"]


def test_three_white_soldiers(cfg):
    rows = [(70.0, 71.1, 69.9, 71.0), (70.5, 72.1, 70.4, 72.0), (71.5, 73.2, 71.4, 73.1)]
    assert flags(rows, cfg)["pat_three_white_soldiers"]


def test_long_upper_wick_rejection(cfg):
    r = flags([(100.0, 103.0, 99.9, 100.4)], cfg, up=True)
    assert r["pat_long_upper_wick_rejection"]


def test_inside_bar_breakout(cfg):
    rows = [(100.0, 102.0, 98.0, 101.0), (100.5, 101.5, 99.0, 100.8), (100.9, 103.5, 100.8, 103.2)]
    assert flags(rows, cfg, up=True)["pat_inside_bar_breakout"]


def test_patterns_are_causal_no_lookahead(cfg):
    df = make_ohlcv(11, 300, vol=0.02)
    full = sp.detect_patterns(df, cfg["patterns"])
    cols = [c for c in full.columns if c.startswith(("pat_", "geo_")) or c in ("atr", "rsi", "rel_vol", "ema20")]
    for cut in (120, 200, 260, 299):
        part = sp.detect_patterns(df.iloc[:cut + 1], cfg["patterns"])
        a, b = part.iloc[-1][cols].astype(float).values, full.iloc[cut][cols].astype(float).values
        assert np.allclose(a, b, equal_nan=True), f"look-ahead at bar {cut}"


def test_explanation_contains_formula_numbers(cfg):
    df = from_rows(trend_rows(60, 100, 70) + [(69.0, 69.45, 67.2, 69.4)])
    m = sp.detect_patterns(df, cfg["patterns"])
    text = sp.describe_pattern("hammer", m, len(m) - 1)
    assert "lower wick >= 2x body" in text and "body/range" in text and "relative volume" in text
