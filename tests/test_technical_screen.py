from datetime import date

import numpy as np
import pandas as pd
import pytest

from common import REPO_ROOT, load_yaml
from synth import make_ohlcv, passing_ohlcv
import technical_screen as ts

REF = date(2026, 11, 12)


@pytest.fixture()
def cfg():
    return load_yaml(REPO_ROOT / "config" / "screening.yml")["technical"]


def ev(df, cfg, series="EQ", **nse):
    return ts.evaluate_symbol("TEST", df, {"series": series, **nse} if series else None, cfg, REF, "synthetic")


def test_passing_series_is_really_passing(cfg):
    r = ev(passing_ohlcv(), cfg)
    assert r["technical_pass"] and r["shortlisted"], r["reasons_fail"]
    assert r["trend_ok"] and r["rsi_ok"] and r["macd_ok"] and r["volume_ok"] and r["momentum_ok"]
    assert 50 <= r["rsi14"] <= 70 and r["rel_volume"] >= 1.5
    assert "EMA20" in r["reasons_pass"] and r["reasons_fail"] == ""


def test_downtrend_fails_with_reason(cfg):
    r = ev(make_ohlcv(7, 300, drift=-0.002), cfg)
    assert not r["technical_pass"] and not r["trend_ok"]
    assert "NOT aligned" in r["reasons_fail"]


def test_low_volume_fails(cfg):
    r = ev(passing_ohlcv(last_vol_mult=1.0), cfg)
    assert not r["volume_ok"] and not r["shortlisted"]
    assert "Relative volume" in r["reasons_fail"]


def test_thresholds_are_configurable(cfg):
    base = ev(passing_ohlcv(), cfg)
    tight = {**cfg, "rsi": {"min": 90, "max": 95}}
    assert base["technical_pass"] and not ev(passing_ohlcv(), tight)["technical_pass"]
    easy = {**cfg, "min_relative_volume": 0.1}
    assert ev(passing_ohlcv(last_vol_mult=1.0), easy)["volume_ok"]


def test_insufficient_history(cfg):
    r = ev(passing_ohlcv(n=100), cfg)
    assert not r["history_ok"] and not r["technical_pass"] and not r["shortlisted"]
    assert "Insufficient history" in r["reasons_fail"] and "100 bars" in r["reasons_fail"]


@pytest.mark.parametrize("series,expected_ok", [("EQ", True), ("BE", False), ("BZ", False), ("SM", False)])
def test_series_gate_on_shortlist(cfg, series, expected_ok):
    r = ev(passing_ohlcv(), cfg, series=series)
    assert r["technical_pass"]                      # indicators identical
    assert r["series_ok"] is expected_ok and r["shortlisted"] is expected_ok


def test_unknown_series_cannot_be_shortlisted(cfg):
    r = ev(passing_ohlcv(), cfg, series=None)
    assert r["series"] == "UNKNOWN" and r["technical_pass"] and not r["shortlisted"]
    assert "Series unknown" in r["reasons_fail"]


def test_stale_data_flagged_and_blocked(cfg):
    stale = passing_ohlcv(end="2026-10-20")
    r = ev(stale, cfg)
    assert r["data_status"] == "STALE" and not r["shortlisted"]
    assert "stale" in r["reasons_fail"].lower()


def test_missing_data(cfg):
    for bad in (None, pd.DataFrame(), pd.DataFrame({"Close": [1, 2]})):
        r = ev(bad, cfg)
        assert r["data_status"] == "NO_DATA" and not r["shortlisted"] and "No usable price history" in r["reasons_fail"]


def test_price_mismatch_blocks_shortlist(cfg):
    df = passing_ohlcv()
    last = df["Close"].iloc[-1]
    r = ev(df, cfg, close=last * 1.2, source_date="2026-11-11")
    assert r["price_mismatch"] and not r["shortlisted"]
    ok = ev(df, cfg, close=last, source_date="2026-11-11")
    assert not ok["price_mismatch"] and ok["shortlisted"]


def test_watchlist_parsing(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("# comment\nsymbol,exchange,notes\n tcs ,NSE,x\nTCS,NSE,dup\nABC,BSE,bse only\nM&M,,blank exchange\n")
    entries, problems = ts.load_watchlist(p)
    assert [e["symbol"] for e in entries] == ["TCS", "M&M"]
    assert any("duplicate" in x for x in problems) and any("BSE" in x for x in problems)
    bad = tmp_path / "b.csv"
    bad.write_text("ticker,notes\nX,y\n")
    with pytest.raises(ValueError):
        ts.load_watchlist(bad)


def test_circuit_day_counter():
    df = make_ohlcv(9, 30, vol=0.001)
    c = df["Close"].copy()
    c.iloc[-1] = c.iloc[-2] * 1.05
    df["Close"] = c
    df["High"] = np.maximum(df["High"], df["Close"])
    up, dn = ts.count_circuit_days(df, 5.0, 10, 0.15)
    assert up == 1 and dn == 0
    assert all(np.isnan(v) for v in ts.count_circuit_days(df, float("nan"), 10, 0.15))
