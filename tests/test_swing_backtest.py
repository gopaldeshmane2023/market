import shutil

import numpy as np
import pandas as pd
import pytest

from common import REPO_ROOT, load_yaml
from helpers import make_sandbox
from synth import make_ohlcv
import swing_backtest as bt
import swing_engine as se


@pytest.fixture()
def cfg():
    return load_yaml(REPO_ROOT / "config" / "swing.yml")


def frame(seed=21, n=700, **kw):
    return make_ohlcv(seed, n, drift=kw.pop("drift", 0.001), vol=kw.pop("vol", 0.015), volume=2e6, **kw)


def test_entry_is_next_open_and_costs_are_applied(cfg):
    df = frame()
    sigs, _ = bt.collect("X", df, cfg, None, None)
    assert len(sigs) > 20
    b = cfg["backtest"]
    slip = b["slippage_pct_each_side"] / 100
    cost = b["cost_round_trip_pct"] / 100 + 2 * slip
    s = sigs[len(sigs) // 2]
    t = df.index.get_loc(s["date"])
    assert s["entry"] == pytest.approx(df["Open"].iloc[t + 1] * (1 + slip))               # next session's open, never the signal close
    assert s["ret_1"] == pytest.approx(df["Close"].iloc[t + 1] / s["entry"] - 1 - cost)
    assert s["ret_3"] == pytest.approx(df["Close"].iloc[t + 3] / s["entry"] - 1 - cost)
    assert s["mfe_3"] == pytest.approx(df["High"].iloc[t + 1:t + 4].max() / s["entry"] - 1)
    assert s["mae_3"] == pytest.approx(df["Low"].iloc[t + 1:t + 4].min() / s["entry"] - 1)
    assert s["mae_3"] <= s["mfe_3"]


def test_stop_tracking_and_stop_adjusted_return(cfg):
    df = frame(seed=22)
    sigs, _ = bt.collect("X", df, cfg, None, None)
    b = cfg["backtest"]
    slip = b["slippage_pct_each_side"] / 100
    hit = [s for s in sigs if s["stophit_5"] == 1.0][:5]
    assert hit
    for s in hit:
        t = df.index.get_loc(s["date"])
        lows = df["Low"].iloc[t + 1:t + 6].values
        assert (lows <= s["stop"]).any()
        assert s["retstop_5"] <= max(0.0, (s["stop"] * (1 - slip) / s["entry"] - 1 - b["cost_round_trip_pct"] / 100) + 1e-9) or s["retstop_5"] < 0
        assert s["retstop_5"] >= (df["Low"].iloc[t + 1:t + 6].min() * (1 - slip) / s["entry"] - 1 - b["cost_round_trip_pct"] / 100) - 1e-9
    safe = [s for s in sigs if s["stophit_5"] == 0.0][:5]
    for s in safe:
        assert s["retstop_5"] == pytest.approx(s["ret_5"])


def test_signals_do_not_depend_on_future_bars(cfg):
    df = frame(seed=23)
    cut = 520
    a, _ = bt.collect("X", df, cfg, None, None)
    garbage = df.copy()
    rng = np.random.default_rng(0)
    for col in ("Open", "High", "Low", "Close"):
        garbage.iloc[cut:, garbage.columns.get_loc(col)] = rng.uniform(20, 400, len(df) - cut)
    garbage["High"] = garbage[["Open", "High", "Low", "Close"]].max(axis=1)
    garbage["Low"] = garbage[["Open", "High", "Low", "Close"]].min(axis=1)
    b, _ = bt.collect("X", garbage, cfg, None, None)
    keyset = lambda ss: {(s["date"], s["tag"]) for s in ss if s["date"] < df.index[cut - 1]}  # noqa: E731
    assert keyset(a) == keyset(b) and len(keyset(a)) > 5


def test_stats_math_and_flags(cfg):
    rng = np.random.default_rng(1)
    n = 40
    r = rng.normal(0.004, 0.02, n)
    sel = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=n), "ret_5": r, "retstop_5": r * 0.8, "mfe_5": np.abs(r) + .01,
                        "mae_5": -np.abs(r) - .01, "stophit_5": (r < -0.02).astype(float)})
    base = pd.DataFrame({"horizon": 5, "ret": rng.normal(0.0, 0.02, 400), "date": pd.date_range("2024-01-01", periods=400)})
    st = bt.stats(sel, 5, base, 30)
    assert st["n"] == n and st["sample_flag"] == "OK"
    assert st["win_rate"] == pytest.approx((r > 0).mean() * 100)
    assert st["avg_ret"] == pytest.approx(r.mean() * 100) and st["median_ret"] == pytest.approx(np.median(r) * 100)
    assert st["profit_factor"] == pytest.approx(r[r > 0].sum() / -r[r < 0].sum())
    assert st["edge_vs_baseline"] == pytest.approx(st["avg_ret"] - base["ret"].mean() * 100)
    assert np.isfinite(st["t_stat"])
    small = bt.stats(sel.head(10), 5, base, 30)
    assert small["sample_flag"] == "INSUFFICIENT SAMPLE" and np.isnan(small["profit_factor"])
    assert bt.stats(sel.head(0), 5, base, 30)["sample_flag"] == "NONE"


def test_max_drawdown():
    assert bt.max_drawdown(np.array([0.02, 0.01, -0.03, -0.01, 0.05])) == pytest.approx(-4.0)
    assert bt.max_drawdown(np.array([0.01, 0.02])) == pytest.approx(0.0)
    assert np.isnan(bt.max_drawdown(np.array([])))


def test_variants_filter_correctly():
    sig = pd.DataFrame({"vol_ok": [True, False, True, True], "trend_ok": [True, True, False, True], "liq_ok": [True, True, True, False],
                        "sector_ok": [True, None, False, True]})
    assert len(bt.select_variant(sig, "alone")) == 4 and len(bt.select_variant(sig, "volume")) == 3
    assert len(bt.select_variant(sig, "trend")) == 3 and len(bt.select_variant(sig, "liquidity")) == 3
    assert len(bt.select_variant(sig, "sector")) == 2 and len(bt.select_variant(sig, "combined")) == 1
    assert len(bt.select_variant(sig, "combined_sector")) == 1


def test_regime_series():
    up = make_ohlcv(3, 500, drift=0.003, vol=0.004)
    reg = bt.regime_series(up, up.index)
    assert reg.iloc[-1] == "BULL" and reg.iloc[10] == "UNKNOWN"
    down = make_ohlcv(3, 500, drift=-0.003, vol=0.004)
    assert bt.regime_series(down, down.index).iloc[-1] == "BEAR"
    assert (bt.regime_series(None, up.index) == "UNKNOWN").all()


def test_in_and_out_of_sample_split_by_date(cfg):
    df = frame(seed=24, n=800)
    s, b = bt.collect("X", df, cfg, None, None)
    sigs, base = pd.DataFrame(s), b
    res = bt.evaluate(sigs, base, cfg)
    assert set(res["period"]) == {"ALL", "IN_SAMPLE", "OUT_OF_SAMPLE"}
    a = res[(res.pattern == "confirmed_breakout") & (res.variant == "alone") & (res.horizon == 5)].set_index("period")["n"]
    assert a["ALL"] == a["IN_SAMPLE"] + a["OUT_OF_SAMPLE"] and a["IN_SAMPLE"] > 0 and a["OUT_OF_SAMPLE"] > 0
    split = res.attrs["split_date"]
    assert sigs[sigs["date"] >= split].shape[0] > 0 and split > sigs["date"].min()


def test_verdict_wording(cfg):
    assert "insufficient" in bt.verdict(pd.Series({"n": 5, "t_stat": 3.0, "edge_vs_baseline": 1.0}), 30)
    assert bt.verdict(pd.Series({"n": 100, "t_stat": 3.0, "edge_vs_baseline": 1.0}), 30).startswith("positive edge")
    assert bt.verdict(pd.Series({"n": 100, "t_stat": -2.5, "edge_vs_baseline": -1.0}), 30).startswith("NEGATIVE")
    assert bt.verdict(pd.Series({"n": 100, "t_stat": 0.5, "edge_vs_baseline": 0.2}), 30).startswith("no reliable edge")


def test_end_to_end_reports_and_incomplete_flag(tmp_path):
    cfg_dir, data, rep = make_sandbox(tmp_path)
    shutil.copy(REPO_ROOT / "config" / "swing.yml", cfg_dir / "swing.yml")
    frames = {f"S{i}": frame(seed=30 + i, n=620) for i in range(3)}
    (cfg_dir / "watchlist.csv").write_text("symbol,exchange,notes\n" + "\n".join(f"{s},NSE," for s in frames) + "\n")
    rc = bt.main(["--config-dir", str(cfg_dir), "--data-dir", str(data), "--reports-dir", str(rep)],
                 provider=lambda s, p: (frames[s], "synthetic"), index_provider=lambda t, p: (frame(seed=99, n=620), "x"))
    assert rc == 0
    for f in ("swing_backtest.md", "swing_backtest.csv", "swing_backtest_trades.csv", "swing_backtest.json"):
        assert (rep / f).exists()
    md = (rep / "swing_backtest.md").read_text()
    assert "VALIDATION INCOMPLETE" in md and "Survivorship" in md and "**not tuned**" in md
    assert "INSUFFICIENT SAMPLE" in (rep / "swing_backtest.csv").read_text()


def test_no_history_fails_visibly(tmp_path, capsys):
    cfg_dir, data, rep = make_sandbox(tmp_path)
    shutil.copy(REPO_ROOT / "config" / "swing.yml", cfg_dir / "swing.yml")
    (cfg_dir / "watchlist.csv").write_text("symbol,exchange,notes\nAAA,NSE,\n")
    rc = bt.main(["--config-dir", str(cfg_dir), "--data-dir", str(data), "--reports-dir", str(rep)],
                 provider=lambda s, p: (None, "x"), index_provider=lambda t, p: (None, "x"))
    assert rc == 1 and "::error::" in capsys.readouterr().out
