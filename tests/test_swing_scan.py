import json
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from common import IST, REPO_ROOT, load_yaml
from helpers import SEC_HEADER, make_sandbox, sec_line
from synth import append_bars, breakout_ohlcv, make_ohlcv
import nse_data
import swing_engine as se
import swing_scan as sc


@pytest.fixture()
def cfg():
    return load_yaml(REPO_ROOT / "config" / "swing.yml")


def ist(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=IST)


# ------------------------------------------------------------------ completeness
def test_forming_bar_is_excluded_until_session_closed(cfg):
    df = make_ohlcv(1, 50, end="2026-11-11")
    d, forming, note = sc.split_complete(df, ist(2026, 11, 11, 11, 0), cfg)                  # mid-session
    assert forming is not None and d.index[-1].date().isoformat() == "2026-11-10" and "forming" in note
    d, forming, _ = sc.split_complete(df, ist(2026, 11, 11, 15, 40), cfg)                    # closed but inside the buffer
    assert forming is not None
    d, forming, _ = sc.split_complete(df, ist(2026, 11, 11, 16, 5), cfg)                     # after close + buffer
    assert forming is None and d.index[-1].date().isoformat() == "2026-11-11"
    d, forming, _ = sc.split_complete(df, ist(2026, 11, 12, 7, 0), cfg)                      # next morning
    assert forming is None and len(d) == len(df)


# ------------------------------------------------------------------ trade plan
def plan_for(variant, cfg, adv_cr, seed=1):
    df = breakout_ohlcv(variant, seed=seed)
    prep = se.prepare(df, cfg)
    ev = se.evaluate_bar(prep, len(df) - 1)
    return ev, se.trade_plan(ev, cfg, 800000, adv_cr)


def test_trade_plan_math(cfg):
    ev, p = plan_for("confirmed", cfg, adv_cr=100.0)
    s = ev["primary"]
    assert p["entry_ref"] == pytest.approx(ev["close"]) and p["entry_basis"].startswith("last close")
    assert p["invalidation"] == pytest.approx(s["structure_low"] - cfg["trade_plan"]["stop_buffer_atr"] * ev["atr"])
    rps = p["entry_ref"] - p["invalidation"]
    assert p["risk_per_share"] == pytest.approx(rps)
    assert p["qty"] == int(min(800000 * 0.0025 // rps, 800000 * 0.25 // p["entry_ref"]))
    assert p["risk_amount"] <= 800000 * 0.0025 + 1e-6 and p["exposure"] == pytest.approx(p["qty"] * p["entry_ref"])


def test_participation_cap_reduces_size(cfg):
    _, p = plan_for("confirmed", cfg, adv_cr=0.5)                    # Rs0.5 crore ADV -> 1% = Rs50,000 per order
    assert p["exposure"] <= 50_000 + p["entry_ref"]


def test_unconfirmed_setup_references_trigger_not_close(cfg):
    ev, p = plan_for("near", cfg, adv_cr=100.0)
    assert ev["primary"]["confirmed"] is False
    assert p["entry_ref"] == pytest.approx(max(ev["primary"]["trigger"], ev["close"])) and "NOT confirmed" in p["entry_basis"]


def test_resistance_too_close_and_missing_invalidation(cfg):
    ev, _ = plan_for("confirmed", cfg, adv_cr=100.0)
    close = ev["close"]
    ev["zones_now"] = [{"low": close * 1.004, "high": close * 1.006, "mid": close * 1.005, "strength": 2.0, "tol": 0.1}]
    p = se.trade_plan(ev, cfg, 800000, 100.0)
    assert p["rr"] is not None and p["rr"] < cfg["trade_plan"]["min_rr"] and any("too close" in x for x in p["issues"])
    ev["primary"] = dict(ev["primary"], structure_low=close * 1.01)
    assert any("invalidation" in x for x in se.trade_plan(ev, cfg, 800000, 100.0)["issues"])
    ev["primary"] = dict(ev["primary"], structure_low=close * 0.7)
    assert any("no reasonable invalidation" in x for x in se.trade_plan(ev, cfg, 800000, 100.0)["issues"])


# ------------------------------------------------------------------ grading
def test_grade_caps_and_gates(cfg):
    conf = {"confirmed": True, "quality": "STRONG"}
    unconf = {"confirmed": False, "quality": None}
    early = {"confirmed": True, "quality": "EARLY_REVERSAL"}
    ok_plan, no_room = {"rr": 2.5}, {"rr": None}
    assert sc.grade(90, conf, ok_plan, cfg, [], 50) == "A+"
    assert sc.grade(90, conf, no_room, cfg, [], 50) == "A"                     # open air: reward not estimable => cannot be A+
    assert sc.grade(90, unconf, ok_plan, cfg, [], 50) == "B"                   # unconfirmed never above B
    assert sc.grade(90, early, ok_plan, cfg, [], 50) == "B"                    # early reversal is classified separately
    assert sc.grade(90, conf, ok_plan, cfg, [{"severity": "MEDIUM"}], 50) == "A"
    assert sc.grade(90, conf, ok_plan, cfg, [], 10) == "A"                     # weak delivery blocks A+
    assert sc.grade(20, conf, ok_plan, cfg, [], 50) == "Rejected"
    assert sc.grade(40, conf, ok_plan, cfg, [], 50) == "C"


# ------------------------------------------------------------------ end to end
def build_env(tmp_path, with_global=False):
    cfg_dir, data, rep = make_sandbox(tmp_path)
    shutil.copy(REPO_ROOT / "config" / "swing.yml", cfg_dir / "swing.yml")
    frames = {"BRK": breakout_ohlcv("confirmed", seed=1), "NEAR": breakout_ohlcv("near", seed=2), "WICK": breakout_ohlcv("wick_reject", seed=3),
              "BESYM": breakout_ohlcv("confirmed", seed=4), "THIN": breakout_ohlcv("confirmed", seed=5, volume=3000)}
    (cfg_dir / "watchlist.csv").write_text("symbol,exchange,notes\n" + "\n".join(f"{s},NSE," for s in list(frames) + ["GHOST"]) + "\n")
    rows = []
    for s, series in (("BRK", "EQ"), ("NEAR", "EQ"), ("WICK", "EQ"), ("BESYM", "BE"), ("THIN", "EQ"), ("GHOST", "EQ")):
        d = frames.get(s, frames["BRK"])
        c = round(float(d.Close.iloc[-1]), 2)
        rows.append(sec_line(s, series, "11-Nov-2026", c, c, c * 1.01, c * .99, c, int(d.Volume.iloc[-1])))
    (data / "sec_bhavdata_full_11112026.csv").write_text(SEC_HEADER + "\n" + "\n".join(rows) + "\n")
    nse_data.today_ist = lambda: datetime(2026, 11, 12).date()
    base = ["--config-dir", str(cfg_dir), "--data-dir", str(data), "--reports-dir", str(rep)]
    assert nse_data.main(base) == 0

    def provider(sym, period):
        if sym not in frames:
            raise RuntimeError("no data")
        return frames[sym], "synthetic"
    return base, rep, frames, provider, cfg_dir


def run_scan(tmp_path, now=None, index_provider=None, **kw):
    base, rep, frames, provider, cfg_dir = build_env(tmp_path, **kw)
    rc = sc.main(base, provider=provider, index_provider=index_provider or (lambda t, p: (None, "x")), now_ist=now or ist(2026, 11, 12, 19, 0))
    assert rc == 0
    cands = pd.read_csv(rep / "swing_candidates.csv", keep_default_na=False)
    rej = pd.read_csv(rep / "swing_rejected.csv", keep_default_na=False)
    return rep, cands.set_index("symbol"), rej.set_index("symbol"), cfg_dir


def test_scan_classifies_and_gates(tmp_path):
    rep, cands, rej, _ = run_scan(tmp_path)
    assert cands.loc["BRK", "grade"] in ("A+", "A", "B") and cands.loc["BRK", "breakout_status"] == "CONFIRMED_BREAKOUT"
    assert cands.loc["BRK", "setup_confirmed"] in (True, "True") and cands.loc["BRK", "invalidation"] < cands.loc["BRK", "entry_ref"]
    # a stock NEAR resistance is never presented as a confirmed breakout
    near = cands.loc["NEAR"] if "NEAR" in cands.index else rej.loc["NEAR"]
    assert near["breakout_status"] == "POTENTIAL_BREAKOUT"
    if "NEAR" in cands.index:
        assert cands.loc["NEAR", "grade"] in ("B", "C") and cands.loc["NEAR", "setup_confirmed"] in (False, "False")
    assert "BE is trade-to-trade" in rej.loc["BESYM", "reasons"]
    assert "Low liquidity" in rej.loc["THIN", "reasons"]
    assert "No usable price history" in rej.loc["GHOST", "reasons"]
    assert "invalidated" in rej.loc["WICK", "reasons"].lower() and rej.loc["WICK", "breakout_status"] == "RESISTANCE_REJECTION"
    for f in ("swing_report.md", "swing_report.json", "swing_candidates.csv", "swing_rejected.csv"):
        assert (rep / f).exists()
    md = (rep / "swing_report.md").read_text()
    assert "Liquidity primer" in md and "order-book depth" in md and "Why it could fail" in md and "Trade plan" in md
    assert "not targets" in md


def test_unknown_sector_and_global_are_reported_not_invented(tmp_path):
    rep, cands, _, _ = run_scan(tmp_path)
    assert "sector" in cands.loc["BRK", "unverified_factors"] and "global" in cands.loc["BRK", "unverified_factors"]
    js = json.loads((rep / "swing_report.json").read_text())
    brk = [c for c in js["candidates"] if c["symbol"] == "BRK"][0]
    assert brk["score_components"]["sector"]["value"] is None and brk["score_components"]["global"]["value"] is None
    assert "UNKNOWN" in (rep / "swing_report.md").read_text()


def test_forming_candle_is_only_pending_never_confirmed(tmp_path):
    # run while the session is still open: today's (breakout) candle is excluded and can only be BREAKOUT_PENDING_CLOSE
    rep, cands, rej, _ = run_scan(tmp_path, now=ist(2026, 11, 11, 11, 30))
    allrows = pd.concat([cands, rej])
    assert allrows.loc["BRK", "breakout_status"] == "BREAKOUT_PENDING_CLOSE"
    assert allrows.loc["BRK", "data_date"] == "2026-11-10"
    if "BRK" in cands.index:
        assert not (str(cands.loc["BRK", "setup"]).startswith("A:") and cands.loc["BRK", "setup_confirmed"] in (True, "True"))
    assert "forming" in (rep / "swing_report.md").read_text().lower()


def test_sector_and_global_context_used_when_available(tmp_path):
    base, rep, frames, provider, cfg_dir = build_env(tmp_path)
    (cfg_dir / "sector_map.csv").write_text("symbol,company,industry,sector,macro_sector,sector_index\nBRK,Brk Ltd,Widgets,IT,IT,^TESTIT\n")
    gj = {"generated_at_ist": "2026-11-12 07:00", "score": {"score": 40.0, "assessment": "POSITIVE", "coverage_pct": 100},
          "factors": [], "macro_risks": ["Brent crude rose 3%+"], "sectors": [{"sector": "IT", "tilt": 30.0, "label": "TAILWIND", "ambiguous": False}]}
    (rep / "global_report.json").write_text(json.dumps(gj))
    up = make_ohlcv(5, 400, drift=0.002, vol=0.005)
    rc = sc.main(base, provider=provider, index_provider=lambda t, p: (up, "x"), now_ist=ist(2026, 11, 12, 19, 0))
    assert rc == 0
    js = json.loads((rep / "swing_report.json").read_text())
    brk = [c for c in js["candidates"] if c["symbol"] == "BRK"][0]
    assert brk["score_components"]["sector"]["value"] is not None and brk["score_components"]["global"]["value"] is not None
    assert brk["sector"] == "IT" and brk["sector_status"] == "UP" and "tilt" in brk["global_alignment"]
    assert "Macro headwinds" in (rep / "swing_report.md").read_text()


def test_missing_nse_data_rejects_everything_with_reason(tmp_path):
    base, rep, frames, provider, cfg_dir = build_env(tmp_path)
    (rep / "nse_eod_processed.csv").unlink()
    assert sc.main(base, provider=provider, index_provider=lambda t, p: (None, "x"), now_ist=ist(2026, 11, 12, 19, 0)) == 0
    cands = pd.read_csv(rep / "swing_candidates.csv")
    rej = pd.read_csv(rep / "swing_rejected.csv", keep_default_na=False)
    assert cands.empty and rej["reasons"].str.contains("Series unknown").sum() >= 4
    assert "No NSE EOD file" in (rep / "swing_report.md").read_text()


def test_participation_and_max_order_reported(tmp_path):
    _, cands, _, _ = run_scan(tmp_path)
    r = cands.loc["BRK"]
    assert float(r["adv20_cr"]) > 20 and float(r["participation_8L_pct"]) < 1.0
    assert float(r["max_order_value"]) == pytest.approx(float(r["adv20_cr"]) * 1e7 * 0.01, rel=0.02)
