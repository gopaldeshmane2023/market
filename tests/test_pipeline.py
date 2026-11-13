"""End-to-end: NSE files -> technical -> risk -> final report, entirely offline with synthetic data."""
import json
from datetime import date

import pandas as pd
import pytest

import final_report
import nse_data
import risk_filter
import technical_screen as ts
from helpers import make_sandbox, sec_line, SEC_HEADER
from synth import make_ohlcv, passing_ohlcv


@pytest.fixture(autouse=True)
def fixed_today(monkeypatch):
    monkeypatch.setattr(ts, "today_ist", lambda: date(2026, 11, 12))
    monkeypatch.setattr(nse_data, "today_ist", lambda: date(2026, 11, 12))


def provider_for(frames):
    def provider(symbol, period):
        if symbol not in frames:
            raise RuntimeError("symbol not on Yahoo")
        return frames[symbol], "synthetic"
    return provider


def build(tmp_path, with_nse=True):
    cfg, data, rep = make_sandbox(tmp_path)
    good = passing_ohlcv()
    frames = {"GOOD": good, "BESYM": passing_ohlcv(), "THIN": passing_ohlcv(volume=2_000),
              "DOWN": make_ohlcv(7, 300, drift=-0.002)}
    (cfg / "watchlist.csv").write_text("symbol,exchange,notes\nGOOD,NSE,mine\nBESYM,NSE,\nTHIN,NSE,\nDOWN,NSE,\nGHOST,NSE,no data\nBSEONLY,BSE,\n")
    if with_nse:
        rows = []
        for sym, series in (("GOOD", "EQ"), ("BESYM", "BE"), ("THIN", "EQ"), ("DOWN", "EQ"), ("GHOST", "EQ")):
            c = round(float(frames.get(sym, good)["Close"].iloc[-1]), 2)
            rows.append(sec_line(sym, series, "11-Nov-2026", c, c, c * 1.01, c * 0.99, c, 1_000_000))
        (data / "sec_bhavdata_full_11112026.csv").write_text(SEC_HEADER + "\n" + "\n".join(rows) + "\n")
    base = ["--config-dir", str(cfg), "--data-dir", str(data), "--reports-dir", str(rep)]
    return base, rep, frames


def run_all(base, frames):
    assert nse_data.main(base + ["--optional"]) == 0
    assert ts.main(base, provider=provider_for(frames)) == 0
    assert risk_filter.main(base) == 0
    assert final_report.main(base) == 0


def test_full_pipeline_with_nse_data(tmp_path):
    base, rep, frames = build(tmp_path)
    run_all(base, frames)
    for f in ("nse_eod_processed.csv", "technical_screen_full.csv", "technical_shortlist.csv", "risk_report.csv",
              "risk_report.md", "risk_report.json", "final_shortlist.csv", "premarket_report.md", "run_status.json"):
        assert (rep / f).exists(), f
    full = pd.read_csv(rep / "technical_screen_full.csv").set_index("symbol")
    assert set(full.index) == {"GOOD", "BESYM", "THIN", "DOWN", "GHOST"}          # BSE-only symbol skipped
    assert full.loc["GOOD", "shortlisted"] and not full.loc["BESYM", "shortlisted"] and not full.loc["DOWN", "technical_pass"]
    assert full.loc["GHOST", "data_status"] == "NO_DATA"
    risk = pd.read_csv(rep / "risk_report.csv", keep_default_na=False).set_index("symbol")
    assert risk.loc["GOOD", "risk_status"] == "PASS" and risk.loc["BESYM", "risk_status"] == "REJECT"
    assert risk.loc["THIN", "risk_status"] == "REJECT" and "liquidity" in risk.loc["THIN", "reject_reasons"].lower()
    assert risk.loc["GHOST", "risk_status"] == "REJECT"
    final = pd.read_csv(rep / "final_shortlist.csv")
    assert list(final["symbol"]) == ["GOOD"]
    report = (rep / "premarket_report.md").read_text()
    assert "### GOOD" in report and "Reasons for caution" in report
    assert "stop loss" not in report.lower().replace("no entry prices, targets or stop losses", "")
    status = json.loads((rep / "run_status.json").read_text())
    assert status["nse_data"]["status"] == "DEGRADED" and status["technical_screen"]["status"] == "DEGRADED"   # GHOST has no data


def test_pipeline_without_nse_data_shortlists_nothing_and_says_why(tmp_path):
    base, rep, frames = build(tmp_path, with_nse=False)
    assert nse_data.main(base + ["--optional"]) == 0
    status = json.loads((rep / "run_status.json").read_text())
    assert status["nse_data"]["status"] == "SKIPPED"
    assert ts.main(base, provider=provider_for(frames)) == 0
    assert risk_filter.main(base) == 0
    assert final_report.main(base) == 0
    full = pd.read_csv(rep / "technical_screen_full.csv")
    assert not full["shortlisted"].any() and (full.loc[full.data_status == "OK", "series"] == "UNKNOWN").all()
    assert (pd.read_csv(rep / "risk_report.csv")["risk_status"] == "REJECT").all()
    assert "No NSE series data was available" in (rep / "premarket_report.md").read_text()


def test_universe_nse_requires_nse_data(tmp_path):
    base, rep, frames = build(tmp_path, with_nse=False)
    assert ts.main(base + ["--universe", "nse"], provider=provider_for(frames)) == 1


def test_universe_nse_scans_liquid_eq_only(tmp_path):
    base, rep, frames = build(tmp_path)
    nse_data.main(base)
    assert ts.main(base + ["--universe", "nse"], provider=provider_for(frames)) == 0
    full = pd.read_csv(rep / "technical_screen_full.csv")
    assert "BESYM" not in set(full["symbol"])                                    # BE series never enters NSE universe


def test_risk_filter_fails_visibly_without_technical_output(tmp_path, capsys):
    cfg, data, rep = make_sandbox(tmp_path)
    assert risk_filter.main(["--config-dir", str(cfg), "--data-dir", str(data), "--reports-dir", str(rep)]) == 1
    assert "::error::" in capsys.readouterr().out


def test_empty_watchlist_skips_cleanly(tmp_path):
    base, rep, frames = build(tmp_path, with_nse=False)
    (tmp_path / "config" / "watchlist.csv").write_text("symbol,exchange,notes\n")
    assert ts.main(base + ["--optional"], provider=provider_for(frames)) == 0
    assert json.loads((rep / "run_status.json").read_text())["technical_screen"]["status"] == "SKIPPED"
