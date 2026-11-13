import copy
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from common import REPO_ROOT, load_yaml
import global_report as g

NOW = datetime(2026, 11, 12, 1, 30, tzinfo=timezone.utc)


@pytest.fixture()
def cfg():
    c = load_yaml(REPO_ROOT / "config" / "market_impact.yml")
    g.validate_config(c)
    return c


def frame(latest, prev, date="2026-11-11"):
    idx = pd.to_datetime(["2026-11-10" if date == "2026-11-11" else "2026-10-01", date])
    return pd.DataFrame({"Close": [prev, latest]}, index=idx)


def build(cfg, moves: dict, default=0.0, stale=(), drop=()):
    """moves: factor key -> pct move (for ^TNX: bp)."""
    def fetch(ticker, period):
        key = next(k for k, v in cfg["factors"].items() if v["ticker"] == ticker)
        if key in drop:
            return pd.DataFrame()
        m = moves.get(key, default)
        if key == "us10y":
            return frame(4.25 + m / 100, 4.25, "2026-10-01" if key in stale else "2026-11-11")
        return frame(100 * (1 + m / 100), 100, "2026-10-01" if key in stale else "2026-11-11")
    return g.build_report(cfg, fetch, NOW)


def test_repo_config_is_valid_and_has_all_required_factors(cfg):
    tickers = {v["ticker"] for v in cfg["factors"].values()}
    assert tickers == {"^GSPC", "^IXIC", "^DJI", "^VIX", "DX-Y.NYB", "BZ=F", "GC=F", "SI=F", "HG=F", "INR=X", "^TNX"}
    w = {k: v["weight"] for k, v in cfg["factors"].items()}
    assert (w["sp500"], w["nasdaq"], w["dow"], w["vix"], w["dxy"], w["brent"], w["gold"], w["copper"], w["usdinr"],
            w["us10y"]) == (1.0, 1.0, 0.5, 1.0, 1.0, 1.5, 0.25, 0.5, 1.5, 1.0)


def test_bad_config_rejected(cfg):
    bad = copy.deepcopy(cfg)
    bad["factors"]["vix"]["direction"] = 2
    bad["factors"]["brent"]["weight"] = -1
    bad["sectors"]["IT"]["drivers"]["nope"] = 1
    with pytest.raises(ValueError) as e:
        g.validate_config(bad)
    assert "direction" in str(e.value) and "weight" in str(e.value) and "nope" in str(e.value)


@pytest.mark.parametrize("move", [-90, -5, 0, 5, 1000])
def test_score_always_bounded(cfg, move):
    for m in (move, -move if abs(move) < 100 else -90):
        rep = build(cfg, {k: m for k in cfg["factors"]})
        s = rep["score"]["score"]
        assert -100 <= s <= 100


def test_extremes_hit_exactly_plus_and_minus_100(cfg):
    best = {k: (1000 if v["direction"] > 0 else -90) for k, v in cfg["factors"].items()}
    worst = {k: (-90 if v["direction"] > 0 else 1000) for k, v in cfg["factors"].items()}
    assert build(cfg, best)["score"]["score"] == 100.0
    assert build(cfg, worst)["score"]["score"] == -100.0


def test_direction_semantics(cfg):
    vix_up = build(cfg, {"vix": 20})
    f = next(x for x in vix_up["factors"] if x["key"] == "vix")
    assert f["contribution_points"] < 0 and vix_up["score"]["score"] < 0
    spx_up = build(cfg, {"sp500": 2})
    assert spx_up["score"]["score"] > 0
    brent_up = build(cfg, {"brent": 3})
    assert brent_up["score"]["score"] < 0
    inr_up = build(cfg, {"usdinr": 0.5})
    assert inr_up["score"]["score"] < 0


def test_contributions_sum_to_score(cfg):
    rep = build(cfg, {"sp500": 1.0, "vix": -4, "brent": 1.2, "usdinr": 0.2, "us10y": 7})
    total = sum(f["contribution_points"] for f in rep["factors"] if f["included_in_score"])
    assert total == pytest.approx(rep["score"]["score"], abs=0.06)


def test_silver_tracked_but_not_scored(cfg):
    rep = build(cfg, {"silver": 50})
    silver = next(f for f in rep["factors"] if f["key"] == "silver")
    assert not silver["included_in_score"] and silver["exclusion_reason"].startswith("weight 0")
    assert rep["score"]["score"] == 0.0


def test_missing_data_renormalises_and_is_reported(cfg):
    rep = build(cfg, {"sp500": 2.0}, drop=["dow", "copper"])
    assert rep["score"]["score"] is not None
    missing = [f for f in rep["factors"] if f["status"] == "MISSING"]
    assert {f["key"] for f in missing} == {"dow", "copper"} and all(not f["included_in_score"] for f in missing)
    assert rep["score"]["coverage_pct"] < 100


def test_insufficient_coverage_gives_no_score(cfg):
    drop = ["sp500", "nasdaq", "vix", "dxy", "brent"]
    rep = build(cfg, {}, drop=drop)
    assert rep["score"]["score"] is None and rep["score"]["assessment"] == "INSUFFICIENT_DATA"
    assert "not produced" in g.render_markdown(rep)


def test_stale_excluded_by_default_and_includable(cfg):
    rep = build(cfg, {"brent": 3.0}, stale=["brent"])
    brent = next(f for f in rep["factors"] if f["key"] == "brent")
    assert brent["status"] == "STALE" and brent["stale"] and not brent["included_in_score"]
    assert brent["age_days"] > cfg["settings"]["max_age_days"]
    c2 = copy.deepcopy(cfg)
    c2["settings"]["stale_policy"] = "include"
    rep2 = build(c2, {"brent": 3.0}, stale=["brent"])
    assert next(f for f in rep2["factors"] if f["key"] == "brent")["included_in_score"]


def test_single_row_and_garbage_data(cfg):
    spec = cfg["factors"]["sp500"]
    one = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-11-11"]))
    assert g.summarize_factor("sp500", spec, one, NOW, cfg["settings"])["status"] == "INSUFFICIENT_HISTORY"
    neg = pd.DataFrame({"Close": [-1.0, 5.0]}, index=pd.to_datetime(["2026-11-10", "2026-11-11"]))
    assert g.summarize_factor("sp500", spec, neg, NOW, cfg["settings"])["status"] == "INVALID"
    assert g.summarize_factor("sp500", spec, None, NOW, cfg["settings"])["status"] == "MISSING"


def tnx(cfg, latest, prev, setting="auto"):
    s = dict(cfg["settings"], tnx_convention=setting)
    return g.summarize_factor("us10y", cfg["factors"]["us10y"], frame(latest, prev), NOW, s)


def test_tnx_percent_convention_basis_points(cfg):
    f = tnx(cfg, 4.30, 4.25)
    assert f["status"] == "OK" and f["tnx_convention"] == "percent"
    assert f["latest_close"] == pytest.approx(4.30) and f["change_bp"] == pytest.approx(5.0)


def test_tnx_tenths_convention_basis_points(cfg):
    f = tnx(cfg, 43.0, 42.5)
    assert f["status"] == "OK" and f["tnx_convention"] == "tenths"
    assert f["latest_close"] == pytest.approx(4.30) and f["change_bp"] == pytest.approx(5.0)


def test_tnx_forced_convention_and_ambiguity_and_suspect(cfg):
    assert tnx(cfg, 43.0, 42.5, "percent")["status"] == "INVALID"          # 43% yield is implausible
    amb = tnx(cfg, 9.5, 9.4)
    assert any("AMBIGUOUS" in n for n in amb["notes"])
    sus = tnx(cfg, 4.95, 4.25)                                              # +70bp in a day
    assert sus["status"] == "SUSPECT" and not build(cfg, {"us10y": 70})["factors"][-1]["included_in_score"]


def test_sector_tilt_and_labels(cfg):
    rep = build(cfg, {"nasdaq": 2.5, "usdinr": 0.5})
    it = next(s for s in rep["sectors"] if s["sector"] == "IT")
    assert it["tilt"] > 15 and it["label"] == "TAILWIND"
    rep2 = build(cfg, {"brent": 3.0})
    auto = next(s for s in rep2["sectors"] if s["sector"] == "Auto")
    assert auto["label"] == "HEADWIND"
    names = {s["sector"] for s in rep["sectors"]}
    assert names == {"IT", "Pharma", "Auto", "Defence", "Metals", "PSU banks", "Private banks", "FMCG", "Chemicals",
                     "Oil and gas", "Realty", "Electronics", "Jewellery"}


def test_sector_without_data_says_no_data(cfg):
    rep = build(cfg, {}, drop=list(cfg["factors"]))
    assert all(s["label"] == "NO DATA" for s in rep["sectors"])


def test_risk_flags_fire_only_on_usable_data(cfg):
    rep = build(cfg, {"brent": 4.0, "vix": 12})
    assert any("Brent" in m for m in rep["macro_risks"]) and any("VIX" in m for m in rep["macro_risks"])
    rep2 = build(cfg, {"brent": 4.0}, stale=["brent"])
    assert not any("Brent" in m for m in rep2["macro_risks"])


def test_fetch_exception_is_contained(cfg):
    def fetch(ticker, period):
        if ticker == "^VIX":
            raise RuntimeError("boom")
        return frame(101, 100)
    rep = g.build_report(cfg, fetch, NOW)
    vix = next(f for f in rep["factors"] if f["key"] == "vix")
    assert vix["status"] == "FETCH_ERROR" and "boom" in vix["notes"][0]
    assert rep["score"]["score"] is not None


def test_main_writes_reports_and_status(cfg, tmp_path):
    def fetch(ticker, period):
        return frame(4.3, 4.25) if ticker == "^TNX" else frame(101, 100)
    rc = g.main(["--reports-dir", str(tmp_path)], fetcher=fetch)
    assert rc == 0
    md = (tmp_path / "global_report.md").read_text()
    assert "not** a buy signal" in md and "clip(m_i / full_move_i" in md and "Limitations" in md
    import json
    js = json.loads((tmp_path / "global_report.json").read_text())
    assert -100 <= js["score"]["score"] <= 100 and len(js["factors"]) == 11
    assert json.loads((tmp_path / "run_status.json").read_text())["global_report"]["status"] in ("OK", "DEGRADED")


def test_main_fails_visibly_when_everything_is_missing(tmp_path, capsys):
    rc = g.main(["--reports-dir", str(tmp_path)], fetcher=lambda t, p: pd.DataFrame())
    assert rc == 1
    assert "::error::" in capsys.readouterr().out
