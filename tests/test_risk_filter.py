import numpy as np
import pytest

from common import REPO_ROOT, load_yaml
from risk_filter import evaluate_risk


@pytest.fixture()
def cfg():
    return load_yaml(REPO_ROOT / "config" / "screening.yml")["risk"]


def good(**over):
    row = dict(series="EQ", master_series="EQ", data_status="OK", last_close=250.0, avg_traded_value_cr20=25.0,
               avg_vol20=900_000, delivery_pct=45.0, circuit_status="NONE", price_band_pct=20.0,
               circuit_up_days=0, circuit_down_days=0, surveillance="NONE", security_status="LISTED_IN_MASTER",
               price_mismatch=False)
    row.update(over)
    return row


def test_clean_eq_stock_passes_but_is_not_called_safe(cfg):
    r = evaluate_risk(good(), cfg)
    assert r["status"] == "PASS" and r["reject_reasons"] == [] and r["warning_reasons"] == []
    assert not any("safe" in x.lower() for x in r["reject_reasons"] + r["warning_reasons"])


@pytest.mark.parametrize("series", ["BE", "BZ", "BT", "SM", "ST", "X", "Z"])
def test_restricted_series_rejected(cfg, series):
    r = evaluate_risk(good(series=series), cfg)
    assert r["status"] == "REJECT"
    assert any(series in x for x in r["reject_reasons"])


@pytest.mark.parametrize("series", [None, "", "nan", np.nan, "UNKNOWN", "QQ"])
def test_unknown_series_rejected_until_verified(cfg, series):
    r = evaluate_risk(good(series=series), cfg)
    assert r["status"] == "REJECT"
    assert any("verified" in x for x in r["reject_reasons"])


def test_series_conflict_with_master_is_warning(cfg):
    r = evaluate_risk(good(master_series="BE"), cfg)
    assert r["status"] == "WARNING" and any("conflict" in x.lower() for x in r["warning_reasons"])


def test_liquidity_tiers(cfg):
    assert evaluate_risk(good(avg_traded_value_cr20=1.0), cfg)["status"] == "REJECT"
    thin = evaluate_risk(good(avg_traded_value_cr20=3.0), cfg)
    assert thin["status"] == "WARNING" and thin["liquidity_status"] == "THIN"
    assert evaluate_risk(good(avg_traded_value_cr20=50.0), cfg)["liquidity_status"] == "HIGH"
    low_vol = evaluate_risk(good(avg_vol20=50_000), cfg)
    assert low_vol["status"] == "REJECT" and any("volume" in x.lower() for x in low_vol["reject_reasons"])


def test_missing_liquidity_data_rejected(cfg):
    r = evaluate_risk(good(avg_traded_value_cr20=np.nan, avg_vol20=np.nan), cfg)
    assert r["status"] == "REJECT" and sum("unavailable" in x for x in r["reject_reasons"]) == 2


def test_min_price(cfg):
    r = evaluate_risk(good(last_close=12.5), cfg)
    assert r["status"] == "REJECT" and any("minimum" in x for x in r["reject_reasons"])


def test_circuits(cfg):
    assert evaluate_risk(good(circuit_status="LOWER_CIRCUIT"), cfg)["status"] == "REJECT"
    up = evaluate_risk(good(circuit_status="UPPER_CIRCUIT"), cfg)
    assert up["status"] == "WARNING"
    rep = evaluate_risk(good(circuit_up_days=2, circuit_down_days=1), cfg)
    assert rep["status"] == "WARNING" and any("Repeated" in x for x in rep["warning_reasons"])
    strict = {**cfg, "upper_circuit_action": "reject"}
    assert evaluate_risk(good(circuit_status="UPPER_CIRCUIT"), strict)["status"] == "REJECT"


def test_unassessed_circuit_is_a_note_not_a_warning(cfg):
    r = evaluate_risk(good(circuit_status="UNKNOWN", price_band_pct=np.nan, surveillance="NO_DATA"), cfg)
    assert r["status"] == "PASS" and len(r["notes"]) >= 2


def test_surveillance_configurable(cfg):
    assert evaluate_risk(good(surveillance="ASM Stage I"), cfg)["status"] == "WARNING"
    assert evaluate_risk(good(surveillance="GSM 2"), {**cfg, "surveillance_action": "reject"})["status"] == "REJECT"


def test_delivery_missing_and_low(cfg):
    miss = evaluate_risk(good(delivery_pct=np.nan), cfg)
    assert miss["status"] == "WARNING" and any("Delivery" in x for x in miss["warning_reasons"])
    low = evaluate_risk(good(delivery_pct=10.0), cfg)
    assert low["status"] == "WARNING"


def test_stale_and_no_data(cfg):
    st = evaluate_risk(good(data_status="STALE", data_date="2026-10-01", data_age_days=40), cfg)
    assert st["status"] == "WARNING" and any("Stale" in x for x in st["warning_reasons"])
    assert evaluate_risk(good(data_status="STALE"), {**cfg, "stale_action": "reject"})["status"] == "REJECT"
    nd = evaluate_risk(good(data_status="NO_DATA", last_close=np.nan, avg_traded_value_cr20=np.nan, avg_vol20=np.nan), cfg)
    assert nd["status"] == "REJECT"


def test_not_in_master_warns_and_spread(cfg):
    assert evaluate_risk(good(security_status="NOT_IN_MASTER"), cfg)["status"] == "WARNING"
    wide = evaluate_risk(good(bid=100.0, ask=103.0), cfg)
    assert wide["status"] == "WARNING" and any("spread" in x.lower() for x in wide["warning_reasons"])


def test_reject_dominates_warning(cfg):
    r = evaluate_risk(good(series="BE", delivery_pct=np.nan), cfg)
    assert r["status"] == "REJECT" and r["warning_reasons"]
