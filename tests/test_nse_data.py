from datetime import date

import numpy as np
import pandas as pd
import pytest

import nse_data as nd
from helpers import SEC_HEADER, make_sandbox, sec_line


def write_sec(path, rows, header=SEC_HEADER):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_sec_bhavdata_parsing_units_and_dash_delivery(tmp_path):
    p = tmp_path / "sec_bhavdata_full_11112026.csv"
    write_sec(p, [sec_line("ABC", "EQ", "11-Nov-2026", 100, 101, 105, 99, 104, 1_000_000),
                  sec_line("XYZ", "BE", "11-Nov-2026", 50, 50, 52, 49, 51, 200_000, "-", "-")])
    df, meta = nd.parse_price_file(p, min_rows=1)
    abc = df[df["symbol"] == "ABC"].iloc[0]
    assert meta["format"] == "sec_bhavdata_full" and meta["source_date"] == "2026-11-11" and meta["has_delivery"]
    assert abc["traded_value_cr"] == pytest.approx(104 * 1_000_000 / 1e7)       # lakh -> crore
    assert abc["delivery_pct"] == pytest.approx(45.5) and abc["series"] == "EQ"
    assert np.isnan(df[df["symbol"] == "XYZ"].iloc[0]["delivery_pct"])           # '-' is NaN, never 0


def test_udiff_and_legacy_formats(tmp_path):
    u = tmp_path / "BhavCopy_NSE_CM_0_0_0_20261111_F_0000.csv"
    u.write_text("TradDt,TckrSymb,SctySrs,OpnPric,HghPric,LwPric,ClsPric,PrvsClsgPric,TtlTradgVol,TtlTrfVal\n"
                 "2026-11-11,ABC,EQ,10,11,9,10.5,10,1000,10500000\n")
    df, meta = nd.parse_price_file(u, min_rows=1)
    assert meta["format"] == "udiff" and not meta["has_delivery"]
    assert df.iloc[0]["traded_value_cr"] == pytest.approx(1.05) and np.isnan(df.iloc[0]["delivery_pct"])
    l = tmp_path / "cm11NOV2026bhav.csv"
    l.write_text("SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP\n"
                 "ABC,EQ,10,11,9,10.5,10.5,10,1000,10500000,11-NOV-2026\n")
    df, meta = nd.parse_price_file(l, min_rows=1)
    assert meta["format"] == "legacy" and df.iloc[0]["traded_value_cr"] == pytest.approx(1.05)


def test_validation_errors_are_explicit(tmp_path):
    p = tmp_path / "sec_bhavdata_full_x.csv"
    p.write_text("SYMBOL,SERIES,CLOSE_PRICE\nA,EQ,1\n")
    with pytest.raises(nd.ValidationError) as e:
        nd.parse_price_file(p, min_rows=1)
    assert "unrecognised NSE format" in str(e.value)
    p2 = tmp_path / "e.csv"
    p2.write_text("")
    with pytest.raises(nd.ValidationError):
        nd.parse_price_file(p2, min_rows=1)
    p3 = tmp_path / "small.csv"
    write_sec(p3, [sec_line("A", "EQ", "11-Nov-2026", 1, 1, 1, 1, 1, 10)])
    with pytest.raises(nd.ValidationError) as e:
        nd.parse_price_file(p3, min_rows=500)
    assert "truncated" in str(e.value)
    p4 = tmp_path / "baddate.csv"
    write_sec(p4, [sec_line("A", "EQ", "not-a-date", 1, 1, 1, 1, 1, 10)])
    with pytest.raises(nd.ValidationError) as e:
        nd.parse_price_file(p4, min_rows=1)
    assert "date" in str(e.value).lower()


def test_quality_flags_and_duplicates(tmp_path):
    p = tmp_path / "q.csv"
    write_sec(p, [sec_line("A", "EQ", "11-Nov-2026", 10, 10, 9, 11, 10, 100),       # high < low
                  sec_line("A", "EQ", "11-Nov-2026", 10, 10, 11, 9, 10, 100),       # duplicate
                  sec_line("B", "EQ", "11-Nov-2026", 10, 10, 11, 9, 10, 100, "10", "150")])  # delivery > 100
    df, meta = nd.parse_price_file(p, min_rows=1)
    assert len(df) == 2 and any("duplicate" in w for w in meta["warnings"])
    assert "high_below_low" in df[df.symbol == "A"].iloc[0]["data_quality"]
    b = df[df.symbol == "B"].iloc[0]
    assert "bad_delivery_pct" in b["data_quality"] and np.isnan(b["delivery_pct"])


def test_multiple_dates_uses_latest_with_warning(tmp_path):
    p = tmp_path / "m.csv"
    write_sec(p, [sec_line("A", "EQ", "10-Nov-2026", 10, 10, 11, 9, 10, 100),
                  sec_line("A2", "EQ", "11-Nov-2026", 10, 10, 11, 9, 10, 100)])
    df, meta = nd.parse_price_file(p, min_rows=1)
    assert list(df["symbol"]) == ["A2"] and any("different dates" in w for w in meta["warnings"])


def test_circuit_classification():
    assert nd.classify_circuit(105, 100, 105, 99, 5, 0.15) == "UPPER_CIRCUIT"
    assert nd.classify_circuit(95, 100, 101, 95, 5, 0.15) == "LOWER_CIRCUIT"
    assert nd.classify_circuit(104, 100, 105, 99, 5, 0.15) == "NONE"
    assert nd.classify_circuit(105, 100, 105, 99, np.nan, 0.15) == "UNKNOWN"


def test_enrich_with_master_band_surveillance(tmp_path):
    cfg, data, rep = make_sandbox(tmp_path)
    write_sec(data / "sec_bhavdata_full_11112026.csv",
              [sec_line("ABC", "EQ", "11-Nov-2026", 100, 100, 105, 100, 105, 1_000_000),
               sec_line("QQQ", "EQ", "11-Nov-2026", 100, 100, 101, 99, 100, 1_000_000)])
    (data / "nse_security_master.csv").write_text("SYMBOL,NAME OF COMPANY, SERIES,DATE OF LISTING\nABC,Abc Ltd,EQ,01-JAN-2000\n")
    (data / "nse_price_band.csv").write_text("symbol,price_band_pct\nABC,5%\n")
    (data / "nse_surveillance.csv").write_text("symbol,list,stage\nQQQ,ASM,Stage 1\n")
    assert nd.main(["--config-dir", str(cfg), "--data-dir", str(data), "--reports-dir", str(rep)]) == 0
    out = pd.read_csv(rep / "nse_eod_processed.csv", keep_default_na=False).set_index("symbol")
    assert out.loc["ABC", "circuit_status"] == "UPPER_CIRCUIT" and out.loc["ABC", "security_status"] == "LISTED_IN_MASTER"
    assert out.loc["QQQ", "security_status"] == "NOT_IN_MASTER" and out.loc["QQQ", "surveillance"] == "ASM Stage 1"
    assert out.loc["ABC", "surveillance"] == "NONE"
    assert (rep / "nse_data_report.md").exists() and (rep / "nse_data_report.json").exists()


def test_missing_files_skipped_when_optional_else_fail(tmp_path, capsys):
    cfg, data, rep = make_sandbox(tmp_path)
    args = ["--config-dir", str(cfg), "--data-dir", str(data), "--reports-dir", str(rep)]
    assert nd.main(args + ["--optional"]) == 0
    assert "SKIPPED" in capsys.readouterr().out
    assert not (rep / "nse_eod_processed.csv").exists()          # a missing download is not an empty market
    assert nd.main(args) == 1


def test_invalid_file_fails_even_when_optional(tmp_path):
    cfg, data, rep = make_sandbox(tmp_path)
    (data / "sec_bhavdata_full_bad.csv").write_text("FOO,BAR\n1,2\n")
    assert nd.main(["--config-dir", str(cfg), "--data-dir", str(data), "--reports-dir", str(rep), "--optional"]) == 1


def test_latest_source_date_wins(tmp_path):
    cfg, data, rep = make_sandbox(tmp_path)
    write_sec(data / "sec_bhavdata_full_10112026.csv", [sec_line("OLD", "EQ", "10-Nov-2026", 1, 1, 1, 1, 1, 10)])
    write_sec(data / "sec_bhavdata_full_11112026.csv", [sec_line("NEW", "EQ", "11-Nov-2026", 1, 1, 1, 1, 1, 10)])
    nd.main(["--config-dir", str(cfg), "--data-dir", str(data), "--reports-dir", str(rep)])
    assert list(pd.read_csv(rep / "nse_eod_processed.csv")["symbol"]) == ["NEW"]


def test_placeholder_master_is_unverified_not_an_error(tmp_path):
    from pathlib import Path
    from common import REPO_ROOT
    m, note = nd.load_security_master(REPO_ROOT / "data" / "nse_security_master.csv")
    assert m is None and "UNVERIFIED" in note
