#!/usr/bin/env python3
"""Historical evaluation of the swing patterns/setups — does the pattern actually add value?

Method (all choices are in config/swing.yml -> backtest):
  * Walk forward bar by bar: the setup on day t is detected with swing_engine.evaluate_bar(prep, t), which only
    uses bars <= t (no look-ahead). Entry is the NEXT session's open (entry_mode next_open) plus slippage, because
    you cannot trade the close you just computed.
  * Horizons 1,2,3,5,7 sessions after entry: close-to-close net return, max favourable/adverse excursion (MFE/MAE),
    whether the technical invalidation level was touched, and a stop-adjusted return (exit at the stop, or the open
    if it gaps through). Costs: round-trip charges + slippage each side (assumptions, configurable).
  * Compare: pattern alone vs +volume, +trend, +sector, +liquidity, and combined. Baseline = every bar of the same
    stocks (unconditional), so "edge" = pattern average minus baseline average, with a rough t-statistic.
  * Regime = Nifty trend on the signal date. In-sample / out-of-sample split by DATE (last 40% reported separately).
    Thresholds are NOT tuned here, and nothing is optimised on any sample.

Known biases you must read: survivorship (current symbols only), overlapping signals (t-stats are optimistic),
unknown intraday order of stop vs high, no delivery history, free Yahoo data. Results are evidence, not proof.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import swing_engine as se
import technical_screen as ts
from common import add_dir_args, dirs_from_args, fmt, gh_error, gh_warning, load_yaml, now_utc, update_status, write_json

PHASE = "swing_backtest"
TAGS = ["hammer_at_support", "engulfing_at_support", "confirmed_breakout", "breakout_retest", "fvg_continuation",
        "flag_breakout", "ema_pullback_reversal"]
VARIANTS = ["alone", "volume", "trend", "sector", "liquidity", "combined", "combined_sector"]


def tags_for_bar(ev: dict, i: int, df: pd.DataFrame) -> dict[str, float]:
    """tag -> structural low used for the technical invalidation level (None if the tag is absent)."""
    out: dict[str, float] = {}
    for rec in ev["patterns"]:
        if rec["bar_index"] == i and rec["at_support"]:
            if rec["name"] == "hammer":
                out["hammer_at_support"] = rec["lowest_low"]
            elif rec["name"] == "bullish_engulfing":
                out["engulfing_at_support"] = rec["lowest_low"]
    bo = ev["breakout"]
    if bo["status"] == "CONFIRMED_BREAKOUT":
        out["confirmed_breakout"] = min(float(df["Low"].iloc[i]), bo["level"])
    if ev["retest"]["status"] == "BREAKOUT_RETEST":
        out["breakout_retest"] = min(ev["retest"]["retest_low"], ev["retest"]["level"])
    for s in ev["setups"]:
        if s["code"] == "F" and s["confirmed"]:
            out["fvg_continuation"] = s["structure_low"]
        if s["code"] == "D" and s["confirmed"]:
            out["ema_pullback_reversal"] = s["structure_low"]
    fl = ev["flag"]
    if fl:
        out["flag_breakout"] = min(fl["flag_low"], float(df["Low"].iloc[i]))
    return out


def regime_series(bench: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.Series:
    if bench is None or len(bench) < 210:
        return pd.Series("UNKNOWN", index=index)
    c = bench["Close"]
    e50, e200 = ts.ema(c, 50), ts.ema(c, 200)
    reg = pd.Series("SIDEWAYS", index=c.index)
    reg[(c > e50) & (e50 > e200)] = "BULL"
    reg[(c < e50) & (e50 < e200)] = "BEAR"
    reg[e200.isna()] = "UNKNOWN"
    return reg.reindex(index, method="ffill").fillna("UNKNOWN")


def sector_ok_series(df: pd.DataFrame, sec: pd.DataFrame | None, lookback: int) -> pd.Series | None:
    if sec is None or len(sec) < 60:
        return None
    c = sec["Close"]
    up = (c > ts.ema(c, 20)) & (ts.ema(c, 20) > ts.ema(c, 50))
    s_ret = c / c.shift(lookback) - 1
    stock_ret = df["Close"] / df["Close"].shift(lookback) - 1
    up_a = up.reindex(df.index, method="ffill")
    rs = stock_ret - s_ret.reindex(df.index, method="ffill")
    return ((up_a == True) & (rs > 0)).where(up_a.notna() & rs.notna())   # noqa: E712


def collect(symbol: str, df: pd.DataFrame, cfg: dict, bench: pd.DataFrame | None, sec: pd.DataFrame | None) -> tuple[list[dict], pd.DataFrame]:
    """Signals + baseline returns for one symbol. Only information up to each signal date is used."""
    bt, tp = cfg["backtest"], cfg["trade_plan"]
    hs = [int(h) for h in bt["horizons"]]
    cost = float(bt["cost_round_trip_pct"]) / 100.0 + 2 * float(bt["slippage_pct_each_side"]) / 100.0
    slip = float(bt["slippage_pct_each_side"]) / 100.0
    prep = se.prepare(df, cfg)
    n = len(df)
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    atr = prep.m["atr"].values
    regime = regime_series(bench, df.index)
    secok = sector_ok_series(df, sec, int(cfg["sector"]["outperform_lookback"]))
    adv = (df["Close"] * df["Volume"]).shift(1).rolling(20, min_periods=20).mean() / 1e7
    warm = int(bt["warmup_bars"])
    maxh = max(hs)

    def outcome(t: int, stop: float | None) -> dict | None:
        """Entry next open; metrics per horizon. Needs bars up to t+maxh to exist for that horizon."""
        if t + 1 >= n:
            return None
        entry = (o[t + 1] if bt["entry_mode"] == "next_open" else c[t]) * (1 + slip)
        res: dict = {"entry": float(entry)}
        for hz in hs:
            end = t + hz
            if end >= n:
                res[f"ret_{hz}"] = res[f"mfe_{hz}"] = res[f"mae_{hz}"] = res[f"stophit_{hz}"] = res[f"retstop_{hz}"] = np.nan
                continue
            first = t + 1 if bt["entry_mode"] == "next_open" else t + 1
            hi, lo = h[first:end + 1], l[first:end + 1]
            res[f"ret_{hz}"] = float(c[end] / entry - 1 - cost)
            res[f"mfe_{hz}"] = float(hi.max() / entry - 1)
            res[f"mae_{hz}"] = float(lo.min() / entry - 1)
            if stop is not None:
                hit = np.where(lo <= stop)[0]
                if len(hit):
                    k = first + int(hit[0])
                    exit_px = min(o[k], stop) if k > first or bt["entry_mode"] != "next_open" else min(stop, o[k])
                    res[f"stophit_{hz}"] = 1.0
                    res[f"retstop_{hz}"] = float(exit_px * (1 - slip) / entry - 1 - float(bt["cost_round_trip_pct"]) / 100.0)
                else:
                    res[f"stophit_{hz}"] = 0.0
                    res[f"retstop_{hz}"] = res[f"ret_{hz}"]
            else:
                res[f"stophit_{hz}"] = np.nan
                res[f"retstop_{hz}"] = np.nan
        return res

    signals: list[dict] = []
    for i in range(warm, n - 1):
        ev = se.evaluate_bar(prep, i)
        tg = tags_for_bar(ev, i, df)
        if not tg:
            continue
        rv = prep.m["rel_vol"].iloc[i]
        trend_ok = bool(ev["trend"]["established_uptrend"] and ev["trend"]["weekly_ok"])
        liq_ok = bool(np.isfinite(adv.iloc[i]) and adv.iloc[i] >= float(bt["liquidity_min_value_cr"]) and c[i] >= float(cfg["risk"]["min_price"]))
        vol_ok = bool(np.isfinite(rv) and rv >= float(bt["volume_min_rel"]))
        sk = None if secok is None or pd.isna(secok.iloc[i]) else bool(secok.iloc[i])
        for tag, low in tg.items():
            entry_ref = c[i]
            stop = low - float(tp["stop_buffer_atr"]) * (atr[i] if np.isfinite(atr[i]) else 0)
            if not np.isfinite(stop) or stop >= entry_ref or (entry_ref - stop) / entry_ref * 100 > float(tp["max_stop_pct"]):
                continue                                         # same "reasonable invalidation" gate as the live scan
            out = outcome(i, stop)
            if out is None:
                continue
            signals.append({"symbol": symbol, "date": df.index[i], "tag": tag, "vol_ok": vol_ok, "trend_ok": trend_ok, "liq_ok": liq_ok,
                            "sector_ok": sk, "regime": regime.iloc[i], "stop": float(stop), **out})
    base_rows = []
    for hz in hs:
        idx = np.arange(warm, n - hz - 1)
        if len(idx) == 0:
            continue
        entry = (o[idx + 1] if bt["entry_mode"] == "next_open" else c[idx]) * (1 + slip)
        ret = c[idx + hz] / entry - 1 - cost
        base_rows.append(pd.DataFrame({"date": df.index[idx], "horizon": hz, "ret": ret, "regime": regime.iloc[idx].values}))
    return signals, (pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame(columns=["date", "horizon", "ret", "regime"]))


def select_variant(sig: pd.DataFrame, variant: str) -> pd.DataFrame:
    if variant == "alone":
        return sig
    if variant == "volume":
        return sig[sig["vol_ok"]]
    if variant == "trend":
        return sig[sig["trend_ok"]]
    if variant == "liquidity":
        return sig[sig["liq_ok"]]
    if variant == "sector":
        return sig[sig["sector_ok"] == True]   # noqa: E712
    if variant == "combined":
        return sig[sig["vol_ok"] & sig["trend_ok"] & sig["liq_ok"]]
    if variant == "combined_sector":
        return sig[sig["vol_ok"] & sig["trend_ok"] & sig["liq_ok"] & (sig["sector_ok"] == True)]   # noqa: E712
    raise ValueError(variant)


def max_drawdown(rets: np.ndarray) -> float:
    """Peak-to-trough decline of the cumulative sum of per-trade net returns in date order (equal-size trades), in %."""
    if len(rets) == 0:
        return float("nan")
    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(np.r_[0.0, cum])[1:]
    return float((cum - peak).min() * 100)


def stats(sel: pd.DataFrame, hz: int, base: pd.DataFrame, min_n: int) -> dict:
    r = sel[f"ret_{hz}"].dropna()
    rs = sel[f"retstop_{hz}"].dropna()
    n = int(len(r))
    out = {"n": n, "sample_flag": "OK" if n >= min_n else ("NONE" if n == 0 else "INSUFFICIENT SAMPLE")}
    if n == 0:
        return out
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    out.update(win_rate=float((r > 0).mean() * 100), avg_ret=float(r.mean() * 100), median_ret=float(r.median() * 100),
               profit_factor=float(wins / losses) if losses > 0 and n >= min_n else float("nan"),
               max_drawdown=max_drawdown(sel.sort_values("date").loc[r.index, f"ret_{hz}"].values),
               avg_mfe=float(sel[f"mfe_{hz}"].mean() * 100), avg_mae=float(sel[f"mae_{hz}"].mean() * 100),
               stop_hit_rate=float(sel[f"stophit_{hz}"].mean() * 100), avg_ret_stop=float(rs.mean() * 100) if len(rs) else float("nan"),
               win_rate_stop=float((rs > 0).mean() * 100) if len(rs) else float("nan"))
    b = base[base["horizon"] == hz]["ret"].dropna()
    if len(b):
        out["baseline_avg"] = float(b.mean() * 100)
        out["edge_vs_baseline"] = out["avg_ret"] - out["baseline_avg"]
        se_ = r.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        out["t_stat"] = float((r.mean() - b.mean()) / se_) if se_ and np.isfinite(se_) and se_ > 0 else float("nan")
    return out


def evaluate(signals: pd.DataFrame, base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    bt = cfg["backtest"]
    hs = [int(h) for h in bt["horizons"]]
    min_n = int(bt["min_sample"])
    rows = []
    if signals.empty and base.empty:
        return pd.DataFrame()
    alld = pd.concat([signals["date"], base["date"]]) if len(base) else signals["date"]
    split = alld.min() + (alld.max() - alld.min()) * (1 - float(bt["oos_fraction"]))
    periods = {"ALL": lambda d: d.notna(), "IN_SAMPLE": lambda d: d < split, "OUT_OF_SAMPLE": lambda d: d >= split}
    for pname, f in periods.items():
        b_p = base[f(base["date"])] if len(base) else base
        for tag in TAGS:
            s_t = signals[(signals["tag"] == tag) & f(signals["date"])] if len(signals) else signals
            for v in VARIANTS:
                sel = select_variant(s_t, v) if len(s_t) else s_t
                for hz in hs:
                    st = stats(sel, hz, b_p, min_n)
                    rows.append({"pattern": tag, "variant": v, "horizon": hz, "period": pname, **st})
        for hz in hs:        # baseline row for reference
            b = b_p[b_p["horizon"] == hz]["ret"]
            rows.append({"pattern": "BASELINE_ALL_BARS", "variant": "alone", "horizon": hz, "period": pname, "n": int(len(b)),
                         "sample_flag": "OK" if len(b) >= min_n else "INSUFFICIENT SAMPLE",
                         "win_rate": float((b > 0).mean() * 100) if len(b) else np.nan, "avg_ret": float(b.mean() * 100) if len(b) else np.nan,
                         "median_ret": float(b.median() * 100) if len(b) else np.nan})
    res = pd.DataFrame(rows)
    res.attrs["split_date"] = split
    return res


def regime_table(signals: pd.DataFrame, base: pd.DataFrame, cfg: dict, hz: int = 5) -> pd.DataFrame:
    rows = []
    min_n = int(cfg["backtest"]["min_sample"])
    for tag in TAGS:
        for reg in ("BULL", "SIDEWAYS", "BEAR", "UNKNOWN"):
            s = signals[(signals["tag"] == tag) & (signals["regime"] == reg)] if len(signals) else signals
            r = s[f"ret_{hz}"].dropna() if len(s) else pd.Series(dtype=float)
            if len(r):
                rows.append({"pattern": tag, "regime": reg, "horizon": hz, "n": len(r), "win_rate": (r > 0).mean() * 100,
                             "avg_ret": r.mean() * 100, "sample_flag": "OK" if len(r) >= min_n else "INSUFFICIENT SAMPLE"})
    for reg in ("BULL", "SIDEWAYS", "BEAR"):
        b = base[(base["horizon"] == hz) & (base["regime"] == reg)]["ret"] if len(base) else pd.Series(dtype=float)
        if len(b):
            rows.append({"pattern": "BASELINE_ALL_BARS", "regime": reg, "horizon": hz, "n": len(b), "win_rate": (b > 0).mean() * 100,
                         "avg_ret": b.mean() * 100, "sample_flag": "OK"})
    return pd.DataFrame(rows)


def verdict(row: pd.Series, min_n: int) -> str:
    if row.get("n", 0) < min_n:
        return "insufficient sample: no conclusion"
    t, e = row.get("t_stat", np.nan), row.get("edge_vs_baseline", np.nan)
    if not np.isfinite(t):
        return "no baseline comparison"
    if t >= 2 and e > 0:
        return "positive edge vs baseline (t>=2) — still subject to the biases listed"
    if t <= -2 and e < 0:
        return "NEGATIVE edge vs baseline"
    return "no reliable edge vs baseline"


def render_md(res: pd.DataFrame, reg: pd.DataFrame, meta: dict, cfg: dict) -> str:
    bt = cfg["backtest"]
    min_n = int(bt["min_sample"])
    L = ["# TradeSentinel — Swing Pattern Backtest", "", f"Generated {meta['generated']} UTC · {meta['symbols']} symbols · {meta['bars']:,} bars · "
         f"signals {meta['signals']:,} · out-of-sample from {pd.Timestamp(meta['split']).date() if meta.get('split') is not None else 'n/a'}", "",
         "> **Purpose: test whether these patterns add value, not to prove they do.** Read the biases section before trusting any number.", ""]
    if meta["signals"] == 0:
        L += ["**No setups were detected in the supplied history — validation is INCOMPLETE.** Provide longer history or more symbols.", ""]
    if meta["complete"] is False:
        L += [f"**VALIDATION INCOMPLETE:** {meta['incomplete_reason']}", ""]
    L += ["## Assumptions", "",
          f"- Entry: {bt['entry_mode']} (+{bt['slippage_pct_each_side']}% slippage); costs {bt['cost_round_trip_pct']}% round trip + slippage each side. These are **assumptions**, not your actual charges.",
          f"- Horizons {bt['horizons']} sessions; stop-adjusted returns exit at the technical invalidation (or the open if it gaps through). Intraday order of stop vs high is unknowable from daily bars.",
          f"- A sample below {min_n} trades is marked INSUFFICIENT SAMPLE. Thresholds in `swing.yml` were set from general convention and are **not tuned** on this data.", "",
          "## Headline: 5-session net returns, ALL / out-of-sample", "",
          "| Pattern | Variant | Period | n | Win % | Avg % | Median % | PF | Max DD % | Stop-hit % | Baseline avg % | Edge | t | Reading |", "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|"]
    show = res[(res["horizon"] == 5) & (res["variant"].isin(["alone", "combined"])) & (res["period"].isin(["ALL", "OUT_OF_SAMPLE"]))]
    for _, r in show.iterrows():
        if r["pattern"] == "BASELINE_ALL_BARS":
            L.append(f"| **baseline (all bars)** | | {r['period']} | {int(r['n'])} | {fmt(r['win_rate'], 1)} | {fmt(r['avg_ret'], 2)} | {fmt(r['median_ret'], 2)} | | | | | | | reference |")
            continue
        L.append(f"| {r['pattern']} | {r['variant']} | {r['period']} | {int(r['n'])} | {fmt(r.get('win_rate'), 1)} | {fmt(r.get('avg_ret'), 2)} | {fmt(r.get('median_ret'), 2)} | "
                 f"{fmt(r.get('profit_factor'), 2)} | {fmt(r.get('max_drawdown'), 1)} | {fmt(r.get('stop_hit_rate'), 0)} | {fmt(r.get('baseline_avg'), 2)} | "
                 f"{fmt(r.get('edge_vs_baseline'), 2)} | {fmt(r.get('t_stat'), 1)} | {verdict(r, min_n)} |")
    L += ["", "Full grid (every pattern × variant × horizon × period) is in `swing_backtest.csv`; individual trades in `swing_backtest_trades.csv`.", "",
          "## Does adding filters help? (5-session, ALL period: average net %, n)", "", "| Pattern | " + " | ".join(VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    sub = res[(res["horizon"] == 5) & (res["period"] == "ALL") & (res["pattern"] != "BASELINE_ALL_BARS")]
    for tag in TAGS:
        cells = []
        for v in VARIANTS:
            r = sub[(sub["pattern"] == tag) & (sub["variant"] == v)]
            cells.append("n/a" if r.empty or r.iloc[0]["n"] == 0 else f"{fmt(r.iloc[0].get('avg_ret'), 2)} (n={int(r.iloc[0]['n'])})")
        L.append(f"| {tag} | " + " | ".join(cells) + " |")
    L += ["", "## By market regime (Nifty trend on the signal date), 5-session", ""]
    if reg.empty or (reg["regime"] == "UNKNOWN").all():
        L += ["Regime analysis incomplete: benchmark history unavailable or too short.", ""]
    else:
        L += ["| Pattern | Regime | n | Win % | Avg % | Sample |", "|---|---|--:|--:|--:|---|"]
        L += [f"| {r['pattern']} | {r['regime']} | {int(r['n'])} | {r['win_rate']:.1f} | {r['avg_ret']:.2f} | {r['sample_flag']} |" for _, r in reg.iterrows()] + [""]
    L += ["## Biases and limits (read these)", "",
          "- **Survivorship bias:** only symbols that exist today are tested; delisted/collapsed stocks are missing, which flatters every result.",
          "- **Overlapping signals:** consecutive days on one stock are not independent; t-statistics are optimistic.",
          "- **Sector variant:** needs sector index history for the symbol; where unavailable it is excluded, so its n is smaller and not comparable one-to-one.",
          "- **Delivery % is not tested:** free data has no daily delivery history. **Liquidity** uses each stock's own traded-value history, not exchange order-book data.",
          "- **Data quality:** Yahoo prices (split-adjusted close, unadjusted OHLC quirks) can contain errors.",
          "- A pattern being popular is not evidence. If an `Edge` is small, negative or its sample is small, treat the pattern as unproven and rely on the hard risk gates, not the pattern.", ""]
    return "\n".join(L)


def run(args, provider=None, index_provider=None) -> int:
    from swing_scan import default_index_provider, load_sector_map
    config_dir, data_dir, reports = dirs_from_args(args)
    cfg = load_yaml(config_dir / "swing.yml")
    bt = cfg["backtest"]
    provider = provider or ts.default_history_provider(data_dir)
    index_provider = index_provider or default_index_provider
    entries, problems = ts.load_watchlist(config_dir / "watchlist.csv")
    symbols = [e["symbol"] for e in entries]
    if args.universe in ("nse", "both"):
        p = reports / "nse_eod_processed.csv"
        if p.exists():
            nse = pd.read_csv(p)
            pool = nse[(nse["series"] == "EQ") & (nse["traded_value_cr"] >= float(bt["liquidity_min_value_cr"]) / 4)]
            symbols += [s for s in pool.sort_values("traded_value_cr", ascending=False)["symbol"] if s not in symbols]
        elif args.universe == "nse":
            gh_error("swing_backtest: --universe nse needs NSE data.")
            return 1
    if args.universe == "nse":
        symbols = [s for s in symbols if s not in {e["symbol"] for e in entries}]
    symbols = symbols[:(args.max_symbols or int(bt["max_symbols"]))]
    if not symbols:
        update_status(reports, PHASE, "SKIPPED", "empty universe")
        gh_warning("swing_backtest: empty universe.")
        return 0 if args.optional else 1

    smap = load_sector_map(config_dir / cfg["sector"]["map_file"]).set_index("symbol").to_dict("index") if (config_dir / cfg["sector"]["map_file"]).exists() else {}
    cache: dict = {}

    def idx(t):
        if t and t not in cache:
            df, _ = index_provider(t, bt["history_period"])
            cache[t] = ts.clean_ohlcv(df)
        return cache.get(t)
    bench = idx(cfg["sector"]["benchmark"])
    all_sig, all_base, bars, used = [], [], 0, 0
    for k, sym in enumerate(symbols, 1):
        try:
            df, _ = provider(sym, bt["history_period"])
        except Exception as exc:  # noqa: BLE001
            print(f"[swing_backtest] {sym}: {type(exc).__name__}: {exc}")
            continue
        d = ts.clean_ohlcv(df)
        if d is None or len(d) < int(bt["warmup_bars"]) + 30:
            continue
        sec = idx((smap.get(sym) or {}).get("sector_index", ""))
        s, b = collect(sym, d, cfg, bench, sec)
        all_sig += s
        all_base.append(b)
        bars += len(d)
        used += 1
        if k % 10 == 0:
            print(f"[swing_backtest] {k}/{len(symbols)} symbols, {len(all_sig)} signals")
    sig = pd.DataFrame(all_sig)
    base = pd.concat(all_base, ignore_index=True) if all_base else pd.DataFrame(columns=["date", "horizon", "ret", "regime"])
    if used == 0:
        gh_error("swing_backtest: no symbol had enough history.")
        update_status(reports, PHASE, "FAILED", "no usable history")
        return 1
    res = evaluate(sig, base, cfg)
    reg = regime_table(sig, base, cfg)
    complete, reason = True, ""
    if len(sig) < int(bt["min_sample"]):
        complete, reason = False, f"only {len(sig)} setups detected in total (< {bt['min_sample']}); most statistics are INSUFFICIENT SAMPLE."
    elif used < 20:
        complete, reason = False, f"only {used} symbols tested; patterns need a broad universe (use --universe both / nse) for reliable conclusions."
    meta = {"generated": now_utc().isoformat(timespec="seconds"), "symbols": used, "bars": bars, "signals": len(sig), "split": res.attrs.get("split_date"),
            "complete": complete, "incomplete_reason": reason}
    res.to_csv(reports / "swing_backtest.csv", index=False)
    sig.to_csv(reports / "swing_backtest_trades.csv", index=False)
    (reports / "swing_backtest.md").write_text(render_md(res, reg, meta, cfg), encoding="utf-8")
    write_json(reports / "swing_backtest.json", {**meta, "results": res.to_dict("records"), "regimes": reg.to_dict("records"),
                                                  "assumptions": {k: bt[k] for k in ("entry_mode", "cost_round_trip_pct", "slippage_pct_each_side", "horizons", "min_sample", "oos_fraction")},
                                                  "watchlist_problems": problems})
    update_status(reports, PHASE, "OK" if complete else "DEGRADED", f"{used} symbols, {len(sig)} setups" + ("" if complete else f"; {reason}"))
    if not complete:
        gh_warning(f"swing_backtest: validation incomplete: {reason}")
    print(f"[swing_backtest] symbols={used} signals={len(sig)} complete={complete}")
    return 0


def main(argv=None, provider=None, index_provider=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dir_args(ap)
    ap.add_argument("--universe", choices=["watchlist", "nse", "both"], default="watchlist")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--optional", action="store_true")
    args = ap.parse_args(argv)
    try:
        return run(args, provider, index_provider)
    except Exception as exc:  # noqa: BLE001
        gh_error(f"swing_backtest failed: {type(exc).__name__}: {exc}")
        update_status(Path(args.reports_dir), PHASE, "FAILED", f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
