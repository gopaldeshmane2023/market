# TradeSentinel

A personal, **GitHub-Actions-only** research pipeline for Indian equities (NSE). It produces downloadable reports; it
has no web app, database, broker integration or order placement. No API keys, no paid data.

**WAIT. CONFIRM. PROTECT.**

> Research tooling only, not investment advice. Indicator filters are not forecasts. **PASS never means "safe", and EQ
> series never guarantees an exit** — circuits, missing buyers, thin order-book depth and exchange restrictions can
> still trap you. (T+1 settlement and trade-to-trade classification are different concepts.)

## Folder structure

```
tradesentinel/
├── .github/workflows/premarket.yml   # Mon-Fri 07:00 IST + manual run
├── .github/workflows/swing.yml       # weekdays 19:30 IST (after close) + Saturday review + manual run
├── config/
│   ├── market_impact.yml             # global-score weights, directions, risk flags, sector sensitivities
│   ├── screening.yml                 # every technical + risk threshold (daily screen)
│   ├── swing.yml                     # WEEKLY SWING strategy thresholds (separate)
│   ├── sector_map.csv                # your symbol -> industry/sector/sector-index map (starter, verify)
│   └── watchlist.csv                 # YOUR symbols (symbol,exchange,notes)
├── data/
│   └── nse_security_master.csv       # placeholder (headers only) -> replace with NSE EQUITY_L.csv
├── scripts/
│   ├── common.py  global_report.py  nse_data.py  technical_screen.py  risk_filter.py  final_report.py
│   └── swing_patterns.py  swing_levels.py  swing_engine.py  swing_scan.py  swing_backtest.py   # weekly swing
├── tests/                            # 142 offline tests on synthetic data
├── reports/.gitkeep                  # outputs land here (not committed)
├── requirements.txt  .gitignore  README.md
```

## What is implemented, and how far it can be trusted

| Phase | Status | Depends on |
|---|---|---|
| 1 Global report (`global_report.py`) | **Fully implemented.** Live fetch from Yahoo not verifiable offline, so run the workflow once and read the report. | Yahoo Finance via `yfinance` (free, unofficial) |
| 2 NSE data (`nse_data.py`) | **Implemented and validated against synthetic files** in the layouts I know (below). **Manual-input dependent.** Auto-download is best-effort and untested against the live NSE site. | Official NSE files you place in `data/` |
| 3 Technical screen (`technical_screen.py`) | **Fully implemented** and unit-tested. History comes from Yahoo (`SYMBOL.NS`) or `data/history/SYMBOL.csv`. | Yahoo (or local CSV) |
| 4 Risk filter (`risk_filter.py`) | **Fully implemented** rules. Circuits, ASM/GSM and spreads are only assessed if you supply the data; otherwise they are explicitly "not assessed". | NSE series/delivery data; optional band/surveillance CSVs |
| Final report (`final_report.py`) | Implemented | Outputs above |

Without NSE files, every symbol's **series is UNKNOWN**, so by design nothing is shortlisted and the risk filter rejects
everything ("unknown series: rejected until verified"). The report says so plainly. That is the capital-protection
default, not a bug.

## Upload to GitHub

1. Create a new **private** repository on github.com (e.g. `tradesentinel`), empty (no README).
2. Unzip this project. In the folder run:
   ```bash
   git init && git add . && git commit -m "TradeSentinel"
   git branch -M main
   git remote add origin https://github.com/<you>/tradesentinel.git
   git push -u origin main
   ```
   (Or use *Add file → Upload files* in the browser; keep the `.github/workflows/` path intact.)
3. Repo **Settings → Actions → General**: make sure Actions are enabled. No secrets are needed.

## Run the workflow manually

*Actions* tab → **TradeSentinel pre-market** → **Run workflow**. Options:
`universe` = `watchlist` (default), `nse` (liquid EQ stocks from the NSE file, capped by `technical.universe.max_symbols`)
or `both`; `try_nse_download` = attempt NSE's public archive download first.

The scheduled run is `30 1 * * 1-5` (UTC) = 07:00 IST. GitHub may start it up to ~an hour late. Scheduled runs on
repositories with no activity for 60 days are paused by GitHub; re-enable them in the Actions tab.

## The two files to open first

After a run, open the artifact and read just these two:

- **`TradeSentinel_results.xlsx`** (Excel) — sheet **Top_Candidates** is your daily list (5-10 ranked stocks); **Why_Filtered**
  shows how many stocks fell out at each step; All_Screened / Rejected hold everything else with reasons.
- **`premarket_report.md`** — same list in plain English, with a 30-second "how to read" guide.

**Why the list is never blank:** stocks are ranked by a 0-100 score with partial credit. `STRICT_PASS` = every bullish
rule met; `NEAR_MISS` = best remaining stock that passed the hard safety rules, with the missing rule named. Hard safety
rules (EQ series only, liquidity, data quality, no lower circuit) are **never** relaxed to reach the minimum, so on a bad day
you may see fewer than 5 and the report tells you why. Change the size of the list in `config/screening.yml` → `output`.
The other CSV files are detail for debugging.

## Download and read the artifact

Open the finished run → **Artifacts** → `tradesentinel-reports` (kept 30 days) → unzip. Start with
`premarket_report.md` (it is also shown on the run's Summary page). Then:

| File | Contents |
|---|---|
| `premarket_report.md` | run status, global score, sector tilts, shortlist with reasons for selection **and** caution |
| `global_report.md/.json` | factor-by-factor numbers, formula, methodology, limitations |
| `nse_data_report.md/.json`, `nse_eod_processed.csv` | NSE source date, download time, validation results |
| `top_candidates.csv`, `TradeSentinel_results.xlsx` | the ranked daily list (see above) |
| `technical_screen_full.csv`, `technical_shortlist.csv` | every symbol with pass/fail reasons; `technical_shortlist.csv` is the strict-all-rules list and can be empty |
| `funnel.json` | counts explaining where stocks dropped out |
| `risk_report.md/.csv/.json` | PASS / WARNING / REJECT with exact reasons |
| `final_shortlist.csv` | indicator-pass + EQ + not REJECT |
| `run_status.json` | OK / DEGRADED / SKIPPED / FAILED per phase |

No entry prices, targets or stop losses are ever generated.

## Update your watchlist

Edit `config/watchlist.csv` (in GitHub: open file → pencil icon → commit):

```
symbol,exchange,notes
KALYANKJIL,NSE,from screener.in 12 Nov
```
Use the **NSE trading symbol** (not the Screener.in company name). Exchange must be `NSE`; others are skipped with a
message. `#` lines are comments. The three rows shipped (RELIANCE, TCS, HDFCBANK) are **samples** — delete them. Screener.in
is never scraped: export your screen as CSV, copy the symbols over.

## Supplying NSE data (Phase 2)

Download from NSE's official reports pages (no CAPTCHA/login is ever bypassed), unzip, and commit into `data/`:

| File to place in `data/` | Where it comes from | Gives |
|---|---|---|
| `sec_bhavdata_full_DDMMYYYY.csv` (preferred) | NSE "Full Bhavcopy and Security-wise Delivery Positions" | OHLC, prev close, volume, turnover, **delivery qty/%**, series |
| `BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv` | NSE CM (UDiFF) bhavcopy | same, no delivery |
| `cm*bhav.csv` | legacy CM bhavcopy | same, no delivery |
| `nse_security_master.csv` | NSE `EQUITY_L.csv` (listed equities) | listing check, series cross-check |
| `nse_price_band.csv` *(optional)* | you, from NSE price-band list: `symbol,price_band_pct` | circuit detection |
| `nse_surveillance.csv` *(optional)* | you, from NSE ASM/GSM lists: `symbol,list[,stage]` | surveillance flags |

The file must be the **latest trading day** — a stale file is flagged with its age. Column names are checked before
processing; if NSE changes a layout the run **fails with the exact missing columns** instead of guessing. A missing or
truncated file is never treated as an empty market. Because committing a bhavcopy every day is tedious, you can set a
repository variable `NSE_AUTO_DOWNLOAD=true` (Settings → Secrets and variables → Actions → Variables) so scheduled runs
attempt the public archive download; if NSE blocks GitHub's servers the run says so and falls back to the files in `data/`.
Optional price-history override: `data/history/SYMBOL.csv` with `Date,Open,High,Low,Close,Volume`.

## Data sources and limits (all free)

- **Yahoo Finance (`yfinance`)** — unofficial; may lag, have gaps, return partial sessions or occasionally break. Futures
  tickers roll contracts. Used for global factors and for price history.
- **NSE official files** — authoritative, but manual (or best-effort download); published after market close.
- **Screener.in** — manual exports only.
- No bid/ask data in free files, so spread checks are not assessed. Circuits/ASM/GSM need your optional CSVs.
- Circuit detection is an **approximation** (move within `circuit_tolerance_pct` of today's band and close at the day's
  extreme); repeated-circuit counts apply today's band to earlier days.

## How the global score works

`signal = clip(move / full_move, -1, 1)`, `points = direction × weight × signal`,
`score = 100 × Σpoints / Σweights` over factors with usable data — always within -100…+100. Weights/directions are in
`config/market_impact.yml` and are **heuristic**, not fitted correlations. Missing/stale factors are excluded and the rest
renormalised; below 60% weight coverage no score is produced. A high score is context, **never a buy signal**.

**`^TNX`:** Yahoo has quoted it both as yield×10 (42.5) and as percent (4.25). The script auto-detects from magnitude
(≥15 ⇒ tenths), prints the decision, validates the converted yield (0–15%) and the daily move (>60bp ⇒ SUSPECT, excluded),
and reports the change in **basis points**. Force a convention with `tnx_convention` if needed.

## Screening and risk rules (all in `config/screening.yml`)

Bullish screen: EMA20>EMA50>EMA200, RSI14 50–70, MACD>signal, relative volume ≥1.5 (vs the **prior** 20 sessions),
positive daily and 5-day momentum, ≥210 bars, EQ only. Indicators are computed in pandas/numpy (Wilder RSI/ATR).
Risk: **REJECT** for non-EQ (BE/BZ/SM… known-restricted, unknown = rejected until verified), missing data, price <₹20,
average traded value <₹2 cr or volume <1 lakh shares, lower circuit. **WARNING** for thin liquidity, low/missing delivery %,
upper circuit, repeated circuits, ASM/GSM, stale data, Yahoo-vs-NSE price mismatch, not in security master.


## Weekly Swing Strategy (separate from the daily screen)

An independent 1-7 session price-action scanner. It reuses the existing data modules (`nse_data`, `technical_screen`
indicators/history/watchlist, `risk_filter` series-liquidity-circuit-surveillance gates, `global_report`) but has its own
config (`config/swing.yml`), scripts, reports and workflow (`swing.yml`). It does not touch intraday screening or the
daily pipeline.

**The one rule that matters:** a stock near resistance is **not** a confirmed breakout. `CONFIRMED_BREAKOUT` requires a
*completed* daily candle that **closes** above resistance + buffer (max of 0.25% or 0.1 ATR), with body/close-position/
upper-wick/volume quality checks. Statuses are never mixed: `POTENTIAL_BREAKOUT`, `BREAKOUT_PENDING_CLOSE`
(today's candle still forming: never confirmed), `CONFIRMED_BREAKOUT`, `BREAKOUT_RETEST`, `FAILED_BREAKOUT`,
`RESISTANCE_REJECTION` (crossed intraday but closed below, or long upper wick with a weak close). Resistance comes from
bars up to the *previous* day only, pivots are used only after their right-hand bars have closed, and weekly values use
completed weeks only, so no future candle ever confirms an earlier date (tests assert this).

| Piece | What it does |
|---|---|
| `swing_patterns.py` | 22 candle patterns from explicit formulas (hammer, engulfing, morning star ... shooting star, distribution, exhaustion), each with shape/volume/location/context strength and a formula explanation |
| `swing_levels.py` | support/resistance **zones** (pivots, 20/50-day highs/lows, volume-supported zones, repeated rejections), breakout/retest/failed logic, three-candle FVGs, weekly trend, flag/pennant/triangle/cup |
| `swing_engine.py` | setups kept separate: A confirmed breakout, B retest, C support reversal, D pullback in uptrend, E consolidation/flag/triangle/cup breakout, F FVG continuation; formula-derived invalidation and reward/risk |
| `swing_scan.py` | hard gates -> 0-100 score -> grade A+/A/B/C/Rejected; reports |
| `swing_backtest.py` | walk-forward test of each pattern vs an unconditional baseline |

**Hard gates beat the score:** not EQ, liquidity below Rs20 cr 20-day average traded value, price below Rs50, ASM/GSM,
lower circuit, stale/missing data, invalidated setup (bearish engulfing, evening star, failed breakout, rejection ...),
no reasonable invalidation (none below entry, or >10% away), next resistance too close (reward/risk below 1.5).
Unconfirmed setups and early reversals are capped at **B**. A+ also needs a measurable reward/risk and no medium warnings.

**Liquidity and capital:** participation = order value / 20-day average traded value (Rs, not share count). Cap 1%.
For each candidate the report shows what an Rs8 lakh order would be as % of daily turnover and the maximum order at the
cap. Position size = min(risk budget 0.25% of capital / stop distance, 25% of capital, participation cap). Turnover is **not**
order-book depth; depth is not available free, so execution capacity is always marked *not verified*.

**Reference levels, not targets:** invalidation = structural low of the setup minus 0.2 ATR; "next resistance" is shown
only as a distance for reward/risk. If nothing overhead is found, reward/risk is `n/a` (cannot be A+). Nothing is
predicted.

**Sector and global context** (supporting, 10% + 5% of the score): sector trend and stock-vs-sector relative strength need
`config/sector_map.csv` (symbol -> sector index ticker, a starter file you must verify) and Yahoo sector-index data
(e.g. `^CNXIT`); if either is missing the factor is shown as **UNKNOWN** and the score is renormalised, never invented.
Global alignment is read from the existing `global_report.json`. Nifty direction uses `^NSEI`.

### Run it

*Actions -> **TradeSentinel weekly swing** -> Run workflow*: `universe` (watchlist/nse/both), `mode` (daily/weekend),
`run_backtest`, `try_nse_download`. Scheduled: weekdays 19:30 IST (after the close, so candles are complete) and Saturday
10:00 IST with the backtest. Download artifact `tradesentinel-swing-reports`; read `swing_report.md` first.
Files: `swing_report.md/json`, `swing_candidates.csv` (A+/A/B/C), `swing_rejected.csv` (every reason),
`swing_backtest.md/csv/json`, `swing_backtest_trades.csv`. Locally: `python scripts/swing_scan.py --universe watchlist`.

**Completeness rule:** Yahoo has no "candle complete" flag, so a candle dated today is treated as complete only after
15:30 IST + 20 minutes; otherwise it is excluded and a crossing can only be reported as `BREAKOUT_PENDING_CLOSE`.
When you supply the NSE EOD file its date/close are cross-checked against the last Yahoo bar. Run it after ~19:30 IST.

### Backtest: read it before trusting a grade

`swing_backtest.py` records every setup using only data available that day, enters at the **next open**, and measures
1/2/3/5/7-session net returns (costs and slippage are *assumptions* in `swing.yml`), max favourable/adverse excursion,
invalidation hits and stop-adjusted returns; shows win rate, average, median, profit factor (only with n >= 30), max
drawdown, regime (Nifty trend) splits, and edge vs. an all-bars baseline with a rough t-stat. It compares each pattern
alone vs +volume, +trend, +sector, +liquidity and combined, and reports in-sample vs out-of-sample (last 40% by date);
**thresholds are not tuned on any sample.** Known biases: survivorship (today's symbols only), overlapping signals
(optimistic t-stats), no delivery history, sector variant only where index data exists. With too few symbols or setups
the report says **VALIDATION INCOMPLETE / INSUFFICIENT SAMPLE** instead of drawing conclusions. A well-known pattern is not
evidence of profit: on pure random data the module reports no edge, as it should.

### Swing data sources and limits
Same free sources as above (Yahoo daily OHLCV, NSE official files you supply, your sector map). 4-hour confirmation is
optional/off (`use_4h: false`): free Yahoo intraday history is short and unreliable, so it is not used by default.

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```
Tests use synthetic data only (no network): indicator maths, EQ/BE/BZ/unknown series, missing/stale handling, score bounds,
`^TNX` conventions, NSE file formats and validation, and an end-to-end run that checks every report is generated. The
workflow runs them before each report.
