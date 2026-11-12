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
├── config/
│   ├── market_impact.yml             # global-score weights, directions, risk flags, sector sensitivities
│   ├── screening.yml                 # every technical + risk threshold
│   └── watchlist.csv                 # YOUR symbols (symbol,exchange,notes)
├── data/
│   └── nse_security_master.csv       # placeholder (headers only) -> replace with NSE EQUITY_L.csv
├── scripts/
│   ├── common.py  global_report.py  nse_data.py  technical_screen.py  risk_filter.py  final_report.py
├── tests/                            # 89 offline tests on synthetic data
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

## Download and read the artifact

Open the finished run → **Artifacts** → `tradesentinel-reports` (kept 30 days) → unzip. Start with
`premarket_report.md` (it is also shown on the run's Summary page). Then:

| File | Contents |
|---|---|
| `premarket_report.md` | run status, global score, sector tilts, shortlist with reasons for selection **and** caution |
| `global_report.md/.json` | factor-by-factor numbers, formula, methodology, limitations |
| `nse_data_report.md/.json`, `nse_eod_processed.csv` | NSE source date, download time, validation results |
| `technical_screen_full.csv`, `technical_shortlist.csv` | every symbol with pass/fail reasons; indicator-pass list |
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

## Tests

```bash
pip install -r requirements.txt
python -m pytest tests -q
```
Tests use synthetic data only (no network): indicator maths, EQ/BE/BZ/unknown series, missing/stale handling, score bounds,
`^TNX` conventions, NSE file formats and validation, and an end-to-end run that checks every report is generated. The
workflow runs them before each report.
