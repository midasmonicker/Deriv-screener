# deriv-vol-lab

Research and analytics tools for Deriv Volatility Indices.

## Setup

Requires Python 3.11 or newer.

```sh
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Set `DERIV_APP_ID`, `DB_PATH`, and `LOG_LEVEL` in the environment as needed.
The typed settings model provides defaults so the project can be run before
those values are configured.

## Development

```sh
make lint
make test
make run
```

The package uses a `src` layout; importable code lives in `src/deriv_vol_lab/`.

## Deriv market data

`deriv_vol_lab.data.DerivWebSocketClient` connects to the public Deriv
WebSocket API using an application ID. It validates the local volatility-index
registry against `active_symbols` when connecting, pages historical ticks or
candles backwards in requests of at most 5,000 records, and exposes live tick
and OHLC subscriptions as async streams. Closed clients raise
`ClientClosedError`; an interrupted request raises retryable
`ConnectionLostError`. Failed resubscriptions deliver their terminal error once
and then end the stream. Unexpected connection-manager failures are logged,
reported to in-flight callers, and retried with the configured backoff.

## Storage and quality checks

`deriv_vol_lab.data.DuckDBStorage` persists ticks in `ticks` and candles in
granularity-specific tables such as `candles_60`. Both use `(symbol, epoch)`
keys with idempotent upserts. Use `export_parquet("ticks", path)` or pass a
candle granularity to export a table to Parquet. `find_gaps` reports expected
cadence-aligned epochs; `await backfill_gaps(client, ...)` fetches and stores
the missing history. `quality_report(records=raw_observations)` checks raw
input order and duplicate epochs before the unique-key upsert, as well as
non-positive prices, >8-sigma local log-return jumps, and UTC daily coverage.

## Features

`deriv_vol_lab.features` provides causal, pandas-index-aligned indicators:
Wilder RSI, EMA, ATR, Bollinger z-score, annualized close-to-close and
Parkinson volatility, rolling autocorrelation for lags 1–10, rolling Hurst
estimates (rescaled range and variance ratio), and rolling return skew and
Fisher excess kurtosis. Rolling features return `NaN` until their full window
is available. Volatility annualization uses a 365-day year and the supplied
sampling granularity in seconds.

`screen_candles` builds the latest-candle screener with descriptive flags and
per-flag forward-return reliability fields. Candle epochs must be unique; gaps
split the data into contiguous segments so rolling features and outcomes do not
span missing candles. Realized volatility is annualized using the observed
elapsed seconds in each rolling window.

The default chronological partition of the requested lookback is 60%
calibration, 10% later holdout evaluation, and 30% current display. The latest
contiguous segment within calibration supplies historical flag edges and
exploratory p-values, so calibration outcomes do not bridge gaps. The holdout
is reserved and does not affect either reliability or displayed current features;
the final display window supplies current screen values. Unadjusted p-values
are explicitly named `*_p_value_unadjusted_exploratory`; adjusted values use
Benjamini-Hochberg across all available flag/symbol tests in the screen result.
This correction does not remove other multiple-testing or data-snooping risks.
Stored candles can be screened with
`python -m deriv_vol_lab screen --granularity 60`; use `--latest`, `--seed`, and
`--permutations` to control its lookback and reproducible reliability test. The
CLI explains the temporal separation and exploratory/adjusted p-value labels.

## Statistical diagnostics

`deriv_vol_lab.stats` provides rolling realized-versus-stated volatility
confidence intervals, Ljung-Box, runs, variance-ratio, and ARCH-LM tests with
Benjamini-Hochberg adjustment across symbols and tests, seeded driftless GBM
null paths, and multi-horizon signal evaluation. Signal evaluation excludes
zero-return ties from binomial hit trials, reports tie counts and effective
sample size, and uses circular-shift inference that preserves signal ordering.
For horizons longer than one candle, inferential tests use non-overlapping
decisions; the binomial p-value is reported as descriptive and explicitly
marked non-inferential.
Optionally, a seeded stationary block bootstrap estimates a dependence-aware
mean-return p-value. Rolling windows, symbols, tests, signals, and horizons all
create multiple-testing risk; the statistical API docstrings describe where
correction is applied and where further family-wide adjustment remains
necessary.

## Event-driven backtesting

`deriv_vol_lab.backtest` supports configurable multiplier and fixed-payout
contracts. Candle epochs are UTC epoch seconds at candle open; a signal becomes
available at that candle's close (`epoch + granularity`), enters at the next
contiguous candle's open, and exits at the execution candle's close (or at a
configured stop-out). Gaps split execution segments, are not traded across, and
are counted in the report. The annualized Sharpe uses stake-normalized,
time-aligned per-candle returns including flat and missing-candle periods, with
`365*24*3600/granularity` periods per year. Reports also include per-trade mean
and standard deviation of net PnL, max drawdown, expectancy, net win rate, and
payout-implied break-even rate, plus expanding-window walk-forward
out-of-sample summaries. Pooled walk-forward reports reject overlapping test
windows; `walk_forward_splits` can still generate overlapping splits for
separate use.
Deflated Sharpe and White's Reality Check helpers address strategy-search
bias; neither removes the need for pre-specified hypotheses and untouched
out-of-sample evaluation.

## Risk analysis

`deriv_vol_lab.risk` provides volatility-targeted, fixed-fractional, and
fractional-Kelly sizing. Maximum modeled loss includes the product stop-out,
spread, and commission; configurable risk caps apply before the final stake is
returned. Volatility targeting uses each symbol's registered stated annualized
volatility and is intended for multiplier contracts. Kelly sizing requires
out-of-sample estimates of win probability and net win/loss ratio; it does not
infer a profitable edge from the driftless null.

`monte_carlo_risk_report` simulates seeded, driftless price paths at the
registered one- or two-second tick cadence, aggregates trades to valid
60-second horizons, and reports risk of reaching a configured equity floor
plus the 95th/99th percentiles of each path's worst drawdown. Simulated
directions are independent random long/short choices and stake is fixed.
`print_risk_report` writes these estimates to standard output; tail estimates
are model-based diagnostics, not guarantees. `LossCircuitBreaker` tracks
realized net loss by UTC day and consecutive losing trades. Daily loss resets
at UTC midnight; a consecutive-loss halt persists until explicitly reset.

`portfolio_risk_view` accepts time-aligned returns for all ten registered
indices, indexed by increasing UTC epoch seconds. It reports the empirical
cross-symbol correlation matrix and compares it with a theoretical annual
portfolio volatility calculated under the independent-process assumption.
Rows with missing returns are excluded rather than forward-filled; finite
sample correlations are diagnostics and can deviate from zero by chance.

## Snapshot API and dashboard

The FastAPI read-only app is `deriv_vol_lab.api.app:app` and can be run locally
with `uvicorn deriv_vol_lab.api.app:app`. It exposes `GET /screen`,
`GET /health`, `GET /data-quality`, and `GET /stats/vol-monitor`; the root path
serves the sortable descriptive dashboard. Each endpoint reads the same
validated JSON snapshot and performs no WebSocket, DuckDB, or background-worker
work while serving requests. `SNAPSHOT_PATH` defaults to
`snapshots/latest.json`.

Market-data ingestion is a separate short-lived process:

```sh
DERIV_APP_ID=your-public-app-id python -m deriv_vol_lab ingest-snapshot --days 2
```

The `Scheduled analytics snapshot` GitHub Actions workflow runs every six hours
and can also be started manually. Configure the repository variable
`DERIV_APP_ID` (public market-data access only) and allow the workflow to write
to the default branch. It fetches recent 60-second candles over Deriv's public
WebSocket API, computes screen/quality/volatility-monitor views, and commits the
latest JSON snapshot. Connecting the repository to Vercel causes that snapshot
commit to redeploy the read-only Python function; there is no persistent
WebSocket worker and the deployed function never mutates storage. The generated
snapshot contains public market data and analytics and is committed to the
repository, so do not add secrets or private data to it. The first dashboard
deployment reports a degraded health state until the scheduled job publishes
its first snapshot.

The volatility monitor uses a 500-return 99% chi-square window with a
60-second sampling interval (the candle cadence, not each symbol's faster
native tick cadence). The endpoint serves the latest window per symbol. A
missing window means there is not yet sufficient snapshot history; it is not
interpreted as a passing volatility check.
