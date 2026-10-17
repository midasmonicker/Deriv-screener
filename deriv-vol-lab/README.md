# deriv-vol-lab

Research and analytics tools for Deriv Volatility Indices.

## Research findings

The central question for this project was whether the screener's descriptive
flags (overbought, stretched, volatility mismatch, trending and mean-reverting
regimes) carry any tradable edge on Deriv's volatility indices. On about 60
days of stored 60-second candles for all thirteen indices (86,400 candles each),
the answer is **no detectable edge**:

- 1,120 hypothesis tests (flag x horizon x direction x symbol/pooled x
  discovery/confirmation) produced **0 confirmed pooled edges** and 0
  Bonferroni discoveries. The best single-symbol corrected p-value was 0.45 and
  the best pooled confirmation q-value was 0.68.
- The method was validated first: on 20 seeded driftless random-walk datasets
  run through the same pipeline, 4.87% of raw tests passed p < 0.05 (about 5%
  expected) and 0.00% survived Benjamini-Hochberg correction.
- A fixed-payout rise/fall check (`scripts/payout_breakeven.py`) compared each
  row's win rate with the break-even rate `1 / (1 + payout)`. At assumed payouts
  of 90%, 95% and 100% no row beat break-even after correction. Payouts are
  assumptions, not broker quotes; verify them against live prices.

Limits: one roughly 60-day stretch of one synthetic process; rare flags (such as
volatility mismatch at long horizons) have too few independent trades to detect
any realistic edge; costs and spreads were set to zero, which favours the
flags; and failing to detect an edge is not proof that none exists. Treat the
screener as descriptive analytics on random processes, not a source of signals.

The same evidence is served by `GET /research/flag-edge` and shown on the
dashboard. To refresh it after a new report:

```sh
python scripts/flag_edge_report.py
python scripts/payout_breakeven.py reports/flag_edge_YYYYMMDD.csv
python scripts/build_research_summary.py reports/flag_edge_YYYYMMDD.csv
```

The last command rewrites `src/deriv_vol_lab/api/research_summary.json`, which
ships with the package.

## Setup

Requires Python 3.11 or newer.

```sh
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Set `DB_PATH` and `LOG_LEVEL` in the environment as needed. `DERIV_APP_ID` is
optional; when supplied, the client sends it in the `Deriv-App-ID` header.
`DERIV_ENDPOINT` defaults to
`wss://api.derivws.com/trading/v1/options/ws/public`.
`DERIV_HANDSHAKE_MAX_ATTEMPTS` defaults to 5 and bounds initial WebSocket
handshake retries.
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

`deriv_vol_lab.data.DerivWebSocketClient` connects to Deriv's public
market-data WebSocket at `api.derivws.com` without requiring an application ID.
When provided, the ID is sent as the `Deriv-App-ID` handshake header, never as
a query parameter. It validates the local volatility-index registry against
`active_symbols` using `underlying_symbol`, pages historical ticks or
candles backwards in requests of at most 5,000 records, and exposes live tick
and OHLC subscriptions as async streams. Closed clients raise
`ClientClosedError`; an interrupted request raises retryable
`ConnectionLostError`. Failed resubscriptions deliver their terminal error once
and then end the stream. Unexpected connection-manager failures are logged,
reported to in-flight callers, and retried with the configured backoff.
Tick-history responses require both `history.times` and `history.prices`.
Historical candles receive their requested granularity because the public
response omits it; live OHLC updates use `open_time` as the candle epoch.

### Resumable candle backfill

Fetch and store the last 60 days of 60-second candles for all registered
volatility indices using the public market-data WebSocket:

```sh
python scripts/backfill.py
python scripts/backfill.py --days 90
python scripts/backfill.py --days 60 --max-seconds 1800
```

The command resumes around the latest stored candle with a two-candle overlap,
checks stored gaps, and writes only missing epochs. History pages are traversed
backwards by candle epoch; short non-empty pages do not terminate pagination.
Each symbol's output includes gaps, duplicate epochs, and daily coverage. A
failed WebSocket handshake or an expired `--max-seconds` deadline exits with
status 2. The backfill is manual and makes read-only market-data requests; it
does not use trade or payment scopes.

## Storage and quality checks

`deriv_vol_lab.data.DuckDBStorage` persists ticks in `ticks` and candles in
granularity-specific tables such as `candles_60`. Both use `(symbol, epoch)`
keys with idempotent upserts. Use `export_parquet("ticks", path)` or pass a
candle granularity to export a table to Parquet. `find_gaps` reports expected
cadence-aligned epochs; `await backfill_gaps(client, ...)` fetches and stores
the missing history. `quality_report(records=raw_observations)` checks raw
input order and duplicate epochs before the unique-key upsert, as well as
non-positive prices, >8-sigma local log-return jumps, and UTC daily coverage.

### Live volatility smoke check

After installing the project, run the network smoke check manually:

```sh
python scripts/smoke_live.py
# Optionally choose a different backfill period
python scripts/smoke_live.py --days 3
```

It uses only Deriv's public market-data WebSocket API (no auth token or trading
scopes; an application ID is optional), backfills seven days of 60-second
candles for every registered volatility symbol present in `active_symbols`,
skips symbols not returned by the endpoint, and stores data in the local DuckDB file
configured by `DB_PATH` (default `data/deriv-vol-lab.duckdb`). It prints
per-symbol missing-candle counts, coverage, and the latest valid realized-to-
stated volatility ratio with its 99% confidence interval. The breach-rate test
uses non-overlapping windows (step equal to the monitor window), a one-sided
binomial test against a 1% breach rate, and Benjamini-Hochberg adjustment
across symbols at 1%. The latest-window test is also BH-adjusted across
symbols at 1%. The overlapping-window breach fraction is still printed as
“overlapping, informational” and does not affect the exit status. A warning is
printed when fewer than ten non-overlapping windows are available. Coverage
below 99% and insufficient volatility history remain failures. If all
handshake attempts fail, it prints the HTTP status and endpoint host on one
line and exits with code 2. This intentionally networked check is manual and
is not part of CI. The output also includes the aligned pairwise correlation
matrix for 1-minute log returns, the maximum absolute correlation and its
`4/sqrt(n)` reference threshold, and the number of symbol pairs with realized-
volatility ratios equal to three decimals. These cross-symbol diagnostics are
informational and never change the exit code.

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
elapsed seconds in each rolling window. Reliability uses the same cost-aware
forward-trade calculation as the backtester: a signal known at candle close
enters at the next candle's open and exits at the close of candle `t+h`; product
costs and stop-out terms are supplied through `ProductSpec`.

The requested lookback is partitioned chronologically at floored integer
cutoffs: `calibration_end = floor(N * calibration_fraction)` and
`evaluation_end = floor(N * (calibration_fraction + evaluation_fraction))`.
The default fractions are 60% calibration, 10% later holdout evaluation, and
the remainder current display; reported candle counts are the resulting
integer slice lengths and can differ fractionally from those percentages for
small or non-divisible lookbacks. The latest contiguous segment within
calibration supplies historical flag edges and
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

### Offline flag-edge report

Measure whether any existing screener flag has a repeatable edge using only
locally stored 60-second candles:

```sh
python scripts/flag_edge_report.py
python scripts/flag_edge_report.py --db data/deriv-vol-lab.duckdb --symbols R_75 1HZ75V
python scripts/flag_edge_report.py --horizons 1 5 15 60 --alpha 0.05 --seed 12345
python scripts/flag_edge_report.py --commission-per-trade 0.01 --spread-fraction 0.002
python scripts/flag_edge_report.py --jobs 2
python scripts/flag_edge_report.py --quick
```

This command makes no network or WebSocket requests and opens DuckDB read-only.
It requires at least 2,000 stored candles per selected symbol; `--symbols`
defaults to all thirteen registered indices. Signals use the screener's causal
flag definitions. Forward trades enter at the next candle open and exit at the
close of candle `t+h`; the shared backtester applies the selected `ProductSpec`
and suppresses trades across gaps. Its default terms match the screener's
current multiplier default (stake 1, multiplier 1, no commission or spread).
Use `--stake`, `--multiplier`, `--commission-per-trade`, `--spread-fraction`,
and `--stop-out-fraction` to supply different ProductSpec terms; they are
printed in the report.

The first 70% of each symbol's candles are discovery and the final 30% are
untouched confirmation. Circular-shift tests are the primary inference;
binomial hit-rate tests are reported and separately BH-adjusted. For horizons
longer than one candle, inference uses a non-overlapping decision sample.
Stationary-bootstrap intervals and an 80%-power minimum detectable hit-rate
deviation accompany each test. A 20-replication driftless-GBM null control runs
through the same pipeline and exits with code 3 if more than 2x alpha of its
tests survive BH; missing or undersized stored history exits with code 2.
Progress, elapsed time, and per-replication ETA are flushed to stderr. Null
replications run in separate processes; `--jobs` defaults to CPU count minus
one, and each replication's seed is independent of worker assignment. Even
quick mode enforces the 2,000-permutation and 20-replication minima.

`--quick` uses the first two selected symbols and first two selected horizons,
skips null-control inference, and writes only `quick_*.md`. The header marks it
`NOT VALID EVIDENCE`; it is for checking the pipeline, not interpreting results.

The dated CSV contains every symbol and pooled test row. The Markdown report
lists data windows, the null result, the strongest pooled tests, and only edges
that are significant in discovery and replicate with positive net returns in
confirmation. Results are descriptive analytics on random processes; a
non-significant result means no edge was detectable at this sample size, not
that an edge is proven absent. This report does not imply predictive power.

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
are counted in the report. Annualized Sharpe uses stake-normalized,
per-candle returns for actually evaluated test candles, including flat
candles; missing candles and gaps between non-adjacent walk-forward folds are
not treated as evaluated periods.
`365*24*3600/granularity` periods per year is applied to those evaluated-candle
statistics. Pooled drawdown is reported as the worst drawdown among individual
test folds, without stitching their equity curves. Reports also include per-trade mean
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

`portfolio_risk_view` accepts time-aligned returns for all thirteen registered
indices, indexed by increasing UTC epoch seconds. It reports the empirical
cross-symbol correlation matrix and compares it with a theoretical annual
portfolio volatility calculated under the independent-process assumption.
Rows with missing returns are excluded rather than forward-filled; finite
sample correlations are diagnostics and can deviate from zero by chance.

## Snapshot API and dashboard

The FastAPI read-only app is `deriv_vol_lab.api.app:app` and can be run locally
with `uvicorn deriv_vol_lab.api.app:app`. It exposes `GET /screen`,
`GET /health`, `GET /data-quality`, `GET /stats/vol-monitor`, and
`GET /research/flag-edge` (the frozen offline research evidence); the root path
serves the sortable descriptive dashboard. Each endpoint reads the same
validated JSON snapshot and performs no WebSocket, DuckDB, or background-worker
work while serving requests. `SNAPSHOT_PATH` defaults to
`snapshots/latest.json`.

Market-data ingestion is a separate short-lived process:

```sh
python -m deriv_vol_lab ingest-snapshot --days 2
```

The `Scheduled analytics snapshot` GitHub Actions workflow runs every six hours
and can also be started manually. An application ID is optional; when
configured as the `DERIV_APP_ID` repository variable, it is sent in the
`Deriv-App-ID` handshake header. Allow the workflow to write to the default
branch. It fetches recent 60-second candles over Deriv's public
WebSocket API, computes screen/quality/volatility-monitor views, and commits the
latest JSON snapshot. Connecting the repository to Vercel causes that snapshot
commit to redeploy the read-only Python function; there is no persistent
WebSocket worker and the deployed function never mutates storage. The generated
snapshot contains public market data and analytics and is committed to the
repository, so do not add secrets or private data to it. The first dashboard
deployment reports a degraded health state until the scheduled job publishes
its first snapshot.

The volatility monitor uses a 500-valid-return 99% chi-square window with a
60-second sampling interval (the candle cadence, not each symbol's faster
native tick cadence). Returns spanning missing candles are excluded rather
than annualized at the candle rate. The endpoint serves the latest valid
window per symbol, or an `insufficient_data` status with no alert when fewer
than 500 valid returns are available; insufficient history is not interpreted
as a passing volatility check.
