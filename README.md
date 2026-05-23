# Stock Momentum Engine

A systematic momentum screener for equities and ETFs. Fetches price data, applies a configurable stack of gate filters and scoring filters, and produces a ranked list of picks — automatically every Sunday via GitHub Actions.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [CLI Reference](#cli-reference)
- [Filters Reference](#filters-reference)
- [Universes](#universes)
- [Backtesting](#backtesting)
- [Weekly Automation](#weekly-automation)
- [Known Limitations](#known-limitations)
- [Assumptions](#assumptions)
- [Contributing](#contributing)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

---

## How It Works

```
Universe (YAML)
    ↓
Fetcher — downloads OHLCV price data (yfinance, cached)
    ↓
Gate Filters — hard boolean pass/fail (AND logic)
    TrendFilter       → Close >= EMA50 >= EMA100
    MinReturnFilter   → 1-year return >= rfr (per-universe)
    ↓
Scoring Filters — continuous signal (higher = better)
    MomentumFilter    → sum-of-ordinal-ranks across 3m/6m/9m/12m periods
    VolumeFilter      → volume trend multiplier (weight 0.3)
    ↓
Percentile-rank normalise → weighted average → final_score → rank
    ↓
DiversificationFilter — greedy rank-aware deduplication
    Remove correlated near-clones, keep the higher-ranked one
    ↓
Ranked picks → CSV + Markdown report
```

---

## Methodology

### Why Momentum?

Momentum — the tendency of assets that have performed well recently to continue outperforming — is one of the most robust and well-documented anomalies in financial markets. First formalised by Jegadeesh & Titman (1993), it has been replicated across asset classes, geographies, and time periods.

This engine implements a systematic momentum strategy with three layers of signal validation:

1. **Trend confirmation** — the stock must be above its EMA50 and EMA100 (configurable, default 50/100)
2. **Momentum scoring** — rank by risk-adjusted returns over multiple lookback periods
3. **Volume confirmation** — price moves backed by growing volume are more reliable

### Scoring Modes

**Returns mode** (`--mode returns`)
Simple percentage return over each lookback period. Fast, interpretable, matches the original notebook logic exactly.

```
score = (Close[-1] / Close[-window_start]) - 1
```

**Sharpe mode** (`--mode sharpe`)
Risk-adjusted return — divides cumulative return by annualised volatility. Penalises volatile stocks even if they have high raw returns.

```
sharpe = (sum(daily_returns) - rfr * period/12) / (std(daily_returns) * sqrt(n_days))
```

**Sortino mode** (`--mode sortino`)
Like Sharpe but only penalises downside volatility. Stocks with many small up days and few large down days score well.

```
sortino = (sum(daily_returns) - rfr * period/12) / (downside_std * sqrt(n_days))
```

### Multi-Period Scoring

Each scoring mode runs across four lookback periods (default: 3m, 6m, 9m, 12m). Rather than averaging raw scores — which lets a single strong month dominate — the engine ranks all tickers independently for each period and sums the ordinal ranks. The ticker with the lowest rank sum wins: it was consistently near the top across all timeframes, not just a flash performer in one.

```
period_scores:   3M    6M    9M   12M
  TCS          2.1   0.3   0.2   0.1   ← strong 3M, weak everywhere else
  INFY         0.8   0.7   0.8   0.7   ← consistently solid

rank each period:
  TCS          1     3     3     3     → sum = 10  (loses)
  INFY         2     1     1     1     → sum =  5  (wins)
```

### Volume Confirmation

The volume scorer computes:
```
vol_trend   = avg_volume_21d / avg_volume_252d
consistency = avg_volume_63d / avg_volume_252d
vol_score   = vol_trend * sqrt(consistency)
```

`vol_score > 1` means price move is backed by growing volume — more trustworthy momentum.
`vol_score < 1` means price move on declining volume — penalised.

Volume is weighted at 30% relative to momentum (70%) in the composite score. This breaks ties in favour of higher-volume stocks without overriding the momentum signal.

### Diversification

After scoring and ranking, a greedy rank-aware deduplication step removes correlated near-clones. If two stocks have correlation > 0.85 over the last 6 months, only the higher-ranked one survives. This prevents the portfolio from being dominated by a single sector.

---

## Getting Started

### Prerequisites

- Python 3.11+
- Internet connection (for yfinance price fetches)

### Installation

```bash
git clone https://github.com/bhaveshraj15/stock_momentum_engine.git
cd stock_momentum_engine
pip install -r requirements.txt
```

### Quick Start

```bash
# Run with all defaults — Nifty 50, returns mode, top 10
python pipeline.py

# Run on ETF India universe, sharpe mode
python pipeline.py --universe config/universes/etf_india_2026.yaml --mode sharpe

# Run all three modes on DAX 40, top 5 picks
python pipeline.py --universe config/universes/dax40.yaml --mode returns --top-n 5
python pipeline.py --universe config/universes/dax40.yaml --mode sharpe  --top-n 5
python pipeline.py --universe config/universes/dax40.yaml --mode sortino --top-n 5
```

---

## Running the Pipeline

### Basic Usage

```bash
python pipeline.py [options]
```

Output is printed to the console and saved as a CSV to `output/runs/`.

### Common Recipes

```bash
# Fresh run ignoring cache
python pipeline.py --refresh

# Only passing tickers in the CSV, top 10
python pipeline.py --passing-only --top-n 10

# Add volume buildup gate (strict — removes flat-volume stocks)
python pipeline.py --volume-confirm

# Pure momentum ranking, no volume scorer
python pipeline.py --no-volume-score

# Skip diversification
python pipeline.py --no-diversify

# Skip all gate filters (score the entire universe)
python pipeline.py --no-gates

# Custom lookback periods
python pipeline.py --lookbacks 3 6 9 12

# Correlation pre-filter for large universes
python pipeline.py --universe config/universes/etf_india_2026.yaml --corr-filter
```

---

## CLI Reference

### Universe & Data

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `config/universes/nifty50.yaml` | Path to universe YAML |
| `--lookback-days` | `730` | Days of price history to fetch |
| `--refresh` | `false` | Force re-download, ignore cache |
| `--cache-dir` | `data/cache` | Cache directory |

### Scoring

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `returns` | Scoring mode: `returns`, `sharpe`, `sortino` |
| `--lookbacks` | `3 6 9 12` | Lookback periods in months |
| `--no-volume-score` | off | Disable volume scorer (on by default at 0.3 weight) |
| `--volume-weight` | `0.3` | Weight of volume scorer relative to momentum |

### Gate Filters

| Flag | Default | Description |
|------|---------|-------------|
| `--no-gates` | off | Skip all gate filters |
| `--ema-fast` | `50` | Fast EMA span for TrendFilter |
| `--ema-slow` | `100` | Slow EMA span for TrendFilter |
| `--volume-confirm` | off | Add volume buildup gate: vol_21d > vol_63d > vol_252d (opt-in) |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `csv` | Output format: `csv`, `numpy` |
| `--output-dir` | `output/runs` | Output directory |
| `--top-n` | all | Print/save top N tickers |
| `--passing-only` | off | CSV contains only gate-passing tickers |

### Post-Processing

| Flag | Default | Description |
|------|---------|-------------|
| `--no-diversify` | off | Skip diversification filter (on by default) |
| `--div-threshold` | `0.85` | Max correlation between final picks |
| `--max-picks` | none | Hard cap on final output size |
| `--corr-filter` | off | Apply correlation pre-filter (large universes) |
| `--ccp` | `1.75e-4` | Correlation cap parameter for `--corr-filter` |

### Misc

| Flag | Default | Description |
|------|---------|-------------|
| `-v, --verbose` | off | Enable debug logging |

---

## Filters Reference

### Gate Filters (pass/fail)

Default gate stack (2 filters, AND logic):

| Filter | Params | What It Checks |
|--------|--------|----------------|
| `TrendFilter` | EMA 50/100 (configurable) | Close >= EMA50 >= EMA100 |
| `MinReturnFilter` | min_return=rfr (from universe YAML) | 1-year return >= threshold |

Additional filters available but not in defaults:

| Filter | What It Checks |
|--------|----------------|
| `High52wFilter` | Close within 20% of 52-week high |
| `UpDaysFilter` | >= 50% positive days in last 6 months |
| `VolumeFilter (confirm)` | vol_21d > vol_63d > vol_252d — opt-in via `--volume-confirm` |

All gates use AND logic. Tickers with insufficient price history are excluded rather than using partial data.

### Scoring Filters (continuous)

| Filter | Weight | What It Scores |
|--------|--------|---------------|
| `MomentumFilter` | 0.7 | Sum-of-ordinal-ranks across 3m/6m/9m/12m periods |
| `VolumeFilter (score)` | 0.3 | Volume trend confirmation multiplier |

Scorer outputs are percentile-rank normalised to (0, 1] before combining — outlier-immune and scale-independent. Weights are applied per-ticker using only scorers that produced a valid signal.

### Post-Scoring Filters

| Filter | Default Params | What It Does |
|--------|---------------|--------------|
| `DiversificationFilter` | threshold=0.85, window=126 | Greedy rank-aware deduplication |
| `CorrelationFilter` | ccp=1.75e-4, window=252 | Universe pre-screen (opt-in, large universes) |

---

## Universes

Universes are defined as YAML files in `config/universes/`. Each file specifies a name and a list of ticker symbols.

```yaml
meta:
  name: "Nifty 50"
  exchange: "NSE"
  currency: "INR"
  rfr: 0.065       # risk-free rate — used in Sharpe/Sortino and gate filter
  suffix: ".NS"

tickers:
  - RELIANCE.NS
  - TCS.NS
  - HDFCBANK.NS
  ...
```

### Available Universes

| File | Description |
|------|-------------|
| `nifty50.yaml` | Nifty 50 — Indian large-cap equities |
| `etf_india_2026.yaml` | Indian ETF universe |
| `dax40.yaml` | DAX 40 — German large-cap equities |

### Adding a New Universe

1. Copy `config/universes/universe_template.yaml` and rename it
2. Fill in the `meta` block: `name`, `exchange`, `currency`, `rfr`, `suffix`
3. Add your tickers (inline list or `csv_source`)
4. Run: `python pipeline.py --universe config/universes/my_universe.yaml`

---

## Backtesting

The backtest engine replays the momentum strategy on historical data using a strict walk-forward simulation — no lookahead, no future prices ever touch the scoring pipeline.

### How It Works

```
For each rebalance date R:
    prices.loc[:R]  ← slice to R only (zero lookahead)
        ↓
    Full scoring pipeline (same gates + scorers as pipeline.py)
        ↓
    DiversificationFilter → top_n picks
        ↓
    Portfolio.rebalance_to(picks, prices_on_R, R)

Every trading day:
    Portfolio.update_prices(day_prices, day)  ← mark to market
        ↓
Equity curve = NAV history / initial_cash × 100  (indexed to 100)
```

### Running a Backtest

```bash
# Basic run — Nifty 50, 2018–2024, monthly rebalance
python backtest_run.py \
    --universe config/universes/nifty50.yaml \
    --start 2018-01-01 \
    --end   2024-12-31

# Sharpe mode, weekly rebalance, top 5 picks
python backtest_run.py \
    --universe config/universes/nifty50.yaml \
    --start 2018-01-01 --end 2024-12-31 \
    --mode sharpe --rebalance weekly --top-n 5

# No gate filters, no diversification — score the full universe
python backtest_run.py \
    --start 2018-01-01 --end 2024-12-31 \
    --no-gates --no-diversify

# Save all outputs to CSV
python backtest_run.py \
    --start 2018-01-01 --end 2024-12-31 \
    --output csv
```

### Backtest CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `config/universes/nifty50.yaml` | Path to universe YAML |
| `--start` | required | Backtest start date `YYYY-MM-DD` |
| `--end` | required | Backtest end date `YYYY-MM-DD` |
| `--initial-cash` | `1,000,000` | Starting cash for portfolio simulation |
| `--top-n` | `10` | Tickers to hold per period |
| `--rebalance` | `monthly` | `weekly`, `monthly`, `quarterly` |
| `--mode` | `returns` | Scoring mode: `returns`, `sharpe`, `sortino` |
| `--lookbacks` | `3 6 9 12` | Lookback periods in months |
| `--lookback-days` | `730` | Days of price history fed to scorer |
| `--ema-fast` | `50` | Fast EMA span for TrendFilter |
| `--ema-slow` | `100` | Slow EMA span for TrendFilter |
| `--no-gates` | off | Skip all gate filters |
| `--volume-confirm` | off | Add volume buildup gate (opt-in) |
| `--no-volume-score` | off | Disable volume scorer |
| `--volume-weight` | `0.3` | Weight of volume scorer |
| `--no-diversify` | off | Skip diversification filter |
| `--div-threshold` | `0.85` | Max correlation between final picks |
| `--max-picks` | none | Hard cap on final output after diversification |
| `--benchmark` | none | Benchmark ticker (e.g. `^NSEI`, `^GDAXI`) |
| `--cache-dir` | `data/cache` | Cache directory |
| `--output` | none | `csv` — save all outputs to `output/backtests/` |
| `-v, --verbose` | off | Print rebalance blocks with buys/sells |

### Metrics

| Metric | Description |
|--------|-------------|
| Total Return | `(Final NAV / Initial NAV) - 1` |
| CAGR | Compound annual growth rate |
| Max Drawdown | Peak-to-trough decline (worst case) |
| Sharpe Ratio | Annualised excess return / annualised volatility |
| Calmar Ratio | CAGR / \|Max Drawdown\| |

### Output Files (with `--output csv`)

```
output/backtests/
  backtest_<universe>_<ts>_equity.csv        — daily NAV equity curve
  backtest_<universe>_<ts>_rebalances.csv    — rebalance history with picks
  portfolio_<ts>_holdings.csv               — current open positions
  portfolio_<ts>_transactions.csv           — all buys and sells
  portfolio_<ts>_closed_positions.csv       — closed round-trips with P&L
  portfolio_<ts>_nav_history.csv            — daily NAV snapshots
```

---

## Weekly Automation

The engine runs automatically every Sunday at 5:00pm IST via GitHub Actions, producing picks for Monday morning.

### What Runs

- 3 universes × 3 modes = 9 pipeline runs
- Top 10 picks per run
- Cache deleted after all runs (fresh fetch next week)
- Results committed to `picks/weekly` branch (main stays clean)

### Output on `picks/weekly`

```
output/runs/
  nifty_50_20260427_returns.csv
  nifty_50_20260427_sharpe.csv
  nifty_50_20260427_sortino.csv
  etf_india_2026_20260427_returns.csv
  ...
  picks_20260427.md        ← full report, all universes
```

### Manual Trigger

Go to **Actions** → **Weekly Momentum Picks** → **Run workflow**. You can select:
- Which universes to run
- Which modes to run
- How many top picks

### Viewing Picks

Switch to the `picks/weekly` branch on GitHub and open `picks_YYYYMMDD.md` for the full report.

---

## Known Limitations

**Survivorship bias**
Universe files contain current index constituents. Stocks that were delisted or removed from the index during the backtest period are not included, which overstates historical performance.

**Look-ahead bias**
All filters use data available at the time of scoring. However, yfinance adjusted close prices are adjusted retroactively for splits and dividends — historical prices may differ from what was available in real time.

**Data quality**
yfinance is not a professional data source. Missing bars, incorrect splits, and stale prices are possible, especially for less liquid tickers. Always sanity-check picks before trading.

**Execution assumptions**
The engine assumes you can buy at Monday's open at the price seen on Sunday. In reality, gaps, liquidity constraints, and market impact will affect execution, especially for smaller ETFs.

**No position sizing**
The engine ranks picks but does not size positions. Equal weighting is assumed. Risk-based sizing (e.g. volatility targeting, Kelly) is not implemented.

---

## Assumptions

| Assumption | Value | Reason |
|------------|-------|--------|
| Risk-free rate | per-universe YAML (`rfr`) | INR=6.5%, EUR=4%, USD=5% — set in universe metadata |
| Trading days per year | 252 | Market standard |
| Trading days per month | 21 | 252 / 12 approximation |
| Minimum volume | 100,000 | Liquidity floor from original notebook |
| Diversification threshold | 0.85 | Standard near-clone removal |
| Volume scorer weight | 0.3 | Confirmation signal, not primary signal |

---

## Contributing

### Adding a New Filter

All filters inherit from `BaseFilter` (`filters/base_filter.py`). Implement two methods:

```python
from filters.base_filter import BaseFilter
import pandas as pd

class MyFilter(BaseFilter):

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        # Return a float score per ticker (higher = better)
        # Return NaN for tickers that cannot be scored
        ...

    def filter(self, scores: pd.Series) -> pd.Series:
        # Return True for tickers that pass, False for those that don't
        # For pure scorers (no hard gate), return all True
        ...
```

Then register it in `filters/__init__.py` and wire it into `pipeline.py`.

### Code Conventions

- All filter params go through `self.params` dict — no hardcoded values in logic
- Insufficient history → return `NaN`, never fall back to partial data
- Gate filters override `apply()` directly; scorers use `compute()` + `filter()`
- Log at `DEBUG` for per-ticker detail, `INFO` for summary counts

---

## Project Structure

```
stock_momentum_engine/
├── pipeline.py                  Screener — fetch, filter, score, export
├── backtest_run.py              Single backtest run CLI
├── config/
│   └── universes/               Universe YAML files
│       ├── nifty50.yaml
│       ├── etf_india_2026.yaml
│       └── dax40.yaml
├── data/
│   ├── fetcher.py               yfinance downloader with caching
│   ├── loader.py                Universe YAML loader
│   └── cache/                   Cached price data (gitignored)
├── filters/
│   ├── base_filter.py           Abstract base class
│   ├── trend_filter.py          EMA alignment gate
│   ├── high52w_filter.py        52-week high proximity gate (opt-in)
│   ├── min_return_filter.py     Minimum return gate
│   ├── up_days_filter.py        Up-days breadth gate (opt-in)
│   ├── volume_filter.py         Volume gate / confirm / scorer
│   ├── momentum_filter.py       Sum-of-ranks momentum scorer
│   ├── correlation_filter.py    Universe pre-screen
│   └── diversification_filter.py Post-scorer deduplication
├── scoring/
│   └── scorer.py                Orchestrates gates + scorers → ranked output
├── backtest/
│   ├── engine.py                Walk-forward backtest engine
│   ├── portfolio.py             Position sizing, execution, NAV tracking
│   ├── metrics.py               CAGR, Sharpe, Calmar, drawdown
│   ├── result.py                Result container — summary(), to_csv()
│   └── reporter.py              Verbose rebalance block formatter
├── output/
│   ├── exporter.py              CSV and numpy export
│   ├── runs/                    Screener output (gitignored except picks/weekly)
│   └── backtests/               Backtest output CSVs (gitignored)
├── scripts/
│   └── generate_report.py       Markdown report generator
└── .github/
    └── workflows/
        └── weekly_picks.yml     Sunday automation workflow
```

---

## Acknowledgements

- **yfinance** — price data
- **pandas / numpy** — data processing
- **Jegadeesh & Titman (1993)** — "Returns to Buying Winners and Selling Losers" — foundational momentum paper
- **Asness, Moskowitz & Pedersen (2013)** — "Value and Momentum Everywhere" — cross-asset momentum evidence
- **FactorLab Capital Services** ([factorlab.smallcase.com](https://factorlab.smallcase.com)) — original inspiration for the momentum factor methodology and Indian market application
