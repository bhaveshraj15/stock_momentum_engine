# Stock Momentum Engine

A systematic momentum screener for equities and ETFs. Fetches price data, applies a configurable stack of gate filters and scoring filters, diversifies the output, and produces a ranked list of picks — automatically every Sunday via GitHub Actions.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Methodology](#methodology)
- [Getting Started](#getting-started)
- [Running the Pipeline](#running-the-pipeline)
- [CLI Reference](#cli-reference)
- [Filters Reference](#filters-reference)
- [Universes](#universes)
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
    TrendFilter       → Close >= EMA100 >= EMA200
    High52wFilter     → within 20% of 52-week high
    MinReturnFilter   → 1-year return >= 6.5%
    UpDaysFilter      → >= 50% up days in last 6 months
    VolumeFilter      → avg daily volume >= 100k
    ↓
Scoring Filters — continuous signal (higher = better)
    MomentumFilter    → weighted multi-period return/sharpe/sortino
    VolumeFilter      → volume trend multiplier (weight 0.3)
    ↓
Min-max normalise → weighted average → final_score → rank
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

1. **Trend confirmation** — the stock must be in a medium and long-term uptrend (EMA alignment)
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

Each scoring mode runs across multiple lookback periods (default: 1m, 3m, 6m, 12m) and combines them into a weighted average. This captures both short-term momentum and longer-term trend persistence.

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

# Custom lookbacks with front-weighted scoring
python pipeline.py --lookbacks 1 3 6 12 --weights 4 3 2 1

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
| `--lookbacks` | `1 3 6 12` | Lookback periods in months |
| `--weights` | equal | Weight per lookback period |
| `--no-volume-score` | off | Disable volume scorer (on by default at 0.3 weight) |
| `--volume-weight` | `0.3` | Weight of volume scorer relative to momentum |

### Gate Filters

| Flag | Default | Description |
|------|---------|-------------|
| `--no-gates` | off | Skip all gate filters |
| `--no-volume-gate` | off | Skip avg-volume gate |
| `--min-volume` | `100000` | Minimum average daily volume |
| `--volume-confirm` | off | Add volume buildup gate (opt-in) |

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

| Filter | Default Params | What It Checks |
|--------|---------------|----------------|
| `TrendFilter` | EMA 100/200 | Close >= EMA100 >= EMA200 |
| `High52wFilter` | window=252, proximity=0.80 | Close within 20% of 52-week high |
| `MinReturnFilter` | min_return=6.5%, window=252 | 1-year return >= threshold |
| `UpDaysFilter` | min_up_pct=50%, window=126 | >= 50% positive days in 6 months |
| `VolumeFilter (gate)` | min_avg_vol=100k, window=252 | Average daily volume >= minimum |
| `VolumeFilter (confirm)` | window=252 | vol_21d > vol_63d > vol_252d (opt-in) |

All gates use AND logic — a ticker must pass every gate to reach the scorer. Tickers with insufficient price history (< required window) are excluded rather than using partial data.

### Scoring Filters (continuous)

| Filter | Weight | What It Scores |
|--------|--------|---------------|
| `MomentumFilter` | 0.7 | Multi-period return / sharpe / sortino |
| `VolumeFilter (score)` | 0.3 | Volume trend confirmation multiplier |

Scorer outputs are min-max normalised to [0, 1] before combining. Weights are applied per-ticker using only the scorers that produced a valid signal — a ticker missing one scorer's data is not penalised against tickers with full coverage.

### Post-Scoring Filters

| Filter | Default Params | What It Does |
|--------|---------------|--------------|
| `DiversificationFilter` | threshold=0.85, window=126 | Greedy rank-aware deduplication |
| `CorrelationFilter` | ccp=1.75e-4, window=252 | Universe pre-screen (opt-in, large universes) |

---

## Universes

Universes are defined as YAML files in `config/universes/`. Each file specifies a name and a list of ticker symbols.

```yaml
name: Nifty 50
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

1. Create `config/universes/my_universe.yaml`
2. Add `name` and `tickers` (use yfinance ticker format, e.g. `.NS` for NSE, `.DE` for Deutsche Börse)
3. Run: `python pipeline.py --universe config/universes/my_universe.yaml`

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
| Risk-free rate | 6.5% p.a. | Matches Indian T-bill rate at notebook creation |
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
├── pipeline.py                  Main entry point
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
│   ├── high52w_filter.py        52-week high proximity gate
│   ├── min_return_filter.py     Minimum return gate
│   ├── up_days_filter.py        Up-days breadth gate
│   ├── volume_filter.py         Volume gate / confirm / scorer
│   ├── momentum_filter.py       Multi-period momentum scorer
│   ├── correlation_filter.py    Universe pre-screen
│   └── diversification_filter.py Post-scorer deduplication
├── scoring/
│   └── scorer.py                Orchestrates gates + scorers → ranked output
├── output/
│   ├── exporter.py              CSV and numpy export
│   └── runs/                    Output files (gitignored except on picks/weekly)
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
