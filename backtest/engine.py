"""
backtest/engine.py
------------------
Walk-forward backtesting engine for the momentum strategy.

Algorithm
---------
1. Fetch all price history from (start - lookback_days) to end.
2. Generate rebalance dates (first trading day of each period).
3. For every rebalance date R:
       - Slice prices strictly to [:R]  ← no lookahead
       - Run the full Scorer pipeline on that slice
       - Call portfolio.rebalance_to(selected_tickers, prices_on_R, R)
4. Every trading day: portfolio.update_prices(day_prices, day)
5. Equity curve = portfolio NAV history / initial_cash.

Usage
-----
    from backtest.engine import BacktestEngine

    engine = BacktestEngine(
        universe     = "config/universes/nifty50.yaml",
        start        = "2020-01-01",
        end          = "2024-12-31",
        top_n        = 10,
        rebalance    = "monthly",
        mode         = "returns",
        initial_cash = 1_000_000,
    )
    result = engine.run()
    print(result.summary())
    print(result.portfolio.summary())
    result.to_csv()
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rebalance date generation
# ---------------------------------------------------------------------------

def _rebalance_dates(
    price_index: pd.DatetimeIndex,
    start:       pd.Timestamp,
    end:         pd.Timestamp,
    freq:        str,
) -> List[pd.Timestamp]:
    """
    First available trading day of each calendar period in [start, end].
    freq: "weekly" | "monthly" | "quarterly"
    """
    freq_map = {
        "weekly":    "W-MON",
        "monthly":   "MS",
        "quarterly": "QS",
    }
    if freq not in freq_map:
        raise ValueError(f"rebalance must be one of {list(freq_map)}. Got {freq!r}.")

    period_starts = pd.date_range(start=start, end=end, freq=freq_map[freq])

    dates = []
    for ps in period_starts:
        candidates = price_index[(price_index >= ps) & (price_index <= end)]
        if len(candidates) > 0:
            dates.append(candidates[0])

    seen, unique = set(), []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward momentum strategy backtester with Portfolio integration.

    Parameters
    ----------
    universe : str | Path
        Path to a universe YAML config.
    start : str
        Backtest start date "YYYY-MM-DD".
    end : str
        Backtest end date "YYYY-MM-DD".
    top_n : int
        Tickers to hold per period (default 10).
    rebalance : str
        "weekly" | "monthly" (default) | "quarterly".
    lookback_days : int
        Days of price history fed to the Scorer at each rebalance (default 730).
    mode : str
        Momentum scoring mode: "returns" | "sharpe" | "sortino".
    lookbacks : list[int]
        Multi-period lookbacks in months for MomentumFilter (default [3,6,9,12]).
    no_gates : bool
        Skip all gate filters.
    ema_fast : int
        Fast EMA span for TrendFilter (default 50).
    ema_slow : int
        Slow EMA span for TrendFilter (default 100).
    volume_confirm : bool
        Add VolumeFilter confirm gate (vol_21d > vol_63d > vol_252d).
    no_volume_score : bool
        Disable VolumeFilter scorer (on by default at weight 0.3).
    volume_weight : float
        Weight of VolumeFilter scorer (default 0.3).
    no_diversify : bool
        Skip diversification filter after scoring.
    div_threshold : float
        Max correlation allowed between final picks (default 0.85).
    max_picks : int | None
        Hard cap on final tickers after diversification.
    benchmark : str | None
        Optional benchmark ticker (e.g. "^NSEI", "^GDAXI").
    initial_cash : float
        Starting cash for the Portfolio simulation (default 1,000,000).
    cache_dir : str
        Directory for the yfinance parquet cache.
    """

    def __init__(
        self,
        universe:        str | Path,
        start:           str,
        end:             str,
        top_n:           int                   = 10,
        rebalance:       str                   = "monthly",
        lookback_days:   int                   = 730,
        mode:            str                   = "returns",
        lookbacks:       Optional[List[int]]   = None,
        no_gates:        bool                  = False,
        ema_fast:        int                   = 50,
        ema_slow:        int                   = 100,
        volume_confirm:  bool                  = False,
        no_volume_score: bool                  = False,
        volume_weight:   float                 = 0.3,
        no_diversify:    bool                  = False,
        div_threshold:   float                 = 0.85,
        max_picks:       Optional[int]         = None,
        benchmark:       Optional[str]         = None,
        initial_cash:    float                 = 1_000_000,
        cache_dir:       str                   = "data/cache",
        verbose:         bool                  = False,
    ):
        self.universe        = Path(universe)
        self.start           = pd.Timestamp(start)
        self.end             = pd.Timestamp(end)
        self.top_n           = top_n
        self.rebalance       = rebalance
        self.lookback_days   = lookback_days
        self.mode            = mode
        self.lookbacks       = lookbacks or [3, 6, 9, 12]
        self.no_gates        = no_gates
        self.ema_fast        = ema_fast
        self.ema_slow        = ema_slow
        self.volume_confirm  = volume_confirm
        self.no_volume_score = no_volume_score
        self.volume_weight   = volume_weight
        self.no_diversify    = no_diversify
        self.div_threshold   = div_threshold
        self.max_picks       = max_picks
        self.benchmark       = benchmark
        self.initial_cash    = float(initial_cash)
        self.cache_dir       = cache_dir
        self.verbose         = verbose

        if self.start >= self.end:
            raise ValueError(f"start ({start}) must be before end ({end}).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> "BacktestResult":  # type: ignore[name-defined]
        from backtest.result    import BacktestResult
        from backtest.metrics   import compute_metrics
        from backtest.portfolio import Portfolio

        # ── 1. Universe ───────────────────────────────────────────────
        from data.loader import UniverseLoader
        loader   = UniverseLoader(self.universe)
        tickers  = loader.get_tickers()
        currency = loader.get_currency() or "?"
        rfr      = loader.get_rfr()
        logger.info(
            "Universe: '%s' — %d tickers  currency=%s  rfr=%.1f%%",
            loader.get_name(), len(tickers), currency, rfr * 100,
        )

        # ── 2. Fetch all prices (universe + optional benchmark) ───────
        from data.fetcher import Fetcher
        fetch_start = (self.start - timedelta(days=self.lookback_days + 60)).strftime("%Y-%m-%d")
        fetch_end   = self.end.strftime("%Y-%m-%d")

        all_requested = list(tickers) + ([self.benchmark] if self.benchmark else [])

        logger.info("Fetching price data (%s → %s)…", fetch_start, fetch_end)
        fetcher = Fetcher(cache_dir=self.cache_dir)
        raw     = fetcher.fetch(all_requested, start=fetch_start, end=fetch_end)

        # Separate benchmark
        bench_close = None
        if self.benchmark:
            avail = raw.columns.get_level_values(1)
            if self.benchmark in avail:
                bench_close   = raw["Close"][[self.benchmark]]
                raw           = raw.loc[:, avail != self.benchmark]
                logger.info("Benchmark '%s' separated.", self.benchmark)
            else:
                logger.warning("Benchmark '%s' not found in fetched data.", self.benchmark)

        price_index = raw.index
        close       = raw["Close"]

        # ── 3. Rebalance dates ────────────────────────────────────────
        rb_dates = _rebalance_dates(price_index, self.start, self.end, self.rebalance)
        if not rb_dates:
            raise RuntimeError(
                f"No rebalance dates found between {self.start.date()} and {self.end.date()}."
            )
        logger.info(
            "%d rebalance dates  (%s → %s)",
            len(rb_dates), rb_dates[0].date(), rb_dates[-1].date(),
        )

        # ── 4. Build scorer (stateless — reused across all rebalances) ─
        scorer = self._build_scorer(rfr=rfr)

        # ── 5. Create portfolio ───────────────────────────────────────
        portfolio = Portfolio(
            initial_cash = self.initial_cash,
            currency     = currency,
        )
        logger.info(
            "Portfolio created — initial cash: %s %.0f",
            currency, self.initial_cash,
        )

        # ── 6. Walk-forward loop ──────────────────────────────────────
        rebalance_log: List[dict] = []
        rb_dates_set = set(rb_dates)

        backtest_days = price_index[(price_index >= self.start) & (price_index <= self.end)]
        if len(backtest_days) == 0:
            raise RuntimeError("No trading days found in backtest range.")

        min_rows = max(22, self.lookbacks[-1] * 21 + 2)

        # Precompute rebalance selections so we can run the loop in one pass
        selected_on: dict = {}   # {date: [tickers]}
        for i, rdate in enumerate(rb_dates):
            prices_slice = raw.loc[:rdate]
            if len(prices_slice) < min_rows:
                logger.warning(
                    "Rebalance %d/%d (%s) skipped — only %d rows (need %d).",
                    i + 1, len(rb_dates), rdate.date(), len(prices_slice), min_rows,
                )
                rebalance_log.append({
                    "date": rdate, "selected": [], "n_passed": 0, "skipped": True,
                })
                continue

            selected = self._select_tickers(prices_slice, scorer)
            selected_on[rdate] = selected
            rebalance_log.append({
                "date": rdate, "selected": selected, "n_passed": len(selected), "skipped": False,
            })
            logger.info(
                "Signal %3d/%d  %s  →  %d tickers  %s",
                i + 1, len(rb_dates), rdate.date(), len(selected),
                selected[:5],
            )

        # ── 7. Daily simulation — rebalance + update_prices each day ──
        if self.verbose:
            from backtest.reporter import format_rebalance_block

        for day in backtest_days:
            price_row = close.loc[day].dropna()
            price_dict = price_row.to_dict()   # {ticker: price}

            # Execute rebalance if today is a rebalance date
            if day in rb_dates_set and day in selected_on:
                rb_event = portfolio.rebalance_to(selected_on[day], price_dict, day)
                if self.verbose:
                    print(format_rebalance_block(day, rb_event, portfolio))

            # Mark to market and record NAV snapshot
            portfolio.update_prices(price_dict, day)

        logger.info(
            "Simulation complete — final NAV: %s %.0f  (%.2f%% return)",
            currency, portfolio.nav,
            portfolio.total_return_pct,
        )

        # ── 8. Equity curve from portfolio NAV history ────────────────
        nav_df       = portfolio.nav_history_df
        nav_series   = nav_df["nav"]

        # Anchor at 1.0 one day before start
        pre_days = price_index[price_index < self.start]
        anchor   = pre_days[-1] if len(pre_days) > 0 else backtest_days[0] - pd.Timedelta("1D")

        anchor_nav   = pd.Series([self.initial_cash], index=[anchor])
        nav_extended = pd.concat([anchor_nav, nav_series])
        equity_curve = nav_extended / self.initial_cash * 100   # indexed to 100

        portfolio_returns = nav_series.pct_change().fillna(0.0)
        portfolio_returns = pd.concat([
            pd.Series([0.0], index=[anchor]),
            portfolio_returns,
        ])

        # ── 9. Benchmark equity curve ─────────────────────────────────
        bench_equity = None
        if bench_close is not None:
            bret = bench_close.iloc[:, 0].pct_change().reindex(backtest_days).fillna(0.0)
            b_ext        = pd.concat([pd.Series([0.0], index=[anchor]), bret])
            bench_equity = (1 + b_ext).cumprod() * 100   # indexed to 100
            bench_equity.name = self.benchmark

        # ── 10. Metrics ───────────────────────────────────────────────
        metrics       = compute_metrics(equity_curve, rfr=rfr)
        bench_metrics = compute_metrics(bench_equity, rfr=rfr) if bench_equity is not None else None

        # ── 11. Return result ─────────────────────────────────────────
        rb_df = pd.DataFrame(rebalance_log).set_index("date")

        return BacktestResult(
            equity_curve      = equity_curve,
            benchmark_equity  = bench_equity,
            portfolio_returns = portfolio_returns,
            rebalance_log     = rb_df,
            metrics           = metrics,
            bench_metrics     = bench_metrics,
            params            = self._params_dict(loader.get_name(), currency),
            portfolio         = portfolio,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_scorer(self, rfr: float = 0.065):
        from filters.momentum_filter import MomentumFilter
        from filters.volume_filter   import VolumeFilter
        from scoring.scorer          import Scorer, default_gates

        momentum = MomentumFilter(params={
            "lookbacks":      self.lookbacks,
            "mode":           self.mode,
            "risk_free_rate": rfr,
        })

        gates = [] if self.no_gates else default_gates(rfr=rfr, ema_fast=self.ema_fast, ema_slow=self.ema_slow)
        if self.volume_confirm:
            gates.append(VolumeFilter(params={"mode": "confirm"}, name="VolumeConfirm"))

        scorers = [momentum]
        weights = [1 - self.volume_weight]
        if not self.no_volume_score:
            scorers.append(VolumeFilter(params={"mode": "score"}, name="VolumeScore"))
            weights.append(self.volume_weight)

        return Scorer(gates=gates, scorers=scorers, weights=weights)

    def _select_tickers(self, prices_slice: pd.DataFrame, scorer) -> List[str]:
        try:
            result  = scorer.run(prices_slice)
            passing = result[result["rank"].notna()]

            if not self.no_diversify and len(passing) > 1:
                from filters.diversification_filter import DiversificationFilter
                div    = DiversificationFilter(params={
                    "threshold":   self.div_threshold,
                    "max_tickers": self.max_picks,
                })
                report   = div.get_report(result, prices_slice)
                accepted = report[report["status"] == "accepted"].sort_values("rank")
                return accepted.index.tolist()[: self.top_n]

            return passing.sort_values("rank").index.tolist()[: self.top_n]
        except Exception as exc:
            logger.warning("Scorer failed on slice: %s", exc)
            return []

    def _params_dict(self, universe_name: str, currency: str) -> dict:
        return {
            "universe":        universe_name,
            "start":           self.start.strftime("%Y-%m-%d"),
            "end":             self.end.strftime("%Y-%m-%d"),
            "top_n":           self.top_n,
            "rebalance":       self.rebalance,
            "lookback_days":   self.lookback_days,
            "mode":            self.mode,
            "lookbacks":       self.lookbacks,
            "no_gates":        self.no_gates,
            "ema_fast":        self.ema_fast,
            "ema_slow":        self.ema_slow,
            "volume_confirm":  self.volume_confirm,
            "no_volume_score": self.no_volume_score,
            "volume_weight":   self.volume_weight,
            "no_diversify":    self.no_diversify,
            "div_threshold":   self.div_threshold,
            "max_picks":       self.max_picks,
            "benchmark":       self.benchmark,
            "initial_cash":    self.initial_cash,
            "currency":        currency,
        }

    def __repr__(self) -> str:
        return (
            f"BacktestEngine("
            f"universe='{self.universe.stem}', "
            f"start='{self.start.date()}', "
            f"end='{self.end.date()}', "
            f"top_n={self.top_n}, "
            f"rebalance='{self.rebalance}', "
            f"mode='{self.mode}', "
            f"initial_cash={self.initial_cash:,.0f})"
        )
