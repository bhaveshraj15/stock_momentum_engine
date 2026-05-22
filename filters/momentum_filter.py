"""
filters/momentum_filter.py
--------------------------
Momentum filter: continuous scoring filter.

Directly translated from the notebook functions:
    calc_return()            → mode="returns"
    calculate_sharpe_ratio() → mode="sharpe"
    calculate_sortino_ratio() → mode="sortino"

For each ticker, computes a score across multiple lookback
periods (default: 1m, 3m, 6m, 12m) and combines them into
a single weighted score. Higher score = stronger momentum.

Slice logic (matches notebook exactly):
    window = close[(period * -21) - 1:]
    score  = close.iloc[-1] / close.iloc[0] - 1

The -1 ensures iloc[0] lands on the exact boundary price
of the lookback window, not one day inside it.

Params
------
lookbacks       : list of int    Lookback periods in months (default [1, 3, 6, 12])
weights         : list of float  One weight per lookback    (default equal weights)
mode            : str            "returns" | "sharpe" | "sortino" (default "returns")
risk_free_rate  : float          Annual risk-free rate, used in sharpe/sortino
                                 (default 0.065 = 6.5%, matches notebook)

Usage:
    from filters.momentum_filter import MomentumFilter

    f = MomentumFilter()                          # defaults
    f = MomentumFilter(params={
            "lookbacks": [3, 6, 12],
            "weights":   [1, 2, 3],
            "mode":      "sharpe",
        })

    scores = f.apply(prices)   # pd.Series: ticker → float score
"""

import logging

import numpy as np
import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)

_DEFAULT_LOOKBACKS      = [1, 3, 6, 12]
_DEFAULT_MODE           = "returns"
_DEFAULT_RISK_FREE_RATE = 0.065   # 6.5% p.a. — matches notebook exactly


class MomentumFilter(BaseFilter):
    """
    Multi-period momentum scorer.

    Returns a weighted combination of per-period scores.
    Plugs into Scorer as a scoring filter (not a gate).
    """

    def _validate_params(self) -> None:
        mode = self.params.get("mode", _DEFAULT_MODE)
        if mode not in ("returns", "sharpe", "sortino"):
            raise ValueError(
                f"MomentumFilter: mode must be 'returns', 'sharpe', or "
                f"'sortino'. Got '{mode}'."
            )

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        """
        Compute momentum score using sum-of-ordinal-ranks across periods.

        For each lookback period p:
            raw_score[ticker, p] = metric value (return / sharpe / sortino)

        Then rank ALL tickers by each period separately (rank 1 = best).
        Sum the ranks per ticker — lowest sum = most consistently strong.
        Return the negative sum so higher output = better (Scorer expects that).

        A ticker ranked #1 in all 4 periods gets sum=4 → output=-4 (best).
        A ticker ranked last in all 4 periods gets sum=4N → output=-4N (worst).
        """
        lookbacks      = self.params.get("lookbacks",      _DEFAULT_LOOKBACKS)
        mode           = self.params.get("mode",           _DEFAULT_MODE)
        risk_free_rate = self.params.get("risk_free_rate", _DEFAULT_RISK_FREE_RATE)

        close   = self._get_close(prices)
        tickers = close.columns.tolist()

        # ── Build raw score matrix: rows=tickers, cols=periods ────────
        period_scores = pd.DataFrame(index=tickers, columns=lookbacks, dtype=float)

        for ticker in tickers:
            series = close[ticker].dropna()
            for period in lookbacks:
                needed = (period * 21) + 1
                if len(series) < needed:
                    logger.debug(
                        "%s: %s — %d rows, need %d for %dm.",
                        self.name, ticker, len(series), needed, period,
                    )
                    period_scores.loc[ticker, period] = float("nan")
                    continue

                window = series.iloc[(period * -21) - 1:]
                score  = self._compute_period_score(window, period, mode, risk_free_rate)
                period_scores.loc[ticker, period] = (
                    score if (score is not None and np.isfinite(score)) else float("nan")
                )

        # ── Rank each period column independently (rank 1 = best) ─────
        rank_matrix = period_scores.rank(
            ascending=False, method="min", na_option="keep"
        )

        # ── Sum ranks across all periods; NaN if ticker missed all ────
        sum_of_ranks = rank_matrix.sum(axis=1, min_count=1)

        # ── Negate: higher output = lower rank sum = better ───────────
        result = -sum_of_ranks
        result.name = self.name

        logger.info(
            "%s (%s): scored %d/%d tickers via sum-of-ranks over %s",
            self.name, mode, int(result.notna().sum()), len(tickers), lookbacks,
        )
        return result

    def filter(self, scores: pd.Series) -> pd.Series:
        """
        No hard gate — MomentumFilter is a pure scorer.
        All tickers with a valid (non-NaN) score pass through.
        """
        return pd.Series(scores.notna(), index=scores.index)

    # ------------------------------------------------------------------
    # Per-period score helpers
    # ------------------------------------------------------------------

    def _compute_period_score(
        self,
        window: pd.Series,
        period: int,
        mode: str,
        risk_free_rate: float,
    ):
        if mode == "returns":
            return self._calc_return(window)

        daily_returns = window.pct_change().dropna()
        if daily_returns.empty:
            return None

        if mode == "sharpe":
            return self._calc_sharpe(daily_returns, period, risk_free_rate)

        if mode == "sortino":
            return self._calc_sortino(daily_returns, period, risk_free_rate)

        return None

    def _calc_return(self, window: pd.Series) -> float:
        """
        Percentage return over the window.
        Matches notebook: (close.iloc[-1] / close.iloc[0]) - 1
        """
        return (window.iloc[-1] / window.iloc[0]) - 1

    def _calc_sharpe(
        self,
        daily_returns: pd.Series,
        period: int,
        risk_free_rate: float,
    ) -> float:
        """
        Sharpe ratio. Matches notebook exactly:
            sum_ret    = daily_returns.sum()
            period_std = daily_returns.std() * sqrt(period * 21)
            sharpe     = (sum_ret - rfr * (period/12)) / period_std
        """
        sum_ret    = daily_returns.sum()
        period_std = daily_returns.std() * np.sqrt(period * 21)

        if period_std == 0:
            return float("nan")

        return (sum_ret - risk_free_rate * (period / 12)) / period_std

    def _calc_sortino(
        self,
        daily_returns: pd.Series,
        period: int,
        risk_free_rate: float,
    ) -> float:
        """
        Sortino ratio. Matches notebook exactly:
            downside   = daily_returns where return < 0
            period_std = downside.std() * sqrt(period * 21)
            sortino    = (sum_ret - rfr * (period/12)) / period_std
        """
        sum_ret    = daily_returns.sum()
        downside   = daily_returns.where(daily_returns < 0)
        period_std = downside.std() * np.sqrt(period * 21)

        if period_std == 0 or np.isnan(period_std):
            return float("nan")

        return (sum_ret - risk_free_rate * (period / 12)) / period_std