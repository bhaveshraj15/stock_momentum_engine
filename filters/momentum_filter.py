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
        lookbacks = self.params.get("lookbacks", _DEFAULT_LOOKBACKS)
        weights   = self.params.get("weights")
        mode      = self.params.get("mode", _DEFAULT_MODE)

        if weights is not None and len(weights) != len(lookbacks):
            raise ValueError(
                f"MomentumFilter: weights length ({len(weights)}) must match "
                f"lookbacks length ({len(lookbacks)})."
            )
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
        Compute weighted momentum score for each ticker.

        For each lookback period:
            window = close[(period * -21) - 1:]        ← notebook boundary fix
            score  = close.iloc[-1] / close.iloc[0] - 1  (returns mode)
                   | sharpe ratio                         (sharpe mode)
                   | sortino ratio                        (sortino mode)

        Final score = weighted average across all periods.
        """
        lookbacks      = self.params.get("lookbacks",      _DEFAULT_LOOKBACKS)
        weights        = self.params.get("weights",        [1.0] * len(lookbacks))
        mode           = self.params.get("mode",           _DEFAULT_MODE)
        risk_free_rate = self.params.get("risk_free_rate", _DEFAULT_RISK_FREE_RATE)

        if weights is None:
            weights = [1.0] * len(lookbacks)

        close = self._get_close(prices)

        results = {}
        for ticker in close.columns:
            series = close[ticker].dropna()

            period_scores  = []
            period_weights = []

            for period, weight in zip(lookbacks, weights):
                needed = (period * 21) + 1
                if len(series) < needed:
                    logger.debug(
                        "%s: %s — %d rows available, need %d for %dm. Skipping.",
                        self.name, ticker, len(series), needed, period,
                    )
                    continue

                # Exact notebook slice: (period * -21) - 1
                window = series.iloc[(period * -21) - 1:]
                score  = self._compute_period_score(
                    window, period, mode, risk_free_rate
                )

                if score is not None and np.isfinite(score):
                    period_scores.append(score)
                    period_weights.append(weight)

            if not period_scores:
                results[ticker] = float("nan")
                continue

            total_weight   = sum(period_weights)
            weighted_score = sum(
                s * w for s, w in zip(period_scores, period_weights)
            )
            results[ticker] = weighted_score / total_weight

            logger.debug(
                "%s | %-20s  score=%.4f",
                self.name, ticker, results[ticker],
            )

        logger.info(
            "%s (%s): scored %d/%d tickers",
            self.name, mode,
            sum(1 for v in results.values() if np.isfinite(v)),
            len(results),
        )
        return pd.Series(results, name=self.name)

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