"""
filters/up_days_filter.py
-------------------------
Up-days breadth gate: hard boolean filter.

Directly translated from the notebook condition:
    six_month_data = df['Close'][-126:]
    up_days = (six_month_data.pct_change() > 0).sum()
    up_days_pct = up_days / len(six_month_data) * 100
    up_days_pct > 50

A ticker passes only if more than X% of its trading days
in the lookback window were positive (close > previous close).
This filters out volatile stocks that only look good on paper
but lack consistent directional movement.

Params
------
min_up_pct  : float   Minimum % of up days as a decimal (default 0.50 = 50%)
window_days : int     Lookback window in trading days   (default 126 = ~6 months)

Usage:
    from filters.up_days_filter import UpDaysFilter

    f = UpDaysFilter()                                           # defaults
    f = UpDaysFilter(params={"min_up_pct": 0.55})               # stricter
    f = UpDaysFilter(params={"min_up_pct": 0.50,
                             "window_days": 63})                 # 3m window
"""

import logging

import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)

_DEFAULT_MIN_UP_PCT  = 0.50    # 50% — matches notebook exactly
_DEFAULT_WINDOW_DAYS = 126     # ~6 months of trading days


class UpDaysFilter(BaseFilter):
    """
    Up-days breadth gate.

    Passes a ticker only when:
        (count of days where Close > prev Close) / window_days  >=  min_up_pct
    """

    def _validate_params(self) -> None:
        pct = self.params.get("min_up_pct", _DEFAULT_MIN_UP_PCT)
        if not (0.0 < pct < 1.0):
            raise ValueError(
                f"UpDaysFilter: min_up_pct must be in (0, 1), got {pct}."
            )

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        """Returns 1.0 for every ticker — pure gate."""
        close = self._get_close(prices)
        return pd.Series(1.0, index=close.columns, name=self.name)

    def filter(self, scores: pd.Series) -> pd.Series:
        """Not used standalone — apply() overrides the full pipeline."""
        return pd.Series(True, index=scores.index)

    def apply(self, prices: pd.DataFrame) -> pd.Series:
        """
        Full gate: compute up-days % for every ticker over the window.
        Returns 1.0 for passing tickers, NaN for failing ones.
        """
        min_up_pct   = self.params.get("min_up_pct",   _DEFAULT_MIN_UP_PCT)
        window_days  = self.params.get("window_days",  _DEFAULT_WINDOW_DAYS)

        close = self._get_close(prices)

        results = {}
        for ticker in close.columns:
            series = close[ticker].dropna()

            # Use available history if shorter than window
            window_data = series.iloc[-window_days:] if len(series) >= window_days \
                          else series

            daily_returns = window_data.pct_change().dropna()

            if daily_returns.empty:
                logger.debug("%s: %s — no return data, skipping.", self.name, ticker)
                results[ticker] = float("nan")
                continue

            up_days     = (daily_returns > 0).sum()
            up_days_pct = up_days / len(daily_returns)

            passes = bool(up_days_pct >= min_up_pct)
            results[ticker] = 1.0 if passes else float("nan")

            logger.debug(
                "%s | %-20s  UpDays=%.1f%%  Threshold=%.1f%%  → %s",
                self.name, ticker,
                up_days_pct * 100, min_up_pct * 100,
                "PASS" if passes else "FAIL",
            )

        passed = sum(1 for v in results.values() if v == 1.0)
        logger.info(
            "%s: %d/%d tickers passed up-days >= %.0f%% gate",
            self.name, passed, len(results), min_up_pct * 100,
        )
        return pd.Series(results, name=self.name)