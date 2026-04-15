"""
filters/min_return_filter.py
----------------------------
Minimum return gate: hard boolean filter.

Directly translated from the notebook condition:
    one_year_return = (df['Close'][-1] / df['Close'][-252] - 1) * 100
    one_year_return >= 6.5

A ticker passes only if its return over the lookback window
meets or exceeds the minimum threshold. Filters out flat,
stagnant, or negative-momentum stocks entirely.

Params
------
min_return   : float   Minimum return as a decimal (default 0.065 = 6.5%)
window_days  : int     Lookback window in trading days (default 252 = 1 year)

Usage:
    from filters.min_return_filter import MinReturnFilter

    f = MinReturnFilter()                                        # defaults
    f = MinReturnFilter(params={"min_return": 0.10})             # 10% minimum
    f = MinReturnFilter(params={"min_return": 0.065,
                                "window_days": 126})             # 6m window
"""

import logging

import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)

_DEFAULT_MIN_RETURN  = 0.065   # 6.5% — matches notebook exactly
_DEFAULT_WINDOW_DAYS = 252     # 1 year of trading days


class MinReturnFilter(BaseFilter):
    """
    Minimum absolute return gate.

    Passes a ticker only when:
        (Close[-1] / Close[-window_days] - 1)  >=  min_return
    """

    def _validate_params(self) -> None:
        window = self.params.get("window_days", _DEFAULT_WINDOW_DAYS)
        if window < 1:
            raise ValueError(
                f"MinReturnFilter: window_days must be >= 1, got {window}."
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
        Full gate: check minimum return for every ticker.
        Returns 1.0 for passing tickers, NaN for failing ones.
        """
        min_return   = self.params.get("min_return",   _DEFAULT_MIN_RETURN)
        window_days  = self.params.get("window_days",  _DEFAULT_WINDOW_DAYS)

        close = self._get_close(prices)

        results = {}
        for ticker in close.columns:
            series = close[ticker].dropna()

            if len(series) <= window_days:
                # Not enough history — use whatever we have
                logger.debug(
                    "%s: %s — only %d rows available, need %d. Using full history.",
                    self.name, ticker, len(series), window_days,
                )
                start_price = series.iloc[0]
            else:
                # Exact boundary capture — same as notebook's (period*-21)-1 logic
                start_price = series.iloc[-(window_days + 1)]

            end_price = series.iloc[-1]
            ret = (end_price / start_price) - 1

            passes = bool(ret >= min_return)
            results[ticker] = 1.0 if passes else float("nan")

            logger.debug(
                "%s | %-20s  Return=%.2f%%  Threshold=%.2f%%  → %s",
                self.name, ticker,
                ret * 100, min_return * 100,
                "PASS" if passes else "FAIL",
            )

        passed = sum(1 for v in results.values() if v == 1.0)
        logger.info(
            "%s: %d/%d tickers passed min-return >= %.1f%% gate",
            self.name, passed, len(results), min_return * 100,
        )
        return pd.Series(results, name=self.name)