"""
filters/high52w_filter.py
-------------------------
52-week high proximity filter: hard boolean gate.

Directly translated from the notebook condition:
    high_52_week = df['Close'][-252:].max()
    within_20_pct_high = df['Close'][-1] >= high_52_week * 0.8

A ticker passes only if its latest close is within X% of its
52-week (252 trading days) high. Stocks trading far below their
annual peak are in drawdown and excluded from the momentum universe.

compute() returns 1.0 for all tickers — pure gate, not a scorer.

Params
------
window_days  : int    Lookback window in trading days  (default 252)
min_proximity: float  Minimum ratio of close/high      (default 0.80)
                      i.e. within 20% of the 52w high

Usage:
    from filters.high52w_filter import High52wFilter

    f = High52wFilter()                                       # defaults
    f = High52wFilter(params={"min_proximity": 0.85})         # within 15%

    scores = f.apply(prices)   # NaN for tickers that fail the gate
"""

import logging

import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW    = 252
_DEFAULT_PROXIMITY = 0.80


class High52wFilter(BaseFilter):
    """
    52-week high proximity gate.

    Passes a ticker only when:
        Close[-1]  >=  max(Close[-window_days:])  *  min_proximity
    """

    def _validate_params(self) -> None:
        proximity = self.params.get("min_proximity", _DEFAULT_PROXIMITY)
        if not (0.0 < proximity <= 1.0):
            raise ValueError(
                f"High52wFilter: min_proximity must be in (0, 1], got {proximity}."
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
        Full gate: check 52-week high proximity for every ticker.
        Returns 1.0 for passing tickers, NaN for failing ones.
        """
        window    = self.params.get("window_days",   _DEFAULT_WINDOW)
        proximity = self.params.get("min_proximity", _DEFAULT_PROXIMITY)

        close = self._get_close(prices)

        results = {}
        for ticker in close.columns:
            series = close[ticker].dropna()

            if len(series) < window:
                logger.debug(
                    "%s: %s — only %d rows, need %d. Skipping.",
                    self.name, ticker, len(series), window,
                )
                results[ticker] = float("nan")
                continue

            window_data = series.iloc[-window:]

            high_52w    = window_data.max()
            last_close  = series.iloc[-1]
            threshold   = high_52w * proximity

            passes = bool(last_close >= threshold)
            results[ticker] = 1.0 if passes else float("nan")

            logger.debug(
                "%s | %-20s  Close=%.2f  52wHigh=%.2f  "
                "Threshold=%.2f (%.0f%%)  → %s",
                self.name, ticker,
                last_close, high_52w, threshold, proximity * 100,
                "PASS" if passes else "FAIL",
            )

        passed = sum(1 for v in results.values() if v == 1.0)
        logger.info(
            "%s: %d/%d tickers passed within-%.0f%%-of-52w-high gate",
            self.name, passed, len(results), (1 - proximity) * 100,
        )
        return pd.Series(results, name=self.name)