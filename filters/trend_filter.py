"""
filters/trend_filter.py
-----------------------
Trend filter: hard boolean gate based on EMA alignment.

Directly translated from the notebook condition:
    df['Close'][-1] >= df['EMA100'][-1] >= df['EMA200'][-1]

A ticker passes only if its latest close sits above both its
EMA100 and EMA200, and EMA100 sits above EMA200.
This confirms the stock is in a medium and long-term uptrend.

compute() returns 1.0 for all tickers — this is a pure gate,
not a scoring filter. All the work happens inside apply().

Params
------
fast_span : int   EMA span for the faster average  (default 100)
slow_span : int   EMA span for the slower average  (default 200)

Usage:
    from filters.trend_filter import TrendFilter

    f = TrendFilter()                                        # defaults
    f = TrendFilter(params={"fast_span": 50, "slow_span": 200})

    scores = f.apply(prices)   # NaN for tickers that fail the gate
"""

import logging

import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)

_DEFAULT_FAST = 100
_DEFAULT_SLOW = 200


class TrendFilter(BaseFilter):
    """
    EMA alignment gate.

    Passes a ticker only when:
        Close[-1]  >=  EMA_fast[-1]  >=  EMA_slow[-1]
    """

    def _validate_params(self) -> None:
        fast = self.params.get("fast_span", _DEFAULT_FAST)
        slow = self.params.get("slow_span", _DEFAULT_SLOW)
        if fast >= slow:
            raise ValueError(
                f"TrendFilter: fast_span ({fast}) must be < slow_span ({slow})."
            )

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        """
        Returns 1.0 for every ticker.
        TrendFilter is a pure gate — scoring is not its job.
        """
        close = self._get_close(prices)
        return pd.Series(1.0, index=close.columns, name=self.name)

    def filter(self, scores: pd.Series) -> pd.Series:
        """Not used standalone — apply() overrides the full pipeline."""
        return pd.Series(True, index=scores.index)

    def apply(self, prices: pd.DataFrame) -> pd.Series:
        """
        Full gate: compute EMA alignment for every ticker.
        Returns 1.0 for passing tickers, NaN for failing ones.
        """
        fast_span = self.params.get("fast_span", _DEFAULT_FAST)
        slow_span = self.params.get("slow_span", _DEFAULT_SLOW)

        close = self._get_close(prices)

        results = {}
        for ticker in close.columns:
            series = close[ticker].dropna()

            if len(series) < slow_span:
                logger.debug(
                    "%s: %s skipped — %d rows available, need %d for EMA%d",
                    self.name, ticker, len(series), slow_span, slow_span,
                )
                results[ticker] = float("nan")
                continue

            ema_fast = series.ewm(span=fast_span, adjust=False).mean()
            ema_slow = series.ewm(span=slow_span, adjust=False).mean()

            last_close    = series.iloc[-1]
            last_ema_fast = ema_fast.iloc[-1]
            last_ema_slow = ema_slow.iloc[-1]

            passes = bool(last_close >= last_ema_fast >= last_ema_slow)
            results[ticker] = 1.0 if passes else float("nan")

            logger.debug(
                "%s | %-20s  Close=%.2f  EMA%d=%.2f  EMA%d=%.2f  → %s",
                self.name, ticker,
                last_close, fast_span, last_ema_fast, slow_span, last_ema_slow,
                "PASS" if passes else "FAIL",
            )

        passed = sum(1 for v in results.values() if v == 1.0)
        logger.info(
            "%s: %d/%d tickers passed EMA%d >= EMA%d gate",
            self.name, passed, len(results), fast_span, slow_span,
        )
        return pd.Series(results, name=self.name)