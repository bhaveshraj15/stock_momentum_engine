"""
filters/base_filter.py
----------------------
Abstract base class for all filters in the momentum engine.

Every filter — momentum, trend, volatility, correlation — must inherit
from BaseFilter and implement two methods:

    compute(prices)  →  pd.Series   scored float per ticker
    filter(scores)   →  pd.Series   bool mask (True = keep)

The only method the outside world (scorer.py) calls is:

    apply(prices)    →  pd.Series   scores with excluded tickers set to NaN

This clean contract means:
  - scorer.py never needs to know what's inside a filter
  - RL/ML extensions can wrap or replace apply() without touching filter logic
  - All config flows through params dict — no hardcoded values

Usage:
    # You never instantiate BaseFilter directly — only subclasses
    from filters.momentum_filter import MomentumFilter

    f = MomentumFilter(params={"lookbacks": [3, 6, 12], "weights": [1, 1, 1]})
    scores = f.apply(prices)   # pd.Series: ticker → float | NaN
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BaseFilter(ABC):
    """
    Abstract base class for all filters.

    Parameters
    ----------
    params : dict
        Filter-specific configuration. Each subclass defines
        what keys it expects. Passed as plain dict so it can
        be loaded directly from default_params.yaml later.

    name : str, optional
        Human-readable label used in logs and reports.
        Defaults to the class name.
    """

    def __init__(self, params: Optional[dict] = None, name: Optional[str] = None):
        self.params = params or {}
        self.name   = name or self.__class__.__name__
        self._validate_params()
        logger.debug("Initialised filter: %s | params: %s", self.name, self.params)

    # ------------------------------------------------------------------
    # Methods subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self, prices: pd.DataFrame) -> pd.Series:
        """
        Compute a numeric score for each ticker.

        Parameters
        ----------
        prices : pd.DataFrame
            MultiIndex columns (field, ticker) as returned by Fetcher.
            Index is a DatetimeIndex of trading days.
            Minimum required field: "Close".

        Returns
        -------
        pd.Series
            Index  : ticker symbols (str)
            Values : float scores — higher is better.
                     Return NaN for tickers that could not be scored.
        """

    @abstractmethod
    def filter(self, scores: pd.Series) -> pd.Series:
        """
        Apply a hard boolean gate on top of the computed scores.

        Use this when a ticker must be fully excluded regardless of
        its score — e.g. "price must be above SMA200, no exceptions".
        For filters with no hard gate, simply return all-True.

        Parameters
        ----------
        scores : pd.Series
            Output of compute() — ticker → float.

        Returns
        -------
        pd.Series
            Index  : same tickers as scores
            Values : bool — True = keep, False = exclude entirely.
        """

    # ------------------------------------------------------------------
    # The one method scorer.py calls
    # ------------------------------------------------------------------

    def apply(self, prices: pd.DataFrame) -> pd.Series:
        """
        Full pipeline: compute scores → apply hard gate → NaN excluded tickers.

        Parameters
        ----------
        prices : pd.DataFrame
            MultiIndex OHLCV DataFrame from Fetcher.

        Returns
        -------
        pd.Series
            Ticker → float score, NaN for any excluded ticker.
        """
        scores = self.compute(prices)

        if not isinstance(scores, pd.Series):
            raise TypeError(
                f"{self.name}.compute() must return a pd.Series, "
                f"got {type(scores).__name__}."
            )

        mask = self.filter(scores)

        if not isinstance(mask, pd.Series) or mask.dtype != bool:
            raise TypeError(
                f"{self.name}.filter() must return a bool pd.Series, "
                f"got {type(mask).__name__} dtype={getattr(mask, 'dtype', '?')}."
            )

        excluded = (~mask).sum()
        if excluded:
            logger.debug(
                "%s excluded %d ticker(s) via hard gate: %s",
                self.name,
                excluded,
                mask[~mask].index.tolist(),
            )

        # Set excluded tickers to NaN — scorer drops them
        scores[~mask] = float("nan")
        return scores

    # ------------------------------------------------------------------
    # Param validation hook — optional override in subclasses
    # ------------------------------------------------------------------

    def _validate_params(self) -> None:
        """
        Override in subclasses to assert required params are present.

        Example:
            def _validate_params(self):
                assert "lookbacks" in self.params, "lookbacks param required"
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _get_close(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the Close price matrix from a MultiIndex OHLCV DataFrame.
        Returns a flat DataFrame: index=Date, columns=tickers.
        """
        if "Close" not in prices.columns.get_level_values(0):
            raise KeyError(
                f"{self.name}: 'Close' field not found in prices DataFrame. "
                f"Available fields: "
                f"{prices.columns.get_level_values(0).unique().tolist()}"
            )
        return prices["Close"]

    def _trading_days(self, months: int) -> int:
        """Convert calendar months to approximate trading days (21 days/month)."""
        return months * 21

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', params={self.params})"
        )