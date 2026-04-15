"""
filters/correlation_filter.py
------------------------------
Correlation filter: universe pre-screening step.

Directly translated from correlation_test.ipynb logic:

    # Compute daily % change for each ticker
    change = (Close / Open - 1) * 100

    # Build correlation matrix
    corr_matrix = change.corr()

    # Keep tickers that appear in low-correlation pairs
    pairs = corr_matrix.where(|corr| <= ccp).stack()
    opt_tickers = unique tickers from those pairs

The notebook uses ccp = 1.75e-4 as the correlation cap parameter.
This keeps tickers that are nearly uncorrelated with at least one
other ticker — producing a diversified sub-universe before scoring.

WHERE IT FITS IN THE PIPELINE
──────────────────────────────
This is a PRE-FILTER on the universe, not a gate or scorer.
It runs BEFORE the 4 gate filters, reducing a large universe
(e.g. 322 ETFs) down to a diversified subset (e.g. ~90 tickers).

    Full universe (322 ETFs)
        ↓
    CorrelationFilter  →  ~90 low-corr tickers
        ↓
    4 gate filters (Trend, High52w, MinReturn, UpDays)
        ↓
    MomentumFilter  →  ranked output

Params
------
ccp          : float   Correlation cap parameter. Tickers in pairs
                       where |corr| <= ccp are kept.
                       Default 1.75e-4 — matches notebook exactly.
               Lower = stricter (fewer tickers kept).
               Higher = more permissive (more tickers kept).
               4.0e-4 was noted in notebook as an alternative.

window_days  : int     Days of history for correlation. Default 252.

Usage:
    from filters.correlation_filter import CorrelationFilter

    cf = CorrelationFilter()                         # defaults
    cf = CorrelationFilter(params={"ccp": 4.0e-4})  # more permissive

    kept = cf.apply_to_universe(prices, tickers)     # list of tickers
    matrix = cf.get_correlation_matrix(prices)       # for inspection
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_CCP         = 1.75e-4   # matches notebook exactly
_DEFAULT_WINDOW_DAYS = 252


class CorrelationFilter:
    """
    Universe-level diversification pre-filter.

    Reduces a large ticker universe to a subset of tickers that
    are not highly correlated with each other.

    Not a BaseFilter subclass — operates on the full universe list
    rather than scoring individual tickers. Runs before gate filters.

    Parameters
    ----------
    params : dict
        ccp         : float   Correlation cap (default 1.75e-4)
        window_days : int     Days of history for correlation (default 252)
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        ccp = self.params.get("ccp", _DEFAULT_CCP)
        if ccp <= 0:
            raise ValueError(f"CorrelationFilter: ccp must be > 0, got {ccp}.")
        logger.debug("CorrelationFilter initialised — ccp=%.2e", ccp)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def apply_to_universe(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Filter universe down to low-correlation tickers.

        Parameters
        ----------
        prices  : pd.DataFrame
            MultiIndex OHLCV DataFrame from Fetcher.
            Must contain both "Open" and "Close" fields.
        tickers : list of str, optional
            Subset of tickers to consider. If None, uses all
            tickers in prices.

        Returns
        -------
        list of str
            Tickers that appear in at least one low-correlation pair.
        """
        ccp         = self.params.get("ccp",         _DEFAULT_CCP)
        window_days = self.params.get("window_days", _DEFAULT_WINDOW_DAYS)

        change = self._compute_daily_change(prices, window_days)

        if tickers:
            available = [t for t in tickers if t in change.columns]
            missing   = [t for t in tickers if t not in change.columns]
            if missing:
                logger.warning(
                    "CorrelationFilter: %d tickers missing from prices: %s",
                    len(missing), missing[:5],
                )
            change = change[available]

        if change.empty or change.shape[1] < 2:
            logger.warning(
                "CorrelationFilter: need >= 2 tickers, got %d. "
                "Returning input unchanged.",
                change.shape[1],
            )
            return list(change.columns)

        # Build correlation matrix of daily % changes
        corr_matrix = change.corr()

        # Keep tickers that appear in any pair where |corr| <= ccp
        corr_arr  = corr_matrix.values.copy()          # writable numpy array
        mask      = np.abs(corr_arr) <= ccp
        np.fill_diagonal(mask, False)                  # remove self-pairs

        # Rebuild as DataFrame for .stack()
        low_corr_df = pd.DataFrame(
            np.where(mask, corr_arr, np.nan),
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
        pairs       = low_corr_df.stack()
        opt_tickers = sorted(
            set(ticker for pair in pairs.index for ticker in pair)
        )

        n_in  = change.shape[1]
        n_out = len(opt_tickers)
        logger.info(
            "CorrelationFilter (ccp=%.2e): %d → %d tickers "
            "(removed %d highly correlated)",
            ccp, n_in, n_out, n_in - n_out,
        )

        return opt_tickers

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_correlation_matrix(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return the full correlation matrix for inspection or plotting.

        Example:
            import seaborn as sns
            matrix = cf.get_correlation_matrix(prices)
            sns.heatmap(matrix, annot=True)
        """
        window_days = self.params.get("window_days", _DEFAULT_WINDOW_DAYS)
        change = self._compute_daily_change(prices, window_days)
        if tickers:
            change = change[[t for t in tickers if t in change.columns]]
        return change.corr()

    def get_low_corr_pairs(
        self,
        prices: pd.DataFrame,
        tickers: Optional[List[str]] = None,
    ) -> pd.Series:
        """
        Return all ticker pairs with |correlation| <= ccp.
        Useful for debugging which pairs drive the filter output.

        Returns pd.Series indexed by (ticker_a, ticker_b).
        """
        ccp         = self.params.get("ccp",         _DEFAULT_CCP)
        window_days = self.params.get("window_days", _DEFAULT_WINDOW_DAYS)
        change      = self._compute_daily_change(prices, window_days)

        if tickers:
            change = change[[t for t in tickers if t in change.columns]]

        corr_matrix   = change.corr()
        corr_arr      = corr_matrix.values.copy()
        mask          = np.abs(corr_arr) <= ccp
        np.fill_diagonal(mask, False)
        low_corr_df   = pd.DataFrame(
            np.where(mask, corr_arr, np.nan),
            index=corr_matrix.index,
            columns=corr_matrix.columns,
        )
        return low_corr_df.stack().sort_values()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_daily_change(
        self,
        prices: pd.DataFrame,
        window_days: int,
    ) -> pd.DataFrame:
        """
        Compute daily % change: (Close / Open - 1) * 100
        Matches notebook exactly.
        """
        fields = prices.columns.get_level_values(0).unique().tolist()

        if "Open" not in fields or "Close" not in fields:
            raise KeyError(
                "CorrelationFilter requires both 'Open' and 'Close' fields. "
                f"Available: {fields}"
            )

        open_df  = prices["Open"].iloc[-window_days:]
        close_df = prices["Close"].iloc[-window_days:]

        common   = open_df.columns.intersection(close_df.columns)
        change   = (close_df[common] / open_df[common] - 1) * 100
        change   = change.dropna(how="all")

        logger.debug(
            "Daily change matrix: %d days × %d tickers",
            len(change), change.shape[1],
        )
        return change

    def __repr__(self) -> str:
        ccp = self.params.get("ccp", _DEFAULT_CCP)
        return f"CorrelationFilter(ccp={ccp:.2e})"