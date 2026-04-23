"""
filters/diversification_filter.py
-----------------------------------
Diversification filter: post-scorer greedy deduplication.

Runs AFTER momentum scoring and ranking. Takes the ranked ticker list
and removes duplicate exposures — if two tickers are too correlated,
only the higher-ranked one survives.

Algorithm (greedy, rank-aware):
    1. Sort tickers by momentum rank (best first)
    2. Start with an empty accepted list
    3. For each ticker (in rank order):
          Compute correlation with every already-accepted ticker
          If max correlation > threshold → skip (too similar to something better)
          Otherwise → accept
    4. Return accepted list

This guarantees:
    - No two tickers in the final list are too similar to each other
    - When two tickers conflict, the higher-ranked one always wins
    - Sector diversity happens naturally — you keep the best bank ETF,
      best IT ETF, best gold ETF, etc.

WHERE IT FITS IN THE PIPELINE
──────────────────────────────
Runs as the final step, after scoring and ranking:

    Full universe (322 ETFs)
        ↓
    CorrelationFilter (pre-screen)   →  ~90 tickers
        ↓
    4 gate filters                   →  ~20-30 tickers
        ↓
    MomentumFilter → ranked list
        ↓
    DiversificationFilter            →  clean, diverse final picks

Params
------
threshold   : float   Max allowed correlation between any two accepted
                      tickers. Default 0.85.
                      0.85 = standard — removes near-duplicates
                      0.90 = permissive — only removes near-clones
                      0.70 = strict — enforces strong diversification

window_days : int     Days of returns used for correlation. Default 126
                      (6 months — recent correlation matters more).

max_tickers : int     Optional hard cap on final output size.
                      Default None (no cap, keep all that pass).

Usage:
    from filters.diversification_filter import DiversificationFilter

    df = DiversificationFilter()
    df = DiversificationFilter(params={"threshold": 0.80, "max_tickers": 10})

    # Pass scored result DataFrame from Scorer + prices
    final_tickers = df.apply(result, prices)

    # Or get the full deduplication report
    report = df.get_report(result, prices)
"""

import logging
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD   = 0.85
_DEFAULT_WINDOW_DAYS = 126    # 6 months — recent correlation is what matters


class DiversificationFilter:
    """
    Post-scorer greedy diversification filter.

    Takes the ranked scorer output and removes correlated duplicates,
    always keeping the higher-ranked ticker when two conflict.

    Parameters
    ----------
    params : dict
        threshold   : float   Max correlation allowed (default 0.85)
        window_days : int     Days of returns for correlation (default 126)
        max_tickers : int     Optional cap on output size (default None)
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        threshold = self.params.get("threshold", _DEFAULT_THRESHOLD)
        if not (0.0 < threshold < 1.0):
            raise ValueError(
                f"DiversificationFilter: threshold must be in (0, 1), "
                f"got {threshold}."
            )
        logger.debug(
            "DiversificationFilter initialised — threshold=%.2f", threshold
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def apply(
        self,
        result: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> List[str]:
        """
        Apply greedy diversification to a scored result.

        Returns
        -------
        list of str
            Final diversified ticker list, ordered by rank (best first).
        """
        accepted, _ = self._greedy(result, prices)
        return accepted

    # ------------------------------------------------------------------
    # Report — useful for inspecting what was removed and why
    # ------------------------------------------------------------------

    def get_report(
        self,
        result: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a DataFrame showing every ticker's fate and reason.

        Columns:
            rank          Original momentum rank
            final_score   Original score
            status        "accepted" | "removed" | "no_data"
            removed_by    Ticker that caused removal (if removed)
            max_corr      Highest correlation with any accepted ticker
        """
        ranked = (
            result[result["rank"].notna()]
            .sort_values("rank")
            .index.tolist()
        )
        _, rows = self._greedy(result, prices)
        report  = pd.DataFrame(rows, index=ranked)
        report.index.name = "ticker"
        return report

    # ------------------------------------------------------------------
    # Shared greedy algorithm
    # ------------------------------------------------------------------

    def _greedy(
        self,
        result: pd.DataFrame,
        prices: pd.DataFrame,
    ):
        """
        Run the greedy rank-aware deduplication.

        Returns (accepted, rows) where:
            accepted  : list of str — tickers that survived
            rows      : list of dict — one entry per ranked ticker for get_report()
        """
        threshold   = self.params.get("threshold",   _DEFAULT_THRESHOLD)
        window_days = self.params.get("window_days", _DEFAULT_WINDOW_DAYS)
        max_tickers = self.params.get("max_tickers", None)

        ranked = (
            result[result["rank"].notna()]
            .sort_values("rank")
            .index.tolist()
        )

        if len(ranked) == 0:
            logger.warning("DiversificationFilter: no ranked tickers to process.")
            return [], []

        returns  = self._compute_returns(prices, window_days)
        accepted: List[str]  = []
        rows:     List[Dict] = []

        for ticker in ranked:
            rank  = result.loc[ticker, "rank"]
            score = result.loc[ticker, "final_score"]

            if ticker not in returns.columns:
                logger.debug("DiversificationFilter: %s not in returns, skipping.", ticker)
                rows.append({
                    "rank": rank, "final_score": score,
                    "status": "no_data", "removed_by": None, "max_corr": None,
                })
                continue

            if len(accepted) == 0:
                accepted.append(ticker)
                rows.append({
                    "rank": rank, "final_score": score,
                    "status": "accepted", "removed_by": None, "max_corr": 0.0,
                })
                logger.debug("  ACCEPT %s (first ticker)", ticker)
                continue

            accepted_in_returns = [t for t in accepted if t in returns.columns]
            corr_series = (
                returns[[ticker] + accepted_in_returns]
                .corr()[ticker]
                .drop(ticker)
                .abs()
            )
            max_corr = corr_series.max()
            conflict = corr_series.idxmax()

            if max_corr > threshold:
                rows.append({
                    "rank": rank, "final_score": score,
                    "status": "removed", "removed_by": conflict,
                    "max_corr": round(max_corr, 4),
                })
                logger.debug(
                    "  SKIP   %-20s  max_corr=%.3f with %s (threshold=%.2f)",
                    ticker, max_corr, conflict, threshold,
                )
            else:
                accepted.append(ticker)
                rows.append({
                    "rank": rank, "final_score": score,
                    "status": "accepted", "removed_by": None,
                    "max_corr": round(max_corr, 4),
                })
                logger.debug(
                    "  ACCEPT %-20s  max_corr=%.3f (below threshold)",
                    ticker, max_corr,
                )

            if max_tickers and len(accepted) >= max_tickers:
                logger.debug("  Reached max_tickers=%d, stopping.", max_tickers)
                break

        logger.info(
            "DiversificationFilter (threshold=%.2f): %d → %d tickers "
            "(removed %d correlated duplicates)",
            threshold, len(ranked), len(accepted), len(ranked) - len(accepted),
        )
        return accepted, rows

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_returns(
        self,
        prices: pd.DataFrame,
        window_days: int,
    ) -> pd.DataFrame:
        """
        Compute daily log-returns from Close prices over the window.
        """
        fields = prices.columns.get_level_values(0).unique().tolist()
        if "Close" not in fields:
            raise KeyError(
                f"DiversificationFilter requires 'Close'. "
                f"Available: {fields}"
            )

        close   = prices["Close"].iloc[-window_days:]
        returns = np.log(close / close.shift(1)).dropna(how="all")
        return returns

    def __repr__(self) -> str:
        threshold = self.params.get("threshold", _DEFAULT_THRESHOLD)
        return f"DiversificationFilter(threshold={threshold:.2f})"