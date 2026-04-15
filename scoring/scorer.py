"""
scoring/scorer.py
-----------------
Combines multiple filters into a single ranked output.

The scorer separates filters into two categories:

  GATE filters   — return 1.0 or NaN (boolean pass/fail)
                   Applied first as AND logic: a ticker must pass
                   ALL gates to remain in the universe.
                   e.g. TrendFilter, High52wFilter

  SCORE filters  — return a continuous float (higher = better)
                   Applied only to tickers that survived the gates.
                   Their outputs are normalised 0-1 and combined
                   as a weighted average into a final rank score.
                   e.g. MomentumFilter (coming next)

Output
------
A pd.DataFrame with columns:
  - one column per filter (raw output of apply())
  - "final_score"  weighted combined score (NaN = gated out)
  - "rank"         integer rank, 1 = best (NaN = gated out)

Usage:
    from filters.trend_filter   import TrendFilter
    from filters.high52w_filter import High52wFilter
    from scoring.scorer         import Scorer

    scorer = Scorer(
        gates=[TrendFilter(), High52wFilter()],
        scorers=[],          # add MomentumFilter here later
    )

    result = scorer.run(prices)
    print(result[["final_score", "rank"]].dropna().sort_values("rank"))
"""

import logging
from typing import List, Optional

import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)


def default_gates() -> List[BaseFilter]:
    """
    Returns the full v1 gate stack:
        1. TrendFilter          — Close >= EMA100 >= EMA200
        2. High52wFilter        — within 20% of 52-week high
        3. MinReturnFilter      — 1-year return >= 6.5%
        4. UpDaysFilter         — >50% up days in last 6 months
        5. VolumeFilter gate    — avg volume >= 100k (removes illiquid)
        6. VolumeFilter confirm — vol_21d > vol_63d > vol_252d (removes fakes)

    Usage:
        scorer = Scorer(gates=default_gates())
    """
    from filters.trend_filter      import TrendFilter
    from filters.high52w_filter    import High52wFilter
    from filters.min_return_filter import MinReturnFilter
    from filters.up_days_filter    import UpDaysFilter
    from filters.volume_filter     import VolumeFilter

    return [
        TrendFilter(),
        High52wFilter(),
        MinReturnFilter(),
        UpDaysFilter(),
        VolumeFilter(params={"mode": "gate"}),
        VolumeFilter(params={"mode": "confirm"}, name="VolumeConfirm"),
    ]


class Scorer:
    """
    Runs gate filters then scoring filters and produces a ranked DataFrame.

    Parameters
    ----------
    gates : list of BaseFilter
        Hard boolean filters — ticker must pass ALL of them.
    scorers : list of BaseFilter
        Continuous scoring filters — combined as weighted average.
    weights : list of float, optional
        One weight per scorer. Defaults to equal weights.
        Ignored if scorers is empty.
    """

    def __init__(
        self,
        gates:   Optional[List[BaseFilter]] = None,
        scorers: Optional[List[BaseFilter]] = None,
        weights: Optional[List[float]]      = None,
    ):
        self.gates   = gates   or []
        self.scorers = scorers or []
        self.weights = weights

        if self.weights is not None and len(self.weights) != len(self.scorers):
            raise ValueError(
                f"Scorer: weights length ({len(self.weights)}) must match "
                f"scorers length ({len(self.scorers)})."
            )

        logger.info(
            "Scorer initialised — %d gate(s): %s | %d scorer(s): %s",
            len(self.gates),   [f.name for f in self.gates],
            len(self.scorers), [f.name for f in self.scorers],
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Run the full pipeline: gates → scorers → rank.

        Parameters
        ----------
        prices : pd.DataFrame
            MultiIndex OHLCV DataFrame from Fetcher.

        Returns
        -------
        pd.DataFrame
            Index   : ticker symbols
            Columns : one per filter + "final_score" + "rank"
            Tickers that failed any gate have NaN in final_score/rank.
        """
        all_tickers = prices.columns.get_level_values(1).unique().tolist()
        result      = pd.DataFrame(index=all_tickers)

        # ---- Step 1: Apply gate filters (AND logic) ------------------
        gate_mask = pd.Series(True, index=all_tickers)

        for gate in self.gates:
            logger.info("Applying gate: %s", gate.name)
            gate_scores = gate.apply(prices).reindex(all_tickers)
            result[gate.name] = gate_scores

            # Ticker fails if gate returned NaN
            gate_mask = gate_mask & gate_scores.notna()

        passed_tickers = gate_mask[gate_mask].index.tolist()
        logger.info(
            "Gates complete — %d/%d tickers passed",
            len(passed_tickers), len(all_tickers),
        )

        if not passed_tickers:
            logger.warning("No tickers passed all gate filters.")
            result["final_score"] = float("nan")
            result["rank"]        = float("nan")
            return result

        # Subset prices to passing tickers only for scorer efficiency
        passing_prices = prices.loc[
            :, prices.columns.get_level_values(1).isin(passed_tickers)
        ]

        # ---- Step 2: Apply scoring filters ---------------------------
        if not self.scorers:
            # No scorers — every passing ticker gets equal score
            result.loc[passed_tickers, "final_score"] = 1.0
            logger.info("No scorers configured — all passing tickers score equally.")
        else:
            score_matrix = pd.DataFrame(index=passed_tickers)

            for scorer_filter in self.scorers:
                logger.info("Applying scorer: %s", scorer_filter.name)
                raw = scorer_filter.apply(passing_prices).reindex(passed_tickers)
                result.loc[passed_tickers, scorer_filter.name] = raw

                # Min-max normalise to [0, 1] so scales are comparable
                score_matrix[scorer_filter.name] = self._minmax(raw)

            weights = self._resolve_weights()
            result.loc[passed_tickers, "final_score"] = (
                score_matrix.mul(weights, axis=1).sum(axis=1) / sum(weights)
            )

        # ---- Step 3: Rank --------------------------------------------
        result["rank"] = (
            result["final_score"]
            .rank(ascending=False, method="min", na_option="keep")
            .where(result["final_score"].notna())
        )

        top5 = (
            result[["final_score", "rank"]]
            .dropna()
            .sort_values("rank")
            .head(5)
        )
        logger.info("Scoring complete — top 5:\n%s", top5.to_string())

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _minmax(self, s: pd.Series) -> pd.Series:
        """Normalise a Series to [0, 1]. Returns 1.0 everywhere if all equal."""
        valid = s.dropna()
        if valid.empty:
            return s
        lo, hi = valid.min(), valid.max()
        if hi == lo:
            return s.where(s.isna(), 1.0)
        return (s - lo) / (hi - lo)

    def _resolve_weights(self) -> List[float]:
        """Return weights list — equal if not specified."""
        return self.weights if self.weights else [1.0] * len(self.scorers)

    def __repr__(self) -> str:
        return (
            f"Scorer(gates={[f.name for f in self.gates]}, "
            f"scorers={[f.name for f in self.scorers]})"
        )