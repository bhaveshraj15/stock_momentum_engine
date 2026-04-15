"""
filters/volume_filter.py
-------------------------
Volume filter: three modes for different stages of the pipeline.

Directly incorporates the notebook condition:
    one_year_avg_vol > 1e5   ← mode="gate"

Three modes
-----------

MODE 1 — "gate"
    Hard exclude low-volume / illiquid tickers.
    Ticker fails if avg volume over window < min_avg_volume.
    Runs as a gate filter alongside TrendFilter etc.

    Params:
        min_avg_volume : float   Minimum average daily volume (default 1e5)
        window_days    : int     Lookback for average (default 252 = 1 year)

MODE 2 — "confirm"
    Boolean gate using 3-window volume pattern.
    Passes only if volume is genuinely building:
        vol_21d > vol_63d > vol_252d
    Removes tickers with fake pumps (spike-only, no sustained buildup).
    Runs as a gate filter.

    Params:
        window_days : int   Annual baseline window (default 252)

MODE 3 — "score"
    Continuous scorer — amplifies or penalises momentum score based
    on whether price move is backed by volume.

    Formula:
        vol_trend    = vol_21d / vol_252d       (recent vs annual)
        consistency  = vol_63d / vol_252d       (medium vs annual)
        vol_score    = vol_trend * sqrt(consistency)

    Effect:
        Price up + volume building   → score amplified (vol_score > 1)
        Price up + flat volume       → score neutral   (vol_score ≈ 1)
        Price up + declining volume  → score penalised (vol_score < 1)
        Price up + spike only        → partially penalised

    Runs as a scoring filter alongside MomentumFilter.

WHERE IT FITS IN THE PIPELINE
──────────────────────────────

    Full universe
        ↓
    CorrelationFilter                     ← pre-screen
        ↓
    TrendFilter                           ← gate
    High52wFilter                         ← gate
    MinReturnFilter                       ← gate
    UpDaysFilter                          ← gate
    VolumeFilter(mode="gate")             ← gate  ← remove illiquid
    VolumeFilter(mode="confirm")          ← gate  ← remove fake pumps
        ↓
    MomentumFilter                        ← scorer
    VolumeFilter(mode="score")            ← scorer ← amplify/penalise
        ↓
    DiversificationFilter                 ← post-dedup

Usage:
    from filters.volume_filter import VolumeFilter

    # Gate: remove illiquid tickers (< 100k daily volume)
    vg = VolumeFilter(params={"mode": "gate", "min_avg_volume": 1e5})

    # Confirm: only pass tickers with genuine volume buildup
    vc = VolumeFilter(params={"mode": "confirm"})

    # Score: amplify/penalise based on volume trend
    vs = VolumeFilter(params={"mode": "score"})
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from filters.base_filter import BaseFilter

logger = logging.getLogger(__name__)

_DEFAULT_MIN_AVG_VOLUME = 1e5    # 100,000 units — matches notebook exactly
_DEFAULT_WINDOW_DAYS    = 252    # 1 year


class VolumeFilter(BaseFilter):
    """
    Volume filter with three modes: gate, confirm, score.

    See module docstring for full details on each mode.
    """

    def _validate_params(self) -> None:
        mode = self.params.get("mode", "gate")
        if mode not in ("gate", "confirm", "score"):
            raise ValueError(
                f"VolumeFilter: mode must be 'gate', 'confirm', or 'score'. "
                f"Got '{mode}'."
            )
        if mode == "gate":
            min_vol = self.params.get("min_avg_volume", _DEFAULT_MIN_AVG_VOLUME)
            if min_vol <= 0:
                raise ValueError(
                    f"VolumeFilter: min_avg_volume must be > 0, got {min_vol}."
                )

    # ------------------------------------------------------------------
    # BaseFilter interface
    # ------------------------------------------------------------------

    def compute(self, prices: pd.DataFrame) -> pd.Series:
        mode = self.params.get("mode", "gate")

        if mode == "gate":
            return self._compute_gate(prices)
        elif mode == "confirm":
            return self._compute_confirm(prices)
        elif mode == "score":
            return self._compute_score(prices)

    def filter(self, scores: pd.Series) -> pd.Series:
        mode = self.params.get("mode", "gate")

        if mode in ("gate", "confirm"):
            # Hard gate — NaN means excluded
            return pd.Series(scores.notna(), index=scores.index)
        else:
            # Score mode — all non-NaN scores pass through
            return pd.Series(scores.notna(), index=scores.index)

    # ------------------------------------------------------------------
    # Mode 1 — gate: remove illiquid tickers
    # ------------------------------------------------------------------

    def _compute_gate(self, prices: pd.DataFrame) -> pd.Series:
        """
        Returns 1.0 if avg volume >= min_avg_volume, else NaN.
        Matches notebook condition: one_year_avg_vol > 1e5
        """
        volume     = self._get_volume(prices)
        min_vol    = self.params.get("min_avg_volume", _DEFAULT_MIN_AVG_VOLUME)
        window     = self.params.get("window_days",    _DEFAULT_WINDOW_DAYS)

        results = {}
        for ticker in volume.columns:
            series = volume[ticker].dropna()
            if series.empty:
                results[ticker] = float("nan")
                continue

            window_data = series.iloc[-window:] if len(series) >= window else series
            avg_vol     = window_data.mean()
            passes      = bool(avg_vol >= min_vol)
            results[ticker] = 1.0 if passes else float("nan")

            logger.debug(
                "%s | %-20s  AvgVol=%.0f  Min=%.0f  → %s",
                self.name, ticker, avg_vol, min_vol,
                "PASS" if passes else "FAIL",
            )

        passed = sum(1 for v in results.values() if v == 1.0)
        logger.info(
            "%s (gate): %d/%d tickers passed avg-volume >= %.0f gate",
            self.name, passed, len(results), min_vol,
        )
        return pd.Series(results, name=self.name)

    # ------------------------------------------------------------------
    # Mode 2 — confirm: genuine volume buildup pattern
    # ------------------------------------------------------------------

    def _compute_confirm(self, prices: pd.DataFrame) -> pd.Series:
        """
        Returns 1.0 only if vol_21d > vol_63d > vol_252d (genuine buildup).
        Rejects spike-only patterns where medium-term volume hasn't grown.
        """
        volume  = self._get_volume(prices)
        results = {}

        for ticker in volume.columns:
            series = volume[ticker].dropna()

            if len(series) < 252:
                logger.debug(
                    "%s: %s — only %d rows, need 252. Skipping confirm.",
                    self.name, ticker, len(series),
                )
                results[ticker] = float("nan")
                continue

            vol_21d  = series.iloc[-21:].mean()
            vol_63d  = series.iloc[-63:].mean()
            vol_252d = series.iloc[-252:].mean()

            # Genuine buildup: all three windows increasing
            passes = bool(vol_21d > vol_63d > vol_252d)
            results[ticker] = 1.0 if passes else float("nan")

            logger.debug(
                "%s | %-20s  21d=%.0f  63d=%.0f  252d=%.0f  → %s",
                self.name, ticker,
                vol_21d, vol_63d, vol_252d,
                "PASS" if passes else "FAIL",
            )

        passed = sum(1 for v in results.values() if v == 1.0)
        logger.info(
            "%s (confirm): %d/%d tickers passed vol_21d > vol_63d > vol_252d",
            self.name, passed, len(results),
        )
        return pd.Series(results, name=self.name)

    # ------------------------------------------------------------------
    # Mode 3 — score: volume-confirms-price signal
    # ------------------------------------------------------------------

    def _compute_score(self, prices: pd.DataFrame) -> pd.Series:
        """
        Returns a continuous multiplier score:
            vol_trend   = vol_21d  / vol_252d   (recent vs annual)
            consistency = vol_63d  / vol_252d   (medium vs annual)
            score       = vol_trend * sqrt(consistency)

        score > 1 → price move backed by growing volume (amplify)
        score < 1 → price move on declining volume (penalise)
        score = 1 → flat volume (neutral)
        """
        volume  = self._get_volume(prices)
        results = {}

        for ticker in volume.columns:
            series = volume[ticker].dropna()

            if len(series) < 252:
                logger.debug(
                    "%s: %s — only %d rows, need 252 for score. "
                    "Returning neutral score 1.0.",
                    self.name, ticker, len(series),
                )
                results[ticker] = 1.0   # neutral — don't penalise for lack of data
                continue

            vol_21d  = series.iloc[-21:].mean()
            vol_63d  = series.iloc[-63:].mean()
            vol_252d = series.iloc[-252:].mean()

            if vol_252d == 0:
                results[ticker] = float("nan")
                continue

            vol_trend   = vol_21d  / vol_252d
            consistency = vol_63d  / vol_252d

            # Protect against negative sqrt input
            consistency_safe = max(consistency, 0.0)
            score = vol_trend * np.sqrt(consistency_safe)

            results[ticker] = round(score, 4)

            logger.debug(
                "%s | %-20s  trend=%.3f  consistency=%.3f  score=%.4f",
                self.name, ticker, vol_trend, consistency, score,
            )

        logger.info(
            "%s (score): computed volume score for %d tickers",
            self.name, len(results),
        )
        return pd.Series(results, name=self.name)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_volume(self, prices: pd.DataFrame) -> pd.DataFrame:
        fields = prices.columns.get_level_values(0).unique().tolist()
        if "Volume" not in fields:
            raise KeyError(
                f"VolumeFilter requires 'Volume' field. "
                f"Available: {fields}"
            )
        return prices["Volume"]