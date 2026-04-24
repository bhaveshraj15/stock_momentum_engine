"""
output/exporter.py
------------------
Exports scorer results and price data to multiple formats.

Supported formats:
    csv     → plain CSV for downstream tools
    numpy   → .npy matrix of Close prices or scores

Usage:
    from output.exporter import Exporter

    exporter = Exporter(output_dir="output/runs")

    exporter.to_csv(result, universe_name="nifty50")
    exporter.to_numpy(prices, field="Close", universe_name="nifty50")
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = "output/runs"


class Exporter:
    """
    Writes scorer output and price data to disk in multiple formats.

    Parameters
    ----------
    output_dir : str | Path
        Directory where output files are written.
        Created automatically if it does not exist.
    """

    def __init__(self, output_dir: str | Path = _DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(
        self,
        result: pd.DataFrame,
        universe_name: str = "universe",
        filename: Optional[str] = None,
        passing_only: bool = False,
        top_n: Optional[int] = None,
        mode: str = "",
    ) -> Path:
        """
        Write the scorer result DataFrame to a .csv file.

        Parameters
        ----------
        result       : pd.DataFrame   Output of Scorer.run()
        universe_name: str
        filename     : str, optional
        passing_only : bool           If True, only write tickers that passed all gates
        top_n        : int, optional  If set, only write the top N ranked tickers
        mode         : str            Scoring mode included in filename (e.g. "returns")

        Returns
        -------
        Path to the written file.
        """
        path = self._build_path(universe_name, "csv", filename, mode=mode)

        df = result.copy()
        df.index.name = "Ticker"

        if passing_only:
            df = df[df["rank"].notna()]

        df = df.sort_values("rank", na_position="last")

        if top_n:
            df = df.head(top_n)

        df.to_csv(path)

        logger.info("CSV saved → %s  (%d rows)", path.name, len(df))
        return path

    # ------------------------------------------------------------------
    # Numpy export
    # ------------------------------------------------------------------

    def to_numpy(
        self,
        prices: pd.DataFrame,
        field: str = "Close",
        universe_name: str = "universe",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save a price field as a raw .npy matrix.

        Shape: (trading_days, n_tickers)
        Column order is alphabetical by ticker.
        A companion .csv metadata file is written alongside,
        recording ticker order and date range.

        Parameters
        ----------
        prices       : pd.DataFrame   MultiIndex OHLCV from Fetcher
        field        : str            Price field to extract (default "Close")
        universe_name: str
        filename     : str, optional

        Returns
        -------
        Path to the .npy file.
        """
        path = self._build_path(universe_name, "npy", filename)

        if field not in prices.columns.get_level_values(0):
            raise KeyError(
                f"Field '{field}' not found. "
                f"Available: {prices.columns.get_level_values(0).unique().tolist()}"
            )

        matrix = prices[field].sort_index(axis=1)   # alphabetical tickers
        arr    = matrix.to_numpy()

        np.save(path, arr)

        # Write metadata so the matrix can be reconstructed
        meta_path = path.with_suffix(".meta.csv")
        meta = pd.DataFrame({
            "ticker":     matrix.columns.tolist(),
            "col_index":  range(len(matrix.columns)),
        })
        meta.to_csv(meta_path, index=False)

        logger.info(
            "Numpy saved → %s  shape=%s  meta → %s",
            path.name, arr.shape, meta_path.name,
        )
        return path

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _build_path(
        self,
        universe_name: str,
        ext: str,
        filename: Optional[str],
        mode: str = "",
    ) -> Path:
        if filename:
            return self.output_dir / filename
        date = datetime.now().strftime("%Y%m%d")
        suffix = f"_{mode}" if mode else ""
        return self.output_dir / f"{universe_name}_{date}{suffix}.{ext}"

    def __repr__(self) -> str:
        return f"Exporter(output_dir='{self.output_dir}')"