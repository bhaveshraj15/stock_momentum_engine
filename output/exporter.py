"""
output/exporter.py
------------------
Exports scorer results and price data to multiple formats.

Supported formats:
    xlsx    → formatted Excel workbook (openpyxl)
    csv     → plain CSV for downstream tools
    numpy   → .npy matrix of Close prices or scores

Usage:
    from output.exporter import Exporter

    exporter = Exporter(output_dir="output/runs")

    # Export scorer result DataFrame
    exporter.to_excel(result, universe_name="nifty50")
    exporter.to_csv(result,   universe_name="nifty50")

    # Export raw price matrix as numpy
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
    # Excel export
    # ------------------------------------------------------------------

    def to_excel(
        self,
        result: pd.DataFrame,
        universe_name: str = "universe",
        filename: Optional[str] = None,
    ) -> Path:
        """
        Write the scorer result DataFrame to a formatted .xlsx file.

        Sheet layout:
            Row 1       : Title + run timestamp
            Row 2       : Column headers (bold, blue background)
            Row 3+      : Data rows
                          Passing tickers (rank not NaN) → white background
                          Gated-out tickers              → light gray background

        Parameters
        ----------
        result       : pd.DataFrame   Output of Scorer.run()
        universe_name: str            Used in filename and sheet title
        filename     : str, optional  Override auto-generated filename

        Returns
        -------
        Path to the written file.
        """
        from openpyxl import Workbook
        from openpyxl.styles import (
            Alignment, Font, PatternFill, Border, Side
        )
        from openpyxl.utils import get_column_letter

        path = self._build_path(universe_name, "xlsx", filename)

        wb = Workbook()
        ws = wb.active
        ws.title = "Momentum Scores"

        # ---- Style constants ----
        font_header  = Font(name="Arial", bold=True, color="FFFFFF", size=11)
        font_title   = Font(name="Arial", bold=True, size=13)
        font_normal  = Font(name="Arial", size=10)
        font_muted   = Font(name="Arial", size=10, color="999999")

        fill_header  = PatternFill("solid", start_color="1F4E79")   # dark blue
        fill_pass    = PatternFill("solid", start_color="FFFFFF")    # white
        fill_fail    = PatternFill("solid", start_color="F2F2F2")    # light gray
        fill_rank1   = PatternFill("solid", start_color="E2EFDA")    # light green — top rank

        thin_border  = Border(
            bottom=Side(style="thin", color="DDDDDD")
        )

        align_center = Alignment(horizontal="center", vertical="center")
        align_left   = Alignment(horizontal="left",   vertical="center")

        # ---- Row 1: Title ----
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        ws.merge_cells(f"A1:{get_column_letter(len(result.columns) + 1)}1")
        ws["A1"] = f"Momentum Engine — {universe_name.upper()}   |   {timestamp}"
        ws["A1"].font      = font_title
        ws["A1"].alignment = align_left
        ws.row_dimensions[1].height = 28

        # ---- Row 2: Headers ----
        headers = ["Ticker"] + list(result.columns)
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=2, column=col_idx, value=header)
            cell.font      = font_header
            cell.fill      = fill_header
            cell.alignment = align_center
        ws.row_dimensions[2].height = 22

        # ---- Rows 3+: Data ----
        # Sort: passing tickers by rank first, then gated-out alphabetically
        passing = result[result["rank"].notna()].sort_values("rank")
        failing = result[result["rank"].isna()].sort_index()
        sorted_result = pd.concat([passing, failing])

        for row_idx, (ticker, row) in enumerate(sorted_result.iterrows(), start=3):
            is_passing = pd.notna(row.get("rank"))
            is_rank1   = is_passing and row.get("rank") == 1.0

            fill = fill_rank1 if is_rank1 else (fill_pass if is_passing else fill_fail)
            font = font_normal if is_passing else font_muted

            # Ticker column
            cell = ws.cell(row=row_idx, column=1, value=ticker)
            cell.font      = Font(name="Arial", bold=True, size=10,
                                  color="000000" if is_passing else "999999")
            cell.fill      = fill
            cell.border    = thin_border
            cell.alignment = align_left

            # Data columns
            for col_idx, col_name in enumerate(result.columns, start=2):
                value = row[col_name]
                # Format floats
                if isinstance(value, float):
                    if col_name == "rank":
                        display = int(value) if pd.notna(value) else ""
                    elif col_name == "final_score":
                        display = round(value, 4) if pd.notna(value) else ""
                    else:
                        display = round(value, 4) if pd.notna(value) else ""
                else:
                    display = value if pd.notna(value) else ""

                cell = ws.cell(row=row_idx, column=col_idx, value=display)
                cell.font      = font
                cell.fill      = fill
                cell.border    = thin_border
                cell.alignment = align_center

            ws.row_dimensions[row_idx].height = 18

        # ---- Column widths ----
        ws.column_dimensions["A"].width = 22   # Ticker
        for col_idx, col_name in enumerate(result.columns, start=2):
            letter = get_column_letter(col_idx)
            if col_name == "rank":
                ws.column_dimensions[letter].width = 8
            elif col_name == "final_score":
                ws.column_dimensions[letter].width = 14
            else:
                ws.column_dimensions[letter].width = 18

        # ---- Freeze panes at row 3 ----
        ws.freeze_panes = "A3"

        wb.save(path)
        logger.info("Excel saved → %s  (%d rows)", path.name, len(result))
        return path

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(
        self,
        result: pd.DataFrame,
        universe_name: str = "universe",
        filename: Optional[str] = None,
        passing_only: bool = False,
    ) -> Path:
        """
        Write the scorer result DataFrame to a .csv file.

        Parameters
        ----------
        result       : pd.DataFrame   Output of Scorer.run()
        universe_name: str
        filename     : str, optional
        passing_only : bool           If True, only write tickers that passed all gates

        Returns
        -------
        Path to the written file.
        """
        path = self._build_path(universe_name, "csv", filename)

        df = result.copy()
        df.index.name = "Ticker"

        if passing_only:
            df = df[df["rank"].notna()]

        df = df.sort_values("rank", na_position="last")
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
    ) -> Path:
        if filename:
            return self.output_dir / filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{universe_name}_{timestamp}.{ext}"

    def __repr__(self) -> str:
        return f"Exporter(output_dir='{self.output_dir}')"