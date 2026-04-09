"""
data/fetcher.py
---------------
Downloads OHLCV price data for a given universe using yfinance.

Design principles:
  - Pure yfinance + standard requests (no third-party HTTP overrides)
  - Local parquet cache to avoid repeated downloads
  - Returns a clean wide DataFrame: index=Date, columns=MultiIndex(field, ticker)
  - Graceful handling of missing / delisted tickers
  - Configurable via plain kwargs — no hidden globals

Typical usage:
    from data.loader  import UniverseLoader
    from data.fetcher import Fetcher

    tickers = UniverseLoader("config/universes/nifty50.yaml").get_tickers()

    fetcher = Fetcher(cache_dir="data/cache")
    prices  = fetcher.fetch(tickers, start="2022-01-01", end="2024-12-31")

    # prices is a pd.DataFrame with MultiIndex columns: (field, ticker)
    close   = prices["Close"]          # shape: (trading_days, n_tickers)
    volume  = prices["Volume"]
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Fields we keep from yfinance — everything else is dropped
_KEEP_FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# Minimum number of trading rows we require for a ticker to be kept
_MIN_ROWS = 20


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(tickers: List[str], start: str, end: str) -> str:
    """Deterministic filename based on universe + date range."""
    payload = json.dumps({"tickers": sorted(tickers), "start": start, "end": end})
    digest = hashlib.md5(payload.encode()).hexdigest()[:12]
    return f"ohlcv_{start}_{end}_{digest}.parquet"


def _load_cache(cache_path: Path) -> Optional[pd.DataFrame]:
    if cache_path.exists():
        logger.info("Cache hit  → %s", cache_path.name)
        return pd.read_parquet(cache_path)
    return None


def _save_cache(df: pd.DataFrame, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    logger.info("Cached     → %s", cache_path.name)


# ---------------------------------------------------------------------------
# Core download
# ---------------------------------------------------------------------------

def _download_batch(
    tickers: List[str],
    start: str,
    end: str,
    chunk_size: int,
    progress: bool,
) -> pd.DataFrame:
    """
    Download via yfinance in chunks to stay within rate limits.
    Returns a wide DataFrame with MultiIndex columns (field, ticker).
    """
    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    logger.info("Downloading %d tickers in %d chunk(s)…", len(tickers), len(chunks))

    for idx, chunk in enumerate(chunks, 1):
        logger.debug("  Chunk %d/%d — %s", idx, len(chunks), chunk)
        try:
            raw = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                auto_adjust=True,       # adjust for splits/dividends inline
                progress=progress,
                threads=True,           # parallel downloads within chunk
                # NOTE: no session= parameter — uses standard requests internally
            )
        except Exception as exc:
            logger.error("Chunk %d failed entirely: %s", idx, exc)
            failed.extend(chunk)
            continue

        if raw.empty:
            logger.warning("Chunk %d returned no data.", idx)
            failed.extend(chunk)
            continue

        # yfinance returns a MultiIndex DataFrame when >1 ticker is requested,
        # but a flat DataFrame for a single ticker — normalise both cases.
        if len(chunk) == 1:
            ticker = chunk[0]
            raw.columns = pd.MultiIndex.from_tuples(
                [(col, ticker) for col in raw.columns]
            )

        frames.append(raw)

    if not frames:
        raise RuntimeError("No data downloaded — all chunks failed.")

    combined = pd.concat(frames, axis=1)

    if failed:
        logger.warning("Tickers with no data: %s", failed)

    return combined


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    1. Keep only OHLCV fields.
    2. Drop tickers with too few rows (delisted / bad data).
    3. Forward-fill small gaps (≤5 days), then drop remaining NaNs.
    4. Ensure index is a proper DatetimeIndex.
    """
    # --- Keep known fields only ---
    available_fields = [f for f in _KEEP_FIELDS if f in df.columns.get_level_values(0)]
    df = df[available_fields]

    # --- Drop tickers below minimum row threshold (based on Close) ---
    close = df["Close"]
    valid_tickers = [
        t for t in close.columns
        if close[t].notna().sum() >= _MIN_ROWS
    ]
    dropped = set(close.columns) - set(valid_tickers)
    if dropped:
        logger.warning("Dropped tickers (insufficient data): %s", sorted(dropped))

    df = df.loc[:, df.columns.get_level_values(1).isin(valid_tickers)]

    # --- Forward-fill small gaps then drop rows where all tickers are NaN ---
    df = df.ffill(limit=5)
    df = df.dropna(how="all")

    # --- Ensure DatetimeIndex ---
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Public Fetcher class
# ---------------------------------------------------------------------------

class Fetcher:
    """
    Downloads and caches OHLCV data for any ticker universe.

    Parameters
    ----------
    cache_dir : str | Path
        Directory where parquet cache files are stored.
        Pass None to disable caching entirely.
    chunk_size : int
        Number of tickers per yfinance batch request (default 50).
        Lower this if you hit rate limits.
    progress : bool
        Show yfinance download progress bar (default False).
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = "data/cache",
        chunk_size: int = 50,
        progress: bool = False,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.chunk_size = chunk_size
        self.progress = progress

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fetch(
        self,
        tickers: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        lookback_days: int = 730,   # used only when start is None
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for the given tickers.

        Parameters
        ----------
        tickers : list of str
            Ticker symbols (e.g. ["RELIANCE.NS", "TCS.NS"]).
        start : str, optional
            Start date "YYYY-MM-DD". Defaults to today − lookback_days.
        end : str, optional
            End date "YYYY-MM-DD". Defaults to today.
        lookback_days : int
            Days of history when start is not specified (default 730 = ~2 years).
        force_refresh : bool
            Ignore cache and re-download even if a cached file exists.

        Returns
        -------
        pd.DataFrame
            MultiIndex columns: (field, ticker) where field ∈ {Open, High, Low, Close, Volume}.
            Index: DatetimeIndex of trading days.

        Example
        -------
        >>> fetcher = Fetcher()
        >>> df = fetcher.fetch(["TCS.NS", "INFY.NS"], start="2023-01-01")
        >>> close = df["Close"]  # shape: (trading_days, 2)
        """
        # --- Resolve dates ---
        end_str   = end   or datetime.today().strftime("%Y-%m-%d")
        start_str = start or (
            datetime.today() - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        tickers = [str(t).strip() for t in tickers if t]
        if not tickers:
            raise ValueError("ticker list is empty.")

        logger.info(
            "Fetcher.fetch — %d tickers | %s → %s",
            len(tickers), start_str, end_str,
        )

        # --- Check cache ---
        cache_path = None
        if self.cache_dir and not force_refresh:
            key = _cache_key(tickers, start_str, end_str)
            cache_path = self.cache_dir / key
            cached = _load_cache(cache_path)
            if cached is not None:
                return cached

        # --- Download ---
        raw = _download_batch(tickers, start_str, end_str, self.chunk_size, self.progress)

        # --- Clean ---
        df = _clean(raw, tickers)

        # --- Cache ---
        if self.cache_dir and cache_path:
            _save_cache(df, cache_path)

        logger.info(
            "Fetch complete — shape: %s, tickers retained: %d",
            df.shape,
            df.columns.get_level_values(1).nunique(),
        )
        return df

    # ------------------------------------------------------------------
    # Convenience exports
    # ------------------------------------------------------------------

    def get_close(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """Shortcut: returns only the Close price matrix."""
        return self.fetch(tickers, **kwargs)["Close"]

    def get_returns(self, tickers: List[str], **kwargs) -> pd.DataFrame:
        """Shortcut: returns daily log-returns of Close prices."""
        close = self.get_close(tickers, **kwargs)
        return np.log(close / close.shift(1)).dropna(how="all")

    def to_numpy(self, tickers: List[str], field: str = "Close", **kwargs) -> np.ndarray:
        """
        Return price data as a raw numpy matrix.
        Shape: (trading_days, n_tickers). Column order matches input tickers.
        """
        df = self.fetch(tickers, **kwargs)[field]
        # Reindex columns to match requested ticker order (drop any missing)
        available = [t for t in tickers if t in df.columns]
        return df[available].to_numpy()

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Delete all .parquet cache files managed by this fetcher."""
        if not self.cache_dir or not self.cache_dir.exists():
            logger.info("No cache directory found — nothing to clear.")
            return
        removed = 0
        for p in self.cache_dir.glob("ohlcv_*.parquet"):
            p.unlink()
            removed += 1
        logger.info("Cleared %d cache file(s) from %s", removed, self.cache_dir)

    def list_cache(self) -> List[Path]:
        """List all cached parquet files."""
        if not self.cache_dir or not self.cache_dir.exists():
            return []
        return sorted(self.cache_dir.glob("ohlcv_*.parquet"))

    def __repr__(self) -> str:
        return (
            f"Fetcher(cache_dir='{self.cache_dir}', "
            f"chunk_size={self.chunk_size})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python -m data.fetcher)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=f"%(levelname)s  %(message)s")

    test_tickers = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
    fetcher = Fetcher(cache_dir="./data/cache", progress=True)

    df = fetcher.fetch(test_tickers, lookback_days=90)
    print("\nDataFrame shape :", df.shape)
    print("Fields          :", df.columns.get_level_values(0).unique().tolist())
    print("Tickers         :", df.columns.get_level_values(1).unique().tolist())
    print("\nClose tail:")
    print(df["Close"].tail(3))

    arr = fetcher.to_numpy(test_tickers, field="Close", lookback_days=90)
    print("\nNumpy array shape:", arr.shape)
    print("Cached files     :", [p.name for p in fetcher.list_cache()])