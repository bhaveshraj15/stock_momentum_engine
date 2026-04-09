"""
data/loader.py
--------------
Loads a stock/ETF universe from a YAML config file.

Supports:
  - Inline ticker lists in YAML
  - Loading tickers from a CSV file (column mapping configurable)
  - Auto-appending exchange suffixes (e.g. .NS, .DE)
  - Returning universe metadata alongside tickers

Usage:
    from data.loader import UniverseLoader

    loader = UniverseLoader("config/universes/etf_india.yaml")
    tickers = loader.get_tickers()
    meta    = loader.get_meta()

    # Or one-liner:
    tickers = UniverseLoader.from_yaml("config/universes/nifty50.yaml").get_tickers()
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

import yaml
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_suffix(ticker: str, suffix: str) -> str:
    """Append exchange suffix if the ticker doesn't already carry one."""
    if not suffix:
        return ticker
    return ticker if ticker.endswith(suffix) else ticker + suffix


def _load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Universe config not found: {path}")
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class UniverseLoader:
    """
    Loads and validates a trading universe from a YAML config file.

    YAML schema
    -----------
    meta:
      name:        str          # human-readable label
      exchange:    str          # e.g. "NSE", "NYSE", "XETRA"
      currency:    str          # e.g. "INR", "USD", "EUR"
      suffix:      str          # ticker suffix, e.g. ".NS"  (can be empty "")
      description: str          # optional free-text

    tickers:                    # Option A — inline list
      - RELIANCE.NS
      - TCS.NS
      ...

    # OR

    csv_source:                 # Option B — load from CSV
      path:          str        # path to CSV file
      ticker_column: str        # column name that holds the ticker symbol
      suffix:        str        # override meta.suffix for CSV tickers (optional)
    """

    def __init__(self, config_path: str | Path):
        self._config_path = Path(config_path)
        self._raw: dict = _load_yaml(self._config_path)
        self._meta: dict = self._raw.get("meta", {})
        self._tickers: List[str] = self._resolve_tickers()
        logger.info(
            "Loaded universe '%s' — %d tickers from %s",
            self._meta.get("name", "unnamed"),
            len(self._tickers),
            self._config_path.name,
        )

    # ------------------------------------------------------------------
    # Class-method constructor (convenience)
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "UniverseLoader":
        """Alias for the constructor — improves readability at call sites."""
        return cls(config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tickers(self) -> List[str]:
        """Return the validated, suffix-corrected list of ticker symbols."""
        return list(self._tickers)  # return a copy

    def get_meta(self) -> Dict:
        """Return universe metadata (name, exchange, currency, etc.)."""
        return dict(self._meta)

    def get_name(self) -> str:
        return self._meta.get("name", self._config_path.stem)

    def get_suffix(self) -> str:
        return self._meta.get("suffix", "")

    def get_currency(self) -> str:
        return self._meta.get("currency", "")

    def __len__(self) -> int:
        return len(self._tickers)

    def __repr__(self) -> str:
        return (
            f"UniverseLoader(name='{self.get_name()}', "
            f"tickers={len(self._tickers)}, "
            f"source='{self._config_path.name}')"
        )

    # ------------------------------------------------------------------
    # Internal resolution
    # ------------------------------------------------------------------

    def _resolve_tickers(self) -> List[str]:
        """
        Resolve tickers from either inline YAML list or a CSV source.
        Returns a deduplicated, suffix-normalised list.
        """
        suffix = self._meta.get("suffix", "")

        # --- Option A: inline ticker list ---
        if "tickers" in self._raw:
            raw_list: List[str] = self._raw["tickers"]
            if not isinstance(raw_list, list):
                raise ValueError("'tickers' in YAML must be a list of strings.")
            tickers = [_ensure_suffix(str(t).strip(), suffix) for t in raw_list if t]

        # --- Option B: CSV source ---
        elif "csv_source" in self._raw:
            tickers = self._load_from_csv(suffix)

        else:
            raise ValueError(
                f"Universe config '{self._config_path}' must contain either "
                "'tickers' (inline list) or 'csv_source' (CSV path)."
            )

        # Deduplicate while preserving order
        seen, unique = set(), []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        # Warn on empty universe
        if not unique:
            logger.warning("Universe '%s' resolved to 0 tickers!", self.get_name())

        return unique

    def _load_from_csv(self, default_suffix: str) -> List[str]:
        """
        Load tickers from a CSV file specified in the 'csv_source' block.

        csv_source:
          path:          "data/ind_nifty50list.csv"
          ticker_column: "Symbol"          # column holding raw symbols
          suffix:        ".NS"             # overrides meta.suffix if set
        """
        csv_cfg: dict = self._raw["csv_source"]

        csv_path = Path(csv_cfg.get("path", ""))
        if not csv_path.exists():
            # Try relative to the YAML config location
            csv_path = self._config_path.parent / csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV source not found: {csv_cfg.get('path')}")

        col: str = csv_cfg.get("ticker_column", "Symbol")
        suffix: str = csv_cfg.get("suffix", default_suffix)

        df = pd.read_csv(csv_path)
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

        tickers = [
            _ensure_suffix(str(sym).strip(), suffix)
            for sym in df[col].dropna()
            if str(sym).strip()
        ]
        logger.info("Loaded %d tickers from CSV: %s", len(tickers), csv_path.name)
        return tickers


# ---------------------------------------------------------------------------
# Utility: list available universe configs
# ---------------------------------------------------------------------------

def list_universes(config_dir: str | Path = "config/universes") -> List[Path]:
    """
    Return a list of all .yaml universe config files in the given directory.

    Example:
        for p in list_universes():
            print(UniverseLoader(p))
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        logger.warning("Universe config directory not found: %s", config_dir)
        return []
    return sorted(config_dir.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python -m data.loader)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    base = Path(__file__).parent.parent / "config" / "universes"
    for yaml_path in list_universes(base):
        try:
            ul = UniverseLoader(yaml_path)
            print(ul)
        except Exception as exc:
            print(f"  ERROR loading {yaml_path.name}: {exc}")