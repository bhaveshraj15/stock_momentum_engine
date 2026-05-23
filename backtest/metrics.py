"""
backtest/metrics.py
-------------------
Performance metrics derived from a NAV-indexed equity curve.
"""

import numpy as np
import pandas as pd
from typing import Dict


def compute_metrics(equity: pd.Series, rfr: float = 0.0) -> Dict:
    """
    Compute core performance metrics from a daily equity curve.

    Parameters
    ----------
    equity : pd.Series
        Daily NAV index (starts at 100.0 by convention).
    rfr : float
        Annual risk-free rate used for Sharpe calculation (default 0.0).

    Returns
    -------
    dict with keys:
        total_return, cagr, max_drawdown, n_years,
        sharpe, calmar
    """
    equity = equity.dropna()
    if len(equity) < 2:
        return {}

    n_years      = (equity.index[-1] - equity.index[0]).days / 365.25
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr         = (equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 1e-6)) - 1

    rolling_max  = equity.cummax()
    max_drawdown = (equity / rolling_max - 1).min()

    # Sharpe — annualised excess return / annualised volatility
    daily_returns = equity.pct_change().dropna()
    daily_rfr     = (1 + rfr) ** (1 / 252) - 1
    excess        = daily_returns - daily_rfr
    ann_vol       = daily_returns.std() * np.sqrt(252)
    sharpe        = (excess.mean() * 252) / ann_vol if ann_vol > 0 else float("nan")

    # Calmar — CAGR / |max drawdown|
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else float("nan")

    return {
        "total_return": round(total_return, 4),
        "cagr":         round(cagr,         4),
        "max_drawdown": round(max_drawdown,  4),
        "n_years":      round(n_years,       2),
        "sharpe":       round(sharpe,        3),
        "calmar":       round(calmar,        3),
    }
