"""
backtest/result.py
------------------
Container returned by BacktestEngine.run().
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import pandas as pd

if TYPE_CHECKING:
    from backtest.portfolio import Portfolio

logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Container for all outputs of a backtest run.

    Attributes
    ----------
    equity_curve      : pd.Series  — daily NAV index (starts at 100.0)
    benchmark_equity  : pd.Series | None
    portfolio_returns : pd.Series  — daily return series
    rebalance_log     : pd.DataFrame — one row per rebalance date
    metrics           : dict       — strategy performance stats
    bench_metrics     : dict | None
    params            : dict       — configuration used
    portfolio         : Portfolio  — full account: holdings, trades, NAV history
    """

    def __init__(
        self,
        equity_curve:      pd.Series,
        portfolio_returns: pd.Series,
        rebalance_log:     pd.DataFrame,
        metrics:           dict,
        params:            dict,
        portfolio:         "Portfolio",
        benchmark_equity:  Optional[pd.Series] = None,
        bench_metrics:     Optional[dict]       = None,
    ):
        self.equity_curve      = equity_curve
        self.benchmark_equity  = benchmark_equity
        self.portfolio_returns = portfolio_returns
        self.rebalance_log     = rebalance_log
        self.metrics           = metrics
        self.bench_metrics     = bench_metrics
        self.params            = params
        self.portfolio         = portfolio

    # ------------------------------------------------------------------
    def summary(self) -> str:
        p  = self.params
        m  = self.metrics
        bm = self.bench_metrics

        sep = "─" * 55
        cur = p.get("currency", "")

        lines = [
            sep,
            f"  Backtest  ·  {p.get('universe', '')}  ({p.get('mode', '')} mode)",
            f"  {p.get('start', '')} → {p.get('end', '')}",
            f"  Top {p.get('top_n', '?')}  |  Rebalance: {p.get('rebalance', '')}  |  No-gates: {p.get('no_gates', False)}",
            f"  Initial Cash: {cur} {p.get('initial_cash', 0):>12,.0f}",
            sep,
            f"  {'Metric':<22}  {'Strategy':>10}  {'Benchmark':>10}",
            f"  {'──────':<22}  {'────────':>10}  {'─────────':>10}",
        ]

        def fmt_pct(v): return f"{v*100:+.2f}%" if v is not None else "─"
        def fmt_f(v):   return f"{v:.2f}"       if v is not None else "─"

        def row(label, key, is_pct=True):
            sv = m.get(key)
            bv = bm.get(key) if bm else None
            f  = fmt_pct if is_pct else fmt_f
            lines.append(f"  {label:<22}  {f(sv):>10}  {f(bv):>10}")

        row("Total Return",  "total_return")
        row("CAGR",          "cagr")
        row("Max Drawdown",  "max_drawdown")
        row("Sharpe Ratio",  "sharpe",     is_pct=False)
        row("Calmar Ratio",  "calmar",     is_pct=False)
        row("Years",         "n_years",    is_pct=False)

        lines.append(sep)

        nav_end    = self.equity_curve.iloc[-1]
        bench_str  = f"{self.benchmark_equity.iloc[-1]:.1f}" if self.benchmark_equity is not None else "─"
        lines.append(f"  {'NAV Index (start=100)':<22}  {nav_end:>10.1f}  {bench_str:>10}")
        lines.append(sep)

        # Portfolio snapshot
        port = self.portfolio
        lines += [
            f"  Final NAV         {cur} {port.nav:>12,.0f}",
            f"  Cash Remaining    {cur} {port.cash:>12,.0f}",
            f"  Realized P&L      {cur} {port.realized_pnl:>+12,.0f}",
            f"  Unrealized P&L    {cur} {port.unrealized_pnl:>+12,.0f}",
            f"  Total Trades      {len(port.transactions_df):>15}",
            f"  Closed Positions  {len(port.closed_positions_df):>15}",
            f"  Open Positions    {len(port._holdings):>15}",
        ]

        lines.append(sep)

        rb_total = len(self.rebalance_log)
        rb_skip  = int(self.rebalance_log["skipped"].sum()) if "skipped" in self.rebalance_log.columns else 0
        lines.append(f"  Rebalances    {rb_total}  (skipped: {rb_skip})")
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    def to_csv(self, output_dir: str = "output/backtests") -> Dict[str, Path]:
        """
        Save all outputs to CSV files.

        Files written:
            backtest_<name>_<ts>_equity.csv        — daily equity curve
            backtest_<name>_<ts>_rebalances.csv    — rebalance history
            portfolio_<ts>_holdings.csv            — current open positions
            portfolio_<ts>_transactions.csv        — all trades (buys + sells)
            portfolio_<ts>_closed_positions.csv    — closed round-trips with P&L
            portfolio_<ts>_nav_history.csv         — daily NAV snapshots
        """
        out  = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        univ = self.params.get("universe", "unknown").lower().replace(" ", "_")
        stem = f"backtest_{univ}_{ts}"

        paths: Dict[str, Path] = {}

        # ── equity curve ─────────────────────────────────────────────
        eq_path = out / f"{stem}_equity.csv"
        eq_df   = pd.DataFrame({
            "equity":  self.equity_curve,
            "returns": self.portfolio_returns.reindex(self.equity_curve.index).fillna(0.0),
        })
        if self.benchmark_equity is not None:
            eq_df["benchmark"] = self.benchmark_equity
        eq_df.index.name = "date"
        eq_df.to_csv(eq_path)
        paths["equity"] = eq_path
        logger.info("Saved equity       → %s", eq_path)

        # ── rebalance log ─────────────────────────────────────────────
        rb_path = out / f"{stem}_rebalances.csv"
        log = self.rebalance_log.copy()
        if "selected" in log.columns:
            log["selected"] = log["selected"].apply(
                lambda x: ",".join(x) if isinstance(x, list) else ""
            )
        log.to_csv(rb_path)
        paths["rebalances"] = rb_path
        logger.info("Saved rebalances   → %s", rb_path)

        # ── portfolio logs ────────────────────────────────────────────
        port_paths = self.portfolio.to_csv(output_dir=output_dir)
        paths.update(port_paths)

        return paths

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        m = self.metrics
        return (
            f"BacktestResult("
            f"CAGR={m.get('cagr', 0)*100:.2f}%, "
            f"MaxDD={m.get('max_drawdown', 0)*100:.2f}%, "
            f"NAV={self.equity_curve.iloc[-1]:.1f})"
        )
