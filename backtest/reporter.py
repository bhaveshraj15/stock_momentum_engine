"""
backtest/reporter.py
--------------------
Human-readable rebalance event formatter.

Produces a print block after each rebalance matching the style:

    ++++++++++++++ [ header ] ++++++++++++++

    Outgoing Stocks:  2021-02-02
    ['ASIANPAINT.NS', ...]

    SELL:  ASIANPAINT.NS at 2463.65 x 3 = 7390.95  P/L: -902.55  on 2021-02-02

    Cash after SELL: 41317.35    P/L: -878.85

    ---------------------------------------------------------------

    Cash/Stock: 10285.50 x 5

    Incoming Stocks:  2021-02-02
    ['ADANIPORTS.NS', ...]

    BUY:  ADANIPORTS.NS at 542.75 x 18 = 9769.50  on 2021-02-02

    ---------------------------------------------------------------

    Portfolio:
                         Buy_Date         Buy   Shares  Total_Value
    ADANIENT.NS        2021-01-01      479.55     20.0      9591.00

    Cash: 3393.99    Invested: 95727.16

    ---------------------------------------------------------------

    Total: 99121.15    Accumulated P/L: -878.85
"""

from __future__ import annotations

import pandas as pd

WIDTH = 100


def _plus(label: str = "") -> str:
    if label:
        pad   = WIDTH - len(label) - 2
        left  = pad // 2
        right = pad - left
        return "+" * left + " " + label + " " + "+" * right
    return "+" * WIDTH


def _dash() -> str:
    return "-" * WIDTH


def format_rebalance_block(
    date,
    rb_event: dict,
    portfolio,
) -> str:
    """
    Build a human-readable string for one rebalance event.

    Parameters
    ----------
    date       : rebalance date (anything pd.Timestamp accepts)
    rb_event   : dict returned by Portfolio.rebalance_to()
    portfolio  : Portfolio instance *after* the rebalance has been executed
    """
    sold           = rb_event.get("sold",  [])
    bought         = rb_event.get("bought", [])
    held           = rb_event.get("held",   [])
    cash_from_sells = rb_event.get("cash_from_sells", 0.0)
    sell_pnl       = rb_event.get("sell_pnl", 0.0)
    cash_available  = rb_event.get("cash_available", portfolio.cash)
    n_new           = rb_event.get("n_new", 0)

    date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
    lines: list[str] = []

    # ── Header ───────────────────────────────────────────────────────
    lines.append(_plus(date_str))
    lines.append("")

    # ── Outgoing / SELL section ───────────────────────────────────────
    if sold:
        out_tickers = [s["ticker"] for s in sold]
        lines.append(f"Outgoing Stocks:    {date_str}")
        lines.append(repr(out_tickers))
        lines.append("")
        for s in sold:
            lines.append(
                f"SELL:    {s['ticker']} at {s['price']} x {s['qty']}"
                f" = {s['sell_value']}"
                f"  P/L: {s['pnl']:+.2f}"
                f"  on {date_str}"
            )
        lines.append("")
        lines.append(
            f"Cash after SELL: {cash_available:.2f}    P/L: {sell_pnl:+.2f}"
        )
        lines.append("")
    else:
        lines.append(f"No exits on {date_str}")
        lines.append("")

    lines.append(_dash())
    lines.append("")

    # ── Incoming / BUY section ────────────────────────────────────────
    if bought and n_new > 0:
        cash_per = cash_available / n_new
        lines.append(f"Cash/Stock: {cash_per:.2f} x {n_new}")
        lines.append("")
        in_tickers = [b["ticker"] for b in bought]
        lines.append(f"Incoming Stocks:    {date_str}")
        lines.append(repr(in_tickers))
        lines.append("")
        for b in bought:
            lines.append(
                f"BUY:    {b['ticker']} at {b['price']} x {b['qty']}"
                f" = {b['cost']}"
                f"  on {date_str}"
            )
        lines.append("")
    elif held and not sold and not bought:
        lines.append(f"No changes — holding {len(held)} existing position(s).")
        lines.append("")
    else:
        lines.append("No new entries this period.")
        lines.append("")

    lines.append(_dash())
    lines.append("")

    # ── Portfolio snapshot table ──────────────────────────────────────
    lines.append("Portfolio:")
    h_df = portfolio.holdings_df
    if not h_df.empty:
        col_w = max(len(t) for t in h_df.index) + 2
        header = (
            f"  {'':>{col_w}}"
            f"  {'Buy_Date':>12}"
            f"  {'Buy':>10}"
            f"  {'Shares':>8}"
            f"  {'Total_Value':>12}"
        )
        lines.append(header)
        for ticker, row in h_df.iterrows():
            od  = str(row["open_date"])[:10]
            lines.append(
                f"  {ticker:>{col_w}}"
                f"  {od:>12}"
                f"  {row['avg_cost']:>10.2f}"
                f"  {row['qty']:>8}"
                f"  {row['cost_basis']:>12.2f}"
            )
    lines.append("")

    invested = portfolio.invested_at_cost
    lines.append(f"Cash: {portfolio.cash:.2f}    Invested: {invested:.2f}")
    lines.append("")

    lines.append(_dash())
    lines.append("")

    total           = portfolio.cash + invested
    accumulated_pnl = portfolio.realized_pnl
    lines.append(f"Total:    {total:.2f}    Accumulated P/L: {accumulated_pnl:+.2f}")
    lines.append("")

    return "\n".join(lines)
