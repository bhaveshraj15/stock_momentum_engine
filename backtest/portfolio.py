"""
backtest/portfolio.py
---------------------
Portfolio — simulates a DMAT account with cash, holdings, and trade logs.

Tracks four things at all times:
    1. Cash balance
    2. Open holdings   (qty, avg cost, current mark-to-market)
    3. All transactions (every buy and sell ever executed)
    4. Closed positions (completed round-trips with realized P&L)
    5. NAV history      (daily snapshot: cash + market value)

Usage (standalone):
    from backtest.portfolio import Portfolio

    port = Portfolio(initial_cash=1_000_000)

    port.buy("TCS.NS",      qty=10, price=3800.0, date="2024-01-02")
    port.buy("INFY.NS",     qty=15, price=1700.0, date="2024-01-02")

    port.update_prices({"TCS.NS": 3950.0, "INFY.NS": 1720.0}, date="2024-01-03")

    print(port.holdings_df)
    print(port.nav)

Usage (inside BacktestEngine):
    port.rebalance_to(["TCS.NS","INFY.NS","WIPRO.NS"], price_dict, date)
    port.update_prices(price_dict, date)   # call every trading day
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Simulates a DMAT brokerage account.

    Parameters
    ----------
    initial_cash : float
        Starting cash balance.
    currency : str
        Display label only (default "INR").
    fees_pct : float
        Fee as a fraction of trade value, applied on each side.
        0.001 = 0.1%.  Default 0.0 (no fees).
    """

    def __init__(
        self,
        initial_cash: float,
        currency:     str   = "INR",
        fees_pct:     float = 0.0,
    ):
        if initial_cash <= 0:
            raise ValueError(f"initial_cash must be > 0, got {initial_cash}")

        self.initial_cash = float(initial_cash)
        self.cash         = float(initial_cash)
        self.currency     = currency
        self.fees_pct     = fees_pct

        # ── Internal state ─────────────────────────────────────────
        # { ticker: {qty, avg_cost, last_price, last_updated, open_date} }
        self._holdings: Dict[str, dict] = {}

        # List of every buy / sell executed
        self._transactions: List[dict] = []

        # Completed round-trips (qty reaches 0 after last sell)
        self._closed_positions: List[dict] = []

        # Daily NAV snapshots
        self._nav_history: List[dict] = []

    # ================================================================
    # Core trade operations
    # ================================================================

    def buy(
        self,
        ticker: str,
        qty:    int,
        price:  float,
        date,
        fees:   Optional[float] = None,
    ) -> None:
        """
        Execute a buy order.

        Parameters
        ----------
        ticker : str
        qty    : int    Integer number of shares.
        price  : float  Execution price per share.
        date   : date-like
        fees   : float  Override calculated fees (optional).
        """
        if qty <= 0:
            raise ValueError(f"buy qty must be > 0, got {qty}")
        if price <= 0:
            raise ValueError(f"buy price must be > 0, got {price}")

        fees = float(fees) if fees is not None else price * qty * self.fees_pct
        total_cost = price * qty + fees

        if total_cost > self.cash + 1e-6:
            raise ValueError(
                f"Insufficient cash to buy {qty} × {ticker} @ {price:.2f}  "
                f"(need {total_cost:.2f}, have {self.cash:.2f})"
            )

        self.cash -= total_cost

        if ticker in self._holdings:
            h = self._holdings[ticker]
            new_qty      = h["qty"] + qty
            h["avg_cost"] = (h["qty"] * h["avg_cost"] + qty * price) / new_qty
            h["qty"]      = new_qty
        else:
            self._holdings[ticker] = {
                "qty":          qty,
                "avg_cost":     price,
                "last_price":   price,
                "last_updated": date,
                "open_date":    date,
            }

        self._holdings[ticker]["last_price"]   = price
        self._holdings[ticker]["last_updated"] = date

        self._transactions.append({
            "date":       date,
            "ticker":     ticker,
            "action":     "BUY",
            "qty":        qty,
            "price":      round(price, 4),
            "value":      round(qty * price, 2),
            "fees":       round(fees, 2),
            "avg_cost":   round(self._holdings[ticker]["avg_cost"], 4),
            "cash_after": round(self.cash, 2),
        })

        logger.debug("BUY  %s  qty=%d  @%.2f  cash_after=%.2f", ticker, qty, price, self.cash)

    def sell(
        self,
        ticker: str,
        qty:    int,
        price:  float,
        date,
        fees:   Optional[float] = None,
    ) -> None:
        """
        Execute a sell order.  Records realized P&L.
        When remaining qty reaches 0, moves to closed_positions.
        """
        if ticker not in self._holdings:
            raise ValueError(f"No open position in {ticker}")
        if qty <= 0:
            raise ValueError(f"sell qty must be > 0, got {qty}")
        if price <= 0:
            raise ValueError(f"sell price must be > 0, got {price}")

        h = self._holdings[ticker]
        if qty > h["qty"]:
            raise ValueError(
                f"Trying to sell {qty} of {ticker} but only {h['qty']} held"
            )

        fees         = float(fees) if fees is not None else price * qty * self.fees_pct
        proceeds     = price * qty - fees
        avg_cost     = h["avg_cost"]
        realized_pnl = (price - avg_cost) * qty - fees
        realized_pct = (price / avg_cost - 1) * 100 if avg_cost > 0 else 0.0

        self.cash += proceeds
        h["qty"] -= qty

        self._transactions.append({
            "date":              date,
            "ticker":            ticker,
            "action":            "SELL",
            "qty":               qty,
            "price":             round(price, 4),
            "value":             round(qty * price, 2),
            "fees":              round(fees, 2),
            "avg_cost":          round(avg_cost, 4),
            "proceeds":          round(proceeds, 2),
            "realized_pnl":      round(realized_pnl, 2),
            "realized_pnl_pct":  round(realized_pct, 4),
            "cash_after":        round(self.cash, 2),
        })

        if h["qty"] == 0:
            # Full close — move to closed positions log
            open_date  = h["open_date"]
            cost_basis = avg_cost * qty
            self._closed_positions.append({
                "ticker":             ticker,
                "open_date":          open_date,
                "close_date":         date,
                "qty":                qty,
                "avg_buy_price":      round(avg_cost, 4),
                "sell_price":         round(price, 4),
                "cost_basis":         round(cost_basis, 2),
                "proceeds":           round(proceeds, 2),
                "realized_pnl":       round(realized_pnl, 2),
                "realized_pnl_pct":   round(realized_pct, 4),
                "holding_days":       (pd.Timestamp(date) - pd.Timestamp(open_date)).days,
            })
            del self._holdings[ticker]
            logger.debug(
                "CLOSED %s  pnl=%.2f (%.2f%%)  held=%d days",
                ticker, realized_pnl, realized_pct,
                self._closed_positions[-1]["holding_days"],
            )
        else:
            h["last_price"]   = price
            h["last_updated"] = date

        logger.debug("SELL %s  qty=%d  @%.2f  pnl=%.2f  cash_after=%.2f",
                     ticker, qty, price, realized_pnl, self.cash)

    # ================================================================
    # Price update & NAV snapshot
    # ================================================================

    def update_prices(self, price_dict: Dict[str, float], date) -> None:
        """
        Mark holdings to market and record a NAV snapshot.
        Call once per trading day.

        Parameters
        ----------
        price_dict : {ticker: close_price}  — NaN prices are ignored.
        date       : date of the snapshot.
        """
        for ticker, h in self._holdings.items():
            p = price_dict.get(ticker)
            if p and pd.notna(p) and p > 0:
                h["last_price"]   = p
                h["last_updated"] = date

        self._nav_history.append({
            "date":           date,
            "nav":            round(self.nav, 2),
            "cash":           round(self.cash, 2),
            "invested":       round(self.invested_value, 2),
            "n_positions":    len(self._holdings),
            "realized_pnl":   round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.unrealized_pnl, 2),
        })

    # ================================================================
    # Rebalancing
    # ================================================================

    def rebalance_to(
        self,
        target_tickers: List[str],
        price_dict:     Dict[str, float],
        date,
    ) -> Dict:
        """
        Rebalance to the new target ticker list.

        Algorithm
        ---------
        1. Mark all holdings to current prices.
        2. Sell positions that are NOT in the new target list.
        3. Split the proceeds equally across tickers that are NEW to the portfolio.
        4. Leave existing holdings that are still in the target list untouched.

        Returns a rich event dict consumed by the reporter.
        """
        valid: Dict[str, float] = {
            t: price_dict[t]
            for t in target_tickers
            if t in price_dict and pd.notna(price_dict.get(t)) and price_dict[t] > 0
        }

        if not valid:
            logger.warning("rebalance_to: no valid prices available on %s", date)
            return {"sold": [], "bought": [], "held": [],
                    "cash_from_sells": 0.0, "sell_pnl": 0.0,
                    "cash_available": self.cash, "n_new": 0}

        # ── Step 1: mark current holdings to market ───────────────────
        for ticker, h in self._holdings.items():
            p = price_dict.get(ticker)
            if p and pd.notna(p) and p > 0:
                h["last_price"] = p

        current = set(self._holdings.keys())
        target  = set(valid.keys())
        to_exit  = current - target
        to_enter = target  - current
        to_hold  = current & target

        # ── Step 2: sell exits — capture details before the sell ──────
        cash_before  = self.cash
        sold_details = []

        for ticker in sorted(to_exit):
            h        = self._holdings[ticker]
            qty      = h["qty"]
            avg_cost = h["avg_cost"]
            price    = price_dict.get(ticker, h["last_price"])
            self.sell(ticker, qty, price, date)
            sold_details.append({
                "ticker":     ticker,
                "qty":        qty,
                "price":      price,
                "sell_value": round(qty * price, 2),
                "pnl":        round((price - avg_cost) * qty, 2),
            })

        cash_from_sells = self.cash - cash_before   # total sell proceeds
        cash_available  = self.cash                 # used for "Cash/Stock" calc
        n_new           = len(to_enter)

        # ── Step 3: buy new entries with proceeds ─────────────────────
        bought_details = []

        if to_enter and cash_available > 0:
            cash_per_new = cash_available / n_new
            for ticker in sorted(to_enter):
                price = valid[ticker]
                qty   = int(cash_per_new / price)
                if qty <= 0:
                    logger.warning(
                        "rebalance_to: %.2f insufficient for %s @ %.2f — skipped",
                        cash_per_new, ticker, price,
                    )
                    continue
                if qty * price > self.cash:
                    qty = int(self.cash / price)
                if qty > 0:
                    cost = qty * price
                    self.buy(ticker, qty, price, date)
                    bought_details.append({
                        "ticker": ticker,
                        "qty":    qty,
                        "price":  price,
                        "cost":   round(cost, 2),
                    })

        sell_pnl = sum(s["pnl"] for s in sold_details)
        logger.info(
            "Rebalance %s  |  sold=%d  bought=%d  held=%d  proceeds=%.0f  pnl=%+.0f  cash=%.0f",
            date, len(sold_details), len(bought_details), len(to_hold),
            cash_from_sells, sell_pnl, self.cash,
        )

        return {
            "sold":            sold_details,
            "bought":          bought_details,
            "held":            sorted(to_hold),
            "cash_from_sells": cash_from_sells,
            "sell_pnl":        sell_pnl,
            "cash_available":  cash_available,
            "n_new":           n_new,
        }

    # ================================================================
    # Properties
    # ================================================================

    @property
    def nav(self) -> float:
        """Current Net Asset Value = cash + market value of all holdings."""
        return self.cash + self.invested_value

    @property
    def invested_value(self) -> float:
        return sum(h["qty"] * h["last_price"] for h in self._holdings.values())

    @property
    def invested_at_cost(self) -> float:
        """Total capital deployed at average cost (cost basis of open positions)."""
        return sum(h["qty"] * h["avg_cost"] for h in self._holdings.values())

    @property
    def unrealized_pnl(self) -> float:
        return sum(
            (h["last_price"] - h["avg_cost"]) * h["qty"]
            for h in self._holdings.values()
        )

    @property
    def realized_pnl(self) -> float:
        return sum(
            tx.get("realized_pnl", 0.0)
            for tx in self._transactions
            if tx["action"] == "SELL"
        )

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def total_return_pct(self) -> float:
        return (self.nav / self.initial_cash - 1) * 100

    # ── DataFrames ────────────────────────────────────────────────────

    @property
    def holdings_df(self) -> pd.DataFrame:
        """Open positions with unrealized P&L."""
        if not self._holdings:
            return pd.DataFrame(columns=[
                "qty", "avg_cost", "last_price",
                "cost_basis", "market_value",
                "unrealized_pnl", "unrealized_pnl_pct",
                "weight_pct", "open_date",
            ])

        nav_now = self.nav
        rows = []
        for ticker, h in self._holdings.items():
            cb   = h["qty"] * h["avg_cost"]
            mv   = h["qty"] * h["last_price"]
            upnl = mv - cb
            rows.append({
                "ticker":              ticker,
                "qty":                 h["qty"],
                "avg_cost":            round(h["avg_cost"], 2),
                "last_price":          round(h["last_price"], 2),
                "cost_basis":          round(cb, 2),
                "market_value":        round(mv, 2),
                "unrealized_pnl":      round(upnl, 2),
                "unrealized_pnl_pct":  round(upnl / cb * 100, 2) if cb else 0.0,
                "weight_pct":          round(mv / nav_now * 100, 2) if nav_now else 0.0,
                "open_date":           h["open_date"],
            })

        df = (
            pd.DataFrame(rows)
            .set_index("ticker")
            .sort_values("market_value", ascending=False)
        )
        return df

    @property
    def transactions_df(self) -> pd.DataFrame:
        """All executed trades (buys and sells) in chronological order."""
        if not self._transactions:
            return pd.DataFrame()
        return pd.DataFrame(self._transactions)

    @property
    def closed_positions_df(self) -> pd.DataFrame:
        """Completed round-trip positions with realized P&L."""
        if not self._closed_positions:
            return pd.DataFrame()
        df = pd.DataFrame(self._closed_positions)
        df = df.sort_values("close_date", ascending=False).reset_index(drop=True)
        return df

    @property
    def nav_history_df(self) -> pd.DataFrame:
        """Daily NAV snapshots with daily returns."""
        if not self._nav_history:
            return pd.DataFrame()
        df = pd.DataFrame(self._nav_history).set_index("date")
        df["daily_return_pct"] = df["nav"].pct_change() * 100
        return df

    # ================================================================
    # Summary
    # ================================================================

    def summary(self) -> str:
        sep = "─" * 52
        cp  = self.closed_positions_df
        tx  = self.transactions_df

        n_wins  = int((cp["realized_pnl"] > 0).sum()) if not cp.empty and "realized_pnl" in cp else 0
        n_loss  = int((cp["realized_pnl"] <= 0).sum()) if not cp.empty and "realized_pnl" in cp else 0
        avg_hold = int(cp["holding_days"].mean()) if not cp.empty and "holding_days" in cp else 0

        lines = [
            sep,
            f"  Portfolio Summary",
            sep,
            f"  Initial Cash      {self.currency} {self.initial_cash:>15,.0f}",
            f"  Current Cash      {self.currency} {self.cash:>15,.2f}",
            f"  Invested Value    {self.currency} {self.invested_value:>15,.2f}",
            f"  NAV               {self.currency} {self.nav:>15,.2f}",
            f"  Total Return      {'':>9}{self.total_return_pct:>+10.2f}%",
            sep,
            f"  Realized P&L      {self.currency} {self.realized_pnl:>+14,.2f}",
            f"  Unrealized P&L    {self.currency} {self.unrealized_pnl:>+14,.2f}",
            f"  Total P&L         {self.currency} {self.total_pnl:>+14,.2f}",
            sep,
            f"  Open Positions    {len(self._holdings):>15}",
            f"  Closed Positions  {len(self._closed_positions):>15}",
            f"  Total Trades      {len(self._transactions):>15}",
            f"  Wins / Losses     {n_wins:>5} / {n_loss:<5}",
            f"  Avg Hold (days)   {avg_hold:>15}",
            sep,
        ]
        return "\n".join(lines)

    # ================================================================
    # Persistence
    # ================================================================

    def to_csv(self, output_dir: str = "output/portfolio") -> Dict[str, Path]:
        """
        Save all four logs to CSV files.  Returns a dict of {label: Path}.

        Files written:
            holdings.csv          — current open positions
            transactions.csv      — all executed trades
            closed_positions.csv  — completed round-trips with P&L
            nav_history.csv       — daily NAV snapshots
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"portfolio_{ts}"

        paths: Dict[str, Path] = {}

        def _save(df: pd.DataFrame, label: str) -> Path:
            path = out / f"{stem}_{label}.csv"
            df.to_csv(path)
            logger.info("Saved %-20s → %s", label, path)
            return path

        paths["holdings"]         = _save(self.holdings_df,        "holdings")
        paths["transactions"]     = _save(self.transactions_df,     "transactions")
        paths["closed_positions"] = _save(self.closed_positions_df, "closed_positions")
        paths["nav_history"]      = _save(self.nav_history_df,      "nav_history")

        return paths

    # ================================================================
    # Internals
    # ================================================================

    def _nav_with(self, price_dict: Dict[str, float]) -> float:
        """NAV using given prices for holdings (fallback to last_price)."""
        invested = sum(
            h["qty"] * price_dict.get(t, h["last_price"])
            for t, h in self._holdings.items()
        )
        return self.cash + invested

    def __repr__(self) -> str:
        return (
            f"Portfolio("
            f"nav={self.nav:,.0f} {self.currency}, "
            f"cash={self.cash:,.0f}, "
            f"positions={len(self._holdings)}, "
            f"return={self.total_return_pct:+.2f}%)"
        )
