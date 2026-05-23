"""
backtest_run.py
---------------
CLI for the walk-forward momentum backtester.

Usage examples
--------------
    # Basic — Nifty 50, monthly rebalance, top 10, 2020-2024
    python backtest_run.py --start 2020-01-01 --end 2024-12-31

    # DAX40, quarterly rebalance, top 5, sharpe mode
    python backtest_run.py --universe config/universes/dax40.yaml \\
                           --start 2018-01-01 --end 2024-12-31 \\
                           --rebalance quarterly --top-n 5 --mode sharpe

    # Compare against Nifty index benchmark, skip gates
    python backtest_run.py --start 2019-01-01 --end 2024-12-31 \\
                           --benchmark "^NSEI" --no-gates

    # Save output to CSV
    python backtest_run.py --start 2020-01-01 --end 2024-12-31 --output csv
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backtest_run",
        description="Momentum Engine — walk-forward backtester",
    )
    p.add_argument(
        "--universe",
        type=str,
        default="config/universes/nifty50.yaml",
        help="Universe YAML config (default: nifty50)",
    )
    p.add_argument(
        "--start",
        type=str,
        required=True,
        help="Backtest start date YYYY-MM-DD",
    )
    p.add_argument(
        "--end",
        type=str,
        required=True,
        help="Backtest end date YYYY-MM-DD",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of tickers to hold per period (default 10)",
    )
    p.add_argument(
        "--rebalance",
        choices=["weekly", "monthly", "quarterly"],
        default="monthly",
        help="Rebalance frequency (default: monthly)",
    )
    p.add_argument(
        "--mode",
        choices=["returns", "sharpe", "sortino"],
        default="returns",
        help="Momentum scoring mode (default: returns)",
    )
    p.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=[3, 6, 9, 12],
        help="Lookback periods in months (default: 3 6 9 12)",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Days of price history fed to scorer at each rebalance (default 730)",
    )
    p.add_argument(
        "--no-gates",
        action="store_true",
        help="Skip all gate filters",
    )
    p.add_argument(
        "--ema-fast",
        type=int,
        default=50,
        help="Fast EMA span for TrendFilter gate (default: 50)",
    )
    p.add_argument(
        "--ema-slow",
        type=int,
        default=100,
        help="Slow EMA span for TrendFilter gate (default: 100)",
    )
    p.add_argument(
        "--volume-confirm",
        action="store_true",
        help="Add VolumeFilter confirm gate (vol_21d > vol_63d > vol_252d)",
    )
    p.add_argument(
        "--no-volume-score",
        action="store_true",
        help="Disable VolumeFilter scorer (runs by default at weight 0.3)",
    )
    p.add_argument(
        "--volume-weight",
        type=float,
        default=0.3,
        help="Weight for VolumeFilter scorer (default: 0.3)",
    )
    p.add_argument(
        "--no-diversify",
        action="store_true",
        help="Skip diversification filter (runs by default)",
    )
    p.add_argument(
        "--div-threshold",
        type=float,
        default=0.85,
        help="Max correlation allowed between final picks (default: 0.85)",
    )
    p.add_argument(
        "--max-picks",
        type=int,
        default=None,
        help="Hard cap on final tickers after diversification",
    )
    p.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Benchmark ticker for comparison (e.g. '^NSEI', '^GDAXI')",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Price cache directory (default: data/cache)",
    )
    p.add_argument(
        "--initial-cash",
        type=float,
        default=1_000_000,
        help="Starting portfolio cash (default 1,000,000)",
    )
    p.add_argument(
        "--output",
        nargs="+",
        choices=["csv"],
        default=[],
        help="Output formats: csv",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/backtests",
        help="Output directory (default: output/backtests)",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    p.add_argument(
        "--verbose-rebalance",
        action="store_true",
        help="Print human-readable rebalance log after each rebalance",
    )
    return p


def run(args: argparse.Namespace) -> None:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from backtest.engine import BacktestEngine

    engine = BacktestEngine(
        universe      = args.universe,
        start         = args.start,
        end           = args.end,
        top_n         = args.top_n,
        rebalance     = args.rebalance,
        lookback_days = args.lookback_days,
        mode          = args.mode,
        lookbacks     = args.lookbacks,
        no_gates        = args.no_gates,
        ema_fast        = args.ema_fast,
        ema_slow        = args.ema_slow,
        volume_confirm  = args.volume_confirm,
        no_volume_score = args.no_volume_score,
        volume_weight   = args.volume_weight,
        no_diversify    = args.no_diversify,
        div_threshold   = args.div_threshold,
        max_picks       = args.max_picks,
        benchmark       = args.benchmark,
        initial_cash  = args.initial_cash,
        cache_dir     = args.cache_dir,
        verbose       = args.verbose_rebalance,
    )

    logger.info("Starting backtest: %s", engine)
    result = engine.run()

    print("\n" + result.summary())
    print("\n" + result.portfolio.summary())

    # Show open holdings and recent closed trades
    holdings = result.portfolio.holdings_df
    if not holdings.empty:
        print("\n  Open Holdings:")
        print(holdings[["qty","avg_cost","last_price","market_value","unrealized_pnl","unrealized_pnl_pct","weight_pct"]].to_string())

    closed = result.portfolio.closed_positions_df
    if not closed.empty:
        n_show = min(10, len(closed))
        print(f"\n  Last {n_show} Closed Positions:")
        print(closed.head(n_show)[["ticker","open_date","close_date","qty","avg_buy_price","sell_price","realized_pnl","realized_pnl_pct","holding_days"]].to_string(index=False))

    if "csv" in args.output:
        paths = result.to_csv(output_dir=args.output_dir)
        for label, path in paths.items():
            logger.info("Saved %s → %s", label, path)

    logger.info("Backtest complete.")


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    if not Path(args.universe).exists():
        logger.error("Universe config not found: %s", args.universe)
        sys.exit(1)

    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Backtest failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
