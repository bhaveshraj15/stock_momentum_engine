"""
pipeline.py
-----------
Main entry point for the momentum engine.

Wires together the full v1 pipeline:
    loader  →  fetcher  →  filters  →  scorer  →  exporter

Usage (from the repo root):

    # Run with defaults (nifty50, returns mode, all 4 gates)
    python pipeline.py

    # Specify universe and output format
    python pipeline.py --universe config/universes/dax40.yaml --output xlsx

    # Use sharpe mode, custom lookbacks
    python pipeline.py --universe config/universes/etf_india.yaml \
                       --mode sharpe \
                       --lookbacks 3 6 12 \
                       --output xlsx csv

    # Force refresh cache
    python pipeline.py --universe config/universes/nifty50.yaml --refresh

Example output:
    INFO  Loaded universe 'Nifty 50' — 50 tickers
    INFO  Fetching price data...
    INFO  Fetch complete — shape: (504, 5, 50)
    INFO  Running scorer...
    INFO  Gates complete — 23/50 tickers passed
    INFO  Scoring complete — top 5:
               final_score  rank
    RELIANCE       0.9821   1.0
    TCS            0.8743   2.0
    ...
    INFO  Excel saved → output/runs/nifty50_20240415_093012.xlsx
"""

import argparse
import logging
import sys
from pathlib import Path

# ── logging setup ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline",
        description="Momentum Engine v1 — stock/ETF momentum screener",
    )
    p.add_argument(
        "--universe",
        type=str,
        default="config/universes/nifty50.yaml",
        help="Path to universe YAML config (default: nifty50)",
    )
    p.add_argument(
        "--output",
        nargs="+",
        choices=["xlsx", "csv", "numpy"],
        default=["xlsx"],
        help="Output format(s): xlsx csv numpy (default: xlsx)",
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
        default=[1, 3, 6, 12],
        help="Lookback periods in months (default: 1 3 6 12)",
    )
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for each lookback (default: equal weights)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="output/runs",
        help="Directory for output files (default: output/runs)",
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Directory for price data cache (default: data/cache)",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=730,
        help="Days of price history to fetch (default: 730 ≈ 2 years)",
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download, ignore cache",
    )
    p.add_argument(
        "--passing-only",
        action="store_true",
        help="Export only tickers that passed all gate filters",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Print top N tickers to console after scoring",
    )
    p.add_argument(
        "--no-gates",
        action="store_true",
        help="Skip gate filters, score all tickers",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def run(args: argparse.Namespace) -> None:
    """Execute the full pipeline with the given args."""

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 1. Load universe ─────────────────────────────────────────────
    from data.loader import UniverseLoader
    logger.info("Loading universe: %s", args.universe)
    loader  = UniverseLoader(args.universe)
    tickers = loader.get_tickers()
    logger.info(
        "Universe '%s' — %d tickers", loader.get_name(), len(tickers)
    )

    # ── 2. Fetch prices ──────────────────────────────────────────────
    from data.fetcher import Fetcher
    logger.info(
        "Fetching %d days of price data%s...",
        args.lookback_days,
        " (force refresh)" if args.refresh else "",
    )
    fetcher = Fetcher(
        cache_dir=args.cache_dir,
        progress=args.verbose,
    )
    prices = fetcher.fetch(
        tickers,
        lookback_days=args.lookback_days,
        force_refresh=args.refresh,
    )
    n_tickers = prices.columns.get_level_values(1).nunique()
    logger.info(
        "Prices ready — %d trading days × %d tickers",
        len(prices), n_tickers,
    )

    # ── 3. Build scorer ──────────────────────────────────────────────
    from filters.momentum_filter import MomentumFilter
    from scoring.scorer import Scorer, default_gates

    momentum = MomentumFilter(params={
        "lookbacks": args.lookbacks,
        "weights":   args.weights,
        "mode":      args.mode,
    })

    gates = [] if args.no_gates else default_gates()

    scorer = Scorer(
        gates=gates,
        scorers=[momentum],
    )

    # ── 4. Run scorer ────────────────────────────────────────────────
    logger.info("Running scorer...")
    result = scorer.run(prices)

    # ── 5. Print summary ─────────────────────────────────────────────
    passing = result[result["rank"].notna()]
    logger.info(
        "%d/%d tickers passed all gates", len(passing), len(result)
    )

    top_n = args.top_n or min(10, len(passing))
    if len(passing) > 0:
        top = (
            passing[["final_score", "rank"]]
            .sort_values("rank")
            .head(top_n)
        )
        print(f"\n{'─'*40}")
        print(f"  Top {top_n} — {loader.get_name()}  ({args.mode} mode)")
        print(f"{'─'*40}")
        print(top.to_string())
        print(f"{'─'*40}\n")
    else:
        logger.warning("No tickers passed the gate filters.")

    # ── 6. Export ────────────────────────────────────────────────────
    from output.exporter import Exporter
    exporter = Exporter(output_dir=args.output_dir)
    universe_name = loader.get_name().lower().replace(" ", "_")

    for fmt in args.output:
        if fmt == "xlsx":
            path = exporter.to_excel(result, universe_name=universe_name)
            logger.info("Saved → %s", path)

        elif fmt == "csv":
            path = exporter.to_csv(
                result,
                universe_name=universe_name,
                passing_only=args.passing_only,
            )
            logger.info("Saved → %s", path)

        elif fmt == "numpy":
            path = exporter.to_numpy(
                prices,
                field="Close",
                universe_name=universe_name,
            )
            logger.info("Saved → %s", path)

    logger.info("Pipeline complete.")


def main() -> None:
    parser = build_arg_parser()
    args   = parser.parse_args()

    universe_path = Path(args.universe)
    if not universe_path.exists():
        logger.error("Universe config not found: %s", args.universe)
        sys.exit(1)

    try:
        run(args)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()