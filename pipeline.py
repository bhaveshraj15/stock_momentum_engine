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
        choices=["csv", "numpy"],
        default=["csv"],
        help="Output format(s): csv numpy (default: csv)",
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
        help="Skip all gate filters, score the entire universe",
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
        help="Weight for VolumeFilter scorer (default: 0.3, giving 70-30 split with momentum)",
    )
    p.add_argument(
        "--corr-filter",
        action="store_true",
        help="Apply correlation pre-filter before gates (recommended for large universes)",
    )
    p.add_argument(
        "--ccp",
        type=float,
        default=1.75e-4,
        help="Correlation cap parameter for --corr-filter (default: 1.75e-4)",
    )
    p.add_argument(
        "--no-diversify",
        action="store_true",
        help="Skip the diversification filter (runs by default)",
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
        help="Hard cap on final number of tickers after diversification",
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

    # ── 3. Correlation pre-filter (optional) ────────────────────────
    if args.corr_filter:
        from filters.correlation_filter import CorrelationFilter
        cf = CorrelationFilter(params={"ccp": args.ccp})
        logger.info(
            "Applying correlation pre-filter (ccp=%.2e)...", args.ccp
        )
        kept_tickers = cf.apply_to_universe(prices)
        if not kept_tickers:
            logger.error(
                "Correlation filter (ccp=%.2e) removed all %d tickers — "
                "every pair exceeds the correlation cap. "
                "Try a higher --ccp value.",
                args.ccp, n_tickers,
            )
            sys.exit(1)
        logger.info(
            "Correlation filter: %d → %d tickers",
            n_tickers, len(kept_tickers),
        )
        # Subset prices to kept tickers
        prices = prices.loc[
            :, prices.columns.get_level_values(1).isin(kept_tickers)
        ]

    # ── 4. Build scorer ──────────────────────────────────────────────
    # ── 4. Build scorer ──────────────────────────────────────────────
    from filters.momentum_filter import MomentumFilter
    from filters.volume_filter   import VolumeFilter
    from scoring.scorer          import Scorer, default_gates

    momentum = MomentumFilter(params={
        "lookbacks": args.lookbacks,
        "mode":      args.mode,
    })

    # Build gate list
    if args.no_gates:
        gates = []
    else:
        gates = default_gates()
        if args.volume_confirm:
            gates.append(
                VolumeFilter(params={"mode": "confirm"}, name="VolumeConfirm")
            )
            logger.info("VolumeFilter confirm gate added.")

    # Build scorer list
    scorers = [momentum]
    weights = [0.7]

    if not args.no_volume_score:
        vol_scorer = VolumeFilter(
            params={"mode": "score"},
            name="VolumeScore",
        )
        scorers.append(vol_scorer)
        weights.append(args.volume_weight)
        logger.info(
            "VolumeFilter scorer active (weight=%.1f)", args.volume_weight
        )

    scorer = Scorer(gates=gates, scorers=scorers, weights=weights)

    # ── 5. Run scorer ────────────────────────────────────────────────
    logger.info("Running scorer...")
    result = scorer.run(prices)

    # ── 6. Diversification filter (on by default, --no-diversify to skip) ──
    passing = result[result["rank"].notna()]
    logger.info("%d/%d tickers passed all gates", len(passing), len(result))

    if len(passing) == 0:
        logger.warning("No tickers passed the gate filters.")

    elif not args.no_diversify:
        from filters.diversification_filter import DiversificationFilter
        div = DiversificationFilter(params={
            "threshold":   args.div_threshold,
            "max_tickers": args.max_picks,
        })
        logger.info(
            "Applying diversification filter (threshold=%.2f)...",
            args.div_threshold,
        )
        report = div.get_report(result, prices)
        logger.info(
            "Diversification: %d → %d final picks",
            len(passing), (report["status"] == "accepted").sum(),
        )

        accepted = report[report["status"] == "accepted"][["rank", "final_score"]]
        removed  = report[report["status"] == "removed"][["rank", "final_score", "removed_by", "max_corr"]]

        top_n = args.top_n or len(accepted)
        print(f"\n{'─'*40}")
        print(f"  {loader.get_name()}  ({args.mode} mode)  — final picks")
        print(f"{'─'*40}")
        print(accepted.head(top_n).to_string())
        if not removed.empty:
            print(f"\n  Removed by diversification ({len(removed)}):")
            print(removed.to_string())
        print(f"{'─'*40}\n")

    else:
        # No diversification — print top-N from scorer directly
        top_n = args.top_n or min(10, len(passing))
        top   = passing[["final_score", "rank"]].sort_values("rank").head(top_n)
        print(f"\n{'─'*40}")
        print(f"  Top {top_n} — {loader.get_name()}  ({args.mode} mode)")
        print(f"{'─'*40}")
        print(top.to_string())
        print(f"{'─'*40}\n")

    # ── 8. Export ────────────────────────────────────────────────────
    from output.exporter import Exporter
    exporter = Exporter(output_dir=args.output_dir)
    universe_name = loader.get_name().lower().replace(" ", "_")

    for fmt in args.output:
        if fmt == "csv":
            path = exporter.to_csv(
                result,
                universe_name=universe_name,
                passing_only=args.passing_only,
                top_n=args.top_n,
                mode=args.mode,
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