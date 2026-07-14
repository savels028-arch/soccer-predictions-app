"""Run static, leakage-free Asian Handicap sanity benchmarks.

The benchmark blindly backs every available home or away quote.  It does not
fit or select a strategy, so it cannot be mistaken for a trained champion.
Use ``--include-proxies`` only to compare executable bookmaker results with
non-executable market-average/maximum reference prices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.asian_handicap import (
    EXECUTABLE_ASIAN_BASES,
    PROXY_ASIAN_BASES,
    fixed_blind_asian_benchmark,
)
from research.dataset import LATEST_COMPLETE_SEASON, MIN_SEASON, load_canonical_matches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=MIN_SEASON)
    parser.add_argument("--end-season", type=int, default=LATEST_COMPLETE_SEASON)
    parser.add_argument(
        "--leagues",
        nargs="*",
        help="Optional research league codes; defaults to every canonical league",
    )
    parser.add_argument(
        "--include-proxies",
        action="store_true",
        help="Also report non-executable market average and maximum prices",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON artifact path; JSON is always printed to stdout",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    matches, manifest = load_canonical_matches(
        leagues=args.leagues,
        start=args.start_season,
        end=args.end_season,
    )
    bases = EXECUTABLE_ASIAN_BASES + (PROXY_ASIAN_BASES if args.include_proxies else ())
    payload = {
        "benchmark": "fixed_blind_asian_handicap",
        "selection": "none",
        "point_in_time_odds": "Football-Data pre-closing source columns",
        "source_caveats": [
            "Historical quotes are not proof that the price was available at bet time.",
            "Football-Data warns that Pinnacle prices from 2025-07-23 onward may be stale.",
        ],
        "dataset_id": manifest["dataset_id"],
        "source_matches": manifest["rows"],
        "season_start": args.start_season,
        "season_end": args.end_season,
        "results": fixed_blind_asian_benchmark(matches, bases=bases),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
