#!/usr/bin/env python3
"""Generate the compact public historical strategy-zoo artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from research.dataset import (
    CSV_LEAGUES,
    LATEST_COMPLETE_SEASON,
    MAX_SEASON,
    MIN_SEASON,
    assert_public_canonical_coverage,
    load_canonical_matches,
)
from research.pattern_zoo import (
    DEFAULT_PUBLIC_PATH,
    MAX_PUBLIC_BYTES,
    ROI_BOOTSTRAP_RESAMPLES,
    StrategyZooValidationError,
    build_strategy_zoo,
    load_strategy_zoo,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-season", type=int, default=MIN_SEASON)
    parser.add_argument(
        "--source-end-season",
        type=int,
        default=MAX_SEASON,
        help="Latest locally available source season; incomplete years require an explicit quarantine cutoff.",
    )
    parser.add_argument("--complete-through-season", type=int, default=LATEST_COMPLETE_SEASON)
    parser.add_argument("--display-through-season", type=int, default=MAX_SEASON)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--generated-at")
    parser.add_argument("--output", type=Path, default=DEFAULT_PUBLIC_PATH)
    return parser


def _atomic_write(path: Path, payload: dict) -> tuple[int, str]:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > MAX_PUBLIC_BYTES:
        raise ValueError(f"artifact is {len(encoded)} bytes; limit is {MAX_PUBLIC_BYTES}")
    artifact_bytes = encoded + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary_name = handle.name
            handle.write(artifact_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return len(artifact_bytes), hashlib.sha256(artifact_bytes).hexdigest()


def _atomic_write_checksum(path: Path, digest: str) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="ascii",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(f"{digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _guard_against_regression(path: Path, payload: dict) -> None:
    """Keep a stale or partial cache from replacing a larger public artifact."""

    if not path.exists():
        return
    try:
        previous_dataset = load_strategy_zoo(path)["dataset"]
    except StrategyZooValidationError:
        # A stricter current schema may legitimately reject an artifact made
        # by the immediately preceding generator version.  Preserve the two
        # monotonicity fields without weakening validation of the new output.
        try:
            previous_dataset = json.loads(path.read_text(encoding="utf-8"))["dataset"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise RuntimeError("cannot validate the previous Strategy Zoo dataset guard") from exc
        if (
            isinstance(previous_dataset.get("completeThroughSeason"), bool)
            or not isinstance(previous_dataset.get("completeThroughSeason"), int)
            or isinstance(previous_dataset.get("evaluatedMatches"), bool)
            or not isinstance(previous_dataset.get("evaluatedMatches"), int)
            or previous_dataset["evaluatedMatches"] < 0
        ):
            raise RuntimeError("previous Strategy Zoo dataset guard is malformed")
    next_dataset = payload["dataset"]
    if next_dataset["completeThroughSeason"] < previous_dataset["completeThroughSeason"]:
        raise RuntimeError("refusing to regress the Strategy Zoo complete-season cutoff")
    shrinking = (
        next_dataset["completeThroughSeason"] >= previous_dataset["completeThroughSeason"]
        and next_dataset["evaluatedMatches"] < previous_dataset["evaluatedMatches"]
    )
    if not shrinking:
        return

    # The source-file fingerprint can stay unchanged when a parser correction
    # starts rejecting rows whose explicit Div value contradicts the league in
    # the filename.  Permit only that exact, fully accounted correction.  A
    # stale/partial cache still fails because it either changes the dataset
    # fingerprint or cannot explain every missing row as a newly quarantined
    # league mismatch.
    previous_mismatches = previous_dataset.get("leagueMismatchRows", 0)
    next_mismatches = next_dataset.get("leagueMismatchRows", 0)
    correction_rows = next_mismatches - previous_mismatches
    evaluated_reduction = (
        previous_dataset["evaluatedMatches"] - next_dataset["evaluatedMatches"]
    )
    match_reduction = previous_dataset.get("matches", 0) - next_dataset.get("matches", 0)
    previous_source_id = previous_dataset.get(
        "sourceDatasetId",
        previous_dataset.get("datasetId"),
    )
    next_source_id = next_dataset.get(
        "sourceDatasetId",
        next_dataset.get("datasetId"),
    )
    accounted_parser_correction = (
        previous_source_id == next_source_id
        and correction_rows > 0
        and evaluated_reduction == correction_rows
        and match_reduction == correction_rows
    )
    if not accounted_parser_correction:
        raise RuntimeError("refusing to replace Strategy Zoo with a smaller evaluated dataset")


def _missing_canonical_files(
    file_hashes: dict[str, str],
    *,
    start_season: int,
    end_season: int,
) -> set[str]:
    expected = {
        f"{season % 100:02d}{(season + 1) % 100:02d}_{file_code}.csv"
        for season in range(start_season, end_season + 1)
        for file_code in CSV_LEAGUES
    }
    # Football-Data does not publish Belgian B1 files for these two earliest
    # canonical seasons; every other requested season/league file is required.
    known_unpublished = {"9394_B1.csv", "9495_B1.csv"}
    return expected.difference(known_unpublished).difference(file_hashes)


def build_from_canonical_cache(
    *,
    start_season: int = MIN_SEASON,
    source_end_season: int = MAX_SEASON,
    complete_through_season: int = LATEST_COMPLETE_SEASON,
    display_through_season: int = MAX_SEASON,
    bootstrap_resamples: int = ROI_BOOTSTRAP_RESAMPLES,
    generated_at: str | None = None,
) -> dict:
    """Build one complete artifact from the fixed canonical cache."""

    if not start_season <= complete_through_season <= source_end_season:
        raise ValueError("expected start-season <= complete-through-season <= source-end-season")
    if display_through_season != source_end_season:
        raise ValueError("display-through-season must equal source-end-season")
    matches, manifest = load_canonical_matches(start=start_season, end=source_end_season)
    if start_season == MIN_SEASON and source_end_season == MAX_SEASON:
        assert_public_canonical_coverage(
            matches,
            manifest,
            start_season=start_season,
            end_season=source_end_season,
        )
    if manifest.get("end_season") != source_end_season:
        raise RuntimeError("canonical cache does not contain the requested source-end season")
    observed_seasons = {int(match["season"]) for match in matches}
    expected_seasons = set(range(start_season, source_end_season + 1))
    if observed_seasons != expected_seasons:
        missing = ", ".join(str(season) for season in sorted(expected_seasons - observed_seasons))
        raise RuntimeError(f"canonical cache has missing seasons: {missing or 'unexpected coverage'}")
    missing_files = _missing_canonical_files(
        manifest.get("file_hashes", {}),
        start_season=start_season,
        end_season=source_end_season,
    )
    if missing_files:
        missing = ", ".join(sorted(missing_files))
        raise RuntimeError(f"canonical cache is missing requested files: {missing}")
    return build_strategy_zoo(
        matches,
        manifest,
        generated_at=generated_at,
        bootstrap_resamples=bootstrap_resamples,
        complete_through_season=complete_through_season,
        display_through_season=display_through_season,
    )


def verify_artifact_against_canonical(
    path: Path | str = DEFAULT_PUBLIC_PATH,
) -> dict:
    """Reject any public artifact not reproducible from canonical match rows.

    Structural validation and a checksum alone cannot detect coherent but
    invented P&L.  The publication boundary therefore rebuilds every metric
    from the immutable local CSV cache and requires exact equality.
    """

    artifact = load_strategy_zoo(path, require_checksum=True)
    dataset = artifact["dataset"]
    public_seasons = list(range(MIN_SEASON, MAX_SEASON + 1))
    if (
        artifact["seasons"] != public_seasons
        or dataset["completeThroughSeason"] != LATEST_COMPLETE_SEASON
        or dataset["quarantinedSeasons"]
    ):
        raise StrategyZooValidationError(
            "strategy zoo artifact does not cover the complete public canonical range"
        )
    expected = build_from_canonical_cache(
        start_season=MIN_SEASON,
        source_end_season=MAX_SEASON,
        complete_through_season=LATEST_COMPLETE_SEASON,
        display_through_season=MAX_SEASON,
        bootstrap_resamples=ROI_BOOTSTRAP_RESAMPLES,
        generated_at=artifact["generatedAt"],
    )
    if expected != artifact:
        raise StrategyZooValidationError(
            "strategy zoo artifact is not reproducible from the canonical cache"
        )
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output.resolve() == DEFAULT_PUBLIC_PATH.resolve() and (
        args.start_season != MIN_SEASON
        or args.source_end_season != MAX_SEASON
        or args.complete_through_season != LATEST_COMPLETE_SEASON
        or args.display_through_season != MAX_SEASON
        or args.bootstrap_resamples != ROI_BOOTSTRAP_RESAMPLES
    ):
        raise RuntimeError(
            "refusing to publish Strategy Zoo with non-canonical coverage or settings"
        )
    payload = build_from_canonical_cache(
        start_season=args.start_season,
        source_end_season=args.source_end_season,
        complete_through_season=args.complete_through_season,
        display_through_season=args.display_through_season,
        generated_at=args.generated_at,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    _guard_against_regression(args.output, payload)
    size, digest = _atomic_write(args.output, payload)
    _atomic_write_checksum(args.output.with_suffix(".sha256"), digest)
    statuses: dict[str, int] = {}
    for strategy in payload["strategies"]:
        status = strategy["status"]
        statuses[status] = statuses.get(status, 0) + 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": size,
                "sha256": digest,
                "matches": payload["dataset"]["matches"],
                "evaluated_matches": payload["dataset"]["evaluatedMatches"],
                "strategies": len(payload["strategies"]),
                "rivalry_patterns": len(payload["rivalryPatterns"]),
                "statuses": statuses,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
