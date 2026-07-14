#!/usr/bin/env python3
"""Build the reproducible, forecast-only AIBets national-team model bundle.

The upstream dataset is pinned to an immutable commit and SHA-256.  The script
stores only a compact, normalized 1990+ snapshot plus a small model artifact;
it never vendors the upstream repository.  Selection uses 2010-2017 and the
reported holdout is the untouched 2018-2025 period.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predictions.international_model import (  # noqa: E402
    EloParameters,
    InternationalMatch,
    MANIFEST_SCHEMA,
    MODEL_SCHEMA,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URL,
    VALIDATED_STATUS,
    normalize_team_name,
    walk_forward_predictions,
)


SOURCE_COMMIT = PINNED_SOURCE_COMMIT
SOURCE_URL = PINNED_SOURCE_URL
SOURCE_SHA256 = PINNED_SOURCE_SHA256
SOURCE_REPOSITORY = "https://github.com/martj42/international_results"
OPENFOOTBALL_MIRROR = "https://github.com/openfootball/internationals"
LICENSE_URL = (
    "https://github.com/martj42/international_results/blob/"
    f"{SOURCE_COMMIT}/LICENSE"
)

SNAPSHOT_NAME = "results_1990_plus.csv.gz"
ARTIFACT_NAME = "international_elo_v1.json"
REPORT_NAME = "validation_report.md"
MANIFEST_NAME = "manifest.json"
START_DATE = date(1990, 1, 1)
CALIBRATION_START = date(2010, 1, 1)
CALIBRATION_END = date(2017, 12, 31)
HOLDOUT_START = date(2018, 1, 1)
HOLDOUT_END = date(2025, 12, 31)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source(local_source: Path | None = None) -> bytes:
    if local_source is not None:
        payload = local_source.read_bytes()
    else:
        request = Request(SOURCE_URL, headers={"User-Agent": "AIBets-research/1.0"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
    actual = sha256_bytes(payload)
    if actual != SOURCE_SHA256:
        raise RuntimeError(
            f"source checksum mismatch: expected {SOURCE_SHA256}, got {actual}"
        )
    return payload


def parse_matches(payload: bytes, as_of: date) -> Tuple[List[InternationalMatch], Dict[str, str], int]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    required = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "neutral",
    }
    if not reader.fieldnames or not required.issubset(reader.fieldnames):
        raise RuntimeError("upstream results.csv schema changed")

    matches: List[InternationalMatch] = []
    display_names: Dict[str, str] = {}
    skipped_unscored = 0
    seen = set()
    for row in reader:
        match_date = date.fromisoformat(row["date"])
        if match_date < START_DATE or match_date > as_of:
            continue
        try:
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
        except (TypeError, ValueError):
            skipped_unscored += 1
            continue
        home = normalize_team_name(row["home_team"])
        away = normalize_team_name(row["away_team"])
        if not home or not away or home == away:
            continue
        display_names.setdefault(home, row["home_team"].strip())
        display_names.setdefault(away, row["away_team"].strip())
        tournament = row["tournament"].strip()
        neutral = row["neutral"].strip().upper() == "TRUE"
        key = (
            match_date,
            home,
            away,
            home_score,
            away_score,
            tournament,
            neutral,
        )
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            InternationalMatch(
                match_date=match_date,
                home_team=home,
                away_team=away,
                home_score=home_score,
                away_score=away_score,
                tournament=tournament,
                neutral=neutral,
            )
        )
    matches.sort(
        key=lambda m: (m.match_date, m.home_team, m.away_team, m.tournament)
    )
    if not matches:
        raise RuntimeError("no scored international matches in snapshot")
    return matches, display_names, skipped_unscored


def write_snapshot(path: Path, matches: Sequence[InternationalMatch]) -> None:
    text_buffer = io.StringIO(newline="")
    writer = csv.writer(text_buffer, lineterminator="\n")
    writer.writerow(
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        ]
    )
    for match in matches:
        writer.writerow(
            [
                match.match_date.isoformat(),
                match.home_team,
                match.away_team,
                match.home_score,
                match.away_score,
                match.tournament,
                "TRUE" if match.neutral else "FALSE",
            ]
        )
    payload = text_buffer.getvalue().encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(payload)


OUTCOMES = ("home", "draw", "away")


def _argmax(probabilities: Mapping[str, float]) -> str:
    return max(OUTCOMES, key=lambda outcome: probabilities[outcome])


def outcome_priors(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    counts = {outcome: 1.0 for outcome in OUTCOMES}
    for prediction in predictions:
        counts[prediction["outcome"]] += 1.0
    total = sum(counts.values())
    return {outcome: counts[outcome] / total for outcome in OUTCOMES}


def metrics(
    predictions: Sequence[Mapping[str, Any]],
    baseline_probabilities: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    if not predictions:
        return {
            "matches": 0,
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "top_label_ece": None,
            "outcome_counts": {outcome: 0 for outcome in OUTCOMES},
        }
    correct = 0
    brier = 0.0
    log_loss = 0.0
    bins = [{"count": 0, "confidence": 0.0, "correct": 0} for _ in range(10)]
    outcome_counts = {outcome: 0 for outcome in OUTCOMES}
    for prediction in predictions:
        probs = baseline_probabilities or prediction["probabilities"]
        actual = prediction["outcome"]
        outcome_counts[actual] += 1
        chosen = _argmax(probs)
        confidence = float(probs[chosen])
        is_correct = int(chosen == actual)
        correct += is_correct
        brier += sum(
            (float(probs[outcome]) - float(outcome == actual)) ** 2
            for outcome in OUTCOMES
        )
        log_loss -= math.log(max(float(probs[actual]), 1e-15))
        bin_index = min(9, int(confidence * 10))
        bins[bin_index]["count"] += 1
        bins[bin_index]["confidence"] += confidence
        bins[bin_index]["correct"] += is_correct

    total = len(predictions)
    ece = 0.0
    calibration_bins = []
    for index, item in enumerate(bins):
        if not item["count"]:
            continue
        avg_confidence = item["confidence"] / item["count"]
        avg_accuracy = item["correct"] / item["count"]
        ece += (item["count"] / total) * abs(avg_confidence - avg_accuracy)
        calibration_bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": item["count"],
                "avg_confidence": round(avg_confidence, 6),
                "accuracy": round(avg_accuracy, 6),
            }
        )
    return {
        "matches": total,
        "accuracy": round(correct / total, 6),
        "brier": round(brier / total, 6),
        "log_loss": round(log_loss / total, 6),
        "top_label_ece": round(ece, 6),
        "outcome_counts": outcome_counts,
        "calibration_bins": calibration_bins,
    }


def parameter_grid() -> Iterable[EloParameters]:
    for k_factor in (20.0, 30.0, 40.0):
        for home_advantage in (40.0, 70.0, 100.0):
            for temperature in (0.85, 1.0, 1.15):
                for draw_base in (0.24, 0.27, 0.30):
                    for draw_decay in (0.3, 0.6, 0.9):
                        yield EloParameters(
                            k_factor=k_factor,
                            home_advantage=home_advantage,
                            temperature=temperature,
                            draw_base=draw_base,
                            draw_decay=draw_decay,
                            min_team_matches=8,
                        )


def verify_point_in_time_batching() -> bool:
    """Executable invariant: same-date input order cannot alter pre-match state."""
    params = EloParameters()
    fixtures = [
        InternationalMatch(date(2000, 1, 1), "Alpha", "Beta", 2, 0, "Friendly", True),
        InternationalMatch(date(2000, 1, 1), "Alpha", "Gamma", 0, 1, "Friendly", True),
    ]
    forward, state_forward = walk_forward_predictions(fixtures, params)
    reverse, state_reverse = walk_forward_predictions(list(reversed(fixtures)), params)
    return (
        forward == reverse
        and state_forward.ratings == state_reverse.ratings
        and all(row["home_rating"] == 1500.0 for row in forward)
    )


def tune_parameters(matches: Sequence[InternationalMatch]) -> Tuple[EloParameters, Dict[str, Any], int]:
    selection_rows = [m for m in matches if m.match_date <= CALIBRATION_END]
    best: Tuple[Tuple[float, float, float, Tuple[float, ...]], EloParameters, Dict[str, Any]] | None = None
    evaluated = 0
    for params in parameter_grid():
        predictions, _ = walk_forward_predictions(
            selection_rows,
            params,
            score_from=CALIBRATION_START,
            score_through=CALIBRATION_END,
        )
        report = metrics(predictions)
        tie_break = (
            params.k_factor,
            params.home_advantage,
            params.temperature,
            params.draw_base,
            params.draw_decay,
        )
        score = (
            float(report["log_loss"]),
            float(report["brier"]),
            -float(report["accuracy"]),
            tie_break,
        )
        if best is None or score < best[0]:
            best = (score, params, report)
        evaluated += 1
    if best is None:
        raise RuntimeError("parameter grid produced no model")
    return best[1], best[2], evaluated


def validation_bundle(
    matches: Sequence[InternationalMatch],
    params: EloParameters,
    selection_metrics: Mapping[str, Any],
    grid_size: int,
) -> Dict[str, Any]:
    through_holdout = [m for m in matches if m.match_date <= HOLDOUT_END]
    calibration_predictions, _ = walk_forward_predictions(
        through_holdout,
        params,
        score_from=CALIBRATION_START,
        score_through=CALIBRATION_END,
    )
    priors = outcome_priors(calibration_predictions)
    holdout_predictions, _ = walk_forward_predictions(
        through_holdout,
        params,
        score_from=HOLDOUT_START,
        score_through=HOLDOUT_END,
    )
    holdout_metrics = metrics(holdout_predictions)
    baseline_metrics = metrics(holdout_predictions, priors)
    world_cup_predictions = [
        prediction
        for prediction in holdout_predictions
        if prediction["tournament"] == "FIFA World Cup"
    ]
    world_cup_calibration_predictions = [
        prediction
        for prediction in calibration_predictions
        if prediction["tournament"] == "FIFA World Cup"
    ]
    world_cup_priors = outcome_priors(world_cup_calibration_predictions)
    world_cup_metrics = metrics(world_cup_predictions)
    world_cup_baseline = metrics(world_cup_predictions, world_cup_priors)

    gates = {
        "enough_holdout_matches": holdout_metrics["matches"] >= 6_000,
        "enough_world_cup_matches": world_cup_metrics["matches"] >= 100,
        "beats_prior_accuracy": (
            holdout_metrics["accuracy"] > baseline_metrics["accuracy"]
        ),
        "beats_prior_brier": holdout_metrics["brier"] < baseline_metrics["brier"],
        "beats_prior_log_loss": (
            holdout_metrics["log_loss"] < baseline_metrics["log_loss"]
        ),
        "calibration_within_limit": holdout_metrics["top_label_ece"] <= 0.08,
        "world_cup_beats_prior_accuracy": (
            world_cup_metrics["accuracy"] > world_cup_baseline["accuracy"]
        ),
        "world_cup_beats_prior_brier": (
            world_cup_metrics["brier"] < world_cup_baseline["brier"]
        ),
        "world_cup_beats_prior_log_loss": (
            world_cup_metrics["log_loss"] < world_cup_baseline["log_loss"]
        ),
        "world_cup_calibration_within_limit": (
            world_cup_metrics["top_label_ece"] <= 0.10
        ),
        "point_in_time_batching": verify_point_in_time_batching(),
    }
    return {
        "selection_period": {
            "start": CALIBRATION_START.isoformat(),
            "end": CALIBRATION_END.isoformat(),
            "grid_candidates": grid_size,
            "metrics": selection_metrics,
        },
        "holdout_period": {
            "start": HOLDOUT_START.isoformat(),
            "end": HOLDOUT_END.isoformat(),
        },
        "holdout": holdout_metrics,
        "fixed_prior_baseline": {
            "probabilities": {k: round(v, 8) for k, v in priors.items()},
            "metrics": baseline_metrics,
        },
        "world_cup_holdout": world_cup_metrics,
        "world_cup_fixed_prior_probabilities": {
            k: round(v, 8) for k, v in world_cup_priors.items()
        },
        "world_cup_fixed_prior_baseline": world_cup_baseline,
        "gates": gates,
        "limitations": [
            "No historical pre-match odds; ROI and betting edge are not validated.",
            "Source full-time scores may include extra time, so this is not a 90-minute betting settlement backtest.",
            "No lineups, injuries, player availability, travel or current FIFA ranking snapshots.",
        ],
    }


def pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value) * 100:.2f}%"


def report_markdown(artifact: Mapping[str, Any]) -> str:
    validation = artifact["validation"]
    holdout = validation["holdout"]
    baseline = validation["fixed_prior_baseline"]["metrics"]
    world_cup = validation["world_cup_holdout"]
    wc_baseline = validation["world_cup_fixed_prior_baseline"]
    gates = validation["gates"]
    status = artifact["status"]
    return f"""# International / World Cup model validation

Status: **{status}**. This is a calibrated outcome forecast, not a validated
betting strategy. There are no historical pre-match odds in the source, so the
report makes no ROI claim and the live pipeline must not create coupon/P&L bets.

## Frozen design

- Source: Mart Jürisoo international results (CC0), mirrored by OpenFootball
- Source commit: `{SOURCE_COMMIT}`
- Source SHA-256: `{SOURCE_SHA256}`
- Model selection/calibration: {CALIBRATION_START} through {CALIBRATION_END}
- Untouched holdout: {HOLDOUT_START} through {HOLDOUT_END}
- Point-in-time rule: predict every date batch before applying any result from it
- Selected parameters: `{json.dumps(artifact['parameters'], sort_keys=True)}`

## Honest holdout

| Scope | Matches | Accuracy | Brier (lower) | Log loss (lower) | Top-label ECE |
|---|---:|---:|---:|---:|---:|
| Elo model, all internationals | {holdout['matches']:,} | {pct(holdout['accuracy'])} | {holdout['brier']:.4f} | {holdout['log_loss']:.4f} | {pct(holdout['top_label_ece'])} |
| Fixed prior baseline | {baseline['matches']:,} | {pct(baseline['accuracy'])} | {baseline['brier']:.4f} | {baseline['log_loss']:.4f} | {pct(baseline['top_label_ece'])} |
| World Cup 2018 + 2022 only | {world_cup['matches']:,} | {pct(world_cup['accuracy'])} | {world_cup['brier']:.4f} | {world_cup['log_loss']:.4f} | {pct(world_cup['top_label_ece'])} |
| World Cup fixed prior baseline | {wc_baseline['matches']:,} | {pct(wc_baseline['accuracy'])} | {wc_baseline['brier']:.4f} | {wc_baseline['log_loss']:.4f} | {pct(wc_baseline['top_label_ece'])} |

## Fail-closed gates

""" + "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`" for name, passed in gates.items()
    ) + """

## Limitations

""" + "\n".join(f"- {item}" for item in validation["limitations"]) + "\n"


def build_bundle(payload: bytes, output_dir: Path, as_of: date) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matches, display_names, skipped_unscored = parse_matches(payload, as_of)
    snapshot_path = output_dir / SNAPSHOT_NAME
    write_snapshot(snapshot_path, matches)
    snapshot_sha = sha256_file(snapshot_path)

    params, selection_metrics, grid_size = tune_parameters(matches)
    validation = validation_bundle(matches, params, selection_metrics, grid_size)
    _, final_state = walk_forward_predictions(matches, params)
    all_gates_pass = all(validation["gates"].values())
    status = VALIDATED_STATUS if all_gates_pass else "REJECTED"
    training_cutoff = max(match.match_date for match in matches)
    artifact: Dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "model_version": "international_elo_forecast_only_v1",
        "status": status,
        "decision_scope": "forecast_only_no_historical_odds",
        "source_sha256": SOURCE_SHA256,
        "normalized_snapshot_sha256": snapshot_sha,
        "dataset": {
            "start": min(match.match_date for match in matches).isoformat(),
            "end": training_cutoff.isoformat(),
            "matches": len(matches),
            "teams": len(final_state.ratings),
            "skipped_unscored_rows": skipped_unscored,
        },
        "training_cutoff": training_cutoff.isoformat(),
        "parameters": params.as_dict(),
        "ratings": {
            team: round(rating, 8)
            for team, rating in sorted(final_state.ratings.items())
        },
        "match_counts": dict(sorted(final_state.match_counts.items())),
        "display_names": dict(sorted(display_names.items())),
        "world_cup_hosts": ["Canada", "Mexico", "United States"],
        "validation": validation,
    }
    artifact_path = output_dir / ARTIFACT_NAME
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifact_sha = sha256_file(artifact_path)
    report_path = output_dir / REPORT_NAME
    report_path.write_text(report_markdown(artifact), encoding="utf-8")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "name": "Mart Jürisoo international_results",
            "repository": SOURCE_REPOSITORY,
            "openfootball_mirror": OPENFOOTBALL_MIRROR,
            "commit": SOURCE_COMMIT,
            "url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "license": "CC0-1.0",
            "license_url": LICENSE_URL,
        },
        "snapshot": {
            "path": SNAPSHOT_NAME,
            "sha256": snapshot_sha,
            "rows": len(matches),
            "start": matches[0].match_date.isoformat(),
            "end": training_cutoff.isoformat(),
        },
        "artifact": {
            "path": ARTIFACT_NAME,
            "sha256": artifact_sha,
            "status": status,
        },
        "report": {"path": REPORT_NAME},
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {"artifact": artifact, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        help="Use a local copy of the pinned results.csv (still checksum-verified)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "international",
    )
    parser.add_argument(
        "--as-of",
        type=date.fromisoformat,
        default=date(2026, 7, 14),
        help="Ignore results after this date (default: reproducible 2026-07-14 snapshot)",
    )
    args = parser.parse_args()
    bundle = build_bundle(fetch_source(args.source), args.output_dir, args.as_of)
    artifact = bundle["artifact"]
    holdout = artifact["validation"]["holdout"]
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "dataset": artifact["dataset"],
                "parameters": artifact["parameters"],
                "holdout": holdout,
                "world_cup_holdout": artifact["validation"]["world_cup_holdout"],
                "gates": artifact["validation"]["gates"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == VALIDATED_STATUS else 2


if __name__ == "__main__":
    raise SystemExit(main())
