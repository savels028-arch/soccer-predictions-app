"""Command-line entry point for the AIBets research strategy zoo."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import scipy
import sklearn

from research.dataset import LATEST_COMPLETE_SEASON, load_canonical_matches
from research.engine import ResearchConfig, run_nested_strategy_zoo
from research.features import build_feature_frame
from research.registry import register_research_result


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DATA = ROOT / "data" / "research"
FEATURE_CONFIG: Dict[str, Any] = {
    "windows": (5, 10, 20),
    "team_prior_strength": 5.0,
    "league_prior_strength": 40.0,
    "league_window": 500,
    "elo_initial": 1500.0,
    "elo_home_advantage": 65.0,
    "elo_k_factor": 20.0,
    "include_unshrunk_history": False,
}
FEATURE_SOURCE_PATHS = (
    ROOT / "research" / "asian_handicap.py",
    ROOT / "research" / "dataset.py",
    ROOT / "research" / "features.py",
    ROOT / "src" / "api" / "csv_football_client.py",
    ROOT / "config" / "settings.py",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=_json_default)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(paths) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for path in sorted({Path(value).resolve() for value in paths}, key=str):
        if not path.is_file() or ROOT not in path.parents:
            raise FileNotFoundError(f"research source is missing or outside the repository: {path}")
        result[path.relative_to(ROOT).as_posix()] = _sha256_file(path)
    return result


def _canonical_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_state(source_paths) -> Dict[str, Any]:
    relative_paths = [Path(path).resolve().relative_to(ROOT).as_posix() for path in source_paths]
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
        repository_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.splitlines()
        source_status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *relative_paths,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return {
            "head": None,
            "repository_dirty": None,
            "research_sources_dirty": None,
            "research_source_status": [],
        }
    return {
        "head": head,
        "repository_dirty": bool(repository_status),
        "research_sources_dirty": bool(source_status),
        # Status codes and paths only. Never persist diff or file contents here.
        "research_source_status": source_status,
    }


def _research_code_provenance() -> Dict[str, Any]:
    paths = tuple(sorted((ROOT / "research").glob("*.py"))) + (
        ROOT / "src" / "api" / "csv_football_client.py",
        ROOT / "config" / "settings.py",
    )
    hashes = _source_hashes(paths)
    return {
        "fingerprint_sha256": _canonical_fingerprint(hashes),
        "source_sha256": hashes,
        "git": _git_state(paths),
    }


def _feature_cache_fingerprint(
    dataset_id: str,
    feature_source_hashes: Dict[str, str],
    feature_config: Dict[str, Any],
) -> str:
    return _canonical_fingerprint(
        {
            "dataset_id": dataset_id,
            "feature_source_sha256": feature_source_hashes,
            "feature_config": feature_config,
        }
    )


def _load_or_build_features(
    matches,
    dataset_manifest: Dict[str, Any],
    *,
    rebuild: bool,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    cache_dir = RESEARCH_DATA / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_source_hashes = _source_hashes(FEATURE_SOURCE_PATHS)
    fingerprint = _feature_cache_fingerprint(
        str(dataset_manifest["dataset_id"]),
        feature_source_hashes,
        FEATURE_CONFIG,
    )
    cache_path = cache_dir / f"features_{dataset_manifest['dataset_id']}_{fingerprint[:20]}.pkl"
    if cache_path.exists() and not rebuild:
        print(f"feature cache: {cache_path}", flush=True)
        frame = pd.read_pickle(cache_path)
        if frame.attrs.get("feature_cache_fingerprint") != fingerprint:
            raise ValueError("feature cache metadata does not match its deterministic key")
    else:
        print("building leakage-free point-in-time feature frame...", flush=True)
        frame = build_feature_frame(matches, **FEATURE_CONFIG)
        frame.attrs.update(
            {
                "feature_cache_fingerprint": fingerprint,
                "feature_config": dict(FEATURE_CONFIG),
                "feature_source_sha256": feature_source_hashes,
            }
        )
        frame.to_pickle(cache_path)
        print(f"feature cache written: {cache_path} ({len(frame):,} rows)", flush=True)
    try:
        recorded_cache_path = cache_path.relative_to(ROOT).as_posix()
    except ValueError:
        recorded_cache_path = str(cache_path)
    cache_metadata = {
        "path": recorded_cache_path,
        "sha256": _sha256_file(cache_path),
        "fingerprint_sha256": fingerprint,
        "config": dict(FEATURE_CONFIG),
        "source_sha256": feature_source_hashes,
    }
    return frame, cache_metadata


def _report_markdown(result: Dict[str, Any], dataset: Dict[str, Any]) -> str:
    lines = [
        "# AIBets strategy-zoo research report",
        "",
        f"Generated: {result['generated_at']}",
        f"Dataset: `{dataset['dataset_id']}` — {dataset['rows']:,} matches, {dataset['start_date']} to {dataset['end_date']}",
        "",
        "## Method",
        "",
        "Nested expanding walk-forward. Models train through S-2; the first half of S-1 calibrates probabilities; the second half selects a fixed policy; season S is the untouched outer test. A 1% odds-profit haircut is applied by default. CLV is measured only when opening and closing prices come from the same source family.",
        "",
        "## Outer-test results",
        "",
        "| Market | Track | Bets | Hit rate | Profit (u) | ROI | 95% ROI CI | Positive seasons | Mean CLV | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for market, tracks in result["summaries"].items():
        for track, summary in tracks.items():
            bootstrap = summary.get("bootstrap", {})
            stability = summary.get("stability", {})
            closing = summary.get("closing_line", {})
            gate = result["promotion_gates"][market][track]
            ci = (
                f"{float(bootstrap.get('ci_lower', 0.0)):.1%}…{float(bootstrap.get('ci_upper', 0.0)):.1%}"
                if bootstrap
                else "—"
            )
            mean_clv = closing.get("mean_clv")
            mean_clv_text = f"{float(mean_clv):+.2%}" if mean_clv is not None else "—"
            lines.append(
                f"| {market} | {track} | {int(summary.get('bets', 0)):,} | "
                f"{float(summary.get('hit_rate', 0.0)):.1%} | {float(summary.get('profit', 0.0)):+.1f} | "
                f"{float(summary.get('roi', 0.0)):+.2%} | {ci} | "
                f"{float(stability.get('positive_season_rate', 0.0)):.1%} | "
                f"{mean_clv_text} | "
                f"{'PASS' if gate['passed'] else 'NO'} |"
            )
    lines.extend(
        [
            "",
            "## Fixed-policy audit",
            "",
            "| Market | Candidate | Family | Side | Odds basis / band | Development bets | Development ROI | Seasons | Holdout bets | Holdout ROI |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for market in result["summaries"]:
        lock = result.get("locked_strategies", {}).get(market, {}).get("executable") or {}
        selected = lock.get("selected")
        candidate_label = "eligible"
        summary_track = "locked_executable"
        if selected is None:
            selected = lock.get("diagnostic_selected")
            candidate_label = "diagnostic only"
            summary_track = "locked_diagnostic_executable"
        if not selected:
            lines.append(f"| {market} | none | — | — | — | 0 | — | 0 | 0 | — |")
            continue
        spec = selected["spec"]
        holdout = result["summaries"][market][summary_track]
        lines.append(
            f"| {market} | {candidate_label} | {spec['family']} | {spec['side']} | "
            f"{spec['odds_basis']} / {float(spec['min_odds']):.2f}–{float(spec['max_odds']):.2f} | "
            f"{int(selected.get('bets', 0)):,} | {float(selected.get('roi', 0.0)):+.2%} | "
            f"{int(selected.get('seasons', 0))} | {int(holdout.get('bets', 0)):,} | "
            f"{float(holdout.get('roi', 0.0)):+.2%} |"
        )
    lines.extend(
        [
            "",
            "## Promotion decision",
            "",
            f"**{result['champion_candidate']['status']}**",
            "",
            "Proxy/average/max-price results are diagnostic upper bounds and can never be promoted. Even a passing historical candidate starts in prospective shadow mode; it does not become a live betting recommendation automatically.",
            "",
            "## Research sources",
            "",
            "- Football-Data historical results and odds: https://www.football-data.co.uk/data.php",
            "- Dixon & Coles dynamic Poisson football model: https://doi.org/10.1111/1467-9876.00065",
            "- Scikit-learn time-series validation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html",
            "- Scikit-learn probability calibration: https://scikit-learn.org/stable/modules/calibration.html",
        ]
    )
    return "\n".join(lines) + "\n"


def audit(args: argparse.Namespace) -> int:
    _, manifest = load_canonical_matches(leagues=args.leagues, start=args.start_season, end=args.end_season)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def backtest(args: argparse.Namespace) -> int:
    code_provenance = _research_code_provenance()
    matches, dataset_manifest = load_canonical_matches(
        leagues=args.leagues,
        start=args.start_season,
        end=args.end_season,
    )
    print(
        f"dataset {dataset_manifest['dataset_id']}: {dataset_manifest['rows']:,} matches "
        f"({dataset_manifest['start_date']}…{dataset_manifest['end_date']})",
        flush=True,
    )
    frame, feature_cache = _load_or_build_features(
        matches,
        dataset_manifest,
        rebuild=args.rebuild_features,
    )
    markets = tuple(value.strip() for value in args.markets.split(",") if value.strip())
    config = ResearchConfig(
        first_test_season=args.first_test_season,
        last_test_season=args.last_test_season,
        markets=markets,
        min_train_seasons=args.min_train_seasons,
        min_selection_bets=args.min_selection_bets,
        odds_haircut=args.odds_haircut,
        random_state=args.seed,
        include_boosting=not args.no_boosting,
        include_isotonic=args.isotonic,
        bootstrap_resamples=args.bootstrap_resamples,
        policy_lock_season=args.policy_lock_season,
    )
    print(f"research config: {asdict(config)}", flush=True)
    result = run_nested_strategy_zoo(frame, config, progress=lambda message: print(message, flush=True))
    ending_source_hashes = _research_code_provenance()["source_sha256"]
    if ending_source_hashes != code_provenance["source_sha256"]:
        raise RuntimeError(
            "research source changed during the run; refusing to publish non-reproducible artifacts"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{dataset_manifest['dataset_id']}_{'full' if config.include_boosting else 'baseline'}"
    run_dir = RESEARCH_DATA / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "run_id": run_id,
        "dataset": dataset_manifest,
        "features": dict(frame.attrs),
        "feature_cache": feature_cache,
        "config": asdict(config),
        "git_sha": code_provenance["git"]["head"],
        "code": code_provenance,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "fold_results.json", result["folds"])
    _write_json(run_dir / "summary.json", result["summaries"])
    _write_json(run_dir / "promotion_gates.json", result["promotion_gates"])
    _write_json(run_dir / "locked_strategies.json", result["locked_strategies"])
    _write_json(run_dir / "champion_candidate.json", result["champion_candidate"])
    pd.DataFrame(result["bets"]).to_csv(run_dir / "bets.csv.gz", index=False, compression="gzip")
    (run_dir / "report.md").write_text(_report_markdown(result, dataset_manifest), encoding="utf-8")
    _, registry_event = register_research_result(
        RESEARCH_DATA / "runs" / "strategy_registry.json",
        result,
        run_id=run_id,
        dataset_id=dataset_manifest["dataset_id"],
        evaluated_at=result["generated_at"],
        git_sha=manifest["git_sha"],
    )
    _write_json(run_dir / "registry_evaluation.json", registry_event)
    print(f"run saved: {run_dir}", flush=True)
    print(
        f"shadow registry: {registry_event['status']} "
        f"({', '.join(registry_event['registered_markets']) or 'no markets'})",
        flush=True,
    )
    print(_report_markdown(result, dataset_manifest), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AIBets leakage-free strategy research")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("audit", "backtest"):
        command = subparsers.add_parser(name)
        command.add_argument("--start-season", type=int, default=1993)
        command.add_argument("--end-season", type=int, default=LATEST_COMPLETE_SEASON)
        command.add_argument("--leagues", nargs="*", default=None)
    backtest_parser = subparsers.choices["backtest"]
    backtest_parser.add_argument("--first-test-season", type=int, default=2012)
    backtest_parser.add_argument(
        "--last-test-season", type=int, default=LATEST_COMPLETE_SEASON
    )
    backtest_parser.add_argument("--markets", default="1x2,ou25")
    backtest_parser.add_argument("--min-train-seasons", type=int, default=5)
    backtest_parser.add_argument("--min-selection-bets", type=int, default=40)
    backtest_parser.add_argument("--odds-haircut", type=float, default=0.01)
    backtest_parser.add_argument("--seed", type=int, default=20260714)
    backtest_parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    backtest_parser.add_argument(
        "--policy-lock-season",
        type=int,
        default=2023,
        help="First untouched season for the fixed outer-of-outer policy holdout",
    )
    backtest_parser.add_argument("--no-boosting", action="store_true")
    backtest_parser.add_argument("--isotonic", action="store_true")
    backtest_parser.add_argument("--rebuild-features", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return audit(args) if args.command == "audit" else backtest(args)


if __name__ == "__main__":
    raise SystemExit(main())
