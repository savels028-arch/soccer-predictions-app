from pathlib import Path

from research import run_zoo
from research.dataset import LATEST_COMPLETE_SEASON


def _match():
    return {
        "api_id": 1,
        "match_date": "2024-08-01T19:00:00+00:00",
        "league_code": "PL",
        "league_name": "Premier League",
        "season": 2024,
        "home_team_name": "Alpha",
        "away_team_name": "Beta",
        "home_score": 2,
        "away_score": 1,
        "home_odds": 1.8,
        "draw_odds": 3.5,
        "away_odds": 4.5,
        "extra_data": {},
    }


def test_cli_defaults_stop_at_latest_complete_local_season():
    parser = run_zoo.build_parser()

    audit_args = parser.parse_args(["audit"])
    backtest_args = parser.parse_args(["backtest"])

    assert audit_args.end_season == LATEST_COMPLETE_SEASON == 2025
    assert backtest_args.end_season == LATEST_COMPLETE_SEASON
    assert backtest_args.last_test_season == LATEST_COMPLETE_SEASON


def test_feature_cache_key_changes_with_code_or_explicit_config():
    sources = {"research/features.py": "abc"}
    config = {"windows": (5, 10, 20), "elo_k_factor": 20.0}

    original = run_zoo._feature_cache_fingerprint("dataset", sources, config)
    changed_code = run_zoo._feature_cache_fingerprint(
        "dataset", {"research/features.py": "def"}, config
    )
    changed_config = run_zoo._feature_cache_fingerprint(
        "dataset", sources, {**config, "elo_k_factor": 21.0}
    )

    assert original != changed_code
    assert original != changed_config


def test_feature_cache_records_fingerprint_config_sources_and_file_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(run_zoo, "RESEARCH_DATA", tmp_path / "research-data")
    manifest = {"dataset_id": "dataset-1"}

    frame, metadata = run_zoo._load_or_build_features(
        [_match()],
        manifest,
        rebuild=False,
    )

    recorded_path = Path(metadata["path"])
    cache_path = recorded_path if recorded_path.is_absolute() else run_zoo.ROOT / recorded_path
    assert len(frame) == 1
    assert cache_path.is_file()
    assert metadata["sha256"] == run_zoo._sha256_file(cache_path)
    assert frame.attrs["feature_cache_fingerprint"] == metadata["fingerprint_sha256"]
    assert frame.attrs["feature_config"] == run_zoo.FEATURE_CONFIG
    assert metadata["source_sha256"]


def test_code_provenance_hashes_actual_untracked_or_dirty_sources():
    provenance = run_zoo._research_code_provenance()

    assert len(provenance["fingerprint_sha256"]) == 64
    assert provenance["source_sha256"]["research/run_zoo.py"] == run_zoo._sha256_file(
        Path(run_zoo.__file__)
    )
    assert "repository_dirty" in provenance["git"]
    assert "research_source_status" in provenance["git"]
