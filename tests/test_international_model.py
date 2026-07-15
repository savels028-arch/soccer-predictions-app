from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import pytest

from src.predictions.international_model import (
    EloParameters,
    InternationalEloState,
    InternationalMatch,
    InternationalModelUnavailable,
    MODEL_SCHEMA,
    MANIFEST_SCHEMA,
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URL,
    VALIDATED_STATUS,
    ValidatedInternationalModel,
    load_validated_international_model,
    normalize_team_name,
    walk_forward_predictions,
)


def test_international_source_snapshot_is_pinned_through_2026_07_14():
    assert PINNED_SOURCE_COMMIT == "23449460b67a975bcf84d1042472c4f8da507f9c"
    assert (
        PINNED_SOURCE_SHA256
        == "126d983c1e0f6849f1d75222da93aa0e8559ce3fb32ab17da02c526c4b69288e"
    )
    assert PINNED_SOURCE_URL.endswith(
        "/23449460b67a975bcf84d1042472c4f8da507f9c/results.csv"
    )


def test_espn_and_current_world_cup_team_aliases_normalize_to_dataset_keys():
    assert normalize_team_name("USA") == "united states"
    assert normalize_team_name("U.S.A.") == "united states"
    assert normalize_team_name("Korea Republic") == "south korea"
    assert normalize_team_name("IR Iran") == "iran"
    assert normalize_team_name("Türkiye") == "turkey"
    assert normalize_team_name("Cabo Verde") == "cape verde"
    assert normalize_team_name("Côte d’Ivoire") == "ivory coast"
    assert normalize_team_name("Curaçao") == "curacao"
    assert normalize_team_name("Czechia") == "czech republic"
    assert normalize_team_name("Congo DR") == "dr congo"


def test_same_date_matches_are_predicted_before_any_same_date_update():
    params = EloParameters()
    fixtures = [
        InternationalMatch(date(2020, 1, 1), "Alpha", "Beta", 4, 0, "Friendly", True),
        InternationalMatch(date(2020, 1, 1), "Alpha", "Gamma", 0, 1, "Friendly", True),
    ]

    predictions, forward_state = walk_forward_predictions(fixtures, params)
    reversed_predictions, reversed_state = walk_forward_predictions(
        list(reversed(fixtures)), params
    )

    assert predictions == reversed_predictions
    assert forward_state.ratings == reversed_state.ratings
    assert [row["home_rating"] for row in predictions] == [1500.0, 1500.0]


def test_prediction_refuses_cutoff_unknown_teams_and_infers_2026_host_advantage():
    artifact = _artifact()
    model = ValidatedInternationalModel(artifact)

    with pytest.raises(InternationalModelUnavailable, match="fixture_not_after"):
        model.predict_fixture("France", "Spain", "2026-07-10")
    with pytest.raises(InternationalModelUnavailable, match="insufficient_home"):
        model.predict_fixture("Atlantis", "Spain", "2026-07-11")

    neutral = model.predict_fixture("France", "Spain", "2026-07-11")
    host = model.predict_fixture("United States", "Spain", "2026-07-11")
    assert neutral["neutral"] is True
    assert host["neutral"] is False
    assert sum(neutral["probabilities"].values()) == pytest.approx(1.0)
    assert host["decision_scope"] == "forecast_only_no_historical_odds"


def test_loader_fails_closed_on_snapshot_tampering(tmp_path: Path):
    manifest_path = _write_bundle(tmp_path)
    loaded = load_validated_international_model(manifest_path)
    assert loaded.training_cutoff == date(2026, 7, 10)

    (tmp_path / "snapshot.csv.gz").write_bytes(b"tampered")
    with pytest.raises(InternationalModelUnavailable, match="snapshot_checksum"):
        load_validated_international_model(manifest_path)


def test_loader_rejects_failed_or_missing_validation_gate(tmp_path: Path):
    artifact = _artifact()
    artifact["validation"]["gates"]["world_cup_beats_prior_brier"] = False
    manifest_path = _write_bundle(tmp_path, artifact=artifact)
    with pytest.raises(InternationalModelUnavailable, match="validation_gates_failed"):
        load_validated_international_model(manifest_path)


def test_checked_in_bundle_covers_all_2026_world_cup_teams_and_is_forecast_only():
    model = load_validated_international_model()
    participants = {
        "Algeria", "Argentina", "Australia", "Austria", "Belgium",
        "Bosnia and Herzegovina", "Brazil", "Canada", "Cape Verde",
        "Colombia", "Croatia", "Curaçao", "Czech Republic", "DR Congo",
        "Ecuador", "Egypt", "England", "France", "Germany", "Ghana",
        "Haiti", "Iran", "Iraq", "Ivory Coast", "Japan", "Jordan",
        "Mexico", "Morocco", "Netherlands", "New Zealand", "Norway",
        "Panama", "Paraguay", "Portugal", "Qatar", "Saudi Arabia",
        "Scotland", "Senegal", "South Africa", "South Korea", "Spain",
        "Sweden", "Switzerland", "Tunisia", "Turkey", "United States",
        "Uruguay", "Uzbekistan",
    }
    missing = [
        team
        for team in sorted(participants)
        if model.state.match_counts.get(normalize_team_name(team), 0)
        < model.params.min_team_matches
    ]
    assert missing == []

    forecast = model.predict_fixture("England", "Argentina", "2026-07-15")
    assert forecast["model_version"] == "international_elo_forecast_only_v1"
    assert forecast["decision_scope"] == "forecast_only_no_historical_odds"


def _artifact():
    gates = {
        "enough_holdout_matches": True,
        "enough_world_cup_matches": True,
        "beats_prior_accuracy": True,
        "beats_prior_brier": True,
        "beats_prior_log_loss": True,
        "calibration_within_limit": True,
        "world_cup_beats_prior_accuracy": True,
        "world_cup_beats_prior_brier": True,
        "world_cup_beats_prior_log_loss": True,
        "world_cup_calibration_within_limit": True,
        "point_in_time_batching": True,
    }
    return {
        "schema": MODEL_SCHEMA,
        "model_version": "international_elo_forecast_only_v1",
        "status": VALIDATED_STATUS,
        "source_sha256": PINNED_SOURCE_SHA256,
        "normalized_snapshot_sha256": "placeholder",
        "training_cutoff": "2026-07-10",
        "parameters": EloParameters().as_dict(),
        "ratings": {
            "france": 1650.0,
            "spain": 1640.0,
            "united states": 1530.0,
        },
        "match_counts": {"france": 100, "spain": 100, "united states": 100},
        "world_cup_hosts": ["Canada", "Mexico", "United States"],
        "validation": {
            "holdout": {"accuracy": 0.55},
            "world_cup_holdout": {"accuracy": 0.52},
            "gates": gates,
        },
    }


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bundle(tmp_path: Path, artifact=None) -> Path:
    snapshot = b"deterministic snapshot"
    snapshot_sha = _sha(snapshot)
    artifact = artifact or _artifact()
    artifact["normalized_snapshot_sha256"] = snapshot_sha
    artifact_payload = (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    (tmp_path / "snapshot.csv.gz").write_bytes(snapshot)
    (tmp_path / "artifact.json").write_bytes(artifact_payload)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source": {
            "commit": PINNED_SOURCE_COMMIT,
            "sha256": artifact["source_sha256"],
            "url": PINNED_SOURCE_URL,
        },
        "snapshot": {"path": "snapshot.csv.gz", "sha256": snapshot_sha},
        "artifact": {"path": "artifact.json", "sha256": _sha(artifact_payload)},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path
