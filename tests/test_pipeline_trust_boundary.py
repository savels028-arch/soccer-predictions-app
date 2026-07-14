import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import ML_SETTINGS
from run_pipeline import (
    PredictionPipeline,
    PublicCacheSyncFailed,
    _build_finished_betting_result,
)
from src.firestore_writer import FirestoreWriter
from src.predictions.prediction_engine import current_season_start


class _StoredDocument:
    def __init__(self, stored):
        self.stored = stored

    def set(self, payload, **_kwargs):
        self.stored.update(payload)

    def get(self):
        return SimpleNamespace(exists=False)


class _StoredCollection:
    def __init__(self, stored):
        self.stored = stored

    def document(self, _doc_id):
        return _StoredDocument(self.stored)


class _StoredDb:
    def __init__(self, stored):
        self.stored = stored

    def collection(self, _name):
        return _StoredCollection(self.stored)


def _writer_with_store(stored):
    writer = object.__new__(FirestoreWriter)
    writer.db = _StoredDb(stored)
    return writer


def test_model_output_persists_only_verified_actionable_recommendation():
    rejected = {}
    writer = _writer_with_store(rejected)
    writer.save_model_output(
        "unverified",
        {"home": 0.65, "draw": 0.2, "away": 0.15},
        recommended_bet="HOME",
        decision_status="BET",
        odds_at_pick={"home": 1.8, "draw": 3.5, "away": 5.0},
    )

    assert rejected["decisionStatus"] == "ABSTAIN"
    assert rejected["decisionReason"] == "unverified_or_missing_pick_odds"
    assert rejected["eligibleForBetting"] is False
    assert "recommendedBet" not in rejected
    assert "evaluationMode" not in rejected

    accepted = {}
    writer = _writer_with_store(accepted)
    writer.save_model_output(
        "verified",
        {"home": 0.65, "draw": 0.2, "away": 0.15},
        recommended_bet="2",
        decision_status="BET",
        odds_at_pick={"home": 1.8, "draw": 3.5, "away": 5.0},
        odds_basis="verified_pre_match_odds",
        odds_source="danske_spil",
    )

    assert accepted["decisionStatus"] == "BET"
    assert accepted["recommendedBet"] == "AWAY"
    assert accepted["eligibleForBetting"] is True
    assert accepted["evaluationMode"] == "forward_only"


def test_finished_result_uses_recommendation_not_probability_winner():
    match = {
        "match_date": "2026-07-15T18:00:00Z",
        "home_team_name": "Alpha",
        "away_team_name": "Beta",
        "league_code": "PL",
    }
    output = {
        "generatedAt": "2026-07-15T12:00:00Z",
        "finalProbability": {"home": 0.7, "draw": 0.2, "away": 0.1},
        "recommendedBet": "AWAY",
        "decisionStatus": "BET",
        "eligibleForBetting": True,
        "evaluationMode": "forward_only",
        "oddsAtPick": {"home": 1.8, "draw": 3.4, "away": 5.2},
        "oddsBasis": "verified_pre_match_odds",
        "oddsSource": "danske_spil",
        "edge": {"away": 0.04},
    }

    result = _build_finished_betting_result(
        match,
        output,
        {"closingOdds": {"away": 4.8}},
        "AWAY",
        0,
        1,
        "ML Ensemble v1",
    )

    assert result["predictedOutcome"] == "AWAY"
    assert result["recommendedBet"] == "AWAY"
    assert result["isCorrect"] is True
    assert result["oddsAtPick"] == 5.2
    assert result["profit"] == 4.2
    assert result["eligibleForBetting"] is True


def test_unverified_or_late_result_keeps_accuracy_but_nulls_pnl():
    match = {
        "match_date": "2026-07-15T18:00:00Z",
        "home_team_name": "Alpha",
        "away_team_name": "Beta",
        "league_code": "PL",
    }
    output = {
        "generatedAt": "2026-07-15T19:00:00Z",
        "finalProbability": {"home": 0.6, "draw": 0.25, "away": 0.15},
        "recommendedBet": "HOME",
        "decisionStatus": "BET",
        "eligibleForBetting": True,
        "evaluationMode": "forward_only",
        "oddsAtPick": {"home": 1.9, "draw": 3.4, "away": 4.2},
        "oddsBasis": "verified_pre_match_odds",
        "oddsSource": "danske_spil",
    }

    result = _build_finished_betting_result(
        match, output, {}, "HOME", 2, 0, "ML Ensemble v1"
    )

    assert result["isCorrect"] is True
    assert result["eligibleForBetting"] is False
    assert result["evaluationMode"] == "accuracy_only"
    assert result["odds"] is None
    assert result["oddsAtPick"] is None
    assert result["profit"] is None


def test_prediction_result_persistence_recomputes_and_fail_closes_pnl():
    stored = {}
    writer = _writer_with_store(stored)
    writer.save_prediction_result({
        "matchDate": "2026-07-15",
        "homeTeam": "Alpha",
        "awayTeam": "Beta",
        "predictedOutcome": "HOME",
        "recommendedBet": "HOME",
        "decisionStatus": "BET",
        "evaluationMode": "forward_only",
        "eligibleForBetting": True,
        "oddsBasis": "verified_pre_match_odds",
        "oddsSource": "danske_spil",
        "oddsAtPick": 2.25,
        "odds": 99,
        "profit": 98,
        "isCorrect": False,
    })
    assert stored["odds"] == 2.25
    assert stored["profit"] == -1.0

    unverified = {}
    writer = _writer_with_store(unverified)
    writer.save_prediction_result({
        "matchDate": "2026-07-16",
        "homeTeam": "Gamma",
        "awayTeam": "Delta",
        "predictedOutcome": "HOME",
        "recommendedBet": "HOME",
        "decisionStatus": "BET",
        "evaluationMode": "forward_only",
        "eligibleForBetting": True,
        "oddsAtPick": 2.0,
        "odds": 2.0,
        "profit": 1.0,
        "isCorrect": True,
    })
    assert unverified["eligibleForBetting"] is False
    assert unverified["odds"] is None
    assert unverified["profit"] is None


def test_paper_cache_requires_every_forward_only_marker():
    valid = {
        "matchDate": "2026-07-15",
        "leagueCode": "PL",
        "predictedOutcome": "AWAY",
        "recommendedBet": "AWAY",
        "decisionStatus": "BET",
        "evaluationMode": "forward_only",
        "eligibleForBetting": True,
        "oddsBasis": "verified_pre_match_odds",
        "oddsSource": "danske_spil",
        "oddsAtPick": 2.5,
        "odds": 2.5,
        "profit": 1.5,
        "isCorrect": True,
    }
    legacy = {**valid, "matchDate": "2026-07-16"}
    legacy.pop("evaluationMode")

    writer = object.__new__(FirestoreWriter)
    writer.get_all_prediction_results = lambda limit=2000: [legacy, valid]
    caches = {}
    writer.write_cache = lambda name, payload: caches.update({name: payload})

    payload = writer.refresh_paper_trading_cache(stake=100, bankroll=1000)

    assert payload["totalBets"] == 1
    assert payload["totalWon"] == 1
    assert payload["totalProfit"] == 150
    assert payload["currentBankroll"] == 1150
    assert payload["excludedUnverifiedBets"] == 1
    assert caches["paper_trading"] == payload


def test_public_cache_uses_recommended_outcome_for_all_actionable_views():
    class Writer:
        def __init__(self):
            self.cache = {}

        def write_cache(self, name, data):
            self.cache[name] = data

        def refresh_coupon_history_cache(self):
            return {}

        def refresh_prediction_history_cache(self):
            return {}

        def refresh_paper_trading_cache(self, **_kwargs):
            return {"totalBets": 0, "totalProfit": 0, "roi": 0}

        def refresh_forecast_history_cache(self):
            return {}

        def refresh_source_weights_cache(self):
            return {}

    pipeline = object.__new__(PredictionPipeline)
    pipeline.fs = Writer()
    pipeline._matches = []
    pipeline._odds = [{
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_odds": 1.6,
        "draw_odds": 3.8,
        "away_odds": 5.5,
    }]
    pipeline._ml_preds = {"fixture": {
        "home_team": "Alpha",
        "away_team": "Beta",
        "league": "PL",
        "match_date": "2026-07-15T18:00:00Z",
        "ensemble": {"home": 0.7, "draw": 0.2, "away": 0.1},
        "edge": {"home": 0.01, "away": 0.07},
        "recommended": "AWAY",
        "decision_status": "BET",
        "models": {},
    }}

    pipeline.write_legacy_cache()

    assert pipeline.fs.cache["ai_predictions"][0]["predicted_outcome"] == "AWAY"
    market = pipeline.fs.cache["ml_predictions"]["odds_matches"][0]
    assert market["ai_prediction"] == "AWAY"
    assert market["ai_confidence"] == 10
    assert market["value_edge"] == 7.0


def test_daily_coupon_cannot_replace_recommendation_with_probability_winner(monkeypatch):
    class Writer:
        def __init__(self):
            self.coupon = None

        def save_daily_coupon(self, date_str, picks, total_odds):
            self.coupon = (date_str, picks, total_odds)

        def save_no_coupon(self, *_args):
            raise AssertionError("the recommended pick should qualify")

        def refresh_coupon_history_cache(self):
            return {}

        def save_pick_snapshot(self, *_args, **_kwargs):
            return None

    pipeline = object.__new__(PredictionPipeline)
    pipeline.fs = Writer()
    pipeline.db = SimpleNamespace(save_pick_snapshot=lambda *_args, **_kwargs: None)
    pipeline.ab_enabled = False
    pipeline._ml_preds_v2 = {}
    pipeline._stats = {"pick_snapshots": 0}
    pipeline._odds = [{
        "home_team": "Alpha",
        "away_team": "Beta",
        "home_odds": 1.6,
        "draw_odds": 3.8,
        "away_odds": 5.5,
    }]
    pipeline._ml_preds = {"fixture": {
        "home_team": "Alpha",
        "away_team": "Beta",
        "league": "PL",
        "match_date": "2026-07-15T18:00:00Z",
        "ensemble": {"home": 0.7, "draw": 0.2, "away": 0.1},
        "edge": {"home": 0.01, "away": 0.07},
        "recommended": "AWAY",
        "decision_status": "BET",
        "eligible_for_betting": True,
        "evaluation_mode": "forward_only",
        "odds_at_pick": {"home": 1.6, "draw": 3.8, "away": 5.5},
        "odds_basis": "verified_pre_match_odds",
        "odds_source": "danske_spil",
        "confidence": 0.1,
        "models": {},
    }}
    monkeypatch.setitem(ML_SETTINGS, "coupon", {
        "strategy": "walk_forward_value_coupon",
        "min_edge_pct": None,
        "min_confidence_pct": 0,
        "min_picks": 1,
        "max_picks": 1,
        "max_per_league": 1,
        "allowed_leagues": ["PL"],
        "skip_high_disagreement": False,
        "sort_by": "confidence",
    })

    pipeline.build_daily_coupon()

    assert pipeline.fs.coupon is not None
    assert pipeline.fs.coupon[1][0]["pick"] == "AWAY"
    assert pipeline.fs.coupon[1][0]["odds"] == 5.5
    assert pipeline.fs.coupon[1][0]["oddsBasis"] == "verified_pre_match_odds"


def test_daily_coupon_persistence_requires_and_exposes_verified_odds_basis():
    pick = {
        "pick": "AWAY",
        "odds": 5.5,
        "decisionStatus": "BET",
        "evaluationMode": "forward_only",
        "eligibleForBetting": True,
        "oddsBasis": "verified_pre_match_odds",
        "oddsSource": "danske_spil",
    }
    stored = {}
    writer = _writer_with_store(stored)

    writer.save_daily_coupon("2026-07-15", [pick], 5.5)

    assert stored["oddsBasis"] == "verified_pre_match_odds"
    assert stored["oddsSource"] == "danske_spil"
    assert stored["evaluationMode"] == "forward_only"
    assert stored["eligibleForBetting"] is True

    with pytest.raises(ValueError, match="verified_forward_only"):
        writer.save_daily_coupon(
            "2026-07-16",
            [{**pick, "oddsBasis": "unavailable_no_verified_odds"}],
            5.5,
        )


def test_pipeline_sync_failure_is_nonzero_after_writer_returns_failure():
    pipeline = object.__new__(PredictionPipeline)
    pipeline._stats = {}
    pipeline.fs = SimpleNamespace(
        sync_public_cache=lambda: SimpleNamespace(
            synced=False,
            reason="http_503",
            attempted_at="2026-07-14T12:00:00Z",
        )
    )

    with pytest.raises(PublicCacheSyncFailed, match="http_503"):
        pipeline.sync_public_cache()

    assert pipeline._stats["public_cache_synced"] is False


def test_training_season_rolls_over_in_august():
    assert current_season_start(date(2026, 7, 31)) == 2025
    assert current_season_start(date(2026, 8, 1)) == 2026


def test_local_wrapper_has_shared_lock_and_mandatory_sync_secret():
    wrapper = (
        Path(__file__).resolve().parents[1] / "scripts" / "run-local-pipeline.sh"
    ).read_text(encoding="utf-8")

    assert '.runtime/pipeline.lock' in wrapper
    assert "skipping overlapping job" in wrapper
    assert "Missing AIBets public-cache sync credential" in wrapper
    assert "refresh-football-data-cache.py --season" in wrapper
    assert "refusing to train on a stale or incomplete cache" in wrapper
