from datetime import date, timedelta

import pytest

from run_pipeline import (
    INTERNATIONAL_FORECAST_ONLY_REASON,
    INTERNATIONAL_MODEL_ABSTAIN_REASON,
    ML_SETTINGS,
    PredictionPipeline,
    _model_scope_abstention_reason,
)
from src.firestore_writer import FirestoreWriter
from src.predictions.international_model import load_validated_international_model


class _BombModel:
    is_trained = True

    def predict_proba(self, _features):
        raise AssertionError("club model must not run for an international fixture")


class _FakeEngine:
    is_trained = True
    models = {"club_model": _BombModel()}


class _GuardedDb:
    def compute_team_stats_from_matches(self, *_args, **_kwargs):
        raise AssertionError("club team history must not be read for an international fixture")


class _ModelOutputWriter:
    def __init__(self):
        self.outputs = []

    def save_model_output(self, *args, **kwargs):
        self.outputs.append((args, kwargs))


def _world_cup_match():
    return {
        "api_id": 26071401,
        "home_team_name": "France",
        "away_team_name": "Spain",
        "league_code": "WC",
        "league_name": "FIFA World Cup",
        "match_date": (date.today() + timedelta(days=1)).isoformat() + "T20:00:00Z",
        "status": "SCHEDULED",
    }


def _guarded_pipeline():
    pipeline = object.__new__(PredictionPipeline)
    pipeline.engine = _FakeEngine()
    pipeline.engine_v2 = None
    pipeline.ab_enabled = False
    pipeline.db = _GuardedDb()
    pipeline.fs = _ModelOutputWriter()
    pipeline._matches = [_world_cup_match()]
    pipeline._odds = [{
        "home_team": "France",
        "away_team": "Spain",
        "home_odds": 2.5,
        "draw_odds": 3.1,
        "away_odds": 2.9,
    }]
    pipeline._ai_preds = []
    pipeline._ml_preds = {}
    pipeline._ml_preds_v2 = {}
    pipeline._stats = {"ml_predictions": 0}
    pipeline._stats_v2 = {"ml_predictions": 0}
    return pipeline


def test_model_scope_guard_only_blocks_unvalidated_international_leagues():
    assert _model_scope_abstention_reason("WC") == INTERNATIONAL_MODEL_ABSTAIN_REASON
    assert _model_scope_abstention_reason("wc") == INTERNATIONAL_MODEL_ABSTAIN_REASON
    assert _model_scope_abstention_reason("PL") is None


def test_world_cup_fixture_stays_visible_but_club_model_abstains():
    pipeline = _guarded_pipeline()

    predictions = pipeline.run_ml_predictions()

    assert len(predictions) == 1
    prediction = next(iter(predictions.values()))
    assert prediction["league"] == "WC"
    assert prediction["decision_status"] == "ABSTAIN"
    assert prediction["decision_reason"] == INTERNATIONAL_MODEL_ABSTAIN_REASON
    assert prediction["recommended"] is None
    assert prediction["confidence"] == 0.0
    assert prediction["edge"] == {}
    assert prediction["models"] == {}
    assert prediction["ensemble"] == pytest.approx({
        "home": 1 / 3,
        "draw": 1 / 3,
        "away": 1 / 3,
    })

    assert len(pipeline.fs.outputs) == 1
    _, kwargs = pipeline.fs.outputs[0]
    assert kwargs["decision_status"] == "ABSTAIN"
    assert kwargs["decision_reason"] == INTERNATIONAL_MODEL_ABSTAIN_REASON
    assert kwargs["confidence"] == 0.0
    assert kwargs["model_version"] == "international_shadow_abstain"


def test_validated_international_forecast_is_visible_but_still_non_bet_abstain():
    pipeline = _guarded_pipeline()
    pipeline.international_model = load_validated_international_model()

    predictions = pipeline.run_ml_predictions()

    prediction = next(iter(predictions.values()))
    assert prediction["decision_status"] == "ABSTAIN"
    assert prediction["decision_reason"] == INTERNATIONAL_FORECAST_ONLY_REASON
    assert prediction["forecast_status"] == "VALIDATED_FORECAST_ONLY"
    assert prediction["recommended"] is None
    assert prediction["edge"] == {}
    assert prediction["models"]["international_elo"] == prediction["ensemble"]
    assert sum(prediction["ensemble"].values()) == pytest.approx(1.0)
    assert prediction["ensemble"] != pytest.approx({
        "home": 1 / 3,
        "draw": 1 / 3,
        "away": 1 / 3,
    })

    _, kwargs = pipeline.fs.outputs[0]
    assert kwargs["decision_status"] == "ABSTAIN"
    assert kwargs["decision_reason"] == INTERNATIONAL_FORECAST_ONLY_REASON
    assert kwargs["model_version"] == "international_elo_forecast_only_v1"
    assert kwargs["confidence"] == 0.0
    assert kwargs.get("recommended_bet") is None
    assert kwargs["forecast_status"] == "VALIDATED_FORECAST_ONLY"
    assert kwargs["forecast_outcome"] == prediction["forecast_outcome"]
    assert kwargs["forecast_confidence"] == prediction["forecast_confidence"]


def test_started_international_fixture_never_runs_or_overwrites_forecast():
    class BombInternationalModel:
        def predict_fixture(self, *_args, **_kwargs):
            raise AssertionError("a forecast must never be generated after kickoff")

    pipeline = _guarded_pipeline()
    pipeline.international_model = BombInternationalModel()
    pipeline._matches[0]["status"] = "IN_PLAY"
    pipeline._matches[0]["match_date"] = (
        date.today() - timedelta(days=1)
    ).isoformat() + "T20:00:00Z"

    predictions = pipeline.run_ml_predictions()

    prediction = next(iter(predictions.values()))
    assert prediction["decision_status"] == "ABSTAIN"
    assert "forecast_status" not in prediction
    assert pipeline.fs.outputs == []


def test_legacy_cache_marks_abstention_and_excludes_it_from_betting_odds():
    pipeline = _guarded_pipeline()
    pipeline.run_ml_predictions()

    class CacheWriter:
        def __init__(self):
            self.cache = {}

        def write_cache(self, name, payload):
            self.cache[name] = payload

        def refresh_coupon_history_cache(self):
            return {}

        def refresh_prediction_history_cache(self):
            return {}

        def refresh_paper_trading_cache(self, **_kwargs):
            return {"totalBets": 0, "totalProfit": 0, "roi": 0}

    cache_writer = CacheWriter()
    pipeline.fs = cache_writer
    pipeline.write_legacy_cache()

    cached = cache_writer.cache["ai_predictions"]
    assert len(cached) == 1
    assert cached[0]["predicted_outcome"] == "ABSTAIN"
    assert cached[0]["confidence"] == 0
    assert cached[0]["abstain_reason"] == INTERNATIONAL_MODEL_ABSTAIN_REASON
    assert cached[0]["sources"] == []
    assert cache_writer.cache["ml_predictions"]["odds_matches"] == []


def test_abstention_never_becomes_coupon_candidate_if_league_is_enabled(monkeypatch):
    pipeline = _guarded_pipeline()
    pipeline.run_ml_predictions()

    class CouponWriter:
        def __init__(self):
            self.skipped = []

        def save_daily_coupon(self, *_args, **_kwargs):
            raise AssertionError("an abstained fixture must never enter a coupon")

        def save_no_coupon(self, date_str, reason, meta):
            self.skipped.append((date_str, reason, meta))

        def refresh_coupon_history_cache(self):
            return {}

    coupon_writer = CouponWriter()
    pipeline.fs = coupon_writer
    monkeypatch.setitem(ML_SETTINGS, "coupon", {
        "strategy": "historical_h2h_coupon",
        "min_h2h_matches": 1,
        "min_h2h_rate_pct": 0,
        "min_edge_pct": None,
        "min_confidence_pct": 0,
        "min_picks": 1,
        "max_picks": 1,
        "max_per_league": 1,
        "allowed_leagues": ["WC"],
    })

    pipeline.build_daily_coupon()

    assert len(coupon_writer.skipped) == 1
    assert coupon_writer.skipped[0][1] == "not_enough_quality_picks"
    assert coupon_writer.skipped[0][2]["candidateCount"] == 0


def test_firestore_model_output_persists_abstention_metadata():
    stored = {}

    class Document:
        def set(self, payload):
            stored.update(payload)

    class Collection:
        def document(self, _mid):
            return Document()

    class Db:
        def collection(self, _name):
            return Collection()

    writer = object.__new__(FirestoreWriter)
    writer.db = Db()
    writer.save_model_output(
        "wc-fixture",
        {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3},
        confidence=0.0,
        model_version="international_shadow_abstain",
        decision_status="ABSTAIN",
        decision_reason=INTERNATIONAL_MODEL_ABSTAIN_REASON,
    )

    assert stored["decisionStatus"] == "ABSTAIN"
    assert stored["decisionReason"] == INTERNATIONAL_MODEL_ABSTAIN_REASON
    assert "recommendedBet" not in stored
    assert "edge" not in stored


def test_firestore_result_history_rejects_and_filters_abstentions():
    class ResultDocument:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return self.payload

    class ResultQuery:
        def order_by(self, *_args, **_kwargs):
            return self

        def limit(self, _limit):
            return self

        def get(self):
            return [
                ResultDocument({"predictedOutcome": "HOME", "isCorrect": True}),
                ResultDocument({"predictedOutcome": "ABSTAIN", "isCorrect": False}),
                ResultDocument({"decisionStatus": "ABSTAIN", "predictedOutcome": "AWAY"}),
            ]

        def document(self, _doc_id):
            raise AssertionError("an abstention must be rejected before any Firestore write")

    class ResultDb:
        def collection(self, _name):
            return ResultQuery()

    writer = object.__new__(FirestoreWriter)
    writer.db = ResultDb()

    assert writer.save_prediction_result({"predictedOutcome": "ABSTAIN"}) is False
    assert writer.get_all_prediction_results() == [
        {"predictedOutcome": "HOME", "isCorrect": True}
    ]
