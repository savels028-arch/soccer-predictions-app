from run_pipeline import PredictionPipeline, _build_finished_forecast_result
from src.firestore_writer import (
    FirestoreWriter,
    NON_BETTING_FORECAST_SCOPE,
    VALIDATED_FORECAST_ONLY,
    build_forecast_history_payload,
)


def _stored_forecast(generated_at="2026-07-14T12:00:00Z"):
    return {
        "generatedAt": generated_at,
        "finalProbability": {"home": 0.56, "draw": 0.24, "away": 0.20},
        "forecastStatus": VALIDATED_FORECAST_ONLY,
        "forecastOutcome": "HOME",
        "forecastConfidence": 0.56,
        "decisionStatus": "ABSTAIN",
        "evaluationScope": NON_BETTING_FORECAST_SCOPE,
        "eligibleForBetting": False,
        "modelVersion": "international_elo_forecast_only_v1",
    }


def _finished_match():
    return {
        "home_team_name": "France",
        "away_team_name": "Spain",
        "league_code": "WC",
        "match_date": "2026-07-14T20:00:00Z",
        "status": "FINISHED",
        "home_score": 2,
        "away_score": 1,
    }


def test_finished_forecast_requires_persisted_pre_match_metadata():
    result = _build_finished_forecast_result(
        "2026-07-14_france_spain",
        _finished_match(),
        _stored_forecast(),
        "HOME",
        2,
        1,
    )

    assert result is not None
    assert result["isCorrect"] is True
    assert result["decisionStatus"] == "ABSTAIN"
    assert result["evaluationScope"] == NON_BETTING_FORECAST_SCOPE
    assert result["eligibleForBetting"] is False
    assert result["forecastConfidence"] == 56.0
    assert result["brierScore"] > 0
    assert result["logLoss"] > 0
    assert "odds" not in result
    assert "profit" not in result

    assert _build_finished_forecast_result(
        "late", _finished_match(), _stored_forecast("2026-07-14T21:00:00Z"), "HOME", 2, 1
    ) is None
    legacy = _stored_forecast()
    legacy.pop("generatedAt")
    assert _build_finished_forecast_result(
        "legacy", _finished_match(), legacy, "HOME", 2, 1
    ) is None
    unvalidated = _stored_forecast()
    unvalidated["forecastStatus"] = "UNVALIDATED"
    assert _build_finished_forecast_result(
        "unvalidated", _finished_match(), unvalidated, "HOME", 2, 1
    ) is None


def test_forecast_result_collection_uses_allow_list_and_stays_non_betting():
    stored = {}
    collections = []

    class Snapshot:
        exists = False

    class Document:
        def get(self):
            return Snapshot()

        def set(self, payload):
            stored.update(payload)

    class Collection:
        def document(self, _doc_id):
            return Document()

    class Db:
        def collection(self, name):
            collections.append(name)
            return Collection()

    writer = object.__new__(FirestoreWriter)
    writer.db = Db()
    result = _build_finished_forecast_result(
        "2026-07-14_france_spain",
        _finished_match(),
        _stored_forecast(),
        "HOME",
        2,
        1,
    )
    result.update({"odds": 2.5, "profit": 150, "recommendedBet": "HOME"})

    assert writer.save_forecast_result(result) is True
    assert collections == ["forecast_results"]
    assert stored["decisionStatus"] == "ABSTAIN"
    assert stored["eligibleForBetting"] is False
    assert "odds" not in stored
    assert "profit" not in stored
    assert "recommendedBet" not in stored


def test_forecast_history_payload_filters_scope_and_has_no_betting_metrics():
    valid = _build_finished_forecast_result(
        "2026-07-14_france_spain",
        _finished_match(),
        _stored_forecast(),
        "HOME",
        2,
        1,
    )
    invalid = {**valid, "matchId": "invalid", "decisionStatus": "BET"}
    payload = build_forecast_history_payload([valid, invalid])

    assert payload["scope"] == NON_BETTING_FORECAST_SCOPE
    assert payload["summary"]["totalForecasts"] == 1
    assert payload["summary"]["correctForecasts"] == 1
    assert payload["summary"]["forecastAccuracy"] == 100.0
    assert payload["summary"]["byCompetition"] == [
        {"competition": "WC", "total": 1, "correct": 1, "accuracy": 100.0}
    ]
    assert set(payload["summary"]).isdisjoint({"profit", "roi", "hitRate", "totalBets"})
    assert all("odds" not in result and "profit" not in result for result in payload["results"])


def test_finished_pipeline_writes_only_separate_forecast_result():
    saved_forecasts = []
    refreshes = []

    class Snapshot:
        exists = True

        def to_dict(self):
            return _stored_forecast()

    class Document:
        def get(self):
            return Snapshot()

    class Collection:
        def document(self, _doc_id):
            return Document()

    class Db:
        def collection(self, name):
            assert name == "model_outputs"
            return Collection()

    class Writer:
        db = Db()

        def update_match_result(self, *_args):
            pass

        def get_match(self, _mid):
            return {"status": "FINISHED"}

        def save_forecast_result(self, result):
            saved_forecasts.append(result)
            return True

        def save_prediction_result(self, _result):
            raise AssertionError("forecast-only output must not enter prediction_results")

        def refresh_coupon_history_cache(self):
            refreshes.append("coupon")

        def refresh_prediction_history_cache(self):
            refreshes.append("prediction")

        def refresh_paper_trading_cache(self, **_kwargs):
            refreshes.append("pnl")

        def refresh_forecast_history_cache(self):
            refreshes.append("forecast")

    pipeline = object.__new__(PredictionPipeline)
    pipeline.fs = Writer()
    pipeline._matches = [_finished_match()]
    pipeline._stats = {}
    pipeline.ab_enabled = False
    pipeline._evaluate_coupons = lambda _matches: None

    pipeline.evaluate_finished()

    assert len(saved_forecasts) == 1
    assert saved_forecasts[0]["evaluationScope"] == NON_BETTING_FORECAST_SCOPE
    assert pipeline._stats["results_saved"] == 0
    assert pipeline._stats["forecast_results_saved"] == 1
    assert refreshes.count("forecast") == 1
    assert refreshes.count("prediction") == 1
    assert refreshes.count("pnl") == 1


def test_model_output_persists_forecast_metadata_but_no_recommendation():
    stored = {}

    class Document:
        def set(self, payload):
            stored.update(payload)

    class Collection:
        def document(self, _mid):
            return Document()

    class Db:
        def collection(self, name):
            assert name == "model_outputs"
            return Collection()

    writer = object.__new__(FirestoreWriter)
    writer.db = Db()
    writer.save_model_output(
        "fixture",
        {"home": 0.56, "draw": 0.24, "away": 0.20},
        confidence=0,
        model_version="international_elo_forecast_only_v1",
        decision_status="ABSTAIN",
        forecast_status=VALIDATED_FORECAST_ONLY,
        forecast_outcome="HOME",
        forecast_confidence=0.56,
    )

    assert stored["forecastStatus"] == VALIDATED_FORECAST_ONLY
    assert stored["forecastOutcome"] == "HOME"
    assert stored["forecastConfidence"] == 0.56
    assert stored["evaluationScope"] == NON_BETTING_FORECAST_SCOPE
    assert stored["eligibleForBetting"] is False
    assert "recommendedBet" not in stored
