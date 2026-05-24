import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.web.app import create_flask_app


class FakeDb:
    def __init__(self, predictions=None):
        self.saved_results = []
        self.predictions = predictions or []

    def get_match_count(self):
        return 1

    def get_prediction_count(self):
        return 0

    def get_predictions_by_teams(self, home, away):
        return self.predictions

    def save_prediction_result(self, result):
        self.saved_results.append(result)
        return True

    def get_prediction_results_summary(self):
        return {"overall": {"total": len(self.saved_results), "correct": 0}}

    def get_all_prediction_results(self):
        return self.saved_results


class FakeAggregator:
    def __init__(self):
        self.match_force_calls = []
        self.consensus_force_calls = []

    def fetch_todays_matches(self, force_refresh=False):
        self.match_force_calls.append(force_refresh)
        return [
            {
                "api_id": 1,
                "league_code": "PL",
                "match_date": "2026-05-07T20:00:00",
                "status": "FINISHED",
                "home_team_name": "Home",
                "away_team_name": "Away",
                "home_score": 2,
                "away_score": 1,
            }
        ]

    def fetch_ai_predictions(self, force_refresh=False):
        return []

    def build_consensus_with_danske_spil(self, prediction_engine=None, matches=None, force_refresh=False):
        self.consensus_force_calls.append(force_refresh)
        return {
            "all_consensus": [],
            "playable": [],
            "agreed": [],
            "stats": {
                "total_matches": 0,
                "ai_predictions": 0,
                "ml_predictions": 0,
                "ds_events": 0,
                "agreed": 0,
                "playable": 0,
            },
        }


class FakeEngine:
    is_trained = True

    def predict_all_matches(self, matches):
        return {}


def _client(predictions=None):
    db = FakeDb(predictions=predictions)
    aggregator = FakeAggregator()
    app = create_flask_app(
        db,
        aggregator,
        FakeEngine(),
        start_background_workers=False,
    )
    return app.test_client(), aggregator, db


def test_matches_route_respects_force_refresh_flag():
    client, aggregator, _ = _client()

    client.get("/api/matches")
    client.get("/api/matches?force=1")

    assert aggregator.match_force_calls[:2] == [False, True]


def test_history_update_forces_fresh_match_data():
    client, aggregator, _ = _client()

    response = client.post("/api/history/update")

    assert response.status_code == 200
    assert aggregator.match_force_calls[-1] is True


def test_consensus_route_forces_fresh_matches_and_odds_by_default():
    client, aggregator, _ = _client()

    response = client.get("/api/consensus_danske_spil")

    assert response.status_code == 200
    assert aggregator.match_force_calls[-1] is True
    assert aggregator.consensus_force_calls == [True]
    assert "last_update" in response.get_json()


def test_history_update_normalizes_legacy_model_outcomes():
    client, _, db = _client(predictions=[{
        "model_name": "ensemble",
        "predicted_outcome": "Home Win",
        "confidence": 0.62,
        "home_win_prob": 0.62,
        "draw_prob": 0.21,
        "away_win_prob": 0.17,
    }])

    response = client.post("/api/history/update")

    assert response.status_code == 200
    assert db.saved_results[0]["predicted_outcome"] == "HOME_WIN"
    assert db.saved_results[0]["actual_outcome"] == "HOME_WIN"
    assert db.saved_results[0]["is_correct"] is True
