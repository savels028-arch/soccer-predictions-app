import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import (
    LABEL_AWAY,
    LABEL_DRAW,
    LABEL_HOME,
    _bet_label_for_style,
    _build_over_under_rows,
    _build_pattern_predictions,
    _compute_team_stats_snapshot,
    _csv_match_to_history_prediction,
    _enrich_directed_h2h_history,
    _enrich_strategy_zoo_history,
    _record_finished_match,
    _robust_score,
    simulate_coupon_batches,
    simulate_coupon_bankroll,
    simulate_flat_bankroll,
    walk_forward_csv_strategy_zoo,
)
from config.settings import LEAGUES, ML_SETTINGS_V2
from src.api.csv_football_client import FootballDataCSVClient, _season_code_for_year
from src.api.free_football_client import FreeFootballClient
from src.database.db_manager import DatabaseManager
from src.predictions.feature_engineering import FeatureEngineer, FeatureEngineerV2
from src.predictions.models import RandomForestModel
from src.predictions.prediction_engine import PredictionEngine
from run_pipeline import PredictionPipeline, _calibrate_probs_by_league, _historical_h2h_coupon_pick
from src.firestore_writer import build_coupon_history_payload


def test_world_cup_is_available_from_espn_feed():
    assert LEAGUES["WC"]["name"] == "FIFA World Cup"
    assert FreeFootballClient.ESPN_LEAGUES["WC"] == "fifa.world"


def test_csv_history_supports_1990s_warmup_seasons():
    assert _season_code_for_year(1993) == "9394"
    assert _season_code_for_year(1999) == "9900"
    assert _season_code_for_year(1992) is None
    assert FootballDataCSVClient.LEAGUE_CSV_MAP["BEL1"] == "B1"
    assert "BSA" not in FootballDataCSVClient.LEAGUE_CSV_MAP


def test_csv_parser_preserves_legacy_market_prices_and_stable_ids():
    client = FootballDataCSVClient()
    row = {
        "Date": "14/08/05",
        "HomeTeam": "Alpha",
        "AwayTeam": "Beta",
        "FTHG": "2",
        "FTAG": "1",
        "FTR": "H",
        # The incomplete B365 quote must not be mixed with another book.
        "B365H": "1.80",
        "BWH": "1.90",
        "BWD": "3.40",
        "BWA": "4.20",
        "BbAvH": "1.92",
        "BbAvD": "3.35",
        "BbAvA": "4.10",
        "BbMxH": "2.00",
        "BbMxD": "3.50",
        "BbMxA": "4.40",
        "BbAv>2.5": "1.85",
        "BbAv<2.5": "2.02",
        "BbMx>2.5": "1.92",
        "BbMx<2.5": "2.10",
        "BbAHh": "-0.50",
        "BbAvAHH": "1.95",
        "BbAvAHA": "1.93",
        "PAHH": "1.97",
        "PAHA": "1.91",
        "AHCh": "-0.75",
        "PCAHH": "2.01",
        "PCAHA": "1.89",
    }

    first = client._normalize_csv_row(row, "PL", LEAGUES["PL"], 2005)
    second = client._normalize_csv_row(row, "PL", LEAGUES["PL"], 2005)

    assert first["api_id"] == second["api_id"]
    assert first["home_odds"] == 1.9
    assert first["draw_odds"] == 3.4
    assert first["away_odds"] == 4.2
    assert first["extra_data"]["avg_home_odds"] == 1.92
    assert first["extra_data"]["max_away_odds"] == 4.4
    assert first["extra_data"]["avg_over25"] == 1.85
    assert first["extra_data"]["avg_under25"] == 2.02
    assert first["extra_data"]["asian_handicap_line"] == -0.5
    assert first["extra_data"]["pinnacle_asian_home"] == 1.97
    assert first["extra_data"]["pinnacle_asian_away"] == 1.91
    assert first["extra_data"]["asian_handicap_close_line"] == -0.75
    assert first["extra_data"]["pinnacle_close_asian_home"] == 2.01
    assert first["extra_data"]["pinnacle_close_asian_away"] == 1.89


def test_over_under_history_uses_unpriced_matches_and_legacy_average_odds():
    matches = [
        {
            "match_date": "1999-01-01T15:00:00",
            "league_code": "PL",
            "season": 1998,
            "home_team_name": "Alpha",
            "away_team_name": "Beta",
            "home_score": 2,
            "away_score": 1,
            "extra_data": {},
        },
        {
            "match_date": "2005-01-01T15:00:00",
            "league_code": "PL",
            "season": 2004,
            "home_team_name": "Alpha",
            "away_team_name": "Beta",
            "home_score": 1,
            "away_score": 0,
            "extra_data": {"avg_over25": 1.9, "avg_under25": 2.0},
        },
    ]

    rows = _build_over_under_rows(matches)

    assert len(rows) == 1
    assert rows[0]["odds_basis"] == "average_open"
    assert rows[0]["team_history_matches"] == 1
    assert rows[0]["pair_history_matches"] == 1


def test_over_under_history_batches_simultaneous_matches_before_updates():
    priced = {"avg_over25": 1.9, "avg_under25": 2.0}
    matches = [
        {
            "match_date": "2005-01-01T15:00:00",
            "league_code": "PL",
            "season": 2004,
            "home_team_name": "Alpha",
            "away_team_name": "Beta",
            "home_score": 2,
            "away_score": 1,
            "extra_data": priced,
        },
        {
            "match_date": "2005-01-01T15:00:00",
            "league_code": "PL",
            "season": 2004,
            "home_team_name": "Gamma",
            "away_team_name": "Delta",
            "home_score": 0,
            "away_score": 0,
            "extra_data": priced,
        },
        {
            "match_date": "2005-01-02T15:00:00",
            "league_code": "PL",
            "season": 2004,
            "home_team_name": "Alpha",
            "away_team_name": "Gamma",
            "home_score": 1,
            "away_score": 1,
            "extra_data": priced,
        },
    ]

    rows = _build_over_under_rows(matches)

    assert rows[0]["league_history_matches"] == 0
    assert rows[1]["league_history_matches"] == 0
    assert rows[2]["league_history_matches"] == 2


def test_odds_only_cache_refresh_preserves_existing_predictions():
    class FakeWriter:
        def __init__(self):
            self.writes = []

        def write_cache(self, name, data):
            self.writes.append(name)

        def refresh_coupon_history_cache(self):
            return {}

        def refresh_prediction_history_cache(self):
            return {}

        def refresh_paper_trading_cache(self, **kwargs):
            return {"totalBets": 0, "totalProfit": 0, "roi": 0}

    pipeline = object.__new__(PredictionPipeline)
    pipeline.fs = FakeWriter()
    pipeline._matches = []
    pipeline._ml_preds = {}
    pipeline._odds = []

    pipeline.write_legacy_cache(preserve_predictions=True)

    assert "matches" in pipeline.fs.writes
    assert "ai_predictions" not in pipeline.fs.writes
    assert "ml_predictions" not in pipeline.fs.writes


def test_legacy_cache_never_substitutes_model_probabilities_for_market_odds():
    class FakeWriter:
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
    pipeline.fs = FakeWriter()
    pipeline._matches = []
    pipeline._odds = []
    pipeline._ml_preds = {
        "fixture": {
            "home_team": "Alpha",
            "away_team": "Beta",
            "league": "PL",
            "match_date": "2026-07-15T18:00:00Z",
            "ensemble": {"home": 0.7, "draw": 0.2, "away": 0.1},
            "edge": {"home": 0.2},
            "recommended": "HOME",
            "decision_status": "BET",
            "models": {},
        },
    }

    pipeline.write_legacy_cache()

    market = pipeline.fs.cache["ml_predictions"]["odds_matches"][0]
    assert market["odds_1"] == 0
    assert market["odds_x"] == 0
    assert market["odds_2"] == 0
    assert market["odds_available"] is False
    assert market["value_bet"] is False
    assert market["value_edge"] == 0


def _match(api_id, home, away, when, home_score, away_score, season=2024):
    return {
        "api_id": api_id,
        "home_team_name": home,
        "away_team_name": away,
        "league_code": "PL",
        "season": season,
        "match_date": when,
        "status": "FINISHED",
        "home_score": home_score,
        "away_score": away_score,
    }


def _prediction(when, predicted, actual, odds, season=2024, edge=0.06, confidence=0.55):
    return {
        "match_date": when,
        "league": "PL",
        "season": season,
        "home": f"Home {when}",
        "away": f"Away {when}",
        "predicted": predicted,
        "actual": actual,
        "confidence": confidence,
        "edge": edge,
        "home_odds": odds if predicted == LABEL_HOME else 2.0,
        "draw_odds": odds if predicted == LABEL_DRAW else 3.0,
        "away_odds": odds if predicted == LABEL_AWAY else 2.5,
    }


def _base_stats(team_name):
    return {
        "team_name": team_name,
        "league_code": "PL",
        "season": 2024,
        "matches_played": 6,
        "wins": 3,
        "draws": 2,
        "losses": 1,
        "goals_scored": 10,
        "goals_conceded": 6,
        "clean_sheets": 2,
        "home_wins": 2,
        "home_draws": 1,
        "home_losses": 0,
        "away_wins": 1,
        "away_draws": 1,
        "away_losses": 1,
        "home_goals_scored": 6,
        "home_goals_conceded": 2,
        "away_goals_scored": 4,
        "away_goals_conceded": 4,
        "form": "WWDLW",
        "avg_goals_scored": 1.67,
        "avg_goals_conceded": 1.0,
    }


class DummyModel:
    def __init__(self):
        self.is_trained = True
        self.seen_dims = []

    def predict_proba(self, X):
        arr = np.asarray(X)
        self.seen_dims.append(arr.shape[-1])
        return np.array([[0.2, 0.3, 0.5]])


def test_build_training_data_uses_only_pre_match_stats(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "train.db")
    matches = [
        _match(1, "A", "X", "2024-01-01T12:00:00", 2, 0),
        _match(2, "B", "P", "2024-01-01T15:00:00", 0, 1),
        _match(3, "A", "Y", "2024-01-02T12:00:00", 0, 1),
        _match(4, "B", "Q", "2024-01-02T15:00:00", 1, 1),
        _match(5, "A", "Z", "2024-01-03T12:00:00", 3, 0),
        _match(6, "B", "R", "2024-01-03T15:00:00", 0, 2),
        _match(7, "A", "B", "2024-01-04T12:00:00", 1, 0),
        _match(8, "A", "F", "2024-01-05T12:00:00", 0, 4),
        _match(9, "G", "B", "2024-01-05T15:00:00", 2, 1),
    ]

    for match in matches:
        db.upsert_match(match)

    # Poison the cache with full-season stats. The training builder must ignore them.
    for team in ["A", "B", "X", "Y", "Z", "P", "Q", "R", "F", "G"]:
        stats = db.compute_team_stats_from_matches(team, "PL", 2024)
        if stats.get("matches_played", 0):
            db.upsert_team_stats(stats)

    X, y, dates = FeatureEngineer.build_training_data(matches, db)

    assert dates == ["2024-01-04T12:00:00"]
    assert y.tolist() == [0]
    assert X[0][0] == pytest.approx(2 / 3, rel=1e-3)
    assert X[0][1] == pytest.approx(0.0, abs=1e-6)
    assert X[0][2] == pytest.approx(1 / 3, rel=1e-3)
    assert X[0][3] == pytest.approx(0.0, abs=1e-6)
    assert X[0][4] == pytest.approx(1 / 3, rel=1e-3)
    assert X[0][5] == pytest.approx(2 / 3, rel=1e-3)


def test_v2_models_expect_v2_feature_count():
    model = RandomForestModel(config=ML_SETTINGS_V2, suffix="_test_v2")
    assert model._expected_features() == len(FeatureEngineerV2.FEATURE_NAMES)


def test_prediction_engine_v2_uses_v2_feature_width(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "predict.db")
    db.upsert_team_stats(_base_stats("Home"))
    db.upsert_team_stats(_base_stats("Away"))

    engine = PredictionEngine(db_manager=db, config=ML_SETTINGS_V2, suffix="_test_v2", version_label="v2")
    base_model = DummyModel()
    ensemble_model = DummyModel()
    engine.models = {"dummy": base_model}
    engine.ensemble = ensemble_model

    match = {
        "api_id": 100,
        "home_team_name": "Home",
        "away_team_name": "Away",
        "league_code": "PL",
        "season": 2024,
        "match_date": "2024-02-01T20:00:00",
        "home_odds": 2.1,
        "draw_odds": 3.4,
        "away_odds": 3.6,
    }

    predictions = engine.predict_match(match)

    expected = len(FeatureEngineerV2.FEATURE_NAMES)
    assert base_model.seen_dims == [expected]
    assert ensemble_model.seen_dims == [expected]
    assert {
        p["predicted_outcome"]
        for p in predictions
        if p["model_name"] in {"dummy", "ensemble"}
    } == {"AWAY_WIN"}


def test_database_tracks_odds_pick_context_and_league_priors(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "tracking.db")
    db.save_odds_snapshot(
        "2024-01-01_home_away",
        "2024-01-01T12:00:00",
        "Home",
        "Away",
        "danske_spil",
        {"home": 2.0, "draw": 3.5, "away": 4.0},
    )
    latest = db.get_latest_odds_snapshot("2024-01-01_home_away")
    assert latest["source"] == "danske_spil"
    assert latest["implied_home"] == pytest.approx(0.4828)

    pick_id = db.save_pick_snapshot(
        "2024-01-01_home_away",
        "2024-01-01T12:00:00",
        "Home",
        "Away",
        "ML Ensemble v1",
        "HOME",
        2.1,
        probability=0.52,
        edge=0.04,
        closing_odds=2.0,
    )
    assert pick_id > 0

    db.upsert_match_context(
        "2024-01-01_home_away",
        "api_football",
        "2024-01-01T12:00:00",
        "Home",
        "Away",
        {
            "summary": {
                "home_missing_players": 2,
                "away_missing_players": 1,
                "home_lineup_players": 11,
                "away_lineup_players": 11,
                "home_player_rating_avg": 6.9,
            }
        },
    )
    context = db.get_match_context("2024-01-01_home_away")
    assert context["home_missing_players"] == 2
    assert context["home_player_rating_avg"] == pytest.approx(6.9)

    for idx in range(90):
        db.upsert_match({
            "api_id": 10_000 + idx,
            "league_code": "PL",
            "season": 2024,
            "match_date": f"2024-02-{(idx % 28) + 1:02d}T12:00:00",
            "status": "FINISHED",
            "home_team_name": f"H{idx}",
            "away_team_name": f"A{idx}",
            "home_score": 2 if idx % 3 == 0 else 1,
            "away_score": 1 if idx % 3 == 0 else (1 if idx % 3 == 1 else 2),
        })
    priors = db.get_league_outcome_priors("PL", min_matches=80)
    assert priors == pytest.approx({"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}, rel=1e-3)


def test_get_h2h_falls_back_to_finished_matches_when_cache_empty(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "h2h.db")
    db.upsert_match(_match(301, "Arsenal", "Everton", "2023-01-01T15:00:00", 2, 0))
    db.upsert_match(_match(302, "Everton", "Arsenal", "2024-01-01T15:00:00", 1, 3))
    scheduled = _match(303, "Arsenal", "Everton", "2024-05-01T15:00:00", None, None)
    scheduled["status"] = "SCHEDULED"
    db.upsert_match(scheduled)

    cache_count = db.conn.execute("SELECT COUNT(*) FROM head_to_head").fetchone()[0]
    assert cache_count == 0

    rows = db.get_h2h("Arsenal FC", "Everton", limit=10, before_date="2025-01-01")

    assert len(rows) == 2
    assert {row["match_date"] for row in rows} == {
        "2023-01-01T15:00:00",
        "2024-01-01T15:00:00",
    }
    assert all(row["home_score"] is not None and row["away_score"] is not None for row in rows)


def test_league_calibration_blends_toward_priors():
    calibrated, meta = _calibrate_probs_by_league(
        {"home": 0.60, "draw": 0.20, "away": 0.20},
        {"home": 0.45, "draw": 0.30, "away": 0.25},
        0.20,
    )

    assert meta["applied"] is True
    assert calibrated["home"] == pytest.approx(0.57)
    assert calibrated["draw"] == pytest.approx(0.22)
    assert calibrated["away"] == pytest.approx(0.21)


def test_coupon_history_payload_keeps_skipped_days_and_excludes_bad_legacy_docs():
    payload = build_coupon_history_payload([
        {
            "date": "2026-05-24",
            "status": "skipped",
            "reason": "not_enough_quality_picks",
            "picks": [],
        },
        {
            "date": "2026-05-23",
            "status": "lost",
            "picks": [{"league": "PL"}],
            "pickResults": ["lost"],
        },
        {
            "date": "2026-05-22",
            "status": "won",
            "picks": [{"league": "PL"}],
            "pickResults": [],
        },
        {
            "date": "2026-05-21",
            "status": "pending",
            "picks": [],
        },
    ])

    assert payload["total"] == 2
    assert payload["skipped"] == 1
    assert payload["lost"] == 1
    assert payload["totalPicks"] == 1
    assert payload["coupons"][0]["status"] == "skipped"
    assert payload["coupons"][0]["reason"] == "not_enough_quality_picks"


def test_prediction_results_normalize_legacy_outcome_labels(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "results.db")

    was_saved = db.save_prediction_result({
        "match_date": "2024-02-01T20:00:00",
        "home_team": "Home",
        "away_team": "Away",
        "league_code": "PL",
        "home_score": 2,
        "away_score": 1,
        "actual_outcome": "HOME_WIN",
        "predicted_outcome": "Home Win",
        "confidence": 0.61,
        "source": "ensemble",
        "is_correct": False,
    })

    results = db.get_all_prediction_results()

    assert was_saved is True
    assert results[0]["actual_outcome"] == "HOME_WIN"
    assert results[0]["predicted_outcome"] == "HOME_WIN"
    assert results[0]["is_correct"] == 1

    was_updated = db.save_prediction_result({
        "match_date": "2024-02-01T20:00:00",
        "home_team": "Home",
        "away_team": "Away",
        "league_code": "PL",
        "home_score": 2,
        "away_score": 1,
        "actual_outcome": "HOME_WIN",
        "predicted_outcome": "HOME-WIN",
        "confidence": 0.66,
        "source": "ensemble",
        "is_correct": False,
    })

    results = db.get_all_prediction_results()
    assert was_updated is False
    assert len(results) == 1
    assert results[0]["confidence"] == pytest.approx(0.66)


def test_prediction_accuracy_crosscheck_accepts_legacy_labels(tmp_path):
    db = DatabaseManager(db_path=tmp_path / "accuracy.db")
    match_id = db.upsert_match(_match(100, "Home", "Away", "2024-02-01T20:00:00", 2, 1))

    db.save_prediction({
        "match_id": match_id,
        "match_date": "2024-02-01T20:00:00",
        "home_team": "Home",
        "away_team": "Away",
        "league_code": "PL",
        "model_name": "ensemble",
        "predicted_outcome": "Home Win",
        "confidence": 0.61,
    })

    accuracy = db.get_prediction_accuracy()

    assert accuracy["by_model_crosscheck"][0]["model_name"] == "ensemble"
    assert accuracy["by_model_crosscheck"][0]["total"] == 1
    assert accuracy["by_model_crosscheck"][0]["correct"] == 1


def test_backtest_records_skipped_matches_into_rolling_history():
    team_idx = defaultdict(list)
    h2h_idx = defaultdict(list)

    warmup_matches = [
        _match(200, "Home", "Opp A", "2024-08-01T12:00:00", 1, 0),
        _match(201, "Home", "Opp B", "2024-08-08T12:00:00", 2, 2),
        _match(202, "Home", "Opp C", "2024-08-15T12:00:00", 0, 1),
    ]

    for match in warmup_matches:
        assert _compute_team_stats_snapshot(team_idx["Home"], "Home", "PL", 2024) is None
        assert _record_finished_match(team_idx, h2h_idx, match) is True

    stats = _compute_team_stats_snapshot(team_idx["Home"], "Home", "PL", 2024)

    assert stats["matches_played"] == 3
    assert len(team_idx["Home"]) == 3
    assert len(h2h_idx[("Home", "Opp A")]) == 1


def test_flat_bankroll_uses_fixed_stake_and_tracks_seasons():
    predictions = [
        _prediction("2024-01-01T12:00:00", LABEL_HOME, LABEL_HOME, 2.0),
        _prediction("2024-01-02T12:00:00", LABEL_HOME, LABEL_AWAY, 2.0),
        _prediction("2024-01-03T12:00:00", LABEL_DRAW, LABEL_DRAW, 3.0),
    ]

    result = simulate_flat_bankroll(predictions, starting_bankroll=10_000, stake=100)

    assert result["bets"] == 3
    assert result["wins"] == 2
    assert result["staked"] == 300
    assert result["returned"] == 500
    assert result["final_bankroll"] == 10_200
    assert result["profit"] == 200
    assert result["max_drawdown"] == 100
    assert result["by_season"]["2024"]["bets"] == 3


def test_coupon_bankroll_groups_by_day_and_respects_max_legs():
    predictions = [
        _prediction(f"2024-01-01T1{i}:00:00", LABEL_HOME, LABEL_HOME, 2.0, confidence=0.70 - i * 0.01)
        for i in range(5)
    ]

    result = simulate_coupon_bankroll(
        predictions,
        starting_bankroll=10_000,
        stake=100,
        min_legs=2,
        max_legs=3,
    )

    assert result["coupons"] == 2
    assert result["winning_coupons"] == 2
    assert result["legs_played"] == 5
    assert result["avg_legs"] == 2.5
    assert result["staked"] == 200
    assert result["returned"] == 1_200
    assert result["final_bankroll"] == 11_000


def test_coupon_bankroll_can_limit_league_concentration():
    predictions = [
        dict(_prediction("2024-01-01T10:00:00", LABEL_HOME, LABEL_HOME, 2.0, confidence=0.90), league="PL"),
        dict(_prediction("2024-01-01T11:00:00", LABEL_HOME, LABEL_HOME, 2.0, confidence=0.89), league="PL"),
        dict(_prediction("2024-01-01T12:00:00", LABEL_HOME, LABEL_HOME, 2.0, confidence=0.88), league="PD"),
        dict(_prediction("2024-01-01T13:00:00", LABEL_HOME, LABEL_HOME, 2.0, confidence=0.87), league="SA"),
    ]

    result = simulate_coupon_bankroll(
        predictions,
        starting_bankroll=10_000,
        stake=100,
        min_legs=2,
        max_legs=3,
        max_per_league=1,
        sort_by="confidence",
    )

    assert result["coupons"] == 1
    assert result["legs_played"] == 3
    assert result["returned"] == 800


def test_optimizer_score_rejects_small_or_fragile_samples():
    simulation = {
        "bets": 12,
        "profit": 1200,
        "max_drawdown": 200,
        "by_season": {
            "2023": {"profit": 1200},
            "2024": {"profit": -100},
            "2025": {"profit": -100},
        },
    }

    score, eligible, reasons = _robust_score(simulation, "bets", 100)

    assert score == pytest.approx(1200 - 100 - 3000 - 25)
    assert eligible is False
    assert "too_few_bets" in reasons
    assert "not_profitable_enough_seasons" in reasons


def test_least_likely_bet_style_can_be_simulated():
    predictions = [
        {
            **_prediction(
                f"2024-01-{day:02d}T12:00:00",
                LABEL_HOME,
                LABEL_AWAY,
                1.5,
                edge=0.10,
                confidence=0.70,
            ),
            "home_prob": 0.70,
            "draw_prob": 0.20,
            "away_prob": 0.10,
            "away_odds": 12.0,
        }
        for day in range(1, 21)
    ]

    normal = simulate_flat_bankroll(predictions, starting_bankroll=10_000, stake=100)
    contrarian = simulate_flat_bankroll(
        predictions,
        starting_bankroll=10_000,
        stake=100,
        bet_label_fn=lambda p: _bet_label_for_style(p, "least_likely"),
    )

    assert _bet_label_for_style(predictions[0], "least_likely") == LABEL_AWAY
    assert normal["profit"] < 0
    assert contrarian["profit"] > 0


def test_historical_pattern_predictions_use_only_prior_matches():
    def match(day, home_score, away_score):
        actual = LABEL_HOME if home_score > away_score else LABEL_DRAW if home_score == away_score else LABEL_AWAY
        return {
            "match_date": f"2024-01-{day:02d}T12:00:00",
            "league": "PL",
            "season": 2024,
            "home": "Arsenal",
            "away": "Everton",
            "home_score": home_score,
            "away_score": away_score,
            "actual": actual,
            "predicted": LABEL_DRAW,
            "confidence": 0.33,
            "home_prob": 0.33,
            "draw_prob": 0.33,
            "away_prob": 0.34,
            "home_odds": 2.0,
            "draw_odds": 3.2,
            "away_odds": 4.0,
            "edge": 0.0,
            "kelly": 0.0,
        }

    matches = [
        match(1, 2, 0),
        match(2, 1, 0),
        match(3, 3, 1),
    ]

    selected = _build_pattern_predictions(
        matches,
        pattern="directed_h2h_outcome",
        min_matches=2,
        min_rate=1.0,
        max_odds=None,
    )

    assert len(selected) == 1
    assert selected[0]["match_date"] == "2024-01-03T12:00:00"
    assert selected[0]["predicted"] == LABEL_HOME
    assert selected[0]["pattern_history_count"] == 2


def test_historical_edge_h2h_enrichment_uses_only_prior_matches():
    matches = [
        {
            "match_date": "2024-01-01T12:00:00",
            "league": "PL",
            "season": 2024,
            "home": "Arsenal",
            "away": "Everton",
            "actual": LABEL_HOME,
        },
        {
            "match_date": "2024-02-01T12:00:00",
            "league": "PL",
            "season": 2024,
            "home": "Arsenal",
            "away": "Everton",
            "actual": LABEL_HOME,
        },
        {
            "match_date": "2024-03-01T12:00:00",
            "league": "PL",
            "season": 2024,
            "home": "Arsenal",
            "away": "Everton",
            "actual": LABEL_AWAY,
        },
    ]

    enriched = _enrich_directed_h2h_history(matches)

    assert enriched[0]["_h2h_count"] == 0
    assert enriched[1]["_h2h_count"] == 1
    assert enriched[1]["_h2h_label"] == LABEL_HOME
    assert enriched[2]["_h2h_count"] == 2
    assert enriched[2]["_h2h_label"] == LABEL_HOME
    assert enriched[2]["_h2h_rate"] == pytest.approx(1.0)


def test_csv_match_to_history_prediction_requires_scores_and_odds():
    match = {
        "match_date": "2000-08-19T15:00:00",
        "league_code": "PL",
        "season": 2000,
        "home_team_name": "Charlton",
        "away_team_name": "Man City",
        "home_score": 4,
        "away_score": 0,
        "home_odds": 2.1,
        "draw_odds": 3.2,
        "away_odds": 3.6,
        "extra_data": {
            "avg_home_odds": 2.05,
            "avg_draw_odds": 3.15,
            "avg_away_odds": 3.5,
            "max_home_odds": 2.2,
            "max_draw_odds": 3.3,
            "max_away_odds": 3.8,
            "b365_close_home": 2.0,
            "b365_close_draw": 3.1,
            "b365_close_away": 3.4,
            "avg_close_home_odds": 2.02,
            "avg_close_draw_odds": 3.12,
            "avg_close_away_odds": 3.45,
        },
    }

    prediction = _csv_match_to_history_prediction(match)

    assert prediction["home"] == "Charlton"
    assert prediction["away"] == "Man City"
    assert prediction["actual"] == LABEL_HOME
    assert prediction["home_odds"] == pytest.approx(2.1)
    assert prediction["avg_home_odds"] == pytest.approx(2.05)
    assert prediction["max_away_odds"] == pytest.approx(3.8)
    assert prediction["b365_close_home"] == pytest.approx(2.0)
    assert prediction["avg_close_away_odds"] == pytest.approx(3.45)

    match_without_odds = dict(match)
    match_without_odds["draw_odds"] = None
    assert _csv_match_to_history_prediction(match_without_odds) is None


def test_strategy_zoo_pair_history_uses_only_prior_matches():
    matches = []
    for month, home, away, actual in [
        (1, "A", "B", LABEL_HOME),
        (2, "B", "A", LABEL_AWAY),
        (3, "A", "B", LABEL_HOME),
    ]:
        matches.append({
            "match_date": f"2024-{month:02d}-01T12:00:00",
            "league": "PL",
            "season": 2024,
            "home": home,
            "away": away,
            "actual": actual,
            "predicted": -1,
            "home_odds": 1.8,
            "draw_odds": 3.3,
            "away_odds": 4.5,
        })

    enriched = _enrich_strategy_zoo_history(matches)

    assert enriched[0]["_pair_count"] == 0
    assert enriched[1]["_pair_count"] == 1
    assert enriched[1]["_pair_label"] == LABEL_AWAY
    assert enriched[2]["_pair_count"] == 2
    assert enriched[2]["_pair_label"] == LABEL_HOME
    assert enriched[2]["_pair_rate"] == pytest.approx(1.0)


def test_coupon_batch_simulation_matches_coupon_bankroll():
    predictions = [
        _prediction("2024-01-01T12:00:00", LABEL_HOME, LABEL_HOME, 1.7, confidence=0.8),
        _prediction("2024-01-01T14:00:00", LABEL_AWAY, LABEL_AWAY, 2.1, confidence=0.7),
        _prediction("2024-01-02T12:00:00", LABEL_HOME, LABEL_AWAY, 1.6, confidence=0.8),
        _prediction("2024-01-02T14:00:00", LABEL_DRAW, LABEL_DRAW, 3.2, confidence=0.6),
    ]

    batches, skipped = _build_coupon_batches_for_test(predictions)
    direct = simulate_coupon_bankroll(predictions, max_legs=2, sort_by="confidence", max_per_league=2)
    from_batches = simulate_coupon_batches(
        batches,
        max_legs=2,
        sort_by="confidence",
        max_per_league=2,
        skipped_no_odds=skipped,
    )

    assert from_batches["profit"] == pytest.approx(direct["profit"])
    assert from_batches["coupons"] == direct["coupons"]


def _build_coupon_batches_for_test(predictions):
    from backtest import _build_coupon_batches

    return _build_coupon_batches(
        predictions,
        max_legs=2,
        sort_by="confidence",
        max_per_league=2,
    )


def test_strategy_zoo_walk_forward_selects_from_prior_seasons():
    matches = []
    for season, date, home_score, away_score in [
        (2020, "2020-08-01T12:00:00", 2, 0),
        (2021, "2021-08-01T12:00:00", 3, 1),
        (2022, "2022-08-01T12:00:00", 2, 1),
        (2023, "2023-08-01T12:00:00", 1, 0),
    ]:
        matches.append({
            "match_date": date,
            "league_code": "PL",
            "season": season,
            "home_team_name": "Arsenal",
            "away_team_name": "Everton",
            "home_score": home_score,
            "away_score": away_score,
            "home_odds": 1.8,
            "draw_odds": 3.4,
            "away_odds": 4.8,
        })

    result = walk_forward_csv_strategy_zoo(
        matches,
        2020,
        2023,
        first_test_season=2023,
        min_train_bets=1,
    )

    assert result["combined_single"]["bets"] == 1
    assert result["combined_single"]["profit"] > 0
    assert result["folds"][0]["chosen_single"]["test_simulation"]["bets"] == 1


def test_historical_h2h_coupon_pick_uses_exact_home_away_fixture_only():
    h2h = [
        {"home_team": "Arsenal", "away_team": "Everton", "home_score": 2, "away_score": 0},
        {"home_team": "Everton", "away_team": "Arsenal", "home_score": 1, "away_score": 3},
        {"home_team": "Arsenal", "away_team": "Everton", "home_score": 1, "away_score": 0},
        {"home_team": "Everton", "away_team": "Arsenal", "home_score": 0, "away_score": 2},
    ]

    pick = _historical_h2h_coupon_pick(
        "Arsenal",
        "Everton",
        h2h,
        {"home": 1.8, "draw": 3.4, "away": 4.8},
        {
            "min_h2h_matches": 2,
            "min_h2h_rate_pct": 75.0,
            "min_edge_pct": 2.0,
            "odds_min": 1.2,
            "odds_max": 2.5,
        },
    )

    assert pick["pick"] == "home"
    assert pick["h2h_count"] == 2
    assert pick["h2h_rate_pct"] == pytest.approx(100.0)
    assert pick["edge"] > 0
