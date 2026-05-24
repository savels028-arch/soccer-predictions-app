import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.prediction_scraper import PredictionScraper


def test_predictionpitch_json_source_parses_probabilities(monkeypatch):
    scraper = PredictionScraper()
    payload = [
        {
            "homeTeam": "Belgrano",
            "awayTeam": "Union Santa Fe",
            "league": "Liga Profesional",
            "matchTime": "2026-05-12T22:00:00.000Z",
            "homeWinProb": 45.5,
            "drawProb": 31.5,
            "awayWinProb": 23,
            "overUnder25": 38.8,
            "bttsYesProb": 34,
            "bestHomeOdds": 2.46,
            "bestDrawOdds": 3.15,
            "bestAwayOdds": 3.33,
            "isValueBet": True,
            "valueBetMarket": "home",
        }
    ]
    monkeypatch.setattr(scraper, "_safe_get", lambda *_args, **_kwargs: json.dumps(payload))

    predictions = scraper._scrape_predictionpitch(target_date="2026-05-12")

    assert predictions == [
        {
            "home_team": "Belgrano",
            "away_team": "Union Santa Fe",
            "league": "Liga Profesional",
            "kickoff_time": "2026-05-12T22:00:00.000Z",
            "home_win_pct": 45.5,
            "draw_pct": 31.5,
            "away_win_pct": 23.0,
            "predicted_winner": "1",
            "over_under_25": "Under",
            "btts": "No",
            "odds_home": 2.46,
            "odds_draw": 3.15,
            "odds_away": 3.33,
            "value_bet": True,
            "value_bet_market": "home",
        }
    ]


def test_winfulltime_json_source_parses_matches(monkeypatch):
    scraper = PredictionScraper()
    payload = {
        "date": "2026-05-12",
        "matches": [
            {
                "league": "Brazil Serie A",
                "time": "20:30",
                "match": "Santos - Palmeiras",
                "probabilities": {"homeWin": 29, "draw": 31, "awayWin": 40},
                "tip": "2",
                "date": "2026-05-12",
            },
            {
                "league": "Old",
                "time": "10:00",
                "match": "Old Home - Old Away",
                "probabilities": {"homeWin": 60, "draw": 25, "awayWin": 15},
                "tip": "1",
                "date": "2026-05-11",
            },
        ],
    }
    monkeypatch.setattr(scraper, "_safe_get", lambda *_args, **_kwargs: json.dumps(payload))

    predictions = scraper._scrape_winfulltime(target_date="2026-05-12")

    assert predictions == [
        {
            "home_team": "Santos",
            "away_team": "Palmeiras",
            "league": "Brazil Serie A",
            "kickoff_time": "20:30",
            "home_win_pct": 29.0,
            "draw_pct": 31.0,
            "away_win_pct": 40.0,
            "predicted_winner": "2",
        }
    ]
