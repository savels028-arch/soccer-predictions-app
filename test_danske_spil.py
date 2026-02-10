#!/usr/bin/env python3
"""Test Danske Spil Client"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src.api.danske_spil_client import DanskeSpilClient

ds = DanskeSpilClient()
print("OK: DanskeSpilClient importeret")
print(f"OK: BASE_URL = {ds.BASE_URL}")
print(f"OK: Ligaer: {list(ds.LEAGUE_PATHS.keys())}")

# Test normalisering
assert ds._norm("FC Barcelona") == "barcelona", f"Got: {ds._norm('FC Barcelona')}"
assert ds._norm("Arsenal FC") == "arsenal", f"Got: {ds._norm('Arsenal FC')}"
assert ds._norm("Paris Saint-Germain FC") == "psg", f"Got: {ds._norm('Paris Saint-Germain FC')}"
assert ds._norm("FC Bayern München") == "bayern", f"Got: {ds._norm('FC Bayern München')}"
assert ds._norm("Borussia Dortmund") == "dortmund", f"Got: {ds._norm('Borussia Dortmund')}"
print("OK: Normalisering virker")

# Test team name parsing
h, a = ds._parse_team_names("Arsenal FC - Chelsea FC")
assert h == "Arsenal FC" and a == "Chelsea FC", f"Got: {h}, {a}"
print("OK: Team name parsing virker")

# Test matching
pred = [{"home_team": "Arsenal FC", "away_team": "Chelsea FC"}]
ds_events = [{
    "event_id": 123, "home_team": "Arsenal", "away_team": "Chelsea",
    "home_odds": 1.85, "draw_odds": 3.50, "away_odds": 4.20,
    "match_date": "2026-02-10T15:00:00Z", "league": "Premier League",
    "over_25_odds": None, "under_25_odds": None,
    "btts_yes_odds": None, "btts_no_odds": None,
    "deeplink": "https://danskespil.dk/oddset/event/123/test",
}]
matched = ds.match_predictions_with_odds(pred, ds_events)
assert matched[0]["danske_spil"]["available"] == True
assert matched[0]["danske_spil"]["home_odds"] == 1.85
print(f"OK: Matching virker (available={matched[0]['danske_spil']['available']})")

# Test fuzzy matching with slightly different names
pred2 = [{"home_team": "Manchester City FC", "away_team": "Liverpool FC"}]
ds_events2 = [{
    "event_id": 456, "home_team": "Manchester City", "away_team": "Liverpool",
    "home_odds": 1.55, "draw_odds": 4.00, "away_odds": 5.50,
    "match_date": "2026-02-10T17:30:00Z", "league": "Premier League",
    "over_25_odds": 1.70, "under_25_odds": 2.10,
    "btts_yes_odds": 1.65, "btts_no_odds": 2.20,
    "deeplink": None,
}]
matched2 = ds.match_predictions_with_odds(pred2, ds_events2)
assert matched2[0]["danske_spil"]["available"] == True
print(f"OK: Fuzzy matching virker (Man City vs Liverpool)")

# Test non-matching
pred3 = [{"home_team": "Unknown Team FC", "away_team": "Another Team SC"}]
matched3 = ds.match_predictions_with_odds(pred3, ds_events)
assert matched3[0]["danske_spil"]["available"] == False
print("OK: Non-matching returnerer available=False")

print("\n=== ALLE TESTS BESTAAET ===")
