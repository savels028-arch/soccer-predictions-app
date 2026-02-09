"""
API-Football Client (api-sports.io)
Free tier: 100 requests/day
Docs: https://www.api-football.com/documentation-v3
Register: https://dashboard.api-football.com/register
"""
import requests
import logging
import time
from datetime import date
from typing import Optional, List, Dict, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import API_FOOTBALL_BASE_URL, API_FOOTBALL_KEY, LEAGUES

logger = logging.getLogger(__name__)


class ApiFootballClient:
    """Client for API-Football (api-sports.io) v3."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or API_FOOTBALL_KEY
        self.base_url = API_FOOTBALL_BASE_URL
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                "x-apisports-key": self.api_key,
                "x-rapidapi-key": self.api_key,
            })
        self._request_count = 0
        self._day_start = date.today()

    def _check_daily_limit(self):
        if date.today() != self._day_start:
            self._request_count = 0
            self._day_start = date.today()
        if self._request_count >= 95:
            logger.warning("Approaching daily API limit (100 req/day)")

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        self._check_daily_limit()
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            self._request_count += 1
            if resp.status_code == 200:
                data = resp.json()
                if data.get("errors") and len(data["errors"]) > 0:
                    logger.error(f"API errors: {data['errors']}")
                    return None
                return data
            else:
                logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    # ──────────────────────────────────────────
    # FIXTURES / MATCHES
    # ──────────────────────────────────────────
    def get_todays_fixtures(self, league_id: Optional[int] = None) -> List[Dict]:
        today = date.today().isoformat()
        params = {"date": today}
        if league_id:
            params["league"] = league_id

        data = self._get("fixtures", params)
        if data and "response" in data:
            return self._normalize_fixtures(data["response"])
        return []

    def get_fixtures_by_date(self, match_date: str, league_id: Optional[int] = None) -> List[Dict]:
        params = {"date": match_date}
        if league_id:
            params["league"] = league_id
        data = self._get("fixtures", params)
        if data and "response" in data:
            return self._normalize_fixtures(data["response"])
        return []

    def get_live_fixtures(self) -> List[Dict]:
        data = self._get("fixtures", {"live": "all"})
        if data and "response" in data:
            return self._normalize_fixtures(data["response"])
        return []

    def get_league_fixtures(self, league_id: int, season: int) -> List[Dict]:
        data = self._get("fixtures", {"league": league_id, "season": season})
        if data and "response" in data:
            return self._normalize_fixtures(data["response"])
        return []

    # ──────────────────────────────────────────
    # PREDICTIONS (API built-in)
    # ──────────────────────────────────────────
    def get_prediction(self, fixture_id: int) -> Optional[Dict]:
        data = self._get("predictions", {"fixture": fixture_id})
        if data and "response" in data and len(data["response"]) > 0:
            pred = data["response"][0]
            return {
                "winner": pred.get("predictions", {}).get("winner", {}),
                "win_or_draw": pred.get("predictions", {}).get("win_or_draw"),
                "under_over": pred.get("predictions", {}).get("under_over"),
                "goals_home": pred.get("predictions", {}).get("goals", {}).get("home"),
                "goals_away": pred.get("predictions", {}).get("goals", {}).get("away"),
                "advice": pred.get("predictions", {}).get("advice"),
                "percent": pred.get("predictions", {}).get("percent", {}),
                "comparison": pred.get("comparison", {}),
                "h2h": pred.get("h2h", []),
            }
        return None

    # ──────────────────────────────────────────
    # TEAM STATISTICS
    # ──────────────────────────────────────────
    def get_team_statistics(self, team_id: int, league_id: int, season: int) -> Optional[Dict]:
        data = self._get("teams/statistics", {
            "team": team_id, "league": league_id, "season": season
        })
        if data and "response" in data:
            return data["response"]
        return None

    # ──────────────────────────────────────────
    # STANDINGS
    # ──────────────────────────────────────────
    def get_standings(self, league_id: int, season: int) -> List[Dict]:
        data = self._get("standings", {"league": league_id, "season": season})
        if data and "response" in data and len(data["response"]) > 0:
            league_data = data["response"][0].get("league", {})
            standings = league_data.get("standings", [])
            if standings:
                return standings[0]
        return []

    # ──────────────────────────────────────────
    # HEAD TO HEAD
    # ──────────────────────────────────────────
    def get_head_to_head(self, team1_id: int, team2_id: int, last: int = 10) -> List[Dict]:
        h2h_str = f"{team1_id}-{team2_id}"
        data = self._get("fixtures/headtohead", {"h2h": h2h_str, "last": last})
        if data and "response" in data:
            return self._normalize_fixtures(data["response"])
        return []

    # ──────────────────────────────────────────
    # ODDS
    # ──────────────────────────────────────────
    def get_odds(self, fixture_id: int) -> Optional[Dict]:
        data = self._get("odds", {"fixture": fixture_id})
        if data and "response" in data and len(data["response"]) > 0:
            bookmakers = data["response"][0].get("bookmakers", [])
            if bookmakers:
                bets = bookmakers[0].get("bets", [])
                for bet in bets:
                    if bet.get("name") == "Match Winner":
                        values = bet.get("values", [])
                        odds = {}
                        for v in values:
                            if v["value"] == "Home":
                                odds["home"] = float(v["odd"])
                            elif v["value"] == "Draw":
                                odds["draw"] = float(v["odd"])
                            elif v["value"] == "Away":
                                odds["away"] = float(v["odd"])
                        return odds
        return None

    # ──────────────────────────────────────────
    # NORMALIZATION
    # ──────────────────────────────────────────
    def _normalize_fixtures(self, fixtures: List[Dict]) -> List[Dict]:
        normalized = []
        for f in fixtures:
            fixture = f.get("fixture", {})
            league = f.get("league", {})
            teams = f.get("teams", {})
            goals = f.get("goals", {})
            score = f.get("score", {})

            # Map API status to standard status
            status_map = {
                "NS": "SCHEDULED", "TBD": "SCHEDULED",
                "1H": "IN_PLAY", "HT": "HALFTIME", "2H": "IN_PLAY",
                "ET": "IN_PLAY", "BT": "IN_PLAY", "P": "IN_PLAY",
                "FT": "FINISHED", "AET": "FINISHED", "PEN": "FINISHED",
                "PST": "POSTPONED", "CANC": "CANCELLED",
                "SUSP": "SUSPENDED", "INT": "SUSPENDED",
                "LIVE": "IN_PLAY",
            }
            raw_status = fixture.get("status", {}).get("short", "NS")

            # Find league code from config
            league_code = ""
            league_id = league.get("id")
            for code, info in LEAGUES.items():
                if info.get("api_id") == league_id:
                    league_code = code
                    break

            match = {
                "api_id": fixture.get("id"),
                "league_code": league_code,
                "league_name": league.get("name", ""),
                "season": league.get("season"),
                "matchday": league.get("round", ""),
                "match_date": fixture.get("date", ""),
                "status": status_map.get(raw_status, raw_status),
                "home_team_name": teams.get("home", {}).get("name", "Unknown"),
                "away_team_name": teams.get("away", {}).get("name", "Unknown"),
                "home_team_crest": teams.get("home", {}).get("logo", ""),
                "away_team_crest": teams.get("away", {}).get("logo", ""),
                "home_score": goals.get("home"),
                "away_score": goals.get("away"),
                "home_ht_score": score.get("halftime", {}).get("home"),
                "away_ht_score": score.get("halftime", {}).get("away"),
                "venue": fixture.get("venue", {}).get("name", ""),
                "referee": fixture.get("referee", ""),
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "extra_data": {
                    "home_team_id": teams.get("home", {}).get("id"),
                    "away_team_id": teams.get("away", {}).get("id"),
                    "league_id": league_id,
                    "elapsed": fixture.get("status", {}).get("elapsed"),
                },
            }
            normalized.append(match)
        return normalized

    def is_available(self) -> bool:
        """Check if API is reachable."""
        data = self._get("status")
        return data is not None
