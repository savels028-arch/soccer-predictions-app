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
    # LINEUPS / INJURIES / PLAYER CONTEXT
    # ──────────────────────────────────────────
    def get_fixture_lineups(self, fixture_id: int) -> List[Dict]:
        data = self._get("fixtures/lineups", {"fixture": fixture_id})
        if data and "response" in data:
            return data["response"] or []
        return []

    def get_fixture_injuries(self, fixture_id: int) -> List[Dict]:
        data = self._get("injuries", {"fixture": fixture_id})
        if data and "response" in data:
            return data["response"] or []
        return []

    def get_fixture_player_stats(self, fixture_id: int) -> List[Dict]:
        data = self._get("fixtures/players", {"fixture": fixture_id})
        if data and "response" in data:
            return data["response"] or []
        return []

    def get_fixture_statistics(self, fixture_id: int) -> List[Dict]:
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        if data and "response" in data:
            return data["response"] or []
        return []

    @staticmethod
    def _team_lineup_count(lineups: List[Dict], team_id: Optional[int]) -> int:
        for row in lineups:
            team = row.get("team", {})
            if team_id and team.get("id") != team_id:
                continue
            start_xi = row.get("startXI") or []
            if start_xi:
                return len(start_xi)
        return 0

    @staticmethod
    def _team_missing_count(injuries: List[Dict], team_id: Optional[int]) -> int:
        count = 0
        for row in injuries:
            team = row.get("team", {})
            if team_id and team.get("id") != team_id:
                continue
            count += 1
        return count

    @staticmethod
    def _team_rating_avg(player_stats: List[Dict], team_id: Optional[int]) -> Optional[float]:
        ratings = []
        for team_row in player_stats:
            team = team_row.get("team", {})
            if team_id and team.get("id") != team_id:
                continue
            for player_row in team_row.get("players", []) or []:
                for stat in player_row.get("statistics", []) or []:
                    rating = stat.get("games", {}).get("rating")
                    if rating in (None, ""):
                        continue
                    try:
                        ratings.append(float(rating))
                    except (TypeError, ValueError):
                        continue
        if not ratings:
            return None
        return round(sum(ratings) / len(ratings), 3)

    @staticmethod
    def _team_stat_value(statistics: List[Dict], team_id: Optional[int], stat_name: str) -> Optional[float]:
        wanted = stat_name.strip().lower()
        for team_row in statistics:
            team = team_row.get("team", {})
            if team_id and team.get("id") != team_id:
                continue
            for stat in team_row.get("statistics", []) or []:
                if str(stat.get("type", "")).strip().lower() != wanted:
                    continue
                value = stat.get("value")
                if value in (None, ""):
                    return None
                try:
                    return float(str(value).replace("%", ""))
                except (TypeError, ValueError):
                    return None
        return None

    def get_fixture_context(self, fixture_id: int, home_team_id: Optional[int] = None,
                            away_team_id: Optional[int] = None) -> Dict:
        """Bundle the pre-match data that can move a prediction edge."""
        lineups = self.get_fixture_lineups(fixture_id)
        injuries = self.get_fixture_injuries(fixture_id)
        players = self.get_fixture_player_stats(fixture_id)
        statistics = self.get_fixture_statistics(fixture_id)

        summary = {
            "home_missing_players": self._team_missing_count(injuries, home_team_id),
            "away_missing_players": self._team_missing_count(injuries, away_team_id),
            "home_lineup_players": self._team_lineup_count(lineups, home_team_id),
            "away_lineup_players": self._team_lineup_count(lineups, away_team_id),
            "home_player_rating_avg": self._team_rating_avg(players, home_team_id),
            "away_player_rating_avg": self._team_rating_avg(players, away_team_id),
            "home_xg": self._team_stat_value(statistics, home_team_id, "Expected Goals"),
            "away_xg": self._team_stat_value(statistics, away_team_id, "Expected Goals"),
        }
        summary["has_lineups"] = bool(summary["home_lineup_players"] or summary["away_lineup_players"])
        summary["has_injuries"] = bool(summary["home_missing_players"] or summary["away_missing_players"])
        summary["has_player_stats"] = bool(
            summary["home_player_rating_avg"] is not None
            or summary["away_player_rating_avg"] is not None
        )
        summary["has_xg"] = bool(summary["home_xg"] is not None or summary["away_xg"] is not None)

        return {
            "fixtureId": fixture_id,
            "summary": summary,
            "lineups": lineups,
            "injuries": injuries,
            "playerStats": players,
            "fixtureStatistics": statistics,
        }

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
