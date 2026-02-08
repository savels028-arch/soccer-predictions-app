"""
Football-Data.org API Client
Free tier: 10 requests/minute
Docs: https://www.football-data.org/documentation/api
"""
import requests
import logging
import time
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import FOOTBALL_DATA_BASE_URL, FOOTBALL_DATA_API_KEY, LEAGUES

logger = logging.getLogger(__name__)


class FootballDataClient:
    """Client for football-data.org API (v4)."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or FOOTBALL_DATA_API_KEY
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-Auth-Token": self.api_key})
        self._last_request_time = 0
        self._min_interval = 6.5  # Free tier: ~10 req/min

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {wait:.1f}s")
            time.sleep(wait)
        self._last_request_time = time.time()

    def _get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        self._rate_limit()
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                logger.warning("Rate limit hit, waiting 60s...")
                time.sleep(60)
                return self._get(endpoint, params)
            else:
                logger.error(f"API error {resp.status_code}: {resp.text[:200]}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    # ──────────────────────────────────────────
    # MATCHES
    # ──────────────────────────────────────────
    def get_todays_matches(self, league_codes: Optional[List[str]] = None) -> List[Dict]:
        """Get all matches for today."""
        today = date.today().isoformat()
        params = {"dateFrom": today, "dateTo": today}
        if league_codes:
            params["competitions"] = ",".join(league_codes)

        data = self._get("matches", params)
        if data and "matches" in data:
            return self._normalize_matches(data["matches"])
        return []

    def get_matches_by_date_range(self, date_from: str, date_to: str,
                                   league_codes: Optional[List[str]] = None) -> List[Dict]:
        """Get matches in a date range."""
        params = {"dateFrom": date_from, "dateTo": date_to}
        if league_codes:
            params["competitions"] = ",".join(league_codes)

        data = self._get("matches", params)
        if data and "matches" in data:
            return self._normalize_matches(data["matches"])
        return []

    def get_league_matches(self, league_code: str, season: Optional[int] = None,
                            matchday: Optional[int] = None) -> List[Dict]:
        """Get matches for a specific league."""
        endpoint = f"competitions/{league_code}/matches"
        params = {}
        if season:
            params["season"] = season
        if matchday:
            params["matchday"] = matchday

        data = self._get(endpoint, params)
        if data and "matches" in data:
            return self._normalize_matches(data["matches"])
        return []

    def get_live_matches(self) -> List[Dict]:
        """Get currently live matches."""
        params = {"status": "LIVE,IN_PLAY,PAUSED,HALFTIME"}
        data = self._get("matches", params)
        if data and "matches" in data:
            return self._normalize_matches(data["matches"])
        return []

    # ──────────────────────────────────────────
    # STANDINGS
    # ──────────────────────────────────────────
    def get_standings(self, league_code: str, season: Optional[int] = None) -> List[Dict]:
        endpoint = f"competitions/{league_code}/standings"
        params = {}
        if season:
            params["season"] = season
        data = self._get(endpoint, params)
        if data and "standings" in data:
            for standing_type in data["standings"]:
                if standing_type.get("type") == "TOTAL":
                    return standing_type.get("table", [])
        return []

    # ──────────────────────────────────────────
    # TEAMS
    # ──────────────────────────────────────────
    def get_teams(self, league_code: str, season: Optional[int] = None) -> List[Dict]:
        endpoint = f"competitions/{league_code}/teams"
        params = {}
        if season:
            params["season"] = season
        data = self._get(endpoint, params)
        if data and "teams" in data:
            return data["teams"]
        return []

    # ──────────────────────────────────────────
    # HEAD TO HEAD
    # ──────────────────────────────────────────
    def get_head_to_head(self, match_id: int) -> Optional[Dict]:
        data = self._get(f"matches/{match_id}/head2head")
        return data

    # ──────────────────────────────────────────
    # NORMALIZATION
    # ──────────────────────────────────────────
    def _normalize_matches(self, raw_matches: List[Dict]) -> List[Dict]:
        """Normalize API response into standard format."""
        normalized = []
        for m in raw_matches:
            comp = m.get("competition", {})
            score = m.get("score", {})
            full_time = score.get("fullTime", {})
            half_time = score.get("halfTime", {})
            home_team = m.get("homeTeam", {})
            away_team = m.get("awayTeam", {})
            odds = m.get("odds", {})

            match = {
                "api_id": m.get("id"),
                "league_code": comp.get("code", ""),
                "league_name": comp.get("name", ""),
                "season": m.get("season", {}).get("startDate", "")[:4] if m.get("season") else None,
                "matchday": m.get("matchday"),
                "match_date": m.get("utcDate", ""),
                "status": m.get("status", "SCHEDULED"),
                "home_team_name": home_team.get("name", "Unknown"),
                "away_team_name": away_team.get("name", "Unknown"),
                "home_team_crest": home_team.get("crest", ""),
                "away_team_crest": away_team.get("crest", ""),
                "home_score": full_time.get("home"),
                "away_score": full_time.get("away"),
                "home_ht_score": half_time.get("home"),
                "away_ht_score": half_time.get("away"),
                "venue": m.get("venue", ""),
                "referee": "",
                "home_odds": odds.get("homeWin"),
                "draw_odds": odds.get("draw"),
                "away_odds": odds.get("awayWin"),
                "extra_data": {
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "competition_id": comp.get("id"),
                },
            }

            # Parse referee
            referees = m.get("referees", [])
            if referees:
                match["referee"] = referees[0].get("name", "")

            # Parse season year
            season_data = m.get("season", {})
            if season_data:
                start = season_data.get("startDate", "")
                if start:
                    try:
                        match["season"] = int(start[:4])
                    except (ValueError, IndexError):
                        pass

            normalized.append(match)
        return normalized

    def is_available(self) -> bool:
        """Check if API is reachable and key is valid."""
        data = self._get("competitions")
        return data is not None
