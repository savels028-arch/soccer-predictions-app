"""
Free Football Data Client - NO API KEY REQUIRED
Uses multiple free, open data sources:
  1. football-data.org free tier (no key = limited but works for some endpoints)
  2. OpenLigaDB (German Bundesliga, free, no auth)
  3. ESPN hidden API (live scores, no auth)
  4. TheSportsDB (free tier, no key needed)
"""
import requests
import logging
import time
import random
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LEAGUES

logger = logging.getLogger(__name__)


class FreeFootballClient:
    """
    Aggregates data from multiple FREE football APIs.
    No API keys or registration required.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SoccerPredictionsPro/1.0",
            "Accept": "application/json",
        })
        self._last_request = 0
        self._min_interval = 0.5  # be polite but not too slow

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _safe_get(self, url: str, params: Optional[Dict] = None, timeout: int = 15) -> Optional[Any]:
        """Safe HTTP GET with error handling."""
        self._rate_limit()
        try:
            resp = self.session.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            else:
                logger.debug(f"HTTP {resp.status_code} from {url}")
                return None
        except Exception as e:
            logger.debug(f"Request failed {url}: {e}")
            return None

    # ═══════════════════════════════════════════════════
    #  SOURCE 1: TheSportsDB (free, no key needed)
    #  https://www.thesportsdb.com/api.php
    # ═══════════════════════════════════════════════════
    SPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"

    # League IDs in TheSportsDB
    SPORTSDB_LEAGUES = {
        "PL":  4328,  # English Premier League
        "PD":  4335,  # Spanish La Liga
        "BL1": 4331,  # German Bundesliga
        "SA":  4332,  # Italian Serie A
        "FL1": 4334,  # French Ligue 1
        "DED": 4337,  # Dutch Eredivisie
        "PPL": 4344,  # Portuguese Primeira Liga
    }

    def _sportsdb_get_todays_matches(self) -> List[Dict]:
        """Get today's matches from TheSportsDB."""
        matches = []
        # TheSportsDB: events on a specific day by league
        today_str = date.today().strftime("%Y-%m-%d")

        for league_code, sdb_id in self.SPORTSDB_LEAGUES.items():
            data = self._safe_get(
                f"{self.SPORTSDB_BASE}/eventsday.php",
                params={"d": today_str, "l": sdb_id}
            )
            if data and data.get("events"):
                for ev in data["events"]:
                    match = self._normalize_sportsdb_event(ev, league_code)
                    if match:
                        matches.append(match)

        return matches

    def _sportsdb_get_upcoming(self, league_code: str) -> List[Dict]:
        """Get next 15 upcoming events for a league from TheSportsDB."""
        sdb_id = self.SPORTSDB_LEAGUES.get(league_code)
        if not sdb_id:
            return []

        data = self._safe_get(
            f"{self.SPORTSDB_BASE}/eventsnextleague.php",
            params={"id": sdb_id}
        )
        matches = []
        if data and data.get("events"):
            for ev in data["events"]:
                match = self._normalize_sportsdb_event(ev, league_code)
                if match:
                    matches.append(match)
        return matches

    def _sportsdb_get_last_results(self, league_code: str) -> List[Dict]:
        """Get last 15 results for a league from TheSportsDB."""
        sdb_id = self.SPORTSDB_LEAGUES.get(league_code)
        if not sdb_id:
            return []

        data = self._safe_get(
            f"{self.SPORTSDB_BASE}/eventspastleague.php",
            params={"id": sdb_id}
        )
        matches = []
        if data and data.get("events"):
            for ev in data["events"]:
                match = self._normalize_sportsdb_event(ev, league_code)
                if match:
                    matches.append(match)
        return matches

    def _sportsdb_get_season_events(self, league_code: str, season: str) -> List[Dict]:
        """Get all events for a league+season from TheSportsDB."""
        sdb_id = self.SPORTSDB_LEAGUES.get(league_code)
        if not sdb_id:
            return []

        # TheSportsDB uses season format like "2024-2025"
        season_str = f"{season}-{season+1}" if isinstance(season, int) else str(season)

        data = self._safe_get(
            f"{self.SPORTSDB_BASE}/eventsseason.php",
            params={"id": sdb_id, "s": season_str}
        )
        matches = []
        if data and data.get("events"):
            for ev in data["events"]:
                match = self._normalize_sportsdb_event(ev, league_code)
                if match:
                    matches.append(match)
        return matches

    def _sportsdb_get_livescores(self) -> List[Dict]:
        """Get live scores from TheSportsDB (free tier)."""
        # The v2 livescores endpoint - try soccer sport ID 100
        data = self._safe_get(
            f"{self.SPORTSDB_BASE}/livescore.php",
            params={"s": "Soccer"}
        )
        matches = []
        if data and data.get("events"):
            for ev in data["events"]:
                match = self._normalize_sportsdb_livescore(ev)
                if match:
                    matches.append(match)
        return matches

    def _normalize_sportsdb_event(self, ev: Dict, league_code: str = "") -> Optional[Dict]:
        """Normalize TheSportsDB event to our standard format."""
        try:
            home_score = None
            away_score = None
            status = "SCHEDULED"

            if ev.get("intHomeScore") is not None and ev.get("intHomeScore") != "":
                try:
                    home_score = int(ev["intHomeScore"])
                    away_score = int(ev.get("intAwayScore", 0))
                    status = "FINISHED"
                except (ValueError, TypeError):
                    pass

            # Parse date
            match_date = ev.get("dateEvent", "")
            match_time = ev.get("strTime", "00:00:00")
            if match_date:
                try:
                    dt = datetime.strptime(f"{match_date} {match_time[:5]}", "%Y-%m-%d %H:%M")
                    match_date = dt.isoformat()
                except:
                    match_date = f"{match_date}T{match_time}"

            league_info = LEAGUES.get(league_code, {})

            return {
                "api_id": int(ev.get("idEvent", 0)),
                "league_code": league_code,
                "league_name": ev.get("strLeague", league_info.get("name", "")),
                "season": ev.get("strSeason", ""),
                "matchday": ev.get("intRound", ""),
                "match_date": match_date,
                "status": status,
                "home_team_name": ev.get("strHomeTeam", "Unknown"),
                "away_team_name": ev.get("strAwayTeam", "Unknown"),
                "home_team_crest": ev.get("strHomeTeamBadge", ""),
                "away_team_crest": ev.get("strAwayTeamBadge", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_ht_score": None,
                "away_ht_score": None,
                "venue": ev.get("strVenue", ""),
                "referee": ev.get("strReferee", ""),
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "extra_data": {
                    "source": "thesportsdb",
                    "sportsdb_id": ev.get("idEvent"),
                },
            }
        except Exception as e:
            logger.debug(f"Error normalizing SportsDB event: {e}")
            return None

    def _normalize_sportsdb_livescore(self, ev: Dict) -> Optional[Dict]:
        """Normalize TheSportsDB livescore event."""
        try:
            home_score = None
            away_score = None
            status = "IN_PLAY"

            if ev.get("intHomeScore") is not None:
                try:
                    home_score = int(ev["intHomeScore"])
                    away_score = int(ev.get("intAwayScore", 0))
                except (ValueError, TypeError):
                    pass

            progress = ev.get("strProgress", "")
            if progress and "FT" in progress.upper():
                status = "FINISHED"
            elif progress and "HT" in progress.upper():
                status = "HALFTIME"

            # Detect league code
            league_name = ev.get("strLeague", "")
            league_code = self._detect_league_code(league_name)

            return {
                "api_id": int(ev.get("idEvent", random.randint(500000, 599999))),
                "league_code": league_code,
                "league_name": league_name,
                "season": ev.get("strSeason", "2025"),
                "matchday": "",
                "match_date": datetime.now().isoformat(),
                "status": status,
                "home_team_name": ev.get("strHomeTeam", "Unknown"),
                "away_team_name": ev.get("strAwayTeam", "Unknown"),
                "home_team_crest": "",
                "away_team_crest": "",
                "home_score": home_score,
                "away_score": away_score,
                "home_ht_score": None,
                "away_ht_score": None,
                "venue": "",
                "referee": "",
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "extra_data": {
                    "source": "thesportsdb_live",
                    "elapsed": ev.get("strProgress", ""),
                },
            }
        except Exception as e:
            logger.debug(f"Error normalizing livescore: {e}")
            return None

    # ═══════════════════════════════════════════════════
    #  SOURCE 2: ESPN hidden API (no auth needed)
    # ═══════════════════════════════════════════════════
    ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

    # ESPN league slugs
    ESPN_LEAGUES = {
        "PL":  "eng.1",
        "PD":  "esp.1",
        "BL1": "ger.1",
        "SA":  "ita.1",
        "FL1": "fra.1",
        "DED": "ned.1",
        "CL":  "uefa.champions",
        "EL":  "uefa.europa",
    }

    def _espn_get_scoreboard(self, league_code: str, match_date: Optional[str] = None) -> List[Dict]:
        """Get scores from ESPN for a specific league."""
        slug = self.ESPN_LEAGUES.get(league_code)
        if not slug:
            return []

        url = f"{self.ESPN_BASE}/{slug}/scoreboard"
        params = {}
        if match_date:
            # ESPN wants YYYYMMDD format
            params["dates"] = match_date.replace("-", "")

        data = self._safe_get(url, params)
        matches = []
        if data and "events" in data:
            for ev in data["events"]:
                match = self._normalize_espn_event(ev, league_code)
                if match:
                    matches.append(match)
        return matches

    def _espn_get_all_today(self) -> List[Dict]:
        """Get today's matches from ESPN across top leagues."""
        all_matches = []
        # Only fetch top 6 leagues to avoid being too slow
        top_leagues = ["PL", "PD", "BL1", "SA", "FL1", "DED"]
        for league_code in top_leagues:
            try:
                matches = self._espn_get_scoreboard(league_code)
                all_matches.extend(matches)
            except Exception as e:
                logger.debug(f"ESPN error for {league_code}: {e}")
        return all_matches

    def _normalize_espn_event(self, ev: Dict, league_code: str) -> Optional[Dict]:
        """Normalize ESPN event to our standard format."""
        try:
            competitions = ev.get("competitions", [])
            if not competitions:
                return None

            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None

            # ESPN: first competitor is usually home
            home = None
            away = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home = c
                elif c.get("homeAway") == "away":
                    away = c

            if not home or not away:
                home, away = competitors[0], competitors[1]

            # Status
            status_data = comp.get("status", {}).get("type", {})
            espn_state = status_data.get("state", "pre")
            status_map = {
                "pre": "SCHEDULED",
                "in": "IN_PLAY",
                "post": "FINISHED",
            }
            status = status_map.get(espn_state, "SCHEDULED")

            home_score = None
            away_score = None
            if status in ("IN_PLAY", "FINISHED"):
                try:
                    home_score = int(home.get("score", 0))
                    away_score = int(away.get("score", 0))
                except (ValueError, TypeError):
                    pass

            league_info = LEAGUES.get(league_code, {})

            # Parse date
            match_date = ev.get("date", "")

            return {
                "api_id": int(ev.get("id", random.randint(600000, 699999))),
                "league_code": league_code,
                "league_name": league_info.get("name", ev.get("name", "")),
                "season": 2025,
                "matchday": "",
                "match_date": match_date,
                "status": status,
                "home_team_name": home.get("team", {}).get("displayName", "Unknown"),
                "away_team_name": away.get("team", {}).get("displayName", "Unknown"),
                "home_team_crest": home.get("team", {}).get("logo", ""),
                "away_team_crest": away.get("team", {}).get("logo", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_ht_score": None,
                "away_ht_score": None,
                "venue": comp.get("venue", {}).get("fullName", ""),
                "referee": "",
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "extra_data": {
                    "source": "espn",
                    "espn_id": ev.get("id"),
                    "elapsed": comp.get("status", {}).get("displayClock", ""),
                },
            }
        except Exception as e:
            logger.debug(f"Error normalizing ESPN event: {e}")
            return None

    # ═══════════════════════════════════════════════════
    #  SOURCE 3: OpenLigaDB (Bundesliga, free, no auth)
    #  https://api.openligadb.de
    # ═══════════════════════════════════════════════════
    OPENLIGA_BASE = "https://api.openligadb.de"

    def _openliga_get_matches(self, season: int = 2024) -> List[Dict]:
        """Get Bundesliga matches from OpenLigaDB."""
        data = self._safe_get(
            f"{self.OPENLIGA_BASE}/getmatchdata/bl1/{season}"
        )
        matches = []
        if data and isinstance(data, list):
            for m in data:
                match = self._normalize_openliga_match(m)
                if match:
                    matches.append(match)
        return matches

    def _openliga_get_current_matchday(self) -> List[Dict]:
        """Get current Bundesliga matchday."""
        data = self._safe_get(
            f"{self.OPENLIGA_BASE}/getmatchdata/bl1"
        )
        matches = []
        if data and isinstance(data, list):
            for m in data:
                match = self._normalize_openliga_match(m)
                if match:
                    matches.append(match)
        return matches

    def _normalize_openliga_match(self, m: Dict) -> Optional[Dict]:
        """Normalize OpenLigaDB match."""
        try:
            home_team = m.get("team1", {})
            away_team = m.get("team2", {})

            # Get final result
            home_score = None
            away_score = None
            status = "SCHEDULED"

            results = m.get("matchResults", [])
            for r in results:
                if r.get("resultTypeID") == 2:  # End result
                    home_score = r.get("pointsTeam1")
                    away_score = r.get("pointsTeam2")
                    status = "FINISHED"
                    break

            if m.get("matchIsFinished"):
                status = "FINISHED"

            match_date = m.get("matchDateTimeUTC", m.get("matchDateTime", ""))

            return {
                "api_id": m.get("matchID", random.randint(400000, 499999)),
                "league_code": "BL1",
                "league_name": "Bundesliga",
                "season": 2024,
                "matchday": m.get("group", {}).get("groupOrderID", ""),
                "match_date": match_date,
                "status": status,
                "home_team_name": home_team.get("teamName", "Unknown"),
                "away_team_name": away_team.get("teamName", "Unknown"),
                "home_team_crest": home_team.get("teamIconUrl", ""),
                "away_team_crest": away_team.get("teamIconUrl", ""),
                "home_score": home_score,
                "away_score": away_score,
                "home_ht_score": None,
                "away_ht_score": None,
                "venue": m.get("location", {}).get("locationStadium", "") if m.get("location") else "",
                "referee": "",
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "extra_data": {
                    "source": "openligadb",
                },
            }
        except Exception as e:
            logger.debug(f"Error normalizing OpenLigaDB match: {e}")
            return None

    # ═══════════════════════════════════════════════════
    #  UNIFIED PUBLIC METHODS
    # ═══════════════════════════════════════════════════
    def _detect_league_code(self, league_name: str) -> str:
        """Try to detect league code from league name."""
        name_lower = league_name.lower() if league_name else ""
        mapping = {
            "premier league": "PL", "premier": "PL",
            "la liga": "PD", "laliga": "PD", "primera": "PD",
            "bundesliga": "BL1",
            "serie a": "SA", "seria a": "SA",
            "ligue 1": "FL1",
            "eredivisie": "DED",
            "champions": "CL",
            "europa league": "EL",
        }
        for key, code in mapping.items():
            if key in name_lower:
                return code
        return ""

    def get_todays_matches(self) -> List[Dict]:
        """Get today's matches from all free sources."""
        all_matches = []
        seen_keys = set()

        # 1. Try ESPN (most reliable, real-time)
        logger.info("Fetching from ESPN...")
        try:
            espn = self._espn_get_all_today()
            for m in espn:
                key = f"{m['home_team_name']}_{m['away_team_name']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_matches.append(m)
            logger.info(f"ESPN: {len(espn)} matches")
        except Exception as e:
            logger.error(f"ESPN failed: {e}")

        # 2. Try TheSportsDB
        logger.info("Fetching from TheSportsDB...")
        try:
            sdb = self._sportsdb_get_todays_matches()
            for m in sdb:
                key = f"{m['home_team_name']}_{m['away_team_name']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_matches.append(m)
            logger.info(f"TheSportsDB: {len(sdb)} matches")
        except Exception as e:
            logger.error(f"TheSportsDB failed: {e}")

        logger.info(f"Total today's matches: {len(all_matches)}")
        return all_matches

    def get_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Get upcoming matches from free sources."""
        all_matches = []
        seen_keys = set()

        # ESPN: get each day
        for d in range(1, min(days + 1, 8)):
            target_date = (date.today() + timedelta(days=d)).strftime("%Y-%m-%d")
            for league_code in list(self.ESPN_LEAGUES.keys())[:6]:
                try:
                    matches = self._espn_get_scoreboard(league_code, target_date)
                    for m in matches:
                        key = f"{m['home_team_name']}_{m['away_team_name']}_{m.get('match_date', '')[:10]}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            all_matches.append(m)
                except Exception:
                    pass

        # TheSportsDB upcoming
        for league_code in list(self.SPORTSDB_LEAGUES.keys())[:5]:
            try:
                matches = self._sportsdb_get_upcoming(league_code)
                for m in matches:
                    key = f"{m['home_team_name']}_{m['away_team_name']}_{m.get('match_date', '')[:10]}"
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_matches.append(m)
            except Exception:
                pass

        return all_matches

    def get_live_matches(self) -> List[Dict]:
        """Get live matches from free sources."""
        all_matches = []
        seen_keys = set()

        # ESPN live (check all leagues for in-progress games)
        for league_code in self.ESPN_LEAGUES:
            try:
                matches = self._espn_get_scoreboard(league_code)
                for m in matches:
                    if m.get("status") == "IN_PLAY":
                        key = f"{m['home_team_name']}_{m['away_team_name']}"
                        if key not in seen_keys:
                            seen_keys.add(key)
                            all_matches.append(m)
            except Exception:
                pass

        # TheSportsDB live
        try:
            sdb_live = self._sportsdb_get_livescores()
            for m in sdb_live:
                key = f"{m['home_team_name']}_{m['away_team_name']}"
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_matches.append(m)
        except Exception:
            pass

        return all_matches

    def get_historical_matches(self, league_code: str, season: int = 2024) -> List[Dict]:
        """Get historical matches for ML training."""
        matches = []

        # TheSportsDB season data
        try:
            sdb_matches = self._sportsdb_get_season_events(league_code, season)
            finished = [m for m in sdb_matches if m.get("status") == "FINISHED"]
            matches.extend(finished)
            logger.info(f"TheSportsDB historical for {league_code}/{season}: {len(finished)} matches")
        except Exception as e:
            logger.error(f"TheSportsDB historical error: {e}")

        # Also get recent results
        try:
            recent = self._sportsdb_get_last_results(league_code)
            for m in recent:
                if m.get("status") == "FINISHED":
                    # Avoid duplicates
                    existing_ids = {x.get("api_id") for x in matches}
                    if m.get("api_id") not in existing_ids:
                        matches.append(m)
        except Exception:
            pass

        # OpenLigaDB for Bundesliga
        if league_code == "BL1":
            try:
                ol_matches = self._openliga_get_matches(season)
                existing_keys = {f"{x['home_team_name']}_{x['away_team_name']}" for x in matches}
                for m in ol_matches:
                    key = f"{m['home_team_name']}_{m['away_team_name']}"
                    if key not in existing_keys:
                        matches.append(m)
            except Exception:
                pass

        logger.info(f"Total historical for {league_code}: {len(matches)} matches")
        return matches

    def is_available(self) -> bool:
        """Check if free APIs are reachable."""
        # Quick check: try ESPN
        try:
            resp = self.session.get(
                f"{self.ESPN_BASE}/eng.1/scoreboard",
                timeout=5
            )
            return resp.status_code == 200
        except:
            return False
