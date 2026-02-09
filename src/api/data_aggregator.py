"""
Data Aggregator - Combines data from multiple sources
Handles caching, deduplication, and normalization.
Uses FREE APIs (no registration needed) as primary sources.
"""
import logging
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
import random

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LEAGUES, DATA_SETTINGS
from src.api.free_football_client import FreeFootballClient
from src.api.csv_football_client import FootballDataCSVClient

logger = logging.getLogger(__name__)


class DataAggregator:
    """Aggregates data from multiple football APIs and generates demo data as fallback.
    
    Priority order:
      1. Football-Data.co.uk CSV (historical with odds+stats) - NO key needed
      2. Free APIs (ESPN, TheSportsDB, OpenLigaDB) - NO key needed  
      3. Football-Data.org (if API key provided)
      4. API-Football (if API key provided)
      5. Demo data (fallback)
    """

    def __init__(self, db_manager, fd_client=None, af_client=None):
        self.db = db_manager
        self.fd_client = fd_client
        self.af_client = af_client
        self.free_client = FreeFootballClient()
        self.csv_client = FootballDataCSVClient()

    # ──────────────────────────────────────────
    # MAIN DATA FETCH
    # ──────────────────────────────────────────
    def fetch_todays_matches(self, force_refresh: bool = False) -> List[Dict]:
        """Fetch today's matches from APIs or cache/demo."""
        cache_key = f"todays_matches_{date.today().isoformat()}"

        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached:
                logger.info("Returning cached today's matches")
                return cached

        matches = []

        # 1. Try FREE APIs first (no key needed!)
        try:
            free_matches = self.free_client.get_todays_matches()
            matches.extend(free_matches)
            logger.info(f"Got {len(free_matches)} matches from free APIs (ESPN/TheSportsDB)")
        except Exception as e:
            logger.error(f"Free API error: {e}")

        # 2. Try Football-Data.org (if key provided)
        if self.fd_client and not matches:
            try:
                league_codes = list(LEAGUES.keys())
                fd_matches = self.fd_client.get_todays_matches(league_codes[:5])
                matches.extend(fd_matches)
                logger.info(f"Got {len(fd_matches)} matches from Football-Data.org")
            except Exception as e:
                logger.error(f"Football-Data.org error: {e}")

        # 3. Try API-Football (if key provided)
        if self.af_client and not matches:
            try:
                af_matches = self.af_client.get_todays_fixtures()
                matches.extend(af_matches)
                logger.info(f"Got {len(af_matches)} matches from API-Football")
            except Exception as e:
                logger.error(f"API-Football error: {e}")

        # 4. Fallback to demo data
        if not matches:
            logger.info("No API data available, generating demo matches")
            matches = self._generate_demo_matches()

        # Store in database & cache
        for m in matches:
            try:
                self.db.upsert_match(m)
            except Exception as e:
                logger.error(f"Error storing match: {e}")

        self.db.set_cache(cache_key, matches, DATA_SETTINGS["cache_ttl_minutes"])
        return matches

    def fetch_upcoming_matches(self, days: int = 7) -> List[Dict]:
        """Fetch upcoming matches for the next N days."""
        cache_key = f"upcoming_{days}d_{date.today().isoformat()}"
        cached = self.db.get_cache(cache_key)
        if cached:
            return cached

        matches = []

        # 1. Free APIs first
        try:
            free_matches = self.free_client.get_upcoming_matches(days)
            matches.extend(free_matches)
            logger.info(f"Got {len(free_matches)} upcoming from free APIs")
        except Exception as e:
            logger.error(f"Free API upcoming error: {e}")

        # 2. Football-Data.org fallback
        if not matches and self.fd_client:
            date_from = date.today().isoformat()
            date_to = (date.today() + timedelta(days=days)).isoformat()
            try:
                matches = self.fd_client.get_matches_by_date_range(date_from, date_to)
            except Exception as e:
                logger.error(f"Error fetching upcoming: {e}")

        # 3. Demo fallback
        if not matches:
            matches = self._generate_demo_upcoming(days)

        for m in matches:
            try:
                self.db.upsert_match(m)
            except Exception:
                pass

        self.db.set_cache(cache_key, matches, DATA_SETTINGS["cache_ttl_minutes"])
        return matches

    def fetch_live_matches(self) -> List[Dict]:
        """Fetch currently live matches."""
        matches = []

        # 1. Free APIs first (ESPN + TheSportsDB live)
        try:
            free_live = self.free_client.get_live_matches()
            matches.extend(free_live)
            logger.info(f"Got {len(free_live)} live matches from free APIs")
        except Exception as e:
            logger.error(f"Free API live error: {e}")

        # 2. Football-Data.org
        if not matches and self.fd_client:
            try:
                matches = self.fd_client.get_live_matches()
            except Exception as e:
                logger.error(f"Error fetching live: {e}")

        # 3. API-Football
        if not matches and self.af_client:
            try:
                matches = self.af_client.get_live_fixtures()
            except Exception as e:
                logger.error(f"Error fetching live: {e}")

        # 4. Database / demo fallback
        if not matches:
            matches = self.db.get_live_matches()
            if not matches:
                matches = self._generate_demo_live()

        for m in matches:
            try:
                self.db.upsert_match(m)
            except Exception:
                pass

        return matches

    def fetch_historical_matches(self, league_code: str, season: int = 2025) -> List[Dict]:
        """Fetch historical match data for ML training."""
        cache_key = f"historical_{league_code}_{season}"
        cached = self.db.get_cache(cache_key)
        if cached:
            return cached

        matches = []

        # 1. Football-Data.co.uk CSV (best source: full stats + odds!)
        try:
            csv_matches = self.csv_client.get_season_matches(league_code, season)
            if csv_matches:
                matches.extend(csv_matches)
                logger.info(f"Got {len(csv_matches)} historical from CSV for {league_code}/{season}")
        except Exception as e:
            logger.error(f"CSV historical error: {e}")

        # 2. Also try previous seasons for more training data
        if len(matches) < 100:
            for prev_season in [season - 1, season - 2]:
                try:
                    prev = self.csv_client.get_season_matches(league_code, prev_season)
                    if prev:
                        matches.extend(prev)
                        logger.info(f"Added {len(prev)} matches from {league_code}/{prev_season}")
                except Exception:
                    pass

        # 3. Free APIs (TheSportsDB)
        if not matches:
            try:
                free_hist = self.free_client.get_historical_matches(league_code, season)
                matches.extend(free_hist)
                logger.info(f"Got {len(free_hist)} historical from free APIs for {league_code}/{season}")
            except Exception as e:
                logger.error(f"Free API historical error: {e}")

        # 4. Football-Data.org (if key provided)
        if not matches and self.fd_client:
            try:
                fd_hist = self.fd_client.get_league_matches(league_code, season)
                matches.extend(fd_hist)
            except Exception as e:
                logger.error(f"Error fetching historical: {e}")

        # 5. Check database
        if not matches:
            matches = self.db.get_finished_matches(league_code, season)

        # 6. Generate demo data if nothing else works
        if not matches:
            matches = self._generate_demo_historical(league_code, season)

        self.db.set_cache(cache_key, matches, 60)
        return matches

    # ──────────────────────────────────────────
    # TEAM DATA
    # ──────────────────────────────────────────
    def get_team_form(self, team_name: str, n: int = 5) -> str:
        """Get last N results as string e.g. 'WWDLW'."""
        matches = self.db.get_team_matches(team_name, limit=n)
        form = []
        for m in matches:
            if m["home_score"] is None or m["away_score"] is None:
                continue
            is_home = m["home_team_name"] == team_name
            gs = m["home_score"] if is_home else m["away_score"]
            gc = m["away_score"] if is_home else m["home_score"]
            if gs > gc:
                form.append("W")
            elif gs == gc:
                form.append("D")
            else:
                form.append("L")
        return "".join(form[:n]) if form else "-----"

    # ──────────────────────────────────────────
    # DEMO DATA GENERATORS
    # ──────────────────────────────────────────
    def _generate_demo_matches(self) -> List[Dict]:
        """Generate realistic demo match data for today."""
        today = datetime.now()
        demo_matches = []

        league_fixtures = {
            "PL": [
                ("Arsenal FC", "Chelsea FC"),
                ("Manchester City FC", "Liverpool FC"),
                ("Manchester United FC", "Tottenham Hotspur FC"),
                ("Newcastle United FC", "Aston Villa FC"),
            ],
            "PD": [
                ("FC Barcelona", "Real Madrid CF"),
                ("Atlético de Madrid", "Real Sociedad"),
                ("Sevilla FC", "Valencia CF"),
            ],
            "BL1": [
                ("FC Bayern München", "Borussia Dortmund"),
                ("RB Leipzig", "Bayer 04 Leverkusen"),
                ("VfB Stuttgart", "Eintracht Frankfurt"),
            ],
            "SA": [
                ("SSC Napoli", "Juventus FC"),
                ("AC Milan", "Inter Milano"),
                ("AS Roma", "SS Lazio"),
            ],
            "FL1": [
                ("Paris Saint-Germain FC", "Olympique de Marseille"),
                ("AS Monaco FC", "Olympique Lyonnais"),
            ],
        }

        statuses = ["SCHEDULED", "SCHEDULED", "SCHEDULED", "IN_PLAY", "FINISHED"]
        match_id = 900000

        for league_code, fixtures in league_fixtures.items():
            league_info = LEAGUES.get(league_code, {})
            for i, (home, away) in enumerate(fixtures):
                hour = 13 + i * 2 + random.randint(0, 1)
                minute = random.choice([0, 15, 30, 45])
                match_time = today.replace(hour=min(hour, 22), minute=minute, second=0)
                status = random.choice(statuses)

                home_score = None
                away_score = None
                if status == "FINISHED":
                    home_score = random.randint(0, 4)
                    away_score = random.randint(0, 3)
                elif status == "IN_PLAY":
                    home_score = random.randint(0, 3)
                    away_score = random.randint(0, 2)

                home_odds = round(random.uniform(1.3, 4.5), 2)
                away_odds = round(random.uniform(1.5, 5.0), 2)
                draw_odds = round(random.uniform(3.0, 4.0), 2)

                match_id += 1
                demo_matches.append({
                    "api_id": match_id,
                    "league_code": league_code,
                    "league_name": league_info.get("name", ""),
                    "season": 2025,
                    "matchday": random.randint(20, 30),
                    "match_date": match_time.isoformat(),
                    "status": status,
                    "home_team_name": home,
                    "away_team_name": away,
                    "home_team_crest": "",
                    "away_team_crest": "",
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_ht_score": None,
                    "away_ht_score": None,
                    "venue": f"{home} Stadium",
                    "referee": "",
                    "home_odds": home_odds,
                    "draw_odds": draw_odds,
                    "away_odds": away_odds,
                    "extra_data": {},
                })

        return demo_matches

    def _generate_demo_upcoming(self, days: int) -> List[Dict]:
        """Generate demo upcoming matches."""
        all_matches = []
        for d in range(1, min(days + 1, 8)):
            match_date = datetime.now() + timedelta(days=d)
            teams_pool = [
                ("PL", "Arsenal FC", "Manchester United FC"),
                ("PL", "Chelsea FC", "Liverpool FC"),
                ("PL", "Tottenham Hotspur FC", "Newcastle United FC"),
                ("PD", "FC Barcelona", "Atlético de Madrid"),
                ("PD", "Real Madrid CF", "Sevilla FC"),
                ("BL1", "FC Bayern München", "RB Leipzig"),
                ("SA", "Inter Milano", "AC Milan"),
                ("FL1", "Paris Saint-Germain FC", "AS Monaco FC"),
            ]

            selected = random.sample(teams_pool, min(3, len(teams_pool)))
            for league_code, home, away in selected:
                league_info = LEAGUES.get(league_code, {})
                hour = random.choice([13, 15, 17, 19, 20, 21])
                match_time = match_date.replace(hour=hour, minute=0, second=0)
                all_matches.append({
                    "api_id": random.randint(800000, 899999),
                    "league_code": league_code,
                    "league_name": league_info.get("name", ""),
                    "season": 2025,
                    "matchday": random.randint(20, 30),
                    "match_date": match_time.isoformat(),
                    "status": "SCHEDULED",
                    "home_team_name": home,
                    "away_team_name": away,
                    "home_score": None,
                    "away_score": None,
                    "home_odds": round(random.uniform(1.4, 4.0), 2),
                    "draw_odds": round(random.uniform(3.0, 4.0), 2),
                    "away_odds": round(random.uniform(1.8, 5.0), 2),
                    "extra_data": {},
                })
        return all_matches

    def _generate_demo_live(self) -> List[Dict]:
        """Generate a few demo live matches."""
        live = []
        live_fixtures = [
            ("PL", "Arsenal FC", "Chelsea FC"),
            ("PD", "FC Barcelona", "Real Madrid CF"),
            ("BL1", "FC Bayern München", "Borussia Dortmund"),
        ]
        for league_code, home, away in live_fixtures:
            league_info = LEAGUES.get(league_code, {})
            elapsed = random.randint(15, 85)
            live.append({
                "api_id": random.randint(700000, 799999),
                "league_code": league_code,
                "league_name": league_info.get("name", ""),
                "season": 2025,
                "matchday": 25,
                "match_date": datetime.now().isoformat(),
                "status": "IN_PLAY",
                "home_team_name": home,
                "away_team_name": away,
                "home_score": random.randint(0, 3),
                "away_score": random.randint(0, 2),
                "home_odds": None,
                "draw_odds": None,
                "away_odds": None,
                "extra_data": {"elapsed": elapsed},
            })
        return live

    def _generate_demo_historical(self, league_code: str, season: int) -> List[Dict]:
        """Generate demo historical data for ML training."""
        teams_by_league = {
            "PL": ["Arsenal FC", "Chelsea FC", "Manchester City FC", "Liverpool FC",
                    "Manchester United FC", "Tottenham Hotspur FC", "Newcastle United FC",
                    "Aston Villa FC", "Brighton & Hove Albion FC", "West Ham United FC",
                    "Crystal Palace FC", "Brentford FC", "Fulham FC", "Wolverhampton Wanderers FC",
                    "AFC Bournemouth", "Nottingham Forest FC", "Everton FC", "Leicester City FC",
                    "Ipswich Town FC", "Southampton FC"],
            "PD": ["FC Barcelona", "Real Madrid CF", "Atlético de Madrid", "Real Sociedad",
                    "Athletic Club", "Real Betis", "Villarreal CF", "Valencia CF",
                    "Sevilla FC", "Getafe CF", "Girona FC", "RC Celta de Vigo",
                    "RCD Mallorca", "UD Las Palmas", "CA Osasuna", "Deportivo Alavés",
                    "Rayo Vallecano", "RCD Espanyol", "Real Valladolid CF", "CD Leganés"],
            "BL1": ["FC Bayern München", "Borussia Dortmund", "RB Leipzig",
                     "Bayer 04 Leverkusen", "VfB Stuttgart", "Eintracht Frankfurt",
                     "VfL Wolfsburg", "SC Freiburg", "TSG 1899 Hoffenheim",
                     "1. FC Union Berlin", "Borussia Mönchengladbach", "1. FSV Mainz 05",
                     "FC Augsburg", "SV Werder Bremen", "VfL Bochum 1848",
                     "1. FC Heidenheim 1846", "FC St. Pauli", "Holstein Kiel"],
            "SA": ["SSC Napoli", "Juventus FC", "AC Milan", "Inter Milano",
                    "AS Roma", "SS Lazio", "Atalanta BC", "ACF Fiorentina",
                    "Bologna FC 1909", "Torino FC", "Udinese Calcio", "US Sassuolo",
                    "Empoli FC", "Cagliari Calcio", "Hellas Verona FC",
                    "Genoa CFC", "US Lecce", "Frosinone Calcio",
                    "US Salernitana 1919", "Monza"],
            "FL1": ["Paris Saint-Germain FC", "Olympique de Marseille", "AS Monaco FC",
                     "Olympique Lyonnais", "LOSC Lille", "OGC Nice", "RC Lens",
                     "Stade Rennais FC 1901", "RC Strasbourg Alsace", "Stade Brestois 29",
                     "FC Nantes", "Toulouse FC", "Montpellier HSC", "Stade de Reims",
                     "Le Havre AC", "FC Metz", "Clermont Foot 63", "FC Lorient"],
        }

        teams = teams_by_league.get(league_code, teams_by_league["PL"])
        matches = []
        match_id = 100000 + hash(league_code) % 100000

        # Generate round-robin schedule
        for matchday in range(1, 39):
            for i in range(0, len(teams) - 1, 2):
                if i + 1 < len(teams):
                    home = teams[i]
                    away = teams[(i + matchday) % len(teams)]
                    if home == away:
                        continue

                    days_back = (38 - matchday) * 7 + random.randint(-2, 2)
                    match_date = datetime.now() - timedelta(days=max(days_back, 1))

                    # Simulate realistic scores
                    home_strength = random.uniform(0.3, 0.7)
                    home_score = int(random.gauss(1.5 * home_strength + 0.5, 0.8))
                    away_score = int(random.gauss(1.2 * (1 - home_strength) + 0.4, 0.7))
                    home_score = max(0, min(home_score, 6))
                    away_score = max(0, min(away_score, 5))

                    match_id += 1
                    matches.append({
                        "api_id": match_id,
                        "league_code": league_code,
                        "league_name": LEAGUES.get(league_code, {}).get("name", ""),
                        "season": season,
                        "matchday": matchday,
                        "match_date": match_date.isoformat(),
                        "status": "FINISHED",
                        "home_team_name": home,
                        "away_team_name": away,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_odds": round(random.uniform(1.3, 4.5), 2),
                        "draw_odds": round(random.uniform(3.0, 4.2), 2),
                        "away_odds": round(random.uniform(1.5, 5.0), 2),
                        "extra_data": {},
                    })

        # Store in db
        for m in matches:
            try:
                self.db.upsert_match(m)
            except Exception:
                pass

        return matches
