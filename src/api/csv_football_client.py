"""
Football-Data.co.uk CSV Client
FREE historical match data with full statistics & betting odds.
No API key or registration needed!

Source: https://www.football-data.co.uk/
Covers: 25+ leagues, 20+ seasons, with odds from 6+ bookmakers.
"""
import csv
import io
import logging
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LEAGUES

logger = logging.getLogger(__name__)


class FootballDataCSVClient:
    """
    Downloads and parses free CSV match data from football-data.co.uk.
    Provides historical matches with full stats and betting odds.
    NO API key or registration required.
    """

    BASE_URL = "https://www.football-data.co.uk/mmz4281"

    # Mapping: our league code -> (CSV code, country folder)
    LEAGUE_CSV_MAP = {
        "PL":  "E0",    # England Premier League
        "ELC": "E1",    # England Championship
        "PD":  "SP1",   # Spain La Liga
        "BL1": "D1",    # Germany Bundesliga
        "BL2": "D2",    # Germany 2. Bundesliga
        "SA":  "I1",    # Italy Serie A
        "FL1": "F1",    # France Ligue 1
        "DED": "N1",    # Netherlands Eredivisie
        "PPL": "P1",    # Portugal Primeira Liga
        "BSA": "B1",    # Belgium Jupiler League (not Brazil)
    }

    # Season codes: 2024/25 = "2425", 2023/24 = "2324", etc.
    AVAILABLE_SEASONS = [
        (2024, "2425"),
        (2023, "2324"),
        (2022, "2223"),
        (2021, "2122"),
        (2020, "2021"),
        (2019, "1920"),
        (2018, "1819"),
        (2017, "1718"),
        (2016, "1617"),
        (2015, "1516"),
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SoccerPredictionsPro/1.0",
        })
        self._cache = {}  # in-memory cache: (league, season) -> data
        self._last_request = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request = time.time()

    def get_season_matches(self, league_code: str, season: int) -> List[Dict]:
        """
        Get all matches for a league + season.
        Returns normalized match dicts with full stats and odds.
        
        Args:
            league_code: Our league code (PL, PD, BL1, SA, FL1, etc.)
            season: Start year of season (e.g. 2024 for 2024/25)
        """
        cache_key = (league_code, season)
        if cache_key in self._cache:
            return self._cache[cache_key]

        csv_code = self.LEAGUE_CSV_MAP.get(league_code)
        if not csv_code:
            logger.debug(f"No CSV mapping for league {league_code}")
            return []

        season_code = self._get_season_code(season)
        if not season_code:
            logger.debug(f"No season code for {season}")
            return []

        url = f"{self.BASE_URL}/{season_code}/{csv_code}.csv"
        logger.info(f"Downloading CSV: {url}")

        self._rate_limit()
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"CSV download failed: HTTP {resp.status_code} for {url}")
                return []

            matches = self._parse_csv(resp.text, league_code, season)
            self._cache[cache_key] = matches
            logger.info(f"Parsed {len(matches)} matches from {league_code} {season}/{season+1}")
            return matches

        except Exception as e:
            logger.error(f"Error downloading CSV from {url}: {e}")
            return []

    def get_multi_season_matches(self, league_code: str, num_seasons: int = 3) -> List[Dict]:
        """Get matches from multiple recent seasons for ML training."""
        all_matches = []
        for season, _ in self.AVAILABLE_SEASONS[:num_seasons]:
            matches = self.get_season_matches(league_code, season)
            all_matches.extend(matches)
        return all_matches

    def get_all_leagues_current_season(self) -> List[Dict]:
        """Get current season data for all supported leagues."""
        all_matches = []
        current_season = self.AVAILABLE_SEASONS[0][0]
        for league_code in self.LEAGUE_CSV_MAP:
            matches = self.get_season_matches(league_code, current_season)
            all_matches.extend(matches)
        return all_matches

    def _get_season_code(self, season: int) -> Optional[str]:
        """Convert season year to URL code."""
        for s, code in self.AVAILABLE_SEASONS:
            if s == season:
                return code
        # Try to generate it
        if 2010 <= season <= 2030:
            s1 = str(season)[2:]
            s2 = str(season + 1)[2:]
            return f"{s1}{s2}"
        return None

    def _parse_csv(self, csv_text: str, league_code: str, season: int) -> List[Dict]:
        """Parse CSV text into normalized match dicts."""
        matches = []
        try:
            # Handle BOM and encoding issues
            csv_text = csv_text.lstrip('\ufeff')
            reader = csv.DictReader(io.StringIO(csv_text))

            league_info = LEAGUES.get(league_code, {})

            for row in reader:
                match = self._normalize_csv_row(row, league_code, league_info, season)
                if match:
                    matches.append(match)

        except Exception as e:
            logger.error(f"CSV parse error: {e}")

        return matches

    def _normalize_csv_row(self, row: Dict, league_code: str,
                           league_info: Dict, season: int) -> Optional[Dict]:
        """Normalize a CSV row to our standard match format."""
        try:
            home = row.get("HomeTeam", "").strip()
            away = row.get("AwayTeam", "").strip()
            if not home or not away:
                return None

            # Parse scores
            home_score = self._safe_int(row.get("FTHG"))
            away_score = self._safe_int(row.get("FTAG"))
            home_ht = self._safe_int(row.get("HTHG"))
            away_ht = self._safe_int(row.get("HTAG"))

            # Parse date
            date_str = row.get("Date", "")
            time_str = row.get("Time", "15:00")
            match_date = self._parse_date(date_str, time_str)

            # Parse odds (Bet365 as primary, with fallbacks)
            home_odds = self._safe_float(row.get("B365H")) or self._safe_float(row.get("BWH")) or self._safe_float(row.get("PSH"))
            draw_odds = self._safe_float(row.get("B365D")) or self._safe_float(row.get("BWD")) or self._safe_float(row.get("PSD"))
            away_odds = self._safe_float(row.get("B365A")) or self._safe_float(row.get("BWA")) or self._safe_float(row.get("PSA"))

            # Average odds (most accurate for value betting)
            avg_home_odds = self._safe_float(row.get("AvgH"))
            avg_draw_odds = self._safe_float(row.get("AvgD"))
            avg_away_odds = self._safe_float(row.get("AvgA"))

            # Match stats
            extra_data = {
                "source": "football-data.co.uk",
                # Shots
                "home_shots": self._safe_int(row.get("HS")),
                "away_shots": self._safe_int(row.get("AS")),
                "home_shots_target": self._safe_int(row.get("HST")),
                "away_shots_target": self._safe_int(row.get("AST")),
                # Fouls
                "home_fouls": self._safe_int(row.get("HF")),
                "away_fouls": self._safe_int(row.get("AF")),
                # Corners
                "home_corners": self._safe_int(row.get("HC")),
                "away_corners": self._safe_int(row.get("AC")),
                # Cards
                "home_yellow": self._safe_int(row.get("HY")),
                "away_yellow": self._safe_int(row.get("AY")),
                "home_red": self._safe_int(row.get("HR")),
                "away_red": self._safe_int(row.get("AR")),
                # All odds
                "b365_home": self._safe_float(row.get("B365H")),
                "b365_draw": self._safe_float(row.get("B365D")),
                "b365_away": self._safe_float(row.get("B365A")),
                "avg_home_odds": avg_home_odds,
                "avg_draw_odds": avg_draw_odds,
                "avg_away_odds": avg_away_odds,
                "max_home_odds": self._safe_float(row.get("MaxH")),
                "max_draw_odds": self._safe_float(row.get("MaxD")),
                "max_away_odds": self._safe_float(row.get("MaxA")),
                # Over/under 2.5
                "b365_over25": self._safe_float(row.get("B365>2.5")),
                "b365_under25": self._safe_float(row.get("B365<2.5")),
                # Result
                "ftr": row.get("FTR", ""),  # H, D, A
                "htr": row.get("HTR", ""),  # Half-time result
            }

            # Generate a stable ID
            api_id = hash(f"{league_code}_{season}_{match_date}_{home}_{away}") % 10000000

            return {
                "api_id": api_id,
                "league_code": league_code,
                "league_name": league_info.get("name", ""),
                "season": season,
                "matchday": None,
                "match_date": match_date,
                "status": "FINISHED",
                "home_team_name": home,
                "away_team_name": away,
                "home_team_crest": "",
                "away_team_crest": "",
                "home_score": home_score,
                "away_score": away_score,
                "home_ht_score": home_ht,
                "away_ht_score": away_ht,
                "venue": "",
                "referee": row.get("Referee", ""),
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "extra_data": extra_data,
            }
        except Exception as e:
            logger.debug(f"Error normalizing CSV row: {e}")
            return None

    def _parse_date(self, date_str: str, time_str: str = "15:00") -> str:
        """Parse date from CSV (DD/MM/YYYY or DD/MM/YY)."""
        if not date_str:
            return ""
        try:
            # Try DD/MM/YYYY
            dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M")
            return dt.isoformat()
        except ValueError:
            try:
                # Try DD/MM/YY
                dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%y %H:%M")
                return dt.isoformat()
            except ValueError:
                return date_str

    def _safe_int(self, val) -> Optional[int]:
        if val is None or val == "":
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, val) -> Optional[float]:
        if val is None or val == "":
            return None
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            return None

    def is_available(self) -> bool:
        """Check if football-data.co.uk is reachable."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/2425/E0.csv",
                timeout=5
            )
            return resp.status_code == 200
        except:
            return False
