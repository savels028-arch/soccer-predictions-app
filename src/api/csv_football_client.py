"""
Football-Data.co.uk CSV Client
FREE historical match data with full statistics & betting odds.
No API key or registration needed!

Source: https://www.football-data.co.uk/
Covers: 25+ leagues, 20+ seasons, with odds from 6+ bookmakers.
"""
import csv
import hashlib
import io
import logging
from pathlib import Path
import requests
import time
from datetime import datetime
from typing import List, Dict, Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LEAGUES

logger = logging.getLogger(__name__)


def _season_code_for_year(season: int) -> Optional[str]:
    """Convert season start year to Football-Data URL code."""
    if 1993 <= season <= 2030:
        s1 = str(season)[2:]
        s2 = str(season + 1)[2:]
        return f"{s1}{s2}"
    return None


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
        # B1 is Belgium.  Never expose it as BSA: that app code correctly
        # means Brazil Serie A in every live feed.
        "BEL1": "B1",
    }

    # Season codes: 2024/25 = "2425", 2000/01 = "0001", etc.
    AVAILABLE_SEASONS = [
        (season, _season_code_for_year(season))
        for season in range(2025, 1992, -1)
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SoccerPredictionsPro/1.0",
        })
        self._cache = {}  # in-memory cache: (league, season) -> data
        self._last_request = 0
        self.cache_dir = Path(__file__).resolve().parents[2] / "data" / "cache" / "football_data_csv"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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
        cache_path = self.cache_dir / f"{season_code}_{csv_code}.csv"

        if cache_path.exists():
            try:
                matches = self._parse_csv(cache_path.read_text(encoding="utf-8", errors="ignore"), league_code, season)
                self._cache[cache_key] = matches
                logger.info(f"Parsed {len(matches)} cached matches from {league_code} {season}/{season+1}")
                return matches
            except Exception as e:
                logger.warning(f"CSV cache read failed for {cache_path}: {e}")

        self._rate_limit()
        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"CSV download failed: HTTP {resp.status_code} for {url}")
                return []

            try:
                cache_path.write_text(resp.text, encoding="utf-8")
            except Exception as e:
                logger.debug(f"CSV cache write failed for {cache_path}: {e}")

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
        return _season_code_for_year(season)

    def _parse_csv(self, csv_text: str, league_code: str, season: int) -> List[Dict]:
        """Parse CSV text into normalized match dicts."""
        matches = []
        try:
            # Handle BOM and encoding issues
            csv_text = csv_text.lstrip('\ufeff')
            reader = csv.DictReader(io.StringIO(csv_text))

            league_info = LEAGUES.get(league_code, {})
            if league_code == "BEL1":
                league_info = {
                    "name": "Belgian First Division A",
                    "country": "Belgium",
                    "emoji": "🇧🇪",
                }

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

            # Parse a complete 1X2 quote from one bookmaker. Never mix the
            # home price from one bookmaker with draw/away prices from another.
            home_odds, draw_odds, away_odds = self._first_complete_odds(
                row,
                [
                    ("B365H", "B365D", "B365A"),
                    ("BWH", "BWD", "BWA"),
                    ("PSH", "PSD", "PSA"),
                    ("WHH", "WHD", "WHA"),
                    ("IWH", "IWD", "IWA"),
                    ("GBH", "GBD", "GBA"),
                    ("LBH", "LBD", "LBA"),
                    ("SBH", "SBD", "SBA"),
                    ("SJH", "SJD", "SJA"),
                    ("VCH", "VCD", "VCA"),
                ],
            )

            # Football-Data used BbAv*/BbMx* before the newer Avg*/Max*
            # column names. Preserve both eras for consistent research.
            avg_home_odds = self._first_float(row, "AvgH", "BbAvH")
            avg_draw_odds = self._first_float(row, "AvgD", "BbAvD")
            avg_away_odds = self._first_float(row, "AvgA", "BbAvA")
            max_home_odds = self._first_float(row, "MaxH", "BbMxH")
            max_draw_odds = self._first_float(row, "MaxD", "BbMxD")
            max_away_odds = self._first_float(row, "MaxA", "BbMxA")

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
                "max_home_odds": max_home_odds,
                "max_draw_odds": max_draw_odds,
                "max_away_odds": max_away_odds,
                "b365_close_home": self._safe_float(row.get("B365CH")),
                "b365_close_draw": self._safe_float(row.get("B365CD")),
                "b365_close_away": self._safe_float(row.get("B365CA")),
                "avg_close_home_odds": self._safe_float(row.get("AvgCH")),
                "avg_close_draw_odds": self._safe_float(row.get("AvgCD")),
                "avg_close_away_odds": self._safe_float(row.get("AvgCA")),
                "max_close_home_odds": self._safe_float(row.get("MaxCH")),
                "max_close_draw_odds": self._safe_float(row.get("MaxCD")),
                "max_close_away_odds": self._safe_float(row.get("MaxCA")),
                # Over/under 2.5
                "b365_over25": self._safe_float(row.get("B365>2.5")),
                "b365_under25": self._safe_float(row.get("B365<2.5")),
                "pinnacle_over25": self._safe_float(row.get("P>2.5")),
                "pinnacle_under25": self._safe_float(row.get("P<2.5")),
                "avg_over25": self._first_float(row, "Avg>2.5", "BbAv>2.5"),
                "avg_under25": self._first_float(row, "Avg<2.5", "BbAv<2.5"),
                "max_over25": self._first_float(row, "Max>2.5", "BbMx>2.5"),
                "max_under25": self._first_float(row, "Max<2.5", "BbMx<2.5"),
                "b365_close_over25": self._safe_float(row.get("B365C>2.5")),
                "b365_close_under25": self._safe_float(row.get("B365C<2.5")),
                "avg_close_over25": self._safe_float(row.get("AvgC>2.5")),
                "avg_close_under25": self._safe_float(row.get("AvgC<2.5")),
                "max_close_over25": self._safe_float(row.get("MaxC>2.5")),
                "max_close_under25": self._safe_float(row.get("MaxC<2.5")),
                # Asian handicap (home line and two-way prices). Older files
                # pair B365AHH/AHA with Bet365's own B365AH line. AHh/BbAHh
                # is the shared market/Betbrain line and must stay separate.
                "asian_handicap_line": self._first_float(row, "AHh", "BbAHh"),
                "b365_asian_line": self._safe_float(row.get("B365AH")),
                "b365_asian_home": self._safe_float(row.get("B365AHH")),
                "b365_asian_away": self._safe_float(row.get("B365AHA")),
                "pinnacle_asian_home": self._safe_float(row.get("PAHH")),
                "pinnacle_asian_away": self._safe_float(row.get("PAHA")),
                "avg_asian_home": self._first_float(row, "AvgAHH", "BbAvAHH"),
                "avg_asian_away": self._first_float(row, "AvgAHA", "BbAvAHA"),
                "max_asian_home": self._first_float(row, "MaxAHH", "BbMxAHH"),
                "max_asian_away": self._first_float(row, "MaxAHA", "BbMxAHA"),
                "asian_handicap_close_line": self._safe_float(row.get("AHCh")),
                "b365_close_asian_home": self._safe_float(row.get("B365CAHH")),
                "b365_close_asian_away": self._safe_float(row.get("B365CAHA")),
                "pinnacle_close_asian_home": self._safe_float(row.get("PCAHH")),
                "pinnacle_close_asian_away": self._safe_float(row.get("PCAHA")),
                "avg_close_asian_home": self._safe_float(row.get("AvgCAHH")),
                "avg_close_asian_away": self._safe_float(row.get("AvgCAHA")),
                "max_close_asian_home": self._safe_float(row.get("MaxCAHH")),
                "max_close_asian_away": self._safe_float(row.get("MaxCAHA")),
                # Result
                "ftr": row.get("FTR", ""),  # H, D, A
                "htr": row.get("HTR", ""),  # Half-time result
            }

            # Generate a stable ID
            stable_key = f"{league_code}|{season}|{match_date}|{home}|{away}"
            api_id = int.from_bytes(
                hashlib.sha256(stable_key.encode("utf-8")).digest()[:8],
                "big",
            ) & ((1 << 63) - 1)

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

    def _first_float(self, row: Dict, *keys: str) -> Optional[float]:
        for key in keys:
            value = self._safe_float(row.get(key))
            if value is not None:
                return value
        return None

    def _first_complete_odds(self, row: Dict, triples) -> tuple:
        for home_key, draw_key, away_key in triples:
            odds = (
                self._safe_float(row.get(home_key)),
                self._safe_float(row.get(draw_key)),
                self._safe_float(row.get(away_key)),
            )
            if all(value is not None and value > 1.0 for value in odds):
                return odds
        return None, None, None

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
