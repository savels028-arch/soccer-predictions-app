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
from src.api.prediction_scraper import PredictionScraper
from src.api.danske_spil_client import DanskeSpilClient

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
        self.prediction_scraper = PredictionScraper()
        self.danske_spil = DanskeSpilClient()

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

    # ──────────────────────────────────────────
    # AI PREDICTIONS FROM EXTERNAL SITES
    # ──────────────────────────────────────────
    def fetch_ai_predictions(self, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch consensus AI predictions from multiple external prediction websites.
        Returns list of consensus prediction dicts sorted by number of sources.
        """
        cache_key = f"ai_predictions_{date.today().isoformat()}"

        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached:
                logger.info("Returning cached AI predictions")
                return cached

        try:
            consensus = self.prediction_scraper.get_consensus_predictions()
            logger.info(f"Got {len(consensus)} consensus AI predictions")
        except Exception as e:
            logger.error(f"AI prediction scraper error: {e}")
            consensus = []

        if consensus:
            self.db.set_cache(cache_key, consensus,
                              DATA_SETTINGS["cache_ttl_minutes"])
        return consensus

    # ──────────────────────────────────────────
    # DANSKE SPIL ODDS
    # ──────────────────────────────────────────
    def fetch_danske_spil_odds(self, force_refresh: bool = False) -> List[Dict]:
        """
        Hent alle tilgængelige fodboldkampe med odds fra Danske Spil.
        Returns list of event dicts with 1X2, O/U, BTTS odds.
        """
        cache_key = f"danske_spil_odds_{date.today().isoformat()}"

        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached:
                logger.info("Returning cached Danske Spil odds")
                return cached

        try:
            events = self.danske_spil.get_all_football_odds()
            logger.info(f"Got {len(events)} events from Danske Spil")
        except Exception as e:
            logger.error(f"Danske Spil scraper error: {e}")
            events = []

        if events:
            self.db.set_cache(cache_key, events,
                              DATA_SETTINGS["cache_ttl_minutes"])
        return events

    def match_predictions_with_danske_spil(
        self,
        predictions: List[Dict],
        force_refresh: bool = False,
    ) -> List[Dict]:
        """
        Match appens predictions med Danske Spil odds.
        Returnerer predictions beriget med danske_spil-info.
        """
        ds_events = self.fetch_danske_spil_odds(force_refresh=force_refresh)
        return self.danske_spil.match_predictions_with_odds(predictions, ds_events)

    def build_consensus_with_danske_spil(
        self,
        prediction_engine=None,
        matches: List[Dict] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """
        Automatisk konsensus-analyse:
        1. Hent AI-site predictions (consensus fra 4 sider)
        2. Hent ML-model predictions (ensemble)
        3. Find kampe hvor BEGGE kilder er enige om udfald
        4. Krydsreferér med Danske Spil odds
        5. Beregn value bets

        Returns dict med:
          - all_consensus: alle kampe med source-agreement
          - playable: kun dem der er hos Danske Spil
          - stats: opsummering
        """
        import re

        # ── 1. AI site consensus ──
        ai_consensus = self.fetch_ai_predictions(force_refresh=force_refresh)
        logger.info(f"Konsensus: {len(ai_consensus)} AI-site predictions")

        # ── 2. ML ensemble predictions ──
        ml_predictions = {}
        if prediction_engine and matches:
            try:
                all_preds = prediction_engine.predict_all_matches(matches)
                for match_key, preds in all_preds.items():
                    ensemble = next((p for p in preds if p.get("model_name") == "ensemble"), None)
                    if ensemble:
                        ml_predictions[self._norm_key(
                            ensemble.get("home_team", ""),
                            ensemble.get("away_team", "")
                        )] = ensemble
            except Exception as e:
                logger.warning(f"ML predictions fejl: {e}")

        logger.info(f"Konsensus: {len(ml_predictions)} ML ensemble predictions")

        # ── 3. Danske Spil odds ──
        ds_events = self.fetch_danske_spil_odds(force_refresh=force_refresh)
        ds_index = {}
        for ev in ds_events:
            key = self._norm_key(ev.get("home_team", ""), ev.get("away_team", ""))
            if key:
                ds_index[key] = ev

        logger.info(f"Konsensus: {len(ds_events)} Danske Spil events")

        # ── 4. Byg samlet konsensus ──
        combined = {}

        # Tilføj AI consensus
        for ai in ai_consensus:
            key = self._norm_key(ai.get("home_team", ""), ai.get("away_team", ""))
            if not key:
                continue
            combined[key] = {
                "home_team": ai.get("home_team", ""),
                "away_team": ai.get("away_team", ""),
                "league": ai.get("league", ""),
                "kickoff_time": ai.get("kickoff_time", ""),
                "sources": [],
                "ai_consensus": ai,
                "ml_ensemble": None,
                "danske_spil": None,
                "agreement_level": 0,
                "agreed_outcome": None,
            }
            # AI site predictions
            winner = ai.get("consensus_winner")
            if winner:
                outcome = {"1": "HOME_WIN", "X": "DRAW", "2": "AWAY_WIN"}.get(winner, winner)
                combined[key]["sources"].append({
                    "name": "AI Sites",
                    "type": "ai_consensus",
                    "prediction": outcome,
                    "confidence": ai.get("consensus_confidence"),
                    "num_sources": ai.get("num_sources", 0),
                    "home_pct": ai.get("avg_home_win_pct"),
                    "draw_pct": ai.get("avg_draw_pct"),
                    "away_pct": ai.get("avg_away_win_pct"),
                    "btts": ai.get("btts_consensus"),
                    "over_under": ai.get("over_under_consensus"),
                    "sites": ai.get("sources", []),
                })

        # Tilføj ML ensemble
        for key, ens in ml_predictions.items():
            if key not in combined:
                combined[key] = {
                    "home_team": ens.get("home_team", ""),
                    "away_team": ens.get("away_team", ""),
                    "league": ens.get("league_code", ""),
                    "kickoff_time": "",
                    "sources": [],
                    "ai_consensus": None,
                    "ml_ensemble": None,
                    "danske_spil": None,
                    "agreement_level": 0,
                    "agreed_outcome": None,
                }
            combined[key]["ml_ensemble"] = ens
            combined[key]["sources"].append({
                "name": "ML Ensemble",
                "type": "ml_ensemble",
                "prediction": ens.get("predicted_outcome"),
                "confidence": ens.get("confidence"),
                "home_pct": ens.get("home_win_prob"),
                "draw_pct": ens.get("draw_prob"),
                "away_pct": ens.get("away_win_prob"),
                "suggestion": ens.get("suggestion", ""),
            })

        # ── 5. Beregn enighed og match med DS ──
        all_consensus = []
        for key, entry in combined.items():
            sources = entry["sources"]
            predictions_by_source = [s["prediction"] for s in sources if s.get("prediction")]

            # Find enighed
            if len(predictions_by_source) >= 2:
                from collections import Counter
                counts = Counter(predictions_by_source)
                most_common_outcome, most_common_count = counts.most_common(1)[0]
                entry["agreement_level"] = most_common_count
                entry["agreed_outcome"] = most_common_outcome if most_common_count >= 2 else None
                entry["all_agree"] = most_common_count == len(predictions_by_source)
            elif len(predictions_by_source) == 1:
                entry["agreement_level"] = 1
                entry["agreed_outcome"] = predictions_by_source[0]
                entry["all_agree"] = False
            else:
                entry["agreement_level"] = 0
                entry["agreed_outcome"] = None
                entry["all_agree"] = False

            # Match med Danske Spil (direkte + fuzzy)
            ds_match = ds_index.get(key)
            if not ds_match:
                ds_match = self.danske_spil._fuzzy_find(key, ds_index)
            if ds_match:
                entry["danske_spil"] = ds_match

            all_consensus.append(entry)

        # Sortér: enige + spilbare først, dernæst enige, dernæst resten
        all_consensus.sort(key=lambda x: (
            -(1 if x["danske_spil"] else 0),
            -(x["agreement_level"]),
            -(1 if x["all_agree"] else 0),
        ))

        playable = [x for x in all_consensus if x["danske_spil"] and x["agreed_outcome"]]
        agreed = [x for x in all_consensus if x["agreed_outcome"]]

        stats = {
            "total_matches": len(all_consensus),
            "ai_predictions": len(ai_consensus),
            "ml_predictions": len(ml_predictions),
            "ds_events": len(ds_events),
            "agreed": len(agreed),
            "playable": len(playable),
            "playable_agree_all": sum(1 for x in playable if x.get("all_agree")),
        }
        logger.info(
            "Konsensus færdig: %d kampe, %d enige, %d spilbare hos DS",
            stats["total_matches"], stats["agreed"], stats["playable"]
        )

        return {
            "all_consensus": all_consensus,
            "playable": playable,
            "agreed": agreed,
            "stats": stats,
        }

    @staticmethod
    def _norm_key(home: str, away: str) -> str:
        """Normaliseret match-key for cross-source matching."""
        import re

        def _n(name: str) -> str:
            if not name:
                return ""
            n = name.strip().lower()
            for sfx in (" fc", " sc", " cf", " bc", " fk", " sk"):
                if n.endswith(sfx):
                    n = n[: -len(sfx)].strip()
            for pfx in ("fc ", "sc ", "fk ", "sk ", "ac ", "as "):
                if n.startswith(pfx):
                    n = n[len(pfx):].strip()
            n = (n.replace("ü", "u").replace("ö", "o").replace("é", "e")
                  .replace("á", "a").replace("ñ", "n").replace("ç", "c"))
            n = re.sub(r"[^\w\s]", "", n)
            return re.sub(r"\s+", " ", n).strip()

        h, a = _n(home), _n(away)
        return f"{h}_vs_{a}" if h and a else ""

    def fetch_ai_predictions_raw(self, force_refresh: bool = False) -> List[Dict]:
        """
        Fetch raw (non-consensus) AI predictions from all sources.
        Returns flat list of individual predictions from each site.
        """
        cache_key = f"ai_predictions_raw_{date.today().isoformat()}"

        if not force_refresh:
            cached = self.db.get_cache(cache_key)
            if cached:
                logger.info("Returning cached raw AI predictions")
                return cached

        try:
            preds = self.prediction_scraper.get_all_predictions()
            logger.info(f"Got {len(preds)} raw AI predictions")
        except Exception as e:
            logger.error(f"AI prediction scraper error: {e}")
            preds = []

        if preds:
            self.db.set_cache(cache_key, preds,
                              DATA_SETTINGS["cache_ttl_minutes"])
        return preds

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
