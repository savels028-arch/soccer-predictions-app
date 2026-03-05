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

        # 4. No demo fallback — empty is better than fake data
        if not matches:
            logger.warning("No API data available for today's matches (no demo fallback)")

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

        # 3. No demo fallback
        if not matches:
            logger.warning("No upcoming matches found from any source")

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

        # 4. Database fallback (no demo)
        if not matches:
            matches = self.db.get_live_matches()

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

        # 6. No demo fallback for historical
        if not matches:
            logger.warning(f"No historical data found for {league_code} season {season}")

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


