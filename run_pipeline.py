#!/usr/bin/env python3
"""
AIBets Prediction Pipeline — Main Orchestrator

Connects real data sources → ML models → Firestore.
Replaces the fake hash-based predictions in the Next.js cron.

Data flow:
  1. ESPN/TheSportsDB     → match fixtures + results
  2. Danske Spil (Kambi)  → real 1X2/BTTS/O-U odds
  3. 4 AI prediction sites → external predictions (features for meta-model)
  4. ML Ensemble           → XGBoost + NN + RF predictions
  5. Performance weighting → source-weighted consensus
  6. Value detection       → edge vs market odds
  7. Firestore             → new structured collections + legacy cache

Usage:
  python run_pipeline.py                    # Full pipeline
  python run_pipeline.py --odds-only        # Just update odds
  python run_pipeline.py --evaluate-only    # Just evaluate finished matches
  python run_pipeline.py --train            # Train ML models first, then predict

Requires:
  - Firebase service account key (FIREBASE_SERVICE_ACCOUNT_KEY env var or service-account.json)
  - Python deps: firebase-admin, requests, numpy, scikit-learn, xgboost, pandas
"""

import os
import sys
import json
import math
import logging
import argparse
import traceback
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.firestore_writer import FirestoreWriter, match_id, _normalize_team
from src.api.prediction_scraper import PredictionScraper
from src.api.danske_spil_client import DanskeSpilClient
from src.api.free_football_client import FreeFootballClient
from src.database.db_manager import DatabaseManager
from src.predictions.prediction_engine import PredictionEngine
from src.predictions.feature_engineering import FeatureEngineer

# Optional: CSV + API-Football
try:
    from src.api.csv_football_client import CSVFootballClient
    HAS_CSV = True
except ImportError:
    HAS_CSV = False

try:
    from src.api.api_football_client import ApiFootballClient
    HAS_API_FOOTBALL = True
except ImportError:
    HAS_API_FOOTBALL = False

# Optional: FlashScore
try:
    from src.scrapers.flashscore_scraper import FlashScoreScraper
    HAS_FLASHSCORE = True
except ImportError:
    HAS_FLASHSCORE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ─── Team name matching ─────────────────────

# Common abbreviations/aliases for team name matching
TEAM_ALIASES = {
    "psg": "paris saint germain",
    "paris saint-germain": "paris saint germain",
    "man utd": "manchester united",
    "man city": "manchester city",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "newcastle utd": "newcastle united",
    "leeds utd": "leeds united",
    "west ham utd": "west ham united",
    "sheff utd": "sheffield united",
    "nott'm forest": "nottingham forest",
    "nottingham": "nottingham forest",
    "atletico madrid": "atletico de madrid",
    "atletico": "atletico de madrid",
    "atl. madrid": "atletico de madrid",
    "atlético madrid": "atletico de madrid",
    "atlético de madrid": "atletico de madrid",
    "athletic bilbao": "athletic club",
    "real sociedad": "real sociedad",
    "betis": "real betis",
    "celta vigo": "celta de vigo",
    "cadiz": "cadiz cf",
    "deportivo alaves": "alaves",
    "inter": "inter milan",
    "internazionale": "inter milan",
    "inter milan": "inter milan",
    "ac milan": "milan",
    "napoli": "ssc napoli",
    "hellas verona": "verona",
    "monza": "ac monza",
    "rb leipzig": "rasenballsport leipzig",
    "leipzig": "rasenballsport leipzig",
    "leverkusen": "bayer leverkusen",
    "bayer 04 leverkusen": "bayer leverkusen",
    "gladbach": "borussia monchengladbach",
    "b. monchengladbach": "borussia monchengladbach",
    "dortmund": "borussia dortmund",
    "b. dortmund": "borussia dortmund",
    "bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "fc bayern münchen": "bayern munich",
    "st. pauli": "fc st pauli",
    "mainz 05": "mainz",
    "freiburg": "sc freiburg",
    "augsburg": "fc augsburg",
    "heidenheim": "1. fc heidenheim",
    "lens": "rc lens",
    "lyon": "olympique lyonnais",
    "marseille": "olympique de marseille",
    "om": "olympique de marseille",
    "ol": "olympique lyonnais",
    "monaco": "as monaco",
    "saint etienne": "as saint-etienne",
    "st. etienne": "as saint-etienne",
    "psv": "psv eindhoven",
    "ajax": "afc ajax",
    "feyenoord": "feyenoord rotterdam",
    "az": "az alkmaar",
    "porto": "fc porto",
    "sporting": "sporting cp",
    "sporting lisbon": "sporting cp",
    "benfica": "sl benfica",
}


def _canonical_name(name: str) -> str:
    """Get canonical team name for matching."""
    n = name.lower().strip()
    # Check aliases first
    if n in TEAM_ALIASES:
        n = TEAM_ALIASES[n]
    # Remove common prefixes/suffixes
    for prefix in ["fc ", "cf ", "ac ", "as ", "ss ", "us ", "sc ", "afc ",
                    "rcd ", "sl ", "ssc "]:
        if n.startswith(prefix):
            check = n[len(prefix):]
            if check in TEAM_ALIASES:
                n = TEAM_ALIASES[check]
                break
    return n


def fuzzy_match_teams(name1: str, name2: str) -> bool:
    """Check if two team names likely refer to the same team."""
    c1 = _canonical_name(name1)
    c2 = _canonical_name(name2)
    
    if c1 == c2:
        return True
    
    # One contains the other (after canonicalization)
    if c1 in c2 or c2 in c1:
        return True
    
    # Normalize deeper for word comparison
    n1 = _normalize_team(name1)
    n2 = _normalize_team(name2)
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    
    # Word overlap
    w1 = set(c1.split())
    w2 = set(c2.split())
    w1.discard("")
    w2.discard("")
    if len(w1) == 0 or len(w2) == 0:
        return False
    
    # Remove common noise words
    noise = {"fc", "cf", "ac", "as", "ss", "us", "sc", "afc", "sl", "ssc", "de", "la", "le", "1."}
    w1 -= noise
    w2 -= noise
    if len(w1) == 0 or len(w2) == 0:
        return False
    
    overlap = len(w1 & w2) / min(len(w1), len(w2))
    return overlap >= 0.5


def find_match_in_list(target_home: str, target_away: str,
                       match_list: list) -> Optional[dict]:
    """Find a match in a list by fuzzy team name matching."""
    for m in match_list:
        mh = m.get("home_team") or m.get("home_team_name") or ""
        ma = m.get("away_team") or m.get("away_team_name") or ""
        if fuzzy_match_teams(target_home, mh) and fuzzy_match_teams(target_away, ma):
            return m
    return None


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

class PredictionPipeline:
    def __init__(self):
        log.info("═══ Initializing Pipeline ═══")
        self.fs = FirestoreWriter()
        self.db = DatabaseManager()
        self.espn = FreeFootballClient()
        self.scraper = PredictionScraper()
        self.danske_spil = DanskeSpilClient()
        self.engine = PredictionEngine(db_manager=self.db)
        self.feature_eng = FeatureEngineer()

        self.csv_client = CSVFootballClient() if HAS_CSV else None
        self.api_football = ApiFootballClient() if HAS_API_FOOTBALL else None
        self.flashscore = FlashScoreScraper() if HAS_FLASHSCORE else None

        # Collected data during pipeline run
        self._matches: List[dict] = []
        self._odds: List[dict] = []
        self._ai_preds: List[dict] = []
        self._ml_preds: Dict[str, dict] = {}  # matchId → prediction

        self._stats = {
            "matches_fetched": 0,
            "odds_fetched": 0,
            "ai_predictions": 0,
            "ml_predictions": 0,
            "results_saved": 0,
            "coupons_evaluated": 0,
            "sources_updated": 0,
        }

    # ──────────────────────────────────────
    # STAGE 1: Fetch matches from ESPN
    # ──────────────────────────────────────

    # ESPN league slugs
    ESPN_LEAGUES = ["PL", "PD", "BL1", "SA", "FL1", "CL", "DED", "PPL"]

    def fetch_matches(self) -> List[dict]:
        """Fetch today's + upcoming + recent matches from ESPN/TheSportsDB."""
        log.info("── Stage 1: Fetching matches ──")
        all_matches = []

        # ESPN: today + tomorrow + past 2 days, per league
        for delta in [-2, -1, 0, 1]:
            d = date.today() + timedelta(days=delta)
            d_str = d.strftime("%Y-%m-%d")
            for league in self.ESPN_LEAGUES:
                try:
                    day_matches = self.espn._espn_get_scoreboard(league, d_str)
                    if day_matches:
                        all_matches.extend(day_matches)
                except Exception as e:
                    log.debug(f"  ESPN {league} {d_str} failed: {e}")
            log.info(f"  ESPN {d_str}: {len(all_matches)} total so far")

        # TheSportsDB: today
        try:
            sdb = self.espn._sportsdb_get_todays_matches()
            if sdb:
                log.info(f"  TheSportsDB today: {len(sdb)} matches")
                # Merge without duplicates
                for m in sdb:
                    if not find_match_in_list(
                        m.get("home_team_name", ""), m.get("away_team_name", ""),
                        all_matches
                    ):
                        all_matches.append(m)
        except Exception as e:
            log.warning(f"  TheSportsDB failed: {e}")

        # FlashScore: today + tomorrow
        if self.flashscore:
            try:
                fs_today = self.flashscore.fetch_todays_matches()
                log.info(f"  FlashScore today: {len(fs_today)} matches")
                for fm in fs_today:
                    fh = fm.get("homeTeam", "")
                    fa = fm.get("awayTeam", "")
                    existing = find_match_in_list(fh, fa, all_matches)
                    if existing:
                        # Enrich with FlashScore ID + live data
                        existing["flashscoreId"] = fm.get("flashscore_id", "")
                        if fm.get("status") == "LIVE":
                            existing["status"] = "LIVE"
                            existing["minute"] = fm.get("minute", "")
                        if fm.get("homeScore") is not None:
                            existing["home_score"] = fm["homeScore"]
                            existing["away_score"] = fm.get("awayScore")
                    else:
                        # New match not in ESPN — add it
                        all_matches.append({
                            "home_team_name": fh,
                            "away_team_name": fa,
                            "home_score": fm.get("homeScore"),
                            "away_score": fm.get("awayScore"),
                            "status": fm.get("status", "SCHEDULED"),
                            "league_name": fm.get("league", ""),
                            "match_date": date.today().isoformat(),
                            "flashscoreId": fm.get("flashscore_id", ""),
                            "source": "flashscore",
                        })
            except Exception as e:
                log.warning(f"  FlashScore failed: {e}")

        # Deduplicate
        seen = set()
        deduped = []
        for m in all_matches:
            key = match_id(
                m.get("match_date", ""),
                m.get("home_team_name", m.get("home_team", "")),
                m.get("away_team_name", m.get("away_team", "")),
            )
            if key not in seen:
                seen.add(key)
                deduped.append(m)
        
        self._matches = deduped
        self._stats["matches_fetched"] = len(deduped)
        log.info(f"  Total: {len(deduped)} unique matches")

        # Write to Firestore: matches/ collection
        for m in deduped:
            try:
                self.fs.upsert_match(m)
            except Exception as e:
                log.error(f"  Failed to upsert match: {e}")

        return deduped

    # ──────────────────────────────────────
    # STAGE 2: Fetch real odds from Danske Spil
    # ──────────────────────────────────────

    def fetch_odds(self) -> List[dict]:
        """Fetch real odds from Danske Spil (Kambi API)."""
        log.info("── Stage 2: Fetching Danske Spil odds ──")
        try:
            odds = self.danske_spil.get_all_football_odds() or []
            self._odds = odds
            self._stats["odds_fetched"] = len(odds)
            log.info(f"  Got {len(odds)} odds entries from Kambi")

            # Match odds to our matches and write to Firestore
            matched = 0
            for od in odds:
                oh = od.get("home_team", "")
                oa = od.get("away_team", "")
                
                # Find matching match
                m = find_match_in_list(oh, oa, self._matches)
                if m:
                    mid = match_id(
                        m.get("match_date", ""),
                        m.get("home_team_name", m.get("home_team", "")),
                        m.get("away_team_name", m.get("away_team", "")),
                    )
                    try:
                        self.fs.update_match_odds(mid, {
                            "home_odds": od.get("home_odds", 0),
                            "draw_odds": od.get("draw_odds", 0),
                            "away_odds": od.get("away_odds", 0),
                        })

                        # Also store Danske Spil's implied probabilities as a "prediction"
                        ho = od.get("home_odds", 0)
                        do_ = od.get("draw_odds", 0)
                        ao = od.get("away_odds", 0)
                        if ho > 0 and do_ > 0 and ao > 0:
                            total = 1/ho + 1/do_ + 1/ao
                            self.fs.add_prediction(mid, "danske_spil_market", {
                                "home": round(1/ho / total, 4),
                                "draw": round(1/do_ / total, 4),
                                "away": round(1/ao / total, 4),
                            }, odds_at_scrape={"home": ho, "draw": do_, "away": ao},
                            extra={
                                "overUnder25": {
                                    "over": od.get("over_25_odds"),
                                    "under": od.get("under_25_odds"),
                                },
                                "btts": {
                                    "yes": od.get("btts_yes_odds"),
                                    "no": od.get("btts_no_odds"),
                                },
                                "deeplink": od.get("deeplink", ""),
                            })
                        matched += 1
                    except Exception as e:
                        log.error(f"  Failed to write odds for {oh} vs {oa}: {e}")

            log.info(f"  Matched {matched}/{len(odds)} odds to our matches")

            # FlashScore odds as fallback for matches without Danske Spil odds
            if self.flashscore:
                fs_odds_count = 0
                for m in self._matches:
                    mid = match_id(
                        m.get("match_date", ""),
                        m.get("home_team_name", m.get("home_team", "")),
                        m.get("away_team_name", m.get("away_team", "")),
                    )
                    fs_id = m.get("flashscoreId", "")
                    # Only fetch FlashScore odds if no Danske Spil odds matched
                    if fs_id and not find_match_in_list(
                        m.get("home_team_name", m.get("home_team", "")),
                        m.get("away_team_name", m.get("away_team", "")),
                        odds
                    ):
                        try:
                            fs_odds = self.flashscore.fetch_odds(fs_id)
                            if fs_odds and fs_odds.get("average"):
                                avg = fs_odds["average"]
                                ho = avg.get("home", 0)
                                do_ = avg.get("draw", 0)
                                ao = avg.get("away", 0)
                                if ho and do_ and ao:
                                    self.fs.update_match_odds(mid, {
                                        "home_odds": ho, "draw_odds": do_, "away_odds": ao,
                                    })
                                    total = 1/ho + 1/do_ + 1/ao
                                    self.fs.add_prediction(mid, "flashscore_market", {
                                        "home": round(1/ho / total, 4),
                                        "draw": round(1/do_ / total, 4),
                                        "away": round(1/ao / total, 4),
                                    }, odds_at_scrape={"home": ho, "draw": do_, "away": ao})
                                    fs_odds_count += 1
                        except Exception as e:
                            log.debug(f"  FlashScore odds for {fs_id} failed: {e}")
                log.info(f"  FlashScore fallback odds: {fs_odds_count} matches")

            return odds

        except Exception as e:
            log.error(f"  Danske Spil fetch failed: {e}")
            traceback.print_exc()
            return []

    # ──────────────────────────────────────
    # STAGE 3: Scrape AI prediction sites
    # ──────────────────────────────────────

    def scrape_ai_predictions(self) -> List[dict]:
        """Scrape predictions from 4 AI prediction websites."""
        log.info("── Stage 3: Scraping AI prediction sites ──")
        try:
            preds = self.scraper.get_all_predictions() or []
            self._ai_preds = preds
            self._stats["ai_predictions"] = len(preds)
            log.info(f"  Got {len(preds)} AI predictions from scrapers")

            # Get current market odds for oddsAtScrape
            odds_by_match = {}
            for od in self._odds:
                key = _normalize_team(od.get("home_team", "")) + "_" + _normalize_team(od.get("away_team", ""))
                odds_by_match[key] = {
                    "home": od.get("home_odds", 0),
                    "draw": od.get("draw_odds", 0),
                    "away": od.get("away_odds", 0),
                }

            # Write each prediction to Firestore
            written = 0
            for pred in preds:
                ph = pred.get("home_team", "")
                pa = pred.get("away_team", "")
                source = pred.get("source", "unknown")

                # Find matching match
                m = find_match_in_list(ph, pa, self._matches)
                if not m:
                    continue

                mid = match_id(
                    m.get("match_date", ""),
                    m.get("home_team_name", m.get("home_team", "")),
                    m.get("away_team_name", m.get("away_team", "")),
                )

                # Build probabilities
                h_pct = pred.get("home_win_pct")
                d_pct = pred.get("draw_pct")
                a_pct = pred.get("away_win_pct")
                if h_pct is not None and a_pct is not None:
                    total_pct = (h_pct or 0) + (d_pct or 0) + (a_pct or 0)
                    if total_pct > 0:
                        probs = {
                            "home": round((h_pct or 0) / total_pct, 4),
                            "draw": round((d_pct or 0) / total_pct, 4),
                            "away": round((a_pct or 0) / total_pct, 4),
                        }
                    else:
                        probs = {"home": 0.33, "draw": 0.33, "away": 0.34}
                else:
                    # Only have predicted winner, assign rough probs
                    winner = pred.get("predicted_winner", "1")
                    if winner == "1":
                        probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
                    elif winner == "2":
                        probs = {"home": 0.20, "draw": 0.25, "away": 0.55}
                    else:
                        probs = {"home": 0.30, "draw": 0.40, "away": 0.30}

                # OddsAtScrape
                odds_key = _normalize_team(ph) + "_" + _normalize_team(pa)
                oas = odds_by_match.get(odds_key)

                extra = {}
                if pred.get("btts"):
                    extra["btts"] = pred["btts"]
                if pred.get("over_under_25"):
                    extra["overUnder25"] = pred["over_under_25"]
                if pred.get("predicted_score"):
                    extra["predictedScore"] = pred["predicted_score"]

                try:
                    self.fs.add_prediction(mid, source, probs, oas, extra if extra else None)
                    written += 1
                except Exception as e:
                    log.error(f"  Failed to write prediction for {ph} vs {pa}: {e}")

            log.info(f"  Wrote {written}/{len(preds)} predictions to Firestore")
            return preds

        except Exception as e:
            log.error(f"  AI scraping failed: {e}")
            traceback.print_exc()
            return []

    # ──────────────────────────────────────
    # STAGE 4: ML Ensemble predictions
    # ──────────────────────────────────────

    def train_models(self):
        """Train ML models on historical data."""
        log.info("── Training ML models ──")
        try:
            results = self.engine.train_models(
                league_codes=["PL", "PD", "BL1", "SA", "FL1"],
                callback=lambda t, msg: log.info(f"  [{t}] {msg}")
            )
            log.info(f"  Training results: {results}")
            return results
        except Exception as e:
            log.error(f"  Training failed: {e}")
            traceback.print_exc()
            return {}

    def run_ml_predictions(self) -> Dict[str, dict]:
        """Run ML ensemble on all upcoming matches."""
        log.info("── Stage 4: Running ML predictions ──")

        if not self.engine.is_trained:
            log.warning("  ML models not trained. Attempting to load from disk...")
            if not self.engine.is_trained:
                log.warning("  No trained models found. Run with --train first.")
                log.warning("  Skipping ML predictions.")
                return {}

        upcoming = [m for m in self._matches
                    if m.get("status") in ("SCHEDULED", "IN_PLAY", "pre")]
        log.info(f"  Running predictions for {len(upcoming)} upcoming matches")

        ml_preds = {}
        for m in upcoming:
            home = m.get("home_team_name", m.get("home_team", ""))
            away = m.get("away_team_name", m.get("away_team", ""))
            mid = match_id(m.get("match_date", ""), home, away)

            try:
                # Get odds for this match (from Danske Spil)
                od = find_match_in_list(home, away, self._odds)
                home_odds = od.get("home_odds", 2.5) if od else 2.5
                draw_odds = od.get("draw_odds", 3.3) if od else 3.3  
                away_odds = od.get("away_odds", 3.0) if od else 3.0

                # Get team stats
                league = m.get("league_code", m.get("league", "PL"))
                home_stats = self.db.compute_team_stats_from_matches(home, league, 2025) or {}
                away_stats = self.db.compute_team_stats_from_matches(away, league, 2025) or {}

                # Get H2H
                h2h = self.db.get_h2h(home, away) or []

                # Build features
                features = self.feature_eng.build_match_features(
                    home_stats, away_stats, h2h,
                    home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds,
                )

                # Run all models
                model_results = {}
                for model_name, model in self.engine.models.items():
                    if model.is_trained:
                        try:
                            proba = model.predict_proba(features.reshape(1, -1))
                            if proba is not None and len(proba) > 0:
                                p = proba[0]
                                model_results[model_name] = {
                                    "home": float(p[0]),
                                    "draw": float(p[1]),
                                    "away": float(p[2]),
                                }
                                # Store each model as a separate prediction (source)
                                self.fs.add_prediction(
                                    mid, f"ml_{model_name}",
                                    model_results[model_name],
                                    {"home": home_odds, "draw": draw_odds, "away": away_odds}
                                )
                        except Exception as e:
                            log.debug(f"    Model {model_name} failed for {home} vs {away}: {e}")

                if not model_results:
                    continue

                # Ensemble: weighted average
                weights = {"xgboost": 0.40, "neural_network": 0.35, "random_forest": 0.25}
                ensemble = {"home": 0.0, "draw": 0.0, "away": 0.0}
                total_weight = 0.0
                for model_name, probs in model_results.items():
                    w = weights.get(model_name, 0.25)
                    for k in ["home", "draw", "away"]:
                        ensemble[k] += probs[k] * w
                    total_weight += w
                
                if total_weight > 0:
                    for k in ensemble:
                        ensemble[k] /= total_weight
                        ensemble[k] = round(ensemble[k], 4)

                # Calculate edge vs market
                edge = {}
                if od:
                    ho, do_, ao = home_odds, draw_odds, away_odds
                    if ho > 0 and do_ > 0 and ao > 0:
                        total_impl = 1/ho + 1/do_ + 1/ao
                        fair = {
                            "home": 1/ho / total_impl,
                            "draw": 1/do_ / total_impl,
                            "away": 1/ao / total_impl,
                        }
                        edge = {k: round(ensemble[k] - fair[k], 4) for k in ensemble}

                # Determine recommended bet
                best_outcome = max(ensemble, key=ensemble.get)
                confidence = ensemble[best_outcome]
                recommended = None
                if edge:
                    # Only recommend if edge > 3% and confidence > 50%
                    best_edge_outcome = max(edge, key=edge.get)
                    if edge[best_edge_outcome] > 0.03 and ensemble[best_edge_outcome] > 0.50:
                        recommended = best_edge_outcome.upper()

                # Save model output
                self.fs.save_model_output(
                    mid, ensemble, edge if edge else None,
                    recommended, confidence,
                    model_version=f"ensemble_v1_{len(model_results)}models"
                )

                ml_preds[mid] = {
                    "home_team": home,
                    "away_team": away,
                    "match_date": m.get("match_date", ""),
                    "league": m.get("league_code", m.get("league", "")),
                    "ensemble": ensemble,
                    "edge": edge,
                    "recommended": recommended,
                    "confidence": confidence,
                    "models": model_results,
                }

            except Exception as e:
                log.error(f"  ML prediction failed for {home} vs {away}: {e}")
                traceback.print_exc()

        self._ml_preds = ml_preds
        self._stats["ml_predictions"] = len(ml_preds)
        log.info(f"  Generated {len(ml_preds)} ML predictions")
        return ml_preds

    # ──────────────────────────────────────
    # STAGE 5: Compute meta-features
    # ──────────────────────────────────────

    def compute_meta_features(self):
        """Compute disagreement / meta-features for each match."""
        log.info("── Stage 5: Computing meta-features ──")
        count = 0
        quota_errors = 0

        for m in self._matches:
            if m.get("status") not in ("SCHEDULED", "IN_PLAY", "pre"):
                continue

            # Skip remaining if quota exceeded
            if quota_errors >= 2:
                log.warning("  Skipping remaining meta-features due to quota limits")
                break

            home = m.get("home_team_name", m.get("home_team", ""))
            away = m.get("away_team_name", m.get("away_team", ""))
            mid = match_id(m.get("match_date", ""), home, away)

            try:
                preds = self.fs.get_predictions_for_match(mid)
                if len(preds) < 2:
                    continue

                # Collect probabilities per source
                home_probs = []
                draw_probs = []
                away_probs = []
                sources = []

                for p in preds:
                    probs = p.get("probabilities", {})
                    hp = probs.get("home", 0.33)
                    dp = probs.get("draw", 0.33)
                    ap = probs.get("away", 0.34)
                    home_probs.append(hp)
                    draw_probs.append(dp)
                    away_probs.append(ap)
                    sources.append(p.get("source", ""))

                import numpy as np
                features = {
                    "sourceCount": len(preds),
                    "avgHomeProb": round(float(np.mean(home_probs)), 4),
                    "avgDrawProb": round(float(np.mean(draw_probs)), 4),
                    "avgAwayProb": round(float(np.mean(away_probs)), 4),
                    "stdHomeProb": round(float(np.std(home_probs)), 4),
                    "stdDrawProb": round(float(np.std(draw_probs)), 4),
                    "stdAwayProb": round(float(np.std(away_probs)), 4),
                    "maxMinSpreadHome": round(max(home_probs) - min(home_probs), 4),
                    "maxMinSpreadDraw": round(max(draw_probs) - min(draw_probs), 4),
                    "maxMinSpreadAway": round(max(away_probs) - min(away_probs), 4),
                    "sources": sources,
                }

                # Market comparison
                match_doc = self.fs.get_match(mid)
                if match_doc and "currentOdds" in match_doc:
                    co = match_doc["currentOdds"]
                    ho, do_, ao = co.get("home", 0), co.get("draw", 0), co.get("away", 0)
                    if ho > 0 and do_ > 0 and ao > 0:
                        total = 1/ho + 1/do_ + 1/ao
                        features["marketHomeProb"] = round(1/ho / total, 4)
                        features["marketDrawProb"] = round(1/do_ / total, 4)
                        features["marketAwayProb"] = round(1/ao / total, 4)
                        features["deltaVsMarketHome"] = round(features["avgHomeProb"] - features["marketHomeProb"], 4)
                        features["deltaVsMarketDraw"] = round(features["avgDrawProb"] - features["marketDrawProb"], 4)
                        features["deltaVsMarketAway"] = round(features["avgAwayProb"] - features["marketAwayProb"], 4)

                # Disagreement flag
                max_spread = max(
                    features["maxMinSpreadHome"],
                    features["maxMinSpreadDraw"],
                    features["maxMinSpreadAway"],
                )
                features["disagreementFlag"] = max_spread > 0.15

                self.fs.save_model_features(mid, features)
                count += 1

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "Quota" in err_str or "quota" in err_str:
                    quota_errors += 1
                    log.warning(f"  Quota exceeded for {home} vs {away}, will skip if persistent")
                else:
                    log.error(f"  Meta-features failed for {home} vs {away}: {e}")

        log.info(f"  Computed meta-features for {count} matches")

    # ──────────────────────────────────────
    # STAGE 6: Build daily coupon
    # ──────────────────────────────────────

    def build_daily_coupon(self):
        """Build today's Vinderkupon from model outputs + odds."""
        log.info("── Stage 6: Building daily coupon ──")
        today = date.today().strftime("%Y-%m-%d")

        # Collect candidates: matches with ML predictions + Danske Spil odds
        candidates = []
        for mid, pred in self._ml_preds.items():
            od = find_match_in_list(
                pred["home_team"], pred["away_team"], self._odds
            )
            if not od:
                continue

            best_outcome = max(pred["ensemble"], key=pred["ensemble"].get)
            odds_val = (od.get("home_odds", 0) if best_outcome == "home"
                       else od.get("away_odds", 0) if best_outcome == "away"
                       else od.get("draw_odds", 0))

            if odds_val <= 1.0:
                continue

            edge_val = pred.get("edge", {}).get(best_outcome, 0)

            candidates.append({
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": pred["match_date"],
                "kickoff": "",
                "pick": best_outcome.upper(),
                "odds": round(odds_val, 2),
                "confidence": round(pred["confidence"] * 100, 1),
                "edge": round(edge_val * 100, 1),
            })

        # Sort by edge (value) then confidence
        candidates.sort(key=lambda c: (c["edge"], c["confidence"]), reverse=True)
        picks = candidates[:4]

        if picks:
            total_odds = 1.0
            for p in picks:
                total_odds *= p["odds"]
            self.fs.save_daily_coupon(today, picks, total_odds)
            log.info(f"  Daily coupon: {len(picks)} picks, total odds: {total_odds:.2f}")
        else:
            log.info("  No suitable picks for today's coupon")

    # ──────────────────────────────────────
    # STAGE 7: Evaluate finished matches
    # ──────────────────────────────────────

    def evaluate_finished(self):
        """Evaluate predictions for finished matches, update source performance."""
        log.info("── Stage 7: Evaluating finished matches ──")
        finished = [m for m in self._matches if m.get("status") in ("FINISHED", "post")]
        log.info(f"  {len(finished)} finished matches to evaluate")

        results_saved = 0
        for m in finished:
            home = m.get("home_team_name", m.get("home_team", ""))
            away = m.get("away_team_name", m.get("away_team", ""))
            hs = m.get("home_score")
            as_ = m.get("away_score")
            if hs is None or as_ is None:
                continue

            hs, as_ = int(hs), int(as_)
            actual = "HOME" if hs > as_ else ("AWAY" if hs < as_ else "DRAW")
            mid = match_id(m.get("match_date", ""), home, away)

            # Update match result
            try:
                self.fs.update_match_result(mid, hs, as_)
            except Exception:
                pass

            # Save to legacy prediction_results collection
            # Use the ensemble prediction if available
            model_out = self.fs.get_match(mid)
            if model_out:
                # Look for model output
                mo_doc = self.fs.db.collection("model_outputs").document(mid).get()
                if mo_doc.exists:
                    mo_data = mo_doc.to_dict()
                    fp = mo_data.get("finalProbability", {})
                    predicted = max(fp, key=fp.get).upper() if fp else "HOME"
                    confidence = round(mo_data.get("confidenceScore", 0.5) * 100)
                else:
                    predicted = "HOME"
                    confidence = 50

                saved = self.fs.save_prediction_result({
                    "matchDate": m.get("match_date", "")[:10],
                    "homeTeam": home,
                    "awayTeam": away,
                    "leagueCode": m.get("league_code", m.get("league", "")),
                    "homeScore": hs,
                    "awayScore": as_,
                    "actualOutcome": actual,
                    "predictedOutcome": predicted,
                    "confidence": confidence,
                    "source": "ML Ensemble",
                    "isCorrect": predicted == actual,
                })
                if saved:
                    results_saved += 1

        self._stats["results_saved"] = results_saved
        log.info(f"  Saved {results_saved} new prediction results")

        # Evaluate pending coupons
        self._evaluate_coupons(finished)

    def _evaluate_coupons(self, finished_matches: list):
        """Evaluate pending daily coupons against finished matches."""
        pending = self.fs.get_pending_coupons()
        log.info(f"  Evaluating {len(pending)} pending coupons")

        for coupon in pending:
            picks = coupon.get("picks", [])
            pick_results = []
            all_correct = True
            has_pending = False

            for pick in picks:
                ph = pick.get("home_team", "")
                pa = pick.get("away_team", "")

                m = find_match_in_list(ph, pa, finished_matches)
                if not m or m.get("status") not in ("FINISHED", "post"):
                    pick_results.append("pending")
                    has_pending = True
                    continue

                hs, as_ = int(m.get("home_score", 0)), int(m.get("away_score", 0))
                actual = "HOME" if hs > as_ else ("AWAY" if hs < as_ else "DRAW")

                if pick.get("pick", "").upper() == actual:
                    pick_results.append("won")
                else:
                    pick_results.append("lost")
                    all_correct = False

            if not has_pending:
                self.fs.evaluate_coupon(coupon["id"], pick_results, all_correct)
                self._stats["coupons_evaluated"] += 1
            elif any(r in ("won", "lost") for r in pick_results):
                # Partial update
                self.fs.db.collection("daily_coupons").document(coupon["id"]).update({
                    "pickResults": pick_results,
                })

    # ──────────────────────────────────────
    # STAGE 8: Update source performance
    # ──────────────────────────────────────

    def update_source_performance(self):
        """Calculate and update performance metrics for each prediction source."""
        log.info("── Stage 8: Updating source performance ──")

        # Get all finished matches with results
        finished_mids = []
        for m in self._matches:
            if m.get("status") in ("FINISHED", "post"):
                home = m.get("home_team_name", m.get("home_team", ""))
                away = m.get("away_team_name", m.get("away_team", ""))
                hs = m.get("home_score")
                as_ = m.get("away_score")
                if hs is not None and as_ is not None:
                    mid = match_id(m.get("match_date", ""), home, away)
                    actual = "HOME" if int(hs) > int(as_) else ("AWAY" if int(hs) < int(as_) else "DRAW")
                    finished_mids.append((mid, actual))

        if not finished_mids:
            log.info("  No finished matches to evaluate sources against")
            return

        # Collect predictions per source
        source_results: Dict[str, List[Tuple[Dict, str]]] = {}  # source → [(probs, actual)]
        for mid, actual in finished_mids:
            try:
                preds = self.fs.get_predictions_for_match(mid)
                for p in preds:
                    src = p.get("source", "unknown")
                    probs = p.get("probabilities", {})
                    if src not in source_results:
                        source_results[src] = []
                    source_results[src].append((probs, actual))
            except Exception:
                pass

        # Calculate metrics per source
        for source, results in source_results.items():
            if len(results) < 3:  # Need min 3 predictions to evaluate
                continue

            total = len(results)
            correct = 0
            brier_sum = 0.0
            log_loss_sum = 0.0
            roi_sum = 0.0

            for probs, actual in results:
                predicted = max(probs, key=probs.get)
                if predicted.upper() == actual:
                    correct += 1

                brier_sum += self.fs.brier_score(probs, actual)
                log_loss_sum += self.fs.log_loss_single(probs, actual)

                # ROI: assume flat 1-unit stakes at fair odds (1/prob)
                pred_prob = probs.get(predicted.lower(), 0.33)
                if pred_prob > 0:
                    implied_odds = 1.0 / pred_prob
                    if predicted.upper() == actual:
                        roi_sum += (implied_odds - 1)  # profit
                    else:
                        roi_sum -= 1  # loss

            metrics = {
                "totalPredictions": total,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total > 0 else 0,
                "roi": round(roi_sum / total, 4) if total > 0 else 0,
                "brierScore": round(brier_sum / total, 4) if total > 0 else 0.5,
                "logLoss": round(log_loss_sum / total, 4) if total > 0 else 1.0,
                "weight": 0.1,  # Will be computed below
            }

            self.fs.update_source(source, metrics)
            self._stats["sources_updated"] += 1
            log.info(f"  {source}: accuracy={metrics['accuracy']:.1%}, "
                     f"brier={metrics['brierScore']:.3f}, roi={metrics['roi']:.4f}")

        # Compute normalized weights from inverse Brier Score
        if source_results:
            self._recalculate_weights()

    def _recalculate_weights(self):
        """Recalculate source weights based on inverse Brier Score."""
        snap = self.fs.db.collection("sources").get()
        sources = []
        for doc in snap:
            d = doc.to_dict()
            if d.get("totalPredictions", 0) >= 3:
                bs = d.get("brierScore", 0.5)
                sources.append((doc.id, max(1.0 - bs, 0.01)))  # inverse Brier

        if not sources:
            return

        total_inv = sum(s[1] for s in sources)
        for name, inv_brier in sources:
            weight = round(inv_brier / total_inv, 4)
            self.fs.db.collection("sources").document(name).update({"weight": weight})
            log.info(f"    {name}: weight={weight:.3f}")

    # ──────────────────────────────────────
    # STAGE 9: Write backward-compatible cache
    # ──────────────────────────────────────

    def write_legacy_cache(self):
        """Write to cache/ collection so the existing frontend continues to work."""
        log.info("── Stage 9: Writing legacy cache ──")

        # cache/matches — upcoming matches for dashboard
        today = date.today().strftime("%Y-%m-%d")
        upcoming = []
        for m in self._matches:
            if m.get("status") in ("SCHEDULED", "IN_PLAY", "pre"):
                md = m.get("match_date", "")
                if md[:10] >= today:
                    upcoming.append({
                        "home_team_name": m.get("home_team_name", m.get("home_team", "")),
                        "away_team_name": m.get("away_team_name", m.get("away_team", "")),
                        "home_score": None,
                        "away_score": None,
                        "status": "SCHEDULED",
                        "match_date": md,
                        "league_name": m.get("league_name", ""),
                    })
        self.fs.write_cache("matches", upcoming)

        # cache/ai_predictions — ensemble predictions in old format
        ai_preds = []
        for mid, pred in self._ml_preds.items():
            ens = pred["ensemble"]
            best = max(ens, key=ens.get)
            confidence = round(ens[best] * 100)

            ai_preds.append({
                "matchId": mid,
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": pred["match_date"],
                "kickoff": "",
                "predicted_outcome": best.upper(),
                "confidence": confidence,
                "home_prob": round(ens.get("home", 0.33) * 100),
                "draw_prob": round(ens.get("draw", 0.33) * 100),
                "away_prob": round(ens.get("away", 0.34) * 100),
                "sources": ["ML Ensemble"],
                "consensus": best.upper(),
            })
        self.fs.write_cache("ai_predictions", ai_preds)

        # cache/ml_predictions — odds + predictions in old format
        odds_matches = []
        for mid, pred in self._ml_preds.items():
            od = find_match_in_list(
                pred["home_team"], pred["away_team"], self._odds
            )
            ens = pred["ensemble"]
            best = max(ens, key=ens.get)
            confidence = round(ens[best] * 100)

            ho = od.get("home_odds", 0) if od else 0
            do_ = od.get("draw_odds", 0) if od else 0
            ao = od.get("away_odds", 0) if od else 0

            # Calculate edge
            edge_val = pred.get("edge", {}).get(best, 0)

            odds_matches.append({
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": pred["match_date"],
                "kickoff": "",
                "odds_1": ho if ho else round(100 / max(ens.get("home", 0.33) * 100, 1), 2),
                "odds_x": do_ if do_ else round(100 / max(ens.get("draw", 0.33) * 100, 1), 2),
                "odds_2": ao if ao else round(100 / max(ens.get("away", 0.34) * 100, 1), 2),
                "ai_prediction": best.upper(),
                "ai_confidence": confidence,
                "value_bet": edge_val > 0.03,
                "value_edge": round(edge_val * 100, 1),
            })

        self.fs.write_cache("ml_predictions", {
            "predictions": ai_preds,
            "odds_matches": odds_matches,
        })

        log.info(f"  Wrote {len(upcoming)} matches, {len(ai_preds)} predictions, "
                 f"{len(odds_matches)} odds_matches to cache")

    # ──────────────────────────────────────
    # MAIN RUN
    # ──────────────────────────────────────

    def run_full(self):
        """Full pipeline: fetch → scrape → predict → evaluate → cache."""
        start = datetime.now()
        log.info("╔════════════════════════════════════════╗")
        log.info("║   AIBets Prediction Pipeline v2.0     ║")
        log.info(f"║   {start.strftime('%Y-%m-%d %H:%M:%S')}                  ║")
        log.info("╚════════════════════════════════════════╝")

        # Stage 1: Fetch matches
        self.fetch_matches()

        # Stage 2: Fetch real odds
        self.fetch_odds()

        # Stage 3: Scrape AI sites
        self.scrape_ai_predictions()

        # Stage 4: ML predictions
        self.run_ml_predictions()

        # Stage 5: Meta-features  
        self.compute_meta_features()

        # Stage 6: Daily coupon
        self.build_daily_coupon()

        # Stage 7: Evaluate finished
        self.evaluate_finished()

        # Stage 8: Source performance
        self.update_source_performance()

        # Stage 9: Legacy cache for frontend
        self.write_legacy_cache()

        elapsed = (datetime.now() - start).total_seconds()
        log.info("═══════════════════════════════════════")
        log.info(f"Pipeline complete in {elapsed:.1f}s")
        log.info(f"  Matches:      {self._stats['matches_fetched']}")
        log.info(f"  Odds:         {self._stats['odds_fetched']}")
        log.info(f"  AI Preds:     {self._stats['ai_predictions']}")
        log.info(f"  ML Preds:     {self._stats['ml_predictions']}")
        log.info(f"  Results:      {self._stats['results_saved']}")
        log.info(f"  Coupons:      {self._stats['coupons_evaluated']}")
        log.info(f"  Sources:      {self._stats['sources_updated']}")
        log.info("═══════════════════════════════════════")

        return self._stats

    def run_odds_only(self):
        """Just update odds from Danske Spil."""
        log.info("═ Odds-only update ═")
        self.fetch_matches()
        self.fetch_odds()
        self.write_legacy_cache()

    def run_evaluate_only(self):
        """Just evaluate finished matches."""
        log.info("═ Evaluate-only run ═")
        self.fetch_matches()
        self.evaluate_finished()
        self.update_source_performance()


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AIBets Prediction Pipeline")
    parser.add_argument("--train", action="store_true", help="Train ML models before predicting")
    parser.add_argument("--odds-only", action="store_true", help="Only update odds from Danske Spil")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate finished matches")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = PredictionPipeline()

    if args.train:
        pipeline.train_models()

    if args.odds_only:
        pipeline.run_odds_only()
    elif args.evaluate_only:
        pipeline.run_evaluate_only()
    else:
        pipeline.run_full()


if __name__ == "__main__":
    main()
