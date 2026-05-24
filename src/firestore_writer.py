"""
Firestore Writer — Bridge between Python ML pipeline and Firebase Firestore.

New collection schema (replaces flat cache):
  matches/{matchId}        — Canonical match records with results + closing odds
  predictions/{autoId}     — Per-source per-match predictions with probabilities
  model_outputs/{autoId}   — Meta-model / ensemble final predictions
  sources/{sourceName}     — Source performance metrics + weights
  model_features/{matchId} — Pre-computed ML features for meta-model
  daily_coupons/{date}     — Daily coupon picks (kept from old schema)

Also writes backward-compatible cache/ docs for the existing frontend.
"""

import os
import re
import json
import math
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any

import firebase_admin
from firebase_admin import credentials, firestore

log = logging.getLogger("firestore_writer")


def build_coupon_history_payload(coupons: list) -> dict:
    """Build the frontend coupon_history cache payload from coupon documents."""
    valid_coupons = []
    for data in coupons:
        picks = data.get("picks") or []
        status = data.get("status") or "pending"

        if not picks and status != "skipped":
            continue

        pick_results = data.get("pickResults") or []
        has_results = any(r in ("won", "lost") for r in pick_results)
        if status in ("won", "lost") and not has_results:
            continue

        valid_coupons.append({
            "date": data.get("date"),
            "picks": picks,
            "totalOdds": data.get("totalOdds") or 0,
            "status": status,
            "pickResults": pick_results,
            "reason": data.get("reason"),
            "candidateCount": data.get("candidateCount"),
        })

    won = sum(1 for c in valid_coupons if c["status"] == "won")
    lost = sum(1 for c in valid_coupons if c["status"] == "lost")
    pending = sum(1 for c in valid_coupons if c["status"] == "pending")
    skipped = sum(1 for c in valid_coupons if c["status"] == "skipped")
    decided = won + lost

    total_picks = 0
    correct_picks = 0
    league_map: Dict[str, Dict[str, int]] = {}
    for coupon in valid_coupons:
        pick_results = coupon.get("pickResults") or []
        for idx, pick in enumerate(coupon.get("picks") or []):
            result = pick_results[idx] if idx < len(pick_results) else None
            if result not in ("won", "lost"):
                continue
            total_picks += 1
            if result == "won":
                correct_picks += 1

            league = pick.get("league") or "Ukendt"
            if league not in league_map:
                league_map[league] = {"total": 0, "correct": 0}
            league_map[league]["total"] += 1
            if result == "won":
                league_map[league]["correct"] += 1

    league_stats = [
        {
            "league": league,
            "total": stats["total"],
            "correct": stats["correct"],
            "accuracy": round((stats["correct"] / stats["total"]) * 100) if stats["total"] else 0,
        }
        for league, stats in league_map.items()
    ]

    return {
        "total": len(valid_coupons),
        "won": won,
        "lost": lost,
        "pending": pending,
        "skipped": skipped,
        "winRate": round((won / decided) * 100) if decided else 0,
        "totalPicks": total_picks,
        "correctPicks": correct_picks,
        "pickHitRate": round((correct_picks / total_picks) * 100) if total_picks else 0,
        "leagueStats": league_stats,
        "coupons": valid_coupons,
    }


# ──────────────────────────────────────────
# Team name normalization
# ──────────────────────────────────────────

def _normalize_team(name: str) -> str:
    """Normalize team name for deterministic match IDs."""
    n = name.lower().strip()
    for prefix in ["fc ", "cf ", "ac ", "as ", "ss ", "us ", "sc ", "afc ",
                    "rcd ", "real ", "sporting ", "atletico "]:
        if n.startswith(prefix):
            n = n[len(prefix):]
    for suffix in [" fc", " cf", " sc", " ac", " united", " city"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)]
    n = re.sub(r'[^a-z0-9]', '_', n.strip())
    n = re.sub(r'_+', '_', n).strip('_')
    return n


def match_id(date_str: str, home: str, away: str) -> str:
    """Generate deterministic match ID: {YYYY-MM-DD}_{home}_{away}"""
    d = date_str[:10] if len(date_str) > 10 else date_str
    return f"{d}_{_normalize_team(home)}_{_normalize_team(away)}"


# ──────────────────────────────────────────
# FirestoreWriter
# ──────────────────────────────────────────

class FirestoreWriter:
    def __init__(self):
        if not firebase_admin._apps:
            cred = self._get_credentials()
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        log.info("Firestore client initialized")

    def _get_credentials(self):
        # 1. JSON string in env var (same as TypeScript app)
        key_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
        if key_json:
            try:
                return credentials.Certificate(json.loads(key_json))
            except Exception as e:
                log.warning(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_KEY: {e}")

        # 2. Path to JSON file via standard Google env var
        key_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if key_path and os.path.exists(key_path):
            return credentials.Certificate(key_path)

        # 3. service-account.json in project root
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ["service-account.json", "firebase-key.json", "serviceAccountKey.json"]:
            local_path = os.path.join(root, name)
            if os.path.exists(local_path):
                return credentials.Certificate(local_path)

        raise RuntimeError(
            "No Firebase credentials found.\n"
            "  Set FIREBASE_SERVICE_ACCOUNT_KEY env var (JSON string) or\n"
            "  place service-account.json in project root."
        )

    # ──────────────────────────────────────
    # MATCHES collection
    # ──────────────────────────────────────

    def upsert_match(self, m: dict) -> str:
        """Write or update a match document. Returns match ID."""
        home = m.get("home_team") or m.get("home_team_name") or ""
        away = m.get("away_team") or m.get("away_team_name") or ""
        date_str = m.get("match_date") or m.get("date") or ""
        mid = match_id(date_str, home, away)

        doc: Dict[str, Any] = {
            "homeTeam": home,
            "awayTeam": away,
            "league": m.get("league") or m.get("league_code") or "",
            "leagueName": m.get("league_name") or "",
            "country": m.get("country") or "",
            "kickoff": date_str,
            "status": m.get("status") or "SCHEDULED",
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

        # Add scores if finished
        hs = m.get("home_score")
        as_ = m.get("away_score")
        if hs is not None and as_ is not None:
            try:
                hs, as_ = int(hs), int(as_)
                outcome = "HOME" if hs > as_ else ("AWAY" if hs < as_ else "DRAW")
                doc["result"] = {"homeGoals": hs, "awayGoals": as_, "outcome": outcome}
            except (ValueError, TypeError):
                pass

        self.db.collection("matches").document(mid).set(doc, merge=True)
        return mid

    def update_match_odds(self, mid: str, odds: dict, is_closing: bool = False):
        """Write odds to a match document."""
        home_o = odds.get("home_odds") or odds.get("odds_home") or odds.get("home") or 0
        draw_o = odds.get("draw_odds") or odds.get("odds_draw") or odds.get("draw") or 0
        away_o = odds.get("away_odds") or odds.get("odds_away") or odds.get("away") or 0

        field = "closingOdds" if is_closing else "currentOdds"
        update: Dict[str, Any] = {
            field: {"home": home_o, "draw": draw_o, "away": away_o},
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

        # Margin-removed implied probabilities
        if home_o > 0 and draw_o > 0 and away_o > 0:
            total = 1/home_o + 1/draw_o + 1/away_o
            prob_field = "closingImpliedProb" if is_closing else "currentImpliedProb"
            update[prob_field] = {
                "home": round(1/home_o / total, 4),
                "draw": round(1/draw_o / total, 4),
                "away": round(1/away_o / total, 4),
            }

        self.db.collection("matches").document(mid).update(update)

    def add_odds_snapshot(
        self,
        mid: str,
        source: str,
        odds: dict,
        market: str = "1x2",
        extra: Optional[dict] = None,
    ) -> str:
        """Append an odds snapshot for line movement and CLV tracking."""
        home_o = odds.get("home_odds") or odds.get("odds_home") or odds.get("home") or 0
        draw_o = odds.get("draw_odds") or odds.get("odds_draw") or odds.get("draw") or 0
        away_o = odds.get("away_odds") or odds.get("odds_away") or odds.get("away") or 0
        doc: Dict[str, Any] = {
            "matchId": mid,
            "source": source,
            "market": market,
            "capturedAt": firestore.SERVER_TIMESTAMP,
            "odds": {"home": home_o, "draw": draw_o, "away": away_o},
        }
        if home_o and draw_o and away_o and home_o > 1 and draw_o > 1 and away_o > 1:
            total = 1/home_o + 1/draw_o + 1/away_o
            doc["impliedProbability"] = {
                "home": round(1/home_o / total, 4),
                "draw": round(1/draw_o / total, 4),
                "away": round(1/away_o / total, 4),
            }
        if extra:
            doc["extra"] = extra
        ref = self.db.collection("matches").document(mid).collection("odds_snapshots").add(doc)
        return ref[1].id

    def save_pick_snapshot(self, mid: str, pick: dict) -> str:
        """Persist the exact pick and odds seen when the model made the decision."""
        doc = {
            "matchId": mid,
            "capturedAt": firestore.SERVER_TIMESTAMP,
            **pick,
        }
        ref = self.db.collection("pick_snapshots").add(doc)
        return ref[1].id

    def save_match_context(self, mid: str, context: dict, source: str = "api_football") -> str:
        """Store lineup/injury/player context for a match."""
        doc = {
            "matchId": mid,
            "source": source,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            **context,
        }
        self.db.collection("match_context").document(f"{mid}_{source}").set(doc, merge=True)
        self.db.collection("matches").document(mid).set({
            "contextSummary": context.get("summary", {}),
            "contextUpdatedAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        return f"{mid}_{source}"

    def update_match_result(self, mid: str, home_goals: int, away_goals: int):
        """Update match with final result."""
        outcome = "HOME" if home_goals > away_goals else ("AWAY" if home_goals < away_goals else "DRAW")
        self.db.collection("matches").document(mid).update({
            "result": {"homeGoals": home_goals, "awayGoals": away_goals, "outcome": outcome},
            "status": "FINISHED",
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })

    # ──────────────────────────────────────
    # PREDICTIONS collection
    # ──────────────────────────────────────

    def add_prediction(self, mid: str, source: str,
                       probabilities: Dict[str, float],
                       odds_at_scrape: Dict[str, float] = None,
                       extra: dict = None) -> str:
        """Add a prediction document. One per source per match per scrape-time."""
        doc: Dict[str, Any] = {
            "matchId": mid,
            "source": source,
            "scrapedAt": firestore.SERVER_TIMESTAMP,
            "probabilities": {
                "home": probabilities.get("home", 0.33),
                "draw": probabilities.get("draw", 0.33),
                "away": probabilities.get("away", 0.34),
            },
        }
        if odds_at_scrape:
            doc["oddsAtScrape"] = {
                "home": odds_at_scrape.get("home", 0),
                "draw": odds_at_scrape.get("draw", 0),
                "away": odds_at_scrape.get("away", 0),
            }
        if extra:
            doc.update(extra)  # btts, overUnder25, predictedScore, etc.

        ref = self.db.collection("predictions").add(doc)
        return ref[1].id

    # ──────────────────────────────────────
    # MODEL_OUTPUTS collection
    # ──────────────────────────────────────

    def save_model_output(self, mid: str,
                          final_prob: Dict[str, float],
                          edge: Dict[str, float] = None,
                          recommended_bet: str = None,
                          confidence: float = 0.0,
                          model_version: str = "v1",
                          odds_at_pick: Dict[str, float] = None,
                          calibration: Dict[str, Any] = None,
                          context_summary: Dict[str, Any] = None) -> str:
        """Save meta-model / ensemble output for a match."""
        doc: Dict[str, Any] = {
            "matchId": mid,
            "generatedAt": firestore.SERVER_TIMESTAMP,
            "finalProbability": {
                "home": round(final_prob.get("home", 0.33), 4),
                "draw": round(final_prob.get("draw", 0.33), 4),
                "away": round(final_prob.get("away", 0.34), 4),
            },
            "confidenceScore": round(confidence, 2),
            "modelVersion": model_version,
        }
        if edge:
            doc["edge"] = {k: round(v, 4) for k, v in edge.items()}
        if recommended_bet:
            doc["recommendedBet"] = recommended_bet
        if odds_at_pick:
            doc["oddsAtPick"] = {
                "home": odds_at_pick.get("home", 0),
                "draw": odds_at_pick.get("draw", 0),
                "away": odds_at_pick.get("away", 0),
            }
        if calibration:
            doc["calibration"] = calibration
        if context_summary:
            doc["contextSummary"] = context_summary

        # Use matchId as doc ID so we only keep latest output per match
        self.db.collection("model_outputs").document(mid).set(doc)
        return mid

    # ──────────────────────────────────────
    # SOURCES collection
    # ──────────────────────────────────────

    def update_source(self, source_name: str, metrics: dict):
        """Update source performance metrics."""
        doc = {
            "lastUpdated": firestore.SERVER_TIMESTAMP,
            **metrics,
        }
        self.db.collection("sources").document(source_name).set(doc, merge=True)

    def get_source_weights(self) -> Dict[str, float]:
        """Get current weights for all sources."""
        snap = self.db.collection("sources").get()
        weights = {}
        for doc in snap:
            data = doc.to_dict()
            weights[doc.id] = data.get("weight", 0.1)
        return weights

    # ──────────────────────────────────────
    # MODEL_FEATURES collection
    # ──────────────────────────────────────

    def save_model_features(self, mid: str, features: dict):
        """Save pre-computed features for meta-model training."""
        doc = {
            "matchId": mid,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            **features,
        }
        self.db.collection("model_features").document(mid).set(doc)

    # ──────────────────────────────────────
    # DAILY COUPONS
    # ──────────────────────────────────────

    def save_daily_coupon(self, date_str: str, picks: list, total_odds: float):
        """Save or update daily coupon."""
        ref = self.db.collection("daily_coupons").document(date_str)
        existing = ref.get()
        if existing.exists:
            data = existing.to_dict()
            if data.get("status") not in (None, "pending"):
                return  # Already evaluated

        ref.set({
            "date": date_str,
            "picks": picks,
            "totalOdds": round(total_odds, 2),
            "status": "pending",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })

    def save_no_coupon(self, date_str: str, reason: str, meta: Optional[dict] = None):
        """Record that today's coupon was intentionally skipped."""
        ref = self.db.collection("daily_coupons").document(date_str)
        existing = ref.get()
        if existing.exists:
            data = existing.to_dict()
            if data.get("status") not in (None, "pending", "skipped"):
                return
            if data.get("picks"):
                return

        doc = {
            "date": date_str,
            "picks": [],
            "totalOdds": 0,
            "status": "skipped",
            "reason": reason,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
        if meta:
            doc.update(meta)
        ref.set(doc, merge=True)

    def get_pending_coupons(self) -> list:
        """Get all pending daily coupons."""
        snap = self.db.collection("daily_coupons").where("status", "==", "pending").get()
        return [{"id": d.id, **d.to_dict()} for d in snap]

    def evaluate_coupon(self, date_str: str, pick_results: list,
                        all_correct: bool):
        """Update coupon with evaluation results."""
        self.db.collection("daily_coupons").document(date_str).update({
            "status": "won" if all_correct else "lost",
            "pickResults": pick_results,
            "evaluatedAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        })

    # ──────────────────────────────────────
    # BACKWARD COMPAT: cache/ collection
    # ──────────────────────────────────────

    def write_cache(self, cache_type: str, data: Any):
        """Write to the old cache/ collection for backward compat with frontend."""
        self.db.collection("cache").document(cache_type).set({
            "data": data,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
        })

    def refresh_coupon_history_cache(self, limit: int = 30) -> dict:
        """Refresh cache/coupon_history from daily_coupons."""
        snap = (
            self.db.collection("daily_coupons")
            .order_by("date", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .get()
        )
        payload = build_coupon_history_payload([d.to_dict() for d in snap])
        self.write_cache("coupon_history", payload)
        log.info("Refreshed coupon_history cache with %s coupons", payload["total"])
        return payload

    # ──────────────────────────────────────
    # READS (for evaluation)
    # ──────────────────────────────────────

    def get_match(self, mid: str) -> Optional[dict]:
        """Get a match document."""
        doc = self.db.collection("matches").document(mid).get()
        return doc.to_dict() if doc.exists else None

    def get_matches_by_date(self, date_str: str) -> List[dict]:
        """Get all matches for a specific date."""
        snap = self.db.collection("matches")\
            .where("kickoff", ">=", date_str)\
            .where("kickoff", "<", date_str + "T23:59:59")\
            .get()
        return [{"id": d.id, **d.to_dict()} for d in snap]

    def get_predictions_for_match(self, mid: str) -> List[dict]:
        """Get all predictions for a match."""
        snap = self.db.collection("predictions")\
            .where("matchId", "==", mid)\
            .get()
        return [{"id": d.id, **d.to_dict()} for d in snap]

    def get_all_model_outputs(self, status: str = "SCHEDULED") -> List[dict]:
        """Get all current model outputs."""
        snap = self.db.collection("model_outputs").get()
        return [{"id": d.id, **d.to_dict()} for d in snap]

    def get_finished_matches_without_result(self) -> List[dict]:
        """Get matches that are finished but don't have a result yet."""
        snap = self.db.collection("matches")\
            .where("status", "==", "FINISHED")\
            .get()
        results = []
        for doc in snap:
            data = doc.to_dict()
            if "result" not in data:
                results.append({"id": doc.id, **data})
        return results

    # ──────────────────────────────────────
    # PREDICTION RESULTS (legacy — kept for history page)
    # ──────────────────────────────────────

    def save_prediction_result(self, result: dict) -> bool:
        """Save to prediction_results collection (legacy format for history page)."""
        home = result.get("homeTeam", "")
        away = result.get("awayTeam", "")
        date_str = result.get("matchDate", "")
        doc_id = f"{date_str}_{home}_{away}".replace(" ", "_").replace("/", "_").replace("\\", "_").replace(".", "_").lower()

        ref = self.db.collection("prediction_results").document(doc_id)
        if ref.get().exists:
            return False

        ref.set({**result, "createdAt": firestore.SERVER_TIMESTAMP})
        return True

    # ──────────────────────────────────────
    # BRIER SCORE / EVALUATION HELPERS
    # ──────────────────────────────────────

    @staticmethod
    def brier_score(probs: Dict[str, float], actual: str) -> float:
        """Calculate Brier Score for a single prediction. Lower = better."""
        actual_vec = {"home": 0.0, "draw": 0.0, "away": 0.0}
        actual_vec[actual.lower()] = 1.0
        score = 0.0
        for k in ["home", "draw", "away"]:
            score += (probs.get(k, 0.33) - actual_vec[k]) ** 2
        return score

    @staticmethod
    def log_loss_single(probs: Dict[str, float], actual: str) -> float:
        """Calculate log loss for a single prediction."""
        p = max(probs.get(actual.lower(), 0.33), 1e-15)
        return -math.log(p)

    # ──────────────────────────────────────
    # FEEDBACK LOOP — data for ML retraining
    # ──────────────────────────────────────

    def get_all_prediction_results(self, limit: int = 2000) -> List[Dict]:
        """Download prediction_results from Firestore for ML retraining.

        Returns list of dicts with matchDate, homeTeam, awayTeam, leagueCode,
        homeScore, awayScore, actualOutcome, predictedOutcome, confidence, isCorrect.
        """
        snap = self.db.collection("prediction_results") \
            .order_by("matchDate", direction=firestore.Query.DESCENDING) \
            .limit(limit) \
            .get()
        results = []
        for doc in snap:
            results.append(doc.to_dict())
        log.info(f"Downloaded {len(results)} prediction_results for retraining")
        return results

    def get_source_metrics(self) -> Dict[str, Dict]:
        """Get full metrics (accuracy, brierScore, weight) for all sources."""
        snap = self.db.collection("sources").get()
        metrics = {}
        for doc in snap:
            metrics[doc.id] = doc.to_dict()
        return metrics
