"""
Firestore Writer — Bridge between Python ML pipeline and Firebase Firestore.

New collection schema (replaces flat cache):
  matches/{matchId}        — Canonical match records with results + closing odds
  predictions/{autoId}     — Per-source per-match predictions with probabilities
  model_outputs/{autoId}   — Meta-model / ensemble final predictions
  forecast_results/{id}    — Evaluated international forecasts (never bets)
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
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

import firebase_admin
from firebase_admin import credentials, firestore

from src.public_cache_sync import (
    PublicCacheSyncResult,
    public_cache_document_ids,
    sync_public_cache as publish_public_cache,
    utc_now_iso,
)

log = logging.getLogger("firestore_writer")

VALIDATED_FORECAST_ONLY = "VALIDATED_FORECAST_ONLY"
NON_BETTING_FORECAST_SCOPE = "NON_BETTING_FORECAST"
PUBLIC_CACHE_DOCUMENT_IDS = public_cache_document_ids()


def _normalize_betting_outcome(value: Any) -> Optional[str]:
    """Return the canonical 1X2 betting outcome, or ``None``.

    Keeping this normalization at the persistence boundary prevents callers
    from accidentally turning a diagnostic model winner into an actionable
    pick merely because two outcome-label conventions are in use.
    """
    normalized = str(value or "").strip().upper()
    aliases = {
        "1": "HOME",
        "HOME": "HOME",
        "HOME_WIN": "HOME",
        "X": "DRAW",
        "DRAW": "DRAW",
        "2": "AWAY",
        "AWAY": "AWAY",
        "AWAY_WIN": "AWAY",
    }
    return aliases.get(normalized)


def _verified_pick_odd(value: Any) -> Optional[float]:
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return None
    return odd if math.isfinite(odd) and odd > 1.0 else None


def _forecast_timestamp_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            str(value).replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _validated_forecast_probabilities(raw: Any) -> Optional[Dict[str, float]]:
    if not isinstance(raw, dict):
        return None
    try:
        probabilities = {
            key: float(raw.get(key)) for key in ("home", "draw", "away")
        }
    except (TypeError, ValueError):
        return None
    if any(
        not math.isfinite(value) or value <= 0 or value >= 1
        for value in probabilities.values()
    ):
        return None
    total = sum(probabilities.values())
    if not 0.99 <= total <= 1.01:
        return None
    return {key: value / total for key, value in probabilities.items()}


def build_forecast_history_payload(results: list) -> dict:
    """Build isolated performance history for validated, non-betting forecasts.

    This payload is deliberately incompatible with ``prediction_history`` and
    ``paper_trading``: it has forecast-specific field names and contains no
    odds, stakes, profit or ROI fields.  That makes accidental mixing into the
    betting feedback loop much harder.
    """
    valid_results = []
    for raw in results:
        status = str(raw.get("forecastStatus") or "").strip().upper()
        decision = str(raw.get("decisionStatus") or "").strip().upper()
        scope = str(raw.get("evaluationScope") or "").strip().upper()
        outcome = str(raw.get("forecastOutcome") or "").strip().upper()
        actual = str(raw.get("actualOutcome") or "").strip().upper()
        if (
            status != VALIDATED_FORECAST_ONLY
            or decision != "ABSTAIN"
            or scope != NON_BETTING_FORECAST_SCOPE
            or outcome not in {"HOME", "DRAW", "AWAY"}
            or actual not in {"HOME", "DRAW", "AWAY"}
            or not raw.get("forecastGeneratedAt")
        ):
            continue

        generated_epoch = _forecast_timestamp_epoch(raw.get("forecastGeneratedAt"))
        kickoff_epoch = _forecast_timestamp_epoch(raw.get("matchDate"))
        probabilities = _validated_forecast_probabilities(raw.get("probabilities"))
        try:
            confidence = float(raw.get("forecastConfidence"))
        except (TypeError, ValueError):
            continue
        if (
            generated_epoch is None
            or kickoff_epoch is None
            or generated_epoch > kickoff_epoch
            or probabilities is None
            or not math.isfinite(confidence)
            or confidence < 0
            or confidence > 100
            or outcome != max(probabilities, key=probabilities.get).upper()
        ):
            continue

        actual_vector = {"home": 0.0, "draw": 0.0, "away": 0.0}
        actual_vector[actual.lower()] = 1.0
        brier_score = sum(
            (probabilities[key] - actual_vector[key]) ** 2
            for key in probabilities
        )
        log_loss = -math.log(max(probabilities[actual.lower()], 1e-15))

        # Explicit allow-list: never copy betting fields into this cache.
        result = {
            "matchId": raw.get("matchId"),
            "matchDate": raw.get("matchDate"),
            "homeTeam": raw.get("homeTeam"),
            "awayTeam": raw.get("awayTeam"),
            "leagueCode": raw.get("leagueCode"),
            "homeScore": raw.get("homeScore"),
            "awayScore": raw.get("awayScore"),
            "actualOutcome": actual,
            "forecastOutcome": outcome,
            "forecastConfidence": round(confidence, 1),
            "probabilities": {
                key: round(value, 4) for key, value in probabilities.items()
            },
            "isCorrect": outcome == actual,
            "brierScore": round(brier_score, 4),
            "logLoss": round(log_loss, 4),
            "modelVersion": raw.get("modelVersion"),
            "forecastStatus": VALIDATED_FORECAST_ONLY,
            "decisionStatus": "ABSTAIN",
            "evaluationScope": NON_BETTING_FORECAST_SCOPE,
            "forecastGeneratedAt": raw.get("forecastGeneratedAt"),
        }
        valid_results.append(result)

    total = len(valid_results)
    correct = sum(1 for result in valid_results if result["isCorrect"])
    avg_confidence = (
        sum(result["forecastConfidence"] for result in valid_results) / total
        if total else 0
    )
    avg_brier = (
        sum(result["brierScore"] for result in valid_results) / total
        if total else 0
    )
    avg_log_loss = (
        sum(result["logLoss"] for result in valid_results) / total
        if total else 0
    )
    by_competition: Dict[str, Dict[str, int]] = {}
    for result in valid_results:
        league = result.get("leagueCode") or "Ukendt"
        stats = by_competition.setdefault(league, {"total": 0, "correct": 0})
        stats["total"] += 1
        stats["correct"] += int(result["isCorrect"])

    return {
        "summary": {
            "totalForecasts": total,
            "correctForecasts": correct,
            "forecastAccuracy": round(correct / total * 100, 1) if total else 0,
            "averageConfidence": round(avg_confidence, 1),
            "averageBrierScore": round(avg_brier, 4),
            "averageLogLoss": round(avg_log_loss, 4),
            "byCompetition": [
                {
                    "competition": competition,
                    "total": stats["total"],
                    "correct": stats["correct"],
                    "accuracy": round(stats["correct"] / stats["total"] * 100, 1),
                }
                for competition, stats in sorted(by_competition.items())
            ],
        },
        "results": valid_results,
        "scope": NON_BETTING_FORECAST_SCOPE,
    }


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
            "oddsBasis": data.get("oddsBasis"),
            "oddsSource": data.get("oddsSource"),
            "evaluationMode": data.get("evaluationMode"),
            "eligibleForBetting": data.get("eligibleForBetting") is True,
        })

    won = sum(1 for c in valid_coupons if c["status"] == "won")
    lost = sum(1 for c in valid_coupons if c["status"] == "lost")
    pending = sum(1 for c in valid_coupons if c["status"] == "pending")
    skipped = sum(1 for c in valid_coupons if c["status"] == "skipped")
    coupons_with_picks = [c for c in valid_coupons if c.get("picks")]
    verified_odds_coupons = sum(
        1
        for coupon in coupons_with_picks
        if coupon.get("oddsBasis") == "verified_pre_match_odds"
    )
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
        "verifiedOddsCoupons": verified_odds_coupons,
        "excludedUnverifiedCoupons": len(coupons_with_picks) - verified_odds_coupons,
        "winRate": round((won / decided) * 100) if decided else 0,
        "totalPicks": total_picks,
        "correctPicks": correct_picks,
        "pickHitRate": round((correct_picks / total_picks) * 100) if total_picks else 0,
        "leagueStats": league_stats,
        "coupons": valid_coupons,
    }


def build_source_weights_payload(
    source_metrics: Dict[str, Dict[str, Any]],
    prediction_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Merge real source metrics with evaluated ensemble performance.

    No default accuracy or weight is invented.  Existing source weights come
    directly from Firestore, while the ensemble rows are calculated only from
    persisted, evaluated prediction results.
    """

    payload: Dict[str, Dict[str, Any]] = {}
    for name, raw_metrics in source_metrics.items():
        if not isinstance(raw_metrics, dict):
            continue
        metrics = dict(raw_metrics)
        # Legacy source docs sometimes contain ROI calculated from model-implied
        # odds.  Do not republish it unless its verified price basis is explicit.
        if metrics.get("roiBasis") != "verified_pre_match_odds":
            metrics["roi"] = None
            metrics["roiBets"] = 0
            metrics["roiBasis"] = "unavailable_no_verified_odds"
        try:
            sample_size = int(metrics.get("totalPredictions") or 0)
            brier = float(metrics.get("brierScore"))
        except (TypeError, ValueError):
            sample_size = 0
            brier = float("nan")
        if sample_size < 3 or not math.isfinite(brier):
            metrics.pop("weight", None)
        payload[str(name)] = metrics
    valid_results = [
        result
        for result in prediction_results
        if isinstance(result, dict) and not FirestoreWriter._is_abstention_result(result)
    ]
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for result in valid_results:
        source = str(result.get("source") or "ML Ensemble").strip() or "ML Ensemble"
        grouped.setdefault(source, []).append(result)
    if valid_results:
        grouped["ML Ensemble"] = valid_results

    for source, results in grouped.items():
        total = len(results)
        correct = sum(1 for result in results if bool(result.get("isCorrect")))
        metrics = dict(payload.get(source, {}))
        if source.startswith("ML Ensemble"):
            metrics.pop("weight", None)
        metrics.update({
            "correct": correct,
            "totalPredictions": total,
            "accuracy": round(correct / total, 4) if total else 0,
        })
        payload[source] = metrics

    return payload


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
        self._public_cache_envelopes: Dict[str, Dict[str, Any]] = {}
        self.public_cache_sync_status: Optional[PublicCacheSyncResult] = None
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
                          odds_basis: str = None,
                          odds_source: str = None,
                          calibration: Dict[str, Any] = None,
                          context_summary: Dict[str, Any] = None,
                          decision_status: str = None,
                          decision_reason: str = None,
                          forecast_status: str = None,
                          forecast_outcome: str = None,
                          forecast_confidence: float = None) -> str:
        """Save meta-model / ensemble output for a match."""
        recommended = _normalize_betting_outcome(recommended_bet)
        normalized_decision = str(decision_status or "").strip().upper()
        selected_odd = (
            _verified_pick_odd((odds_at_pick or {}).get(recommended.lower()))
            if recommended
            else None
        )
        actionable = bool(
            recommended
            and normalized_decision == "BET"
            and odds_basis == "verified_pre_match_odds"
            and str(odds_source or "").strip()
            and selected_odd
            and forecast_status != VALIDATED_FORECAST_ONLY
        )
        if normalized_decision == "BET" and not actionable:
            normalized_decision = "ABSTAIN"
            decision_reason = decision_reason or "unverified_or_missing_pick_odds"

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
            "eligibleForBetting": actionable,
        }
        if edge:
            doc["edge"] = {k: round(v, 4) for k, v in edge.items()}
        if actionable:
            doc["recommendedBet"] = recommended
            doc["evaluationMode"] = "forward_only"
        if odds_at_pick:
            doc["oddsAtPick"] = {
                "home": odds_at_pick.get("home", 0),
                "draw": odds_at_pick.get("draw", 0),
                "away": odds_at_pick.get("away", 0),
            }
        if actionable:
            doc["oddsBasis"] = odds_basis
            doc["oddsSource"] = odds_source
        if calibration:
            doc["calibration"] = calibration
        if context_summary:
            doc["contextSummary"] = context_summary
        if normalized_decision:
            doc["decisionStatus"] = normalized_decision
        if decision_reason:
            doc["decisionReason"] = decision_reason
        if forecast_status:
            doc["forecastStatus"] = forecast_status
        if forecast_outcome:
            doc["forecastOutcome"] = forecast_outcome
        if forecast_confidence is not None:
            doc["forecastConfidence"] = round(float(forecast_confidence), 4)
        if forecast_status == VALIDATED_FORECAST_ONLY:
            # A durable trust boundary for every downstream consumer.
            doc["evaluationScope"] = NON_BETTING_FORECAST_SCOPE
            doc["eligibleForBetting"] = False

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
        if not picks:
            raise ValueError("daily_coupon_requires_at_least_one_pick")
        for pick in picks:
            recommendation = _normalize_betting_outcome(pick.get("pick"))
            verified = bool(
                recommendation
                and str(pick.get("decisionStatus") or "").strip().upper() == "BET"
                and str(pick.get("evaluationMode") or "").strip().lower()
                == "forward_only"
                and pick.get("eligibleForBetting") is True
                and pick.get("oddsBasis") == "verified_pre_match_odds"
                and str(pick.get("oddsSource") or "").strip()
                and _verified_pick_odd(pick.get("odds"))
            )
            if not verified:
                raise ValueError("daily_coupon_requires_verified_forward_only_picks")

        odds_sources = sorted({str(pick["oddsSource"]) for pick in picks})
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
            "decisionStatus": "BET",
            "evaluationMode": "forward_only",
            "eligibleForBetting": True,
            "oddsBasis": "verified_pre_match_odds",
            "oddsSource": odds_sources[0] if len(odds_sources) == 1 else odds_sources,
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
        """Write Firestore cache and stage its public Cloudflare envelope."""
        updated_at = utc_now_iso()
        self.db.collection("cache").document(cache_type).set({
            "data": data,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        })
        if cache_type in PUBLIC_CACHE_DOCUMENT_IDS:
            self._public_cache_envelopes[cache_type] = {
                "data": data,
                "updatedAt": updated_at,
            }
        else:
            log.warning(
                "Cache %s was written to Firestore but is not in the public-cache contract",
                cache_type,
            )

    def sync_public_cache(self) -> PublicCacheSyncResult:
        """Publish all cache envelopes staged during this pipeline run once."""
        result = publish_public_cache(dict(self._public_cache_envelopes))
        self.public_cache_sync_status = result
        if result.synced:
            log.info(
                "Public cache synced: %s documents (%s bytes, %s attempt%s)",
                result.cache_count,
                result.byte_count,
                result.attempts,
                "" if result.attempts == 1 else "s",
            )
            self._public_cache_envelopes.clear()
        else:
            log.error(
                "PUBLIC CACHE NOT SYNCED (%s; %s documents). Firestore is current, "
                "but aibets.dk may remain stale until the next successful run.",
                result.reason,
                result.cache_count,
            )
        return result

    def refresh_coupon_history_cache(self, limit: int = 365) -> dict:
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

    def refresh_prediction_history_cache(self, limit: int = 2000) -> dict:
        """Refresh the frontend history cache from evaluated predictions."""
        results = self.get_all_prediction_results(limit=limit)
        total = len(results)
        correct = sum(1 for result in results if result.get("isCorrect"))
        average_confidence = (
            sum(float(result.get("confidence") or 0) for result in results) / total
            if total else 0
        )
        leagues: Dict[str, Dict[str, int]] = {}
        for result in results:
            league = result.get("leagueCode") or "Ukendt"
            stats = leagues.setdefault(league, {"total": 0, "correct": 0})
            stats["total"] += 1
            if result.get("isCorrect"):
                stats["correct"] += 1

        payload = {
            "summary": {
                "totalPredictions": total,
                "correctPredictions": correct,
                "accuracy": round(correct / total * 100) if total else 0,
                "averageConfidence": round(average_confidence),
                "byLeague": [
                    {
                        "league": league,
                        "total": stats["total"],
                        "correct": stats["correct"],
                        "accuracy": round(stats["correct"] / stats["total"] * 100)
                        if stats["total"] else 0,
                    }
                    for league, stats in leagues.items()
                ],
            },
            "results": results,
        }
        self.write_cache("prediction_history", payload)
        log.info("Refreshed prediction_history cache with %s results", total)
        return payload

    def refresh_forecast_history_cache(self, limit: int = 2000) -> dict:
        """Refresh the isolated non-betting forecast history cache."""
        results = self.get_all_forecast_results(limit=limit)
        payload = build_forecast_history_payload(results)
        self.write_cache("forecast_history", payload)
        log.info(
            "Refreshed forecast_history cache with %s non-betting forecasts",
            payload["summary"]["totalForecasts"],
        )
        return payload

    def refresh_paper_trading_cache(
        self,
        stake: float = 100,
        bankroll: float = 10000,
        limit: int = 2000,
    ) -> dict:
        """Recompute paper P&L only from explicitly verified pre-match odds."""
        results = self.get_all_prediction_results(limit=limit)
        total_bets = 0
        total_won = 0
        total_profit = 0.0
        running_profit = 0.0
        by_league: Dict[str, Dict[str, Any]] = {}
        equity_curve = []
        excluded_unverified = 0

        for result in sorted(results, key=lambda item: item.get("matchDate", "")):
            recommendation = _normalize_betting_outcome(result.get("recommendedBet"))
            predicted = _normalize_betting_outcome(result.get("predictedOutcome"))
            odds = _verified_pick_odd(result.get("oddsAtPick"))
            eligible = bool(
                result.get("eligibleForBetting") is True
                and str(result.get("decisionStatus") or "").strip().upper() == "BET"
                and str(result.get("evaluationMode") or "").strip().lower()
                == "forward_only"
                and recommendation
                and recommendation == predicted
                and result.get("oddsBasis") == "verified_pre_match_odds"
                and str(result.get("oddsSource") or "").strip()
                and odds
            )
            if not eligible:
                if any(
                    result.get(field) not in (None, "", 0, 0.0)
                    for field in ("odds", "oddsAtPick", "profit")
                ):
                    excluded_unverified += 1
                continue
            total_bets += 1
            won = bool(result.get("isCorrect"))
            if won:
                total_won += 1
                bet_profit = (odds - 1) * stake
            else:
                bet_profit = -stake
            total_profit += bet_profit
            running_profit += bet_profit

            league = result.get("leagueCode") or "UNK"
            stats = by_league.setdefault(league, {"bets": 0, "won": 0, "profit": 0.0})
            stats["bets"] += 1
            stats["won"] += int(won)
            stats["profit"] += bet_profit
            equity_curve.append({
                "date": result.get("matchDate", ""),
                "profit": round(running_profit),
            })

        league_stats = []
        for league, stats in sorted(
            by_league.items(), key=lambda item: item[1]["profit"], reverse=True
        ):
            staked = stats["bets"] * stake
            league_stats.append({
                "league": league,
                "bets": stats["bets"],
                "won": stats["won"],
                "hitRate": round(stats["won"] / stats["bets"] * 100) if stats["bets"] else 0,
                "profit": round(stats["profit"]),
                "roi": round(stats["profit"] / staked * 100, 1) if staked else 0,
            })

        total_staked = total_bets * stake
        payload = {
            "startingBankroll": bankroll,
            "stakePerBet": stake,
            "totalBets": total_bets,
            "totalWon": total_won,
            "hitRate": round(total_won / total_bets * 100, 1) if total_bets else 0,
            "totalProfit": round(total_profit),
            "totalStaked": total_staked,
            "roi": round(total_profit / total_staked * 100, 1) if total_staked else 0,
            "currentBankroll": bankroll + round(total_profit),
            "byLeague": league_stats,
            "equityCurve": equity_curve[-100:],
            "oddsBasis": "verified_pre_match_odds",
            "excludedUnverifiedBets": excluded_unverified,
        }
        self.write_cache("paper_trading", payload)
        log.info(
            "Refreshed paper_trading cache: %s bets, P&L %+.0f DKK",
            total_bets,
            total_profit,
        )
        return payload

    def refresh_source_weights_cache(self) -> dict:
        """Refresh public source metrics from real sources and evaluations."""
        source_metrics = self.get_source_metrics()
        prediction_results = self.get_all_prediction_results(limit=2000)
        payload = build_source_weights_payload(source_metrics, prediction_results)
        self.write_cache("source_weights", payload)
        log.info(
            "Refreshed source_weights cache with %s evaluated sources/models",
            len(payload),
        )
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

    @staticmethod
    def _is_abstention_result(result: dict) -> bool:
        status = str(
            result.get("decisionStatus") or result.get("decision_status") or ""
        ).strip().upper()
        outcome = str(
            result.get("predictedOutcome") or result.get("predicted_outcome") or ""
        ).strip().upper()
        return status == "ABSTAIN" or outcome == "ABSTAIN"

    def save_prediction_result(self, result: dict) -> bool:
        """Save to prediction_results collection (legacy format for history page)."""
        if self._is_abstention_result(result):
            return False

        normalized = dict(result)
        recommendation = _normalize_betting_outcome(normalized.get("recommendedBet"))
        predicted = _normalize_betting_outcome(normalized.get("predictedOutcome"))
        odds_at_pick = _verified_pick_odd(normalized.get("oddsAtPick"))
        pnl_eligible = bool(
            normalized.get("eligibleForBetting") is True
            and str(normalized.get("decisionStatus") or "").strip().upper() == "BET"
            and str(normalized.get("evaluationMode") or "").strip().lower()
            == "forward_only"
            and recommendation
            and recommendation == predicted
            and normalized.get("oddsBasis") == "verified_pre_match_odds"
            and str(normalized.get("oddsSource") or "").strip()
            and odds_at_pick
        )
        normalized["eligibleForBetting"] = pnl_eligible
        if recommendation:
            normalized["recommendedBet"] = recommendation
        if predicted:
            normalized["predictedOutcome"] = predicted
        if pnl_eligible:
            normalized["oddsAtPick"] = round(odds_at_pick, 2)
            normalized["odds"] = round(odds_at_pick, 2)
            normalized["profit"] = round(
                odds_at_pick - 1.0 if bool(normalized.get("isCorrect")) else -1.0,
                2,
            )
        else:
            # Accuracy/history may still be valid, but legacy or incomplete
            # rows must never manufacture a paper-trading return.
            normalized["oddsAtPick"] = None
            normalized["odds"] = None
            normalized["profit"] = None
            if normalized.get("oddsBasis") != "verified_pre_match_odds":
                normalized["oddsBasis"] = "unavailable_no_verified_odds"

        home = normalized.get("homeTeam", "")
        away = normalized.get("awayTeam", "")
        date_str = normalized.get("matchDate", "")
        doc_id = f"{date_str}_{home}_{away}".replace(" ", "_").replace("/", "_").replace("\\", "_").replace(".", "_").lower()

        ref = self.db.collection("prediction_results").document(doc_id)
        if ref.get().exists:
            return False

        ref.set({**normalized, "createdAt": firestore.SERVER_TIMESTAMP})
        return True

    def save_forecast_result(self, result: dict) -> bool:
        """Persist one pre-match international forecast outside betting data.

        The caller must supply a forecast that was already stored before the
        fixture.  This method does not infer, backfill, or read betting odds.
        """
        status = str(result.get("forecastStatus") or "").strip().upper()
        decision = str(result.get("decisionStatus") or "").strip().upper()
        scope = str(result.get("evaluationScope") or "").strip().upper()
        outcome = str(result.get("forecastOutcome") or "").strip().upper()
        actual = str(result.get("actualOutcome") or "").strip().upper()
        if (
            status != VALIDATED_FORECAST_ONLY
            or decision != "ABSTAIN"
            or scope != NON_BETTING_FORECAST_SCOPE
            or outcome not in {"HOME", "DRAW", "AWAY"}
            or actual not in {"HOME", "DRAW", "AWAY"}
            or not result.get("forecastGeneratedAt")
        ):
            return False

        generated_epoch = _forecast_timestamp_epoch(result.get("forecastGeneratedAt"))
        kickoff_epoch = _forecast_timestamp_epoch(result.get("matchDate"))
        probabilities = _validated_forecast_probabilities(result.get("probabilities"))
        try:
            confidence = float(result.get("forecastConfidence"))
        except (TypeError, ValueError):
            return False
        if (
            generated_epoch is None
            or kickoff_epoch is None
            or generated_epoch > kickoff_epoch
            or probabilities is None
            or not math.isfinite(confidence)
            or confidence < 0
            or confidence > 100
            or outcome != max(probabilities, key=probabilities.get).upper()
        ):
            return False

        actual_vector = {"home": 0.0, "draw": 0.0, "away": 0.0}
        actual_vector[actual.lower()] = 1.0
        brier_score = sum(
            (probabilities[key] - actual_vector[key]) ** 2
            for key in probabilities
        )
        log_loss = -math.log(max(probabilities[actual.lower()], 1e-15))

        mid = str(result.get("matchId") or "").strip()
        if not mid:
            return False
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", mid).lower()
        ref = self.db.collection("forecast_results").document(doc_id)
        if ref.get().exists:
            return False

        # Explicit allow-list prevents odds/profit fields leaking across the
        # forecast/betting boundary even if a caller passes extra properties.
        stored = {
            "matchId": mid,
            "matchDate": result.get("matchDate"),
            "homeTeam": result.get("homeTeam"),
            "awayTeam": result.get("awayTeam"),
            "leagueCode": result.get("leagueCode"),
            "homeScore": result.get("homeScore"),
            "awayScore": result.get("awayScore"),
            "actualOutcome": actual,
            "forecastOutcome": outcome,
            "forecastConfidence": round(confidence, 1),
            "probabilities": {
                key: round(value, 4) for key, value in probabilities.items()
            },
            "isCorrect": outcome == actual,
            "brierScore": round(brier_score, 4),
            "logLoss": round(log_loss, 4),
            "modelVersion": result.get("modelVersion"),
            "forecastStatus": VALIDATED_FORECAST_ONLY,
            "decisionStatus": "ABSTAIN",
            "evaluationScope": NON_BETTING_FORECAST_SCOPE,
            "eligibleForBetting": False,
            "forecastGeneratedAt": result.get("forecastGeneratedAt"),
            "evaluatedAt": firestore.SERVER_TIMESTAMP,
        }
        ref.set(stored)
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
            result = doc.to_dict()
            if not self._is_abstention_result(result):
                results.append(result)
        log.info(f"Downloaded {len(results)} prediction_results for retraining")
        return results

    def get_all_forecast_results(self, limit: int = 2000) -> List[Dict]:
        """Read only evaluated non-betting forecasts for their own cache."""
        snap = self.db.collection("forecast_results") \
            .order_by("matchDate", direction=firestore.Query.DESCENDING) \
            .limit(limit) \
            .get()
        results = []
        for doc in snap:
            result = doc.to_dict()
            if (
                str(result.get("forecastStatus") or "").strip().upper()
                == VALIDATED_FORECAST_ONLY
                and str(result.get("decisionStatus") or "").strip().upper()
                == "ABSTAIN"
                and str(result.get("evaluationScope") or "").strip().upper()
                == NON_BETTING_FORECAST_SCOPE
            ):
                results.append(result)
        return results

    def get_source_metrics(self) -> Dict[str, Dict]:
        """Get full metrics (accuracy, brierScore, weight) for all sources."""
        snap = self.db.collection("sources").get()
        metrics = {}
        for doc in snap:
            metrics[doc.id] = doc.to_dict()
        return metrics
