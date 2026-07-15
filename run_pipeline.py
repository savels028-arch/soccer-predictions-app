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
import time
import numpy as np
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from dateutil import parser as date_parser

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.firestore_writer import (
    FirestoreWriter,
    NON_BETTING_FORECAST_SCOPE,
    VALIDATED_FORECAST_ONLY,
    match_id,
    _normalize_team,
)
from src.api.prediction_scraper import PredictionScraper
from src.api.danske_spil_client import DanskeSpilClient
from src.api.free_football_client import FreeFootballClient
from src.database.db_manager import DatabaseManager
from src.predictions.prediction_engine import PredictionEngine
from src.predictions.feature_engineering import FeatureEngineer, FeatureEngineerV2, EloTracker
from src.predictions.international_model import (
    InternationalModelUnavailable,
    ValidatedInternationalModel,
    try_load_default_international_model,
)
from src.api.data_aggregator import DataAggregator
from config.settings import AB_TEST, DATA_ENRICHMENT, ML_SETTINGS, ML_SETTINGS_V2, PAPER_TRADING

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


# The production ensemble is trained and validated on domestic club leagues.
# International fixtures need a separate national-team model because neutral
# venues, tournament structure and team histories are materially different.
# Keep these fixtures visible, but never let the club model turn its default
# feature vector into a normal prediction or betting recommendation.
UNVALIDATED_INTERNATIONAL_LEAGUES = frozenset({"WC"})
INTERNATIONAL_MODEL_ABSTAIN_REASON = "international_model_not_validated"
INTERNATIONAL_FORECAST_ONLY_REASON = "international_forecast_only_no_odds_validation"


class PublicCacheSyncFailed(RuntimeError):
    """Raised after Firestore writes when the public cache was not published."""


def _model_scope_abstention_reason(league_code: str) -> Optional[str]:
    """Return a stable reason code when the club model is out of scope."""
    code = str(league_code or "").strip().upper()
    if code in UNVALIDATED_INTERNATIONAL_LEAGUES:
        return INTERNATIONAL_MODEL_ABSTAIN_REASON
    return None


def build_model_breakdown(
    predictions: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build a per-fixture breakdown from actual model probability outputs.

    Invalid or incomplete outputs are omitted instead of being replaced with
    neutral/default probabilities.
    """

    breakdown: Dict[str, List[Dict[str, Any]]] = {}
    for mid, prediction in predictions.items():
        models = prediction.get("models")
        if not isinstance(models, dict):
            continue
        rows: List[Dict[str, Any]] = []
        for model_name, raw_probabilities in models.items():
            if not isinstance(raw_probabilities, dict):
                continue
            try:
                probabilities = {
                    outcome: float(raw_probabilities[outcome])
                    for outcome in ("home", "draw", "away")
                }
            except (KeyError, TypeError, ValueError):
                continue
            total = sum(probabilities.values())
            if (
                not all(math.isfinite(value) and value >= 0 for value in probabilities.values())
                or total <= 0
            ):
                continue
            normalized = {
                outcome: value / total for outcome, value in probabilities.items()
            }
            predicted = max(normalized, key=normalized.get)
            source = str(model_name)
            if source in {"xgboost", "neural_network", "random_forest"}:
                source = f"ml_{source}"
            rows.append({
                "source": source,
                "predicted_outcome": predicted.upper(),
                "confidence": round(normalized[predicted] * 100),
                "probabilities": {
                    outcome: round(value, 4)
                    for outcome, value in normalized.items()
                },
            })
        if rows:
            breakdown[str(mid)] = rows
    return breakdown


def _verified_decimal_odd(value: Any) -> float:
    """Return a real decimal market odd, or zero when it is unavailable."""
    try:
        odd = float(value)
    except (TypeError, ValueError):
        return 0.0
    return odd if math.isfinite(odd) and odd > 1.0 else 0.0


def _normalize_actionable_outcome(value: Any) -> Optional[str]:
    """Normalize an explicitly persisted 1X2 recommendation."""
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


def _is_pre_match_fixture(match: Dict[str, Any]) -> bool:
    """Return true only while a fixture kickoff is still in the future."""
    kickoff_epoch = _timestamp_epoch(match.get("match_date") or match.get("date"))
    return kickoff_epoch is not None and kickoff_epoch > datetime.now(timezone.utc).timestamp()


def _build_abstention_prediction(
    match: Dict[str, Any],
    home: str,
    away: str,
    league: str,
    reason: str,
) -> Dict[str, Any]:
    """Build the legacy-compatible shadow record for an abstained fixture."""
    neutral = {"home": 1 / 3, "draw": 1 / 3, "away": 1 / 3}
    return {
        "home_team": home,
        "away_team": away,
        "match_date": match.get("match_date", ""),
        "league": league,
        "ensemble": neutral,
        "raw_ensemble": neutral,
        "edge": {},
        "recommended": None,
        "confidence": 0.0,
        "models": {},
        "calibration": {"applied": False},
        "context_summary": {},
        "decision_status": "ABSTAIN",
        "decision_reason": reason,
    }


def _build_international_forecast_prediction(
    model: Optional[ValidatedInternationalModel],
    match: Dict[str, Any],
    home: str,
    away: str,
    league: str,
) -> Optional[Dict[str, Any]]:
    """Return a non-bet national-team forecast, or ``None`` fail-closed.

    ``decision_status`` deliberately remains ``ABSTAIN``.  Historical source
    data has no pre-match odds, so these calibrated probabilities may be shown
    as a forecast but must stay outside coupons and P&L.
    """
    if model is None:
        return None
    if not _is_pre_match_fixture(match):
        return None
    neutral_raw = match.get("neutral")
    neutral = neutral_raw if isinstance(neutral_raw, bool) else None
    try:
        forecast = model.predict_fixture(
            home,
            away,
            match.get("match_date", ""),
            neutral=neutral,
        )
    except (InternationalModelUnavailable, TypeError, ValueError):
        return None

    prediction = _build_abstention_prediction(
        match,
        home,
        away,
        league,
        INTERNATIONAL_FORECAST_ONLY_REASON,
    )
    probabilities = forecast["probabilities"]
    prediction.update(
        {
            "ensemble": probabilities,
            "raw_ensemble": probabilities,
            "models": {"international_elo": probabilities},
            "calibration": {
                "applied": True,
                "method": "frozen_elo_probability_calibration",
                "holdout_accuracy": model.validation["holdout"]["accuracy"],
                "world_cup_holdout_accuracy": model.validation["world_cup_holdout"]["accuracy"],
            },
            "context_summary": {
                "scope": forecast["decision_scope"],
                "neutral": forecast["neutral"],
                "trainingCutoff": forecast["training_cutoff"],
            },
            "forecast_status": "VALIDATED_FORECAST_ONLY",
            "forecast_outcome": forecast["forecast_outcome"],
            "forecast_confidence": forecast["confidence"],
            "model_version": forecast["model_version"],
        }
    )
    return prediction


def _timestamp_epoch(value: Any) -> Optional[float]:
    """Normalize persisted Firestore/ISO datetimes for a pre-kickoff check."""
    if value is None:
        return None
    try:
        parsed = value if isinstance(value, datetime) else date_parser.parse(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        return None


def _build_finished_forecast_result(
    mid: str,
    match: Dict[str, Any],
    model_output: Dict[str, Any],
    actual: str,
    home_score: int,
    away_score: int,
) -> Optional[Dict[str, Any]]:
    """Evaluate only an already-persisted, validated pre-match forecast.

    No model is loaded or called here. Missing metadata, legacy records and
    outputs created after kickoff fail closed, preventing retroactive forecast
    generation from contaminating the public history.
    """
    if str(model_output.get("forecastStatus") or "").strip().upper() != VALIDATED_FORECAST_ONLY:
        return None
    if str(model_output.get("decisionStatus") or "").strip().upper() != "ABSTAIN":
        return None
    if str(model_output.get("evaluationScope") or "").strip().upper() != NON_BETTING_FORECAST_SCOPE:
        return None

    generated_at = model_output.get("generatedAt")
    kickoff = match.get("match_date") or match.get("date")
    generated_epoch = _timestamp_epoch(generated_at)
    kickoff_epoch = _timestamp_epoch(kickoff)
    if generated_epoch is None or kickoff_epoch is None or generated_epoch > kickoff_epoch:
        return None

    probabilities = model_output.get("finalProbability") or {}
    try:
        probs = {
            key: float(probabilities.get(key))
            for key in ("home", "draw", "away")
        }
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value <= 0 or value >= 1 for value in probs.values()):
        return None
    total = sum(probs.values())
    if not 0.99 <= total <= 1.01:
        return None
    probs = {key: value / total for key, value in probs.items()}

    forecast_outcome = str(model_output.get("forecastOutcome") or "").strip().upper()
    if forecast_outcome not in {"HOME", "DRAW", "AWAY"}:
        return None
    if forecast_outcome != max(probs, key=probs.get).upper():
        return None
    actual = str(actual or "").strip().upper()
    if actual not in {"HOME", "DRAW", "AWAY"}:
        return None

    confidence = model_output.get("forecastConfidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        return None

    actual_vector = {"home": 0.0, "draw": 0.0, "away": 0.0}
    actual_vector[actual.lower()] = 1.0
    brier = sum((probs[key] - actual_vector[key]) ** 2 for key in probs)
    log_loss = -math.log(max(probs[actual.lower()], 1e-15))

    return {
        "matchId": mid,
        "matchDate": kickoff,
        "homeTeam": match.get("home_team_name", match.get("home_team", "")),
        "awayTeam": match.get("away_team_name", match.get("away_team", "")),
        "leagueCode": match.get("league_code", match.get("league", "")),
        "homeScore": int(home_score),
        "awayScore": int(away_score),
        "actualOutcome": actual,
        "forecastOutcome": forecast_outcome,
        "forecastConfidence": round(confidence * 100, 1),
        "probabilities": {key: round(value, 4) for key, value in probs.items()},
        "isCorrect": forecast_outcome == actual,
        "brierScore": round(brier, 4),
        "logLoss": round(log_loss, 4),
        "modelVersion": model_output.get("modelVersion"),
        "forecastStatus": VALIDATED_FORECAST_ONLY,
        "decisionStatus": "ABSTAIN",
        "evaluationScope": NON_BETTING_FORECAST_SCOPE,
        "eligibleForBetting": False,
        "forecastGeneratedAt": generated_at,
    }


def _build_finished_betting_result(
    match: Dict[str, Any],
    model_output: Dict[str, Any],
    match_record: Dict[str, Any],
    actual: str,
    home_score: int,
    away_score: int,
    source: str,
) -> Optional[Dict[str, Any]]:
    """Evaluate the persisted recommendation; never infer one from probabilities."""
    recommendation = _normalize_actionable_outcome(model_output.get("recommendedBet"))
    if (
        str(model_output.get("decisionStatus") or "").strip().upper() != "BET"
        or not recommendation
    ):
        return None

    selected = recommendation.lower()
    odds_at_pick = _verified_decimal_odd(
        (model_output.get("oddsAtPick") or {}).get(selected)
    )
    generated_epoch = _timestamp_epoch(model_output.get("generatedAt"))
    kickoff_epoch = _timestamp_epoch(match.get("match_date") or match.get("date"))
    captured_pre_match = bool(
        generated_epoch is not None
        and kickoff_epoch is not None
        and generated_epoch <= kickoff_epoch
    )
    pnl_eligible = bool(
        model_output.get("eligibleForBetting") is True
        and str(model_output.get("evaluationMode") or "").strip().lower()
        == "forward_only"
        and model_output.get("oddsBasis") == "verified_pre_match_odds"
        and str(model_output.get("oddsSource") or "").strip()
        and odds_at_pick
        and captured_pre_match
    )

    closing_market = (
        match_record.get("closingOdds") or match_record.get("currentOdds") or {}
    )
    closing_odds = _verified_decimal_odd(closing_market.get(selected))
    clv_pct = (
        round((odds_at_pick / closing_odds - 1.0) * 100, 2)
        if pnl_eligible and closing_odds
        else None
    )
    probabilities = model_output.get("finalProbability") or {}
    try:
        confidence = float(probabilities.get(selected)) * 100
    except (TypeError, ValueError):
        confidence = float(model_output.get("confidenceScore") or 0) * 100
    edge_data = model_output.get("edge") or {}
    try:
        edge = round(float(edge_data.get(selected, 0)) * 100, 1)
    except (TypeError, ValueError):
        edge = 0.0
    is_correct = recommendation == str(actual or "").strip().upper()

    return {
        "matchDate": str(match.get("match_date") or match.get("date") or "")[:10],
        "homeTeam": match.get("home_team_name", match.get("home_team", "")),
        "awayTeam": match.get("away_team_name", match.get("away_team", "")),
        "leagueCode": match.get("league_code", match.get("league", "")),
        "homeScore": int(home_score),
        "awayScore": int(away_score),
        "actualOutcome": str(actual or "").strip().upper(),
        "predictedOutcome": recommendation,
        "recommendedBet": recommendation,
        "confidence": round(confidence, 1),
        "source": source,
        "isCorrect": is_correct,
        "hasModelOutput": True,
        "decisionStatus": "BET",
        "evaluationMode": "forward_only" if pnl_eligible else "accuracy_only",
        "eligibleForBetting": pnl_eligible,
        "odds": round(odds_at_pick, 2) if pnl_eligible else None,
        "oddsAtPick": round(odds_at_pick, 2) if pnl_eligible else None,
        "oddsBasis": (
            "verified_pre_match_odds"
            if pnl_eligible
            else "unavailable_no_verified_odds"
        ),
        "oddsSource": model_output.get("oddsSource") if pnl_eligible else None,
        "closingLineValuePct": clv_pct,
        "edge": edge,
        "profit": (
            round(odds_at_pick - 1.0 if is_correct else -1.0, 2)
            if pnl_eligible
            else None
        ),
    }


def _current_season() -> int:
    """Dynamic season: August+ = current year, else previous year."""
    now = date.today()
    return now.year if now.month >= 8 else now.year - 1


def _valid_1x2_odds(odds: Dict[str, float]) -> bool:
    return all(float(odds.get(k) or 0.0) > 1.0 for k in ("home", "draw", "away"))


def _market_implied_probabilities(odds: Dict[str, float]) -> Dict[str, float]:
    if not _valid_1x2_odds(odds):
        return {}
    total = sum(1 / float(odds[k]) for k in ("home", "draw", "away"))
    if total <= 0:
        return {}
    return {k: round((1 / float(odds[k])) / total, 4) for k in ("home", "draw", "away")}


def _normalize_probability_map(probs: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.001, float(probs.get(k, 0.0))) for k in ("home", "draw", "away")}
    total = sum(cleaned.values())
    if total <= 0:
        return {"home": 0.33, "draw": 0.33, "away": 0.34}
    return {k: round(v / total, 4) for k, v in cleaned.items()}


def _calibrate_probs_by_league(
    probs: Dict[str, float],
    priors: Optional[Dict[str, float]],
    strength: float,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Blend model probabilities toward recent league base rates."""
    if not priors or strength <= 0:
        return _normalize_probability_map(probs), {"applied": False}
    strength = min(max(float(strength), 0.0), 0.5)
    calibrated = {
        key: (float(probs.get(key, 0.0)) * (1.0 - strength)) + (float(priors.get(key, 0.0)) * strength)
        for key in ("home", "draw", "away")
    }
    return _normalize_probability_map(calibrated), {
        "applied": True,
        "method": "league_prior_blend",
        "strength": strength,
        "priors": priors,
    }


def _selected_odds(odds: Dict[str, float], pick: Optional[str]) -> float:
    if not pick:
        return 0.0
    return float(odds.get(str(pick).lower()) or 0.0)


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


def _match_from_odds_event(odds_event: Dict[str, Any]) -> Optional[dict]:
    """Create an internal upcoming match from a Kambi/Danske Spil odds event."""
    home = str(odds_event.get("home_team") or "").strip()
    away = str(odds_event.get("away_team") or "").strip()
    if not home or not away:
        return None

    league_code = str(odds_event.get("league_code") or "").strip()
    if not league_code:
        return None
    if league_code not in ML_SETTINGS.get("coupon", {}).get("allowed_leagues", []):
        return None

    raw_date = str(odds_event.get("match_date") or "")
    if not raw_date:
        return None
    try:
        kickoff = date_parser.isoparse(raw_date)
        match_date = kickoff.isoformat()
    except Exception:
        match_date = raw_date

    return {
        "api_id": odds_event.get("event_id"),
        "home_team_name": home,
        "away_team_name": away,
        "home_score": None,
        "away_score": None,
        "status": "SCHEDULED",
        "league_name": odds_event.get("league") or league_code,
        "league_code": league_code,
        "match_date": match_date,
        "source": "danske_spil",
        "home_odds": odds_event.get("home_odds"),
        "draw_odds": odds_event.get("draw_odds"),
        "away_odds": odds_event.get("away_odds"),
        "extra_data": {
            "source": "danske_spil",
            "event_id": odds_event.get("event_id"),
            "deeplink": odds_event.get("deeplink"),
            "over_25_odds": odds_event.get("over_25_odds"),
            "under_25_odds": odds_event.get("under_25_odds"),
            "btts_yes_odds": odds_event.get("btts_yes_odds"),
            "btts_no_odds": odds_event.get("btts_no_odds"),
        },
    }


def _paper_strategy_recommendation(
    league_code: str,
    ensemble: Dict[str, float],
    edge: Dict[str, float],
    odds: Dict[str, float],
    cfg: Dict,
) -> Optional[str]:
    """Return a configured paper-trading pick as home/draw/away, or None."""
    if not cfg.get("enabled", False):
        return None
    allowed_leagues = set(cfg.get("profitable_leagues") or [])
    if allowed_leagues and league_code not in allowed_leagues:
        return None
    if league_code in set(cfg.get("excluded_leagues") or []):
        return None

    style = cfg.get("bet_style", "model_pick")
    if style == "market_underdog":
        valid_odds = {k: v for k, v in odds.items() if v and v > 1.0}
        if not valid_odds:
            return None
        selected = max(valid_odds, key=valid_odds.get)
    elif style == "least_likely":
        selected = min(ensemble, key=ensemble.get)
    else:
        selected = max(ensemble, key=ensemble.get)

    outcome_filter = cfg.get("outcome_filter")
    if outcome_filter and selected != outcome_filter:
        return None

    min_confidence_pct = cfg.get("min_confidence_pct")
    if min_confidence_pct is not None and ensemble.get(selected, 0.0) * 100 < min_confidence_pct:
        return None

    min_edge_pct = cfg.get("min_edge_pct")
    if min_edge_pct is not None:
        if not edge or edge.get(selected, -999.0) * 100 < min_edge_pct:
            return None

    return selected


def _score_value(match: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = match.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def _historical_h2h_coupon_pick(
    home: str,
    away: str,
    h2h: List[Dict[str, Any]],
    odds: Dict[str, float],
    cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return a no-ML H2H coupon pick when exact home-vs-away history dominates."""
    counts = {"home": 0, "draw": 0, "away": 0}

    for match in h2h or []:
        match_home = match.get("home_team_name") or match.get("home_team") or ""
        match_away = match.get("away_team_name") or match.get("away_team") or ""
        home_score = _score_value(match, "home_score", "home_goals")
        away_score = _score_value(match, "away_score", "away_goals")
        if home_score is None or away_score is None:
            continue

        # The validated strategy is directed H2H: same home side, same away side.
        # Reversed fixtures are a separate weaker signal and must not be mixed in.
        if not (fuzzy_match_teams(home, match_home) and fuzzy_match_teams(away, match_away)):
            continue

        if home_score == away_score:
            counts["draw"] += 1
        elif home_score > away_score:
            counts["home"] += 1
        else:
            counts["away"] += 1

    total = sum(counts.values())
    if total < int(cfg.get("min_h2h_matches", 10)):
        return None

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None

    pick, hits = ranked[0]
    confidence = hits / total if total else 0.0
    if confidence * 100 < float(cfg.get("min_h2h_rate_pct", 75.0)):
        return None

    odds_val = float(odds.get(pick) or 0.0)
    if odds_val <= 1.0:
        return None
    odds_min = cfg.get("odds_min")
    odds_max = cfg.get("odds_max")
    if odds_min is not None and odds_val < float(odds_min):
        return None
    if odds_max is not None and odds_val > float(odds_max):
        return None

    edge = confidence - (1.0 / odds_val)
    min_edge_pct = cfg.get("min_edge_pct")
    if min_edge_pct is not None and edge * 100 < float(min_edge_pct):
        return None

    return {
        "pick": pick,
        "confidence": confidence,
        "edge": edge,
        "odds": odds_val,
        "h2h_count": total,
        "h2h_rate_pct": confidence * 100,
    }


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
        self._data_agg = DataAggregator(db_manager=self.db)

        # v1 engine (baseline)
        self.engine = PredictionEngine(
            db_manager=self.db, data_aggregator=self._data_agg,
            config=ML_SETTINGS, suffix="", version_label="v1"
        )
        self.feature_eng = FeatureEngineer()

        # Separate, fail-closed national-team bundle.  Loading verifies the
        # artifact and normalized snapshot checksums plus all frozen holdout
        # gates.  Missing or stale data leaves World Cup fixtures as abstains.
        self.international_model, international_reason = (
            try_load_default_international_model()
        )
        if self.international_model is None:
            log.warning("  International model unavailable: %s", international_reason)
        else:
            log.info(
                "  International forecast-only model loaded (cutoff %s)",
                self.international_model.training_cutoff,
            )

        # v2 engine (challenger) — only if A/B test enabled
        self.ab_enabled = AB_TEST.get("enabled", False)
        if self.ab_enabled:
            self.engine_v2 = PredictionEngine(
                db_manager=self.db, data_aggregator=self._data_agg,
                config=ML_SETTINGS_V2, suffix="_v2", version_label="v2"
            )
            self.feature_eng_v2 = FeatureEngineerV2()
            self.elo_tracker = EloTracker()
            log.info("  A/B test enabled: running v1 + v2 in parallel")
        else:
            self.engine_v2 = None
            self.feature_eng_v2 = None
            self.elo_tracker = None

        self.csv_client = CSVFootballClient() if HAS_CSV else None
        self.api_football = ApiFootballClient() if HAS_API_FOOTBALL else None
        self.flashscore = FlashScoreScraper() if HAS_FLASHSCORE else None

        # Collected data during pipeline run
        self._matches: List[dict] = []
        self._odds: List[dict] = []
        self._ai_preds: List[dict] = []
        self._ml_preds: Dict[str, dict] = {}      # v1 matchId → prediction
        self._ml_preds_v2: Dict[str, dict] = {}   # v2 matchId → prediction

        self._stats = {
            "matches_fetched": 0,
            "odds_fetched": 0,
            "ai_predictions": 0,
            "ml_predictions": 0,
            "ml_predictions_v2": 0,
            "results_saved": 0,
            "results_saved_v2": 0,
            "coupons_evaluated": 0,
            "sources_updated": 0,
            "odds_snapshots": 0,
            "pick_snapshots": 0,
            "match_contexts": 0,
        }
        self._stats_v2 = {
            "ml_predictions": 0,
            "results_saved": 0,
        }

    # ──────────────────────────────────────
    # STAGE 1: Fetch matches from ESPN
    # ──────────────────────────────────────

    # ESPN league slugs
    ESPN_LEAGUES = ["WC", "PL", "PD", "BL1", "SA", "FL1", "CL", "DED", "PPL"]

    def fetch_matches(self) -> List[dict]:
        """Fetch today's + upcoming + recent matches from ESPN/TheSportsDB."""
        log.info("── Stage 1: Fetching matches ──")
        all_matches = []

        # ESPN: past 2 days + today + 3 days ahead, per league
        for delta in [-2, -1, 0, 1, 2, 3]:
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
    # STAGE 1B: Enrich match context
    # ──────────────────────────────────────

    def enrich_match_context(self):
        """Fetch lineups, injuries, player ratings and xG/event summaries when available."""
        if not DATA_ENRICHMENT.get("enabled", True):
            return
        if not DATA_ENRICHMENT.get("api_football_context", True):
            return
        if not self.api_football or not getattr(self.api_football, "api_key", ""):
            log.info("── Stage 1B: Match context skipped — API_FOOTBALL_KEY not set ──")
            return

        log.info("── Stage 1B: Fetching lineup/injury/player context ──")
        max_matches = int(DATA_ENRICHMENT.get("max_context_matches_per_run", 12) or 0)
        if max_matches <= 0:
            return

        enriched = 0
        upcoming = [
            m for m in self._matches
            if m.get("status") in ("SCHEDULED", "IN_PLAY", "pre", "LIVE")
        ]
        for m in upcoming[:max_matches]:
            fixture_id = m.get("api_id")
            extra = m.get("extra_data") or {}
            if isinstance(extra, str):
                try:
                    extra = json.loads(extra)
                except Exception:
                    extra = {}
            home_team_id = extra.get("home_team_id") or m.get("home_team_id")
            away_team_id = extra.get("away_team_id") or m.get("away_team_id")

            if not fixture_id:
                continue

            home = m.get("home_team_name", m.get("home_team", ""))
            away = m.get("away_team_name", m.get("away_team", ""))
            date_str = m.get("match_date", "")
            mid = match_id(date_str, home, away)
            try:
                context = self.api_football.get_fixture_context(
                    int(fixture_id),
                    int(home_team_id) if home_team_id else None,
                    int(away_team_id) if away_team_id else None,
                )
                if not context:
                    continue
                self.db.upsert_match_context(mid, "api_football", date_str, home, away, context)
                self.fs.save_match_context(mid, context, source="api_football")
                enriched += 1
            except Exception as e:
                log.debug(f"  Context fetch failed for {home} vs {away}: {e}")

        self._stats["match_contexts"] = enriched
        log.info(f"  Enriched {enriched} matches with context")

    def _context_summary_for_match(self, mid: str) -> Dict[str, Any]:
        context = self.db.get_match_context(mid)
        if not context:
            return {}
        return {
            "homeMissingPlayers": context.get("home_missing_players", 0),
            "awayMissingPlayers": context.get("away_missing_players", 0),
            "homeLineupPlayers": context.get("home_lineup_players", 0),
            "awayLineupPlayers": context.get("away_lineup_players", 0),
            "homePlayerRatingAvg": context.get("home_player_rating_avg"),
            "awayPlayerRatingAvg": context.get("away_player_rating_avg"),
            "homeXg": context.get("home_xg"),
            "awayXg": context.get("away_xg"),
        }

    def _store_odds_snapshot(self, mid: str, match: Dict[str, Any], source: str,
                             odds: Dict[str, float], extra: Optional[Dict[str, Any]] = None):
        if not _valid_1x2_odds(odds):
            return
        home = match.get("home_team_name", match.get("home_team", ""))
        away = match.get("away_team_name", match.get("away_team", ""))
        date_str = match.get("match_date", "")
        try:
            self.db.save_odds_snapshot(mid, date_str, home, away, source, odds, extra=extra)
        except Exception as e:
            log.debug(f"  Local odds snapshot failed for {home} vs {away}: {e}")
        try:
            self.fs.add_odds_snapshot(mid, source, odds, extra=extra)
        except Exception as e:
            log.debug(f"  Firestore odds snapshot failed for {home} vs {away}: {e}")
        self._stats["odds_snapshots"] += 1

    def _league_calibration_for_match(
        self,
        league_code: str,
        match_date: str,
        probs: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not DATA_ENRICHMENT.get("league_calibration_enabled", True):
            return _normalize_probability_map(probs), {"applied": False}
        priors = self.db.get_league_outcome_priors(
            league_code,
            before_date=match_date,
            min_matches=int(DATA_ENRICHMENT.get("league_calibration_min_matches", 80)),
            limit=int(DATA_ENRICHMENT.get("league_calibration_history_limit", 600)),
        )
        return _calibrate_probs_by_league(
            probs,
            priors,
            float(DATA_ENRICHMENT.get("league_calibration_strength", 0.15)),
        )

    def _save_pick_snapshot(self, mid: str, pred: Dict[str, Any], source: str,
                            pick: str, odds_map: Dict[str, float], probability: float,
                            edge: float, model_version: str, strategy: str,
                            extra: Optional[Dict[str, Any]] = None):
        odds_at_pick = _selected_odds(odds_map, pick)
        if odds_at_pick <= 1.0:
            return
        try:
            self.db.save_pick_snapshot(
                mid,
                pred.get("match_date", ""),
                pred.get("home_team", ""),
                pred.get("away_team", ""),
                source,
                pick,
                odds_at_pick,
                probability=probability,
                edge=edge,
                model_version=model_version,
                strategy=strategy,
                extra=extra,
            )
        except Exception as e:
            log.debug(f"  Local pick snapshot failed for {mid}: {e}")
        try:
            self.fs.save_pick_snapshot(mid, {
                "source": source,
                "modelVersion": model_version,
                "strategy": strategy,
                "pick": pick.upper(),
                "probability": round(probability, 4),
                "edge": round(edge, 4),
                "oddsAtPick": round(odds_at_pick, 2),
                "matchDate": pred.get("match_date", ""),
                "homeTeam": pred.get("home_team", ""),
                "awayTeam": pred.get("away_team", ""),
                "extra": extra or {},
            })
        except Exception as e:
            log.debug(f"  Firestore pick snapshot failed for {mid}: {e}")
        self._stats["pick_snapshots"] += 1

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

            # Match odds to our matches and write to Firestore. If fixture feeds
            # are empty, promote valid Kambi events to first-class upcoming matches.
            matched = 0
            odds_created_matches = 0
            for od in odds:
                oh = od.get("home_team", "")
                oa = od.get("away_team", "")
                
                # Find matching match
                m = find_match_in_list(oh, oa, self._matches)
                if not m:
                    m = _match_from_odds_event(od)
                    if not m:
                        continue
                    mid = match_id(
                        m.get("match_date", ""),
                        m.get("home_team_name", ""),
                        m.get("away_team_name", ""),
                    )
                    if not find_match_in_list(
                        m.get("home_team_name", ""),
                        m.get("away_team_name", ""),
                        self._matches,
                    ):
                        self._matches.append(m)
                        odds_created_matches += 1
                        try:
                            self.fs.upsert_match(m)
                        except Exception as e:
                            log.debug(f"  Failed to upsert Kambi fixture {oh} vs {oa}: {e}")
                        try:
                            self.db.upsert_match(m)
                        except Exception as e:
                            log.debug(f"  Failed to store Kambi fixture locally {oh} vs {oa}: {e}")

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
                        odds_map = {
                            "home": od.get("home_odds", 0),
                            "draw": od.get("draw_odds", 0),
                            "away": od.get("away_odds", 0),
                        }
                        self._store_odds_snapshot(
                            mid,
                            m,
                            "danske_spil",
                            odds_map,
                            extra={
                                "deeplink": od.get("deeplink", ""),
                                "over25": od.get("over_25_odds"),
                                "under25": od.get("under_25_odds"),
                                "bttsYes": od.get("btts_yes_odds"),
                                "bttsNo": od.get("btts_no_odds"),
                            },
                        )

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
            if odds_created_matches:
                self._stats["matches_fetched"] = len(self._matches)
                log.info(f"  Created {odds_created_matches} playable fixtures from Kambi odds")

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
                                    self._store_odds_snapshot(
                                        mid,
                                        m,
                                        "flashscore_average",
                                        {"home": ho, "draw": do_, "away": ao},
                                        extra={"flashscoreId": fs_id},
                                    )
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
                if not oas:
                    try:
                        site_home_odds = float(pred.get("odds_home") or 0)
                        site_draw_odds = float(pred.get("odds_draw") or 0)
                        site_away_odds = float(pred.get("odds_away") or 0)
                    except (TypeError, ValueError):
                        site_home_odds = site_draw_odds = site_away_odds = 0.0
                    if site_home_odds > 1 and site_draw_odds > 1 and site_away_odds > 1:
                        oas = {
                            "home": site_home_odds,
                            "draw": site_draw_odds,
                            "away": site_away_odds,
                        }

                extra = {}
                if pred.get("btts"):
                    extra["btts"] = pred["btts"]
                if pred.get("over_under_25"):
                    extra["overUnder25"] = pred["over_under_25"]
                if pred.get("predicted_score"):
                    extra["predictedScore"] = pred["predicted_score"]
                if pred.get("value_bet"):
                    extra["valueBet"] = True
                if pred.get("value_bet_market"):
                    extra["valueBetMarket"] = pred["value_bet_market"]

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
        """Train ML models on historical data + Firestore prediction_results (feedback loop)."""
        log.info("── Training ML models (with feedback loop) ──")
        try:
            # Download accumulated prediction results from Firestore for retraining
            feedback_matches = []
            try:
                feedback_matches = self.fs.get_all_prediction_results(limit=2000)
                log.info(f"  Loaded {len(feedback_matches)} prediction_results for feedback")
            except Exception as e:
                log.warning(f"  Could not load prediction_results for feedback: {e}")

            league_codes = ["PL", "PD", "BL1", "SA", "FL1", "DED", "PPL"]
            cb = lambda t, msg: log.info(f"  [{t}] {msg}")

            # --- v1 training ---
            results = self.engine.train_models(
                league_codes=league_codes,
                callback=cb,
                extra_matches=feedback_matches,
            )
            log.info(f"  v1 Training results: {results}")

            # --- v2 training (A/B) ---
            results_v2 = {}
            if self.ab_enabled and self.engine_v2:
                log.info("── Training v2 models (A/B challenger) ──")
                try:
                    results_v2 = self.engine_v2.train_models(
                        league_codes=league_codes,
                        callback=lambda t, msg: log.info(f"  [v2|{t}] {msg}"),
                        extra_matches=feedback_matches,
                    )
                    log.info(f"  v2 Training results: {results_v2}")
                except Exception as e:
                    log.error(f"  v2 training failed (non-fatal): {e}")
                    traceback.print_exc()

            return results
        except Exception as e:
            log.error(f"  Training failed: {e}")
            traceback.print_exc()
            return {}

    def _get_dynamic_weights(self) -> Dict[str, float]:
        """Get ensemble weights based on recent model accuracy from Firestore sources.

        Falls back to defaults if no source data is available.
        """
        if not hasattr(self, '_dynamic_weights_cache'):
            defaults = {"xgboost": 0.40, "neural_network": 0.35, "random_forest": 0.25}
            try:
                metrics = self.fs.get_source_metrics()
                # Map source names to model names
                name_map = {
                    "ml_xgboost": "xgboost",
                    "ml_neural_network": "neural_network",
                    "ml_random_forest": "random_forest",
                }
                raw_weights = {}
                for src_name, model_name in name_map.items():
                    src = metrics.get(src_name, {})
                    acc = src.get("accuracy", 0)
                    brier = src.get("brierScore", 0.5)
                    total = src.get("totalPredictions", 0)
                    if total >= 5 and acc > 0:
                        # Weight = inverse Brier * accuracy (rewards calibration + correctness)
                        raw_weights[model_name] = max((1.0 - brier) * acc, 0.01)
                    else:
                        raw_weights[model_name] = defaults.get(model_name, 0.25)

                # Normalize to sum to 1
                total_w = sum(raw_weights.values())
                if total_w > 0:
                    self._dynamic_weights_cache = {k: round(v / total_w, 4) for k, v in raw_weights.items()}
                else:
                    self._dynamic_weights_cache = defaults

                log.info(f"  Dynamic ensemble weights: {self._dynamic_weights_cache}")
            except Exception as e:
                log.warning(f"  Could not compute dynamic weights: {e}. Using defaults.")
                self._dynamic_weights_cache = defaults

        return self._dynamic_weights_cache

    def _weighted_ensemble(self, model_results: Dict[str, Dict]) -> Dict[str, float]:
        """Compute weighted average ensemble from model results."""
        weights = self._get_dynamic_weights()
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
        return ensemble

    def _should_retrain(self) -> bool:
        """Check if models should be retrained (weekly or if model files are stale)."""
        try:
            from pathlib import Path
            model_file = Path("data/models/xgboost_model.pkl")
            if not model_file.exists():
                log.info("  No model file found — will retrain")
                return True
            # Retrain if model file is older than 7 days
            import time
            age_days = (time.time() - model_file.stat().st_mtime) / 86400
            if age_days > 7:
                log.info(f"  Models are {age_days:.1f} days old — will retrain")
                return True
            return False
        except Exception:
            return False

    def run_ml_predictions(self) -> Dict[str, dict]:
        """Run ML ensemble on all upcoming matches."""
        log.info("── Stage 4: Running ML predictions ──")

        upcoming = [m for m in self._matches
                    if m.get("status") in ("SCHEDULED", "IN_PLAY", "pre")]
        has_international = any(
            _model_scope_abstention_reason(m.get("league_code", m.get("league", "")))
            for m in upcoming
        )
        has_club_fixtures = any(
            not _model_scope_abstention_reason(
                m.get("league_code", m.get("league", ""))
            )
            for m in upcoming
        )

        if not self.engine.is_trained and has_club_fixtures:
            log.warning("  ML models not trained. Attempting to retrain...")
            try:
                results = self.train_models()
                if not results or not self.engine.is_trained:
                    log.warning("  Club retraining failed; international fixtures remain available.")
                    if not has_international:
                        return {}
                else:
                    log.info("  Retraining successful, continuing with predictions.")
            except Exception as e:
                log.error(f"  Retraining failed: {e}")
                if not has_international:
                    return {}

        log.info(f"  Running predictions for {len(upcoming)} upcoming matches")

        ml_preds = {}
        for m in upcoming:
            home = m.get("home_team_name", m.get("home_team", ""))
            away = m.get("away_team_name", m.get("away_team", ""))
            mid = match_id(m.get("match_date", ""), home, away)

            try:
                # Get odds for this match (from Danske Spil) — pass None when missing
                od = find_match_in_list(home, away, self._odds)
                home_odds = od.get("home_odds") if od else None
                draw_odds = od.get("draw_odds") if od else None
                away_odds = od.get("away_odds") if od else None

                # Get team stats — dynamic season
                league = m.get("league_code", m.get("league", "PL"))
                abstain_reason = _model_scope_abstention_reason(league)
                if abstain_reason:
                    abstention = _build_international_forecast_prediction(
                        getattr(self, "international_model", None),
                        m,
                        home,
                        away,
                        league,
                    )
                    if abstention is None:
                        abstention = _build_abstention_prediction(
                            m, home, away, league, abstain_reason
                        )
                    odds_map = {
                        "home": home_odds or 0.0,
                        "draw": draw_odds or 0.0,
                        "away": away_odds or 0.0,
                    }
                    forecast_only = (
                        abstention.get("forecast_status") == "VALIDATED_FORECAST_ONLY"
                    )
                    if _is_pre_match_fixture(m):
                        self.fs.save_model_output(
                            mid,
                            abstention["ensemble"],
                            confidence=0.0,
                            model_version=abstention.get(
                                "model_version", "international_shadow_abstain"
                            ),
                            odds_at_pick=odds_map,
                            calibration=abstention.get("calibration") if forecast_only else None,
                            context_summary=abstention.get("context_summary") if forecast_only else None,
                            decision_status="ABSTAIN",
                            decision_reason=abstention["decision_reason"],
                            forecast_status=abstention.get("forecast_status"),
                            forecast_outcome=abstention.get("forecast_outcome"),
                            forecast_confidence=abstention.get("forecast_confidence"),
                        )
                    else:
                        log.info(
                            "  Not persisting an international output after kickoff: %s vs %s",
                            home,
                            away,
                        )
                    ml_preds[mid] = abstention
                    log.info(
                        "  Abstaining for %s vs %s (%s): %s",
                        home,
                        away,
                        league,
                        abstention["decision_reason"],
                    )
                    continue

                if not self.engine.is_trained:
                    log.warning("  Skipping club fixture because the club model is unavailable")
                    continue

                current_season = _current_season()
                home_stats = self.db.compute_team_stats_from_matches(home, league, current_season) or {}
                away_stats = self.db.compute_team_stats_from_matches(away, league, current_season) or {}

                # Get H2H
                h2h = self.db.get_h2h(home, away) or []

                # Q3: Gather AI-site predictions for this match as ML features
                ai_for_match = []
                for ai_pred in self._ai_preds:
                    ai_h = ai_pred.get("home_team", "")
                    ai_a = ai_pred.get("away_team", "")
                    if fuzzy_match_teams(home, ai_h) and fuzzy_match_teams(away, ai_a):
                        h_pct = ai_pred.get("home_win_pct", 0)
                        d_pct = ai_pred.get("draw_pct", 0)
                        a_pct = ai_pred.get("away_win_pct", 0)
                        total = (h_pct or 0) + (d_pct or 0) + (a_pct or 0)
                        if total > 0:
                            ai_for_match.append({
                                "home": (h_pct or 0) / total,
                                "draw": (d_pct or 0) / total,
                                "away": (a_pct or 0) / total,
                            })

                # Build features (with AI consensus)
                features = self.feature_eng.build_match_features(
                    home_stats, away_stats, h2h,
                    home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds,
                    ai_predictions=ai_for_match if ai_for_match else None,
                )

                # NaN guard for prediction features
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

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

                # Ensemble: dynamic weights from source performance (feedback loop)
                weights = self._get_dynamic_weights()
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

                raw_ensemble = _normalize_probability_map(ensemble)
                ensemble, calibration_meta = self._league_calibration_for_match(
                    league,
                    m.get("match_date", ""),
                    raw_ensemble,
                )
                context_summary = self._context_summary_for_match(mid)

                # Calculate edge vs market
                edge = {}
                odds_map = {
                    "home": home_odds or 0.0,
                    "draw": draw_odds or 0.0,
                    "away": away_odds or 0.0,
                }
                fair = _market_implied_probabilities(odds_map)
                if fair:
                    edge = {k: round(ensemble[k] - fair[k], 4) for k in ensemble}

                verified_odds_basis = None
                verified_odds_source = None
                if (
                    od
                    and str(od.get("source") or "").strip().lower() == "danske_spil"
                    and _is_pre_match_fixture(m)
                    and _valid_1x2_odds(odds_map)
                ):
                    verified_odds_basis = "verified_pre_match_odds"
                    verified_odds_source = "danske_spil"

                # A recommendation is actionable only with a verified price
                # captured before kickoff. The probability winner remains a
                # diagnostic forecast and must not silently replace the pick.
                best_outcome = max(ensemble, key=ensemble.get)
                confidence = ensemble[best_outcome]
                recommended_key = None
                if verified_odds_basis:
                    recommended_key = _paper_strategy_recommendation(
                        league,
                        ensemble,
                        edge,
                        odds_map,
                        PAPER_TRADING,
                    )
                recommended = recommended_key.upper() if recommended_key else None
                if recommended_key:
                    confidence = ensemble.get(recommended_key, confidence)
                decision_status = "BET" if recommended_key else "ABSTAIN"
                decision_reason = (
                    None
                    if recommended_key
                    else (
                        "no_verified_pre_match_odds"
                        if PAPER_TRADING.get("enabled", False) and not verified_odds_basis
                        else "no_promoted_strategy"
                    )
                )

                # Save model output
                self.fs.save_model_output(
                    mid, ensemble, edge if edge else None,
                    recommended, confidence,
                    model_version=f"ensemble_v1_{len(model_results)}models",
                    odds_at_pick=odds_map,
                    odds_basis=verified_odds_basis,
                    odds_source=verified_odds_source,
                    calibration=calibration_meta,
                    context_summary=context_summary,
                    decision_status=decision_status,
                    decision_reason=decision_reason,
                )

                if recommended_key:
                    self._save_pick_snapshot(
                        mid,
                        {
                            "home_team": home,
                            "away_team": away,
                            "match_date": m.get("match_date", ""),
                        },
                        "ML Ensemble v1",
                        recommended_key,
                        odds_map,
                        ensemble.get(recommended_key, 0.0),
                        edge.get(recommended_key, 0.0) if edge else 0.0,
                        f"ensemble_v1_{len(model_results)}models",
                        PAPER_TRADING.get("bet_style", "paper_single"),
                        extra={
                            "league": league,
                            "calibration": calibration_meta,
                            "contextSummary": context_summary,
                        },
                    )

                ml_preds[mid] = {
                    "home_team": home,
                    "away_team": away,
                    "match_date": m.get("match_date", ""),
                    "league": m.get("league_code", m.get("league", "")),
                    "ensemble": ensemble,
                    "raw_ensemble": raw_ensemble,
                    "edge": edge,
                    "recommended": recommended,
                    "confidence": confidence,
                    "models": model_results,
                    "calibration": calibration_meta,
                    "context_summary": context_summary,
                    "decision_status": decision_status,
                    "decision_reason": decision_reason,
                    "eligible_for_betting": bool(recommended_key and verified_odds_basis),
                    "evaluation_mode": "forward_only" if recommended_key else None,
                    "odds_at_pick": odds_map,
                    "odds_basis": verified_odds_basis,
                    "odds_source": verified_odds_source,
                }

            except Exception as e:
                log.error(f"  ML prediction failed for {home} vs {away}: {e}")
                traceback.print_exc()

        self._ml_preds = ml_preds
        self._stats["ml_predictions"] = len(ml_preds)
        log.info(f"  Generated {len(ml_preds)} ML v1 predictions")

        # ── v2 predictions (A/B challenger) ──
        if self.ab_enabled and self.engine_v2 and self.engine_v2.is_trained:
            log.info("── Running v2 ML predictions (A/B) ──")
            ml_preds_v2 = {}
            for m in upcoming:
                home = m.get("home_team_name", m.get("home_team", ""))
                away = m.get("away_team_name", m.get("away_team", ""))
                mid = match_id(m.get("match_date", ""), home, away)
                try:
                    od = find_match_in_list(home, away, self._odds)
                    home_odds = od.get("home_odds") if od else None
                    draw_odds = od.get("draw_odds") if od else None
                    away_odds = od.get("away_odds") if od else None

                    league = m.get("league_code", m.get("league", "PL"))
                    abstain_reason = _model_scope_abstention_reason(league)
                    if abstain_reason:
                        abstention_v2 = _build_international_forecast_prediction(
                            getattr(self, "international_model", None),
                            m,
                            home,
                            away,
                            league,
                        )
                        if abstention_v2 is None:
                            abstention_v2 = _build_abstention_prediction(
                                m, home, away, league, abstain_reason
                            )
                        odds_map_v2 = {
                            "home": home_odds or 0.0,
                            "draw": draw_odds or 0.0,
                            "away": away_odds or 0.0,
                        }
                        forecast_only_v2 = (
                            abstention_v2.get("forecast_status")
                            == "VALIDATED_FORECAST_ONLY"
                        )
                        if _is_pre_match_fixture(m):
                            self.fs.save_model_output(
                                f"{mid}_v2",
                                abstention_v2["ensemble"],
                                confidence=0.0,
                                model_version=abstention_v2.get(
                                    "model_version", "international_shadow_abstain_v2"
                                ),
                                odds_at_pick=odds_map_v2,
                                calibration=(
                                    abstention_v2.get("calibration")
                                    if forecast_only_v2 else None
                                ),
                                context_summary=(
                                    abstention_v2.get("context_summary")
                                    if forecast_only_v2 else None
                                ),
                                decision_status="ABSTAIN",
                                decision_reason=abstention_v2["decision_reason"],
                                forecast_status=abstention_v2.get("forecast_status"),
                                forecast_outcome=abstention_v2.get("forecast_outcome"),
                                forecast_confidence=abstention_v2.get("forecast_confidence"),
                            )
                        ml_preds_v2[mid] = abstention_v2
                        continue

                    current_season = _current_season()
                    home_stats = self.db.compute_team_stats_from_matches(home, league, current_season) or {}
                    away_stats = self.db.compute_team_stats_from_matches(away, league, current_season) or {}
                    h2h = self.db.get_h2h(home, away) or []

                    # v2 extra data
                    home_matches = self.db.get_team_matches(home, limit=20)
                    away_matches = self.db.get_team_matches(away, limit=20)
                    home_form_list = FeatureEngineerV2.compute_form_list(home_matches, home)
                    away_form_list = FeatureEngineerV2.compute_form_list(away_matches, away)
                    home_extra = FeatureEngineerV2.compute_csv_extra_averages(home_matches, home)
                    away_extra = FeatureEngineerV2.compute_csv_extra_averages(away_matches, away)
                    match_date = m.get("match_date", "")
                    home_days_rest = FeatureEngineerV2.compute_days_since_last(home_matches, home, match_date)
                    away_days_rest = FeatureEngineerV2.compute_days_since_last(away_matches, away, match_date)
                    home_goals_avg = FeatureEngineerV2.compute_recent_goals_avg(home_matches, home)
                    away_goals_avg = FeatureEngineerV2.compute_recent_goals_avg(away_matches, away)

                    # AI predictions
                    ai_for_match = []
                    for ai_pred in self._ai_preds:
                        ai_h = ai_pred.get("home_team", "")
                        ai_a = ai_pred.get("away_team", "")
                        if fuzzy_match_teams(home, ai_h) and fuzzy_match_teams(away, ai_a):
                            h_pct = ai_pred.get("home_win_pct", 0)
                            d_pct = ai_pred.get("draw_pct", 0)
                            a_pct = ai_pred.get("away_win_pct", 0)
                            total = (h_pct or 0) + (d_pct or 0) + (a_pct or 0)
                            if total > 0:
                                ai_for_match.append({
                                    "home": (h_pct or 0) / total,
                                    "draw": (d_pct or 0) / total,
                                    "away": (a_pct or 0) / total,
                                })

                    home_stats["team_name"] = home
                    away_stats["team_name"] = away
                    features_v2 = FeatureEngineerV2.build_match_features_v2(
                        home_stats, away_stats, h2h,
                        home_odds=home_odds, draw_odds=draw_odds, away_odds=away_odds,
                        ai_predictions=ai_for_match if ai_for_match else None,
                        elo_tracker=self.engine_v2.elo_tracker,
                        home_form_list=home_form_list, away_form_list=away_form_list,
                        home_extra=home_extra, away_extra=away_extra,
                        home_days_rest=home_days_rest, away_days_rest=away_days_rest,
                        home_recent_goals_avg=home_goals_avg,
                        away_recent_goals_avg=away_goals_avg,
                        is_training=False,
                        league_code=league,
                        matchday=m.get("matchday", 0),
                        total_matchdays=38,
                        match_datetime=match_date,
                        home_sos=FeatureEngineerV2.compute_sos(home_matches, home, self.engine_v2.elo_tracker),
                        away_sos=FeatureEngineerV2.compute_sos(away_matches, away, self.engine_v2.elo_tracker),
                    )

                    # NaN guard for prediction features
                    if np.any(np.isnan(features_v2)) or np.any(np.isinf(features_v2)):
                        features_v2 = np.nan_to_num(features_v2, nan=0.0, posinf=0.0, neginf=0.0)

                    # Run v2 models
                    model_results_v2 = {}
                    for model_name, model in self.engine_v2.models.items():
                        if model.is_trained:
                            try:
                                proba = model.predict_proba(features_v2.reshape(1, -1))
                                if proba is not None and len(proba) > 0:
                                    p = proba[0]
                                    model_results_v2[model_name] = {
                                        "home": float(p[0]),
                                        "draw": float(p[1]),
                                        "away": float(p[2]),
                                    }
                            except Exception as e:
                                log.debug(f"    v2 Model {model_name} failed for {home} vs {away}: {e}")

                    if not model_results_v2:
                        continue

                    # Stacking ensemble or weighted average
                    if self.engine_v2.stacking and self.engine_v2.stacking.is_trained:
                        stacking_probs = self.engine_v2.stacking.predict_proba(features_v2.reshape(1, -1))
                        if stacking_probs is not None:
                            # Convert ndarray → dict
                            p = stacking_probs[0] if stacking_probs.ndim > 1 else stacking_probs
                            ensemble_v2 = {"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])}
                        else:
                            ensemble_v2 = self._weighted_ensemble(model_results_v2)
                    else:
                        ensemble_v2 = self._weighted_ensemble(model_results_v2)

                    # Blend Poisson probabilities into ensemble (10% weight)
                    poisson_probs = None
                    try:
                        h_att = home_stats.get("avg_goals_scored", 1.3)
                        h_def = home_stats.get("avg_goals_conceded", 1.1)
                        a_att = away_stats.get("avg_goals_scored", 1.2)
                        a_def = away_stats.get("avg_goals_conceded", 1.2)
                        h_exp, a_exp = self.engine_v2.poisson.predict_score(h_att, h_def, a_att, a_def)
                        poisson_probs = self.engine_v2.poisson.match_outcome_probs(h_exp, a_exp)
                        if poisson_probs:
                            poisson_w = 0.10
                            ml_w = 1.0 - poisson_w
                            ensemble_v2 = {
                                "home": ensemble_v2["home"] * ml_w + poisson_probs.get("home_win", 0.33) * poisson_w,
                                "draw": ensemble_v2["draw"] * ml_w + poisson_probs.get("draw", 0.33) * poisson_w,
                                "away": ensemble_v2["away"] * ml_w + poisson_probs.get("away_win", 0.33) * poisson_w,
                            }
                    except Exception as e:
                        log.debug(f"    Poisson blend skipped: {e}")

                    raw_ensemble_v2 = _normalize_probability_map(ensemble_v2)
                    ensemble_v2, calibration_meta_v2 = self._league_calibration_for_match(
                        league,
                        match_date,
                        raw_ensemble_v2,
                    )
                    context_summary_v2 = self._context_summary_for_match(mid)

                    # Edge calculation
                    edge_v2 = {}
                    odds_map_v2 = {
                        "home": home_odds or 0.0,
                        "draw": draw_odds or 0.0,
                        "away": away_odds or 0.0,
                    }
                    fair_v2 = _market_implied_probabilities(odds_map_v2)
                    if fair_v2:
                        edge_v2 = {k: round(ensemble_v2[k] - fair_v2[k], 4) for k in ensemble_v2}

                    verified_odds_basis_v2 = None
                    verified_odds_source_v2 = None
                    if (
                        od
                        and str(od.get("source") or "").strip().lower() == "danske_spil"
                        and _is_pre_match_fixture(m)
                        and _valid_1x2_odds(odds_map_v2)
                    ):
                        verified_odds_basis_v2 = "verified_pre_match_odds"
                        verified_odds_source_v2 = "danske_spil"

                    best_outcome = max(ensemble_v2, key=ensemble_v2.get)
                    confidence_v2 = ensemble_v2[best_outcome]
                    recommended_v2 = None
                    if (
                        verified_odds_basis_v2
                        and PAPER_TRADING.get("enabled", False)
                        and edge_v2
                    ):
                        best_edge_outcome = max(edge_v2, key=edge_v2.get)
                        if edge_v2[best_edge_outcome] > 0.03 and ensemble_v2[best_edge_outcome] > 0.50:
                            recommended_v2 = best_edge_outcome.upper()
                            confidence_v2 = ensemble_v2[best_edge_outcome]
                    decision_status_v2 = "BET" if recommended_v2 else "ABSTAIN"
                    decision_reason_v2 = (
                        None
                        if recommended_v2
                        else (
                            "no_verified_pre_match_odds"
                            if PAPER_TRADING.get("enabled", False)
                            and not verified_odds_basis_v2
                            else "no_promoted_strategy"
                        )
                    )

                    # Save v2 model output (with _v2 suffix)
                    self.fs.save_model_output(
                        f"{mid}_v2", ensemble_v2, edge_v2 if edge_v2 else None,
                        recommended_v2, confidence_v2,
                        model_version=f"ensemble_v2_{len(model_results_v2)}models",
                        odds_at_pick=odds_map_v2,
                        odds_basis=verified_odds_basis_v2,
                        odds_source=verified_odds_source_v2,
                        calibration=calibration_meta_v2,
                        context_summary=context_summary_v2,
                        decision_status=decision_status_v2,
                        decision_reason=decision_reason_v2,
                    )

                    if recommended_v2:
                        self._save_pick_snapshot(
                            mid,
                            {
                                "home_team": home,
                                "away_team": away,
                                "match_date": match_date,
                            },
                            "ML Ensemble v2",
                            recommended_v2,
                            odds_map_v2,
                            ensemble_v2.get(recommended_v2.lower(), confidence_v2),
                            edge_v2.get(recommended_v2.lower(), 0.0) if edge_v2 else 0.0,
                            f"ensemble_v2_{len(model_results_v2)}models",
                            "v2_edge_filter",
                            extra={
                                "league": league,
                                "calibration": calibration_meta_v2,
                                "contextSummary": context_summary_v2,
                            },
                        )

                    # Compute Poisson predicted score + BTTS/O-U for display
                    poisson_score_str = ""
                    btts_prob = 0.0
                    over25_prob = 0.0
                    try:
                        h_att = home_stats.get("avg_goals_scored", 1.3)
                        h_def = home_stats.get("avg_goals_conceded", 1.1)
                        a_att = away_stats.get("avg_goals_scored", 1.2)
                        a_def = away_stats.get("avg_goals_conceded", 1.2)
                        h_exp, a_exp = self.engine_v2.poisson.predict_score(h_att, h_def, a_att, a_def)
                        poisson_score_str = f"{h_exp}-{a_exp}"
                        # BTTS: P(home>0) * P(away>0) using Poisson
                        from math import exp as m_exp
                        btts_prob = round((1 - m_exp(-h_exp)) * (1 - m_exp(-a_exp)), 4)
                        # O/U 2.5: P(total > 2.5) = 1 - P(total <= 2)
                        total_exp = h_exp + a_exp
                        from math import factorial
                        p_under = sum(
                            (total_exp ** k) * m_exp(-total_exp) / factorial(k)
                            for k in range(3)
                        )
                        over25_prob = round(1 - p_under, 4)
                    except Exception:
                        pass

                    ml_preds_v2[mid] = {
                        "home_team": home,
                        "away_team": away,
                        "match_date": match_date,
                        "league": league,
                        "ensemble": ensemble_v2,
                        "raw_ensemble": raw_ensemble_v2,
                        "edge": edge_v2,
                        "recommended": recommended_v2,
                        "confidence": confidence_v2,
                        "models": model_results_v2,
                        "calibration": calibration_meta_v2,
                        "context_summary": context_summary_v2,
                        "poisson_score": poisson_score_str,
                        "btts_prob": btts_prob,
                        "over25_prob": over25_prob,
                        "decision_status": decision_status_v2,
                        "decision_reason": decision_reason_v2,
                        "eligible_for_betting": bool(
                            recommended_v2 and verified_odds_basis_v2
                        ),
                        "evaluation_mode": "forward_only" if recommended_v2 else None,
                        "odds_at_pick": odds_map_v2,
                        "odds_basis": verified_odds_basis_v2,
                        "odds_source": verified_odds_source_v2,
                    }
                except Exception as e:
                    log.error(f"  v2 prediction failed for {home} vs {away}: {e}")
                    traceback.print_exc()

            self._ml_preds_v2 = ml_preds_v2
            self._stats_v2["ml_predictions"] = len(ml_preds_v2)
            log.info(f"  Generated {len(ml_preds_v2)} ML v2 predictions")

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
        """Build today's coupon only from a strategy that passed promotion gates.

        The current configuration deliberately abstains.  A future promoted
        strategy must persist its exact recommendation and verified at-pick
        bookmaker odds before this method can create an actionable coupon.
        """
        log.info("── Stage 6: Building daily coupon ──")
        today = date.today().strftime("%Y-%m-%d")

        # Use v2 predictions if A/B is enabled and v2 produced results, else v1
        source_preds = self._ml_preds
        version_label = "v1"
        if self.ab_enabled and self._ml_preds_v2:
            source_preds = self._ml_preds_v2
            version_label = "v2"
            log.info(f"  Using {version_label} predictions for coupon ({len(source_preds)} available)")

        # Coupon quality settings (v1 now has its own coupon config too)
        if version_label == "v2":
            coupon_cfg = ML_SETTINGS_V2.get("coupon", {})
        else:
            coupon_cfg = ML_SETTINGS.get("coupon", {})
        min_edge_pct = coupon_cfg.get("min_edge_pct", 5.0)
        min_confidence_pct = coupon_cfg.get("min_confidence_pct", 40.0)
        min_picks = coupon_cfg.get("min_picks", 2)
        max_picks = coupon_cfg.get("max_picks", 6)
        max_per_league = coupon_cfg.get("max_per_league", 2)
        skip_disagreement = coupon_cfg.get("skip_high_disagreement", False)
        sort_by = coupon_cfg.get("sort_by", "edge_x_confidence")
        allowed_leagues = set(coupon_cfg.get("allowed_leagues", []))
        coupon_strategy = coupon_cfg.get("strategy", "walk_forward_value_coupon")
        if coupon_strategy == "disabled_no_promoted_strategy":
            self.fs.save_no_coupon(today, "no_promoted_strategy", {
                "candidateCount": 0,
                "selectedCount": 0,
                "version": version_label,
                "strategy": coupon_strategy,
            })
            self.fs.refresh_coupon_history_cache()
            log.info("  Coupon skipped: no strategy has passed promotion gates")
            return

        # Exclude leagues with negative backtest ROI (e.g. Championship)
        excluded_leagues = set(PAPER_TRADING.get("excluded_leagues", []))

        # Collect candidates: matches with ML predictions + Danske Spil odds
        candidates = []
        skipped_league = 0
        for mid, pred in source_preds.items():
            # Only the saved strategy recommendation is a coupon candidate.
            # ``max(ensemble)`` is a forecast diagnostic, not a bet.
            recommended = _normalize_actionable_outcome(pred.get("recommended"))
            if (
                str(pred.get("decision_status") or "").strip().upper() != "BET"
                or not recommended
                or pred.get("eligible_for_betting") is not True
                or pred.get("evaluation_mode") != "forward_only"
                or pred.get("odds_basis") != "verified_pre_match_odds"
                or not str(pred.get("odds_source") or "").strip()
            ):
                continue

            # Skip excluded leagues (negative ROI in backtest)
            league_code = pred.get("league", "")
            if allowed_leagues and league_code not in allowed_leagues:
                skipped_league += 1
                continue
            if league_code in excluded_leagues:
                skipped_league += 1
                continue

            # Coupon settlement uses the exact, verified price snapshot that
            # accompanied the persisted recommendation, never a later market
            # lookup or a model-implied price.
            odds_map = pred.get("odds_at_pick") or {}
            h2h_meta = None
            best_outcome = recommended.lower()
            if coupon_strategy == "historical_h2h_coupon":
                h2h_limit = max(50, int(coupon_cfg.get("min_h2h_matches", 10)) * 4)
                h2h = self.db.get_h2h(
                    pred["home_team"],
                    pred["away_team"],
                    limit=h2h_limit,
                    before_date=pred.get("match_date"),
                ) or []
                h2h_meta = _historical_h2h_coupon_pick(
                    pred["home_team"],
                    pred["away_team"],
                    h2h,
                    odds_map,
                    coupon_cfg,
                )
                if not h2h_meta:
                    continue
                # H2H may filter/score a promoted pick, but it cannot replace
                # the outcome that was persisted by the prediction strategy.
                if h2h_meta["pick"] != best_outcome:
                    continue
                odds_val = h2h_meta["odds"]
                edge_val = h2h_meta["edge"]
                selection_probability = h2h_meta["confidence"]
                confidence_pct = selection_probability * 100
            else:
                odds_val = odds_map.get(best_outcome, 0)
                edge_val = pred.get("edge", {}).get(best_outcome, 0)
                selection_probability = pred["ensemble"].get(best_outcome, 0.33)
                confidence_pct = selection_probability * 100

            if odds_val <= 1.0:
                continue

            # ── Quality filters ──
            if min_edge_pct is not None and edge_val * 100 < min_edge_pct:
                continue
            if confidence_pct < min_confidence_pct:
                continue

            # Skip if high disagreement between models (optional)
            if skip_disagreement and pred.get("models"):
                model_picks = [max(mp, key=mp.get) for mp in pred["models"].values()]
                if len(set(model_picks)) > 1:
                    continue  # models disagree — skip

            # Kelly criterion: f* = (bp - q) / b where b=odds-1, p=model_prob, q=1-p
            model_prob = selection_probability
            b = odds_val - 1.0
            kelly_fraction = 0.0
            if b > 0 and model_prob > 0:
                kelly_fraction = max(0.0, (b * model_prob - (1 - model_prob)) / b)
            kelly_fraction = min(kelly_fraction, 0.25)  # cap at 25% Kelly

            candidates.append({
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": pred["match_date"],
                "kickoff": "",
                "pick": best_outcome.upper(),
                "odds": round(odds_val, 2),
                "confidence": round(confidence_pct, 1),
                "edge": round(edge_val * 100, 1),
                "ev_score": round(edge_val * selection_probability, 6),
                "kelly_fraction": round(kelly_fraction, 4),
                "version": version_label,
                "strategy": coupon_strategy,
                "decisionStatus": "BET",
                "evaluationMode": "forward_only",
                "eligibleForBetting": True,
                "oddsBasis": "verified_pre_match_odds",
                "oddsSource": pred.get("odds_source"),
            })
            if h2h_meta:
                candidates[-1].update({
                    "h2h_count": h2h_meta["h2h_count"],
                    "h2h_rate": round(h2h_meta["h2h_rate_pct"], 1),
                })

        # ── Sort by configured strategy ── highest quality first
        if sort_by == "confidence":
            candidates.sort(key=lambda c: (c["confidence"], c["edge"], c["odds"]), reverse=True)
        elif sort_by == "edge":
            candidates.sort(key=lambda c: (c["edge"], c["confidence"], c["odds"]), reverse=True)
        else:
            candidates.sort(key=lambda c: (c["ev_score"], c["confidence"], c["edge"]), reverse=True)

        # ── Max per league diversification ──
        picks = []
        league_counts = {}
        for c in candidates:
            lg = c.get("league", "UNK")
            if league_counts.get(lg, 0) >= max_per_league:
                continue
            picks.append(c)
            league_counts[lg] = league_counts.get(lg, 0) + 1
            if len(picks) >= max_picks:
                break

        if len(picks) < min_picks:
            log.info(
                f"  Only {len(picks)} quality picks — skipping coupon instead of "
                "relaxing edge/confidence filters"
            )
            self.fs.save_no_coupon(today, "not_enough_quality_picks", {
                "candidateCount": len(candidates),
                "selectedCount": len(picks),
                "minPicks": min_picks,
                "minEdgePct": min_edge_pct,
                "minConfidencePct": min_confidence_pct,
                "version": version_label,
                "strategy": coupon_strategy,
            })
            self.fs.refresh_coupon_history_cache()
            return

        if picks:
            total_odds = 1.0
            for p in picks:
                total_odds *= p["odds"]
                pick_mid = match_id(p.get("match_date", ""), p.get("home_team", ""), p.get("away_team", ""))
                self._save_pick_snapshot(
                    pick_mid,
                    {
                        "home_team": p.get("home_team", ""),
                        "away_team": p.get("away_team", ""),
                        "match_date": p.get("match_date", ""),
                    },
                    "Daily Coupon",
                    str(p.get("pick", "")).lower(),
                    {
                        str(p.get("pick", "")).lower(): p.get("odds", 0),
                        "home": p.get("odds", 0) if p.get("pick") == "HOME" else 0,
                        "draw": p.get("odds", 0) if p.get("pick") == "DRAW" else 0,
                        "away": p.get("odds", 0) if p.get("pick") == "AWAY" else 0,
                    },
                    float(p.get("confidence", 0.0)) / 100,
                    float(p.get("edge", 0.0)) / 100,
                    version_label,
                    p.get("strategy", coupon_strategy),
                    extra={"couponDate": today, "totalOddsBeforeLeg": round(total_odds, 2)},
                )
            self.fs.save_daily_coupon(today, picks, total_odds)
            self.fs.refresh_coupon_history_cache()
            log.info(f"  Daily coupon ({version_label}): {len(picks)} picks, total odds: {total_odds:.2f}")
            if skipped_league:
                log.info(f"  Skipped {skipped_league} predictions from excluded leagues")
        else:
            self.fs.save_no_coupon(today, "no_suitable_picks", {
                "candidateCount": len(candidates),
                "selectedCount": 0,
                "minPicks": min_picks,
                "version": version_label,
                "strategy": coupon_strategy,
            })
            self.fs.refresh_coupon_history_cache()
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
        results_saved_v2 = 0
        forecast_results_saved = 0
        quota_errors = 0
        for m in finished:
            if quota_errors >= 2:
                log.warning("  Skipping remaining evaluations due to quota limits")
                break
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

            # ── Evaluate v1 ──
            try:
                model_out = self.fs.get_match(mid)
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    quota_errors += 1
                continue
            if model_out:
                mo_doc = self.fs.db.collection("model_outputs").document(mid).get()
                if mo_doc.exists:
                    mo_data = mo_doc.to_dict()
                    forecast_result = _build_finished_forecast_result(
                        mid,
                        m,
                        mo_data,
                        actual,
                        hs,
                        as_,
                    )
                    if forecast_result and self.fs.save_forecast_result(forecast_result):
                        forecast_results_saved += 1
                    result_doc = _build_finished_betting_result(
                        m,
                        mo_data,
                        model_out,
                        actual,
                        hs,
                        as_,
                        "ML Ensemble v1",
                    )
                    if result_doc:
                        saved = self.fs.save_prediction_result(result_doc)
                        if saved:
                            results_saved += 1

            # ── Evaluate v2 (A/B) ──
            if self.ab_enabled:
                try:
                    mo_doc_v2 = self.fs.db.collection("model_outputs").document(f"{mid}_v2").get()
                    if mo_doc_v2.exists:
                        mo_data_v2 = mo_doc_v2.to_dict()
                        result_v2 = _build_finished_betting_result(
                            m,
                            mo_data_v2,
                            model_out,
                            actual,
                            hs,
                            as_,
                            "ML Ensemble v2",
                        )
                        if result_v2:
                            saved_v2 = self.fs.save_prediction_result(result_v2)
                            if saved_v2:
                                results_saved_v2 += 1
                except Exception as e:
                    log.debug(f"  v2 evaluation error for {mid}: {e}")

        self._stats["results_saved"] = results_saved
        self._stats["forecast_results_saved"] = forecast_results_saved
        if self.ab_enabled:
            self._stats_v2["results_saved"] = results_saved_v2
            log.info(f"  Saved {results_saved} v1 + {results_saved_v2} v2 prediction results")
        else:
            log.info(f"  Saved {results_saved} new prediction results")
        log.info(
            "  Saved %s new non-betting forecast results",
            forecast_results_saved,
        )

        # Evaluate pending coupons
        try:
            self._evaluate_coupons(finished)
            self.fs.refresh_coupon_history_cache()
            self.fs.refresh_prediction_history_cache()
            self.fs.refresh_paper_trading_cache(
                stake=PAPER_TRADING.get("stake_per_bet", 100),
                bankroll=PAPER_TRADING.get("starting_bankroll", 10000),
            )
        except Exception as e:
            if "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                log.warning("  Skipping coupon evaluation — Firestore quota exhausted")
            else:
                raise

        # Forecast history is intentionally refreshed independently from all
        # betting/coupon caches so a failure cannot affect either data path.
        try:
            self.fs.refresh_forecast_history_cache()
        except Exception as e:
            if "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                log.warning("  Skipping forecast history refresh — Firestore quota exhausted")
            else:
                log.warning("  Forecast history cache refresh failed: %s", e)

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
        source_results: Dict[str, List[Tuple[Dict, str, Dict, str]]] = {}
        for mid, actual in finished_mids:
            try:
                preds = self.fs.get_predictions_for_match(mid)
                for p in preds:
                    src = p.get("source", "unknown")
                    probs = p.get("probabilities", {})
                    if src not in source_results:
                        source_results[src] = []
                    source_results[src].append((
                        probs,
                        actual,
                        p.get("oddsAtScrape") if isinstance(p.get("oddsAtScrape"), dict) else {},
                        str(p.get("oddsBasis") or ""),
                    ))
            except Exception as e:
                if "429" in str(e) or "Quota" in str(e):
                    log.warning("  Quota hit in Stage 8 — using partial data")
                    break

        # Calculate metrics per source
        for source, results in source_results.items():
            if len(results) < 3:  # Need min 3 predictions to evaluate
                continue

            total = len(results)
            correct = 0
            brier_sum = 0.0
            log_loss_sum = 0.0
            roi_sum = 0.0
            roi_bets = 0

            for probs, actual, odds_at_scrape, odds_basis in results:
                predicted = max(probs, key=probs.get)
                if predicted.upper() == actual:
                    correct += 1

                brier_sum += self.fs.brier_score(probs, actual)
                log_loss_sum += self.fs.log_loss_single(probs, actual)

                # ROI is published only when a verified pre-match market price
                # was stored with the source prediction.  Model-implied/fair
                # odds are not bookmaker prices and must never be substituted.
                actual_odds = (
                    _verified_decimal_odd(odds_at_scrape.get(predicted))
                    if odds_basis == "verified_pre_match_odds"
                    else 0.0
                )
                if actual_odds:
                    roi_bets += 1
                    roi_sum += actual_odds - 1 if predicted.upper() == actual else -1

            metrics = {
                "totalPredictions": total,
                "correct": correct,
                "accuracy": round(correct / total, 4) if total > 0 else 0,
                "roi": round(roi_sum / roi_bets, 4) if roi_bets else None,
                "roiBets": roi_bets,
                "roiBasis": "verified_pre_match_odds" if roi_bets else "unavailable_no_verified_odds",
                "brierScore": round(brier_sum / total, 4) if total > 0 else 0.5,
                "logLoss": round(log_loss_sum / total, 4) if total > 0 else 1.0,
            }

            self.fs.update_source(source, metrics)
            self._stats["sources_updated"] += 1
            roi_label = f"{metrics['roi']:.4f}" if metrics["roi"] is not None else "n/a"
            log.info(f"  {source}: accuracy={metrics['accuracy']:.1%}, "
                     f"brier={metrics['brierScore']:.3f}, roi={roi_label}")

        # Compute normalized weights from inverse Brier Score
        if source_results:
            try:
                self._recalculate_weights()
            except Exception as e:
                if "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    log.warning("  Skipping weight recalculation — Firestore quota exhausted")
                else:
                    raise

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

    def write_legacy_cache(self, preserve_predictions: bool = False):
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
            recommended = _normalize_actionable_outcome(pred.get("recommended"))
            is_actionable = bool(
                recommended
                and str(pred.get("decision_status") or "").strip().upper() == "BET"
            )
            selected = recommended.lower() if recommended else None
            confidence = round(ens.get(selected, 0) * 100) if is_actionable else 0
            outcome = recommended if is_actionable else "ABSTAIN"

            ai_preds.append({
                "matchId": mid,
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": pred["match_date"],
                "kickoff": "",
                "predicted_outcome": outcome,
                "confidence": confidence,
                "home_prob": round(ens.get("home", 0.33) * 100),
                "draw_prob": round(ens.get("draw", 0.33) * 100),
                "away_prob": round(ens.get("away", 0.34) * 100),
                "sources": ["ML Ensemble"] if is_actionable else [],
                "consensus": outcome,
                "decision_status": pred.get("decision_status"),
                "abstain_reason": pred.get("decision_reason"),
                "forecast_status": pred.get("forecast_status"),
                "forecast_outcome": pred.get("forecast_outcome"),
                "forecast_confidence": (
                    round(float(pred.get("forecast_confidence", 0.0)) * 100)
                    if pred.get("forecast_status") == "VALIDATED_FORECAST_ONLY"
                    else None
                ),
            })
        if not preserve_predictions:
            self.fs.write_cache("ai_predictions", ai_preds)

        # cache/ml_predictions — odds + predictions in old format
        odds_matches = []
        for mid, pred in self._ml_preds.items():
            od = find_match_in_list(
                pred["home_team"], pred["away_team"], self._odds
            )
            ens = pred["ensemble"]
            recommended = _normalize_actionable_outcome(pred.get("recommended"))
            is_actionable = bool(
                recommended
                and str(pred.get("decision_status") or "").strip().upper() == "BET"
            )
            selected = recommended.lower() if recommended else None
            confidence = round(ens.get(selected, 0) * 100) if is_actionable else 0

            ho = _verified_decimal_odd(od.get("home_odds")) if od else 0.0
            do_ = _verified_decimal_odd(od.get("draw_odds")) if od else 0.0
            ao = _verified_decimal_odd(od.get("away_odds")) if od else 0.0
            has_complete_market = all(value > 1.0 for value in (ho, do_, ao))

            # Calculate edge
            edge_val = pred.get("edge", {}).get(selected, 0) if selected else 0

            if not is_actionable:
                continue

            odds_matches.append({
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "league": pred["league"],
                "match_date": pred["match_date"],
                "kickoff": "",
                "odds_1": ho,
                "odds_x": do_,
                "odds_2": ao,
                "odds_available": has_complete_market,
                "ai_prediction": recommended,
                "ai_confidence": confidence,
                "value_bet": has_complete_market and edge_val > 0.03,
                "value_edge": round(edge_val * 100, 1) if has_complete_market else 0,
            })

        if not preserve_predictions:
            self.fs.write_cache("ml_predictions", {
                "predictions": ai_preds,
                "odds_matches": odds_matches,
            })
            self.fs.write_cache("model_breakdown", build_model_breakdown(self._ml_preds))
        try:
            self.fs.refresh_coupon_history_cache()
        except Exception as e:
            log.warning(f"  Coupon history cache refresh failed: {e}")

        # Performance caches — history and live paper-trading P&L
        try:
            self.fs.refresh_prediction_history_cache()
            paper = self.fs.refresh_paper_trading_cache(
                stake=PAPER_TRADING.get("stake_per_bet", 100),
                bankroll=PAPER_TRADING.get("starting_bankroll", 10000),
            )
            log.info(
                "  Paper trading: %s bets, P&L %+.0f DKK, ROI %+.1f%%",
                paper["totalBets"],
                paper["totalProfit"],
                paper["roi"],
            )
        except Exception as e:
            log.debug(f"  Paper trading cache skipped: {e}")

        try:
            self.fs.refresh_forecast_history_cache()
        except Exception as e:
            log.debug(f"  Forecast history cache skipped: {e}")

        try:
            self.fs.refresh_source_weights_cache()
        except Exception as e:
            log.warning(f"  Source weights cache refresh failed: {e}")

        prediction_note = "preserved" if preserve_predictions else str(len(ai_preds))
        log.info(f"  Wrote {len(upcoming)} matches, {prediction_note} predictions, "
                 f"{len(odds_matches)} odds_matches to cache")

    def refresh_public_performance_caches(self):
        """Rebuild historical/statistical caches after evaluation finishes."""
        refreshers = (
            ("coupon_history", self.fs.refresh_coupon_history_cache),
            ("prediction_history", self.fs.refresh_prediction_history_cache),
            (
                "paper_trading",
                lambda: self.fs.refresh_paper_trading_cache(
                    stake=PAPER_TRADING.get("stake_per_bet", 100),
                    bankroll=PAPER_TRADING.get("starting_bankroll", 10000),
                ),
            ),
            ("forecast_history", self.fs.refresh_forecast_history_cache),
            ("strategy_zoo", self.fs.refresh_strategy_zoo_cache),
            ("source_weights", self.fs.refresh_source_weights_cache),
        )
        for cache_id, refresh in refreshers:
            try:
                refresh()
            except Exception as e:
                log.warning("  Final %s cache refresh failed: %s", cache_id, e)

    def sync_public_cache(self, *, mode: str):
        """Publish staged caches and fail the run if the website stays stale."""
        try:
            result = self.fs.sync_public_cache(mode=mode)
        except Exception as e:
            # Keep the exception deliberately generic: sync credentials must
            # never be copied into pipeline logs.
            log.error(
                "PUBLIC CACHE NOT SYNCED (unexpected_%s). Firestore remains current; "
                "aibets.dk may be stale.",
                type(e).__name__,
            )
            self._stats["public_cache_synced"] = False
            self._stats["public_cache_sync_reason"] = "unexpected_error"
            raise PublicCacheSyncFailed("public_cache_sync_failed:unexpected_error") from e
        self._stats["public_cache_synced"] = bool(result.synced)
        self._stats["public_cache_sync_reason"] = result.reason
        self._stats["public_cache_sync_attempted_at"] = result.attempted_at
        if not result.synced:
            raise PublicCacheSyncFailed(f"public_cache_sync_failed:{result.reason}")
        return result

    # ──────────────────────────────────────
    # MAIN RUN
    # ──────────────────────────────────────

    def run_full(self):
        """Full pipeline: fetch → scrape → predict → evaluate → cache."""
        start = datetime.now()
        log.info("╔════════════════════════════════════════╗")
        log.info("║   AIBets Prediction Pipeline v3.0     ║")
        log.info(f"║   {start.strftime('%Y-%m-%d %H:%M:%S')}                  ║")
        log.info("╚════════════════════════════════════════╝")

        # Stage 1: Fetch matches
        self.fetch_matches()
        self.enrich_match_context()

        # Auto-retrain if models not trained or older than 7 days (feedback loop)
        if not self.engine.is_trained or self._should_retrain():
            log.info("── Auto-retrain triggered ──")
            self.train_models()

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

        # ── Write cache BEFORE evaluate (critical: ensures predictions reach frontend even if quota runs out) ──
        self.write_legacy_cache()

        # Stage 7: Evaluate finished (may hit Firestore quota — non-fatal)
        try:
            self.evaluate_finished()
        except Exception as e:
            log.warning(f"  Evaluate finished failed (quota?): {e}")

        # Stage 8: Source performance (may hit quota — non-fatal)
        try:
            self.update_source_performance()
        except Exception as e:
            log.warning(f"  Source performance update failed (quota?): {e}")

        # The early cache write keeps today's predictions available even if a
        # later Firestore stage fails.  Rebuild performance caches again here
        # so history, P&L and source metrics include this run's evaluations.
        self.refresh_public_performance_caches()
        self.sync_public_cache(mode="full")

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
        log.info(f"  Odds snaps:   {self._stats['odds_snapshots']}")
        log.info(f"  Pick snaps:   {self._stats['pick_snapshots']}")
        log.info(f"  Context:      {self._stats['match_contexts']}")
        log.info("═══════════════════════════════════════")

        return self._stats

    def run_odds_only(self):
        """Just update odds from Danske Spil."""
        log.info("═ Odds-only update ═")
        self.fetch_matches()
        self.enrich_match_context()
        self.fetch_odds()
        self.write_legacy_cache(preserve_predictions=True)
        self.sync_public_cache(mode="odds")

    def run_evaluate_only(self):
        """Just evaluate finished matches."""
        log.info("═ Evaluate-only run ═")
        self.fetch_matches()
        try:
            self.evaluate_finished()
        except Exception as e:
            if "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                log.warning("  Evaluate stage aborted — Firestore quota exhausted")
            else:
                raise
        try:
            self.update_source_performance()
        except Exception as e:
            if "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                log.warning("  Source performance update skipped — Firestore quota exhausted")
            else:
                log.warning("  Source performance update failed: %s", e)
        self.refresh_public_performance_caches()
        self.sync_public_cache(mode="evaluate")

    def run_context_only(self):
        """Just refresh lineup/injury/player context."""
        log.info("═ Context-only update ═")
        self.fetch_matches()
        self.enrich_match_context()
        try:
            self.update_source_performance()
        except Exception as e:
            if "Quota" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                log.warning("  Source performance update skipped — Firestore quota exhausted")
            else:
                raise


# ─────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AIBets Prediction Pipeline")
    parser.add_argument("--train", action="store_true", help="Train ML models before predicting")
    parser.add_argument("--odds-only", action="store_true", help="Only update odds from Danske Spil")
    parser.add_argument("--context-only", action="store_true", help="Only update lineup/injury/player context")
    parser.add_argument("--evaluate-only", action="store_true", help="Only evaluate finished matches")
    parser.add_argument("--watch", action="store_true", help="Keep running and update on an interval")
    parser.add_argument("--interval-minutes", type=float, default=15.0,
                        help="Minutes between runs in --watch mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    pipeline = PredictionPipeline()

    if args.train:
        pipeline.train_models()

    def run_selected_mode():
        if args.odds_only:
            pipeline.run_odds_only()
        elif args.context_only:
            pipeline.run_context_only()
        elif args.evaluate_only:
            pipeline.run_evaluate_only()
        else:
            pipeline.run_full()

    if args.watch:
        interval_seconds = max(args.interval_minutes, 1.0) * 60
        log.info("Watch mode enabled: running every %.1f minutes", interval_seconds / 60)
        while True:
            try:
                run_selected_mode()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                log.error("Watch mode run failed: %s", e, exc_info=True)
            log.info("Next update in %.1f minutes", interval_seconds / 60)
            time.sleep(interval_seconds)
    else:
        run_selected_mode()


if __name__ == "__main__":
    main()
