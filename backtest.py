#!/usr/bin/env python3
"""
Historical Backtest — v1 vs v2 Pipeline Comparison
===================================================
Runs two evaluation modes on all CSV historical data (10 leagues × 11 seasons):
  1. Simple Holdout: train 2015-2022, test 2023-2025
  2. Walk-Forward: retrain season-by-season, predict next season

Outputs: accuracy, Brier score, log loss, ROI (flat + Kelly), per-league
breakdown, edge-threshold sweep, and calibration table.

Usage:
    python backtest.py                        # Both modes, all leagues
    python backtest.py --holdout              # Holdout only
    python backtest.py --walk-forward         # Walk-forward only
    python backtest.py --leagues PL,PD        # Specific leagues
    python backtest.py --verbose              # Debug logging
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Suppress sklearn CalibratedClassifierCV parallel warnings (268k lines otherwise)
try:
    import sklearn.utils.parallel as _skp
    _skp_orig = _skp.delayed
    def _silent_delayed(function):
        """Wrap sklearn.delayed to suppress its UserWarning about Parallel."""
        return _skp_orig(function)
    _silent_delayed.__doc__ = _skp_orig.__doc__
    # Silence the specific warning at source
    warnings.filterwarnings("ignore", module="sklearn.utils.parallel")
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
except Exception:
    pass

# ── Path setup ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config.settings import ML_SETTINGS, ML_SETTINGS_V2, DB_DIR, LEAGUES
from src.api.csv_football_client import FootballDataCSVClient
from src.database.db_manager import DatabaseManager
from src.predictions.feature_engineering import (
    EloTracker,
    FeatureEngineer,
    FeatureEngineerV2,
)
from src.predictions.models import (
    EnsembleModel,
    LightGBMModel,
    NeuralNetworkModel,
    PoissonModel,
    RandomForestModel,
    StackingEnsemble,
    XGBoostModel,
)

# ── Logging ─────────────────────────────────────────────────
logger = logging.getLogger("backtest")

# ── Constants ───────────────────────────────────────────────
LABEL_HOME = 0
LABEL_DRAW = 1
LABEL_AWAY = 2
OUTCOME_MAP = {LABEL_HOME: "HOME", LABEL_DRAW: "DRAW", LABEL_AWAY: "AWAY"}

# Leagues that have CSV data available
CSV_LEAGUES = list(FootballDataCSVClient.LEAGUE_CSV_MAP.keys())

# Season range available in CSV
ALL_SEASONS = [s for s, _ in FootballDataCSVClient.AVAILABLE_SEASONS]  # [2025,...,2015]
ALL_SEASONS.sort()  # [2015,...,2025]


# ═════════════════════════════════════════════════════════════
#  DATA LOADING
# ═════════════════════════════════════════════════════════════
def load_all_csv_data(leagues: List[str], seasons: List[int] = None) -> List[Dict]:
    """Download and merge CSV data for given leagues & seasons."""
    csv_client = FootballDataCSVClient()
    seasons = seasons or ALL_SEASONS
    all_matches = []
    for league in leagues:
        for season in seasons:
            try:
                matches = csv_client.get_season_matches(league, season)
                all_matches.extend(matches)
                logger.info(f"  {league} {season}/{season+1}: {len(matches)} matches")
            except Exception as e:
                logger.warning(f"  {league} {season}: FAILED ({e})")
    all_matches.sort(key=lambda m: m.get("match_date", ""))
    logger.info(f"Total CSV matches loaded: {len(all_matches)}")
    return all_matches


def populate_db(db: DatabaseManager, matches: List[Dict]):
    """Insert all matches into the backtest database + compute team stats."""
    logger.info("Populating database...")
    for m in matches:
        try:
            db.upsert_match(m)
        except Exception:
            pass

    # Also insert H2H records
    for m in matches:
        if m.get("status") != "FINISHED" or m.get("home_score") is None:
            continue
        try:
            db.add_h2h(
                m["home_team_name"], m["away_team_name"],
                int(m["home_score"]), int(m["away_score"]),
                m.get("league_code", ""), m.get("match_date", ""),
                m.get("season", 0),
            )
        except Exception:
            pass

    # Compute and store team stats per league/season using direct SQL
    # (populate_db can use end-of-season totals since training will recompute as needed)
    teams_done = set()
    # Group matches by (team, league, season) and compute stats directly
    team_season_matches: Dict[tuple, List[Dict]] = defaultdict(list)
    for m in matches:
        if m.get("status") != "FINISHED" or m.get("home_score") is None:
            continue
        lc = m.get("league_code", "")
        season = m.get("season", 0)
        team_season_matches[(m["home_team_name"], lc, season)].append(m)
        team_season_matches[(m["away_team_name"], lc, season)].append(m)

    for (team, lc, season), team_matches in team_season_matches.items():
        if len(team_matches) < 3:
            continue
        stats = _compute_stats_from_list(team, lc, season, team_matches)
        if stats and stats.get("matches_played", 0) >= 3:
            db.upsert_team_stats(stats)
            teams_done.add((team, lc, season))

    logger.info(f"DB populated: {len(matches)} matches, {len(teams_done)} team-season combos")


def _compute_stats_from_list(team_name: str, league_code: str, season: int,
                              matches: List[Dict]) -> Dict:
    """Compute team stats directly from a list of matches (no DB query needed)."""
    stats = {
        "team_name": team_name, "league_code": league_code, "season": season,
        "matches_played": 0, "wins": 0, "draws": 0, "losses": 0,
        "goals_scored": 0, "goals_conceded": 0, "clean_sheets": 0,
        "home_wins": 0, "home_draws": 0, "home_losses": 0,
        "away_wins": 0, "away_draws": 0, "away_losses": 0,
        "home_goals_scored": 0, "home_goals_conceded": 0,
        "away_goals_scored": 0, "away_goals_conceded": 0,
        "form": "",
    }
    form_list = []
    for m in matches:
        hs = m.get("home_score")
        aws = m.get("away_score")
        if hs is None or aws is None:
            continue
        hs, aws = int(hs), int(aws)
        is_home = m.get("home_team_name") == team_name
        if is_home:
            gs, gc = hs, aws
        else:
            gs, gc = aws, hs

        stats["matches_played"] += 1
        stats["goals_scored"] += gs
        stats["goals_conceded"] += gc
        if gc == 0:
            stats["clean_sheets"] += 1

        if gs > gc:
            stats["wins"] += 1
            form_list.append("W")
            if is_home:
                stats["home_wins"] += 1
            else:
                stats["away_wins"] += 1
        elif gs == gc:
            stats["draws"] += 1
            form_list.append("D")
            if is_home:
                stats["home_draws"] += 1
            else:
                stats["away_draws"] += 1
        else:
            stats["losses"] += 1
            form_list.append("L")
            if is_home:
                stats["home_losses"] += 1
            else:
                stats["away_losses"] += 1

        if is_home:
            stats["home_goals_scored"] += gs
            stats["home_goals_conceded"] += gc
        else:
            stats["away_goals_scored"] += gs
            stats["away_goals_conceded"] += gc

    mp = stats["matches_played"]
    if mp > 0:
        stats["avg_goals_scored"] = round(stats["goals_scored"] / mp, 2)
        stats["avg_goals_conceded"] = round(stats["goals_conceded"] / mp, 2)
    else:
        stats["avg_goals_scored"] = 0.0
        stats["avg_goals_conceded"] = 0.0
    stats["form"] = "".join(form_list[-5:])
    return stats


def _build_team_index(matches: List[Dict]) -> Dict[str, List[Dict]]:
    """Build team_name → [matches] index. Only FINISHED matches with scores."""
    idx: Dict[str, List[Dict]] = defaultdict(list)
    for m in matches:
        if m.get("status") != "FINISHED" or m.get("home_score") is None:
            continue
        idx[m["home_team_name"]].append(m)
        idx[m["away_team_name"]].append(m)
    return idx


def _build_h2h_index(matches: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    """Build (team1, team2) → [h2h_records] index (both directions)."""
    idx: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for m in matches:
        if m.get("status") != "FINISHED" or m.get("home_score") is None:
            continue
        h = m["home_team_name"]
        a = m["away_team_name"]
        rec = {
            "home_team": h, "away_team": a,
            "home_goals": int(m["home_score"]), "away_goals": int(m["away_score"]),
            "match_date": m.get("match_date", ""), "season": m.get("season", 0),
        }
        idx[(h, a)].append(rec)
        idx[(a, h)].append(rec)
    return idx


def _quick_stats(team_name: str, league_code: str, season: int,
                 team_matches: List[Dict]) -> Optional[Dict]:
    """Compute stats from a team's match list, filtering by league+season."""
    relevant = [m for m in team_matches
                if m.get("league_code") == league_code and m.get("season") == season]
    if len(relevant) < 3:
        return None
    stats = _compute_stats_from_list(team_name, league_code, season, relevant)
    return stats if stats and stats.get("matches_played", 0) >= 3 else None


# ═════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ═════════════════════════════════════════════════════════════
def create_models(config: Dict, suffix: str, is_v2: bool) -> Dict[str, object]:
    """Create fresh model instances for a backtest run."""
    models = {
        "xgboost": XGBoostModel(config=config, suffix=suffix),
        "neural_network": NeuralNetworkModel(config=config, suffix=suffix),
        "random_forest": RandomForestModel(config=config, suffix=suffix),
    }
    if is_v2 and config.get("lightgbm"):
        models["lightgbm"] = LightGBMModel(config=config, suffix=suffix)
    return models


def train_models(models: Dict, X: np.ndarray, y: np.ndarray, config: Dict):
    """Train all models and stacking ensemble on given data."""
    # NaN guard
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    accuracies = {}
    for name, model in models.items():
        try:
            acc = model.train(X, y)
            accuracies[name] = acc
            logger.info(f"  {name}: accuracy={acc:.4f}")
        except Exception as e:
            logger.error(f"  {name} training failed: {e}")
            accuracies[name] = 0.0

    # Stacking
    stacking = None
    if config.get("ensemble", {}).get("use_stacking", False) and len(X) > 300:
        stacking = StackingEnsemble(models, config, suffix="_bt")
        try:
            stacking.train_meta(X, y)
            logger.info(f"  stacking: accuracy={stacking.accuracy:.4f}")
        except Exception as e:
            logger.warning(f"  stacking failed: {e}")
            stacking = None

    ensemble = EnsembleModel(models, config)
    return models, ensemble, stacking, accuracies


# ═════════════════════════════════════════════════════════════
#  PREDICTION HELPERS
# ═════════════════════════════════════════════════════════════
def predict_single_v1(match: Dict, models: Dict, ensemble: EnsembleModel,
                      stacking: StackingEnsemble, home_stats: Dict,
                      away_stats: Dict, h2h: List[Dict],
                      config: Dict) -> Optional[Dict]:
    """Predict a single match using v1 pipeline (pre-computed stats, no DB)."""
    home_name = match["home_team_name"]
    away_name = match["away_team_name"]
    league_code = match.get("league_code", "")
    season = match.get("season", 2025)
    match_date = match.get("match_date", "")

    home_odds = match.get("home_odds")
    draw_odds = match.get("draw_odds")
    away_odds = match.get("away_odds")

    features = FeatureEngineer.build_match_features(
        home_stats, away_stats, h2h,
        home_odds, draw_odds, away_odds,
        ai_predictions=None,  # no AI data historically
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensemble prediction
    if stacking and stacking.is_trained:
        probs = stacking.predict_proba(features)
    else:
        probs = ensemble.predict_proba(features)
    if probs.ndim > 1:
        probs = probs[0]

    predicted_label = int(np.argmax(probs))
    confidence = float(probs[predicted_label])

    # Edge calculation
    edge = 0.0
    if home_odds and draw_odds and away_odds and home_odds > 1 and draw_odds > 1 and away_odds > 1:
        inv_sum = (1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds)
        fair_odds = [1.0 / home_odds / inv_sum, 1.0 / draw_odds / inv_sum, 1.0 / away_odds / inv_sum]
        edge = float(probs[predicted_label] - fair_odds[predicted_label])

    # Kelly criterion
    actual_odds_map = {LABEL_HOME: home_odds, LABEL_DRAW: draw_odds, LABEL_AWAY: away_odds}
    sel_odds = actual_odds_map.get(predicted_label)
    kelly = 0.0
    if sel_odds and sel_odds > 1.0:
        b = sel_odds - 1.0
        kelly = max(0.0, (b * confidence - (1.0 - confidence)) / b)
        kelly = min(kelly, 0.25)

    hs = int(match["home_score"])
    aws = int(match["away_score"])
    actual_label = LABEL_HOME if hs > aws else (LABEL_DRAW if hs == aws else LABEL_AWAY)

    return {
        "match_date": match_date,
        "league": league_code,
        "season": season,
        "home": home_name,
        "away": away_name,
        "home_score": hs,
        "away_score": aws,
        "actual": actual_label,
        "predicted": predicted_label,
        "confidence": confidence,
        "home_prob": float(probs[0]),
        "draw_prob": float(probs[1]),
        "away_prob": float(probs[2]),
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "edge": edge,
        "kelly": kelly,
        "version": "v1",
    }


def predict_single_v2(match: Dict, models: Dict, ensemble: EnsembleModel,
                      stacking: StackingEnsemble, home_stats: Dict,
                      away_stats: Dict, h2h: List[Dict],
                      config: Dict, elo_tracker: EloTracker,
                      poisson: PoissonModel,
                      home_past: List[Dict], away_past: List[Dict]) -> Optional[Dict]:
    """Predict a single match using v2 pipeline (pre-computed stats, team-specific past, no DB)."""
    home_name = match["home_team_name"]
    away_name = match["away_team_name"]
    league_code = match.get("league_code", "")
    season = match.get("season", 2025)
    match_date = match.get("match_date", "")

    home_odds = match.get("home_odds")
    draw_odds = match.get("draw_odds")
    away_odds = match.get("away_odds")

    # Compute v2 extra features from team-specific past matches (fast: ~300 items vs ~20k)
    home_form = FeatureEngineerV2.compute_form_list(home_past, home_name)
    away_form = FeatureEngineerV2.compute_form_list(away_past, away_name)
    home_extra = FeatureEngineerV2.compute_csv_extra_averages(home_past, home_name)
    away_extra = FeatureEngineerV2.compute_csv_extra_averages(away_past, away_name)
    home_rest = FeatureEngineerV2.compute_days_since_last(home_past, home_name, match_date)
    away_rest = FeatureEngineerV2.compute_days_since_last(away_past, away_name, match_date)
    home_goals = FeatureEngineerV2.compute_recent_goals_avg(home_past, home_name)
    away_goals = FeatureEngineerV2.compute_recent_goals_avg(away_past, away_name)
    home_sos = FeatureEngineerV2.compute_sos(home_past, home_name, elo_tracker)
    away_sos = FeatureEngineerV2.compute_sos(away_past, away_name, elo_tracker)

    features = FeatureEngineerV2.build_match_features_v2(
        home_stats, away_stats, h2h,
        home_odds, draw_odds, away_odds,
        ai_predictions=None,
        elo_tracker=elo_tracker,
        home_form_list=home_form,
        away_form_list=away_form,
        home_extra=home_extra,
        away_extra=away_extra,
        home_days_rest=home_rest,
        away_days_rest=away_rest,
        home_recent_goals_avg=home_goals,
        away_recent_goals_avg=away_goals,
        is_training=True,  # mask AI features (not available historically)
        league_code=league_code,
        matchday=match.get("matchday", 0) or 0,
        total_matchdays=38,
        match_datetime=match_date,
        home_sos=home_sos,
        away_sos=away_sos,
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # ML ensemble prediction
    if stacking and stacking.is_trained:
        probs = stacking.predict_proba(features)
    else:
        probs = ensemble.predict_proba(features)
    if probs.ndim > 1:
        probs = probs[0]

    # Poisson blend (5% — reduced from 10% to prevent overconfidence)
    h_attack = home_stats.get("avg_goals_scored", 1.3)
    h_defense = home_stats.get("avg_goals_conceded", 1.1)
    a_attack = away_stats.get("avg_goals_scored", 1.2)
    a_defense = away_stats.get("avg_goals_conceded", 1.2)
    h_exp, a_exp = poisson.predict_score(h_attack, h_defense, a_attack, a_defense)
    poisson_probs = poisson.match_outcome_probs(h_exp, a_exp)

    blended = np.array([
        probs[0] * 0.95 + poisson_probs["home_win"] * 0.05,
        probs[1] * 0.95 + poisson_probs["draw"] * 0.05,
        probs[2] * 0.95 + poisson_probs["away_win"] * 0.05,
    ])
    total = blended.sum()
    if total > 0:
        blended /= total

    predicted_label = int(np.argmax(blended))
    confidence = float(blended[predicted_label])

    # Edge
    edge = 0.0
    if home_odds and draw_odds and away_odds and home_odds > 1 and draw_odds > 1 and away_odds > 1:
        inv_sum = (1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds)
        fair_odds = [1.0 / home_odds / inv_sum, 1.0 / draw_odds / inv_sum, 1.0 / away_odds / inv_sum]
        edge = float(blended[predicted_label] - fair_odds[predicted_label])

    # Kelly
    actual_odds_map = {LABEL_HOME: home_odds, LABEL_DRAW: draw_odds, LABEL_AWAY: away_odds}
    sel_odds = actual_odds_map.get(predicted_label)
    kelly = 0.0
    if sel_odds and sel_odds > 1.0:
        b = sel_odds - 1.0
        kelly = max(0.0, (b * confidence - (1.0 - confidence)) / b)
        kelly = min(kelly, 0.25)

    # BTTS / O2.5 from Poisson
    btts_prob = (1 - math.exp(-h_exp)) * (1 - math.exp(-a_exp))
    total_exp = h_exp + a_exp
    over25_prob = 1.0 - sum(
        (total_exp ** k) * math.exp(-total_exp) / math.factorial(k) for k in range(3)
    )

    hs = int(match["home_score"])
    aws = int(match["away_score"])
    actual_label = LABEL_HOME if hs > aws else (LABEL_DRAW if hs == aws else LABEL_AWAY)

    return {
        "match_date": match_date,
        "league": league_code,
        "season": season,
        "home": home_name,
        "away": away_name,
        "home_score": hs,
        "away_score": aws,
        "actual": actual_label,
        "predicted": predicted_label,
        "confidence": confidence,
        "home_prob": float(blended[0]),
        "draw_prob": float(blended[1]),
        "away_prob": float(blended[2]),
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "edge": edge,
        "kelly": kelly,
        "poisson_score": f"{h_exp}-{a_exp}",
        "btts_prob": btts_prob,
        "over25_prob": over25_prob,
        "version": "v2",
    }


# ═════════════════════════════════════════════════════════════
#  METRICS CALCULATION
# ═════════════════════════════════════════════════════════════
def compute_metrics(predictions: List[Dict]) -> Dict:
    """Compute all metrics for a set of predictions."""
    if not predictions:
        return {"total": 0}

    total = len(predictions)
    correct = sum(1 for p in predictions if p["predicted"] == p["actual"])
    accuracy = correct / total

    # Brier Score: mean(sum((p_k - actual_k)^2)) for 3 outcomes
    brier_sum = 0.0
    log_loss_sum = 0.0
    for p in predictions:
        actual_vec = np.zeros(3)
        actual_vec[p["actual"]] = 1.0
        pred_vec = np.array([p["home_prob"], p["draw_prob"], p["away_prob"]])
        brier_sum += np.sum((pred_vec - actual_vec) ** 2)
        # Log loss: -log(p_actual)
        p_actual = pred_vec[p["actual"]]
        log_loss_sum += -math.log(max(p_actual, 1e-10))

    brier = brier_sum / total
    log_loss = log_loss_sum / total

    # ROI — flat stake (1 unit on predicted outcome)
    roi_flat = _calc_roi_flat(predictions)

    # ROI — Kelly
    roi_kelly = _calc_roi_kelly(predictions)

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "roi_flat": roi_flat,
        "roi_kelly": roi_kelly,
    }


def _calc_roi_flat(predictions: List[Dict]) -> Dict:
    """Flat-stake ROI: bet 1 unit on every predicted outcome."""
    total_staked = 0
    total_return = 0.0
    bets_with_odds = 0
    for p in predictions:
        odds_map = {LABEL_HOME: p["home_odds"], LABEL_DRAW: p["draw_odds"], LABEL_AWAY: p["away_odds"]}
        sel_odds = odds_map.get(p["predicted"])
        if not sel_odds or sel_odds <= 1.0:
            continue
        total_staked += 1
        bets_with_odds += 1
        if p["predicted"] == p["actual"]:
            total_return += sel_odds
    profit = total_return - total_staked
    roi_pct = (profit / total_staked * 100) if total_staked > 0 else 0.0
    return {"staked": total_staked, "returned": total_return, "profit": profit, "roi_pct": roi_pct}


def _calc_roi_kelly(predictions: List[Dict]) -> Dict:
    """Kelly-stake ROI: stake = kelly_fraction on each bet. Only bet when kelly > 0."""
    total_staked = 0.0
    total_return = 0.0
    n_bets = 0
    for p in predictions:
        k = p.get("kelly", 0.0)
        if k <= 0:
            continue
        odds_map = {LABEL_HOME: p["home_odds"], LABEL_DRAW: p["draw_odds"], LABEL_AWAY: p["away_odds"]}
        sel_odds = odds_map.get(p["predicted"])
        if not sel_odds or sel_odds <= 1.0:
            continue
        total_staked += k
        n_bets += 1
        if p["predicted"] == p["actual"]:
            total_return += k * sel_odds
    profit = total_return - total_staked
    roi_pct = (profit / total_staked * 100) if total_staked > 0 else 0.0
    return {"n_bets": n_bets, "staked": round(total_staked, 2),
            "returned": round(total_return, 2), "profit": round(profit, 2), "roi_pct": roi_pct}


def compute_per_league(predictions: List[Dict]) -> Dict[str, Dict]:
    """Compute metrics per league."""
    by_league = defaultdict(list)
    for p in predictions:
        by_league[p["league"]].append(p)
    return {league: compute_metrics(preds) for league, preds in sorted(by_league.items())}


def compute_edge_threshold_sweep(predictions: List[Dict]) -> List[Dict]:
    """Sweep edge thresholds and compute metrics for each."""
    thresholds = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    results = []
    for thresh in thresholds:
        filtered = [p for p in predictions if p["edge"] >= thresh]
        m = compute_metrics(filtered)
        m["threshold"] = thresh
        results.append(m)
    return results


def compute_betting_experiments(predictions: List[Dict]) -> List[Dict]:
    """Run many betting strategies on the full prediction set and return results."""
    experiments = []

    def _run(name: str, preds: List[Dict], bet_label_fn=None):
        """Run a single experiment. bet_label_fn overrides predicted label if given."""
        staked = 0
        returned = 0.0
        n_correct = 0
        n_total = len(preds)
        kelly_staked = 0.0
        kelly_returned = 0.0
        kelly_bets = 0
        cumulative = []  # (match_idx, running_profit)

        for p in preds:
            label = bet_label_fn(p) if bet_label_fn else p["predicted"]
            odds_map = {LABEL_HOME: p["home_odds"], LABEL_DRAW: p["draw_odds"], LABEL_AWAY: p["away_odds"]}
            sel_odds = odds_map.get(label)
            if not sel_odds or sel_odds <= 1.0:
                continue
            staked += 1
            if label == p["actual"]:
                returned += sel_odds
                n_correct += 1
            cumulative.append(returned - staked)

            # Kelly
            prob = [p["home_prob"], p["draw_prob"], p["away_prob"]][label]
            b = sel_odds - 1.0
            k = max(0.0, (b * prob - (1.0 - prob)) / b) if b > 0 else 0.0
            k = min(k, 0.25)
            if k > 0:
                kelly_staked += k
                kelly_bets += 1
                if label == p["actual"]:
                    kelly_returned += k * sel_odds

        profit = returned - staked
        roi = (profit / staked * 100) if staked > 0 else 0.0
        acc = (n_correct / staked * 100) if staked > 0 else 0.0
        k_profit = kelly_returned - kelly_staked
        k_roi = (k_profit / kelly_staked * 100) if kelly_staked > 0 else 0.0
        # Max drawdown from cumulative
        max_dd = 0.0
        peak = 0.0
        for pl in cumulative:
            if pl > peak:
                peak = pl
            dd = peak - pl
            if dd > max_dd:
                max_dd = dd

        experiments.append({
            "name": name,
            "bets": staked,
            "correct": n_correct,
            "accuracy": acc,
            "profit": profit,
            "roi": roi,
            "kelly_bets": kelly_bets,
            "kelly_profit": k_profit,
            "kelly_roi": k_roi,
            "max_drawdown": max_dd,
        })

    # ── 1. ALL predictions (model picks) ──
    _run("ALL predictions", predictions)

    # ── 2. Confidence thresholds ──
    for thresh in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        filtered = [p for p in predictions if p["confidence"] >= thresh]
        _run(f"Conf >= {thresh:.0%}", filtered)

    # ── 3. Edge thresholds ──
    for thresh in [0.02, 0.05, 0.10, 0.15, 0.20]:
        filtered = [p for p in predictions if p["edge"] >= thresh]
        _run(f"Edge >= {thresh:.0%}", filtered)

    # ── 4. Only specific outcomes ──
    _run("Only HOME picks", [p for p in predictions if p["predicted"] == LABEL_HOME])
    _run("Only DRAW picks", [p for p in predictions if p["predicted"] == LABEL_DRAW])
    _run("Only AWAY picks", [p for p in predictions if p["predicted"] == LABEL_AWAY])

    # ── 5. OPPOSITE of model (always bet against) ──
    def _opposite(p):
        probs = [(p["home_prob"], LABEL_HOME), (p["draw_prob"], LABEL_DRAW), (p["away_prob"], LABEL_AWAY)]
        probs.sort(key=lambda x: x[0])  # lowest prob first
        return probs[0][1]  # bet on LEAST likely outcome
    _run("OPPOSITE (least likely)", predictions, bet_label_fn=_opposite)

    # ── 6. Baselines: always bet Home / Draw / Away ──
    _run("Baseline: always HOME", predictions, bet_label_fn=lambda p: LABEL_HOME)
    _run("Baseline: always DRAW", predictions, bet_label_fn=lambda p: LABEL_DRAW)
    _run("Baseline: always AWAY", predictions, bet_label_fn=lambda p: LABEL_AWAY)

    # ── 7. Favourite (lowest odds) / Underdog (highest odds) ──
    def _favourite(p):
        odds = [(p.get("home_odds") or 99, LABEL_HOME),
                (p.get("draw_odds") or 99, LABEL_DRAW),
                (p.get("away_odds") or 99, LABEL_AWAY)]
        odds.sort(key=lambda x: x[0])
        return odds[0][1]
    _run("Baseline: FAVOURITE", predictions, bet_label_fn=_favourite)

    def _underdog(p):
        odds = [(p.get("home_odds") or 0, LABEL_HOME),
                (p.get("draw_odds") or 0, LABEL_DRAW),
                (p.get("away_odds") or 0, LABEL_AWAY)]
        odds.sort(key=lambda x: x[0], reverse=True)
        return odds[0][1]
    _run("Baseline: UNDERDOG", predictions, bet_label_fn=_underdog)

    # ── 8. Kelly-only (only bet when kelly > 0) ──
    kelly_only = [p for p in predictions if p.get("kelly", 0) > 0]
    _run("Kelly filter (k>0)", kelly_only)

    # ── 9. High-value: edge>5% AND conf>45% ──
    hv = [p for p in predictions if p["edge"] >= 0.05 and p["confidence"] >= 0.45]
    _run("Edge>=5% + Conf>=45%", hv)

    hv2 = [p for p in predictions if p["edge"] >= 0.10 and p["confidence"] >= 0.50]
    _run("Edge>=10% + Conf>=50%", hv2)

    return experiments


def print_betting_experiments(v1_exps: List[Dict], v2_exps: List[Dict]):
    """Print side-by-side betting experiment comparison."""
    print_header("BETTING EXPERIMENTS (1 unit flat stake per bet)")
    print()
    print(f"  {'Strategy':<26} │ {'v1 Bets':>7} {'v1 Acc':>7} {'v1 Profit':>10} {'v1 ROI%':>8} │ {'v2 Bets':>7} {'v2 Acc':>7} {'v2 Profit':>10} {'v2 ROI%':>8}")
    print(f"  {'─'*26} │ {'─'*7} {'─'*7} {'─'*10} {'─'*8} │ {'─'*7} {'─'*7} {'─'*10} {'─'*8}")

    for v1e, v2e in zip(v1_exps, v2_exps):
        name = v1e["name"]
        v1_bets = str(v1e["bets"])
        v1_acc = f"{v1e['accuracy']:.1f}%" if v1e["bets"] else "—"
        v1_p = f"{v1e['profit']:+.1f}u" if v1e["bets"] else "—"
        v1_r = f"{v1e['roi']:+.1f}%" if v1e["bets"] else "—"
        v2_bets = str(v2e["bets"])
        v2_acc = f"{v2e['accuracy']:.1f}%" if v2e["bets"] else "—"
        v2_p = f"{v2e['profit']:+.1f}u" if v2e["bets"] else "—"
        v2_r = f"{v2e['roi']:+.1f}%" if v2e["bets"] else "—"

        # Color profit
        if v1e["bets"] and v1e["profit"] > 0:
            v1_p = f"\033[92m{v1_p}\033[0m"
            v1_r = f"\033[92m{v1_r}\033[0m"
        elif v1e["bets"] and v1e["profit"] < 0:
            v1_p = f"\033[91m{v1_p}\033[0m"
            v1_r = f"\033[91m{v1_r}\033[0m"
        if v2e["bets"] and v2e["profit"] > 0:
            v2_p = f"\033[92m{v2_p}\033[0m"
            v2_r = f"\033[92m{v2_r}\033[0m"
        elif v2e["bets"] and v2e["profit"] < 0:
            v2_p = f"\033[91m{v2_p}\033[0m"
            v2_r = f"\033[91m{v2_r}\033[0m"

        print(f"  {name:<26} │ {v1_bets:>7} {v1_acc:>7} {v1_p:>10} {v1_r:>8} │ {v2_bets:>7} {v2_acc:>7} {v2_p:>10} {v2_r:>8}")
    print()

    # Kelly summary
    print(f"  {'KELLY STAKING':^26}")
    print(f"  {'Strategy':<26} │ {'v1 KBets':>7} {'v1 KProf':>10} {'v1 KROI%':>8} {'v1 MaxDD':>8} │ {'v2 KBets':>7} {'v2 KProf':>10} {'v2 KROI%':>8} {'v2 MaxDD':>8}")
    print(f"  {'─'*26} │ {'─'*7} {'─'*10} {'─'*8} {'─'*8} │ {'─'*7} {'─'*10} {'─'*8} {'─'*8}")

    for v1e, v2e in zip(v1_exps[:5], v2_exps[:5]):  # Top 5 strategies only for Kelly
        name = v1e["name"]
        v1_kb = str(v1e["kelly_bets"])
        v1_kp = f"{v1e['kelly_profit']:+.1f}u" if v1e["kelly_bets"] else "—"
        v1_kr = f"{v1e['kelly_roi']:+.1f}%" if v1e["kelly_bets"] else "—"
        v1_dd = f"{v1e['max_drawdown']:.1f}u"
        v2_kb = str(v2e["kelly_bets"])
        v2_kp = f"{v2e['kelly_profit']:+.1f}u" if v2e["kelly_bets"] else "—"
        v2_kr = f"{v2e['kelly_roi']:+.1f}%" if v2e["kelly_bets"] else "—"
        v2_dd = f"{v2e['max_drawdown']:.1f}u"
        print(f"  {name:<26} │ {v1_kb:>7} {v1_kp:>10} {v1_kr:>8} {v1_dd:>8} │ {v2_kb:>7} {v2_kp:>10} {v2_kr:>8} {v2_dd:>8}")
    print()


def compute_calibration(predictions: List[Dict], n_bins: int = 10) -> List[Dict]:
    """Compute calibration table: predicted probability bins vs actual frequency."""
    bins = [[] for _ in range(n_bins)]
    for p in predictions:
        # Use the confidence (predicted prob for the chosen outcome)
        conf = p["confidence"]
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append(1 if p["predicted"] == p["actual"] else 0)

    table = []
    for i, b in enumerate(bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        n = len(b)
        actual_rate = sum(b) / n if n > 0 else 0.0
        expected_mid = (lo + hi) / 2
        table.append({
            "bin": f"{lo:.0%}-{hi:.0%}",
            "count": n,
            "actual_win_rate": round(actual_rate * 100, 1),
            "expected_mid": round(expected_mid * 100, 1),
            "gap": round((actual_rate - expected_mid) * 100, 1),
        })
    return table


# ═════════════════════════════════════════════════════════════
#  REPORTING (pretty terminal output)
# ═════════════════════════════════════════════════════════════
def print_header(title: str):
    w = 70
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def print_comparison(v1_metrics: Dict, v2_metrics: Dict, label: str = ""):
    """Print side-by-side v1 vs v2 comparison table."""
    print_header(f"RESULTS — {label}")
    print()
    print(f"  {'Metric':<22} {'v1':>12} {'v2':>12}   {'Diff':>10}")
    print(f"  {'─'*22} {'─'*12} {'─'*12}   {'─'*10}")

    def row(name, v1_val, v2_val, fmt=".2f", pct=False):
        sfx = "%" if pct else ""
        v1s = f"{v1_val:{fmt}}{sfx}" if v1_val is not None else "N/A"
        v2s = f"{v2_val:{fmt}}{sfx}" if v2_val is not None else "N/A"
        diff = ""
        if v1_val is not None and v2_val is not None:
            d = v2_val - v1_val
            sign = "+" if d >= 0 else ""
            diff = f"{sign}{d:{fmt}}{sfx}"
            if d > 0:
                diff = f"\033[92m{diff}\033[0m"  # green
            elif d < 0:
                diff = f"\033[91m{diff}\033[0m"  # red
        print(f"  {name:<22} {v1s:>12} {v2s:>12}   {diff:>10}")

    row("Matches", v1_metrics.get("total"), v2_metrics.get("total"), ".0f")
    row("Correct", v1_metrics.get("correct"), v2_metrics.get("correct"), ".0f")
    row("Accuracy", (v1_metrics.get("accuracy", 0) * 100), (v2_metrics.get("accuracy", 0) * 100), ".2f", pct=True)
    row("Brier Score", v1_metrics.get("brier"), v2_metrics.get("brier"), ".4f")
    row("Log Loss", v1_metrics.get("log_loss"), v2_metrics.get("log_loss"), ".4f")

    v1_flat = v1_metrics.get("roi_flat", {})
    v2_flat = v2_metrics.get("roi_flat", {})
    row("ROI Flat", v1_flat.get("roi_pct"), v2_flat.get("roi_pct"), ".2f", pct=True)
    row("Flat Bets", v1_flat.get("staked"), v2_flat.get("staked"), ".0f")

    v1_kelly = v1_metrics.get("roi_kelly", {})
    v2_kelly = v2_metrics.get("roi_kelly", {})
    row("ROI Kelly", v1_kelly.get("roi_pct"), v2_kelly.get("roi_pct"), ".2f", pct=True)
    row("Kelly Bets", v1_kelly.get("n_bets"), v2_kelly.get("n_bets"), ".0f")
    print()


def print_per_league(v1_by_league: Dict, v2_by_league: Dict):
    """Print per-league breakdown."""
    all_leagues = sorted(set(list(v1_by_league.keys()) + list(v2_by_league.keys())))
    if not all_leagues:
        return

    print_header("PER-LEAGUE BREAKDOWN")
    print()
    print(f"  {'League':<6} {'v1 Acc':>8} {'v2 Acc':>8} {'v1 Brier':>9} {'v2 Brier':>9} {'v1 ROI%':>8} {'v2 ROI%':>8} {'v1 N':>6} {'v2 N':>6}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*9} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    for league in all_leagues:
        v1 = v1_by_league.get(league, {})
        v2 = v2_by_league.get(league, {})
        v1_acc = f"{v1.get('accuracy', 0)*100:.1f}%" if v1.get('total') else "—"
        v2_acc = f"{v2.get('accuracy', 0)*100:.1f}%" if v2.get('total') else "—"
        v1_brier = f"{v1.get('brier', 0):.4f}" if v1.get('total') else "—"
        v2_brier = f"{v2.get('brier', 0):.4f}" if v2.get('total') else "—"
        v1_roi = f"{v1.get('roi_flat', {}).get('roi_pct', 0):.1f}%" if v1.get('total') else "—"
        v2_roi = f"{v2.get('roi_flat', {}).get('roi_pct', 0):.1f}%" if v2.get('total') else "—"
        v1_n = str(v1.get('total', 0))
        v2_n = str(v2.get('total', 0))
        print(f"  {league:<6} {v1_acc:>8} {v2_acc:>8} {v1_brier:>9} {v2_brier:>9} {v1_roi:>8} {v2_roi:>8} {v1_n:>6} {v2_n:>6}")
    print()


def print_edge_sweep(v1_sweep: List[Dict], v2_sweep: List[Dict]):
    """Print edge-threshold sweep table."""
    print_header("EDGE-THRESHOLD SWEEP")
    print()
    print(f"  {'Threshold':>10} {'v1 Acc':>8} {'v1 ROI%':>8} {'v1 N':>6}  |  {'v2 Acc':>8} {'v2 ROI%':>8} {'v2 N':>6}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*6}  |  {'─'*8} {'─'*8} {'─'*6}")

    for v1e, v2e in zip(v1_sweep, v2_sweep):
        t = f"{v1e['threshold']*100:.0f}%"
        v1_acc = f"{v1e.get('accuracy', 0)*100:.1f}%" if v1e.get('total') else "—"
        v2_acc = f"{v2e.get('accuracy', 0)*100:.1f}%" if v2e.get('total') else "—"
        v1_roi = f"{v1e.get('roi_flat', {}).get('roi_pct', 0):.1f}%" if v1e.get('total') else "—"
        v2_roi = f"{v2e.get('roi_flat', {}).get('roi_pct', 0):.1f}%" if v2e.get('total') else "—"
        v1_n = str(v1e.get('total', 0))
        v2_n = str(v2e.get('total', 0))
        print(f"  {t:>10} {v1_acc:>8} {v1_roi:>8} {v1_n:>6}  |  {v2_acc:>8} {v2_roi:>8} {v2_n:>6}")
    print()


def print_calibration(v1_cal: List[Dict], v2_cal: List[Dict]):
    """Print calibration comparison table."""
    print_header("CALIBRATION (predicted confidence vs actual win rate)")
    print()
    print(f"  {'Bin':>10} {'v1 Actual':>10} {'v1 N':>6}  |  {'v2 Actual':>10} {'v2 N':>6}  | {'Expected':>9}")
    print(f"  {'─'*10} {'─'*10} {'─'*6}  |  {'─'*10} {'─'*6}  | {'─'*9}")

    for v1c, v2c in zip(v1_cal, v2_cal):
        b = v1c["bin"]
        v1_a = f"{v1c['actual_win_rate']:.1f}%"
        v2_a = f"{v2c['actual_win_rate']:.1f}%"
        exp_ = f"{v1c['expected_mid']:.1f}%"
        print(f"  {b:>10} {v1_a:>10} {v1c['count']:>6}  |  {v2_a:>10} {v2c['count']:>6}  | {exp_:>9}")
    print()


# ═════════════════════════════════════════════════════════════
#  BACKTEST ENGINES
# ═════════════════════════════════════════════════════════════
class BacktestEngine:
    """Core backtest runner."""

    def __init__(self, all_matches: List[Dict], db: DatabaseManager, leagues: List[str]):
        self.all_matches = all_matches  # sorted by date
        self.db = db
        self.leagues = leagues

    def _get_matches_in_seasons(self, seasons: List[int]) -> List[Dict]:
        """Filter matches by season list."""
        s_set = set(seasons)
        return [m for m in self.all_matches
                if m.get("season") in s_set and m.get("status") == "FINISHED"
                and m.get("home_score") is not None]

    def _get_matches_before_date(self, date_str: str) -> List[Dict]:
        """All matches strictly before a date."""
        return [m for m in self.all_matches
                if m.get("match_date", "") < date_str and m.get("status") == "FINISHED"]

    def run_holdout(self, train_seasons: List[int], test_seasons: List[int]) -> Tuple[List[Dict], List[Dict]]:
        """
        Simple holdout: train on train_seasons, predict test_seasons.
        Returns (v1_predictions, v2_predictions).
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"HOLDOUT: train={train_seasons[0]}-{train_seasons[-1]}, test={test_seasons[0]}-{test_seasons[-1]}")
        logger.info(f"{'='*60}")

        train_matches = self._get_matches_in_seasons(train_seasons)
        test_matches = self._get_matches_in_seasons(test_seasons)
        logger.info(f"Train matches: {len(train_matches)}, Test matches: {len(test_matches)}")

        if not train_matches or not test_matches:
            logger.error("Not enough data for holdout")
            return [], []

        # ── Train v1 ──
        logger.info("\n--- Training v1 (42 features) ---")
        X_v1, y_v1, _ = FeatureEngineer.build_training_data(train_matches, self.db)
        logger.info(f"v1 training data: X={X_v1.shape}, y={y_v1.shape}")
        v1_models = create_models(ML_SETTINGS, "_bt_v1", is_v2=False)
        v1_models, v1_ensemble, v1_stacking, v1_acc = train_models(v1_models, X_v1, y_v1, ML_SETTINGS)

        # ── Train v2 ──
        logger.info("\n--- Training v2 (83 features) ---")
        elo_train = EloTracker()
        X_v2, y_v2, _ = FeatureEngineerV2.build_training_data_v2(train_matches, self.db, elo_train)
        logger.info(f"v2 training data: X={X_v2.shape}, y={y_v2.shape}")
        v2_models = create_models(ML_SETTINGS_V2, "_bt_v2", is_v2=True)
        v2_models, v2_ensemble, v2_stacking, v2_acc = train_models(v2_models, X_v2, y_v2, ML_SETTINGS_V2)

        # Poisson calibration using train data
        poisson = PoissonModel()
        self._calibrate_poisson(poisson, train_matches)

        # ELO tracker — process all training matches
        elo_test = EloTracker()
        elo_test.process_matches(sorted(train_matches, key=lambda m: m.get("match_date", "")))

        # ── Predict test matches ──
        # Use db.get_team_stats() to match training feature distribution
        # (training also uses full-season stats, not incremental)
        logger.info(f"\n--- Predicting {len(test_matches)} test matches ---")
        team_idx = _build_team_index(train_matches)  # for v2 form/extras
        h2h_idx = _build_h2h_index(train_matches + test_matches)  # pre-build h2h
        test_sorted = sorted(test_matches, key=lambda m: m.get("match_date", ""))

        v1_preds = []
        v2_preds = []
        t_pred = time.time()
        for i, match in enumerate(test_sorted):
            if i > 0 and i % 200 == 0:
                rate = i / (time.time() - t_pred + 0.001)
                eta = (len(test_sorted) - i) / rate if rate > 0 else 0
                logger.info(f"  {i}/{len(test_sorted)} ({rate:.0f}/s, ETA {eta:.0f}s)")

            home = match["home_team_name"]
            away = match["away_team_name"]
            league = match.get("league_code", "")
            season = match.get("season", 2025)

            # Use DB stats (full-season) to match training features
            home_stats = self.db.get_team_stats(home, league, season)
            away_stats = self.db.get_team_stats(away, league, season)
            if not home_stats or home_stats.get("matches_played", 0) < 3:
                continue
            if not away_stats or away_stats.get("matches_played", 0) < 3:
                continue

            home_stats["team_name"] = home
            away_stats["team_name"] = away
            h2h = h2h_idx.get((home, away), [])[-10:]

            # v1
            p1 = predict_single_v1(match, v1_models, v1_ensemble, v1_stacking,
                                   home_stats, away_stats, h2h, ML_SETTINGS)
            if p1:
                v1_preds.append(p1)

            # v2 — pass team-specific past matches for form/extras
            home_past = team_idx.get(home, [])
            away_past = team_idx.get(away, [])
            p2 = predict_single_v2(match, v2_models, v2_ensemble, v2_stacking,
                                   home_stats, away_stats, h2h, ML_SETTINGS_V2,
                                   elo_test, poisson, home_past, away_past)
            if p2:
                v2_preds.append(p2)

            # Update team_idx for v2 form computation
            if match.get("status") == "FINISHED" and match.get("home_score") is not None:
                team_idx[home].append(match)
                team_idx[away].append(match)

            # Update ELO
            elo_test.update(home, away, int(match["home_score"]), int(match["away_score"]))

        elapsed_pred = time.time() - t_pred
        logger.info(f"Holdout complete: v1={len(v1_preds)}, v2={len(v2_preds)} predictions ({elapsed_pred:.1f}s)")
        return v1_preds, v2_preds

    def run_walk_forward(self, start_test_season: int = 2019) -> Tuple[List[Dict], List[Dict]]:
        """
        Walk-forward: for each test season, retrain on the preceding seasons,
        then predict all matches in the test season.
        Returns (v1_predictions_all, v2_predictions_all).
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"WALK-FORWARD: starting from test season {start_test_season}")
        logger.info(f"{'='*60}")

        available = [s for s in ALL_SEASONS if s <= max(ALL_SEASONS)]
        test_seasons = [s for s in available if s >= start_test_season]

        v1_all_preds = []
        v2_all_preds = []

        for test_season in test_seasons:
            # v1 trains on last 3 seasons before test
            v1_train_seasons = [s for s in available if s < test_season][-3:]
            # v2 trains on last 6 seasons before test
            v2_train_seasons = [s for s in available if s < test_season][-6:]

            if len(v1_train_seasons) < 2:
                logger.info(f"Skipping {test_season}: not enough v1 training seasons")
                continue

            logger.info(f"\n--- Walk-Forward: test={test_season}/{test_season+1} ---")
            logger.info(f"  v1 train: {v1_train_seasons}, v2 train: {v2_train_seasons}")

            v1_train_matches = self._get_matches_in_seasons(v1_train_seasons)
            v2_train_matches = self._get_matches_in_seasons(v2_train_seasons)
            test_matches = self._get_matches_in_seasons([test_season])

            logger.info(f"  v1 train: {len(v1_train_matches)} matches")
            logger.info(f"  v2 train: {len(v2_train_matches)} matches")
            logger.info(f"  test: {len(test_matches)} matches")

            if not test_matches:
                continue

            # Train v1
            X_v1, y_v1, _ = FeatureEngineer.build_training_data(v1_train_matches, self.db)
            if len(X_v1) < 100:
                logger.warning(f"  v1: only {len(X_v1)} samples, skipping")
                continue
            v1_models = create_models(ML_SETTINGS, f"_wf_v1_{test_season}", is_v2=False)
            v1_models, v1_ens, v1_stack, _ = train_models(v1_models, X_v1, y_v1, ML_SETTINGS)

            # Train v2
            elo_train = EloTracker()
            X_v2, y_v2, _ = FeatureEngineerV2.build_training_data_v2(v2_train_matches, self.db, elo_train)
            if len(X_v2) < 100:
                logger.warning(f"  v2: only {len(X_v2)} samples, skipping")
                continue
            v2_models = create_models(ML_SETTINGS_V2, f"_wf_v2_{test_season}", is_v2=True)
            v2_models, v2_ens, v2_stack, _ = train_models(v2_models, X_v2, y_v2, ML_SETTINGS_V2)

            # Poisson calibration on v2 train data
            poisson = PoissonModel()
            self._calibrate_poisson(poisson, v2_train_matches)

            # ELO for test predictions — built from ALL matches before test season
            elo_test = EloTracker()
            all_before = self._get_matches_in_seasons([s for s in available if s < test_season])
            elo_test.process_matches(sorted(all_before, key=lambda m: m.get("match_date", "")))

            # Predict test season — use DB stats to match training
            all_train = v2_train_matches
            team_idx = _build_team_index(all_train)
            test_sorted = sorted(test_matches, key=lambda m: m.get("match_date", ""))

            for match in test_sorted:
                home = match["home_team_name"]
                away = match["away_team_name"]
                league = match.get("league_code", "")
                season_m = match.get("season", 2025)

                home_stats = self.db.get_team_stats(home, league, season_m)
                away_stats = self.db.get_team_stats(away, league, season_m)
                if not home_stats or home_stats.get("matches_played", 0) < 3:
                    elo_test.update(home, away, int(match["home_score"]), int(match["away_score"]))
                    continue
                if not away_stats or away_stats.get("matches_played", 0) < 3:
                    elo_test.update(home, away, int(match["home_score"]), int(match["away_score"]))
                    continue

                home_stats["team_name"] = home
                away_stats["team_name"] = away
                h2h = self.db.get_h2h(home, away) or []

                p1 = predict_single_v1(match, v1_models, v1_ens, v1_stack,
                                       home_stats, away_stats, h2h, ML_SETTINGS)
                if p1:
                    p1["fold_season"] = test_season
                    v1_all_preds.append(p1)

                home_past = team_idx.get(home, [])
                away_past = team_idx.get(away, [])
                p2 = predict_single_v2(match, v2_models, v2_ens, v2_stack,
                                       home_stats, away_stats, h2h, ML_SETTINGS_V2,
                                       elo_test, poisson, home_past, away_past)
                if p2:
                    p2["fold_season"] = test_season
                    v2_all_preds.append(p2)

                if match.get("status") == "FINISHED" and match.get("home_score") is not None:
                    team_idx[home].append(match)
                    team_idx[away].append(match)

                elo_test.update(home, away, int(match["home_score"]), int(match["away_score"]))

            logger.info(f"  Fold {test_season}: v1={len([p for p in v1_all_preds if p.get('fold_season')==test_season])}, "
                        f"v2={len([p for p in v2_all_preds if p.get('fold_season')==test_season])}")

        logger.info(f"\nWalk-forward complete: v1={len(v1_all_preds)}, v2={len(v2_all_preds)} total predictions")
        return v1_all_preds, v2_all_preds

    @staticmethod
    def _calibrate_poisson(poisson: PoissonModel, matches: List[Dict]):
        """Calibrate Poisson from match list (no DB needed)."""
        total_home = 0
        total_away = 0
        n = 0
        for m in matches:
            hs = m.get("home_score")
            aws = m.get("away_score")
            if hs is not None and aws is not None:
                total_home += int(hs)
                total_away += int(aws)
                n += 1
        if n >= 50:
            poisson.avg_home_goals = total_home / n
            poisson.avg_away_goals = total_away / n
            logger.info(f"  Poisson calibrated: home={poisson.avg_home_goals:.3f}, "
                        f"away={poisson.avg_away_goals:.3f} ({n} matches)")


# ═════════════════════════════════════════════════════════════
#  FULL REPORT
# ═════════════════════════════════════════════════════════════
def generate_report(v1_preds: List[Dict], v2_preds: List[Dict], label: str):
    """Generate full comparison report for one backtest mode."""
    v1_metrics = compute_metrics(v1_preds)
    v2_metrics = compute_metrics(v2_preds)

    print_comparison(v1_metrics, v2_metrics, label)

    v1_league = compute_per_league(v1_preds)
    v2_league = compute_per_league(v2_preds)
    print_per_league(v1_league, v2_league)

    v1_edge = compute_edge_threshold_sweep(v1_preds)
    v2_edge = compute_edge_threshold_sweep(v2_preds)
    print_edge_sweep(v1_edge, v2_edge)

    v1_cal = compute_calibration(v1_preds)
    v2_cal = compute_calibration(v2_preds)
    print_calibration(v1_cal, v2_cal)

    # Betting experiments (ALL matches, filters, baselines, opposite, etc.)
    v1_exps = compute_betting_experiments(v1_preds)
    v2_exps = compute_betting_experiments(v2_preds)
    print_betting_experiments(v1_exps, v2_exps)

    return {
        "label": label,
        "v1": v1_metrics,
        "v2": v2_metrics,
        "v1_per_league": {k: v for k, v in v1_league.items()},
        "v2_per_league": {k: v for k, v in v2_league.items()},
        "v1_edge_sweep": v1_edge,
        "v2_edge_sweep": v2_edge,
        "v1_calibration": v1_cal,
        "v2_calibration": v2_cal,
        "v1_experiments": v1_exps,
        "v2_experiments": v2_exps,
    }


# ═════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Historical Backtest — v1 vs v2")
    parser.add_argument("--holdout", action="store_true", help="Run holdout backtest only")
    parser.add_argument("--walk-forward", action="store_true", dest="walk_forward", help="Run walk-forward only")
    parser.add_argument("--leagues", type=str, default=None,
                        help="Comma-separated league codes (e.g. PL,PD). Default: all CSV leagues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    parser.add_argument("--analyze", type=str, default=None,
                        help="Re-run analysis on saved predictions JSON (skip training)")
    args = parser.parse_args()

    # ── Analyze-only mode: re-run experiments on saved predictions ──
    if args.analyze:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        pred_path = Path(args.analyze)
        if not pred_path.exists():
            print(f"File not found: {pred_path}")
            return
        with open(pred_path) as f:
            raw_preds = json.load(f)
        for mode_name, mode_data in raw_preds.items():
            v1_preds = mode_data.get("v1", [])
            v2_preds = mode_data.get("v2", [])
            if v1_preds or v2_preds:
                generate_report(v1_preds, v2_preds, f"Re-analysis: {mode_name}")
        return

    # Neither flag means both
    run_holdout = args.holdout or (not args.holdout and not args.walk_forward)
    run_wf = args.walk_forward or (not args.holdout and not args.walk_forward)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT / "logs" / "backtest.log", mode="w"),
        ],
    )
    # Suppress noisy libraries
    for lib in ["tensorflow", "absl", "h5py", "urllib3", "lightgbm"]:
        logging.getLogger(lib).setLevel(logging.WARNING)

    leagues = args.leagues.split(",") if args.leagues else CSV_LEAGUES
    leagues = [l.strip().upper() for l in leagues]
    logger.info(f"Leagues: {leagues}")

    # ── Load data ──
    t0 = time.time()
    print_header("LOADING CSV DATA")
    all_matches = load_all_csv_data(leagues)
    if not all_matches:
        logger.error("No matches loaded! Exiting.")
        return

    # ── Setup backtest DB (use temp dir to avoid WAL lock issues) ──
    import tempfile
    bt_db_dir = Path(tempfile.mkdtemp(prefix="backtest_"))
    bt_db_path = bt_db_dir / "backtest.db"
    logger.info(f"Backtest DB: {bt_db_path}")
    db = DatabaseManager(db_path=str(bt_db_path))
    populate_db(db, all_matches)

    engine = BacktestEngine(all_matches, db, leagues)
    results = {}

    raw_preds = {}  # store raw predictions for later re-analysis

    # ── Holdout ──
    if run_holdout:
        print_header("HOLDOUT BACKTEST")
        train_seasons = list(range(2015, 2023))   # 2015-2022
        test_seasons = list(range(2023, 2026))     # 2023-2025
        v1_h, v2_h = engine.run_holdout(train_seasons, test_seasons)
        if v1_h or v2_h:
            results["holdout"] = generate_report(v1_h, v2_h, f"Holdout (train {train_seasons[0]}-{train_seasons[-1]}, test {test_seasons[0]}-{test_seasons[-1]})")
            raw_preds["holdout"] = {"v1": v1_h, "v2": v2_h}

    # ── Walk-Forward ──
    if run_wf:
        print_header("WALK-FORWARD BACKTEST")
        v1_wf, v2_wf = engine.run_walk_forward(start_test_season=2019)
        if v1_wf or v2_wf:
            results["walk_forward"] = generate_report(v1_wf, v2_wf, "Walk-Forward (2019-2025)")
            raw_preds["walk_forward"] = {"v1": v1_wf, "v2": v2_wf}

    # ── Save results to JSON ──
    output_path = ROOT / "data" / "backtest_results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\nResults saved to {output_path}")
    except Exception as e:
        logger.warning(f"Could not save JSON: {e}")

    # ── Save raw predictions for later re-analysis ──
    raw_path = ROOT / "data" / "backtest_predictions.json"
    try:
        with open(raw_path, "w") as f:
            json.dump(raw_preds, f, indent=2, default=str)
        logger.info(f"Raw predictions saved to {raw_path}")
    except Exception as e:
        logger.warning(f"Could not save raw predictions: {e}")

    elapsed = time.time() - t0
    print_header(f"BACKTEST COMPLETE — {elapsed/60:.1f} minutes")
    print()


if __name__ == "__main__":
    main()
