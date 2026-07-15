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
from itertools import groupby
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

DEFAULT_STARTING_BANKROLL = 10_000.0
DEFAULT_SINGLE_STAKE = 100.0
DEFAULT_COUPON_STAKE = 100.0
DEFAULT_COUPON_MIN_LEGS = 2
DEFAULT_COUPON_MAX_LEGS = 4
MAX_ROBUST_DRAWDOWN = DEFAULT_STARTING_BANKROLL * 0.35
NEGATIVE_SEASON_PENALTY = DEFAULT_STARTING_BANKROLL * 0.15
TOP_LEAGUE_CODES = {"PL", "PD", "BL1", "SA", "FL1", "DED", "PPL"}
OPT_CONF_THRESHOLDS = [x / 100 for x in range(40, 90, 5)]
OPT_EDGE_THRESHOLDS = [None, 0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
OPT_COUPON_MAX_LEGS = [2, 3, 4, 5, 6]
OPT_COUPON_SORTS = ["confidence", "edge", "edge_x_confidence"]
OPT_COUPON_MAX_PER_LEAGUE = [1, 2, 3]
OPT_SINGLE_BET_STYLES = ["model_pick", "least_likely", "market_underdog"]
PATTERN_MIN_MATCHES = [2, 3, 4, 5]
PATTERN_MIN_HIT_RATES = [0.60, 0.67, 0.75, 0.80, 0.90, 1.00]
PATTERN_MAX_ODDS = [None, 4.0, 3.0, 2.5, 2.0]
HISTORY_EDGE_LABEL_STYLES = [
    "model_pick",
    "max_model_edge",
    "market_favorite",
    "market_underdog",
    "historical_outcome",
]
HISTORY_EDGE_CONF_THRESHOLDS = [None, 0.40, 0.50, 0.55, 0.60, 0.65]
HISTORY_EDGE_MODEL_EDGE_THRESHOLDS = [None, 0.0, 0.02, 0.05, 0.08, 0.10]
HISTORY_EDGE_FILTERS = [(2, 0.60), (2, 0.75), (3, 0.67), (3, 0.80), (4, 0.80), (4, 1.00)]
HISTORY_EDGE_ODDS_BANDS = [None, (1.20, 2.50), (1.50, 3.00), (1.80, 3.50), (2.00, 4.00)]
HISTORY_EDGE_COUPON_TOP_FILTERS = 60
CSV_HISTORY_LABEL_STYLES = ["historical_outcome", "market_favorite", "market_underdog"]
CSV_HISTORY_MIN_MATCHES = [2, 3, 4, 5, 6]
CSV_HISTORY_MIN_RATES = [0.60, 0.67, 0.75, 0.80, 0.90, 1.00]
CSV_HISTORY_EDGE_THRESHOLDS = [None, 0.0, 0.02, 0.05, 0.08, 0.10]
STRATEGY_ZOO_ODDS_BANDS = [
    None,
    (1.01, 1.50),
    (1.20, 2.00),
    (1.20, 2.50),
    (1.50, 2.50),
    (1.50, 3.00),
    (2.00, 3.50),
    (3.00, 8.00),
]
STRATEGY_ZOO_HISTORY_COUNTS = [2, 3, 5, 6, 10]
STRATEGY_ZOO_HISTORY_RATES = [0.55, 0.60, 0.67, 0.75, 0.80, 0.90]
STRATEGY_ZOO_EDGE_THRESHOLDS = [None, 0.0, 0.02, 0.05, 0.08]
STRATEGY_ZOO_COUPON_TOP_FILTERS = 25
STRATEGY_ZOO_WF_SOURCE_ALLOWLIST = {
    "market_favorite",
    "direct_h2h",
    "pair_history",
    "favorite_direct_h2h_agree",
    "favorite_pair_agree",
    "home_any_history",
    "away_any_history",
}
STRATEGY_ZOO_WF_ODDS_BANDS = [None, (1.20, 2.00), (1.20, 2.50), (1.50, 3.00)]
STRATEGY_ZOO_WF_HISTORY_COUNTS = [2, 3, 5, 6, 10]
STRATEGY_ZOO_WF_HISTORY_RATES = [0.60, 0.67, 0.75, 0.80]
STRATEGY_ZOO_WF_EDGE_THRESHOLDS = [None, 0.0, 0.02, 0.05]
STRATEGY_ZOO_WF_MIN_TRAIN_BETS = 150
STRATEGY_ZOO_WF_COUPON_TOP_FILTERS = 8
H2H_COUPON_SWEEP_SOURCES = ["direct_h2h", "favorite_direct_h2h_agree", "favorite_pair_agree"]
H2H_COUPON_SWEEP_COUNTS = [8, 10, 12]
H2H_COUPON_SWEEP_RATES = [0.75, 0.80, 0.85]
H2H_COUPON_SWEEP_EDGES = [0.02, 0.05]
H2H_COUPON_SWEEP_ODDS_BANDS = [(1.20, 2.00), (1.20, 2.50), (1.50, 2.50)]
H2H_COUPON_SWEEP_ODDS_BASES = ["b365", "b365_close", "avg", "avg_close"]
H2H_COUPON_SWEEP_MIN_LEAGUE_MATCHES = [0, 100]
H2H_COUPON_SWEEP_FORM_THRESHOLDS = [None, 0.40]
H2H_COUPON_SWEEP_COMBINED_ODDS_MAX = [None, 4.0, 5.0, 6.0]
H2H_COUPON_SWEEP_TOP_FILTERS = 25
OU_UNDER = 0
OU_OVER = 1
OU_OUTCOME_MAP = {OU_UNDER: "UNDER_2_5", OU_OVER: "OVER_2_5"}
OU_CONF_THRESHOLDS = [None, 0.52, 0.55, 0.58, 0.60, 0.65]
OU_EDGE_THRESHOLDS = [None, 0.0, 0.02, 0.05, 0.08, 0.10]
OU_MIN_TEAM_MATCHES = [3, 5, 8, 10, 15]
OU_MIN_LEAGUE_MATCHES = [25, 50, 100]
OU_MIN_PAIR_MATCHES = [2, 3, 5]
OU_RATE_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]
OU_ODDS_BANDS = [None, (1.20, 1.60), (1.40, 1.90), (1.50, 2.20), (1.80, 2.80)]
OU_WF_SOURCE_ALLOWLIST = {
    "market_favorite",
    "market_underdog",
    "poisson_total",
    "poisson_edge",
    "recent_team_total_rate",
    "league_total_rate",
    "pair_total_history",
    "market_poisson_agree",
}


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
        _append_h2h_match(idx, m)
    return idx


def _append_h2h_match(idx: Dict[Tuple[str, str], List[Dict]], match: Dict):
    """Append a finished match to the H2H index in both team directions."""
    if match.get("status") != "FINISHED" or match.get("home_score") is None:
        return
    h = match["home_team_name"]
    a = match["away_team_name"]
    rec = {
        "home_team": h, "away_team": a,
        "home_score": int(match["home_score"]), "away_score": int(match["away_score"]),
        "match_date": match.get("match_date", ""), "season": match.get("season", 0),
    }
    idx[(h, a)].append(rec)
    idx[(a, h)].append(rec)


def _record_finished_match(
    team_idx: Dict[str, List[Dict]],
    h2h_idx: Dict[Tuple[str, str], List[Dict]],
    match: Dict,
) -> bool:
    """Add a completed test match to rolling backtest history."""
    if match.get("status") != "FINISHED" or match.get("home_score") is None:
        return False
    home = match["home_team_name"]
    away = match["away_team_name"]
    team_idx[home].append(match)
    team_idx[away].append(match)
    _append_h2h_match(h2h_idx, match)
    return True


def _compute_team_stats_snapshot(team_matches: List[Dict], team_name: str,
                                 league_code: str, season: int) -> Optional[Dict]:
    """Compute a pre-match team snapshot from past matches in the same league/season."""
    scoped = [
        m for m in team_matches
        if m.get("league_code") == league_code
        and m.get("season") == season
        and m.get("status") == "FINISHED"
        and m.get("home_score") is not None
    ]
    if len(scoped) < 3:
        return None
    return _compute_stats_from_list(team_name, league_code, season, scoped)


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


def _build_training_data_v1_fast(matches: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build v1 training rows from rolling in-memory match history."""
    X_list = []
    y_list = []
    date_list = []
    team_idx: Dict[str, List[Dict]] = defaultdict(list)
    h2h_idx: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

    for match in sorted(matches, key=lambda m: m.get("match_date", "")):
        if match.get("status") != "FINISHED" or match.get("home_score") is None:
            continue

        home = match["home_team_name"]
        away = match["away_team_name"]
        league = match.get("league_code", "")
        season = match.get("season", 2025)
        home_past = team_idx.get(home, [])
        away_past = team_idx.get(away, [])
        home_stats = _compute_team_stats_snapshot(home_past, home, league, season)
        away_stats = _compute_team_stats_snapshot(away_past, away, league, season)

        if home_stats and away_stats:
            home_stats["team_name"] = home
            away_stats["team_name"] = away
            h2h = h2h_idx.get((home, away), [])[-10:]
            try:
                features = FeatureEngineer.build_match_features(
                    home_stats, away_stats, h2h,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                )
                hs = int(match["home_score"])
                aws = int(match["away_score"])
                label = LABEL_HOME if hs > aws else (LABEL_DRAW if hs == aws else LABEL_AWAY)
                X_list.append(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))
                y_list.append(label)
                date_list.append(match.get("match_date", ""))
            except Exception as e:
                logger.error(f"Fast v1 feature error for {home} vs {away}: {e}")

        _record_finished_match(team_idx, h2h_idx, match)

    if not X_list:
        return np.empty((0, len(FeatureEngineer.FEATURE_NAMES))), np.empty(0), []
    return np.array(X_list), np.array(y_list), date_list


def _build_training_data_v2_fast(matches: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build v2 training rows from rolling in-memory match history."""
    X_list = []
    y_list = []
    date_list = []
    team_idx: Dict[str, List[Dict]] = defaultdict(list)
    h2h_idx: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    elo_tracker = EloTracker()

    for match in sorted(matches, key=lambda m: m.get("match_date", "")):
        if match.get("status") != "FINISHED" or match.get("home_score") is None:
            continue

        home = match["home_team_name"]
        away = match["away_team_name"]
        league = match.get("league_code", "")
        season = match.get("season", 2025)
        match_date = match.get("match_date", "")
        home_past = team_idx.get(home, [])
        away_past = team_idx.get(away, [])
        home_stats = _compute_team_stats_snapshot(home_past, home, league, season)
        away_stats = _compute_team_stats_snapshot(away_past, away, league, season)

        if home_stats and away_stats:
            home_stats["team_name"] = home
            away_stats["team_name"] = away
            h2h = h2h_idx.get((home, away), [])[-10:]
            try:
                features = FeatureEngineerV2.build_match_features_v2(
                    home_stats, away_stats, h2h,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                    ai_predictions=None,
                    elo_tracker=elo_tracker,
                    home_form_list=FeatureEngineerV2.compute_form_list(home_past, home),
                    away_form_list=FeatureEngineerV2.compute_form_list(away_past, away),
                    home_extra=FeatureEngineerV2.compute_csv_extra_averages(home_past, home),
                    away_extra=FeatureEngineerV2.compute_csv_extra_averages(away_past, away),
                    home_days_rest=FeatureEngineerV2.compute_days_since_last(home_past, home, match_date),
                    away_days_rest=FeatureEngineerV2.compute_days_since_last(away_past, away, match_date),
                    home_recent_goals_avg=FeatureEngineerV2.compute_recent_goals_avg(home_past, home),
                    away_recent_goals_avg=FeatureEngineerV2.compute_recent_goals_avg(away_past, away),
                    is_training=True,
                    league_code=league,
                    matchday=match.get("matchday", 0) or 0,
                    total_matchdays=38,
                    match_datetime=match_date,
                    home_sos=FeatureEngineerV2.compute_sos(home_past, home, elo_tracker),
                    away_sos=FeatureEngineerV2.compute_sos(away_past, away, elo_tracker),
                )
                hs = int(match["home_score"])
                aws = int(match["away_score"])
                label = LABEL_HOME if hs > aws else (LABEL_DRAW if hs == aws else LABEL_AWAY)
                X_list.append(np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))
                y_list.append(label)
                date_list.append(match_date)
            except Exception as e:
                logger.error(f"Fast v2 feature error for {home} vs {away}: {e}")

        _record_finished_match(team_idx, h2h_idx, match)
        elo_tracker.update(home, away, int(match["home_score"]), int(match["away_score"]))

    if not X_list:
        return np.empty((0, len(FeatureEngineerV2.FEATURE_NAMES))), np.empty(0), []
    return np.array(X_list), np.array(y_list), date_list


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


def _normalize_prob_matrix(probs: np.ndarray, n_rows: int) -> np.ndarray:
    """Return an n x 3 probability matrix with sane defaults."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] == 1 and n_rows > 1:
        arr = np.repeat(arr, n_rows, axis=0)
    if arr.shape[1] != 3:
        raise ValueError(f"Expected 3 probability columns, got {arr.shape}")
    arr = np.nan_to_num(arr[:, :3], nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    empty_rows = row_sums[:, 0] <= 0
    np.divide(arr, row_sums, out=arr, where=row_sums > 0)
    if np.any(empty_rows):
        arr[empty_rows] = [0.33, 0.33, 0.34]
    return arr


def _model_probs_batch(model, X: np.ndarray) -> np.ndarray:
    """Batch predict one model, falling back to neutral probabilities."""
    n_rows = len(X)
    if not getattr(model, "is_trained", False):
        return np.repeat([[0.33, 0.33, 0.34]], n_rows, axis=0)
    try:
        return _normalize_prob_matrix(model.predict_proba(X), n_rows)
    except Exception as e:
        logger.warning(f"Batch prediction failed for {getattr(model, 'name', '?')}: {e}")
        return np.repeat([[0.33, 0.33, 0.34]], n_rows, axis=0)


def _ensemble_probs_batch(
    models: Dict[str, object],
    weights: Dict[str, float],
    stacking: Optional[StackingEnsemble],
    X: np.ndarray,
) -> np.ndarray:
    """Batch equivalent of StackingEnsemble/EnsembleModel predict_proba."""
    n_rows = len(X)
    if n_rows == 0:
        return np.empty((0, 3))

    if stacking and stacking.is_trained and getattr(stacking, "meta_model", None) is not None:
        meta_X = np.zeros((n_rows, 3 * len(models)))
        for m_idx, (_, model) in enumerate(models.items()):
            meta_X[:, m_idx * 3:(m_idx + 1) * 3] = _model_probs_batch(model, X)
        return _normalize_prob_matrix(stacking.meta_model.predict_proba(meta_X), n_rows)

    weighted = np.zeros((n_rows, 3))
    total_weight = 0.0
    for model_name, model in models.items():
        if not getattr(model, "is_trained", False):
            continue
        weight = weights.get(model_name, 0.33)
        weighted += _model_probs_batch(model, X) * weight
        total_weight += weight

    if total_weight <= 0:
        return np.repeat([[0.33, 0.33, 0.34]], n_rows, axis=0)
    return _normalize_prob_matrix(weighted / total_weight, n_rows)


def _prediction_from_probs(row: Dict, probs: np.ndarray, version: str) -> Dict:
    """Create the stored backtest prediction dict from probabilities."""
    predicted_label = int(np.argmax(probs))
    confidence = float(probs[predicted_label])
    home_odds = row.get("home_odds")
    draw_odds = row.get("draw_odds")
    away_odds = row.get("away_odds")

    edge = 0.0
    if home_odds and draw_odds and away_odds and home_odds > 1 and draw_odds > 1 and away_odds > 1:
        inv_sum = (1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds)
        fair_odds = [1.0 / home_odds / inv_sum, 1.0 / draw_odds / inv_sum, 1.0 / away_odds / inv_sum]
        edge = float(probs[predicted_label] - fair_odds[predicted_label])

    actual_odds_map = {LABEL_HOME: home_odds, LABEL_DRAW: draw_odds, LABEL_AWAY: away_odds}
    sel_odds = actual_odds_map.get(predicted_label)
    kelly = 0.0
    if sel_odds and sel_odds > 1.0:
        b = sel_odds - 1.0
        kelly = max(0.0, (b * confidence - (1.0 - confidence)) / b)
        kelly = min(kelly, 0.25)

    result = {
        "match_date": row["match_date"],
        "league": row["league"],
        "season": row["season"],
        "home": row["home"],
        "away": row["away"],
        "home_score": row["home_score"],
        "away_score": row["away_score"],
        "actual": row["actual"],
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
        "version": version,
    }
    for key in ("poisson_score", "btts_prob", "over25_prob", "fold_season"):
        if key in row:
            result[key] = row[key]
    return result


def _predict_rows_batch(
    rows: List[Dict],
    models: Dict[str, object],
    config: Dict,
    stacking: Optional[StackingEnsemble],
    version: str,
) -> List[Dict]:
    """Batch predict prepared feature rows."""
    if not rows:
        return []
    X = np.vstack([r["features"] for r in rows])
    probs_matrix = _ensemble_probs_batch(models, config.get("ensemble", {}).get("weights", {}), stacking, X)
    return [_prediction_from_probs(row, probs, version) for row, probs in zip(rows, probs_matrix)]


def _predict_v2_rows_batch(
    rows: List[Dict],
    models: Dict[str, object],
    config: Dict,
    stacking: Optional[StackingEnsemble],
) -> List[Dict]:
    """Batch predict v2 rows and apply the same Poisson blend as single-match mode."""
    if not rows:
        return []
    X = np.vstack([r["features"] for r in rows])
    base_probs = _ensemble_probs_batch(models, config.get("ensemble", {}).get("weights", {}), stacking, X)
    blended_rows = []
    for row, probs in zip(rows, base_probs):
        poisson_probs = row["poisson_probs"]
        blended = np.array([
            probs[0] * 0.95 + poisson_probs["home_win"] * 0.05,
            probs[1] * 0.95 + poisson_probs["draw"] * 0.05,
            probs[2] * 0.95 + poisson_probs["away_win"] * 0.05,
        ])
        blended_rows.append(blended)
    probs_matrix = _normalize_prob_matrix(np.vstack(blended_rows), len(rows))
    return [_prediction_from_probs(row, probs, "v2") for row, probs in zip(rows, probs_matrix)]


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


def _prediction_odds(prediction: Dict, label: Optional[int] = None) -> Optional[float]:
    """Return decimal odds for a selected 1X2 label."""
    selected = prediction.get("predicted") if label is None else label
    odds_map = {
        LABEL_HOME: prediction.get("home_odds"),
        LABEL_DRAW: prediction.get("draw_odds"),
        LABEL_AWAY: prediction.get("away_odds"),
    }
    try:
        odds = float(odds_map.get(selected) or 0)
    except (TypeError, ValueError):
        return None
    return odds if odds > 1.0 else None


def _prediction_strength(prediction: Dict) -> float:
    """Ranking score for choosing the strongest bets inside a day/coupon."""
    confidence = float(prediction.get("confidence") or 0.0)
    edge = float(prediction.get("edge") or 0.0)
    return max(edge, 0.0) * confidence


def _coupon_sort_key(prediction: Dict, sort_by: str) -> Tuple:
    """Sort key for selecting the strongest coupon legs."""
    confidence = float(prediction.get("confidence") or 0.0)
    edge = float(prediction.get("edge") or 0.0)
    odds = _prediction_odds(prediction) or 0.0
    if sort_by == "confidence":
        return (confidence, edge, odds)
    if sort_by == "edge":
        return (edge, confidence, odds)
    return (_prediction_strength(prediction), confidence, edge, odds)


def _sorted_predictions(predictions: List[Dict]) -> List[Dict]:
    """Stable chronological ordering for bankroll simulations."""
    return sorted(
        predictions,
        key=lambda p: (
            p.get("match_date", ""),
            p.get("league", ""),
            p.get("home", ""),
            p.get("away", ""),
        ),
    )


def _season_key(prediction: Dict) -> str:
    season = prediction.get("season")
    if season is not None:
        return str(season)
    match_date = prediction.get("match_date") or ""
    return match_date[:4] if match_date else "unknown"


def _empty_period_stats() -> Dict:
    return {
        "bets": 0,
        "wins": 0,
        "staked": 0.0,
        "returned": 0.0,
        "profit": 0.0,
        "roi_pct": 0.0,
        "accuracy": 0.0,
    }


def _finalize_period_stats(periods: Dict[str, Dict]) -> Dict[str, Dict]:
    finalized = {}
    for key in sorted(periods):
        stats = dict(periods[key])
        stats["profit"] = round(stats["returned"] - stats["staked"], 2)
        stats["roi_pct"] = (stats["profit"] / stats["staked"] * 100) if stats["staked"] else 0.0
        stats["accuracy"] = (stats["wins"] / stats["bets"] * 100) if stats["bets"] else 0.0
        stats["staked"] = round(stats["staked"], 2)
        stats["returned"] = round(stats["returned"], 2)
        stats["roi_pct"] = round(stats["roi_pct"], 2)
        stats["accuracy"] = round(stats["accuracy"], 2)
        finalized[key] = stats
    return finalized


def _build_coupon_batches(
    predictions: List[Dict],
    min_legs: int = DEFAULT_COUPON_MIN_LEGS,
    max_legs: int = DEFAULT_COUPON_MAX_LEGS,
    sort_by: str = "edge_x_confidence",
    max_per_league: Optional[int] = None,
) -> Tuple[List[List[Dict]], int]:
    """Group selected predictions into deterministic day-based coupon batches."""
    by_day: Dict[str, List[Dict]] = defaultdict(list)
    skipped_no_odds = 0

    for prediction in _sorted_predictions(predictions):
        if _prediction_odds(prediction) is None:
            skipped_no_odds += 1
            continue
        day = (prediction.get("match_date") or "")[:10] or "unknown"
        by_day[day].append(prediction)

    batches: List[List[Dict]] = []
    for day in sorted(by_day):
        remaining = sorted(
            by_day[day],
            key=lambda p: _coupon_sort_key(p, sort_by),
            reverse=True,
        )

        if max_per_league is None:
            for start in range(0, len(remaining), max_legs):
                legs = remaining[start:start + max_legs]
                if len(legs) >= min_legs:
                    batches.append(legs)
            continue

        while remaining:
            legs = []
            used_ids = set()
            league_counts: Dict[str, int] = defaultdict(int)
            for idx, prediction in enumerate(remaining):
                league = prediction.get("league", "UNK")
                if league_counts[league] >= max_per_league:
                    continue
                legs.append(prediction)
                used_ids.add(idx)
                league_counts[league] += 1
                if len(legs) >= max_legs:
                    break

            if len(legs) < min_legs:
                break
            batches.append(legs)
            remaining = [p for idx, p in enumerate(remaining) if idx not in used_ids]

    return batches, skipped_no_odds


def simulate_flat_bankroll(
    predictions: List[Dict],
    starting_bankroll: float = DEFAULT_STARTING_BANKROLL,
    stake: float = DEFAULT_SINGLE_STAKE,
    bet_label_fn=None,
) -> Dict:
    """
    Simulate a real bankroll: start with 10,000 and stake the same amount
    on every selected match. Stops if the bankroll cannot fund the next bet.
    """
    bankroll = float(starting_bankroll)
    peak = bankroll
    min_bankroll = bankroll
    max_drawdown = 0.0
    staked = 0.0
    returned = 0.0
    bets = 0
    wins = 0
    skipped_no_odds = 0
    stopped = False
    season_stats: Dict[str, Dict] = defaultdict(_empty_period_stats)

    for prediction in _sorted_predictions(predictions):
        label = bet_label_fn(prediction) if bet_label_fn else prediction.get("predicted")
        odds = _prediction_odds(prediction, label)
        if odds is None:
            skipped_no_odds += 1
            continue
        if bankroll < stake:
            stopped = True
            break

        bankroll -= stake
        staked += stake
        bets += 1

        payout = 0.0
        if label == prediction.get("actual"):
            payout = stake * odds
            wins += 1
        bankroll += payout
        returned += payout

        season = _season_key(prediction)
        season_stats[season]["bets"] += 1
        season_stats[season]["wins"] += 1 if payout else 0
        season_stats[season]["staked"] += stake
        season_stats[season]["returned"] += payout

        peak = max(peak, bankroll)
        min_bankroll = min(min_bankroll, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)

    profit = bankroll - starting_bankroll
    roi_pct = (profit / staked * 100) if staked else 0.0
    growth_pct = (profit / starting_bankroll * 100) if starting_bankroll else 0.0
    max_drawdown_pct = (max_drawdown / peak * 100) if peak else 0.0

    return {
        "starting_bankroll": round(starting_bankroll, 2),
        "stake": round(stake, 2),
        "bets": bets,
        "wins": wins,
        "accuracy": round((wins / bets * 100) if bets else 0.0, 2),
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "final_bankroll": round(bankroll, 2),
        "profit": round(profit, 2),
        "growth_pct": round(growth_pct, 2),
        "roi_pct": round(roi_pct, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "min_bankroll": round(min_bankroll, 2),
        "stopped_bankroll_depleted": stopped,
        "skipped_no_odds": skipped_no_odds,
        "by_season": _finalize_period_stats(season_stats),
    }


def simulate_coupon_batches(
    batches: List[List[Dict]],
    starting_bankroll: float = DEFAULT_STARTING_BANKROLL,
    stake: float = DEFAULT_COUPON_STAKE,
    min_legs: int = DEFAULT_COUPON_MIN_LEGS,
    max_legs: int = DEFAULT_COUPON_MAX_LEGS,
    sort_by: str = "edge_x_confidence",
    max_per_league: Optional[int] = None,
    skipped_no_odds: int = 0,
) -> Dict:
    """Simulate already-built coupon batches in chronological order."""
    bankroll = float(starting_bankroll)
    peak = bankroll
    min_bankroll = bankroll
    max_drawdown = 0.0
    staked = 0.0
    returned = 0.0
    coupons = 0
    winning_coupons = 0
    legs_played = 0
    stopped = False
    season_stats: Dict[str, Dict] = defaultdict(_empty_period_stats)

    for legs in batches:
        if bankroll < stake:
            stopped = True
            break

        combined_odds = 1.0
        all_correct = True
        for leg in legs:
            combined_odds *= _prediction_odds(leg) or 1.0
            all_correct = all_correct and leg.get("predicted") == leg.get("actual")

        bankroll -= stake
        staked += stake
        coupons += 1
        legs_played += len(legs)

        payout = 0.0
        if all_correct:
            payout = stake * combined_odds
            winning_coupons += 1
        bankroll += payout
        returned += payout

        season = _season_key(legs[0])
        season_stats[season]["bets"] += 1
        season_stats[season]["wins"] += 1 if payout else 0
        season_stats[season]["staked"] += stake
        season_stats[season]["returned"] += payout

        peak = max(peak, bankroll)
        min_bankroll = min(min_bankroll, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)

    profit = bankroll - starting_bankroll
    roi_pct = (profit / staked * 100) if staked else 0.0
    growth_pct = (profit / starting_bankroll * 100) if starting_bankroll else 0.0
    max_drawdown_pct = (max_drawdown / peak * 100) if peak else 0.0

    return {
        "starting_bankroll": round(starting_bankroll, 2),
        "stake": round(stake, 2),
        "min_legs": min_legs,
        "max_legs": max_legs,
        "sort_by": sort_by,
        "max_per_league": max_per_league,
        "coupons": coupons,
        "winning_coupons": winning_coupons,
        "coupon_hit_rate": round((winning_coupons / coupons * 100) if coupons else 0.0, 2),
        "legs_played": legs_played,
        "avg_legs": round((legs_played / coupons) if coupons else 0.0, 2),
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "final_bankroll": round(bankroll, 2),
        "profit": round(profit, 2),
        "growth_pct": round(growth_pct, 2),
        "roi_pct": round(roi_pct, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "min_bankroll": round(min_bankroll, 2),
        "stopped_bankroll_depleted": stopped,
        "skipped_no_odds": skipped_no_odds,
        "by_season": _finalize_period_stats(season_stats),
    }


def simulate_coupon_bankroll(
    predictions: List[Dict],
    starting_bankroll: float = DEFAULT_STARTING_BANKROLL,
    stake: float = DEFAULT_COUPON_STAKE,
    min_legs: int = DEFAULT_COUPON_MIN_LEGS,
    max_legs: int = DEFAULT_COUPON_MAX_LEGS,
    sort_by: str = "edge_x_confidence",
    max_per_league: Optional[int] = None,
) -> Dict:
    """
    Simulate accumulator coupons. Selected matches are grouped by match day,
    strongest selections first, with at most max_legs per coupon.
    """
    batches, skipped_no_odds = _build_coupon_batches(
        predictions,
        min_legs=min_legs,
        max_legs=max_legs,
        sort_by=sort_by,
        max_per_league=max_per_league,
    )
    return simulate_coupon_batches(
        batches,
        starting_bankroll=starting_bankroll,
        stake=stake,
        min_legs=min_legs,
        max_legs=max_legs,
        sort_by=sort_by,
        max_per_league=max_per_league,
        skipped_no_odds=skipped_no_odds,
    )


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
        bankroll = simulate_flat_bankroll(preds, bet_label_fn=bet_label_fn)

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
            "bankroll_final": bankroll["final_bankroll"],
            "bankroll_profit": bankroll["profit"],
            "bankroll_growth_pct": bankroll["growth_pct"],
            "bankroll_max_drawdown": bankroll["max_drawdown"],
            "bankroll": bankroll,
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

    hc_market = [p for p in predictions if p["edge"] >= 0.0 and p["confidence"] >= 0.65]
    _run("Edge>=0% + Conf>=65%", hc_market)

    return experiments


def compute_coupon_experiments(predictions: List[Dict]) -> List[Dict]:
    """Backtest accumulator/coupon strategies with fixed bankroll and stake."""
    strategies = [
        ("Conf >= 45%", lambda p: p["confidence"] >= 0.45),
        ("Conf >= 50%", lambda p: p["confidence"] >= 0.50),
        ("Conf >= 55%", lambda p: p["confidence"] >= 0.55),
        ("Conf >= 60%", lambda p: p["confidence"] >= 0.60),
        ("Conf >= 65%", lambda p: p["confidence"] >= 0.65),
        ("Edge >= 5%", lambda p: p["edge"] >= 0.05),
        ("Edge >= 8%", lambda p: p["edge"] >= 0.08),
        ("Edge >= 10%", lambda p: p["edge"] >= 0.10),
        ("Edge>=5% + Conf>=45%", lambda p: p["edge"] >= 0.05 and p["confidence"] >= 0.45),
        ("Edge>=5% + Conf>=50%", lambda p: p["edge"] >= 0.05 and p["confidence"] >= 0.50),
        ("Edge>=0% + Conf>=65%", lambda p: p["edge"] >= 0.0 and p["confidence"] >= 0.65),
        ("Edge>=10% + Conf>=50%", lambda p: p["edge"] >= 0.10 and p["confidence"] >= 0.50),
        ("Edge>=10% + Conf>=55%", lambda p: p["edge"] >= 0.10 and p["confidence"] >= 0.55),
    ]
    experiments = []
    for name, predicate in strategies:
        selected = [p for p in predictions if predicate(p)]
        for max_legs in [2, 3, 4, 5, 6]:
            sim = simulate_coupon_bankroll(selected, max_legs=max_legs)
            experiments.append({
                "name": name,
                "max_legs": max_legs,
                "coupons": sim["coupons"],
                "winning_coupons": sim["winning_coupons"],
                "coupon_hit_rate": sim["coupon_hit_rate"],
                "profit": sim["profit"],
                "roi_pct": sim["roi_pct"],
                "final_bankroll": sim["final_bankroll"],
                "growth_pct": sim["growth_pct"],
                "max_drawdown": sim["max_drawdown"],
                "simulation": sim,
            })
    experiments.sort(
        key=lambda e: (e["final_bankroll"], -e["max_drawdown"], e["coupons"]),
        reverse=True,
    )
    return experiments


# ═════════════════════════════════════════════════════════════
#  STRATEGY OPTIMIZER
# ═════════════════════════════════════════════════════════════
def _threshold_label(value: Optional[float]) -> str:
    if value is None:
        return "none"
    return f"{value * 100:.0f}%"


def _outcome_label(label: Optional[int]) -> str:
    if label is None:
        return "all"
    return {LABEL_HOME: "home", LABEL_DRAW: "draw", LABEL_AWAY: "away"}.get(label, str(label))


def _bet_label_for_style(prediction: Dict, bet_style: str) -> int:
    if bet_style == "least_likely":
        probs = [
            (prediction.get("home_prob", 0.0), LABEL_HOME),
            (prediction.get("draw_prob", 0.0), LABEL_DRAW),
            (prediction.get("away_prob", 0.0), LABEL_AWAY),
        ]
        return min(probs, key=lambda item: item[0])[1]

    if bet_style == "market_underdog":
        odds = [
            (prediction.get("home_odds") or 0.0, LABEL_HOME),
            (prediction.get("draw_odds") or 0.0, LABEL_DRAW),
            (prediction.get("away_odds") or 0.0, LABEL_AWAY),
        ]
        return max(odds, key=lambda item: item[0])[1]

    return prediction.get("predicted")


def _league_groups(predictions: List[Dict]) -> Dict[str, List[str]]:
    leagues = sorted({p.get("league", "UNK") for p in predictions if p.get("league")})
    groups = {"all": leagues}

    top = [league for league in leagues if league in TOP_LEAGUE_CODES]
    if top and set(top) != set(leagues):
        groups["top_leagues"] = top

    positive = []
    for league in leagues:
        league_preds = [p for p in predictions if p.get("league") == league]
        if simulate_flat_bankroll(league_preds)["profit"] > 0:
            positive.append(league)
    if positive:
        groups["positive_leagues"] = positive

    for league in leagues:
        groups[f"league:{league}"] = [league]
    return groups


def _filter_predictions(
    predictions: List[Dict],
    confidence_min: Optional[float],
    edge_min: Optional[float],
    outcome: Optional[int],
    leagues: List[str],
) -> List[Dict]:
    league_set = set(leagues)
    selected = []
    for prediction in predictions:
        if league_set and prediction.get("league") not in league_set:
            continue
        if confidence_min is not None and prediction.get("confidence", 0.0) < confidence_min:
            continue
        if edge_min is not None and prediction.get("edge", 0.0) < edge_min:
            continue
        if outcome is not None and prediction.get("predicted") != outcome:
            continue
        selected.append(prediction)
    return selected


def _robust_score(simulation: Dict, count_key: str, min_count: int) -> Tuple[float, bool, List[str]]:
    reasons = []
    count = int(simulation.get(count_key, 0))
    profit = float(simulation.get("profit", 0.0))
    max_drawdown = float(simulation.get("max_drawdown", 0.0))
    seasons = simulation.get("by_season", {})
    positive_seasons = sum(1 for stats in seasons.values() if stats.get("profit", 0.0) > 0)
    negative_seasons = sum(1 for stats in seasons.values() if stats.get("profit", 0.0) < 0)
    required_positive = math.ceil(len(seasons) * 0.70) if seasons else 0
    worst_season_loss = abs(min((stats.get("profit", 0.0) for stats in seasons.values()), default=0.0))

    if count < min_count:
        reasons.append(f"too_few_{count_key}")
    if profit <= 0:
        reasons.append("not_profitable")
    if seasons and positive_seasons < required_positive:
        reasons.append("not_profitable_enough_seasons")
    if any(0 < stats.get("bets", min_count) < max(5, min_count // 10) for stats in seasons.values()):
        reasons.append("thin_season_sample")
    if max_drawdown > MAX_ROBUST_DRAWDOWN:
        reasons.append("drawdown_too_high")

    score = (
        profit
        - (0.5 * max_drawdown)
        - (NEGATIVE_SEASON_PENALTY * negative_seasons)
        - (0.25 * worst_season_loss)
    )
    return round(score, 2), not reasons, reasons


def _trim_candidate(candidate: Dict) -> Dict:
    """Keep optimization JSON readable while preserving the key decision data."""
    keep = {
        "type", "mode", "version", "name", "bet_style", "confidence_min_pct", "edge_min_pct",
        "outcome", "league_filter", "leagues", "max_legs", "sort_by",
        "max_per_league", "score", "eligible", "rejection_reasons",
    }
    trimmed = {k: v for k, v in candidate.items() if k in keep}
    sim = candidate.get("simulation", {})
    trimmed["simulation"] = {
        k: sim.get(k)
        for k in [
            "bets", "wins", "accuracy", "coupons", "winning_coupons",
            "coupon_hit_rate", "final_bankroll", "profit", "roi_pct",
            "max_drawdown", "max_drawdown_pct", "by_season",
        ]
        if k in sim
    }
    return trimmed


def _rank_candidates(candidates: List[Dict]) -> List[Dict]:
    return sorted(
        candidates,
        key=lambda c: (
            1 if c.get("eligible") else 0,
            c.get("score", -10**9),
            c.get("simulation", {}).get("final_bankroll", 0),
            -c.get("simulation", {}).get("max_drawdown", 0),
        ),
        reverse=True,
    )


def _optimize_single(mode: str, version: str, predictions: List[Dict]) -> List[Dict]:
    candidates = []
    confidence_options = [None] + OPT_CONF_THRESHOLDS
    outcome_options = [None, LABEL_HOME, LABEL_DRAW, LABEL_AWAY]
    for league_filter, leagues in _league_groups(predictions).items():
        for bet_style in OPT_SINGLE_BET_STYLES:
            bet_label_fn = None if bet_style == "model_pick" else lambda p, style=bet_style: _bet_label_for_style(p, style)
            for confidence_min in confidence_options:
                for edge_min in OPT_EDGE_THRESHOLDS:
                    base_selected = _filter_predictions(predictions, confidence_min, edge_min, None, leagues)
                    for outcome in outcome_options:
                        if outcome is None:
                            selected = base_selected
                        else:
                            selected = [
                                p for p in base_selected
                                if _bet_label_for_style(p, bet_style) == outcome
                            ]
                        simulation = simulate_flat_bankroll(selected, bet_label_fn=bet_label_fn)
                        score, eligible, reasons = _robust_score(simulation, "bets", 100)
                        name = (
                            f"{version} single style={bet_style} "
                            f"conf>={_threshold_label(confidence_min)} "
                            f"edge>={_threshold_label(edge_min)} outcome={_outcome_label(outcome)} "
                            f"leagues={league_filter}"
                        )
                        candidates.append({
                            "type": "single",
                            "mode": mode,
                            "version": version,
                            "name": name,
                            "bet_style": bet_style,
                            "confidence_min_pct": None if confidence_min is None else round(confidence_min * 100, 1),
                            "edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
                            "outcome": _outcome_label(outcome),
                            "league_filter": league_filter,
                            "leagues": leagues,
                            "simulation": simulation,
                            "score": score,
                            "eligible": eligible,
                            "rejection_reasons": reasons,
                        })
    return candidates


def _optimize_coupons(mode: str, version: str, predictions: List[Dict]) -> List[Dict]:
    candidates = []
    for league_filter, leagues in _league_groups(predictions).items():
        sort_options = ["confidence"] if league_filter.startswith("league:") else OPT_COUPON_SORTS
        max_per_options = [1] if league_filter.startswith("league:") else OPT_COUPON_MAX_PER_LEAGUE
        for confidence_min in OPT_CONF_THRESHOLDS:
            for edge_min in OPT_EDGE_THRESHOLDS:
                selected = _filter_predictions(predictions, confidence_min, edge_min, None, leagues)
                if len(selected) < DEFAULT_COUPON_MIN_LEGS:
                    continue
                for max_legs in OPT_COUPON_MAX_LEGS:
                    for sort_by in sort_options:
                        for max_per_league in max_per_options:
                            simulation = simulate_coupon_bankroll(
                                selected,
                                max_legs=max_legs,
                                sort_by=sort_by,
                                max_per_league=max_per_league,
                            )
                            score, eligible, reasons = _robust_score(simulation, "coupons", 50)
                            name = (
                                f"{version} coupon conf>={_threshold_label(confidence_min)} "
                                f"edge>={_threshold_label(edge_min)} max={max_legs} "
                                f"sort={sort_by} max_per_league={max_per_league} leagues={league_filter}"
                            )
                            candidates.append({
                                "type": "coupon",
                                "mode": mode,
                                "version": version,
                                "name": name,
                                "confidence_min_pct": round(confidence_min * 100, 1),
                                "edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
                                "league_filter": league_filter,
                                "leagues": leagues,
                                "max_legs": max_legs,
                                "sort_by": sort_by,
                                "max_per_league": max_per_league,
                                "simulation": simulation,
                                "score": score,
                                "eligible": eligible,
                                "rejection_reasons": reasons,
                            })
    return candidates


def _outcome_diagnostics(predictions: List[Dict]) -> Dict[str, Dict]:
    diagnostics = {}
    for label, name in [(LABEL_HOME, "home"), (LABEL_DRAW, "draw"), (LABEL_AWAY, "away")]:
        selected = [p for p in predictions if p.get("predicted") == label]
        diagnostics[name] = simulate_flat_bankroll(selected)
    return diagnostics


def _edge_diagnostics(predictions: List[Dict]) -> List[Dict]:
    rows = []
    for edge_min in OPT_EDGE_THRESHOLDS:
        selected = _filter_predictions(predictions, None, edge_min, None, sorted({p.get("league") for p in predictions}))
        sim = simulate_flat_bankroll(selected)
        rows.append({
            "edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
            "bets": sim["bets"],
            "profit": sim["profit"],
            "roi_pct": sim["roi_pct"],
            "final_bankroll": sim["final_bankroll"],
            "max_drawdown": sim["max_drawdown"],
        })
    return rows


def _coupon_leg_diagnostics(candidate: Optional[Dict], predictions: List[Dict]) -> Dict:
    if not candidate:
        return {}
    selected = _filter_predictions(
        predictions,
        (candidate.get("confidence_min_pct") or 0) / 100 if candidate.get("confidence_min_pct") is not None else None,
        (candidate.get("edge_min_pct") or 0) / 100 if candidate.get("edge_min_pct") is not None else None,
        None,
        candidate.get("leagues", []),
    )
    batches, _ = _build_coupon_batches(
        selected,
        max_legs=candidate.get("max_legs", DEFAULT_COUPON_MAX_LEGS),
        sort_by=candidate.get("sort_by", "edge_x_confidence"),
        max_per_league=candidate.get("max_per_league"),
    )
    by_leg: Dict[str, Dict] = defaultdict(lambda: {"legs": 0, "correct": 0})
    for legs in batches:
        for idx, leg in enumerate(legs, start=1):
            key = str(idx)
            by_leg[key]["legs"] += 1
            if leg.get("predicted") == leg.get("actual"):
                by_leg[key]["correct"] += 1
    return {
        leg: {
            "legs": stats["legs"],
            "correct": stats["correct"],
            "hit_rate": round(stats["correct"] / stats["legs"] * 100, 2) if stats["legs"] else 0.0,
        }
        for leg, stats in sorted(by_leg.items(), key=lambda item: int(item[0]))
    }


def _build_failure_report(predictions: List[Dict], best_coupon: Optional[Dict]) -> Dict:
    per_league = compute_per_league(predictions)
    losing_leagues = [
        {
            "league": league,
            "accuracy": round(metrics.get("accuracy", 0.0) * 100, 2),
            "roi_pct": round(metrics.get("roi_flat", {}).get("roi_pct", 0.0), 2),
            "bets": metrics.get("total", 0),
        }
        for league, metrics in per_league.items()
        if metrics.get("roi_flat", {}).get("profit", 0.0) < 0
    ]
    losing_leagues.sort(key=lambda row: row["roi_pct"])
    calibration = compute_calibration(predictions)

    return {
        "losing_leagues": losing_leagues,
        "outcomes": _outcome_diagnostics(predictions),
        "calibration": calibration,
        "edge_signal": _edge_diagnostics(predictions),
        "coupon_leg_failure": _coupon_leg_diagnostics(best_coupon, predictions),
        "missing_data_next": [
            "confirmed starting lineups",
            "injuries and suspensions",
            "player strength or player ratings",
            "opening-to-closing odds movement",
            "xG, shot, event, and tracking data",
        ],
    }


def _version_model_comparison(mode: str, v1_predictions: List[Dict], v2_predictions: List[Dict]) -> Dict:
    v1_metrics = compute_metrics(v1_predictions)
    v2_metrics = compute_metrics(v2_predictions)
    seasons = sorted({
        str(p.get("season") or _safe_year(p.get("match_date")))
        for p in [*v1_predictions, *v2_predictions]
        if (p.get("season") or _safe_year(p.get("match_date"))) is not None
    })
    by_season = {}
    v2_accuracy_wins = 0
    v2_brier_wins = 0
    v2_log_loss_wins = 0
    v2_roi_wins = 0

    for season in seasons:
        v1_season = [
            p for p in v1_predictions
            if str(p.get("season") or _safe_year(p.get("match_date"))) == season
        ]
        v2_season = [
            p for p in v2_predictions
            if str(p.get("season") or _safe_year(p.get("match_date"))) == season
        ]
        if not v1_season or not v2_season:
            continue
        v1s = compute_metrics(v1_season)
        v2s = compute_metrics(v2_season)
        v1_roi = v1s.get("roi_flat", {}).get("roi_pct", 0.0)
        v2_roi = v2s.get("roi_flat", {}).get("roi_pct", 0.0)
        if v2s.get("accuracy", 0.0) > v1s.get("accuracy", 0.0):
            v2_accuracy_wins += 1
        if v2s.get("brier", float("inf")) < v1s.get("brier", float("inf")):
            v2_brier_wins += 1
        if v2s.get("log_loss", float("inf")) < v1s.get("log_loss", float("inf")):
            v2_log_loss_wins += 1
        if v2_roi > v1_roi:
            v2_roi_wins += 1
        by_season[season] = {
            "v1_accuracy_pct": round(v1s.get("accuracy", 0.0) * 100, 2),
            "v2_accuracy_pct": round(v2s.get("accuracy", 0.0) * 100, 2),
            "v1_brier": round(v1s.get("brier", 0.0), 4),
            "v2_brier": round(v2s.get("brier", 0.0), 4),
            "v1_log_loss": round(v1s.get("log_loss", 0.0), 4),
            "v2_log_loss": round(v2s.get("log_loss", 0.0), 4),
            "v1_roi_pct": round(v1_roi, 2),
            "v2_roi_pct": round(v2_roi, 2),
        }

    compared_years = len(by_season)
    required_year_wins = math.ceil(compared_years * 0.60) if compared_years else 0
    v1_roi_total = v1_metrics.get("roi_flat", {}).get("roi_pct", 0.0)
    v2_roi_total = v2_metrics.get("roi_flat", {}).get("roi_pct", 0.0)
    v2_overall_better = (
        v2_metrics.get("accuracy", 0.0) >= v1_metrics.get("accuracy", 0.0)
        and v2_metrics.get("brier", float("inf")) <= v1_metrics.get("brier", float("inf"))
        and v2_metrics.get("log_loss", float("inf")) <= v1_metrics.get("log_loss", float("inf"))
    )
    promote_v2 = bool(
        compared_years
        and v2_overall_better
        and v2_accuracy_wins >= required_year_wins
        and v2_brier_wins >= required_year_wins
        and v2_log_loss_wins >= required_year_wins
    )

    return {
        "mode": mode,
        "promote_v2": promote_v2,
        "required_year_wins": required_year_wins,
        "compared_years": compared_years,
        "v2_accuracy_wins": v2_accuracy_wins,
        "v2_brier_wins": v2_brier_wins,
        "v2_log_loss_wins": v2_log_loss_wins,
        "v2_roi_wins": v2_roi_wins,
        "overall": {
            "v1_accuracy_pct": round(v1_metrics.get("accuracy", 0.0) * 100, 2),
            "v2_accuracy_pct": round(v2_metrics.get("accuracy", 0.0) * 100, 2),
            "v1_brier": round(v1_metrics.get("brier", 0.0), 4),
            "v2_brier": round(v2_metrics.get("brier", 0.0), 4),
            "v1_log_loss": round(v1_metrics.get("log_loss", 0.0), 4),
            "v2_log_loss": round(v2_metrics.get("log_loss", 0.0), 4),
            "v1_roi_pct": round(v1_roi_total, 2),
            "v2_roi_pct": round(v2_roi_total, 2),
        },
        "by_season": by_season,
    }


def _build_model_decision(raw_preds: Dict) -> Dict:
    comparisons = {}
    for mode, mode_data in raw_preds.items():
        v1_predictions = mode_data.get("v1", [])
        v2_predictions = mode_data.get("v2", [])
        if v1_predictions and v2_predictions:
            comparisons[mode] = _version_model_comparison(mode, v1_predictions, v2_predictions)

    gate_mode = "walk_forward" if "walk_forward" in comparisons else next(iter(comparisons), None)
    gate = comparisons.get(gate_mode, {})
    promote_v2 = bool(gate.get("promote_v2"))
    overall = gate.get("overall", {})
    reason = (
        "v2 beats v1 on overall walk-forward calibration and in enough seasons"
        if promote_v2
        else (
            "v2 kept off: it does not beat v1 stably on walk-forward accuracy, Brier, and log loss"
            if gate_mode
            else "v2 kept off: no comparable v1/v2 prediction set"
        )
    )

    return {
        "promote_v2": promote_v2,
        "gate_mode": gate_mode,
        "reason": reason,
        "gate_overall": overall,
        "comparisons": comparisons,
    }


def _first_candidate_for_versions(candidates: List[Dict], allowed_versions: set) -> Optional[Dict]:
    for candidate in candidates:
        if candidate.get("eligible") and candidate.get("version") in allowed_versions:
            return candidate
    for candidate in candidates:
        if candidate.get("version") in allowed_versions:
            return candidate
    return None


def _recommend_config(
    best_single: Optional[Dict],
    best_coupon: Optional[Dict],
    model_decision: Optional[Dict] = None,
) -> Dict:
    """Translate optimizer winners into config-shaped recommendations."""
    model_decision = model_decision or {"promote_v2": False, "reason": "v2 decision unavailable"}
    return {
        "promote_v2": bool(model_decision.get("promote_v2")),
        "model_reason": model_decision.get("reason"),
        "paper_trading": {
            "bet_style": best_single.get("bet_style", "model_pick") if best_single else "model_pick",
            "min_edge_pct": best_single.get("edge_min_pct") if best_single else 5.0,
            "min_confidence_pct": best_single.get("confidence_min_pct") if best_single else 45.0,
            "leagues": best_single.get("leagues") if best_single else [],
            "outcome": best_single.get("outcome") if best_single else "all",
        },
        "coupon": {
            "min_edge_pct": best_coupon.get("edge_min_pct") if best_coupon else None,
            "min_confidence_pct": best_coupon.get("confidence_min_pct") if best_coupon else 65.0,
            "max_picks": best_coupon.get("max_legs") if best_coupon else 4,
            "max_per_league": best_coupon.get("max_per_league") if best_coupon else 2,
            "sort_by": best_coupon.get("sort_by") if best_coupon else "confidence",
        },
    }


# ═════════════════════════════════════════════════════════════
#  HISTORICAL PATTERN BACKTEST
# ═════════════════════════════════════════════════════════════
def _match_identity(prediction: Dict) -> Tuple[str, str, str]:
    return (
        str(prediction.get("league", "")),
        str(prediction.get("home", "")).strip().lower(),
        str(prediction.get("away", "")).strip().lower(),
    )


def _pair_key(prediction: Dict) -> Tuple[str, str, str]:
    league = str(prediction.get("league", ""))
    teams = sorted([
        str(prediction.get("home", "")).strip().lower(),
        str(prediction.get("away", "")).strip().lower(),
    ])
    return (league, teams[0], teams[1])


def _team_key(prediction: Dict, side: str) -> Tuple[str, str]:
    team = prediction.get("home") if side == "home" else prediction.get("away")
    return (str(prediction.get("league", "")), str(team).strip().lower())


def _winner_name(prediction: Dict) -> str:
    if prediction.get("actual") == LABEL_HOME:
        return str(prediction.get("home", "")).strip().lower()
    if prediction.get("actual") == LABEL_AWAY:
        return str(prediction.get("away", "")).strip().lower()
    return "DRAW"


def _team_result(prediction: Dict, team_name: str) -> str:
    winner = _winner_name(prediction)
    team = team_name.strip().lower()
    if winner == "DRAW":
        return "DRAW"
    return "WIN" if winner == team else "LOSS"


def _current_label_for_team_result(prediction: Dict, side: str, result: str) -> Optional[int]:
    if result == "DRAW":
        return LABEL_DRAW
    if side == "home":
        return LABEL_HOME if result == "WIN" else LABEL_AWAY
    return LABEL_AWAY if result == "WIN" else LABEL_HOME


def _dominant_key(counts: Dict, min_matches: int, min_rate: float) -> Tuple[Optional[object], int, float]:
    total = sum(counts.values())
    if total < min_matches:
        return None, total, 0.0
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, total, 0.0
    key, hits = ranked[0]
    rate = hits / total if total else 0.0
    if rate < min_rate:
        return None, total, rate
    return key, total, rate


def _pattern_label(
    prediction: Dict,
    pattern: str,
    histories: Dict,
    min_matches: int,
    min_rate: float,
) -> Tuple[Optional[int], int, float]:
    if pattern == "directed_h2h_outcome":
        counts = histories["directed"].get(_match_identity(prediction), {})
        label, total, rate = _dominant_key(counts, min_matches, min_rate)
        return label, total, rate

    if pattern == "pair_dominant_result":
        counts = histories["pair_winner"].get(_pair_key(prediction), {})
        winner, total, rate = _dominant_key(counts, min_matches, min_rate)
        if winner is None:
            return None, total, rate
        if winner == "DRAW":
            return LABEL_DRAW, total, rate
        current_home = str(prediction.get("home", "")).strip().lower()
        current_away = str(prediction.get("away", "")).strip().lower()
        if winner == current_home:
            return LABEL_HOME, total, rate
        if winner == current_away:
            return LABEL_AWAY, total, rate
        return None, total, rate

    if pattern == "home_team_home_outcome":
        counts = histories["home_side"].get(_team_key(prediction, "home"), {})
        label, total, rate = _dominant_key(counts, min_matches, min_rate)
        return label, total, rate

    if pattern == "away_team_away_outcome":
        counts = histories["away_side"].get(_team_key(prediction, "away"), {})
        label, total, rate = _dominant_key(counts, min_matches, min_rate)
        return label, total, rate

    if pattern == "home_team_any_result":
        key = _team_key(prediction, "home")
        counts = histories["team_any"].get(key, {})
        result, total, rate = _dominant_key(counts, min_matches, min_rate)
        return _current_label_for_team_result(prediction, "home", result) if result else None, total, rate

    if pattern == "away_team_any_result":
        key = _team_key(prediction, "away")
        counts = histories["team_any"].get(key, {})
        result, total, rate = _dominant_key(counts, min_matches, min_rate)
        return _current_label_for_team_result(prediction, "away", result) if result else None, total, rate

    if pattern == "pair_no_draw_favourite":
        counts = histories["pair_label"].get(_pair_key(prediction), {})
        total = sum(counts.values())
        draw_count = counts.get(LABEL_DRAW, 0)
        if total < min_matches:
            return None, total, 0.0
        no_draw_rate = 1.0 - (draw_count / total)
        if no_draw_rate < min_rate:
            return None, total, no_draw_rate
        odds = [
            (prediction.get("home_odds") or 99.0, LABEL_HOME),
            (prediction.get("draw_odds") or 99.0, LABEL_DRAW),
            (prediction.get("away_odds") or 99.0, LABEL_AWAY),
        ]
        label = min(odds, key=lambda item: item[0])[1]
        if label == LABEL_DRAW:
            return None, total, no_draw_rate
        return label, total, no_draw_rate

    return None, 0, 0.0


def _update_pattern_histories(histories: Dict, prediction: Dict):
    actual = prediction.get("actual")
    if actual not in (LABEL_HOME, LABEL_DRAW, LABEL_AWAY):
        return

    directed = histories["directed"][_match_identity(prediction)]
    directed[actual] += 1

    pair = _pair_key(prediction)
    histories["pair_label"][pair][actual] += 1
    histories["pair_winner"][pair][_winner_name(prediction)] += 1

    home_key = _team_key(prediction, "home")
    away_key = _team_key(prediction, "away")
    histories["home_side"][home_key][actual] += 1
    histories["away_side"][away_key][actual] += 1
    histories["team_any"][home_key][_team_result(prediction, str(prediction.get("home", "")))] += 1
    histories["team_any"][away_key][_team_result(prediction, str(prediction.get("away", "")))] += 1


def _build_pattern_predictions(
    matches: List[Dict],
    pattern: str,
    min_matches: int,
    min_rate: float,
    max_odds: Optional[float],
) -> List[Dict]:
    histories = {
        "directed": defaultdict(lambda: defaultdict(int)),
        "pair_label": defaultdict(lambda: defaultdict(int)),
        "pair_winner": defaultdict(lambda: defaultdict(int)),
        "home_side": defaultdict(lambda: defaultdict(int)),
        "away_side": defaultdict(lambda: defaultdict(int)),
        "team_any": defaultdict(lambda: defaultdict(int)),
    }
    selected = []

    for match in _sorted_predictions(matches):
        label, history_count, hit_rate = _pattern_label(match, pattern, histories, min_matches, min_rate)
        if label is not None:
            odds = _prediction_odds(match, label)
            if odds is not None and (max_odds is None or odds <= max_odds):
                pick = dict(match)
                pick["predicted"] = label
                pick["confidence"] = hit_rate
                pick["edge"] = hit_rate - (1 / odds)
                pick["pattern"] = pattern
                pick["pattern_history_count"] = history_count
                pick["pattern_hit_rate"] = round(hit_rate * 100, 2)
                selected.append(pick)
        _update_pattern_histories(histories, match)

    return selected


def optimize_historical_patterns(matches: List[Dict]) -> Dict:
    patterns = [
        "directed_h2h_outcome",
        "pair_dominant_result",
        "home_team_home_outcome",
        "away_team_away_outcome",
        "home_team_any_result",
        "away_team_any_result",
        "pair_no_draw_favourite",
    ]
    candidates = []

    for pattern in patterns:
        logger.info(f"Pattern grid: {pattern}")
        for min_matches in PATTERN_MIN_MATCHES:
            for min_rate in PATTERN_MIN_HIT_RATES:
                for max_odds in PATTERN_MAX_ODDS:
                    selected = _build_pattern_predictions(matches, pattern, min_matches, min_rate, max_odds)
                    simulation = simulate_flat_bankroll(selected)
                    score, eligible, reasons = _robust_score(simulation, "bets", 100)
                    candidates.append({
                        "type": "historical_pattern",
                        "name": (
                            f"{pattern} min_matches={min_matches} "
                            f"min_rate={min_rate:.0%} "
                            f"max_odds={'none' if max_odds is None else f'{max_odds:.1f}'}"
                        ),
                        "pattern": pattern,
                        "min_matches": min_matches,
                        "min_rate_pct": round(min_rate * 100, 1),
                        "max_odds": max_odds,
                        "simulation": simulation,
                        "score": score,
                        "eligible": eligible,
                        "rejection_reasons": reasons,
                    })

    ranked = _rank_candidates(candidates)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "walk_forward_historical_patterns_no_future_leakage",
        "source_matches": len(matches),
        "patterns_tested": patterns,
        "candidate_count": len(candidates),
        "best": _trim_pattern_candidate(ranked[0]) if ranked else None,
        "top": [_trim_pattern_candidate(c) for c in ranked[:50]],
    }


def _trim_pattern_candidate(candidate: Dict) -> Dict:
    sim = candidate.get("simulation", {})
    return {
        "type": candidate.get("type"),
        "name": candidate.get("name"),
        "pattern": candidate.get("pattern"),
        "min_matches": candidate.get("min_matches"),
        "min_rate_pct": candidate.get("min_rate_pct"),
        "max_odds": candidate.get("max_odds"),
        "score": candidate.get("score"),
        "eligible": candidate.get("eligible"),
        "rejection_reasons": candidate.get("rejection_reasons", []),
        "simulation": {
            k: sim.get(k)
            for k in [
                "bets", "wins", "accuracy", "final_bankroll", "profit",
                "roi_pct", "max_drawdown", "max_drawdown_pct", "by_season",
            ]
            if k in sim
        },
    }


def _load_pattern_matches(raw_preds: Dict) -> List[Dict]:
    if isinstance(raw_preds, list):
        return raw_preds
    if "walk_forward" in raw_preds and raw_preds["walk_forward"].get("v1"):
        return raw_preds["walk_forward"]["v1"]
    for mode_data in raw_preds.values():
        if isinstance(mode_data, dict) and mode_data.get("v1"):
            return mode_data["v1"]
    return []


def print_historical_pattern_summary(results: Dict):
    print_header("HISTORICAL PATTERN OPTIMIZATION")
    best = results.get("best") or {}
    sim = best.get("simulation", {})
    print()
    print(f"  Best pattern: {best.get('name', 'N/A')}")
    print(f"    Score: {best.get('score', 0):+.0f} | Eligible: {best.get('eligible')}")
    print(
        f"    Final: {sim.get('final_bankroll', 0):.0f} | "
        f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
        f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
    )
    print(f"    Bets: {sim.get('bets', 0)} | Hit: {sim.get('accuracy', 0):.1f}%")
    print(f"    By season: {sim.get('by_season', {})}")
    print()


# ═════════════════════════════════════════════════════════════
#  MONEY-FIRST HISTORICAL EDGE BACKTEST
# ═════════════════════════════════════════════════════════════
def _enrich_directed_h2h_history(matches: List[Dict]) -> List[Dict]:
    """
    Add exact home-vs-away historical counts before each match.

    The current match is evaluated before it updates the history, so this is
    safe for walk-forward backtests and cannot leak future results.
    """
    histories = defaultdict(lambda: defaultdict(int))
    enriched = []

    for match in _sorted_predictions(matches):
        pick = dict(match)
        key = _match_identity(match)
        counts = histories[key]
        total = sum(counts.values())
        if total:
            label, hits = max(counts.items(), key=lambda item: (item[1], -item[0]))
            pick["_h2h_count"] = total
            pick["_h2h_label"] = label
            pick["_h2h_rate"] = hits / total
        else:
            pick["_h2h_count"] = 0
            pick["_h2h_label"] = None
            pick["_h2h_rate"] = 0.0

        enriched.append(pick)

        actual = match.get("actual")
        if actual in (LABEL_HOME, LABEL_DRAW, LABEL_AWAY):
            histories[key][actual] += 1

    return enriched


def _raw_prediction_sets(raw_preds: Dict) -> List[Tuple[str, str, List[Dict]]]:
    if isinstance(raw_preds, list):
        return [("list", "v1", raw_preds)]

    modes = ["walk_forward"] if raw_preds.get("walk_forward") else list(raw_preds.keys())
    sets = []
    for mode in modes:
        mode_data = raw_preds.get(mode, {})
        if not isinstance(mode_data, dict):
            continue
        for version in ("v1", "v2"):
            predictions = mode_data.get(version, [])
            if predictions:
                sets.append((mode, version, predictions))
    return sets


def _history_edge_league_groups(predictions: List[Dict]) -> Dict[str, List[str]]:
    leagues = sorted({p.get("league") for p in predictions if p.get("league")})
    groups = {"all": leagues}
    top = [league for league in leagues if league in TOP_LEAGUE_CODES]
    if top:
        groups["top_leagues"] = top
    for league in top:
        groups[f"league:{league}"] = [league]
    return groups


def _history_edge_arrays(predictions: List[Dict]) -> Dict[str, object]:
    def _float_col(name: str) -> np.ndarray:
        return np.array([float(p.get(name) or 0.0) for p in predictions], dtype=float)

    return {
        "predictions": predictions,
        "actual": np.array([int(p.get("actual", -1)) for p in predictions], dtype=int),
        "model_pred": np.array([int(p.get("predicted", -1)) for p in predictions], dtype=int),
        "league": np.array([str(p.get("league", "")) for p in predictions], dtype=object),
        "season": np.array([_season_key(p) for p in predictions], dtype=object),
        "h2h_count": np.array([int(p.get("_h2h_count") or 0) for p in predictions], dtype=int),
        "h2h_label": np.array([
            int(p.get("_h2h_label")) if p.get("_h2h_label") is not None else -1
            for p in predictions
        ], dtype=int),
        "h2h_rate": np.array([float(p.get("_h2h_rate") or 0.0) for p in predictions], dtype=float),
        "prob": np.stack([
            _float_col("home_prob"),
            _float_col("draw_prob"),
            _float_col("away_prob"),
        ], axis=1),
        "odds": np.stack([
            _float_col("home_odds"),
            _float_col("draw_odds"),
            _float_col("away_odds"),
        ], axis=1),
    }


def _labels_for_history_edge_style(arrays: Dict[str, object], style: str) -> np.ndarray:
    prob = arrays["prob"]
    odds = arrays["odds"]
    valid = odds > 1.0
    model_pred = arrays["model_pred"]

    if style == "model_pick":
        return model_pred.copy()
    if style == "historical_outcome":
        return arrays["h2h_label"].copy()
    if style == "max_model_edge":
        model_edge = np.where(valid, prob * odds - 1.0, -99.0)
        return np.argmax(model_edge, axis=1).astype(int)
    if style == "market_favorite":
        return np.argmin(np.where(valid, odds, 999.0), axis=1).astype(int)
    if style == "market_underdog":
        return np.argmax(np.where(valid, odds, -1.0), axis=1).astype(int)
    return model_pred.copy()


def _selected_label_values(
    arrays: Dict[str, object],
    labels: np.ndarray,
    style: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prob = arrays["prob"]
    odds = arrays["odds"]
    h2h_rate = arrays["h2h_rate"]
    valid_label = np.isin(labels, [LABEL_HOME, LABEL_DRAW, LABEL_AWAY])
    safe_labels = np.where(valid_label, labels, LABEL_HOME)
    row_idx = np.arange(len(labels))
    selected_odds = odds[row_idx, safe_labels]
    selected_prob = prob[row_idx, safe_labels]

    if style == "historical_outcome":
        selected_confidence = h2h_rate.copy()
        selected_edge = np.where(selected_odds > 1.0, h2h_rate - (1.0 / selected_odds), -99.0)
    else:
        selected_confidence = selected_prob
        selected_edge = np.where(selected_odds > 1.0, selected_prob * selected_odds - 1.0, -99.0)

    valid = valid_label & (selected_odds > 1.0)
    return selected_confidence, selected_edge, selected_odds, valid


def _simulate_flat_arrays(
    mask: np.ndarray,
    labels: np.ndarray,
    arrays: Dict[str, object],
    odds_matrix: Optional[np.ndarray] = None,
) -> Dict:
    idx = np.flatnonzero(mask)
    odds = odds_matrix if odds_matrix is not None else arrays["odds"]
    actual = arrays["actual"]
    season = arrays["season"]
    selected_odds = odds[idx, labels[idx]]
    wins = labels[idx] == actual[idx]
    deltas = np.where(wins, (DEFAULT_SINGLE_STAKE * selected_odds) - DEFAULT_SINGLE_STAKE, -DEFAULT_SINGLE_STAKE)
    cumulative = np.cumsum(deltas) if len(deltas) else np.array([])
    final_bankroll = DEFAULT_STARTING_BANKROLL + (float(cumulative[-1]) if len(cumulative) else 0.0)
    bankroll_curve = DEFAULT_STARTING_BANKROLL + cumulative if len(cumulative) else np.array([])
    if len(bankroll_curve):
        peaks = np.maximum.accumulate(np.concatenate(([DEFAULT_STARTING_BANKROLL], bankroll_curve)))[1:]
        max_drawdown = float(np.max(peaks - bankroll_curve))
        peak_bankroll = float(max(DEFAULT_STARTING_BANKROLL, np.max(bankroll_curve)))
        min_bankroll = float(min(DEFAULT_STARTING_BANKROLL, np.min(bankroll_curve)))
    else:
        max_drawdown = 0.0
        peak_bankroll = DEFAULT_STARTING_BANKROLL
        min_bankroll = DEFAULT_STARTING_BANKROLL

    season_stats: Dict[str, Dict] = defaultdict(_empty_period_stats)
    for selected_idx, odd, won in zip(idx, selected_odds, wins):
        season_key = str(season[selected_idx])
        season_stats[season_key]["bets"] += 1
        season_stats[season_key]["wins"] += 1 if won else 0
        season_stats[season_key]["staked"] += DEFAULT_SINGLE_STAKE
        season_stats[season_key]["returned"] += DEFAULT_SINGLE_STAKE * float(odd) if won else 0.0

    bets = int(len(idx))
    staked = bets * DEFAULT_SINGLE_STAKE
    profit = final_bankroll - DEFAULT_STARTING_BANKROLL
    return {
        "starting_bankroll": round(DEFAULT_STARTING_BANKROLL, 2),
        "stake": round(DEFAULT_SINGLE_STAKE, 2),
        "bets": bets,
        "wins": int(np.sum(wins)) if bets else 0,
        "accuracy": round((float(np.mean(wins)) * 100) if bets else 0.0, 2),
        "staked": round(staked, 2),
        "returned": round(staked + profit, 2),
        "final_bankroll": round(final_bankroll, 2),
        "profit": round(profit, 2),
        "growth_pct": round((profit / DEFAULT_STARTING_BANKROLL * 100) if DEFAULT_STARTING_BANKROLL else 0.0, 2),
        "roi_pct": round((profit / staked * 100) if staked else 0.0, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round((max_drawdown / peak_bankroll * 100) if peak_bankroll else 0.0, 2),
        "min_bankroll": round(min_bankroll, 2),
        "stopped_bankroll_depleted": False,
        "skipped_no_odds": 0,
        "by_season": _finalize_period_stats(season_stats),
    }


def _candidate_rank_tuple(candidate: Dict) -> Tuple:
    sim = candidate.get("simulation", {})
    return (
        1 if candidate.get("eligible") else 0,
        candidate.get("score", -10**9),
        sim.get("profit", 0),
        -sim.get("max_drawdown", 0),
    )


def _remember_top_candidate(candidates: List[Dict], candidate: Dict, limit: int):
    candidates.append(candidate)
    if len(candidates) > limit * 4:
        candidates[:] = sorted(candidates, key=_candidate_rank_tuple, reverse=True)[:limit]


def _history_edge_pick_predictions(
    candidate: Dict,
    arrays: Dict[str, object],
) -> List[Dict]:
    predictions = arrays["predictions"]
    idx = np.flatnonzero(candidate["mask"])
    labels = candidate["labels"]
    confidence = candidate["selected_confidence"]
    edge = candidate["selected_edge"]
    selected = []

    for row_idx in idx:
        pick = dict(predictions[row_idx])
        pick["predicted"] = int(labels[row_idx])
        pick["confidence"] = float(confidence[row_idx])
        pick["edge"] = float(edge[row_idx])
        pick["historical_edge_count"] = int(pick.get("_h2h_count") or 0)
        pick["historical_edge_rate"] = round(float(pick.get("_h2h_rate") or 0.0) * 100, 2)
        selected.append(pick)
    return selected


def _trim_history_edge_candidate(candidate: Dict) -> Dict:
    sim = candidate.get("simulation", {})
    trimmed = {
        "type": candidate.get("type"),
        "mode": candidate.get("mode"),
        "version": candidate.get("version"),
        "name": candidate.get("name"),
        "label_style": candidate.get("label_style"),
        "league_filter": candidate.get("league_filter"),
        "leagues": candidate.get("leagues"),
        "min_h2h_matches": candidate.get("min_h2h_matches"),
        "min_h2h_rate_pct": candidate.get("min_h2h_rate_pct"),
        "confidence_min_pct": candidate.get("confidence_min_pct"),
        "model_edge_min_pct": candidate.get("model_edge_min_pct"),
        "odds_band": candidate.get("odds_band"),
        "max_legs": candidate.get("max_legs"),
        "sort_by": candidate.get("sort_by"),
        "max_per_league": candidate.get("max_per_league"),
        "score": candidate.get("score"),
        "eligible": candidate.get("eligible"),
        "rejection_reasons": candidate.get("rejection_reasons", []),
    }
    trimmed["simulation"] = {
        k: sim.get(k)
        for k in [
            "bets", "wins", "accuracy", "coupons", "winning_coupons",
            "coupon_hit_rate", "final_bankroll", "profit", "roi_pct",
            "max_drawdown", "max_drawdown_pct", "by_season",
        ]
        if k in sim
    }
    return trimmed


def optimize_historical_edge(raw_preds: Dict) -> Dict:
    """Money-first historical edge search: bankroll growth from no-leak H2H signals."""
    all_singles = []
    all_coupons = []
    coupon_filters = []
    evaluated_singles = 0
    evaluated_coupons = 0
    source_sets = _raw_prediction_sets(raw_preds)

    for mode, version, predictions in source_sets:
        logger.info(f"Historical edge grid: {mode}/{version}")
        enriched = _enrich_directed_h2h_history(predictions)
        arrays = _history_edge_arrays(enriched)
        league_groups = _history_edge_league_groups(enriched)

        for label_style in HISTORY_EDGE_LABEL_STYLES:
            labels = _labels_for_history_edge_style(arrays, label_style)
            selected_confidence, selected_edge, selected_odds, valid = _selected_label_values(arrays, labels, label_style)

            for league_filter, leagues in league_groups.items():
                league_mask = np.isin(arrays["league"], leagues)
                for min_h2h_matches, min_h2h_rate in HISTORY_EDGE_FILTERS:
                    h2h_mask = (
                        (arrays["h2h_count"] >= min_h2h_matches)
                        & (arrays["h2h_rate"] >= min_h2h_rate)
                        & (arrays["h2h_label"] == labels)
                    )
                    for confidence_min in HISTORY_EDGE_CONF_THRESHOLDS:
                        confidence_mask = (
                            np.ones(len(enriched), dtype=bool)
                            if confidence_min is None
                            else selected_confidence >= confidence_min
                        )
                        for model_edge_min in HISTORY_EDGE_MODEL_EDGE_THRESHOLDS:
                            edge_mask = (
                                np.ones(len(enriched), dtype=bool)
                                if model_edge_min is None
                                else selected_edge >= model_edge_min
                            )
                            for odds_band in HISTORY_EDGE_ODDS_BANDS:
                                if odds_band is None:
                                    odds_mask = np.ones(len(enriched), dtype=bool)
                                else:
                                    odds_mask = (selected_odds >= odds_band[0]) & (selected_odds <= odds_band[1])

                                mask = valid & league_mask & h2h_mask & confidence_mask & edge_mask & odds_mask
                                if int(np.sum(mask)) < 100:
                                    continue

                                simulation = _simulate_flat_arrays(mask, labels, arrays)
                                score, eligible, reasons = _robust_score(simulation, "bets", 100)
                                evaluated_singles += 1
                                candidate = {
                                    "type": "historical_edge_single",
                                    "mode": mode,
                                    "version": version,
                                    "name": (
                                        f"{version} historical-edge single style={label_style} "
                                        f"leagues={league_filter} h2h>={min_h2h_matches}/"
                                        f"{min_h2h_rate:.0%} conf>={_threshold_label(confidence_min)} "
                                        f"edge>={_threshold_label(model_edge_min)} odds={odds_band or 'any'}"
                                    ),
                                    "label_style": label_style,
                                    "league_filter": league_filter,
                                    "leagues": leagues,
                                    "min_h2h_matches": min_h2h_matches,
                                    "min_h2h_rate_pct": round(min_h2h_rate * 100, 1),
                                    "confidence_min_pct": None if confidence_min is None else round(confidence_min * 100, 1),
                                    "model_edge_min_pct": None if model_edge_min is None else round(model_edge_min * 100, 1),
                                    "odds_band": odds_band,
                                    "simulation": simulation,
                                    "score": score,
                                    "eligible": eligible,
                                    "rejection_reasons": reasons,
                                }
                                _remember_top_candidate(all_singles, _trim_history_edge_candidate(candidate), 80)

                                filter_candidate = dict(candidate)
                                filter_candidate["mask"] = mask.copy()
                                filter_candidate["labels"] = labels.copy()
                                filter_candidate["selected_confidence"] = selected_confidence.copy()
                                filter_candidate["selected_edge"] = selected_edge.copy()
                                filter_candidate["arrays"] = arrays
                                _remember_top_candidate(coupon_filters, filter_candidate, HISTORY_EDGE_COUPON_TOP_FILTERS)

    coupon_filters = sorted(coupon_filters, key=_candidate_rank_tuple, reverse=True)[:HISTORY_EDGE_COUPON_TOP_FILTERS]
    for filter_candidate in coupon_filters:
        selected = _history_edge_pick_predictions(filter_candidate, filter_candidate["arrays"])
        if len(selected) < 100:
            continue
        for max_legs in OPT_COUPON_MAX_LEGS:
            for sort_by in OPT_COUPON_SORTS:
                for max_per_league in OPT_COUPON_MAX_PER_LEAGUE:
                    simulation = simulate_coupon_bankroll(
                        selected,
                        max_legs=max_legs,
                        sort_by=sort_by,
                        max_per_league=max_per_league,
                    )
                    if simulation.get("coupons", 0) < 50:
                        continue
                    score, eligible, reasons = _robust_score(simulation, "coupons", 50)
                    evaluated_coupons += 1
                    candidate = {
                        **{
                            key: filter_candidate.get(key)
                            for key in [
                                "mode", "version", "label_style", "league_filter", "leagues",
                                "min_h2h_matches", "min_h2h_rate_pct", "confidence_min_pct",
                                "model_edge_min_pct", "odds_band",
                            ]
                        },
                        "type": "historical_edge_coupon",
                        "name": (
                            f"{filter_candidate['name']} coupon max={max_legs} "
                            f"sort={sort_by} max_per_league={max_per_league}"
                        ),
                        "max_legs": max_legs,
                        "sort_by": sort_by,
                        "max_per_league": max_per_league,
                        "simulation": simulation,
                        "score": score,
                        "eligible": eligible,
                        "rejection_reasons": reasons,
                    }
                    _remember_top_candidate(all_coupons, _trim_history_edge_candidate(candidate), 80)

    ranked_singles = sorted(all_singles, key=_candidate_rank_tuple, reverse=True)
    ranked_coupons = sorted(all_coupons, key=_candidate_rank_tuple, reverse=True)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "money_first_historical_edge_no_future_leakage",
        "source_sets": [
            {"mode": mode, "version": version, "predictions": len(predictions)}
            for mode, version, predictions in source_sets
        ],
        "rules": {
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "single_stake": DEFAULT_SINGLE_STAKE,
            "coupon_stake": DEFAULT_COUPON_STAKE,
            "single_min_bets": 100,
            "coupon_min_count": 50,
            "h2h_signal": "exact same home team vs exact same away team before current match",
            "no_future_leakage": True,
        },
        "evaluated_single_candidates": evaluated_singles,
        "evaluated_coupon_candidates": evaluated_coupons,
        "best_single": ranked_singles[0] if ranked_singles else None,
        "best_coupon": ranked_coupons[0] if ranked_coupons else None,
        "top_singles": ranked_singles[:50],
        "top_coupons": ranked_coupons[:50],
    }


def print_historical_edge_summary(results: Dict):
    print_header("MONEY-FIRST HISTORICAL EDGE BACKTEST")

    def _print_best(title: str, candidate: Optional[Dict]):
        if not candidate:
            print(f"  {title}: N/A")
            return
        sim = candidate.get("simulation", {})
        count = sim.get("bets", sim.get("coupons", 0))
        hit = sim.get("accuracy", sim.get("coupon_hit_rate", 0.0))
        print()
        print(f"  {title}: {candidate.get('name')}")
        print(
            f"    Final: {sim.get('final_bankroll', 0):.0f} | "
            f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
            f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
        )
        print(f"    Count: {count} | Hit: {hit:.1f}% | Eligible: {candidate.get('eligible')}")
        print(f"    By season: {sim.get('by_season', {})}")

    print(f"  Single candidates evaluated: {results.get('evaluated_single_candidates', 0)}")
    print(f"  Coupon candidates evaluated: {results.get('evaluated_coupon_candidates', 0)}")
    _print_best("Best historical-edge single", results.get("best_single"))
    _print_best("Best historical-edge coupon", results.get("best_coupon"))
    print()


# ═════════════════════════════════════════════════════════════
#  CSV-ONLY HISTORICAL EDGE BACKTEST
# ═════════════════════════════════════════════════════════════
def _csv_match_to_history_prediction(match: Dict) -> Optional[Dict]:
    home_score = match.get("home_score")
    away_score = match.get("away_score")
    if home_score is None or away_score is None:
        return None

    home_odds = match.get("home_odds")
    draw_odds = match.get("draw_odds")
    away_odds = match.get("away_odds")
    if not home_odds or not draw_odds or not away_odds:
        return None

    extra = match.get("extra_data") or {}
    actual = LABEL_HOME if home_score > away_score else LABEL_DRAW if home_score == away_score else LABEL_AWAY
    return {
        "match_date": match.get("match_date", ""),
        "league": match.get("league_code", ""),
        "season": match.get("season"),
        "home": match.get("home_team_name", ""),
        "away": match.get("away_team_name", ""),
        "home_score": home_score,
        "away_score": away_score,
        "actual": actual,
        "predicted": -1,
        "confidence": 0.0,
        "home_prob": 0.0,
        "draw_prob": 0.0,
        "away_prob": 0.0,
        "home_odds": home_odds,
        "draw_odds": draw_odds,
        "away_odds": away_odds,
        "avg_home_odds": extra.get("avg_home_odds"),
        "avg_draw_odds": extra.get("avg_draw_odds"),
        "avg_away_odds": extra.get("avg_away_odds"),
        "max_home_odds": extra.get("max_home_odds"),
        "max_draw_odds": extra.get("max_draw_odds"),
        "max_away_odds": extra.get("max_away_odds"),
        "b365_close_home": extra.get("b365_close_home"),
        "b365_close_draw": extra.get("b365_close_draw"),
        "b365_close_away": extra.get("b365_close_away"),
        "avg_close_home_odds": extra.get("avg_close_home_odds"),
        "avg_close_draw_odds": extra.get("avg_close_draw_odds"),
        "avg_close_away_odds": extra.get("avg_close_away_odds"),
        "max_close_home_odds": extra.get("max_close_home_odds"),
        "max_close_draw_odds": extra.get("max_close_draw_odds"),
        "max_close_away_odds": extra.get("max_close_away_odds"),
        "edge": 0.0,
        "kelly": 0.0,
    }


def _csv_matches_to_history_predictions(matches: List[Dict]) -> List[Dict]:
    predictions = []
    for match in matches:
        prediction = _csv_match_to_history_prediction(match)
        if prediction:
            predictions.append(prediction)
    return _sorted_predictions(predictions)


def _history_label_for_csv_style(prediction: Dict, style: str) -> Optional[int]:
    history_label = prediction.get("_h2h_label")
    if history_label not in (LABEL_HOME, LABEL_DRAW, LABEL_AWAY):
        return None

    odds = {
        LABEL_HOME: prediction.get("home_odds"),
        LABEL_DRAW: prediction.get("draw_odds"),
        LABEL_AWAY: prediction.get("away_odds"),
    }
    valid = {label: odd for label, odd in odds.items() if odd and odd > 1.0}
    if not valid:
        return None

    if style == "historical_outcome":
        return history_label

    if style == "market_favorite":
        favorite = min(valid.items(), key=lambda item: item[1])[0]
        return favorite if favorite == history_label else None

    if style == "market_underdog":
        underdog = max(valid.items(), key=lambda item: item[1])[0]
        return underdog if underdog == history_label else None

    return None


def _build_csv_history_edge_predictions(
    matches: List[Dict],
    label_style: str,
    min_matches: int,
    min_rate: float,
    edge_min: Optional[float],
    odds_band: Optional[Tuple[float, float]],
    leagues: List[str],
) -> List[Dict]:
    league_set = set(leagues)
    selected = []

    for match in matches:
        if league_set and match.get("league") not in league_set:
            continue
        if match.get("_h2h_count", 0) < min_matches:
            continue
        if match.get("_h2h_rate", 0.0) < min_rate:
            continue

        label = _history_label_for_csv_style(match, label_style)
        if label is None:
            continue

        odds = _prediction_odds(match, label)
        if odds is None:
            continue
        if odds_band and not (odds_band[0] <= odds <= odds_band[1]):
            continue

        historical_rate = float(match.get("_h2h_rate") or 0.0)
        historical_edge = historical_rate - (1.0 / odds)
        if edge_min is not None and historical_edge < edge_min:
            continue

        pick = dict(match)
        pick["predicted"] = label
        pick["confidence"] = historical_rate
        pick["edge"] = historical_edge
        pick["historical_edge_count"] = int(match.get("_h2h_count") or 0)
        pick["historical_edge_rate"] = round(historical_rate * 100, 2)
        pick["label_style"] = label_style
        selected.append(pick)

    return selected


def _csv_history_league_groups(matches: List[Dict]) -> Dict[str, List[str]]:
    leagues = sorted({p.get("league") for p in matches if p.get("league")})
    groups = {"all": leagues}
    top = [league for league in leagues if league in TOP_LEAGUE_CODES]
    if top:
        groups["top_leagues"] = top
    for league in top:
        groups[f"league:{league}"] = [league]
    return groups


def optimize_csv_historical_edge(matches: List[Dict], start_season: int, end_season: int) -> Dict:
    """Backtest pure historical H2H/odds edge directly from CSV matches."""
    history_predictions = _csv_matches_to_history_predictions(matches)
    enriched = _enrich_directed_h2h_history(history_predictions)
    league_groups = _csv_history_league_groups(enriched)
    arrays = _history_edge_arrays(enriched)
    odds = arrays["odds"]
    odds_valid = odds > 1.0
    favorite_labels = np.argmin(np.where(odds_valid, odds, 999.0), axis=1).astype(int)
    underdog_labels = np.argmax(np.where(odds_valid, odds, -1.0), axis=1).astype(int)
    h2h_labels = arrays["h2h_label"]
    h2h_rate = arrays["h2h_rate"]
    h2h_count = arrays["h2h_count"]
    all_singles = []
    all_coupons = []
    coupon_filters = []
    evaluated_singles = 0
    evaluated_coupons = 0

    for label_style in CSV_HISTORY_LABEL_STYLES:
        if label_style == "historical_outcome":
            labels = h2h_labels.copy()
        elif label_style == "market_favorite":
            labels = np.where(favorite_labels == h2h_labels, favorite_labels, -1)
        elif label_style == "market_underdog":
            labels = np.where(underdog_labels == h2h_labels, underdog_labels, -1)
        else:
            continue

        selected_confidence, selected_edge, selected_odds, valid = _selected_label_values(
            arrays,
            labels,
            "historical_outcome",
        )

        for league_filter, leagues in league_groups.items():
            league_mask = np.isin(arrays["league"], leagues)
            for min_matches in CSV_HISTORY_MIN_MATCHES:
                for min_rate in CSV_HISTORY_MIN_RATES:
                    h2h_mask = (h2h_count >= min_matches) & (h2h_rate >= min_rate)
                    for edge_min in CSV_HISTORY_EDGE_THRESHOLDS:
                        edge_mask = (
                            np.ones(len(enriched), dtype=bool)
                            if edge_min is None
                            else selected_edge >= edge_min
                        )
                        for odds_band in HISTORY_EDGE_ODDS_BANDS:
                            if odds_band is None:
                                odds_mask = np.ones(len(enriched), dtype=bool)
                            else:
                                odds_mask = (selected_odds >= odds_band[0]) & (selected_odds <= odds_band[1])

                            mask = valid & league_mask & h2h_mask & edge_mask & odds_mask
                            if int(np.sum(mask)) < 100:
                                continue

                            simulation = _simulate_flat_arrays(mask, labels, arrays)
                            score, eligible, reasons = _robust_score(simulation, "bets", 100)
                            evaluated_singles += 1
                            candidate = {
                                "type": "csv_historical_edge_single",
                                "mode": "csv_history",
                                "version": "market",
                                "name": (
                                    f"csv historical-edge single style={label_style} "
                                    f"leagues={league_filter} h2h>={min_matches}/"
                                    f"{min_rate:.0%} edge>={_threshold_label(edge_min)} "
                                    f"odds={odds_band or 'any'}"
                                ),
                                "label_style": label_style,
                                "league_filter": league_filter,
                                "leagues": leagues,
                                "min_h2h_matches": min_matches,
                                "min_h2h_rate_pct": round(min_rate * 100, 1),
                                "model_edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
                                "odds_band": odds_band,
                                "simulation": simulation,
                                "score": score,
                                "eligible": eligible,
                                "rejection_reasons": reasons,
                            }
                            _remember_top_candidate(all_singles, _trim_history_edge_candidate(candidate), 100)

                            filter_candidate = dict(candidate)
                            filter_candidate["mask"] = mask.copy()
                            filter_candidate["labels"] = labels.copy()
                            filter_candidate["selected_confidence"] = selected_confidence.copy()
                            filter_candidate["selected_edge"] = selected_edge.copy()
                            filter_candidate["arrays"] = arrays
                            _remember_top_candidate(coupon_filters, filter_candidate, HISTORY_EDGE_COUPON_TOP_FILTERS)

    coupon_filters = sorted(coupon_filters, key=_candidate_rank_tuple, reverse=True)[:HISTORY_EDGE_COUPON_TOP_FILTERS]
    for filter_candidate in coupon_filters:
        selected = _history_edge_pick_predictions(filter_candidate, filter_candidate["arrays"])
        if len(selected) < 100:
            continue
        for max_legs in OPT_COUPON_MAX_LEGS:
            for sort_by in OPT_COUPON_SORTS:
                for max_per_league in OPT_COUPON_MAX_PER_LEAGUE:
                    simulation = simulate_coupon_bankroll(
                        selected,
                        max_legs=max_legs,
                        sort_by=sort_by,
                        max_per_league=max_per_league,
                    )
                    if simulation.get("coupons", 0) < 50:
                        continue
                    score, eligible, reasons = _robust_score(simulation, "coupons", 50)
                    evaluated_coupons += 1
                    candidate = {
                        **{
                            key: filter_candidate.get(key)
                            for key in [
                                "mode", "version", "label_style", "league_filter", "leagues",
                                "min_h2h_matches", "min_h2h_rate_pct", "model_edge_min_pct",
                                "odds_band",
                            ]
                        },
                        "type": "csv_historical_edge_coupon",
                        "name": (
                            f"{filter_candidate['name']} coupon max={max_legs} "
                            f"sort={sort_by} max_per_league={max_per_league}"
                        ),
                        "max_legs": max_legs,
                        "sort_by": sort_by,
                        "max_per_league": max_per_league,
                        "simulation": simulation,
                        "score": score,
                        "eligible": eligible,
                        "rejection_reasons": reasons,
                    }
                    _remember_top_candidate(all_coupons, _trim_history_edge_candidate(candidate), 100)

    ranked_singles = sorted(all_singles, key=_candidate_rank_tuple, reverse=True)
    ranked_coupons = sorted(all_coupons, key=_candidate_rank_tuple, reverse=True)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "csv_money_first_historical_h2h_odds_edge_no_future_leakage",
        "source": "football-data.co.uk CSV",
        "start_season": start_season,
        "end_season": end_season,
        "source_matches": len(matches),
        "usable_matches": len(history_predictions),
        "rules": {
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "single_stake": DEFAULT_SINGLE_STAKE,
            "coupon_stake": DEFAULT_COUPON_STAKE,
            "h2h_signal": "exact same home team vs exact same away team before current match",
            "edge": "historical hit rate minus break-even odds probability",
            "no_future_leakage": True,
            "note": "This is pure history/market odds. It does not include v1/v2 model confidence.",
        },
        "evaluated_single_candidates": evaluated_singles,
        "evaluated_coupon_candidates": evaluated_coupons,
        "best_single": ranked_singles[0] if ranked_singles else None,
        "best_coupon": ranked_coupons[0] if ranked_coupons else None,
        "top_singles": ranked_singles[:50],
        "top_coupons": ranked_coupons[:50],
    }


def print_csv_historical_edge_summary(results: Dict):
    print_header("CSV MONEY-FIRST HISTORICAL EDGE BACKTEST")
    print(f"  Seasons: {results.get('start_season')}-{results.get('end_season')}")
    print(f"  Source matches: {results.get('source_matches', 0)}")
    print(f"  Usable matches with 1X2 odds: {results.get('usable_matches', 0)}")
    print(f"  Single candidates evaluated: {results.get('evaluated_single_candidates', 0)}")
    print(f"  Coupon candidates evaluated: {results.get('evaluated_coupon_candidates', 0)}")

    for title, candidate in [
        ("Best CSV-history single", results.get("best_single")),
        ("Best CSV-history coupon", results.get("best_coupon")),
    ]:
        if not candidate:
            print(f"  {title}: N/A")
            continue
        sim = candidate.get("simulation", {})
        count = sim.get("bets", sim.get("coupons", 0))
        hit = sim.get("accuracy", sim.get("coupon_hit_rate", 0.0))
        print()
        print(f"  {title}: {candidate.get('name')}")
        print(
            f"    Final: {sim.get('final_bankroll', 0):.0f} | "
            f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
            f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
        )
        print(f"    Count: {count} | Hit: {hit:.1f}% | Eligible: {candidate.get('eligible')}")
        print(f"    By season: {sim.get('by_season', {})}")
    print()


# ═════════════════════════════════════════════════════════════
#  CSV STRATEGY ZOO BACKTEST
# ═════════════════════════════════════════════════════════════
def _set_dominant_history_fields(record: Dict, prefix: str, counts: Dict, label_mapper=None):
    total = sum(counts.values())
    if not total:
        record[f"_{prefix}_count"] = 0
        record[f"_{prefix}_label"] = None
        record[f"_{prefix}_rate"] = 0.0
        return

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        record[f"_{prefix}_count"] = total
        record[f"_{prefix}_label"] = None
        record[f"_{prefix}_rate"] = 0.0
        return

    key, hits = ranked[0]
    label = label_mapper(key) if label_mapper else key
    record[f"_{prefix}_count"] = total
    record[f"_{prefix}_label"] = label
    record[f"_{prefix}_rate"] = hits / total if total else 0.0


def _enrich_strategy_zoo_history(matches: List[Dict]) -> List[Dict]:
    """Add no-leak history fields for many simple strategy families."""
    directed = _enrich_directed_h2h_history(matches)
    pair_winner = defaultdict(lambda: defaultdict(int))
    home_side = defaultdict(lambda: defaultdict(int))
    away_side = defaultdict(lambda: defaultdict(int))
    team_any = defaultdict(lambda: defaultdict(int))
    recent_team_results = defaultdict(list)
    league_season_counts = defaultdict(int)
    enriched = []

    for match in _sorted_predictions(directed):
        pick = dict(match)
        home = str(match.get("home", "")).strip().lower()
        away = str(match.get("away", "")).strip().lower()
        league = str(match.get("league", ""))
        season = match.get("season")
        pair = _pair_key(match)
        home_key = (league, home)
        away_key = (league, away)
        league_season_key = (league, season)
        pick["_league_season_match_number"] = league_season_counts[league_season_key] + 1

        def winner_to_current_label(winner: str) -> Optional[int]:
            if winner == "DRAW":
                return LABEL_DRAW
            if winner == home:
                return LABEL_HOME
            if winner == away:
                return LABEL_AWAY
            return None

        _set_dominant_history_fields(pick, "pair", pair_winner[pair], winner_to_current_label)
        _set_dominant_history_fields(pick, "home_home", home_side[home_key])
        _set_dominant_history_fields(pick, "away_away", away_side[away_key])
        _set_dominant_history_fields(
            pick,
            "home_any",
            team_any[home_key],
            lambda result: _current_label_for_team_result(match, "home", result),
        )
        _set_dominant_history_fields(
            pick,
            "away_any",
            team_any[away_key],
            lambda result: _current_label_for_team_result(match, "away", result),
        )

        for side, key in [("home", home_key), ("away", away_key)]:
            recent = recent_team_results[key][-5:]
            total = len(recent)
            pick[f"_{side}_form_count"] = total
            for result in ("WIN", "DRAW", "LOSS"):
                pick[f"_{side}_form_{result.lower()}_rate"] = (
                    recent.count(result) / total if total else 0.0
                )

        enriched.append(pick)

        actual = match.get("actual")
        if actual in (LABEL_HOME, LABEL_DRAW, LABEL_AWAY):
            pair_winner[pair][_winner_name(match)] += 1
            home_side[home_key][actual] += 1
            away_side[away_key][actual] += 1
            home_result = _team_result(match, str(match.get("home", "")))
            away_result = _team_result(match, str(match.get("away", "")))
            team_any[home_key][home_result] += 1
            team_any[away_key][away_result] += 1
            recent_team_results[home_key].append(home_result)
            recent_team_results[away_key].append(away_result)
            league_season_counts[league_season_key] += 1

    return enriched


def _strategy_zoo_arrays(enriched: List[Dict]) -> Dict[str, object]:
    arrays = _history_edge_arrays(enriched)
    n = len(enriched)

    def _int_field(name: str, default: int = 0) -> np.ndarray:
        return np.array([int(p.get(name) or default) for p in enriched], dtype=int)

    def _label_field(name: str) -> np.ndarray:
        return np.array([
            int(p.get(name)) if p.get(name) in (LABEL_HOME, LABEL_DRAW, LABEL_AWAY) else -1
            for p in enriched
        ], dtype=int)

    def _float_field(name: str) -> np.ndarray:
        return np.array([float(p.get(name) or 0.0) for p in enriched], dtype=float)

    for prefix in ["pair", "home_home", "away_away", "home_any", "away_any"]:
        arrays[f"{prefix}_count"] = _int_field(f"_{prefix}_count")
        arrays[f"{prefix}_label"] = _label_field(f"_{prefix}_label")
        arrays[f"{prefix}_rate"] = _float_field(f"_{prefix}_rate")
    arrays["league_season_match_number"] = _int_field("_league_season_match_number", 1)

    for side in ["home", "away"]:
        arrays[f"{side}_form_count"] = _int_field(f"_{side}_form_count")
        for result in ["win", "draw", "loss"]:
            arrays[f"{side}_form_{result}_rate"] = _float_field(f"_{side}_form_{result}_rate")

    odds = arrays["odds"]
    avg_odds = np.array([
        [
            float(p.get("avg_home_odds") or p.get("home_odds") or 0.0),
            float(p.get("avg_draw_odds") or p.get("draw_odds") or 0.0),
            float(p.get("avg_away_odds") or p.get("away_odds") or 0.0),
        ]
        for p in enriched
    ], dtype=float)
    max_odds = np.array([
        [
            float(p.get("max_home_odds") or p.get("home_odds") or 0.0),
            float(p.get("max_draw_odds") or p.get("draw_odds") or 0.0),
            float(p.get("max_away_odds") or p.get("away_odds") or 0.0),
        ]
        for p in enriched
    ], dtype=float)
    b365_close_odds = np.array([
        [
            float(p.get("b365_close_home") or p.get("home_odds") or 0.0),
            float(p.get("b365_close_draw") or p.get("draw_odds") or 0.0),
            float(p.get("b365_close_away") or p.get("away_odds") or 0.0),
        ]
        for p in enriched
    ], dtype=float)
    avg_close_odds = np.array([
        [
            float(p.get("avg_close_home_odds") or p.get("avg_home_odds") or p.get("home_odds") or 0.0),
            float(p.get("avg_close_draw_odds") or p.get("avg_draw_odds") or p.get("draw_odds") or 0.0),
            float(p.get("avg_close_away_odds") or p.get("avg_away_odds") or p.get("away_odds") or 0.0),
        ]
        for p in enriched
    ], dtype=float)
    arrays["avg_odds"] = avg_odds
    arrays["max_odds"] = max_odds
    arrays["b365_close_odds"] = b365_close_odds
    arrays["avg_close_odds"] = avg_close_odds
    valid = odds > 1.0
    arrays["favorite_label"] = np.argmin(np.where(valid, odds, 999.0), axis=1).astype(int)
    arrays["underdog_label"] = np.argmax(np.where(valid, odds, -1.0), axis=1).astype(int)
    order = np.argsort(np.where(valid, odds, 999.0), axis=1)
    arrays["second_favorite_label"] = order[:, 1].astype(int) if n else np.array([], dtype=int)
    return arrays


def _strategy_zoo_sources(arrays: Dict[str, object]) -> List[Dict]:
    n = len(arrays["predictions"])
    zeros = np.zeros(n, dtype=float)
    ones = np.ones(n, dtype=float)

    def source(name: str, labels, count=None, rate=None, pre_mask=None, needs_history=False):
        return {
            "name": name,
            "labels": labels.astype(int),
            "count": count if count is not None else np.zeros(n, dtype=int),
            "rate": rate if rate is not None else ones,
            "pre_mask": pre_mask if pre_mask is not None else np.ones(n, dtype=bool),
            "needs_history": needs_history,
        }

    favorite = arrays["favorite_label"]
    underdog = arrays["underdog_label"]
    second_favorite = arrays["second_favorite_label"]
    h2h_label = arrays["h2h_label"]
    pair_label = arrays["pair_label"]
    home_form_count = arrays["home_form_count"]
    away_form_count = arrays["away_form_count"]
    home_win = arrays["home_form_win_rate"]
    home_loss = arrays["home_form_loss_rate"]
    home_draw = arrays["home_form_draw_rate"]
    away_win = arrays["away_form_win_rate"]
    away_loss = arrays["away_form_loss_rate"]
    away_draw = arrays["away_form_draw_rate"]

    form_count = np.minimum(home_form_count, away_form_count)
    home_form_rate = (home_win + away_loss) / 2
    away_form_rate = (away_win + home_loss) / 2
    draw_form_rate = (home_draw + away_draw) / 2

    sources = [
        source("always_home", np.full(n, LABEL_HOME), rate=zeros),
        source("always_draw", np.full(n, LABEL_DRAW), rate=zeros),
        source("always_away", np.full(n, LABEL_AWAY), rate=zeros),
        source("market_favorite", favorite, rate=zeros),
        source("market_second_favorite", second_favorite, rate=zeros),
        source("market_underdog", underdog, rate=zeros),
        source("direct_h2h", h2h_label, arrays["h2h_count"], arrays["h2h_rate"], needs_history=True),
        source("pair_history", pair_label, arrays["pair_count"], arrays["pair_rate"], needs_history=True),
        source("home_home_history", arrays["home_home_label"], arrays["home_home_count"], arrays["home_home_rate"], needs_history=True),
        source("away_away_history", arrays["away_away_label"], arrays["away_away_count"], arrays["away_away_rate"], needs_history=True),
        source("home_any_history", arrays["home_any_label"], arrays["home_any_count"], arrays["home_any_rate"], needs_history=True),
        source("away_any_history", arrays["away_any_label"], arrays["away_any_count"], arrays["away_any_rate"], needs_history=True),
        source(
            "favorite_direct_h2h_agree",
            np.where(favorite == h2h_label, favorite, -1),
            arrays["h2h_count"],
            arrays["h2h_rate"],
            needs_history=True,
        ),
        source(
            "favorite_pair_agree",
            np.where(favorite == pair_label, favorite, -1),
            arrays["pair_count"],
            arrays["pair_rate"],
            needs_history=True,
        ),
        source(
            "fade_direct_to_favorite",
            np.where((h2h_label >= 0) & (favorite != h2h_label), favorite, -1),
            arrays["h2h_count"],
            1 - arrays["h2h_rate"],
            needs_history=True,
        ),
        source(
            "home_form_vs_away_poor",
            np.full(n, LABEL_HOME),
            form_count,
            home_form_rate,
            pre_mask=(form_count >= 3),
            needs_history=True,
        ),
        source(
            "away_form_vs_home_poor",
            np.full(n, LABEL_AWAY),
            form_count,
            away_form_rate,
            pre_mask=(form_count >= 3),
            needs_history=True,
        ),
        source(
            "draw_form",
            np.full(n, LABEL_DRAW),
            form_count,
            draw_form_rate,
            pre_mask=(form_count >= 3),
            needs_history=True,
        ),
    ]
    return sources


def _strategy_zoo_trim(candidate: Dict) -> Dict:
    trimmed = _trim_history_edge_candidate(candidate)
    trimmed["strategy"] = candidate.get("strategy")
    trimmed["rate_min_pct"] = candidate.get("rate_min_pct")
    trimmed["history_count_min"] = candidate.get("history_count_min")
    return trimmed


def optimize_csv_strategy_zoo(matches: List[Dict], start_season: int, end_season: int) -> Dict:
    history_predictions = _csv_matches_to_history_predictions(matches)
    enriched = _enrich_strategy_zoo_history(history_predictions)
    arrays = _strategy_zoo_arrays(enriched)
    league_groups = _csv_history_league_groups(enriched)
    sources = _strategy_zoo_sources(arrays)
    odds = arrays["odds"]
    all_singles = []
    all_coupons = []
    coupon_filters = []
    evaluated_singles = 0
    evaluated_coupons = 0

    for src in sources:
        logger.info(f"Strategy zoo source: {src['name']}")
        labels = src["labels"]
        selected_confidence, selected_edge, selected_odds, valid = _selected_label_values(
            arrays,
            labels,
            "historical_outcome" if src["needs_history"] else "market",
        )
        if src["needs_history"]:
            selected_confidence = src["rate"]
            selected_edge = np.where(selected_odds > 1.0, src["rate"] - (1.0 / selected_odds), -99.0)
            count_options = STRATEGY_ZOO_HISTORY_COUNTS
            rate_options = STRATEGY_ZOO_HISTORY_RATES
            edge_options = STRATEGY_ZOO_EDGE_THRESHOLDS
        else:
            selected_confidence = np.where(selected_odds > 1.0, 1.0 / selected_odds, 0.0)
            selected_edge = np.zeros(len(enriched), dtype=float)
            count_options = [None]
            rate_options = [None]
            edge_options = [None]

        for league_filter, leagues in league_groups.items():
            league_mask = np.isin(arrays["league"], leagues)
            for count_min in count_options:
                count_mask = (
                    np.ones(len(enriched), dtype=bool)
                    if count_min is None
                    else src["count"] >= count_min
                )
                for rate_min in rate_options:
                    rate_mask = (
                        np.ones(len(enriched), dtype=bool)
                        if rate_min is None
                        else src["rate"] >= rate_min
                    )
                    for edge_min in edge_options:
                        edge_mask = (
                            np.ones(len(enriched), dtype=bool)
                            if edge_min is None
                            else selected_edge >= edge_min
                        )
                        for odds_band in STRATEGY_ZOO_ODDS_BANDS:
                            if odds_band is None:
                                odds_mask = np.ones(len(enriched), dtype=bool)
                            else:
                                odds_mask = (selected_odds >= odds_band[0]) & (selected_odds <= odds_band[1])

                            mask = (
                                valid
                                & src["pre_mask"]
                                & league_mask
                                & count_mask
                                & rate_mask
                                & edge_mask
                                & odds_mask
                            )
                            if int(np.sum(mask)) < 100:
                                continue

                            simulation = _simulate_flat_arrays(mask, labels, arrays)
                            score, eligible, reasons = _robust_score(simulation, "bets", 100)
                            evaluated_singles += 1
                            name = (
                                f"csv zoo single strategy={src['name']} leagues={league_filter} "
                                f"count>={count_min if count_min is not None else 'none'} "
                                f"rate>={_threshold_label(rate_min)} "
                                f"edge>={_threshold_label(edge_min)} odds={odds_band or 'any'}"
                            )
                            candidate = {
                                "type": "csv_strategy_zoo_single",
                                "mode": "csv_strategy_zoo",
                                "version": "market",
                                "name": name,
                                "strategy": src["name"],
                                "label_style": src["name"],
                                "league_filter": league_filter,
                                "leagues": leagues,
                                "history_count_min": count_min,
                                "rate_min_pct": None if rate_min is None else round(rate_min * 100, 1),
                                "model_edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
                                "odds_band": odds_band,
                                "simulation": simulation,
                                "score": score,
                                "eligible": eligible,
                                "rejection_reasons": reasons,
                            }
                            _remember_top_candidate(all_singles, _strategy_zoo_trim(candidate), 150)

                            filter_candidate = dict(candidate)
                            filter_candidate["mask"] = mask
                            filter_candidate["labels"] = labels
                            filter_candidate["selected_confidence"] = selected_confidence
                            filter_candidate["selected_edge"] = selected_edge
                            filter_candidate["arrays"] = arrays
                            _remember_top_candidate(coupon_filters, filter_candidate, STRATEGY_ZOO_COUPON_TOP_FILTERS)

    coupon_filters = sorted(coupon_filters, key=_candidate_rank_tuple, reverse=True)[:STRATEGY_ZOO_COUPON_TOP_FILTERS]
    seen_coupon_names = set()
    for filter_candidate in coupon_filters:
        selected = _history_edge_pick_predictions(filter_candidate, filter_candidate["arrays"])
        if len(selected) < 100:
            continue
        for max_legs in OPT_COUPON_MAX_LEGS:
            for sort_by in OPT_COUPON_SORTS:
                for max_per_league in OPT_COUPON_MAX_PER_LEAGUE:
                    simulation = simulate_coupon_bankroll(
                        selected,
                        max_legs=max_legs,
                        sort_by=sort_by,
                        max_per_league=max_per_league,
                    )
                    if simulation.get("coupons", 0) < 50:
                        continue
                    score, eligible, reasons = _robust_score(simulation, "coupons", 50)
                    evaluated_coupons += 1
                    name = (
                        f"{filter_candidate['name']} coupon max={max_legs} "
                        f"sort={sort_by} max_per_league={max_per_league}"
                    )
                    if name in seen_coupon_names:
                        continue
                    seen_coupon_names.add(name)
                    candidate = {
                        **{
                            key: filter_candidate.get(key)
                            for key in [
                                "mode", "version", "label_style", "league_filter", "leagues",
                                "history_count_min", "rate_min_pct", "model_edge_min_pct",
                                "odds_band", "strategy",
                            ]
                        },
                        "type": "csv_strategy_zoo_coupon",
                        "name": name,
                        "max_legs": max_legs,
                        "sort_by": sort_by,
                        "max_per_league": max_per_league,
                        "simulation": simulation,
                        "score": score,
                        "eligible": eligible,
                        "rejection_reasons": reasons,
                    }
                    _remember_top_candidate(all_coupons, _strategy_zoo_trim(candidate), 150)

    ranked_singles = sorted(all_singles, key=_candidate_rank_tuple, reverse=True)
    ranked_coupons = sorted(all_coupons, key=_candidate_rank_tuple, reverse=True)
    eligible_singles = [c for c in all_singles if c.get("eligible")]
    eligible_coupons = [c for c in all_coupons if c.get("eligible")]
    singles_by_profit = sorted(
        eligible_singles or all_singles,
        key=lambda c: (c.get("simulation", {}).get("profit", 0), -c.get("simulation", {}).get("max_drawdown", 0)),
        reverse=True,
    )
    coupons_by_profit = sorted(
        eligible_coupons or all_coupons,
        key=lambda c: (c.get("simulation", {}).get("profit", 0), -c.get("simulation", {}).get("max_drawdown", 0)),
        reverse=True,
    )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "csv_strategy_zoo_money_first_backtest_no_future_leakage",
        "source": "football-data.co.uk CSV",
        "start_season": start_season,
        "end_season": end_season,
        "source_matches": len(matches),
        "usable_matches": len(history_predictions),
        "strategies_tested": [src["name"] for src in sources],
        "rules": {
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "single_stake": DEFAULT_SINGLE_STAKE,
            "coupon_stake": DEFAULT_COUPON_STAKE,
            "no_future_leakage": True,
        },
        "evaluated_single_candidates": evaluated_singles,
        "evaluated_coupon_candidates": evaluated_coupons,
        "best_single_by_score": ranked_singles[0] if ranked_singles else None,
        "best_single_by_profit": singles_by_profit[0] if singles_by_profit else None,
        "best_coupon_by_score": ranked_coupons[0] if ranked_coupons else None,
        "best_coupon_by_profit": coupons_by_profit[0] if coupons_by_profit else None,
        "top_singles": ranked_singles[:75],
        "top_singles_by_profit": singles_by_profit[:75],
        "top_coupons": ranked_coupons[:75],
        "top_coupons_by_profit": coupons_by_profit[:75],
    }


def print_strategy_zoo_summary(results: Dict):
    print_header("CSV STRATEGY ZOO BACKTEST")
    print(f"  Seasons: {results.get('start_season')}-{results.get('end_season')}")
    print(f"  Source matches: {results.get('source_matches', 0)}")
    print(f"  Usable matches with 1X2 odds: {results.get('usable_matches', 0)}")
    print(f"  Strategy families: {len(results.get('strategies_tested', []))}")
    print(f"  Single candidates evaluated: {results.get('evaluated_single_candidates', 0)}")
    print(f"  Coupon candidates evaluated: {results.get('evaluated_coupon_candidates', 0)}")

    for title, key in [
        ("Best single by robust score", "best_single_by_score"),
        ("Best single by profit", "best_single_by_profit"),
        ("Best coupon by robust score", "best_coupon_by_score"),
        ("Best coupon by profit", "best_coupon_by_profit"),
    ]:
        candidate = results.get(key)
        if not candidate:
            print(f"  {title}: N/A")
            continue
        sim = candidate.get("simulation", {})
        count = sim.get("bets", sim.get("coupons", 0))
        hit = sim.get("accuracy", sim.get("coupon_hit_rate", 0.0))
        print()
        print(f"  {title}: {candidate.get('name')}")
        print(
            f"    Final: {sim.get('final_bankroll', 0):.0f} | "
            f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
            f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
        )
        print(f"    Count: {count} | Hit: {hit:.1f}% | Eligible: {candidate.get('eligible')}")
        print(f"    Reasons: {candidate.get('rejection_reasons', [])}")
        print(f"    By season: {sim.get('by_season', {})}")
    print()


def _odds_matrix_for_basis(arrays: Dict[str, object], basis: str) -> np.ndarray:
    if basis == "b365_close":
        return arrays.get("b365_close_odds", arrays["odds"])
    if basis == "avg":
        return arrays.get("avg_odds", arrays["odds"])
    if basis == "avg_close":
        return arrays.get("avg_close_odds", arrays.get("avg_odds", arrays["odds"]))
    if basis == "max":
        return arrays.get("max_odds", arrays["odds"])
    return arrays["odds"]


def _recent_form_signal(labels: np.ndarray, arrays: Dict[str, object]) -> np.ndarray:
    home_win = arrays["home_form_win_rate"]
    home_draw = arrays["home_form_draw_rate"]
    home_loss = arrays["home_form_loss_rate"]
    away_win = arrays["away_form_win_rate"]
    away_draw = arrays["away_form_draw_rate"]
    away_loss = arrays["away_form_loss_rate"]
    return np.where(
        labels == LABEL_HOME,
        (home_win + away_loss) / 2.0,
        np.where(
            labels == LABEL_DRAW,
            (home_draw + away_draw) / 2.0,
            (away_win + home_loss) / 2.0,
        ),
    )


def _history_edge_pick_predictions_with_odds(
    candidate: Dict,
    arrays: Dict[str, object],
    odds_matrix: np.ndarray,
) -> List[Dict]:
    selected = _history_edge_pick_predictions(candidate, arrays)
    idx = np.flatnonzero(candidate["mask"])
    labels = candidate["labels"]
    for pick, row_idx in zip(selected, idx):
        pick["home_odds"] = float(odds_matrix[row_idx, LABEL_HOME])
        pick["draw_odds"] = float(odds_matrix[row_idx, LABEL_DRAW])
        pick["away_odds"] = float(odds_matrix[row_idx, LABEL_AWAY])
        pick["odds_basis"] = candidate.get("odds_basis", "b365")
        pick["selected_odds"] = float(odds_matrix[row_idx, labels[row_idx]])
    return selected


def _combined_coupon_odds(legs: List[Dict]) -> float:
    combined = 1.0
    for leg in legs:
        combined *= _prediction_odds(leg) or 1.0
    return combined


def _simulate_coupon_candidate(
    candidate: Dict,
    arrays: Dict[str, object],
    mask: np.ndarray,
    max_legs: int,
    sort_by: str,
    max_per_league: Optional[int],
    combined_odds_max: Optional[float],
) -> Dict:
    odds_matrix = candidate["odds_matrix"]
    selected = _history_edge_pick_predictions_with_odds({**candidate, "mask": mask}, arrays, odds_matrix)
    batches, skipped_no_odds = _build_coupon_batches(
        selected,
        max_legs=max_legs,
        sort_by=sort_by,
        max_per_league=max_per_league,
    )
    if combined_odds_max is not None:
        batches = [
            legs for legs in batches
            if _combined_coupon_odds(legs) <= combined_odds_max
        ]
    return simulate_coupon_batches(
        batches,
        max_legs=max_legs,
        sort_by=sort_by,
        max_per_league=max_per_league,
        skipped_no_odds=skipped_no_odds,
    )


def _strategy_zoo_filter_candidate(
    src: Dict,
    league_filter: str,
    leagues: List[str],
    count_min: Optional[int],
    rate_min: Optional[float],
    edge_min: Optional[float],
    odds_band: Optional[Tuple[float, float]],
    mask: np.ndarray,
    labels: np.ndarray,
    selected_confidence: np.ndarray,
    selected_edge: np.ndarray,
    arrays: Dict[str, object],
) -> Dict:
    name = (
        f"csv zoo single strategy={src['name']} leagues={league_filter} "
        f"count>={count_min if count_min is not None else 'none'} "
        f"rate>={_threshold_label(rate_min)} "
        f"edge>={_threshold_label(edge_min)} odds={odds_band or 'any'}"
    )
    return {
        "type": "csv_strategy_zoo_single",
        "mode": "csv_strategy_zoo_walk_forward",
        "version": "market",
        "name": name,
        "strategy": src["name"],
        "label_style": src["name"],
        "league_filter": league_filter,
        "leagues": leagues,
        "history_count_min": count_min,
        "rate_min_pct": None if rate_min is None else round(rate_min * 100, 1),
        "model_edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
        "odds_band": odds_band,
        "mask": mask,
        "labels": labels,
        "selected_confidence": selected_confidence,
        "selected_edge": selected_edge,
        "arrays": arrays,
    }


def walk_forward_csv_strategy_zoo(
    matches: List[Dict],
    start_season: int,
    end_season: int,
    first_test_season: Optional[int] = None,
    min_train_bets: int = STRATEGY_ZOO_WF_MIN_TRAIN_BETS,
) -> Dict:
    """
    Choose strategies using only earlier seasons, then test the chosen strategy
    on the next season. This estimates how much of the strategy-zoo edge survives
    without hindsight selection.
    """
    history_predictions = _csv_matches_to_history_predictions(matches)
    enriched = _enrich_strategy_zoo_history(history_predictions)
    arrays = _strategy_zoo_arrays(enriched)
    league_groups = _csv_history_league_groups(enriched)
    sources = [
        src for src in _strategy_zoo_sources(arrays)
        if src["name"] in STRATEGY_ZOO_WF_SOURCE_ALLOWLIST
    ]
    season_array = np.array([int(season) for season in arrays["season"]], dtype=int)
    seasons = [
        int(season)
        for season in sorted(set(season_array.tolist()))
        if start_season <= int(season) <= end_season
    ]
    if first_test_season is None:
        first_test_season = max(start_season + 5, seasons[0] if seasons else start_season)

    all_single_test_picks: List[Dict] = []
    all_coupon_test_batches: List[List[Dict]] = []
    total_coupon_skipped_no_odds = 0
    folds = []
    total_evaluated_single_candidates = 0
    total_evaluated_coupon_candidates = 0

    for test_season in seasons:
        if test_season < first_test_season:
            continue
        train_mask = season_array < test_season
        test_mask = season_array == test_season
        if not np.any(test_mask):
            continue

        best_single = None
        top_train_filters = []
        evaluated_single_candidates = 0

        logger.info(f"Walk-forward strategy zoo fold: train < {test_season}, test {test_season}")
        for src in sources:
            labels = src["labels"]
            selected_confidence, selected_edge, selected_odds, valid = _selected_label_values(
                arrays,
                labels,
                "historical_outcome" if src["needs_history"] else "market",
            )
            if src["needs_history"]:
                selected_confidence = src["rate"]
                selected_edge = np.where(selected_odds > 1.0, src["rate"] - (1.0 / selected_odds), -99.0)
                count_options = STRATEGY_ZOO_WF_HISTORY_COUNTS
                rate_options = STRATEGY_ZOO_WF_HISTORY_RATES
                edge_options = STRATEGY_ZOO_WF_EDGE_THRESHOLDS
            else:
                selected_confidence = np.where(selected_odds > 1.0, 1.0 / selected_odds, 0.0)
                selected_edge = np.zeros(len(enriched), dtype=float)
                count_options = [None]
                rate_options = [None]
                edge_options = [None]

            for league_filter, leagues in league_groups.items():
                league_mask = np.isin(arrays["league"], leagues)
                for count_min in count_options:
                    count_mask = (
                        np.ones(len(enriched), dtype=bool)
                        if count_min is None
                        else src["count"] >= count_min
                    )
                    for rate_min in rate_options:
                        rate_mask = (
                            np.ones(len(enriched), dtype=bool)
                            if rate_min is None
                            else src["rate"] >= rate_min
                        )
                        for edge_min in edge_options:
                            edge_mask = (
                                np.ones(len(enriched), dtype=bool)
                                if edge_min is None
                                else selected_edge >= edge_min
                            )
                            for odds_band in STRATEGY_ZOO_WF_ODDS_BANDS:
                                if odds_band is None:
                                    odds_mask = np.ones(len(enriched), dtype=bool)
                                else:
                                    odds_mask = (selected_odds >= odds_band[0]) & (selected_odds <= odds_band[1])

                                base_mask = (
                                    valid
                                    & src["pre_mask"]
                                    & league_mask
                                    & count_mask
                                    & rate_mask
                                    & edge_mask
                                    & odds_mask
                                )
                                candidate_train_mask = base_mask & train_mask
                                if int(np.sum(candidate_train_mask)) < min_train_bets:
                                    continue

                                simulation = _simulate_flat_arrays(candidate_train_mask, labels, arrays)
                                score, eligible, reasons = _robust_score(
                                    simulation,
                                    "bets",
                                    min_train_bets,
                                )
                                candidate = _strategy_zoo_filter_candidate(
                                    src,
                                    league_filter,
                                    leagues,
                                    count_min,
                                    rate_min,
                                    edge_min,
                                    odds_band,
                                    base_mask,
                                    labels,
                                    selected_confidence,
                                    selected_edge,
                                    arrays,
                                )
                                candidate.update({
                                    "simulation": simulation,
                                    "score": score,
                                    "eligible": eligible,
                                    "rejection_reasons": reasons,
                                })
                                evaluated_single_candidates += 1

                                if best_single is None or _candidate_rank_tuple(candidate) > _candidate_rank_tuple(best_single):
                                    best_single = candidate
                                _remember_top_candidate(
                                    top_train_filters,
                                    candidate,
                                    STRATEGY_ZOO_WF_COUPON_TOP_FILTERS,
                                )

        total_evaluated_single_candidates += evaluated_single_candidates

        fold = {
            "test_season": test_season,
            "evaluated_single_candidates": evaluated_single_candidates,
        }
        if best_single is None:
            fold["status"] = "no_train_candidate"
            folds.append(fold)
            continue

        single_test_mask = best_single["mask"] & test_mask
        single_test_picks = _history_edge_pick_predictions(
            {**best_single, "mask": single_test_mask},
            arrays,
        )
        all_single_test_picks.extend(single_test_picks)
        single_test_sim = simulate_flat_bankroll(single_test_picks)
        chosen_single = _strategy_zoo_trim(best_single)
        chosen_single["train_simulation"] = chosen_single.pop("simulation")
        chosen_single["test_simulation"] = single_test_sim
        fold["chosen_single"] = chosen_single

        best_coupon = None
        evaluated_coupon_candidates = 0
        for filter_candidate in sorted(top_train_filters, key=_candidate_rank_tuple, reverse=True):
            coupon_train_picks = _history_edge_pick_predictions(
                {**filter_candidate, "mask": filter_candidate["mask"] & train_mask},
                arrays,
            )
            if len(coupon_train_picks) < min_train_bets:
                continue
            for max_legs in [2, 3, 4]:
                for sort_by in ["confidence", "edge_x_confidence"]:
                    for max_per_league in [1, 2]:
                        train_coupon_sim = simulate_coupon_bankroll(
                            coupon_train_picks,
                            max_legs=max_legs,
                            sort_by=sort_by,
                            max_per_league=max_per_league,
                        )
                        if train_coupon_sim.get("coupons", 0) < max(20, min_train_bets // 4):
                            continue
                        score, eligible, reasons = _robust_score(
                            train_coupon_sim,
                            "coupons",
                            max(20, min_train_bets // 4),
                        )
                        coupon_candidate = {
                            **{
                                key: filter_candidate.get(key)
                                for key in [
                                    "mode", "version", "label_style", "league_filter", "leagues",
                                    "history_count_min", "rate_min_pct", "model_edge_min_pct",
                                    "odds_band", "strategy", "mask", "labels",
                                    "selected_confidence", "selected_edge", "arrays",
                                ]
                            },
                            "type": "csv_strategy_zoo_walk_forward_coupon",
                            "name": (
                                f"{filter_candidate['name']} coupon max={max_legs} "
                                f"sort={sort_by} max_per_league={max_per_league}"
                            ),
                            "max_legs": max_legs,
                            "sort_by": sort_by,
                            "max_per_league": max_per_league,
                            "simulation": train_coupon_sim,
                            "score": score,
                            "eligible": eligible,
                            "rejection_reasons": reasons,
                        }
                        evaluated_coupon_candidates += 1
                        if best_coupon is None or _candidate_rank_tuple(coupon_candidate) > _candidate_rank_tuple(best_coupon):
                            best_coupon = coupon_candidate

        total_evaluated_coupon_candidates += evaluated_coupon_candidates
        fold["evaluated_coupon_candidates"] = evaluated_coupon_candidates

        if best_coupon is not None:
            coupon_test_picks = _history_edge_pick_predictions(
                {**best_coupon, "mask": best_coupon["mask"] & test_mask},
                arrays,
            )
            coupon_batches, skipped_no_odds = _build_coupon_batches(
                coupon_test_picks,
                max_legs=best_coupon["max_legs"],
                sort_by=best_coupon["sort_by"],
                max_per_league=best_coupon["max_per_league"],
            )
            all_coupon_test_batches.extend(coupon_batches)
            total_coupon_skipped_no_odds += skipped_no_odds
            coupon_test_sim = simulate_coupon_batches(
                coupon_batches,
                max_legs=best_coupon["max_legs"],
                sort_by=best_coupon["sort_by"],
                max_per_league=best_coupon["max_per_league"],
                skipped_no_odds=skipped_no_odds,
            )
            chosen_coupon = _strategy_zoo_trim(best_coupon)
            chosen_coupon["train_simulation"] = chosen_coupon.pop("simulation")
            chosen_coupon["test_simulation"] = coupon_test_sim
            fold["chosen_coupon"] = chosen_coupon

        folds.append(fold)

    combined_single = simulate_flat_bankroll(all_single_test_picks)
    max_batch_legs = max((len(batch) for batch in all_coupon_test_batches), default=DEFAULT_COUPON_MAX_LEGS)
    combined_coupon = simulate_coupon_batches(
        all_coupon_test_batches,
        max_legs=max_batch_legs,
        sort_by="walk_forward_selected",
        max_per_league=None,
        skipped_no_odds=total_coupon_skipped_no_odds,
    )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "csv_strategy_zoo_walk_forward_no_hindsight_selection",
        "source": "football-data.co.uk CSV",
        "start_season": start_season,
        "end_season": end_season,
        "first_test_season": first_test_season,
        "source_matches": len(matches),
        "usable_matches": len(history_predictions),
        "strategies_tested": sorted(STRATEGY_ZOO_WF_SOURCE_ALLOWLIST),
        "rules": {
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "single_stake": DEFAULT_SINGLE_STAKE,
            "coupon_stake": DEFAULT_COUPON_STAKE,
            "min_train_bets": min_train_bets,
            "no_future_leakage": True,
            "no_hindsight_strategy_selection": True,
        },
        "evaluated_single_candidates": total_evaluated_single_candidates,
        "evaluated_coupon_candidates": total_evaluated_coupon_candidates,
        "combined_single": combined_single,
        "combined_coupon": combined_coupon,
        "folds": folds,
    }


def print_strategy_zoo_walk_forward_summary(results: Dict):
    print_header("CSV STRATEGY ZOO WALK-FORWARD")
    print(f"  Seasons: {results.get('start_season')}-{results.get('end_season')}")
    print(f"  First test season: {results.get('first_test_season')}")
    print(f"  Source matches: {results.get('source_matches', 0)}")
    print(f"  Usable matches with 1X2 odds: {results.get('usable_matches', 0)}")
    print(f"  Single candidates evaluated: {results.get('evaluated_single_candidates', 0)}")
    print(f"  Coupon candidates evaluated: {results.get('evaluated_coupon_candidates', 0)}")

    for title, key in [
        ("Out-of-sample singles", "combined_single"),
        ("Out-of-sample coupons", "combined_coupon"),
    ]:
        sim = results.get(key, {})
        count = sim.get("bets", sim.get("coupons", 0))
        hit = sim.get("accuracy", sim.get("coupon_hit_rate", 0.0))
        print()
        print(f"  {title}")
        print(
            f"    Final: {sim.get('final_bankroll', 0):.0f} | "
            f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
            f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
        )
        print(f"    Count: {count} | Hit: {hit:.1f}%")
        print(f"    By season: {sim.get('by_season', {})}")

    print()
    print("  Fold choices:")
    for fold in results.get("folds", []):
        single = fold.get("chosen_single")
        coupon = fold.get("chosen_coupon")
        single_profit = single.get("test_simulation", {}).get("profit", 0.0) if single else 0.0
        coupon_profit = coupon.get("test_simulation", {}).get("profit", 0.0) if coupon else 0.0
        print(
            f"    {fold.get('test_season')}: "
            f"single {single_profit:+.0f} {single.get('strategy') if single else 'N/A'} | "
            f"coupon {coupon_profit:+.0f} {coupon.get('strategy') if coupon else 'N/A'}"
        )
    print()


def _poisson_over25_probability(total_lambda: float) -> float:
    total_lambda = max(0.05, float(total_lambda or 0.0))
    under = sum(
        (total_lambda ** goals) * math.exp(-total_lambda) / math.factorial(goals)
        for goals in range(3)
    )
    return max(0.0, min(1.0, 1.0 - under))


def _binary_no_vig_prob(odds_a: float, odds_b: float) -> Tuple[float, float]:
    if not odds_a or not odds_b or odds_a <= 1.0 or odds_b <= 1.0:
        return 0.0, 0.0
    inv_a = 1.0 / odds_a
    inv_b = 1.0 / odds_b
    total = inv_a + inv_b
    if total <= 0:
        return 0.0, 0.0
    return inv_a / total, inv_b / total


def _mean_recent(values: List[float], limit: int) -> float:
    sample = values[-limit:] if limit > 0 else values
    return float(sum(sample) / len(sample)) if sample else 0.0


def _build_over_under_rows(matches: List[Dict]) -> List[Dict]:
    """Build O/U 2.5 rows from as-of history and complete market quotes.

    Unpriced matches still update the rolling football state. Matches sharing a
    kickoff timestamp are evaluated as one batch, so one simultaneous result
    cannot leak into another match's pre-match features.
    """
    team_stats = defaultdict(lambda: {
        "matches": 0,
        "gf": 0.0,
        "ga": 0.0,
        "overs": 0,
        "recent_totals": [],
        "recent_overs": [],
    })
    league_stats = defaultdict(lambda: {"matches": 0, "overs": 0, "recent_overs": []})
    pair_stats = defaultdict(lambda: {"matches": 0, "overs": 0})
    rows = []

    scored_matches = []
    for match in sorted(matches, key=lambda m: m.get("match_date", "")):
        try:
            scored_matches.append((match, int(match.get("home_score")), int(match.get("away_score"))))
        except (TypeError, ValueError):
            continue

    for _, kickoff_group in groupby(
        scored_matches,
        key=lambda item: item[0].get("match_date", ""),
    ):
        pending_updates = []
        for match, hs, aws in kickoff_group:
            league = str(match.get("league_code", ""))
            home = str(match.get("home_team_name", ""))
            away = str(match.get("away_team_name", ""))
            season = int(match.get("season") or 0)
            home_key = (league, home)
            away_key = (league, away)
            pair_key = (league, tuple(sorted([home.lower(), away.lower()])))

            home_hist = team_stats[home_key]
            away_hist = team_stats[away_key]
            league_hist = league_stats[league]
            pair_hist = pair_stats[pair_key]

            home_matches = int(home_hist["matches"])
            away_matches = int(away_hist["matches"])
            league_matches = int(league_hist["matches"])
            pair_matches = int(pair_hist["matches"])

            home_gf = home_hist["gf"] / home_matches if home_matches else 1.30
            home_ga = home_hist["ga"] / home_matches if home_matches else 1.25
            away_gf = away_hist["gf"] / away_matches if away_matches else 1.15
            away_ga = away_hist["ga"] / away_matches if away_matches else 1.35

            expected_home = max(0.15, (home_gf + away_ga) / 2)
            expected_away = max(0.15, (away_gf + home_ga) / 2)
            poisson_over = _poisson_over25_probability(expected_home + expected_away)

            home_recent_over = _mean_recent(home_hist["recent_overs"], 10)
            away_recent_over = _mean_recent(away_hist["recent_overs"], 10)
            recent_team_over = (
                (home_recent_over + away_recent_over) / 2
                if home_hist["recent_overs"] and away_hist["recent_overs"]
                else 0.0
            )
            league_over = league_hist["overs"] / league_matches if league_matches else 0.0
            league_recent_over = _mean_recent(league_hist["recent_overs"], 100)
            pair_over = pair_hist["overs"] / pair_matches if pair_matches else 0.0

            extra = match.get("extra_data") or {}
            quote_options = [
                ("average_open", extra.get("avg_under25"), extra.get("avg_over25")),
                ("bet365_open", extra.get("b365_under25"), extra.get("b365_over25")),
                ("pinnacle_open", extra.get("pinnacle_under25"), extra.get("pinnacle_over25")),
            ]
            odds_basis = None
            under_odds = over_odds = 0.0
            for basis, under_value, over_value in quote_options:
                try:
                    candidate_under = float(under_value or 0.0)
                    candidate_over = float(over_value or 0.0)
                except (TypeError, ValueError):
                    continue
                if candidate_under > 1.0 and candidate_over > 1.0:
                    odds_basis = basis
                    under_odds = candidate_under
                    over_odds = candidate_over
                    break

            if odds_basis is not None:
                market_under, market_over = _binary_no_vig_prob(under_odds, over_odds)
                market_label = OU_UNDER if under_odds < over_odds else OU_OVER
                poisson_label = OU_OVER if poisson_over >= 0.5 else OU_UNDER
                recent_label = OU_OVER if recent_team_over >= 0.5 else OU_UNDER
                league_label = OU_OVER if league_over >= 0.5 else OU_UNDER
                pair_label = OU_OVER if pair_over >= 0.5 else OU_UNDER
                actual = OU_OVER if (hs + aws) > 2.5 else OU_UNDER

                rows.append({
                    "match_date": match.get("match_date", ""),
                    "league": league,
                    "season": season,
                    "home": home,
                    "away": away,
                    "home_score": hs,
                    "away_score": aws,
                    "actual": actual,
                    "odds_basis": odds_basis,
                    "under25_odds": under_odds,
                    "over25_odds": over_odds,
                    "home_history_matches": home_matches,
                    "away_history_matches": away_matches,
                    "team_history_matches": min(home_matches, away_matches),
                    "league_history_matches": league_matches,
                    "pair_history_matches": pair_matches,
                    "poisson_over25_prob": poisson_over,
                    "poisson_under25_prob": 1.0 - poisson_over,
                    "recent_team_over25_rate": recent_team_over,
                    "league_over25_rate": league_over,
                    "league_recent_over25_rate": league_recent_over,
                    "pair_over25_rate": pair_over,
                    "market_under25_prob": market_under,
                    "market_over25_prob": market_over,
                    "market_label": market_label,
                    "poisson_label": poisson_label,
                    "recent_team_label": recent_label,
                    "league_label": league_label,
                    "pair_label": pair_label,
                })

            pending_updates.append((home_key, away_key, pair_key, league, hs, aws))

        for home_key, away_key, pair_key, league, hs, aws in pending_updates:
            total_goals = hs + aws
            is_over = 1 if total_goals > 2.5 else 0
            for key, gf, ga in [(home_key, hs, aws), (away_key, aws, hs)]:
                stats = team_stats[key]
                stats["matches"] += 1
                stats["gf"] += gf
                stats["ga"] += ga
                stats["overs"] += is_over
                stats["recent_totals"].append(total_goals)
                stats["recent_overs"].append(is_over)
            league_hist = league_stats[league]
            league_hist["matches"] += 1
            league_hist["overs"] += is_over
            league_hist["recent_overs"].append(is_over)
            pair_hist = pair_stats[pair_key]
            pair_hist["matches"] += 1
            pair_hist["overs"] += is_over

    return rows


def _over_under_arrays(rows: List[Dict]) -> Dict[str, object]:
    def float_col(name: str) -> np.ndarray:
        return np.array([float(row.get(name) or 0.0) for row in rows], dtype=float)

    def int_col(name: str) -> np.ndarray:
        return np.array([int(row.get(name) or 0) for row in rows], dtype=int)

    odds = np.stack([float_col("under25_odds"), float_col("over25_odds")], axis=1)
    prob = np.stack([float_col("poisson_under25_prob"), float_col("poisson_over25_prob")], axis=1)
    market_prob = np.stack([float_col("market_under25_prob"), float_col("market_over25_prob")], axis=1)
    return {
        "rows": rows,
        "actual": int_col("actual"),
        "season": int_col("season"),
        "league": np.array([str(row.get("league", "")) for row in rows], dtype=object),
        "odds": odds,
        "prob": prob,
        "market_prob": market_prob,
        "market_label": int_col("market_label"),
        "poisson_label": int_col("poisson_label"),
        "recent_team_label": int_col("recent_team_label"),
        "league_label": int_col("league_label"),
        "pair_label": int_col("pair_label"),
        "team_history_matches": int_col("team_history_matches"),
        "league_history_matches": int_col("league_history_matches"),
        "pair_history_matches": int_col("pair_history_matches"),
        "recent_team_over25_rate": float_col("recent_team_over25_rate"),
        "league_over25_rate": float_col("league_over25_rate"),
        "pair_over25_rate": float_col("pair_over25_rate"),
    }


def _ou_league_groups(rows: List[Dict]) -> Dict[str, List[str]]:
    leagues = sorted({row.get("league") for row in rows if row.get("league")})
    groups = {"all": leagues}
    top = [league for league in leagues if league in TOP_LEAGUE_CODES]
    if top:
        groups["top_leagues"] = top
    for league in top:
        groups[f"league:{league}"] = [league]
    return groups


def _ou_source_values(arrays: Dict[str, object], source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(arrays["rows"])
    odds = arrays["odds"]
    prob = arrays["prob"]
    market_prob = arrays["market_prob"]
    labels = np.zeros(n, dtype=int)
    confidence = np.zeros(n, dtype=float)
    pre_mask = np.ones(n, dtype=bool)

    if source == "market_favorite":
        labels = np.argmin(np.where(odds > 1.0, odds, 999.0), axis=1).astype(int)
        confidence = market_prob[np.arange(n), labels]
    elif source == "market_underdog":
        labels = np.argmax(np.where(odds > 1.0, odds, -1.0), axis=1).astype(int)
        confidence = market_prob[np.arange(n), labels]
    elif source in ("poisson_total", "poisson_edge"):
        labels = arrays["poisson_label"].copy()
        confidence = prob[np.arange(n), labels]
        pre_mask = arrays["team_history_matches"] >= 5
        if source == "poisson_edge":
            ev = np.where(odds > 1.0, prob * odds - 1.0, -99.0)
            labels = np.argmax(ev, axis=1).astype(int)
            confidence = prob[np.arange(n), labels]
    elif source == "recent_team_total_rate":
        rate = arrays["recent_team_over25_rate"]
        labels = np.where(rate >= 0.5, OU_OVER, OU_UNDER).astype(int)
        confidence = np.where(labels == OU_OVER, rate, 1.0 - rate)
        pre_mask = arrays["team_history_matches"] >= 5
    elif source == "league_total_rate":
        rate = arrays["league_over25_rate"]
        labels = np.where(rate >= 0.5, OU_OVER, OU_UNDER).astype(int)
        confidence = np.where(labels == OU_OVER, rate, 1.0 - rate)
        pre_mask = arrays["league_history_matches"] >= 50
    elif source == "pair_total_history":
        rate = arrays["pair_over25_rate"]
        labels = np.where(rate >= 0.5, OU_OVER, OU_UNDER).astype(int)
        confidence = np.where(labels == OU_OVER, rate, 1.0 - rate)
        pre_mask = arrays["pair_history_matches"] >= 2
    elif source == "market_poisson_agree":
        labels = arrays["poisson_label"].copy()
        confidence = prob[np.arange(n), labels]
        pre_mask = (arrays["market_label"] == labels) & (arrays["team_history_matches"] >= 5)
    else:
        labels = arrays["poisson_label"].copy()
        confidence = prob[np.arange(n), labels]

    selected_odds = odds[np.arange(n), labels]
    edge = np.where(selected_odds > 1.0, confidence * selected_odds - 1.0, -99.0)
    valid = pre_mask & (selected_odds > 1.0) & np.isfinite(confidence)
    return labels, confidence, edge, valid


def _simulate_ou_arrays(mask: np.ndarray, labels: np.ndarray, arrays: Dict[str, object]) -> Dict:
    idx = np.flatnonzero(mask)
    odds = arrays["odds"]
    actual = arrays["actual"]
    season = arrays["season"]
    selected_odds = odds[idx, labels[idx]]
    wins = labels[idx] == actual[idx]
    deltas = np.where(wins, (DEFAULT_SINGLE_STAKE * selected_odds) - DEFAULT_SINGLE_STAKE, -DEFAULT_SINGLE_STAKE)
    cumulative = np.cumsum(deltas) if len(deltas) else np.array([])
    final_bankroll = DEFAULT_STARTING_BANKROLL + (float(cumulative[-1]) if len(cumulative) else 0.0)
    bankroll_curve = DEFAULT_STARTING_BANKROLL + cumulative if len(cumulative) else np.array([])
    if len(bankroll_curve):
        peaks = np.maximum.accumulate(np.concatenate(([DEFAULT_STARTING_BANKROLL], bankroll_curve)))[1:]
        max_drawdown = float(np.max(peaks - bankroll_curve))
        peak_bankroll = float(max(DEFAULT_STARTING_BANKROLL, np.max(bankroll_curve)))
        min_bankroll = float(min(DEFAULT_STARTING_BANKROLL, np.min(bankroll_curve)))
    else:
        max_drawdown = 0.0
        peak_bankroll = DEFAULT_STARTING_BANKROLL
        min_bankroll = DEFAULT_STARTING_BANKROLL

    season_stats: Dict[str, Dict] = defaultdict(_empty_period_stats)
    for selected_idx, odd, won in zip(idx, selected_odds, wins):
        key = str(int(season[selected_idx]))
        season_stats[key]["bets"] += 1
        season_stats[key]["wins"] += 1 if won else 0
        season_stats[key]["staked"] += DEFAULT_SINGLE_STAKE
        season_stats[key]["returned"] += DEFAULT_SINGLE_STAKE * float(odd) if won else 0.0

    bets = int(len(idx))
    staked = bets * DEFAULT_SINGLE_STAKE
    profit = final_bankroll - DEFAULT_STARTING_BANKROLL
    return {
        "starting_bankroll": round(DEFAULT_STARTING_BANKROLL, 2),
        "stake": round(DEFAULT_SINGLE_STAKE, 2),
        "bets": bets,
        "wins": int(np.sum(wins)) if bets else 0,
        "accuracy": round((float(np.mean(wins)) * 100) if bets else 0.0, 2),
        "staked": round(staked, 2),
        "returned": round(staked + profit, 2),
        "final_bankroll": round(final_bankroll, 2),
        "profit": round(profit, 2),
        "growth_pct": round((profit / DEFAULT_STARTING_BANKROLL * 100) if DEFAULT_STARTING_BANKROLL else 0.0, 2),
        "roi_pct": round((profit / staked * 100) if staked else 0.0, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": round((max_drawdown / peak_bankroll * 100) if peak_bankroll else 0.0, 2),
        "min_bankroll": round(min_bankroll, 2),
        "stopped_bankroll_depleted": False,
        "skipped_no_odds": 0,
        "by_season": _finalize_period_stats(season_stats),
    }


def _trim_ou_candidate(candidate: Dict) -> Dict:
    sim = candidate.get("simulation", {})
    return {
        "type": candidate.get("type"),
        "mode": candidate.get("mode"),
        "market": "over_under_2_5",
        "name": candidate.get("name"),
        "source": candidate.get("source"),
        "league_filter": candidate.get("league_filter"),
        "leagues": candidate.get("leagues"),
        "confidence_min_pct": candidate.get("confidence_min_pct"),
        "edge_min_pct": candidate.get("edge_min_pct"),
        "history_min": candidate.get("history_min"),
        "rate_min_pct": candidate.get("rate_min_pct"),
        "odds_band": candidate.get("odds_band"),
        "score": candidate.get("score"),
        "eligible": candidate.get("eligible"),
        "rejection_reasons": candidate.get("rejection_reasons", []),
        "simulation": {
            k: sim.get(k)
            for k in [
                "bets", "wins", "accuracy", "final_bankroll", "profit",
                "roi_pct", "max_drawdown", "max_drawdown_pct", "by_season",
            ]
            if k in sim
        },
    }


def walk_forward_over_under_25(
    matches: List[Dict],
    start_season: int,
    end_season: int,
    first_test_season: Optional[int] = None,
    min_train_bets: int = STRATEGY_ZOO_WF_MIN_TRAIN_BETS,
) -> Dict:
    rows = _build_over_under_rows(matches)
    arrays = _over_under_arrays(rows)
    season_array = arrays["season"]
    seasons = [
        int(season)
        for season in sorted(set(season_array.tolist()))
        if start_season <= int(season) <= end_season
    ]
    if first_test_season is None:
        first_test_season = max(start_season + 5, seasons[0] if seasons else start_season)

    league_groups = _ou_league_groups(rows)
    all_test_mask = np.zeros(len(rows), dtype=bool)
    all_test_labels = np.zeros(len(rows), dtype=int)
    folds = []
    total_evaluated = 0

    for test_season in seasons:
        if test_season < first_test_season:
            continue
        train_mask = season_array < test_season
        test_mask = season_array == test_season
        if not np.any(test_mask):
            continue

        best = None
        evaluated = 0
        logger.info(f"O/U 2.5 walk-forward fold: train < {test_season}, test {test_season}")

        for source in sorted(OU_WF_SOURCE_ALLOWLIST):
            labels, confidence, edge, valid = _ou_source_values(arrays, source)
            if source == "pair_total_history":
                history_options = OU_MIN_PAIR_MATCHES
                history_array = arrays["pair_history_matches"]
                rate_options = OU_RATE_THRESHOLDS
            elif source == "league_total_rate":
                history_options = OU_MIN_LEAGUE_MATCHES
                history_array = arrays["league_history_matches"]
                rate_options = OU_RATE_THRESHOLDS
            elif source in ("recent_team_total_rate", "poisson_total", "poisson_edge", "market_poisson_agree"):
                history_options = OU_MIN_TEAM_MATCHES
                history_array = arrays["team_history_matches"]
                rate_options = [None]
            else:
                history_options = [None]
                history_array = np.zeros(len(rows), dtype=int)
                rate_options = [None]

            for league_filter, leagues in league_groups.items():
                league_mask = np.isin(arrays["league"], leagues)
                for history_min in history_options:
                    history_mask = (
                        np.ones(len(rows), dtype=bool)
                        if history_min is None
                        else history_array >= history_min
                    )
                    for rate_min in rate_options:
                        rate_mask = (
                            np.ones(len(rows), dtype=bool)
                            if rate_min is None
                            else confidence >= rate_min
                        )
                        for conf_min in OU_CONF_THRESHOLDS:
                            conf_mask = (
                                np.ones(len(rows), dtype=bool)
                                if conf_min is None
                                else confidence >= conf_min
                            )
                            for edge_min in OU_EDGE_THRESHOLDS:
                                edge_mask = (
                                    np.ones(len(rows), dtype=bool)
                                    if edge_min is None
                                    else edge >= edge_min
                                )
                                selected_odds = arrays["odds"][np.arange(len(rows)), labels]
                                for odds_band in OU_ODDS_BANDS:
                                    odds_mask = (
                                        np.ones(len(rows), dtype=bool)
                                        if odds_band is None
                                        else (selected_odds >= odds_band[0]) & (selected_odds <= odds_band[1])
                                    )
                                    base_mask = valid & league_mask & history_mask & rate_mask & conf_mask & edge_mask & odds_mask
                                    candidate_train_mask = base_mask & train_mask
                                    if int(np.sum(candidate_train_mask)) < min_train_bets:
                                        continue

                                    simulation = _simulate_ou_arrays(candidate_train_mask, labels, arrays)
                                    score, eligible, reasons = _robust_score(simulation, "bets", min_train_bets)
                                    candidate = {
                                        "type": "over_under_25_single",
                                        "mode": "over_under_25_walk_forward",
                                        "name": (
                                            f"ou25 source={source} leagues={league_filter} "
                                            f"history>={history_min if history_min is not None else 'none'} "
                                            f"rate>={_threshold_label(rate_min)} "
                                            f"conf>={_threshold_label(conf_min)} "
                                            f"edge>={_threshold_label(edge_min)} odds={odds_band or 'any'}"
                                        ),
                                        "source": source,
                                        "league_filter": league_filter,
                                        "leagues": leagues,
                                        "history_min": history_min,
                                        "rate_min_pct": None if rate_min is None else round(rate_min * 100, 1),
                                        "confidence_min_pct": None if conf_min is None else round(conf_min * 100, 1),
                                        "edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
                                        "odds_band": odds_band,
                                        "mask": base_mask,
                                        "labels": labels,
                                        "simulation": simulation,
                                        "score": score,
                                        "eligible": eligible,
                                        "rejection_reasons": reasons,
                                    }
                                    evaluated += 1
                                    if best is None or _candidate_rank_tuple(candidate) > _candidate_rank_tuple(best):
                                        best = candidate

        total_evaluated += evaluated
        fold = {"test_season": test_season, "evaluated_candidates": evaluated}
        if best is None:
            fold["status"] = "no_train_candidate"
            folds.append(fold)
            continue

        test_selected_mask = best["mask"] & test_mask
        test_simulation = _simulate_ou_arrays(test_selected_mask, best["labels"], arrays)
        chosen = _trim_ou_candidate(best)
        chosen["train_simulation"] = chosen.pop("simulation")
        chosen["test_simulation"] = test_simulation
        fold["chosen_strategy"] = chosen
        folds.append(fold)

        all_test_mask |= test_selected_mask
        all_test_labels[test_selected_mask] = best["labels"][test_selected_mask]

    combined = _simulate_ou_arrays(all_test_mask, all_test_labels, arrays)
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "over_under_25_walk_forward_no_hindsight_selection",
        "source": "football-data.co.uk CSV Bet365 O/U 2.5 odds",
        "start_season": start_season,
        "end_season": end_season,
        "first_test_season": first_test_season,
        "source_matches": len(matches),
        "usable_matches": len(rows),
        "market": "over_under_2_5",
        "rules": {
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "stake": DEFAULT_SINGLE_STAKE,
            "min_train_bets": min_train_bets,
            "no_future_leakage": True,
            "no_hindsight_strategy_selection": True,
        },
        "evaluated_candidates": total_evaluated,
        "combined": combined,
        "folds": folds,
    }


def print_over_under_walk_forward_summary(results: Dict):
    print_header("OVER/UNDER 2.5 WALK-FORWARD")
    print(f"  Seasons: {results.get('start_season')}-{results.get('end_season')}")
    print(f"  First test season: {results.get('first_test_season')}")
    print(f"  Source matches: {results.get('source_matches', 0)}")
    print(f"  Usable matches with O/U odds: {results.get('usable_matches', 0)}")
    print(f"  Candidates evaluated: {results.get('evaluated_candidates', 0)}")

    sim = results.get("combined", {})
    print()
    print("  Out-of-sample singles")
    print(
        f"    Final: {sim.get('final_bankroll', 0):.0f} | "
        f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
        f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
    )
    print(f"    Bets: {sim.get('bets', 0)} | Hit: {sim.get('accuracy', 0):.1f}%")
    print(f"    By season: {sim.get('by_season', {})}")

    print()
    print("  Fold choices:")
    for fold in results.get("folds", []):
        chosen = fold.get("chosen_strategy")
        profit = chosen.get("test_simulation", {}).get("profit", 0.0) if chosen else 0.0
        source = chosen.get("source") if chosen else "N/A"
        print(f"    {fold.get('test_season')}: {profit:+.0f} {source}")
    print()


def optimize_h2h_coupon_criteria(
    matches: List[Dict],
    start_season: int,
    end_season: int,
    first_test_season: int = 2012,
    validation_start_season: int = 2022,
) -> Dict:
    """Train/validation sweep for mature H2H coupon tuning."""
    history_predictions = _csv_matches_to_history_predictions(matches)
    enriched = _enrich_strategy_zoo_history(history_predictions)
    arrays = _strategy_zoo_arrays(enriched)
    season_array = np.array([int(season) for season in arrays["season"]], dtype=int)
    train_mask = (season_array >= first_test_season) & (season_array < validation_start_season)
    validation_mask = (season_array >= validation_start_season) & (season_array <= end_season)
    league_groups_all = _csv_history_league_groups(enriched)
    league_groups = {
        key: value
        for key, value in league_groups_all.items()
        if key == "top_leagues" or key.startswith("league:")
    }
    sources = [
        src for src in _strategy_zoo_sources(arrays)
        if src["name"] in H2H_COUPON_SWEEP_SOURCES
    ]

    top_filters = []
    evaluated_filters = 0
    evaluated_coupons = 0

    for src in sources:
        logger.info(f"H2H coupon criteria source: {src['name']}")
        labels = src["labels"]
        valid_label = np.isin(labels, [LABEL_HOME, LABEL_DRAW, LABEL_AWAY])
        row_idx = np.arange(len(labels))

        for odds_basis in H2H_COUPON_SWEEP_ODDS_BASES:
            odds_matrix = _odds_matrix_for_basis(arrays, odds_basis)
            safe_labels = np.where(valid_label, labels, LABEL_HOME)
            selected_odds = odds_matrix[row_idx, safe_labels]
            selected_confidence = src["rate"]
            selected_edge = np.where(selected_odds > 1.0, selected_confidence - (1.0 / selected_odds), -99.0)
            form_signal = _recent_form_signal(safe_labels, arrays)
            full_form_mask = (arrays["home_form_count"] >= 5) & (arrays["away_form_count"] >= 5)
            valid = valid_label & (selected_odds > 1.0)

            for league_filter, leagues in league_groups.items():
                league_mask = np.isin(arrays["league"], leagues)
                for count_min in H2H_COUPON_SWEEP_COUNTS:
                    count_mask = src["count"] >= count_min
                    for rate_min in H2H_COUPON_SWEEP_RATES:
                        rate_mask = src["rate"] >= rate_min
                        for edge_min in H2H_COUPON_SWEEP_EDGES:
                            edge_mask = (
                                np.ones(len(enriched), dtype=bool)
                                if edge_min is None
                                else selected_edge >= edge_min
                            )
                            for odds_band in H2H_COUPON_SWEEP_ODDS_BANDS:
                                odds_mask = (selected_odds >= odds_band[0]) & (selected_odds <= odds_band[1])
                                for min_league_match in H2H_COUPON_SWEEP_MIN_LEAGUE_MATCHES:
                                    match_number_mask = arrays["league_season_match_number"] >= min_league_match
                                    for form_min in H2H_COUPON_SWEEP_FORM_THRESHOLDS:
                                        form_mask = (
                                            np.ones(len(enriched), dtype=bool)
                                            if form_min is None
                                            else full_form_mask & (form_signal >= form_min)
                                        )
                                        base_mask = (
                                            valid
                                            & src["pre_mask"]
                                            & league_mask
                                            & count_mask
                                            & rate_mask
                                            & edge_mask
                                            & odds_mask
                                            & match_number_mask
                                            & form_mask
                                        )
                                        filter_train_mask = base_mask & train_mask
                                        if int(np.sum(filter_train_mask)) < 80:
                                            continue

                                        simulation = _simulate_flat_arrays(
                                            filter_train_mask,
                                            labels,
                                            arrays,
                                            odds_matrix=odds_matrix,
                                        )
                                        score, eligible, reasons = _robust_score(simulation, "bets", 80)
                                        evaluated_filters += 1
                                        name = (
                                            f"h2h coupon filter strategy={src['name']} leagues={league_filter} "
                                            f"basis={odds_basis} count>={count_min} rate>={rate_min:.0%} "
                                            f"edge>={_threshold_label(edge_min)} odds={odds_band} "
                                            f"league_match>={min_league_match} "
                                            f"form>={_threshold_label(form_min)}"
                                        )
                                        candidate = {
                                            "type": "h2h_coupon_filter",
                                            "mode": "h2h_coupon_criteria",
                                            "version": "market",
                                            "name": name,
                                            "strategy": src["name"],
                                            "league_filter": league_filter,
                                            "leagues": leagues,
                                            "history_count_min": count_min,
                                            "rate_min_pct": round(rate_min * 100, 1),
                                            "model_edge_min_pct": None if edge_min is None else round(edge_min * 100, 1),
                                            "odds_band": odds_band,
                                            "odds_basis": odds_basis,
                                            "min_league_match_number": min_league_match,
                                            "recent_form_min_pct": None if form_min is None else round(form_min * 100, 1),
                                            "mask": base_mask,
                                            "labels": labels,
                                            "selected_confidence": selected_confidence,
                                            "selected_edge": selected_edge,
                                            "odds_matrix": odds_matrix,
                                            "arrays": arrays,
                                            "simulation": simulation,
                                            "score": score,
                                            "eligible": eligible,
                                            "rejection_reasons": reasons,
                                        }
                                        _remember_top_candidate(top_filters, candidate, H2H_COUPON_SWEEP_TOP_FILTERS)

    top_filters = sorted(top_filters, key=_candidate_rank_tuple, reverse=True)[:H2H_COUPON_SWEEP_TOP_FILTERS]
    coupon_candidates = []

    for filter_candidate in top_filters:
        for max_legs in [2, 3]:
            for sort_by in ["confidence", "edge", "edge_x_confidence"]:
                for max_per_league in [1, 2]:
                    for combined_odds_max in H2H_COUPON_SWEEP_COMBINED_ODDS_MAX:
                        train_sim = _simulate_coupon_candidate(
                            filter_candidate,
                            arrays,
                            filter_candidate["mask"] & train_mask,
                            max_legs=max_legs,
                            sort_by=sort_by,
                            max_per_league=max_per_league,
                            combined_odds_max=combined_odds_max,
                        )
                        if train_sim.get("coupons", 0) < 30:
                            continue
                        train_score, train_eligible, train_reasons = _robust_score(train_sim, "coupons", 30)
                        validation_sim = _simulate_coupon_candidate(
                            filter_candidate,
                            arrays,
                            filter_candidate["mask"] & validation_mask,
                            max_legs=max_legs,
                            sort_by=sort_by,
                            max_per_league=max_per_league,
                            combined_odds_max=combined_odds_max,
                        )
                        evaluated_coupons += 1
                        candidate = _strategy_zoo_trim(filter_candidate)
                        candidate.update({
                            "type": "h2h_coupon_criteria",
                            "name": (
                                f"{filter_candidate['name']} coupon max={max_legs} "
                                f"sort={sort_by} max_per_league={max_per_league} "
                                f"combined_odds_max={combined_odds_max or 'none'}"
                            ),
                            "odds_basis": filter_candidate["odds_basis"],
                            "min_league_match_number": filter_candidate["min_league_match_number"],
                            "max_legs": max_legs,
                            "sort_by": sort_by,
                            "max_per_league": max_per_league,
                            "combined_odds_max": combined_odds_max,
                            "train_simulation": train_sim,
                            "validation_simulation": validation_sim,
                            "score": train_score,
                            "eligible": train_eligible,
                            "rejection_reasons": train_reasons,
                        })
                        candidate.pop("simulation", None)
                        coupon_candidates.append(candidate)

    ranked_by_train = sorted(
        coupon_candidates,
        key=lambda c: (
            1 if c.get("eligible") else 0,
            c.get("score", -10**9),
            c.get("train_simulation", {}).get("profit", 0),
            c.get("validation_simulation", {}).get("profit", 0),
            -c.get("validation_simulation", {}).get("max_drawdown", 0),
        ),
        reverse=True,
    )
    ranked_by_validation = sorted(
        coupon_candidates,
        key=lambda c: (
            c.get("validation_simulation", {}).get("profit", 0),
            -c.get("validation_simulation", {}).get("max_drawdown", 0),
            c.get("train_simulation", {}).get("profit", 0),
        ),
        reverse=True,
    )
    robust_validation = [
        c for c in coupon_candidates
        if c.get("validation_simulation", {}).get("profit", 0) > 0
        and c.get("validation_simulation", {}).get("coupons", 0) >= 20
        and c.get("train_simulation", {}).get("profit", 0) > 0
    ]
    ranked_robust_validation = sorted(
        robust_validation,
        key=lambda c: (
            c.get("validation_simulation", {}).get("profit", 0),
            -c.get("validation_simulation", {}).get("max_drawdown", 0),
        ),
        reverse=True,
    )

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "h2h_coupon_criteria_train_validation_sweep",
        "source": "football-data.co.uk CSV",
        "start_season": start_season,
        "end_season": end_season,
        "first_test_season": first_test_season,
        "validation_start_season": validation_start_season,
        "source_matches": len(matches),
        "usable_matches": len(history_predictions),
        "rules": {
            "train_period": f"{first_test_season}-{validation_start_season - 1}",
            "validation_period": f"{validation_start_season}-{end_season}",
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "coupon_stake": DEFAULT_COUPON_STAKE,
            "criteria_tested": [
                "strategy_family", "league_filter", "odds_basis", "history_count",
                "history_rate", "historical_edge", "single_odds_band",
                "minimum_league_season_match_number", "coupon_legs",
                "recent_form_signal", "coupon_sort", "max_per_league",
                "combined_coupon_odds_cap",
            ],
        },
        "evaluated_filters": evaluated_filters,
        "evaluated_coupon_candidates": evaluated_coupons,
        "best_train_selected": ranked_by_train[0] if ranked_by_train else None,
        "best_validation": ranked_by_validation[0] if ranked_by_validation else None,
        "best_robust_validation": ranked_robust_validation[0] if ranked_robust_validation else None,
        "top_train_selected": ranked_by_train[:50],
        "top_validation": ranked_by_validation[:50],
        "top_robust_validation": ranked_robust_validation[:50],
    }


def print_h2h_coupon_criteria_summary(results: Dict):
    print_header("H2H COUPON CRITERIA SWEEP")
    print(f"  Seasons: {results.get('start_season')}-{results.get('end_season')}")
    print(f"  Train: {results.get('rules', {}).get('train_period')}")
    print(f"  Validation: {results.get('rules', {}).get('validation_period')}")
    print(f"  Source matches: {results.get('source_matches', 0)}")
    print(f"  Usable matches with 1X2 odds: {results.get('usable_matches', 0)}")
    print(f"  Filters evaluated: {results.get('evaluated_filters', 0)}")
    print(f"  Coupon candidates evaluated: {results.get('evaluated_coupon_candidates', 0)}")

    for title, key in [
        ("Best train-selected candidate", "best_train_selected"),
        ("Best validation candidate", "best_validation"),
        ("Best robust validation candidate", "best_robust_validation"),
    ]:
        candidate = results.get(key)
        if not candidate:
            print(f"  {title}: N/A")
            continue
        train = candidate.get("train_simulation", {})
        validation = candidate.get("validation_simulation", {})
        print()
        print(f"  {title}: {candidate.get('name')}")
        print(
            f"    Train: final {train.get('final_bankroll', 0):.0f} | "
            f"profit {train.get('profit', 0):+.0f} | coupons {train.get('coupons', 0)} | "
            f"hit {train.get('coupon_hit_rate', 0):.1f}% | DD {train.get('max_drawdown', 0):.0f}"
        )
        print(
            f"    Validation: final {validation.get('final_bankroll', 0):.0f} | "
            f"profit {validation.get('profit', 0):+.0f} | coupons {validation.get('coupons', 0)} | "
            f"hit {validation.get('coupon_hit_rate', 0):.1f}% | DD {validation.get('max_drawdown', 0):.0f}"
        )
    print()


def optimize_predictions(raw_preds: Dict) -> Dict:
    """Grid-search saved prediction JSON and rank robust single/coupon strategies."""
    by_mode = {}
    all_single = []
    all_coupons = []

    for mode, mode_data in raw_preds.items():
        mode_result = {"versions": {}}
        for version in ["v1", "v2"]:
            predictions = mode_data.get(version, [])
            if not predictions:
                continue
            logger.info(f"Optimizing {mode} {version}: {len(predictions)} predictions")
            logger.info("  Single-bet grid...")
            single = _rank_candidates(_optimize_single(mode, version, predictions))
            logger.info("  Coupon grid...")
            coupons = _rank_candidates(_optimize_coupons(mode, version, predictions))
            logger.info(f"  Done: {len(single)} single candidates, {len(coupons)} coupon candidates")
            mode_result["versions"][version] = {
                "top_single": [_trim_candidate(c) for c in single[:25]],
                "top_coupons": [_trim_candidate(c) for c in coupons[:25]],
                "single_candidates": len(single),
                "coupon_candidates": len(coupons),
                "failure_report": _build_failure_report(
                    predictions,
                    coupons[0] if coupons else None,
                ),
            }
            all_single.extend(single)
            all_coupons.extend(coupons)
        by_mode[mode] = mode_result

    ranked_single = _rank_candidates(all_single)
    ranked_coupons = _rank_candidates(all_coupons)
    best_single = ranked_single[0] if ranked_single else None
    best_coupon = ranked_coupons[0] if ranked_coupons else None
    model_decision = _build_model_decision(raw_preds)
    allowed_versions = {"v1", "v2"} if model_decision.get("promote_v2") else {"v1"}
    recommended_single = _first_candidate_for_versions(ranked_single, allowed_versions)
    recommended_coupon = _first_candidate_for_versions(ranked_coupons, allowed_versions)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "objective": "robust_bankroll_from_10000_with_100_flat_stake",
        "rules": {
            "min_single_bets": 100,
            "min_coupon_count": 50,
            "max_drawdown": MAX_ROBUST_DRAWDOWN,
            "required_profitable_seasons": "ceil(70% of evaluated seasons)",
            "min_season_sample": "at least 10 single bets or 5 coupons in active seasons",
            "score": "profit - 0.5 * max_drawdown - 1500 * negative_seasons - 0.25 * worst_season_loss",
        },
        "best_single": _trim_candidate(best_single) if best_single else None,
        "best_coupon": _trim_candidate(best_coupon) if best_coupon else None,
        "recommended_single": _trim_candidate(recommended_single) if recommended_single else None,
        "recommended_coupon": _trim_candidate(recommended_coupon) if recommended_coupon else None,
        "top_single": [_trim_candidate(c) for c in ranked_single[:50]],
        "top_coupons": [_trim_candidate(c) for c in ranked_coupons[:50]],
        "model_decision": model_decision,
        "recommendation": _recommend_config(recommended_single, recommended_coupon, model_decision),
        "by_mode": by_mode,
    }


def print_optimization_summary(optimization: Dict):
    """Print a compact terminal summary of optimizer winners."""
    print_header("STRATEGY OPTIMIZATION")
    best_single = optimization.get("best_single") or {}
    best_coupon = optimization.get("best_coupon") or {}
    recommended_single = optimization.get("recommended_single") or {}
    recommended_coupon = optimization.get("recommended_coupon") or {}

    def _print_winner(label: str, winner: Dict):
        sim = winner.get("simulation", {})
        print()
        print(f"  {label}: {winner.get('name', 'N/A')}")
        print(f"    Score: {winner.get('score', 0):+.0f} | Eligible: {winner.get('eligible')}")
        print(
            f"    Final: {sim.get('final_bankroll', 0):.0f} | "
            f"Profit: {sim.get('profit', 0):+.0f} | ROI: {sim.get('roi_pct', 0):+.1f}% | "
            f"MaxDD: {sim.get('max_drawdown', 0):.0f}"
        )
        if "bets" in sim:
            print(f"    Bets: {sim.get('bets', 0)} | Hit: {sim.get('accuracy', 0):.1f}%")
        if "coupons" in sim:
            print(f"    Coupons: {sim.get('coupons', 0)} | Coupon hit: {sim.get('coupon_hit_rate', 0):.1f}%")
        print(f"    By season: {sim.get('by_season', {})}")

    _print_winner("Best single", best_single)
    _print_winner("Best coupon", best_coupon)
    if recommended_single and recommended_single != best_single:
        _print_winner("Recommended live single", recommended_single)
    if recommended_coupon and recommended_coupon != best_coupon:
        _print_winner("Recommended live coupon", recommended_coupon)

    model_decision = optimization.get("model_decision", {})
    if model_decision:
        print()
        print(f"  v2 gate: promote={model_decision.get('promote_v2')} | {model_decision.get('reason')}")
        if model_decision.get("gate_overall"):
            print(f"    Overall: {model_decision.get('gate_overall')}")
    print()
    print("  Recommendation:")
    print(json.dumps(optimization.get("recommendation", {}), indent=2))
    print()


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


def print_bankroll_experiments(v1_exps: List[Dict], v2_exps: List[Dict]):
    """Print top single-bet strategies by fixed-bankroll growth."""
    print_header("BANKROLL SIMULATION (10,000 start, 100 flat stake)")

    def _top(exps: List[Dict]) -> List[Dict]:
        eligible = [e for e in exps if e.get("bankroll", {}).get("bets", 0) >= 20]
        if not eligible:
            eligible = exps
        return sorted(
            eligible,
            key=lambda e: (e.get("bankroll_final", 0), -e.get("bankroll_max_drawdown", 0), e.get("bets", 0)),
            reverse=True,
        )[:8]

    def _print_block(label: str, exps: List[Dict]):
        print()
        print(f"  {label}")
        print(f"  {'Strategy':<26} {'Bets':>6} {'Win%':>7} {'Final':>10} {'Profit':>10} {'ROI%':>8} {'MaxDD':>10}")
        print(f"  {'-'*26} {'-'*6} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
        for exp in _top(exps):
            bank = exp.get("bankroll", {})
            print(
                f"  {exp['name']:<26} {bank.get('bets', 0):>6} "
                f"{bank.get('accuracy', 0):>6.1f}% {bank.get('final_bankroll', 0):>10.0f} "
                f"{bank.get('profit', 0):>+10.0f} {bank.get('roi_pct', 0):>+7.1f}% "
                f"{bank.get('max_drawdown', 0):>10.0f}"
            )

    _print_block("v1 top single-bet strategies", v1_exps)
    _print_block("v2 top single-bet strategies", v2_exps)
    print()


def print_coupon_experiments(v1_coupons: List[Dict], v2_coupons: List[Dict]):
    """Print top accumulator/coupon strategies by fixed-bankroll growth."""
    print_header("COUPON SIMULATION (10,000 start, 100 per coupon)")

    def _top(exps: List[Dict]) -> List[Dict]:
        eligible = [e for e in exps if e.get("coupons", 0) >= 10]
        if not eligible:
            eligible = exps
        return sorted(
            eligible,
            key=lambda e: (e.get("final_bankroll", 0), -e.get("max_drawdown", 0), e.get("coupons", 0)),
            reverse=True,
        )[:8]

    def _print_block(label: str, exps: List[Dict]):
        print()
        print(f"  {label}")
        print(f"  {'Strategy':<26} {'Max':>3} {'Coupons':>7} {'Hit%':>7} {'Final':>10} {'Profit':>10} {'ROI%':>8} {'MaxDD':>10}")
        print(f"  {'-'*26} {'-'*3} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
        for exp in _top(exps):
            print(
                f"  {exp['name']:<26} {exp.get('max_legs', 0):>3} "
                f"{exp.get('coupons', 0):>7} {exp.get('coupon_hit_rate', 0):>6.1f}% "
                f"{exp.get('final_bankroll', 0):>10.0f} {exp.get('profit', 0):>+10.0f} "
                f"{exp.get('roi_pct', 0):>+7.1f}% {exp.get('max_drawdown', 0):>10.0f}"
            )

    _print_block("v1 top coupon strategies", v1_coupons)
    _print_block("v2 top coupon strategies", v2_coupons)
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
        X_v1, y_v1, _ = _build_training_data_v1_fast(train_matches)
        logger.info(f"v1 training data: X={X_v1.shape}, y={y_v1.shape}")
        v1_models = create_models(ML_SETTINGS, "_bt_v1", is_v2=False)
        v1_models, v1_ensemble, v1_stacking, v1_acc = train_models(v1_models, X_v1, y_v1, ML_SETTINGS)

        # ── Train v2 ──
        logger.info("\n--- Training v2 (83 features) ---")
        X_v2, y_v2, _ = _build_training_data_v2_fast(train_matches)
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
        logger.info(f"\n--- Predicting {len(test_matches)} test matches ---")
        team_idx = _build_team_index(train_matches)
        h2h_idx = _build_h2h_index(train_matches)
        test_sorted = sorted(test_matches, key=lambda m: m.get("match_date", ""))

        v1_rows = []
        v2_rows = []
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
            home_past = team_idx.get(home, [])
            away_past = team_idx.get(away, [])

            home_stats = _compute_team_stats_snapshot(home_past, home, league, season)
            away_stats = _compute_team_stats_snapshot(away_past, away, league, season)
            if home_stats and away_stats:
                home_stats["team_name"] = home
                away_stats["team_name"] = away
                h2h = h2h_idx.get((home, away), [])[-10:]

                hs = int(match["home_score"])
                aws = int(match["away_score"])
                actual_label = LABEL_HOME if hs > aws else (LABEL_DRAW if hs == aws else LABEL_AWAY)
                base_row = {
                    "match_date": match.get("match_date", ""),
                    "league": league,
                    "season": season,
                    "home": home,
                    "away": away,
                    "home_score": hs,
                    "away_score": aws,
                    "actual": actual_label,
                    "home_odds": match.get("home_odds"),
                    "draw_odds": match.get("draw_odds"),
                    "away_odds": match.get("away_odds"),
                }

                v1_features = FeatureEngineer.build_match_features(
                    home_stats, away_stats, h2h,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                    ai_predictions=None,
                )
                v1_row = dict(base_row)
                v1_row["features"] = np.nan_to_num(v1_features, nan=0.0, posinf=0.0, neginf=0.0)
                v1_rows.append(v1_row)

                home_form = FeatureEngineerV2.compute_form_list(home_past, home)
                away_form = FeatureEngineerV2.compute_form_list(away_past, away)
                home_extra = FeatureEngineerV2.compute_csv_extra_averages(home_past, home)
                away_extra = FeatureEngineerV2.compute_csv_extra_averages(away_past, away)
                match_date = match.get("match_date", "")
                h_exp, a_exp = poisson.predict_score(
                    home_stats.get("avg_goals_scored", 1.3),
                    home_stats.get("avg_goals_conceded", 1.1),
                    away_stats.get("avg_goals_scored", 1.2),
                    away_stats.get("avg_goals_conceded", 1.2),
                )
                poisson_probs = poisson.match_outcome_probs(h_exp, a_exp)
                total_exp = h_exp + a_exp
                v2_features = FeatureEngineerV2.build_match_features_v2(
                    home_stats, away_stats, h2h,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                    ai_predictions=None,
                    elo_tracker=elo_test,
                    home_form_list=home_form,
                    away_form_list=away_form,
                    home_extra=home_extra,
                    away_extra=away_extra,
                    home_days_rest=FeatureEngineerV2.compute_days_since_last(home_past, home, match_date),
                    away_days_rest=FeatureEngineerV2.compute_days_since_last(away_past, away, match_date),
                    home_recent_goals_avg=FeatureEngineerV2.compute_recent_goals_avg(home_past, home),
                    away_recent_goals_avg=FeatureEngineerV2.compute_recent_goals_avg(away_past, away),
                    is_training=True,
                    league_code=league,
                    matchday=match.get("matchday", 0) or 0,
                    total_matchdays=38,
                    match_datetime=match_date,
                    home_sos=FeatureEngineerV2.compute_sos(home_past, home, elo_test),
                    away_sos=FeatureEngineerV2.compute_sos(away_past, away, elo_test),
                )
                v2_row = dict(base_row)
                v2_row.update({
                    "features": np.nan_to_num(v2_features, nan=0.0, posinf=0.0, neginf=0.0),
                    "poisson_probs": poisson_probs,
                    "poisson_score": f"{h_exp}-{a_exp}",
                    "btts_prob": (1 - math.exp(-h_exp)) * (1 - math.exp(-a_exp)),
                    "over25_prob": 1.0 - sum(
                        (total_exp ** k) * math.exp(-total_exp) / math.factorial(k)
                        for k in range(3)
                    ),
                })
                v2_rows.append(v2_row)

            _record_finished_match(team_idx, h2h_idx, match)

            # Update ELO
            elo_test.update(home, away, int(match["home_score"]), int(match["away_score"]))

        logger.info(f"  Built feature rows: v1={len(v1_rows)}, v2={len(v2_rows)}")
        v1_preds = _predict_rows_batch(v1_rows, v1_models, ML_SETTINGS, v1_stacking, "v1")
        v2_preds = _predict_v2_rows_batch(v2_rows, v2_models, ML_SETTINGS_V2, v2_stacking)

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

            # Train v1 from rolling in-memory history to avoid DB feature-builder bottlenecks.
            X_v1, y_v1, _ = _build_training_data_v1_fast(v1_train_matches)
            if len(X_v1) < 100:
                logger.warning(f"  v1: only {len(X_v1)} samples, skipping")
                continue
            v1_models = create_models(ML_SETTINGS, f"_wf_v1_{test_season}", is_v2=False)
            v1_models, v1_ens, v1_stack, _ = train_models(v1_models, X_v1, y_v1, ML_SETTINGS)

            # Train v2 from the same leakage-safe in-memory feature path as holdout.
            X_v2, y_v2, _ = _build_training_data_v2_fast(v2_train_matches)
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

            all_train = v2_train_matches
            team_idx = _build_team_index(all_train)
            h2h_idx = _build_h2h_index(all_train)
            test_sorted = sorted(test_matches, key=lambda m: m.get("match_date", ""))

            v1_rows = []
            v2_rows = []
            for match in test_sorted:
                home = match["home_team_name"]
                away = match["away_team_name"]
                league = match.get("league_code", "")
                season_m = match.get("season", 2025)
                home_past = team_idx.get(home, [])
                away_past = team_idx.get(away, [])

                home_stats = _compute_team_stats_snapshot(home_past, home, league, season_m)
                away_stats = _compute_team_stats_snapshot(away_past, away, league, season_m)
                if home_stats and away_stats:
                    home_stats["team_name"] = home
                    away_stats["team_name"] = away
                    h2h = h2h_idx.get((home, away), [])[-10:]

                    hs = int(match["home_score"])
                    aws = int(match["away_score"])
                    actual_label = LABEL_HOME if hs > aws else (LABEL_DRAW if hs == aws else LABEL_AWAY)
                    base_row = {
                        "match_date": match.get("match_date", ""),
                        "league": league,
                        "season": season_m,
                        "home": home,
                        "away": away,
                        "home_score": hs,
                        "away_score": aws,
                        "actual": actual_label,
                        "home_odds": match.get("home_odds"),
                        "draw_odds": match.get("draw_odds"),
                        "away_odds": match.get("away_odds"),
                        "fold_season": test_season,
                    }

                    v1_features = FeatureEngineer.build_match_features(
                        home_stats, away_stats, h2h,
                        match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                        ai_predictions=None,
                    )
                    v1_row = dict(base_row)
                    v1_row["features"] = np.nan_to_num(v1_features, nan=0.0, posinf=0.0, neginf=0.0)
                    v1_rows.append(v1_row)

                    match_date = match.get("match_date", "")
                    h_exp, a_exp = poisson.predict_score(
                        home_stats.get("avg_goals_scored", 1.3),
                        home_stats.get("avg_goals_conceded", 1.1),
                        away_stats.get("avg_goals_scored", 1.2),
                        away_stats.get("avg_goals_conceded", 1.2),
                    )
                    poisson_probs = poisson.match_outcome_probs(h_exp, a_exp)
                    total_exp = h_exp + a_exp
                    v2_features = FeatureEngineerV2.build_match_features_v2(
                        home_stats, away_stats, h2h,
                        match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                        ai_predictions=None,
                        elo_tracker=elo_test,
                        home_form_list=FeatureEngineerV2.compute_form_list(home_past, home),
                        away_form_list=FeatureEngineerV2.compute_form_list(away_past, away),
                        home_extra=FeatureEngineerV2.compute_csv_extra_averages(home_past, home),
                        away_extra=FeatureEngineerV2.compute_csv_extra_averages(away_past, away),
                        home_days_rest=FeatureEngineerV2.compute_days_since_last(home_past, home, match_date),
                        away_days_rest=FeatureEngineerV2.compute_days_since_last(away_past, away, match_date),
                        home_recent_goals_avg=FeatureEngineerV2.compute_recent_goals_avg(home_past, home),
                        away_recent_goals_avg=FeatureEngineerV2.compute_recent_goals_avg(away_past, away),
                        is_training=True,
                        league_code=league,
                        matchday=match.get("matchday", 0) or 0,
                        total_matchdays=38,
                        match_datetime=match_date,
                        home_sos=FeatureEngineerV2.compute_sos(home_past, home, elo_test),
                        away_sos=FeatureEngineerV2.compute_sos(away_past, away, elo_test),
                    )
                    v2_row = dict(base_row)
                    v2_row.update({
                        "features": np.nan_to_num(v2_features, nan=0.0, posinf=0.0, neginf=0.0),
                        "poisson_probs": poisson_probs,
                        "poisson_score": f"{h_exp}-{a_exp}",
                        "btts_prob": (1 - math.exp(-h_exp)) * (1 - math.exp(-a_exp)),
                        "over25_prob": 1.0 - sum(
                            (total_exp ** k) * math.exp(-total_exp) / math.factorial(k)
                            for k in range(3)
                        ),
                    })
                    v2_rows.append(v2_row)

                _record_finished_match(team_idx, h2h_idx, match)

                elo_test.update(home, away, int(match["home_score"]), int(match["away_score"]))

            v1_fold_preds = _predict_rows_batch(v1_rows, v1_models, ML_SETTINGS, v1_stack, "v1")
            v2_fold_preds = _predict_v2_rows_batch(v2_rows, v2_models, ML_SETTINGS_V2, v2_stack)
            v1_all_preds.extend(v1_fold_preds)
            v2_all_preds.extend(v2_fold_preds)

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
    print_bankroll_experiments(v1_exps, v2_exps)

    # Coupon/accumulator experiments grouped by match day.
    v1_coupons = compute_coupon_experiments(v1_preds)
    v2_coupons = compute_coupon_experiments(v2_preds)
    print_coupon_experiments(v1_coupons, v2_coupons)

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
        "v1_coupon_experiments": v1_coupons,
        "v2_coupon_experiments": v2_coupons,
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
    parser.add_argument("--optimize", type=str, default=None,
                        help="Optimize single-bet and coupon strategies from saved predictions JSON")
    parser.add_argument("--history-patterns", type=str, default=None,
                        help="Backtest historical H2H/team-pattern strategies from saved predictions JSON")
    parser.add_argument("--history-edge", type=str, default=None,
                        help="Money-first H2H historical edge backtest from saved predictions JSON")
    parser.add_argument("--history-edge-csv", action="store_true",
                        help="Pure CSV historical H2H/odds edge backtest, no ML predictions required")
    parser.add_argument("--strategy-zoo-csv", action="store_true",
                        help="Backtest many CSV-only money strategies against each other")
    parser.add_argument("--strategy-zoo-walk-forward-csv", action="store_true",
                        help="Walk-forward validate CSV strategy-zoo selection without hindsight")
    parser.add_argument("--over-under25-walk-forward-csv", action="store_true",
                        help="Walk-forward validate Over/Under 2.5 money strategies")
    parser.add_argument("--h2h-coupon-criteria-csv", action="store_true",
                        help="Train/validation sweep for mature H2H coupon criteria")
    parser.add_argument("--start-season", type=int, default=None,
                        help="First season for CSV history modes, e.g. 2000 for 2000/01")
    parser.add_argument("--end-season", type=int, default=None,
                        help="Last season for CSV history modes, e.g. 2025 for 2025/26")
    parser.add_argument("--first-test-season", type=int, default=None,
                        help="First season to test in CSV walk-forward strategy modes")
    parser.add_argument("--min-train-bets", type=int, default=STRATEGY_ZOO_WF_MIN_TRAIN_BETS,
                        help="Minimum training bets required before a walk-forward strategy can be selected")
    parser.add_argument("--validation-start-season", type=int, default=2022,
                        help="First validation season for H2H coupon criteria sweeps")
    args = parser.parse_args()

    # ── Over/Under 2.5 walk-forward mode: money-first goal-market strategy search ──
    if args.over_under25_walk_forward_csv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        start_season = args.start_season if args.start_season is not None else 2000
        end_season = args.end_season if args.end_season is not None else max(ALL_SEASONS)
        if start_season > end_season:
            print(f"Invalid season range: {start_season}>{end_season}")
            return
        leagues = args.leagues.split(",") if args.leagues else CSV_LEAGUES
        seasons = list(range(start_season, end_season + 1))
        matches = load_all_csv_data(leagues, seasons)
        if not matches:
            print("No CSV matches loaded; cannot run Over/Under 2.5 walk-forward")
            return

        ou_results = walk_forward_over_under_25(
            matches,
            start_season,
            end_season,
            first_test_season=args.first_test_season,
            min_train_bets=args.min_train_bets,
        )
        print_over_under_walk_forward_summary(ou_results)

        suffix = f"{start_season}_{end_season}_first{ou_results.get('first_test_season')}_min{args.min_train_bets}"
        ou_path = ROOT / "data" / f"over_under25_walk_forward_csv_{suffix}.json"
        with open(ou_path, "w") as f:
            json.dump(ou_results, f, indent=2, default=str)
        print(f"Over/Under 2.5 results saved to {ou_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results[f"over_under25_walk_forward_csv_{suffix}"] = ou_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── H2H coupon criteria mode: tune knobs with train/validation split ──
    if args.h2h_coupon_criteria_csv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        start_season = args.start_season if args.start_season is not None else 2000
        end_season = args.end_season if args.end_season is not None else max(ALL_SEASONS)
        first_test_season = args.first_test_season if args.first_test_season is not None else 2012
        if start_season > end_season:
            print(f"Invalid season range: {start_season}>{end_season}")
            return
        leagues = args.leagues.split(",") if args.leagues else CSV_LEAGUES
        seasons = list(range(start_season, end_season + 1))
        matches = load_all_csv_data(leagues, seasons)
        if not matches:
            print("No CSV matches loaded; cannot run H2H coupon criteria sweep")
            return

        criteria_results = optimize_h2h_coupon_criteria(
            matches,
            start_season,
            end_season,
            first_test_season=first_test_season,
            validation_start_season=args.validation_start_season,
        )
        print_h2h_coupon_criteria_summary(criteria_results)

        suffix = f"{start_season}_{end_season}_train{first_test_season}_{args.validation_start_season - 1}_val{args.validation_start_season}_{end_season}"
        criteria_path = ROOT / "data" / f"h2h_coupon_criteria_csv_{suffix}.json"
        with open(criteria_path, "w") as f:
            json.dump(criteria_results, f, indent=2, default=str)
        print(f"H2H coupon criteria results saved to {criteria_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results[f"h2h_coupon_criteria_csv_{suffix}"] = criteria_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Strategy zoo walk-forward mode: choose from history, test next season ──
    if args.strategy_zoo_walk_forward_csv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        start_season = args.start_season if args.start_season is not None else 2000
        end_season = args.end_season if args.end_season is not None else max(ALL_SEASONS)
        if start_season > end_season:
            print(f"Invalid season range: {start_season}>{end_season}")
            return
        leagues = args.leagues.split(",") if args.leagues else CSV_LEAGUES
        seasons = list(range(start_season, end_season + 1))
        matches = load_all_csv_data(leagues, seasons)
        if not matches:
            print("No CSV matches loaded; cannot run strategy zoo walk-forward")
            return

        wf_results = walk_forward_csv_strategy_zoo(
            matches,
            start_season,
            end_season,
            first_test_season=args.first_test_season,
            min_train_bets=args.min_train_bets,
        )
        print_strategy_zoo_walk_forward_summary(wf_results)

        suffix = f"{start_season}_{end_season}_first{wf_results.get('first_test_season')}_min{args.min_train_bets}"
        wf_path = ROOT / "data" / f"strategy_zoo_walk_forward_csv_{suffix}.json"
        with open(wf_path, "w") as f:
            json.dump(wf_results, f, indent=2, default=str)
        print(f"Strategy zoo walk-forward results saved to {wf_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results[f"strategy_zoo_walk_forward_csv_{suffix}"] = wf_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Strategy zoo CSV mode: many market/history/form strategy families ──
    if args.strategy_zoo_csv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        start_season = args.start_season if args.start_season is not None else 2000
        end_season = args.end_season if args.end_season is not None else max(ALL_SEASONS)
        if start_season > end_season:
            print(f"Invalid season range: {start_season}>{end_season}")
            return
        leagues = args.leagues.split(",") if args.leagues else CSV_LEAGUES
        seasons = list(range(start_season, end_season + 1))
        matches = load_all_csv_data(leagues, seasons)
        if not matches:
            print("No CSV matches loaded; cannot run strategy zoo backtest")
            return

        zoo_results = optimize_csv_strategy_zoo(matches, start_season, end_season)
        print_strategy_zoo_summary(zoo_results)

        suffix = f"{start_season}_{end_season}"
        zoo_path = ROOT / "data" / f"strategy_zoo_csv_{suffix}.json"
        with open(zoo_path, "w") as f:
            json.dump(zoo_results, f, indent=2, default=str)
        print(f"Strategy zoo results saved to {zoo_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results[f"strategy_zoo_csv_{suffix}"] = zoo_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Pure CSV historical edge mode: can go back to 2000/01 odds data ──
    if args.history_edge_csv:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        start_season = args.start_season if args.start_season is not None else 2000
        end_season = args.end_season if args.end_season is not None else max(ALL_SEASONS)
        if start_season > end_season:
            print(f"Invalid season range: {start_season}>{end_season}")
            return
        leagues = args.leagues.split(",") if args.leagues else CSV_LEAGUES
        seasons = list(range(start_season, end_season + 1))
        matches = load_all_csv_data(leagues, seasons)
        if not matches:
            print("No CSV matches loaded; cannot run historical edge CSV backtest")
            return

        csv_edge_results = optimize_csv_historical_edge(matches, start_season, end_season)
        print_csv_historical_edge_summary(csv_edge_results)

        suffix = f"{start_season}_{end_season}"
        edge_path = ROOT / "data" / f"historical_edge_csv_{suffix}.json"
        with open(edge_path, "w") as f:
            json.dump(csv_edge_results, f, indent=2, default=str)
        print(f"CSV historical edge results saved to {edge_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results[f"historical_edge_csv_{suffix}"] = csv_edge_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Money-first historical edge mode: no ML training, no future leakage ──
    if args.history_edge:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        pred_path = Path(args.history_edge)
        if not pred_path.exists():
            print(f"File not found: {pred_path}")
            return
        with open(pred_path) as f:
            raw_preds = json.load(f)

        edge_results = optimize_historical_edge(raw_preds)
        print_historical_edge_summary(edge_results)

        edge_path = ROOT / "data" / "historical_edge_results.json"
        with open(edge_path, "w") as f:
            json.dump(edge_results, f, indent=2, default=str)
        print(f"Historical edge results saved to {edge_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results["historical_edge"] = edge_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Historical-pattern mode: no ML, no future leakage ──
    if args.history_patterns:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        pred_path = Path(args.history_patterns)
        if not pred_path.exists():
            print(f"File not found: {pred_path}")
            return
        with open(pred_path) as f:
            raw_preds = json.load(f)

        matches = _load_pattern_matches(raw_preds)
        if not matches:
            print(f"No usable match list found in {pred_path}")
            return

        pattern_results = optimize_historical_patterns(matches)
        print_historical_pattern_summary(pattern_results)

        pattern_path = ROOT / "data" / "historical_pattern_results.json"
        with open(pattern_path, "w") as f:
            json.dump(pattern_results, f, indent=2, default=str)
        print(f"Historical pattern results saved to {pattern_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results["historical_patterns"] = pattern_results
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Optimize-only mode: grid-search strategies on saved predictions ──
    if args.optimize:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        pred_path = Path(args.optimize)
        if not pred_path.exists():
            print(f"File not found: {pred_path}")
            return
        with open(pred_path) as f:
            raw_preds = json.load(f)

        optimization = optimize_predictions(raw_preds)
        print_optimization_summary(optimization)

        opt_path = ROOT / "data" / "strategy_optimization.json"
        with open(opt_path, "w") as f:
            json.dump(optimization, f, indent=2, default=str)
        print(f"Optimization saved to {opt_path}")

        results_path = ROOT / "data" / "backtest_results.json"
        existing_results = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing_results = json.load(f)
            except Exception:
                existing_results = {}
        existing_results["strategy_optimization"] = optimization
        with open(results_path, "w") as f:
            json.dump(existing_results, f, indent=2, default=str)
        print(f"Backtest results updated at {results_path}")
        return

    # ── Analyze-only mode: re-run experiments on saved predictions ──
    if args.analyze:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        pred_path = Path(args.analyze)
        if not pred_path.exists():
            print(f"File not found: {pred_path}")
            return
        with open(pred_path) as f:
            raw_preds = json.load(f)
        analysis_results = {}
        for mode_name, mode_data in raw_preds.items():
            v1_preds = mode_data.get("v1", [])
            v2_preds = mode_data.get("v2", [])
            if v1_preds or v2_preds:
                analysis_results[mode_name] = generate_report(v1_preds, v2_preds, f"Re-analysis: {mode_name}")
        if analysis_results:
            output_path = ROOT / "data" / "backtest_results.json"
            with open(output_path, "w") as f:
                json.dump(analysis_results, f, indent=2, default=str)
            print(f"Re-analysis saved to {output_path}")
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
