"""
Prediction Engine - Orchestrates ML predictions for soccer matches.
Trains models, generates predictions, and compares results.
"""
import logging
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import threading

from .models import (
    XGBoostModel, NeuralNetworkModel, RandomForestModel,
    EnsembleModel, PoissonModel, HAS_SKLEARN
)
from .feature_engineering import FeatureEngineer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import LEAGUES

logger = logging.getLogger(__name__)


class PredictionEngine:
    """Main prediction engine that manages ML models and generates predictions."""

    OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    def __init__(self, db_manager, data_aggregator=None):
        self.db = db_manager
        self.data_aggregator = data_aggregator
        self.feature_engineer = FeatureEngineer()
        self.poisson = PoissonModel()

        # Initialize models
        self.xgboost = XGBoostModel()
        self.neural_net = NeuralNetworkModel()
        self.random_forest = RandomForestModel()
        self.ensemble = EnsembleModel({
            "xgboost": self.xgboost,
            "neural_network": self.neural_net,
            "random_forest": self.random_forest,
        })

        self.models = {
            "xgboost": self.xgboost,
            "neural_network": self.neural_net,
            "random_forest": self.random_forest,
        }

        self._training = False
        self._trained = False

        # Try to load pre-trained models
        self._load_models()

    def _load_models(self):
        """Load pre-trained models from disk."""
        any_loaded = False
        for name, model in self.models.items():
            if model.load():
                any_loaded = True
                logger.info(f"Loaded pre-trained model: {name}")
        self._trained = any_loaded

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def is_training(self) -> bool:
        return self._training

    # ──────────────────────────────────────────
    # TRAINING
    # ──────────────────────────────────────────
    def train_models(self, league_codes: List[str] = None,
                     callback=None) -> Dict[str, float]:
        """
        Train all ML models on historical data.

        Args:
            league_codes: List of league codes to train on
            callback: Optional callback function for progress updates

        Returns:
            Dict of model_name -> accuracy
        """
        if self._training:
            logger.warning("Training already in progress")
            return {}

        self._training = True
        results = {}

        try:
            if callback:
                callback("status", "Collecting training data...")

            # Collect historical data
            all_matches = []
            leagues = league_codes or ["PL", "PD", "BL1", "SA", "FL1"]

            for i, league_code in enumerate(leagues):
                if callback:
                    callback("progress", f"Loading {LEAGUES.get(league_code, {}).get('name', league_code)}... ({i+1}/{len(leagues)})")

                if self.data_aggregator:
                    matches = self.data_aggregator.fetch_historical_matches(league_code, 2025)
                    all_matches.extend(matches)

                    matches_prev = self.data_aggregator.fetch_historical_matches(league_code, 2024)
                    all_matches.extend(matches_prev)

            if callback:
                callback("status", f"Building features from {len(all_matches)} matches...")

            # Compute team stats first
            for match in all_matches:
                if match.get("status") == "FINISHED" and match.get("home_score") is not None:
                    for team_name in [match["home_team_name"], match["away_team_name"]]:
                        stats = self.db.compute_team_stats_from_matches(
                            team_name, match.get("league_code", ""),
                            match.get("season", 2025)
                        )
                        if stats.get("matches_played", 0) >= 3:
                            self.db.upsert_team_stats(stats)

            # Build training data
            X, y = self.feature_engineer.build_training_data(all_matches, self.db)

            if len(X) < 50:
                logger.warning(f"Not enough training data: {len(X)} samples")
                if callback:
                    callback("error", f"Not enough training data ({len(X)} matches). Need at least 50.")
                self._training = False
                return {}

            if callback:
                callback("status", f"Training on {len(X)} matches...")

            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            dist = {self.OUTCOME_LABELS.get(int(u), str(u)): int(c) for u, c in zip(unique, counts)}
            logger.info(f"Training data: {len(X)} samples, distribution: {dist}")

            # Train each model
            for name, model in self.models.items():
                if callback:
                    callback("progress", f"Training {name}...")
                try:
                    accuracy = model.train(X, y)
                    results[name] = accuracy
                    logger.info(f"{name}: accuracy = {accuracy:.4f}")

                    # Store performance
                    total = len(X)
                    correct = int(accuracy * total)
                    self.db.update_model_performance(name, total, correct, accuracy)

                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    results[name] = 0.0

            # Ensemble doesn't train separately
            results["ensemble"] = np.mean([v for v in results.values() if v > 0])

            self._trained = True

            if callback:
                callback("done", results)

        except Exception as e:
            logger.error(f"Training error: {e}")
            if callback:
                callback("error", str(e))
        finally:
            self._training = False

        return results

    def train_models_async(self, league_codes: List[str] = None, callback=None):
        """Train models in a background thread."""
        thread = threading.Thread(
            target=self.train_models,
            args=(league_codes, callback),
            daemon=True,
        )
        thread.start()
        return thread

    # ──────────────────────────────────────────
    # PREDICTION
    # ──────────────────────────────────────────
    def predict_match(self, match: Dict) -> List[Dict]:
        """
        Generate predictions for a single match using all models.

        Returns list of prediction dicts (one per model + ensemble + poisson).
        """
        home_name = match["home_team_name"]
        away_name = match["away_team_name"]
        league_code = match.get("league_code", "")
        season = match.get("season", 2025)

        # Get team stats
        home_stats = self.db.get_team_stats(home_name, league_code, season)
        away_stats = self.db.get_team_stats(away_name, league_code, season)

        if not home_stats:
            home_stats = self.db.compute_team_stats_from_matches(home_name, league_code, season)
        if not away_stats:
            away_stats = self.db.compute_team_stats_from_matches(away_name, league_code, season)

        # Add team_name to stats if missing
        home_stats["team_name"] = home_name
        away_stats["team_name"] = away_name

        h2h = self.db.get_h2h(home_name, away_name)

        # Build features
        features = self.feature_engineer.build_match_features(
            home_stats, away_stats, h2h,
            match.get("home_odds"), match.get("draw_odds"), match.get("away_odds")
        )

        predictions = []

        # ML model predictions
        all_models = list(self.models.items()) + [("ensemble", self.ensemble)]
        for model_name, model in all_models:
            try:
                probs = model.predict_proba(features)
                if probs.ndim > 1:
                    probs = probs[0]

                home_prob = float(probs[0])
                draw_prob = float(probs[1])
                away_prob = float(probs[2])

                # Determine predicted outcome
                outcome_idx = int(np.argmax(probs))
                predicted_outcome = self.OUTCOME_LABELS[outcome_idx]
                confidence = float(probs[outcome_idx])

                # Value rating (compare with odds)
                value_rating = self._calculate_value(
                    home_prob, draw_prob, away_prob,
                    match.get("home_odds"), match.get("draw_odds"), match.get("away_odds")
                )

                prediction = {
                    "match_id": match.get("id") or match.get("api_id"),
                    "match_date": match.get("match_date"),
                    "home_team": home_name,
                    "away_team": away_name,
                    "league_code": league_code,
                    "model_name": model_name,
                    "home_win_prob": round(home_prob, 4),
                    "draw_prob": round(draw_prob, 4),
                    "away_win_prob": round(away_prob, 4),
                    "predicted_outcome": predicted_outcome,
                    "confidence": round(confidence, 4),
                    "value_rating": round(value_rating, 2),
                    "suggestion": self._generate_suggestion(
                        home_prob, draw_prob, away_prob,
                        match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"),
                        home_name, away_name, confidence
                    ),
                }
                predictions.append(prediction)

            except Exception as e:
                logger.error(f"Prediction error ({model_name}): {e}")

        # Poisson model prediction
        try:
            home_attack = home_stats.get("avg_goals_scored", 1.3)
            home_defense = home_stats.get("avg_goals_conceded", 1.1)
            away_attack = away_stats.get("avg_goals_scored", 1.2)
            away_defense = away_stats.get("avg_goals_conceded", 1.2)

            h_goals, a_goals = self.poisson.predict_score(
                home_attack, home_defense, away_attack, away_defense
            )
            poisson_probs = self.poisson.match_outcome_probs(h_goals, a_goals)

            outcome_map = {"home_win": "Home Win", "draw": "Draw", "away_win": "Away Win"}
            best = max(poisson_probs, key=poisson_probs.get)

            predictions.append({
                "match_id": match.get("id") or match.get("api_id"),
                "match_date": match.get("match_date"),
                "home_team": home_name,
                "away_team": away_name,
                "league_code": league_code,
                "model_name": "poisson",
                "home_win_prob": poisson_probs["home_win"],
                "draw_prob": poisson_probs["draw"],
                "away_win_prob": poisson_probs["away_win"],
                "predicted_home_goals": h_goals,
                "predicted_away_goals": a_goals,
                "predicted_outcome": outcome_map[best],
                "confidence": round(poisson_probs[best], 4),
                "value_rating": 0.0,
                "suggestion": f"Predicted score: {home_name} {h_goals} - {a_goals} {away_name}",
            })

        except Exception as e:
            logger.error(f"Poisson prediction error: {e}")

        # Save predictions to database
        for pred in predictions:
            try:
                self.db.save_prediction(pred)
            except Exception:
                pass

        return predictions

    def predict_all_matches(self, matches: List[Dict]) -> Dict[str, List[Dict]]:
        """Generate predictions for all provided matches."""
        all_predictions = {}
        for match in matches:
            key = f"{match['home_team_name']} vs {match['away_team_name']}"
            preds = self.predict_match(match)
            all_predictions[key] = preds
        return all_predictions

    # ──────────────────────────────────────────
    # VALUE & SUGGESTIONS
    # ──────────────────────────────────────────
    def _calculate_value(self, home_prob: float, draw_prob: float, away_prob: float,
                         home_odds: float = None, draw_odds: float = None,
                         away_odds: float = None) -> float:
        """Calculate value rating by comparing predicted vs implied probabilities."""
        if not any([home_odds, draw_odds, away_odds]):
            return 0.0

        value_scores = []

        if home_odds and home_odds > 0:
            implied = 1 / home_odds
            edge = home_prob - implied
            if edge > 0:
                value_scores.append(edge * home_odds)

        if draw_odds and draw_odds > 0:
            implied = 1 / draw_odds
            edge = draw_prob - implied
            if edge > 0:
                value_scores.append(edge * draw_odds)

        if away_odds and away_odds > 0:
            implied = 1 / away_odds
            edge = away_prob - implied
            if edge > 0:
                value_scores.append(edge * away_odds)

        return max(value_scores) if value_scores else 0.0

    def _generate_suggestion(self, home_prob: float, draw_prob: float, away_prob: float,
                              home_odds: float = None, draw_odds: float = None,
                              away_odds: float = None,
                              home_name: str = "", away_name: str = "",
                              confidence: float = 0.0) -> str:
        """Generate a human-readable betting suggestion."""
        suggestions = []

        # Find best value bet
        best_outcome = ""
        best_value = 0

        if home_odds and home_odds > 0:
            ev_home = home_prob * home_odds
            if ev_home > 1.0 and home_prob > 0.4:
                suggestions.append(f"💚 VALUE: {home_name} to win @ {home_odds:.2f} (EV: {ev_home:.2f})")
                if ev_home > best_value:
                    best_value = ev_home
                    best_outcome = "home"

        if draw_odds and draw_odds > 0:
            ev_draw = draw_prob * draw_odds
            if ev_draw > 1.1 and draw_prob > 0.28:
                suggestions.append(f"🟡 VALUE: Draw @ {draw_odds:.2f} (EV: {ev_draw:.2f})")
                if ev_draw > best_value:
                    best_value = ev_draw
                    best_outcome = "draw"

        if away_odds and away_odds > 0:
            ev_away = away_prob * away_odds
            if ev_away > 1.0 and away_prob > 0.35:
                suggestions.append(f"💚 VALUE: {away_name} to win @ {away_odds:.2f} (EV: {ev_away:.2f})")
                if ev_away > best_value:
                    best_value = ev_away
                    best_outcome = "away"

        if confidence > 0.6:
            outcome_idx = int(np.argmax([home_prob, draw_prob, away_prob]))
            outcome_names = [f"{home_name} Win", "Draw", f"{away_name} Win"]
            suggestions.insert(0, f"🎯 HIGH CONFIDENCE: {outcome_names[outcome_idx]} ({confidence:.0%})")

        if not suggestions:
            probs = [home_prob, draw_prob, away_prob]
            outcome_idx = int(np.argmax(probs))
            outcome_names = [f"{home_name}", "Draw", f"{away_name}"]
            suggestions.append(f"📊 Lean: {outcome_names[outcome_idx]} ({probs[outcome_idx]:.0%})")

        return " | ".join(suggestions[:3])

    # ──────────────────────────────────────────
    # MODEL COMPARISON
    # ──────────────────────────────────────────
    def get_model_comparison(self) -> List[Dict]:
        """Get comparison of all model performances."""
        performances = self.db.get_all_model_performance()
        if not performances:
            # Return model info even if not in DB yet
            return [
                {
                    "model_name": name,
                    "accuracy": model.accuracy if hasattr(model, 'accuracy') else 0.0,
                    "is_trained": model.is_trained if hasattr(model, 'is_trained') else False,
                    "total_predictions": 0,
                    "correct_predictions": 0,
                }
                for name, model in self.models.items()
            ]
        return performances

    def get_consensus_prediction(self, match: Dict) -> Dict:
        """Get consensus prediction where most models agree."""
        predictions = self.predict_match(match)
        if not predictions:
            return {}

        outcomes = [p["predicted_outcome"] for p in predictions]
        from collections import Counter
        most_common = Counter(outcomes).most_common(1)[0]

        avg_confidence = np.mean([p["confidence"] for p in predictions])
        avg_home = np.mean([p["home_win_prob"] for p in predictions])
        avg_draw = np.mean([p["draw_prob"] for p in predictions])
        avg_away = np.mean([p["away_win_prob"] for p in predictions])

        return {
            "consensus_outcome": most_common[0],
            "agreement": most_common[1] / len(predictions),
            "avg_confidence": round(float(avg_confidence), 4),
            "avg_home_prob": round(float(avg_home), 4),
            "avg_draw_prob": round(float(avg_draw), 4),
            "avg_away_prob": round(float(avg_away), 4),
            "models_agree": most_common[1],
            "models_total": len(predictions),
            "all_predictions": predictions,
        }
