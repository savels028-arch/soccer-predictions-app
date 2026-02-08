"""
ML Models for Soccer Match Prediction
Includes: XGBoost, Neural Network, Random Forest, and Ensemble.
"""
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed. Using fallback models.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, OSError, Exception):
    HAS_XGBOOST = False
    logger.info("XGBoost not available. Will use sklearn GradientBoosting alternative.")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.info("TensorFlow not installed. Neural network model unavailable.")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import ML_SETTINGS, MODELS_DIR


class BaseModel:
    """Base class for prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.accuracy = 0.0
        self.model_path = MODELS_DIR / f"{name}_model.pkl"
        self.scaler_path = MODELS_DIR / f"{name}_scaler.pkl"

    def save(self):
        """Save model and scaler to disk."""
        if self.model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            logger.info(f"Model {self.name} saved to {self.model_path}")

    def load(self) -> bool:
        """Load model and scaler from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                if self.scaler_path.exists():
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info(f"Model {self.name} loaded from disk")
                return True
            except Exception as e:
                logger.error(f"Error loading model {self.name}: {e}")
        return False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities. Returns [home_win, draw, away_win]."""
        raise NotImplementedError


class XGBoostModel(BaseModel):
    """XGBoost classifier for match prediction."""

    def __init__(self):
        super().__init__("xgboost")

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        if not HAS_XGBOOST and not HAS_SKLEARN:
            logger.error("No ML library available")
            return 0.0

        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=ML_SETTINGS["test_size"],
            random_state=ML_SETTINGS["random_state"], stratify=y
        )

        xgb_params = ML_SETTINGS["xgboost"]

        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                subsample=xgb_params["subsample"],
                colsample_bytree=xgb_params["colsample_bytree"],
                objective="multi:softprob",
                num_class=3,
                random_state=ML_SETTINGS["random_state"],
                eval_metric="mlogloss",
                use_label_encoder=False,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                subsample=xgb_params["subsample"],
                random_state=ML_SETTINGS["random_state"],
            )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True

        logger.info(f"XGBoost accuracy: {self.accuracy:.4f}")
        self.save()
        return self.accuracy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.array([[0.33, 0.33, 0.34]])
        X_scaled = self.scaler.transform(X.reshape(1, -1)) if self.scaler else X.reshape(1, -1)
        return self.model.predict_proba(X_scaled)


class NeuralNetworkModel(BaseModel):
    """Simple neural network for match prediction."""

    def __init__(self):
        super().__init__("neural_network")
        self.model_path = MODELS_DIR / "neural_network_model.keras"

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        if not HAS_TF:
            # Fallback to sklearn MLP
            if HAS_SKLEARN:
                from sklearn.neural_network import MLPClassifier
                X_scaled = self.scaler.fit_transform(X) if self.scaler else X
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=ML_SETTINGS["test_size"],
                    random_state=ML_SETTINGS["random_state"], stratify=y
                )
                nn_params = ML_SETTINGS["neural_network"]
                self.model = MLPClassifier(
                    hidden_layer_sizes=tuple(nn_params["hidden_layers"]),
                    max_iter=nn_params["epochs"],
                    learning_rate_init=nn_params["learning_rate"],
                    random_state=ML_SETTINGS["random_state"],
                    early_stopping=True,
                )
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                self.accuracy = accuracy_score(y_test, y_pred)
                self.is_trained = True
                self.save()
                return self.accuracy
            return 0.0

        nn_params = ML_SETTINGS["neural_network"]
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=ML_SETTINGS["test_size"],
            random_state=ML_SETTINGS["random_state"], stratify=y
        )

        model = keras.Sequential([
            keras.layers.Input(shape=(X.shape[1],)),
            keras.layers.Dense(nn_params["hidden_layers"][0], activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(nn_params["hidden_layers"][1], activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(nn_params["hidden_layers"][2], activation='relu'),
            keras.layers.Dense(3, activation='softmax'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_params["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        model.fit(
            X_train, y_train,
            epochs=nn_params["epochs"],
            batch_size=nn_params["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            ],
            verbose=0,
        )

        _, self.accuracy = model.evaluate(X_test, y_test, verbose=0)
        self.model = model
        self.is_trained = True

        model.save(str(self.model_path))
        if self.scaler:
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)

        logger.info(f"Neural Network accuracy: {self.accuracy:.4f}")
        return self.accuracy

    def load(self) -> bool:
        model_file = self.model_path
        if model_file.exists() and HAS_TF:
            try:
                self.model = keras.models.load_model(str(model_file))
                if self.scaler_path.exists():
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                self.is_trained = True
                return True
            except Exception as e:
                logger.error(f"Error loading NN: {e}")
        elif self.model_path.with_suffix('.pkl').exists():
            return super().load()
        return False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            return np.array([[0.33, 0.33, 0.34]])
        X_scaled = self.scaler.transform(X.reshape(1, -1)) if self.scaler else X.reshape(1, -1)
        if HAS_TF and hasattr(self.model, 'predict'):
            probs = self.model.predict(X_scaled, verbose=0)
            return probs
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return np.array([[0.33, 0.33, 0.34]])


class RandomForestModel(BaseModel):
    """Random Forest classifier for match prediction."""

    def __init__(self):
        super().__init__("random_forest")

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        if not HAS_SKLEARN:
            return 0.0

        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=ML_SETTINGS["test_size"],
            random_state=ML_SETTINGS["random_state"], stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=ML_SETTINGS["random_state"],
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True

        logger.info(f"Random Forest accuracy: {self.accuracy:.4f}")
        self.save()
        return self.accuracy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.array([[0.33, 0.33, 0.34]])
        X_scaled = self.scaler.transform(X.reshape(1, -1)) if self.scaler else X.reshape(1, -1)
        return self.model.predict_proba(X_scaled)


class EnsembleModel:
    """Weighted ensemble of multiple models."""

    def __init__(self, models: Dict[str, BaseModel]):
        self.name = "ensemble"
        self.models = models
        self.weights = ML_SETTINGS["ensemble"]["weights"]
        self.accuracy = 0.0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of all model predictions."""
        weighted_probs = np.zeros(3)
        total_weight = 0

        for model_name, model in self.models.items():
            if model.is_trained:
                weight = self.weights.get(model_name, 0.33)
                probs = model.predict_proba(X)
                if probs.ndim > 1:
                    probs = probs[0]
                weighted_probs += probs * weight
                total_weight += weight

        if total_weight > 0:
            weighted_probs /= total_weight

        # Ensure they sum to 1
        total = weighted_probs.sum()
        if total > 0:
            weighted_probs /= total

        return weighted_probs.reshape(1, -1)


class PoissonModel:
    """Poisson distribution model for goal predictions."""

    def __init__(self):
        self.name = "poisson"
        self.avg_home_goals = 1.5
        self.avg_away_goals = 1.2

    def predict_score(self, home_attack: float, home_defense: float,
                      away_attack: float, away_defense: float) -> Tuple[float, float]:
        """Predict expected goals using Poisson model."""
        home_expected = (home_attack * away_defense / self.avg_away_goals) * self.avg_home_goals
        away_expected = (away_attack * home_defense / self.avg_home_goals) * self.avg_away_goals

        home_expected = max(0.1, min(home_expected, 5.0))
        away_expected = max(0.1, min(away_expected, 5.0))

        return round(home_expected, 2), round(away_expected, 2)

    def match_outcome_probs(self, home_expected: float, away_expected: float) -> Dict[str, float]:
        """Calculate match outcome probabilities from expected goals."""
        from math import exp, factorial

        max_goals = 8
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0

        for i in range(max_goals):
            for j in range(max_goals):
                p_i = (home_expected ** i) * exp(-home_expected) / factorial(i)
                p_j = (away_expected ** j) * exp(-away_expected) / factorial(j)
                p = p_i * p_j

                if i > j:
                    home_win_prob += p
                elif i == j:
                    draw_prob += p
                else:
                    away_win_prob += p

        return {
            "home_win": round(home_win_prob, 4),
            "draw": round(draw_prob, 4),
            "away_win": round(away_win_prob, 4),
        }
