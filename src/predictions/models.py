"""
ML Models for Soccer Match Prediction
Includes: XGBoost, Neural Network, Random Forest, Ensemble, and Stacking.
v1: Original baseline models
v2: Parametrised constructors for A/B testing with improved defaults
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
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

    def __init__(self, name: str, config: Dict = None, suffix: str = ""):
        self.name = name
        self.config = config or ML_SETTINGS
        self.suffix = suffix
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
        self.accuracy = 0.0
        self.model_path = MODELS_DIR / f"{name}{suffix}_model.pkl"
        self.scaler_path = MODELS_DIR / f"{name}{suffix}_scaler.pkl"

    def save(self):
        """Save model and scaler to disk."""
        if self.model:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            if self.scaler:
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            logger.info(f"Model {self.name} saved to {self.model_path}")

    def _expected_features(self) -> int:
        """Get expected feature count from FeatureEngineer."""
        try:
            from .feature_engineering import FeatureEngineer
            return len(FeatureEngineer.FEATURE_NAMES)
        except Exception:
            return 0

    def load(self) -> bool:
        """Load model and scaler from disk."""
        expected = self._expected_features()
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                if self.scaler_path.exists():
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                # Check feature dimension matches current expected count
                if expected > 0 and self.scaler and hasattr(self.scaler, 'n_features_in_'):
                    if self.scaler.n_features_in_ != expected:
                        logger.warning(
                            f"Model {self.name} has {self.scaler.n_features_in_} features "
                            f"but expected {expected}. Needs retraining."
                        )
                        self.model = None
                        self.scaler = StandardScaler() if HAS_SKLEARN else None
                        self.is_trained = False
                        return False
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

    def __init__(self, config: Dict = None, suffix: str = ""):
        super().__init__("xgboost", config, suffix)

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        if not HAS_XGBOOST and not HAS_SKLEARN:
            logger.error("No ML library available")
            return 0.0

        # Fix: fit scaler on TRAIN only (no leakage)
        split_idx = int(len(X) * (1 - self.config["test_size"]))
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if self.scaler:
            self.scaler.fit(X_train_raw)
            X_train = self.scaler.transform(X_train_raw)
            X_test = self.scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        xgb_params = self.config["xgboost"]

        if HAS_XGBOOST:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y_train)
            self.model = xgb.XGBClassifier(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                subsample=xgb_params["subsample"],
                colsample_bytree=xgb_params.get("colsample_bytree", 0.8),
                reg_alpha=xgb_params.get("reg_alpha", 0),
                reg_lambda=xgb_params.get("reg_lambda", 1),
                min_child_weight=xgb_params.get("min_child_weight", 1),
                gamma=xgb_params.get("gamma", 0),
                objective="multi:softprob",
                num_class=3,
                random_state=self.config["random_state"],
                eval_metric="mlogloss",
                use_label_encoder=False,
            )
        else:
            sample_weights = None
            self.model = GradientBoostingClassifier(
                n_estimators=xgb_params["n_estimators"],
                max_depth=xgb_params["max_depth"],
                learning_rate=xgb_params["learning_rate"],
                subsample=xgb_params["subsample"],
                random_state=self.config["random_state"],
            )

        self.model.fit(X_train, y_train, sample_weight=sample_weights)

        # Feature importance logging
        if hasattr(self.model, 'feature_importances_'):
            top_idx = np.argsort(self.model.feature_importances_)[-10:][::-1]
            logger.info(f"XGBoost top-10 feature indices: {top_idx.tolist()}")
            logger.info(f"XGBoost top-10 importances: {self.model.feature_importances_[top_idx].tolist()}")

        # Probability calibration (sigmoid = Platt scaling, resists overfitting better)
        try:
            cal = CalibratedClassifierCV(self.model, method='sigmoid', cv=5)
            cal.fit(X_train, y_train)
            self.model = cal
            logger.info("XGBoost wrapped with CalibratedClassifierCV (sigmoid, cv=5)")
        except Exception as e:
            logger.warning(f"Calibration failed, using raw model: {e}")

        # Temporal cross-validation report
        try:
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=tscv, scoring='accuracy')
            logger.info(f"XGBoost temporal CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        except Exception as e:
            logger.debug(f"Temporal CV report failed: {e}")

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True

        logger.info(f"XGBoost accuracy: {self.accuracy:.4f}")
        self.save()
        return self.accuracy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.array([[0.33, 0.33, 0.34]])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict_proba(X_scaled)


class NeuralNetworkModel(BaseModel):
    """Simple neural network for match prediction."""

    def __init__(self, config: Dict = None, suffix: str = ""):
        super().__init__("neural_network", config, suffix)
        self.model_path = MODELS_DIR / f"neural_network{suffix}_model.keras"

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        if not HAS_TF:
            # Fallback to sklearn MLP
            if HAS_SKLEARN:
                from sklearn.neural_network import MLPClassifier
                split_idx = int(len(X) * (1 - self.config["test_size"]))
                X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                if self.scaler:
                    self.scaler.fit(X_train_raw)
                    X_train = self.scaler.transform(X_train_raw)
                    X_test = self.scaler.transform(X_test_raw)
                else:
                    X_train, X_test = X_train_raw, X_test_raw
                nn_params = self.config["neural_network"]
                self.model = MLPClassifier(
                    hidden_layer_sizes=tuple(nn_params["hidden_layers"]),
                    max_iter=nn_params["epochs"],
                    learning_rate_init=nn_params["learning_rate"],
                    random_state=self.config["random_state"],
                    early_stopping=True,
                )
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                self.accuracy = accuracy_score(y_test, y_pred)
                self.is_trained = True
                self.save()
                return self.accuracy
            return 0.0

        nn_params = self.config["neural_network"]

        # Fix: fit scaler on TRAIN only
        split_idx = int(len(X) * (1 - self.config["test_size"]))
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if self.scaler:
            self.scaler.fit(X_train_raw)
            X_train = self.scaler.transform(X_train_raw)
            X_test = self.scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        # Get dropout rates (v2 allows per-layer config)
        dropout_rates = nn_params.get("dropout_rates", [0.3, 0.2, 0.0])
        layers = nn_params["hidden_layers"]

        # Build model dynamically
        inputs = keras.layers.Input(shape=(X.shape[1],))
        x = keras.layers.Dense(layers[0], activation='relu')(inputs)
        if len(dropout_rates) > 0 and dropout_rates[0] > 0:
            x = keras.layers.Dropout(dropout_rates[0])(x)
        x = keras.layers.BatchNormalization()(x)

        for i in range(1, len(layers)):
            x = keras.layers.Dense(layers[i], activation='relu')(x)
            dr = dropout_rates[i] if i < len(dropout_rates) else 0.0
            if dr > 0:
                x = keras.layers.Dropout(dr)(x)
            x = keras.layers.BatchNormalization()(x)

        outputs = keras.layers.Dense(3, activation='softmax')(x)
        model = keras.Model(inputs, outputs)

        # Compute class weights for imbalanced classes
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        cw = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {int(c): w for c, w in zip(classes, cw)}

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=nn_params["learning_rate"]),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        ]
        # v2: add learning rate scheduler
        if nn_params.get("use_lr_scheduler"):
            callbacks.append(
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
            )

        model.fit(
            X_train, y_train,
            epochs=nn_params["epochs"],
            batch_size=nn_params["batch_size"],
            validation_data=(X_test, y_test),
            class_weight=class_weight_dict,
            callbacks=callbacks,
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
        expected = self._expected_features()
        model_file = self.model_path
        if model_file.exists() and HAS_TF:
            try:
                self.model = keras.models.load_model(str(model_file))
                if self.scaler_path.exists():
                    with open(self.scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                # Check feature dimension
                if expected > 0 and self.scaler and hasattr(self.scaler, 'n_features_in_'):
                    if self.scaler.n_features_in_ != expected:
                        logger.warning(
                            f"NN model has {self.scaler.n_features_in_} features "
                            f"but expected {expected}. Needs retraining."
                        )
                        self.model = None
                        self.scaler = StandardScaler() if HAS_SKLEARN else None
                        self.is_trained = False
                        return False
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
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        if HAS_TF and hasattr(self.model, 'predict'):
            probs = self.model.predict(X_scaled, verbose=0)
            return probs
        elif hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        return np.array([[0.33, 0.33, 0.34]])


class RandomForestModel(BaseModel):
    """Random Forest classifier for match prediction."""

    def __init__(self, config: Dict = None, suffix: str = ""):
        super().__init__("random_forest", config, suffix)

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        if not HAS_SKLEARN:
            return 0.0

        # Fix: fit scaler on TRAIN only
        split_idx = int(len(X) * (1 - self.config["test_size"]))
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if self.scaler:
            self.scaler.fit(X_train_raw)
            X_train = self.scaler.transform(X_train_raw)
            X_test = self.scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        rf_params = self.config.get("random_forest", {})
        self.model = RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 300),
            max_depth=rf_params.get("max_depth", 10),
            min_samples_split=rf_params.get("min_samples_split", 5),
            min_samples_leaf=rf_params.get("min_samples_leaf", 2),
            class_weight='balanced',
            random_state=self.config["random_state"],
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        # Probability calibration (sigmoid = Platt scaling, resists overfitting better)
        try:
            cal = CalibratedClassifierCV(self.model, method='sigmoid', cv=5)
            cal.fit(X_train, y_train)
            self.model = cal
            logger.info("RandomForest wrapped with CalibratedClassifierCV (sigmoid, cv=5)")
        except Exception as e:
            logger.warning(f"RF calibration failed, using raw model: {e}")

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True

        logger.info(f"Random Forest accuracy: {self.accuracy:.4f}")
        self.save()
        return self.accuracy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.array([[0.33, 0.33, 0.34]])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict_proba(X_scaled)


class LightGBMModel(BaseModel):
    """LightGBM classifier for match prediction."""

    def __init__(self, config: Dict = None, suffix: str = ""):
        super().__init__("lightgbm", config, suffix)

    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        try:
            import lightgbm as lgb
        except ImportError:
            logger.info("LightGBM not installed — skipping")
            return 0.0

        split_idx = int(len(X) * (1 - self.config["test_size"]))
        X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        if self.scaler:
            self.scaler.fit(X_train_raw)
            X_train = self.scaler.transform(X_train_raw)
            X_test = self.scaler.transform(X_test_raw)
        else:
            X_train, X_test = X_train_raw, X_test_raw

        lgb_params = self.config.get("lightgbm", {})
        self.model = lgb.LGBMClassifier(
            n_estimators=lgb_params.get("n_estimators", 300),
            max_depth=lgb_params.get("max_depth", 7),
            learning_rate=lgb_params.get("learning_rate", 0.05),
            subsample=lgb_params.get("subsample", 0.85),
            colsample_bytree=lgb_params.get("colsample_bytree", 0.8),
            reg_alpha=lgb_params.get("reg_alpha", 0.1),
            reg_lambda=lgb_params.get("reg_lambda", 1.0),
            min_child_samples=lgb_params.get("min_child_samples", 20),
            num_leaves=lgb_params.get("num_leaves", 31),
            objective="multiclass",
            num_class=3,
            class_weight="balanced",
            random_state=self.config["random_state"],
            verbose=-1,
        )
        self.model.fit(X_train, y_train)

        # Feature importance logging
        if hasattr(self.model, 'feature_importances_'):
            top_idx = np.argsort(self.model.feature_importances_)[-10:][::-1]
            logger.info(f"LightGBM top-10 feature indices: {top_idx.tolist()}")

        # Calibration (sigmoid = Platt scaling, resists overfitting better)
        try:
            cal = CalibratedClassifierCV(self.model, method='sigmoid', cv=5)
            cal.fit(X_train, y_train)
            self.model = cal
            logger.info("LightGBM wrapped with CalibratedClassifierCV (sigmoid, cv=5)")
        except Exception as e:
            logger.warning(f"LightGBM calibration failed: {e}")

        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.is_trained = True

        logger.info(f"LightGBM accuracy: {self.accuracy:.4f}")
        self.save()
        return self.accuracy

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.array([[0.33, 0.33, 0.34]])
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict_proba(X_scaled)


class EnsembleModel:
    """Weighted ensemble of multiple models."""

    def __init__(self, models: Dict[str, BaseModel], config: Dict = None):
        self.name = "ensemble"
        self.models = models
        cfg = config or ML_SETTINGS
        self.weights = cfg["ensemble"]["weights"]
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
        self.avg_home_goals = 1.5  # defaults, updated by calibrate()
        self.avg_away_goals = 1.2

    def calibrate(self, db_manager, league_codes: List[str] = None):
        """Q4: Compute real league average goals from historical data."""
        try:
            leagues = league_codes or ["PL", "PD", "BL1", "SA", "FL1"]
            total_home = 0
            total_away = 0
            total_matches = 0
            for lc in leagues:
                matches = db_manager.get_finished_matches(lc)
                for m in (matches or []):
                    hs = m.get("home_score")
                    aws = m.get("away_score")
                    if hs is not None and aws is not None:
                        total_home += hs
                        total_away += aws
                        total_matches += 1
            if total_matches >= 50:
                self.avg_home_goals = total_home / total_matches
                self.avg_away_goals = total_away / total_matches
                logger.info(f"Poisson calibrated: home={self.avg_home_goals:.2f}, "
                            f"away={self.avg_away_goals:.2f} from {total_matches} matches")
            else:
                logger.info(f"Poisson: only {total_matches} matches, keeping defaults")
        except Exception as e:
            logger.warning(f"Poisson calibration failed, keeping defaults: {e}")

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


class StackingEnsemble:
    """
    Stacking meta-learner: trains a LogisticRegression on the concatenated
    probability outputs of all base models.  Falls back to weighted average
    if there aren't enough samples.
    """

    def __init__(self, models: Dict[str, BaseModel], config: Dict = None, suffix: str = ""):
        self.name = "stacking_ensemble"
        self.models = models
        self.config = config or ML_SETTINGS
        self.suffix = suffix
        self.meta_model = None
        self.is_trained = False
        self.accuracy = 0.0
        self.model_path = MODELS_DIR / f"stacking_meta{suffix}.pkl"
        self.fallback = EnsembleModel(models, config)

    def train_meta(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train the stacking meta-learner using TimeSeriesSplit OOF predictions.
        This avoids data leakage by generating out-of-fold predictions from
        temporal cross-validation on the TRAINING portion.
        """
        if not HAS_SKLEARN:
            return 0.0

        from sklearn.model_selection import TimeSeriesSplit

        split_idx = int(len(X) * (1 - self.config["test_size"]))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        if len(X_train) < 200:
            logger.info("Stacking: too few training samples, falling back to weighted ensemble")
            return 0.0

        # Generate OOF predictions via TimeSeriesSplit on training data (batch mode)
        n_splits = min(5, max(2, len(X_train) // 100))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_meta = np.zeros((len(X_train), 3 * len(self.models)))
        oof_mask = np.zeros(len(X_train), dtype=bool)

        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_val = X_train[val_idx]
                for m_idx, (name, model) in enumerate(self.models.items()):
                    if model.is_trained:
                        probs_batch = model.predict_proba(X_val)  # batch prediction
                        if probs_batch.ndim == 1:
                            probs_batch = probs_batch.reshape(1, -1)
                        oof_meta[val_idx, m_idx*3:(m_idx+1)*3] = probs_batch
                    else:
                        oof_meta[val_idx, m_idx*3:(m_idx+1)*3] = [0.33, 0.33, 0.34]
                oof_mask[val_idx] = True

        # Keep only rows that have OOF predictions
        meta_X_train = oof_meta[oof_mask]
        meta_y_train = y_train[oof_mask]

        if len(meta_X_train) < 50:
            logger.info("Stacking: too few OOF samples, falling back to weighted ensemble")
            return 0.0

        # Build test meta-features from base model predictions (batch mode)
        meta_X_test = np.zeros((len(X_test), 3 * len(self.models)))
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            for m_idx, (name, model) in enumerate(self.models.items()):
                if model.is_trained:
                    probs_batch = model.predict_proba(X_test)
                    if probs_batch.ndim == 1:
                        probs_batch = probs_batch.reshape(1, -1)
                    meta_X_test[:, m_idx*3:(m_idx+1)*3] = probs_batch
                else:
                    meta_X_test[:, m_idx*3:(m_idx+1)*3] = [0.33, 0.33, 0.34]

        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver='lbfgs',
            class_weight='balanced',
        )
        self.meta_model.fit(meta_X_train, meta_y_train)

        y_pred = self.meta_model.predict(meta_X_test)
        self.accuracy = accuracy_score(y_pred, y_test) if len(y_test) > 0 else 0.0
        self.is_trained = True

        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.meta_model, f)
        except Exception as e:
            logger.warning(f"Could not save stacking meta-model: {e}")

        logger.info(f"Stacking meta-learner accuracy: {self.accuracy:.4f} "
                     f"(OOF samples={len(meta_X_train)}, test={len(meta_X_test)})")
        return self.accuracy

    def load(self) -> bool:
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.meta_model = pickle.load(f)
                self.is_trained = True
                return True
            except Exception:
                pass
        return False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict using stacking meta-model, fall back to weighted average."""
        if not self.is_trained or self.meta_model is None:
            return self.fallback.predict_proba(X)

        # Build meta-features from base model predictions
        row_feats = []
        for name, model in self.models.items():
            if model.is_trained:
                probs = model.predict_proba(X)
                if probs.ndim > 1:
                    probs = probs[0]
                row_feats.extend(probs.tolist())
            else:
                row_feats.extend([0.33, 0.33, 0.34])

        meta_X = np.array(row_feats).reshape(1, -1)
        probs = self.meta_model.predict_proba(meta_X)
        return probs
