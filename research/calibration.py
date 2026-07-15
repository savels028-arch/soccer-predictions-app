"""Point-in-time probability calibration used by the research strategy zoo.

The calibrators in this module are deliberately small.  They are fitted on a
dedicated calibration slice and must never see the strategy-selection or test
slice.  That separation is enforced by the caller in :mod:`research.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


EPSILON = 1e-9


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Clip and normalize a two-dimensional probability matrix."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("probabilities must be a 2D array with at least two classes")
    values = np.nan_to_num(values, nan=1.0 / values.shape[1], posinf=1.0, neginf=0.0)
    values = np.clip(values, EPSILON, 1.0)
    totals = values.sum(axis=1, keepdims=True)
    bad = totals[:, 0] <= 0.0
    if np.any(bad):
        values[bad] = 1.0 / values.shape[1]
        totals = values.sum(axis=1, keepdims=True)
    return values / totals


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature scaling to probability logits."""

    values = normalize_probabilities(probabilities)
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a positive finite number")
    logits = np.log(values) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
    """Fit one temperature by minimizing multiclass log loss."""

    values = normalize_probabilities(probabilities)
    labels = np.asarray(outcomes, dtype=int)
    if len(values) != len(labels) or not len(labels):
        raise ValueError("probabilities and outcomes must have equal non-zero length")
    if labels.min() < 0 or labels.max() >= values.shape[1]:
        raise ValueError("outcomes contain an invalid class")

    def objective(log_temperature: float) -> float:
        scaled = apply_temperature(values, float(np.exp(log_temperature)))
        actual = np.clip(scaled[np.arange(len(labels)), labels], EPSILON, 1.0)
        return float(-np.log(actual).mean())

    result = minimize_scalar(objective, bounds=(-2.3, 2.3), method="bounded")
    return float(np.exp(result.x)) if result.success else 1.0


@dataclass
class IsotonicOVRCalibrator:
    """One-vs-rest isotonic calibration followed by row normalization."""

    models: Dict[int, IsotonicRegression]
    class_count: int

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        values = normalize_probabilities(probabilities)
        if values.shape[1] != self.class_count:
            raise ValueError("probability class count differs from fitted calibrator")
        calibrated = np.column_stack(
            [self.models[index].predict(values[:, index]) for index in range(self.class_count)]
        )
        return normalize_probabilities(calibrated)


def fit_isotonic_ovr(probabilities: np.ndarray, outcomes: np.ndarray) -> IsotonicOVRCalibrator:
    """Fit an isotonic map for each class on a dedicated calibration set."""

    values = normalize_probabilities(probabilities)
    labels = np.asarray(outcomes, dtype=int)
    if len(values) != len(labels) or not len(labels):
        raise ValueError("probabilities and outcomes must have equal non-zero length")
    models: Dict[int, IsotonicRegression] = {}
    for class_index in range(values.shape[1]):
        target = (labels == class_index).astype(float)
        model = IsotonicRegression(out_of_bounds="clip", y_min=EPSILON, y_max=1.0 - EPSILON)
        if len(np.unique(target)) < 2:
            # A season slice should normally contain every class, but a
            # constant two-point anchor is safer than failing or leaking data.
            rate = float(np.clip(target.mean(), EPSILON, 1.0 - EPSILON))
            model.fit([0.0, 1.0], [rate, rate])
        else:
            model.fit(values[:, class_index], target)
        models[class_index] = model
    return IsotonicOVRCalibrator(models=models, class_count=values.shape[1])


def calibrated_variants(
    calibration_probabilities: np.ndarray,
    calibration_outcomes: np.ndarray,
    target_probabilities: np.ndarray,
    *,
    include_isotonic: bool = True,
) -> Dict[str, np.ndarray]:
    """Return raw, temperature-scaled, and optionally isotonic predictions."""

    calibration_values = normalize_probabilities(calibration_probabilities)
    target_values = normalize_probabilities(target_probabilities)
    temperature = fit_temperature(calibration_values, calibration_outcomes)
    variants = {
        "raw": target_values,
        "temperature": apply_temperature(target_values, temperature),
    }
    if include_isotonic:
        isotonic = fit_isotonic_ovr(calibration_values, calibration_outcomes)
        variants["isotonic"] = isotonic.transform(target_values)
    return variants
