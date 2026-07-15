"""Small, dependency-free metrics for betting strategy research.

All ROI values are returned as fractions (``0.05`` means five percent), with
an explicit ``*_pct`` field where a percentage is useful for reports.  The
bootstrap operates on ordered per-bet profit, so callers should pass bets in
the order in which they would have been placed.
"""

from __future__ import annotations

import math
import random
from numbers import Integral, Real
from statistics import median, pstdev
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DEFAULT_BOOTSTRAP_SEED = 20260714


def _finite_floats(values: Iterable[float], name: str) -> List[float]:
    result: List[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be numeric") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def _binary_outcomes(values: Iterable[Any], name: str = "outcomes") -> List[int]:
    outcomes: List[int] = []
    for index, value in enumerate(values):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be 0 or 1") from exc
        if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise ValueError(f"{name}[{index}] must be 0 or 1")
        outcomes.append(int(numeric))
    return outcomes


def settle_flat_stake(
    outcomes: Sequence[int | bool],
    decimal_odds: Sequence[float],
    *,
    stake: float = 1.0,
) -> List[float]:
    """Return realized profit for one fixed-size stake per selection.

    ``outcomes`` contains 1 for a winning bet and 0 for a losing bet.  Odds
    are decimal odds and must be strictly greater than one.
    """

    wins = _binary_outcomes(outcomes)
    odds = _finite_floats(decimal_odds, "decimal_odds")
    stake_value = float(stake)
    if not math.isfinite(stake_value) or stake_value <= 0.0:
        raise ValueError("stake must be a positive finite number")
    if len(wins) != len(odds):
        raise ValueError("outcomes and decimal_odds must have the same length")
    for index, odd in enumerate(odds):
        if odd <= 1.0:
            raise ValueError(f"decimal_odds[{index}] must be greater than 1")

    return [stake_value * (odd - 1.0) if won else -stake_value for won, odd in zip(wins, odds)]


def max_drawdown(profits: Sequence[float]) -> float:
    """Return the largest peak-to-trough decline in an ordered profit curve."""

    values = _finite_floats(profits, "profits")
    equity = 0.0
    peak = 0.0
    largest = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        largest = max(largest, peak - equity)
    return largest


def flat_stake_metrics(
    outcomes: Sequence[int | bool],
    decimal_odds: Sequence[float],
    *,
    stake: float = 1.0,
) -> Dict[str, float | int]:
    """Summarize flat-stake profit, ROI, hit rate, and drawdown."""

    wins = _binary_outcomes(outcomes)
    profits = settle_flat_stake(wins, decimal_odds, stake=stake)
    bets = len(profits)
    stake_value = float(stake)
    total_staked = bets * stake_value
    profit = math.fsum(profits)
    returned = total_staked + profit
    win_count = sum(wins)
    roi = profit / total_staked if total_staked else 0.0
    hit_rate = win_count / bets if bets else 0.0

    return {
        "bets": bets,
        "wins": win_count,
        "hit_rate": hit_rate,
        "staked": total_staked,
        "returned": returned,
        "profit": profit,
        "roi": roi,
        "roi_pct": roi * 100.0,
        "max_drawdown": max_drawdown(profits),
    }


def _probability_rows(probabilities: Sequence[Any]) -> Tuple[str, List[Any]]:
    rows = list(probabilities)
    if not rows:
        raise ValueError("probabilities must not be empty")
    first = rows[0]
    if isinstance(first, (str, bytes)):
        raise ValueError("probabilities must be numeric")
    try:
        iter(first)
    except TypeError:
        return "binary", _finite_floats(rows, "probabilities")
    return "multiclass", rows


def _validated_probability_inputs(
    probabilities: Sequence[Any],
    outcomes: Sequence[Any],
) -> Tuple[str, List[Any], List[int]]:
    mode, raw_rows = _probability_rows(probabilities)
    raw_outcomes = list(outcomes)
    if len(raw_rows) != len(raw_outcomes):
        raise ValueError("probabilities and outcomes must have the same length")

    if mode == "binary":
        probs = list(raw_rows)
        labels = _binary_outcomes(raw_outcomes)
        for index, probability in enumerate(probs):
            if probability < 0.0 or probability > 1.0:
                raise ValueError(f"probabilities[{index}] must be between 0 and 1")
        return mode, probs, labels

    rows: List[List[float]] = []
    class_count = 0
    for index, raw_row in enumerate(raw_rows):
        try:
            row = _finite_floats(raw_row, f"probabilities[{index}]")
        except TypeError as exc:
            raise ValueError(f"probabilities[{index}] must be a probability vector") from exc
        if index == 0:
            class_count = len(row)
            if class_count < 2:
                raise ValueError("multiclass probabilities need at least two classes")
        elif len(row) != class_count:
            raise ValueError("all probability vectors must have the same length")
        if any(probability < 0.0 or probability > 1.0 for probability in row):
            raise ValueError(f"probabilities[{index}] must be between 0 and 1")
        if not math.isclose(math.fsum(row), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(f"probabilities[{index}] must sum to 1")
        rows.append(row)

    labels: List[int] = []
    for index, value in enumerate(raw_outcomes):
        if isinstance(value, Integral):
            label = value
        elif isinstance(value, Real) and float(value).is_integer():
            label = int(value)
        else:
            raise ValueError(f"outcomes[{index}] must be an integer class index")
        label = int(label)
        if label < 0 or label >= class_count:
            raise ValueError(f"outcomes[{index}] is outside the probability vector")
        labels.append(label)
    return mode, rows, labels


def brier_score(probabilities: Sequence[Any], outcomes: Sequence[Any]) -> float:
    """Return mean Brier score for binary or multiclass probabilities.

    Binary input uses the conventional ``(p - y)^2`` definition.  Multiclass
    input uses the unscaled sum of squared errors across all classes, matching
    the convention already used by AIBets for 1X2 forecasts.
    """

    mode, rows, labels = _validated_probability_inputs(probabilities, outcomes)
    if mode == "binary":
        scores = [(probability - label) ** 2 for probability, label in zip(rows, labels)]
    else:
        scores = [
            math.fsum((probability - (1.0 if class_index == label else 0.0)) ** 2 for class_index, probability in enumerate(row))
            for row, label in zip(rows, labels)
        ]
    return math.fsum(scores) / len(scores)


def log_loss(
    probabilities: Sequence[Any],
    outcomes: Sequence[Any],
    *,
    epsilon: float = 1e-15,
) -> float:
    """Return mean negative log likelihood for binary or multiclass forecasts."""

    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0 or epsilon_value >= 0.5:
        raise ValueError("epsilon must be a finite number between 0 and 0.5")
    mode, rows, labels = _validated_probability_inputs(probabilities, outcomes)
    losses: List[float] = []
    if mode == "binary":
        for probability, label in zip(rows, labels):
            actual_probability = probability if label else 1.0 - probability
            losses.append(-math.log(min(1.0 - epsilon_value, max(epsilon_value, actual_probability))))
    else:
        for row, label in zip(rows, labels):
            losses.append(-math.log(min(1.0 - epsilon_value, max(epsilon_value, row[label]))))
    return math.fsum(losses) / len(losses)


def probability_metrics(probabilities: Sequence[Any], outcomes: Sequence[Any]) -> Dict[str, float]:
    """Return Brier score and log loss for the same forecast sample."""

    return {
        "brier_score": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
    }


def season_stability(
    profits: Sequence[float],
    seasons: Sequence[Any],
    *,
    stake: float = 1.0,
) -> Dict[str, Any]:
    """Summarize how consistently a flat-stake strategy performs by season."""

    values = _finite_floats(profits, "profits")
    labels = list(seasons)
    stake_value = float(stake)
    if not math.isfinite(stake_value) or stake_value <= 0.0:
        raise ValueError("stake must be a positive finite number")
    if len(values) != len(labels):
        raise ValueError("profits and seasons must have the same length")

    grouped: Dict[str, List[float]] = {}
    for profit, season in zip(values, labels):
        key = str(season)
        grouped.setdefault(key, []).append(profit)

    by_season: Dict[str, Dict[str, float | int | bool]] = {}
    season_rois: List[float] = []
    for season, season_profits in grouped.items():
        bet_count = len(season_profits)
        profit = math.fsum(season_profits)
        roi = profit / (bet_count * stake_value)
        season_rois.append(roi)
        by_season[season] = {
            "bets": bet_count,
            "profit": profit,
            "roi": roi,
            "roi_pct": roi * 100.0,
            "profitable": profit > 0.0,
        }

    season_count = len(season_rois)
    profitable_count = sum(roi > 0.0 for roi in season_rois)
    return {
        "n_seasons": season_count,
        "profitable_seasons": profitable_count,
        "positive_season_rate": profitable_count / season_count if season_count else 0.0,
        "mean_season_roi": math.fsum(season_rois) / season_count if season_count else 0.0,
        "median_season_roi": median(season_rois) if season_count else 0.0,
        "season_roi_std": pstdev(season_rois) if season_count > 1 else 0.0,
        "worst_season_roi": min(season_rois) if season_count else 0.0,
        "best_season_roi": max(season_rois) if season_count else 0.0,
        "by_season": by_season,
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def block_bootstrap_roi(
    profits: Sequence[float],
    *,
    stake: float = 1.0,
    block_size: int | None = None,
    n_resamples: int = 2_000,
    confidence: float = 0.95,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> Dict[str, float | int]:
    """Estimate ROI uncertainty with a deterministic circular block bootstrap.

    Contiguous circular blocks preserve short-range correlation better than an
    IID bootstrap.  The default block length is the rounded square root of the
    number of bets; research runs should set it explicitly when domain evidence
    supports another dependence horizon.
    """

    values = _finite_floats(profits, "profits")
    if not values:
        raise ValueError("profits must not be empty")
    stake_value = float(stake)
    if not math.isfinite(stake_value) or stake_value <= 0.0:
        raise ValueError("stake must be a positive finite number")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    confidence_value = float(confidence)
    if not math.isfinite(confidence_value) or confidence_value <= 0.0 or confidence_value >= 1.0:
        raise ValueError("confidence must be between 0 and 1")

    bet_count = len(values)
    selected_block_size = max(1, round(math.sqrt(bet_count))) if block_size is None else block_size
    if isinstance(selected_block_size, bool) or not isinstance(selected_block_size, int):
        raise ValueError("block_size must be an integer")
    if selected_block_size < 1 or selected_block_size > bet_count:
        raise ValueError("block_size must be between 1 and the number of bets")

    rng = random.Random(seed)
    sampled_rois: List[float] = []
    for _ in range(n_resamples):
        sampled: List[float] = []
        while len(sampled) < bet_count:
            start = rng.randrange(bet_count)
            sampled.extend(values[(start + offset) % bet_count] for offset in range(selected_block_size))
        sampled_rois.append(math.fsum(sampled[:bet_count]) / (bet_count * stake_value))

    sampled_rois.sort()
    tail = (1.0 - confidence_value) / 2.0
    point_estimate = math.fsum(values) / (bet_count * stake_value)
    ci_lower = _quantile(sampled_rois, tail)
    ci_upper = _quantile(sampled_rois, 1.0 - tail)
    probability_positive = sum(roi > 0.0 for roi in sampled_rois) / n_resamples
    return {
        "bets": bet_count,
        "block_size": selected_block_size,
        "n_resamples": n_resamples,
        "confidence": confidence_value,
        "seed": seed,
        "roi": point_estimate,
        "bootstrap_mean_roi": math.fsum(sampled_rois) / n_resamples,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "probability_roi_positive": probability_positive,
    }


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "block_bootstrap_roi",
    "brier_score",
    "flat_stake_metrics",
    "log_loss",
    "max_drawdown",
    "probability_metrics",
    "season_stability",
    "settle_flat_stake",
]
