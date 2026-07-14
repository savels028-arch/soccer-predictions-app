"""Pre-declared value-betting strategy grid and leakage-safe selection.

The grid is intentionally broad enough to compare useful approaches while
remaining auditable.  A strategy is selected on an inner selection slice and
is then applied unchanged to the following outer test season.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from research.metrics import flat_stake_metrics, settle_flat_stake


@dataclass(frozen=True)
class StrategySpec:
    market: str
    family: str
    odds_basis: str
    side: str
    min_edge: Optional[float]
    min_confidence: Optional[float]
    min_odds: float
    max_odds: float


DEFAULT_EDGE_THRESHOLDS = (None, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10)
DEFAULT_CONFIDENCE_THRESHOLDS = (None, 0.50, 0.55, 0.60, 0.65)
DEFAULT_ODDS_BANDS = ((1.20, 5.00), (1.20, 2.00), (1.50, 2.50), (1.80, 3.50), (2.00, 5.00))


def _side_names(class_count: int) -> Tuple[str, ...]:
    if class_count == 3:
        return ("all", "no_draw", "home", "draw", "away")
    if class_count == 2:
        return ("all", "under", "over")
    raise ValueError("only binary O/U and three-way 1X2 markets are supported")


def _side_mask(labels: np.ndarray, side: str, class_count: int) -> np.ndarray:
    if side == "all":
        return np.ones(len(labels), dtype=bool)
    if class_count == 3:
        mapping = {"home": 0, "draw": 1, "away": 2}
        if side == "no_draw":
            return labels != 1
    else:
        mapping = {"under": 0, "over": 1}
    if side not in mapping:
        raise ValueError(f"unsupported side {side!r}")
    return labels == mapping[side]


def candidate_mask(
    spec: StrategySpec,
    probabilities: np.ndarray,
    decimal_odds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return selection mask, labels, confidences, and model edges."""

    probs = np.asarray(probabilities, dtype=float)
    odds = np.asarray(decimal_odds, dtype=float)
    if probs.shape != odds.shape or probs.ndim != 2:
        raise ValueError("probabilities and decimal_odds must have the same 2D shape")
    valid_prices = np.isfinite(odds) & (odds > 1.0)
    expected_value = np.where(valid_prices, probs * odds - 1.0, -np.inf)
    labels = np.argmax(expected_value, axis=1).astype(int)
    row_index = np.arange(len(labels))
    selected_odds = odds[row_index, labels]
    confidence = probs[row_index, labels]
    edge = expected_value[row_index, labels]

    mask = valid_prices[row_index, labels]
    mask &= _side_mask(labels, spec.side, probs.shape[1])
    mask &= selected_odds >= spec.min_odds
    mask &= selected_odds <= spec.max_odds
    if spec.min_edge is not None:
        mask &= edge >= spec.min_edge
    if spec.min_confidence is not None:
        mask &= confidence >= spec.min_confidence
    return mask, labels, confidence, edge


def _chronological_block_rate(profits: np.ndarray, block_count: int = 6) -> float:
    if not len(profits):
        return 0.0
    blocks = [block for block in np.array_split(profits, min(block_count, len(profits))) if len(block)]
    return float(np.mean([float(block.sum()) > 0.0 for block in blocks]))


def evaluate_spec(
    spec: StrategySpec,
    probabilities: np.ndarray,
    decimal_odds: np.ndarray,
    outcomes: np.ndarray,
) -> Dict[str, object]:
    """Evaluate one fixed strategy on one chronological slice."""

    labels_y = np.asarray(outcomes, dtype=int)
    mask, labels, confidence, edge = candidate_mask(spec, probabilities, decimal_odds)
    selected = np.flatnonzero(mask)
    selected_labels = labels[selected]
    selected_odds = np.asarray(decimal_odds, dtype=float)[selected, selected_labels]
    wins = (selected_labels == labels_y[selected]).astype(int)
    metrics = flat_stake_metrics(wins.tolist(), selected_odds.tolist())
    profits = np.asarray(settle_flat_stake(wins.tolist(), selected_odds.tolist()), dtype=float)
    if len(profits) > 1:
        standard_error = float(profits.std(ddof=1) / np.sqrt(len(profits)))
    else:
        standard_error = float("inf")
    roi = float(metrics["roi"])
    result: Dict[str, object] = {
        "spec": asdict(spec),
        **metrics,
        "mean_edge": float(edge[selected].mean()) if len(selected) else 0.0,
        "mean_confidence": float(confidence[selected].mean()) if len(selected) else 0.0,
        "mean_odds": float(selected_odds.mean()) if len(selected) else 0.0,
        "roi_standard_error": standard_error,
        "roi_lcb_90": roi - 1.645 * standard_error if np.isfinite(standard_error) else float("-inf"),
        "positive_block_rate": _chronological_block_rate(profits),
    }
    return result


def _strategy_tie_break(result: Mapping[str, object]) -> tuple[object, ...]:
    """Return a stable, conservative tie-break independent of grid order."""

    raw_spec = result.get("spec")
    spec = raw_spec if isinstance(raw_spec, Mapping) else {}
    edge = spec.get("min_edge")
    confidence = spec.get("min_confidence")
    minimum = float(spec.get("min_odds", 0.0) or 0.0)
    maximum = float(spec.get("max_odds", float("inf")) or float("inf"))
    side = str(spec.get("side", ""))
    return (
        float(edge) if edge is not None else -1.0,
        float(confidence) if confidence is not None else -1.0,
        -(maximum - minimum),
        -maximum,
        int(side == "all"),
        str(spec.get("family", "")),
        str(spec.get("odds_basis", "")),
        side,
    )


def _rank_tuple(result: Mapping[str, object]) -> tuple[object, ...]:
    return (
        float(result.get("roi_lcb_90", float("-inf"))),
        float(result.get("roi", float("-inf"))),
        -float(result.get("max_drawdown", float("inf"))),
        int(result.get("bets", 0)),
        *_strategy_tie_break(result),
    )


def select_strategy(
    market: str,
    probability_families: Mapping[str, np.ndarray],
    odds_bases: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    *,
    min_bets: int = 40,
    edge_thresholds: Sequence[Optional[float]] = DEFAULT_EDGE_THRESHOLDS,
    confidence_thresholds: Sequence[Optional[float]] = DEFAULT_CONFIDENCE_THRESHOLDS,
    odds_bands: Sequence[Tuple[float, float]] = DEFAULT_ODDS_BANDS,
    allowed_odds_bases: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    """Select only among profitable, sufficiently sampled inner strategies.

    If no candidate satisfies the pre-declared eligibility rule, ``selected``
    is ``None``.  This prevents the old backtester behaviour where the least
    bad rejected strategy was silently deployed anyway.
    """

    allowed = set(allowed_odds_bases) if allowed_odds_bases is not None else set(odds_bases)
    evaluated = 0
    eligible = 0
    best: Optional[Dict[str, object]] = None

    for family_name, probabilities in probability_families.items():
        class_count = np.asarray(probabilities).shape[1]
        for basis_name, decimal_odds in odds_bases.items():
            if basis_name not in allowed:
                continue
            for side in _side_names(class_count):
                for min_edge in edge_thresholds:
                    for min_confidence in confidence_thresholds:
                        for min_odds, max_odds in odds_bands:
                            spec = StrategySpec(
                                market=market,
                                family=family_name,
                                odds_basis=basis_name,
                                side=side,
                                min_edge=min_edge,
                                min_confidence=min_confidence,
                                min_odds=float(min_odds),
                                max_odds=float(max_odds),
                            )
                            # Every StrategySpec is a different forward policy.
                            # Two thresholds can happen to select the same bets
                            # here and still diverge on the outer test slice, so
                            # deduplicating by realized inner bets would make the
                            # result depend on loop order.
                            result = evaluate_spec(spec, probabilities, decimal_odds, outcomes)
                            evaluated += 1
                            is_eligible = (
                                int(result["bets"]) >= min_bets
                                and float(result["profit"]) > 0.0
                                and float(result["positive_block_rate"]) >= 0.34
                            )
                            result["eligible"] = is_eligible
                            if not is_eligible:
                                continue
                            eligible += 1
                            if best is None or _rank_tuple(result) > _rank_tuple(best):
                                best = result

    return {
        "market": market,
        "evaluated_strategy_specs": evaluated,
        "eligible_strategy_specs": eligible,
        "selected": best,
    }


def apply_selected_strategy(
    selected: Optional[Mapping[str, object]],
    probability_families: Mapping[str, np.ndarray],
    odds_bases: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
) -> Dict[str, object]:
    """Apply an inner-selected strategy unchanged to an outer test slice."""

    if selected is None:
        return {"status": "abstained", "bets": 0, "wins": 0, "profit": 0.0, "roi": 0.0, "roi_pct": 0.0}
    spec = StrategySpec(**dict(selected["spec"]))
    if spec.family not in probability_families or spec.odds_basis not in odds_bases:
        raise KeyError("selected strategy references unavailable predictions or odds")
    result = evaluate_spec(
        spec,
        probability_families[spec.family],
        odds_bases[spec.odds_basis],
        outcomes,
    )
    result["status"] = "tested"
    return result
