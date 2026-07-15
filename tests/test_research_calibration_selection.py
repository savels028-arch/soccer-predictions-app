import numpy as np
import pytest

from research.calibration import (
    apply_temperature,
    calibrated_variants,
    fit_temperature,
    normalize_probabilities,
)
from research.selection import StrategySpec, apply_selected_strategy, candidate_mask, select_strategy


def test_temperature_calibration_is_fitted_without_changing_shape():
    calibration = np.array([[0.95, 0.05], [0.90, 0.10], [0.10, 0.90], [0.05, 0.95]])
    outcomes = np.array([0, 1, 1, 0])
    target = np.array([[0.8, 0.2], [0.3, 0.7]])

    temperature = fit_temperature(calibration, outcomes)
    scaled = apply_temperature(target, temperature)
    variants = calibrated_variants(calibration, outcomes, target)

    assert temperature > 1.0
    assert scaled.shape == target.shape
    assert np.allclose(scaled.sum(axis=1), 1.0)
    assert set(variants) == {"raw", "temperature", "isotonic"}


def test_normalize_probabilities_repairs_invalid_rows_without_nans():
    normalized = normalize_probabilities(np.array([[np.nan, 0.5], [0.0, 0.0], [2.0, 1.0]]))
    assert np.all(np.isfinite(normalized))
    assert np.allclose(normalized.sum(axis=1), 1.0)


def test_candidate_uses_highest_expected_value_and_side_filter():
    probabilities = np.array([[0.60, 0.25, 0.15], [0.40, 0.30, 0.30]])
    odds = np.array([[1.50, 4.20, 8.00], [2.20, 3.40, 4.00]])
    spec = StrategySpec("1x2", "demo", "primary", "no_draw", 0.0, None, 1.2, 10.0)

    mask, labels, _, edge = candidate_mask(spec, probabilities, odds)

    assert labels.tolist() == [2, 2]
    assert mask.tolist() == [True, True]
    assert edge[0] == pytest.approx(0.2)


def test_selection_abstains_when_no_profitable_candidate():
    probabilities = {"bad": np.tile([[0.8, 0.2]], (50, 1))}
    odds = {"book": np.tile([[1.2, 4.0]], (50, 1))}
    outcomes = np.ones(50, dtype=int)

    result = select_strategy(
        "ou25",
        probabilities,
        odds,
        outcomes,
        min_bets=20,
        edge_thresholds=(None,),
        confidence_thresholds=(None,),
        odds_bands=((1.1, 10.0),),
    )

    assert result["selected"] is None
    assert apply_selected_strategy(None, probabilities, odds, outcomes)["status"] == "abstained"


def test_selected_strategy_is_applied_unchanged_to_test_slice():
    selection_probabilities = {"model": np.tile([[0.7, 0.3]], (60, 1))}
    selection_odds = {"book": np.tile([[1.8, 3.0]], (60, 1))}
    selection_outcomes = np.zeros(60, dtype=int)
    selected = select_strategy(
        "ou25",
        selection_probabilities,
        selection_odds,
        selection_outcomes,
        min_bets=20,
        edge_thresholds=(0.0,),
        confidence_thresholds=(None,),
        odds_bands=((1.1, 10.0),),
    )["selected"]

    test_probabilities = {"model": np.tile([[0.7, 0.3]], (10, 1))}
    test_odds = {"book": np.tile([[1.8, 3.0]], (10, 1))}
    test_outcomes = np.array([0] * 6 + [1] * 4)
    result = apply_selected_strategy(selected, test_probabilities, test_odds, test_outcomes)

    assert result["status"] == "tested"
    assert result["spec"] == selected["spec"]
    assert result["bets"] == 10


def test_selection_keeps_specs_that_match_inner_bets_but_diverge_out_of_sample():
    selection_probabilities = {"model": np.tile([[0.70, 0.30]], (60, 1))}
    selection_odds = {"book": np.tile([[1.60, 3.00]], (60, 1))}
    selection_outcomes = np.zeros(60, dtype=int)

    result = select_strategy(
        "ou25",
        selection_probabilities,
        selection_odds,
        selection_outcomes,
        min_bets=20,
        edge_thresholds=(None, 0.10),
        confidence_thresholds=(None,),
        odds_bands=((1.1, 10.0),),
    )

    # One family x one book x three side filters x two edge thresholds.
    # Some specs make the same decisions on this inner slice, but they remain
    # distinct policies because their thresholds can diverge in the future.
    assert result["evaluated_strategy_specs"] == 6
    reversed_result = select_strategy(
        "ou25",
        selection_probabilities,
        selection_odds,
        selection_outcomes,
        min_bets=20,
        edge_thresholds=(0.10, None),
        confidence_thresholds=(None,),
        odds_bands=((1.1, 10.0),),
    )
    assert result["selected"]["spec"] == reversed_result["selected"]["spec"]
    assert result["selected"]["spec"]["min_edge"] == 0.10

    loose = StrategySpec("ou25", "model", "book", "under", None, None, 1.1, 10.0)
    strict = StrategySpec("ou25", "model", "book", "under", 0.10, None, 1.1, 10.0)
    inner_loose = candidate_mask(loose, selection_probabilities["model"], selection_odds["book"])[0]
    inner_strict = candidate_mask(strict, selection_probabilities["model"], selection_odds["book"])[0]
    assert np.array_equal(inner_loose, inner_strict)

    outer_probabilities = np.array([[0.65, 0.35], [0.75, 0.25]])
    outer_odds = np.array([[1.60, 2.00], [1.60, 2.00]])
    outer_loose = candidate_mask(loose, outer_probabilities, outer_odds)[0]
    outer_strict = candidate_mask(strict, outer_probabilities, outer_odds)[0]
    assert outer_loose.tolist() == [True, True]
    assert outer_strict.tolist() == [False, True]
