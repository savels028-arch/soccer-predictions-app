import math

import pytest

from research.metrics import (
    block_bootstrap_roi,
    brier_score,
    flat_stake_metrics,
    log_loss,
    max_drawdown,
    probability_metrics,
    season_stability,
    settle_flat_stake,
)


def test_flat_stake_metrics_include_profit_roi_hit_rate_and_drawdown():
    profits = settle_flat_stake([1, 0, 1, 0], [2.0, 3.0, 1.5, 4.0], stake=10.0)
    metrics = flat_stake_metrics([1, 0, 1, 0], [2.0, 3.0, 1.5, 4.0], stake=10.0)

    assert profits == [10.0, -10.0, 5.0, -10.0]
    assert metrics == {
        "bets": 4,
        "wins": 2,
        "hit_rate": 0.5,
        "staked": 40.0,
        "returned": 35.0,
        "profit": -5.0,
        "roi": -0.125,
        "roi_pct": -12.5,
        "max_drawdown": 15.0,
    }
    assert max_drawdown([20.0, -5.0, -30.0, 10.0, -2.0]) == 35.0


def test_probability_scores_support_binary_and_three_way_forecasts():
    binary_probabilities = [0.8, 0.25]
    binary_outcomes = [1, 0]
    assert brier_score(binary_probabilities, binary_outcomes) == pytest.approx(0.05125)
    assert log_loss(binary_probabilities, binary_outcomes) == pytest.approx(
        -(math.log(0.8) + math.log(0.75)) / 2.0
    )

    probabilities = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]]
    outcomes = [0, 2]
    assert probability_metrics(probabilities, outcomes) == pytest.approx(
        {
            "brier_score": 0.2,
            "log_loss": -(math.log(0.7) + math.log(0.6)) / 2.0,
        }
    )


def test_probability_scores_reject_malformed_forecasts():
    with pytest.raises(ValueError, match="sum to 1"):
        brier_score([[0.6, 0.3, 0.2]], [0])
    with pytest.raises(ValueError, match="same length"):
        log_loss([0.5], [0, 1])


def test_season_stability_reports_positive_rate_and_dispersion():
    result = season_stability(
        [1.0, -0.5, -1.0, -1.0, 2.0, -1.0],
        ["2022", "2022", "2023", "2023", "2024", "2024"],
    )

    assert result["n_seasons"] == 3
    assert result["profitable_seasons"] == 2
    assert result["positive_season_rate"] == pytest.approx(2.0 / 3.0)
    assert result["mean_season_roi"] == pytest.approx(-1.0 / 12.0)
    assert result["median_season_roi"] == 0.25
    assert result["worst_season_roi"] == -1.0
    assert result["best_season_roi"] == 0.5
    assert result["season_roi_std"] > 0.0
    assert result["by_season"]["2024"]["roi_pct"] == 50.0


def test_block_bootstrap_is_deterministic_and_reports_positive_probability():
    profits = [1.4, -1.0, 0.9, -1.0, 1.8, -1.0, 0.7, -1.0]
    first = block_bootstrap_roi(
        profits,
        block_size=2,
        n_resamples=500,
        confidence=0.9,
        seed=42,
    )
    second = block_bootstrap_roi(
        profits,
        block_size=2,
        n_resamples=500,
        confidence=0.9,
        seed=42,
    )

    assert first == second
    assert first["roi"] == pytest.approx(sum(profits) / len(profits))
    assert first["ci_lower"] <= first["bootstrap_mean_roi"] <= first["ci_upper"]
    assert 0.0 <= first["probability_roi_positive"] <= 1.0

    certain = block_bootstrap_roi([0.2, 0.4, 0.1], n_resamples=50, seed=7)
    assert certain["ci_lower"] > 0.0
    assert certain["probability_roi_positive"] == 1.0


def test_metrics_reject_invalid_prices_and_empty_bootstrap():
    with pytest.raises(ValueError, match="greater than 1"):
        flat_stake_metrics([1], [1.0])
    with pytest.raises(ValueError, match="must not be empty"):
        block_bootstrap_roi([])
