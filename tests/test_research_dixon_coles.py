import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skellam

from research.engine import _fixed_family_allowed
from research.models import DixonColesModel, dixon_coles_probabilities


def _synthetic_training_frame() -> pd.DataFrame:
    fixtures = []
    scores = (
        ("ELC", "Promoted FC", "Alpha", 4, 0),
        ("ELC", "Beta", "Promoted FC", 0, 3),
        ("ELC", "Promoted FC", "Beta", 3, 0),
        ("ELC", "Alpha", "Promoted FC", 1, 3),
        ("PL", "Premier A", "Premier B", 2, 1),
        ("PL", "Premier B", "Premier A", 1, 1),
    )
    for round_number in range(5):
        for offset, (league, home, away, home_score, away_score) in enumerate(scores):
            fixtures.append(
                {
                    "match_date": pd.Timestamp("2020-08-01", tz="UTC")
                    + pd.Timedelta(days=7 * (round_number * len(scores) + offset)),
                    "league_code": league,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            )
    return pd.DataFrame(fixtures)


def test_dixon_coles_known_low_score_correction_preserves_probability_mass():
    home_rate = 1.40
    away_rate = 0.90
    rho = -0.10

    one_x_two, over_under = dixon_coles_probabilities(home_rate, away_rate, rho)

    independent_away = skellam.cdf(-1, home_rate, away_rate)
    independent_home = 1.0 - skellam.cdf(0, home_rate, away_rate)
    independent_draw = 1.0 - independent_home - independent_away
    delta = home_rate * away_rate * rho * math.exp(-(home_rate + away_rate))
    expected_1x2 = np.array(
        [independent_home + delta, independent_draw - 2.0 * delta, independent_away + delta]
    )
    total_rate = home_rate + away_rate
    independent_under = math.exp(-total_rate) * (
        1.0 + total_rate + 0.5 * total_rate**2
    )

    np.testing.assert_allclose(one_x_two[0], expected_1x2, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(over_under[0], [independent_under, 1.0 - independent_under])
    assert one_x_two[0].sum() == pytest.approx(1.0)
    assert over_under[0].sum() == pytest.approx(1.0)
    assert np.all(one_x_two >= 0.0)
    assert np.all(over_under >= 0.0)


def test_training_cutoff_and_promoted_or_unseen_team_fallback_are_safe():
    training = _synthetic_training_frame()
    model = DixonColesModel(max_iter=100).fit(training)
    future_date = training["match_date"].max() + pd.Timedelta(days=7)

    assert model.training_cutoff_ == training["match_date"].max()
    assert "promoted fc" in model.groups_["ENG"].attack

    targets = pd.DataFrame(
        {
            "match_date": [future_date, future_date],
            "league_code": ["PL", "PL"],
            "home_team": ["Promoted FC", "Never Seen Home"],
            "away_team": ["Never Seen Away", "Another New Team"],
        }
    )
    rates = model.predict_score_rates(targets)
    english_group = model.groups_["ENG"]
    expected_unseen = np.array(
        [
            math.exp(english_group.intercept + english_group.home_advantage),
            math.exp(english_group.intercept),
        ]
    )

    # An ELC team keeps its learned identity after promotion to PL. Completely
    # unseen teams get neutral attack/defence at the fitted English baseline.
    assert rates[0, 0] > rates[1, 0]
    np.testing.assert_allclose(rates[1], expected_unseen, rtol=0.0, atol=1e-12)

    in_sample_target = targets.iloc[[0]].copy()
    in_sample_target["match_date"] = model.training_cutoff_
    with pytest.raises(ValueError, match="strictly after the training cutoff"):
        model.predict_score_rates(in_sample_target)


def test_dixon_coles_fit_and_predictions_are_deterministic_across_input_order():
    training = _synthetic_training_frame()
    target = pd.DataFrame(
        {
            "match_date": [training["match_date"].max() + pd.Timedelta(days=14)],
            "league_code": ["PL"],
            "home_team": ["Promoted FC"],
            "away_team": ["Premier A"],
        }
    )

    first = DixonColesModel(max_iter=100).fit(training)
    second = DixonColesModel(max_iter=100).fit(
        training.sample(frac=1.0, random_state=20260714).reset_index(drop=True)
    )

    np.testing.assert_allclose(
        first.predict_score_rates(target),
        second.predict_score_rates(target),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        first.predict_proba(target, "1x2"),
        second.predict_proba(target, "1x2"),
        rtol=0.0,
        atol=1e-12,
    )


def test_dixon_coles_can_enter_the_locked_policy_evaluation():
    assert _fixed_family_allowed("dixon_coles")
    assert _fixed_family_allowed("dixon_coles__temperature")
