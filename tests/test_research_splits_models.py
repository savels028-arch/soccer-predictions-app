import numpy as np
import pandas as pd
import pytest

from research.models import baseline_probabilities, infer_feature_columns, odds_matrices
from research.splits import nested_season_folds


def test_nested_fold_keeps_train_calibration_selection_and_test_disjoint():
    rows = []
    for season in range(2000, 2007):
        for day in range(1, 11):
            rows.append({"season": season, "match_date": f"{season}-08-{day:02d}"})
    frame = pd.DataFrame(rows)

    folds = nested_season_folds(
        frame,
        first_test_season=2006,
        min_train_seasons=5,
        min_calibration_rows=5,
        min_selection_rows=5,
    )

    assert len(folds) == 1
    fold = folds[0]
    assert fold.train_seasons == (2000, 2001, 2002, 2003, 2004)
    assert not np.any(fold.train_mask & fold.calibration_mask)
    assert not np.any(fold.calibration_mask & fold.selection_mask)
    assert not np.any(fold.selection_mask & fold.test_mask)
    assert fold.calibration_mask.sum() == 5
    assert fold.selection_mask.sum() == 5
    assert fold.test_mask.sum() == 10


def test_nested_fold_rejects_mislabeled_season_that_breaks_real_time_order():
    rows = []
    for season in range(2000, 2007):
        for day in range(1, 11):
            rows.append({"season": season, "match_date": f"{season}-08-{day:02d}"})
    # A bad source row says season 2004 but occurs after the 2006 test season.
    rows[40]["match_date"] = "2007-01-01"

    with pytest.raises(ValueError, match="violate chronological"):
        nested_season_folds(
            pd.DataFrame(rows),
            first_test_season=2006,
            min_train_seasons=5,
            min_calibration_rows=5,
            min_selection_rows=5,
        )


def test_baselines_and_odds_use_under_then_over_class_order():
    frame = pd.DataFrame(
        {
            "league_code": ["PL"],
            "poisson_under25_prob": [0.60],
            "poisson_over25_prob": [0.40],
            "league_over25_rate": [0.45],
            "market_ou25_avg_under25_prob": [0.55],
            "market_ou25_avg_over25_prob": [0.45],
            "odds_ou25_avg_under25": [1.90],
            "odds_ou25_avg_over25": [2.00],
        }
    )

    probabilities = baseline_probabilities(frame, "ou25")
    odds = odds_matrices(frame, "ou25")

    assert probabilities["poisson"].tolist() == [[0.6, 0.4]]
    assert probabilities["league_prior"].tolist() == [[0.55, 0.45]]
    assert odds["avg"].tolist() == [[1.9, 2.0]]


def test_feature_inference_excludes_prices_targets_and_closing_market_data():
    frame = pd.DataFrame(
        {
            "league_code": ["PL"],
            "target_over25": [1],
            "home_score": [2],
            "elo_difference": [25.0],
            "odds_ou25_close_over25": [1.8],
            "market_ou25_close_over25_prob": [0.55],
            "market_ou25_primary_over25_prob": [0.54],
        }
    )

    columns = infer_feature_columns(frame, "ou25")

    assert columns.football_numeric == ("elo_difference",)
    assert columns.market_numeric == ("elo_difference", "market_ou25_primary_over25_prob")
