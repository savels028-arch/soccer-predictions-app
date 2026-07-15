import numpy as np
import pandas as pd
import pytest

from research.engine import (
    _CandidateAggregate,
    _lock_fixed_strategy,
    _materialize_bets,
    _promotion_gate,
    _same_source_closing_price,
    ResearchConfig,
    haircut_odds,
    run_nested_strategy_zoo,
    summarize_bets,
)
from research.selection import StrategySpec


def test_odds_haircut_reduces_only_profit_component():
    odds = np.array([[2.0, np.nan], [1.5, 1.0]])

    adjusted = haircut_odds(odds, 0.02)

    assert adjusted[0, 0] == pytest.approx(1.98)
    assert adjusted[1, 0] == pytest.approx(1.49)
    assert np.isnan(adjusted[0, 1])
    assert adjusted[1, 1] == 1.0


def test_summary_reports_bootstrap_stability_clv_and_haircuts():
    bets = []
    for index, (won, odds) in enumerate([(1, 2.0), (0, 2.0), (1, 2.2), (1, 1.8)]):
        bets.append(
            {
                "match_date": f"202{index}-01-01T12:00:00+00:00",
                "season": 2020 + index,
                "won": won,
                "decimal_odds": odds,
                "raw_odds": odds,
                "clv": 0.01,
            }
        )

    summary = summarize_bets(bets, bootstrap_resamples=100, seed=7)

    assert summary["bets"] == 4
    assert summary["wins"] == 3
    assert summary["profit"] == pytest.approx(2.0)
    assert summary["stability"]["n_seasons"] == 4
    assert summary["closing_line"]["bets_with_close"] == 4
    assert summary["closing_line"]["coverage_rate"] == 1.0
    assert summary["odds_haircut_scenarios"]["haircut_0pct"] > summary["odds_haircut_scenarios"]["haircut_2pct"]


@pytest.mark.parametrize(
    ("opening", "closing", "expected"),
    [
        ("bet365_open", "bet365_close", 1.95),
        ("pinnacle_open", "pinnacle_close", 1.95),
        ("market_average_open", "market_average_close", 1.95),
        ("market_max_open", "market_max_close", 1.95),
        ("pinnacle_open", "bet365_close", None),
        ("market_average_open", "market_max_close", None),
        ("normalized_primary", "bet365_close", None),
    ],
)
def test_clv_requires_same_known_quote_source(opening, closing, expected):
    assert _same_source_closing_price(opening, closing, 1.95) == expected


def test_materialized_bet_drops_cross_source_close_instead_of_reporting_fake_clv():
    selected = {
        "spec": {
            "market": "ou25",
            "family": "model",
            "odds_basis": "pinnacle",
            "side": "under",
            "min_edge": None,
            "min_confidence": None,
            "min_odds": 1.2,
            "max_odds": 5.0,
        }
    }
    test_frame = pd.DataFrame(
        [
            {
                "match_id": "m1",
                "match_date": "2025-08-01T12:00:00+00:00",
                "league_code": "PL",
                "home_team": "Alpha",
                "away_team": "Beta",
                "odds_ou25_pinnacle_source": "pinnacle_open",
                "odds_ou25_close_source": "bet365_close",
            }
        ]
    )
    probabilities = {"model": np.array([[0.70, 0.30]])}
    raw_odds = {
        "pinnacle": np.array([[1.80, 2.20]]),
        "close": np.array([[1.70, 2.30]]),
    }

    bets = _materialize_bets(
        selected,
        probabilities,
        raw_odds,
        raw_odds,
        np.array([0]),
        test_frame,
        market="ou25",
        track="locked_executable",
        test_season=2025,
    )

    assert len(bets) == 1
    assert bets[0]["opening_odds_source"] == "pinnacle_open"
    assert bets[0]["observed_closing_odds_source"] == "bet365_close"
    assert bets[0]["closing_odds"] is None
    assert bets[0]["clv"] is None


def _passing_gate_summary(*, closes=250, coverage=0.625, seasons=3, mean_clv=0.01):
    return {
        "bets": 400,
        "roi": 0.08,
        "bootstrap": {
            "ci_lower": 0.01,
            "probability_roi_positive": 0.99,
        },
        "stability": {
            "n_seasons": seasons,
            "positive_season_rate": 2 / 3 if seasons >= 3 else 1.0,
        },
        "closing_line": {
            "bets_with_close": closes,
            "coverage_rate": coverage,
            "mean_clv": mean_clv,
        },
    }


def test_promotion_gate_fails_closed_without_enough_same_source_clv():
    no_close = _promotion_gate(_passing_gate_summary(closes=0, coverage=0.0))
    ninety_nine = _promotion_gate(_passing_gate_summary(closes=99, coverage=99 / 400))
    one_hundred_low_coverage = _promotion_gate(
        _passing_gate_summary(closes=100, coverage=0.25)
    )
    enough = _promotion_gate(_passing_gate_summary(closes=200, coverage=0.50))

    assert "fewer_than_100_same_source_closing_observations" in no_close["reasons"]
    assert "fewer_than_100_same_source_closing_observations" in ninety_nine["reasons"]
    assert "fewer_than_100_same_source_closing_observations" not in one_hundred_low_coverage["reasons"]
    assert "same_source_closing_coverage_below_50pct" in one_hundred_low_coverage["reasons"]
    assert enough == {"passed": True, "reasons": []}


def test_promotion_gate_rejects_one_lucky_holdout_season():
    gate = _promotion_gate(_passing_gate_summary(seasons=1))

    assert gate["passed"] is False
    assert "fewer_than_3_outer_test_seasons_with_bets" in gate["reasons"]


def test_strategy_zoo_rejects_lock_config_without_three_holdout_seasons():
    config = ResearchConfig(
        first_test_season=2012,
        last_test_season=2025,
        markets=("ou25",),
        policy_lock_season=2024,
    )

    with pytest.raises(ValueError, match="at least three configured outer holdout seasons"):
        run_nested_strategy_zoo(pd.DataFrame(), config)


def test_fixed_policy_keeps_short_history_candidate_diagnostic_only():
    spec = StrategySpec(
        market="ou25",
        family="market__raw",
        odds_basis="b365",
        side="over",
        min_edge=0.03,
        min_confidence=0.55,
        min_odds=1.5,
        max_odds=2.5,
    )
    aggregate = _CandidateAggregate(
        bets=180,
        wins=100,
        profit=12.0,
        profit_squares=150.0,
        season_profit={2020: 3.0, 2021: 4.0, 2022: 5.0},
        season_bets={2020: 60, 2021: 60, 2022: 60},
    )

    locked = _lock_fixed_strategy({spec: aggregate})

    assert locked["selected"] is None
    assert locked["diagnostic_selected"]["spec"] == spec.__dict__
    assert locked["diagnostic_selected"]["eligible"] is False
    assert locked["diagnostic_selected"]["eligibility_reasons"] == [
        "fewer_than_300_development_bets",
        "fewer_than_5_development_seasons",
    ]
