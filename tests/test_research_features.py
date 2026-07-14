import math

import pandas as pd
import pytest

from research.features import build_feature_frame


def _match(
    date,
    home,
    away,
    home_score,
    away_score,
    *,
    season=1999,
    league="PL",
    odds=None,
    extra=None,
):
    odds = odds or (None, None, None)
    return {
        "match_date": date,
        "league_code": league,
        "league_name": "Premier League",
        "season": season,
        "home_team_name": home,
        "away_team_name": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_odds": odds[0],
        "draw_odds": odds[1],
        "away_odds": odds[2],
        "extra_data": extra or {},
    }


def test_unpriced_1990s_fixture_warms_all_later_point_in_time_state():
    matches = [
        _match("1999-08-14T15:00:00", "Alpha", "Beta", 2, 1),
        _match(
            "2005-08-14T15:00:00",
            "Alpha",
            "Gamma",
            1,
            0,
            season=2005,
            odds=(1.80, 3.40, 4.80),
            extra={"avg_over25": 1.95, "avg_under25": 1.90},
        ),
    ]

    frame = build_feature_frame(matches)

    assert len(frame) == 2  # unpriced matches remain available for modelling
    assert frame.attrs["point_in_time"] is True
    old, later = frame.iloc[0], frame.iloc[1]
    assert old["odds_1x2_primary_available"] == 0
    assert old["odds_ou25_primary_available"] == 0
    assert later["odds_1x2_primary_available"] == 1
    assert later["odds_ou25_primary_available"] == 1
    assert later["home_overall_matches_5"] == 1
    assert later["home_overall_goals_for_5"] == pytest.approx(2.0)
    assert later["home_venue_goals_against_20"] == pytest.approx(1.0)
    assert later["league_history_matches"] == 1
    assert later["home_elo"] > 1500.0
    assert later["home_rest_days"] > 2_000
    assert math.isfinite(later["home_overall_goals_for_shrunk_20"])


def test_same_kickoff_is_a_single_snapshot_before_any_result_update():
    matches = [
        _match("2000-01-01T12:00:00", "Alpha", "Beta", 1, 0, season=1999),
        # Deliberately repeat Alpha to make any within-timestamp leakage visible.
        _match("2000-02-01T15:00:00", "Alpha", "Gamma", 4, 0, season=1999),
        _match("2000-02-01T15:00:00", "Alpha", "Delta", 0, 2, season=1999),
        _match("2000-02-02T15:00:00", "Alpha", "Epsilon", 1, 1, season=1999),
    ]

    frame = build_feature_frame(matches)
    simultaneous = frame.iloc[1:3]
    after = frame.iloc[3]

    assert simultaneous["league_history_matches"].tolist() == [1, 1]
    assert simultaneous["home_overall_matches_5"].tolist() == [1, 1]
    assert simultaneous["home_overall_form_ppg_5"].tolist() == [3.0, 3.0]
    assert simultaneous["home_elo"].nunique() == 1
    assert after["league_history_matches"] == 3
    assert after["home_overall_matches_5"] == 3
    assert after["home_overall_goals_for_5"] == pytest.approx(5 / 3)


def test_team_history_survives_promotion_without_cross_country_name_collision():
    matches = [
        _match("2000-01-01T12:00:00", "United", "Town", 2, 0, league="PL"),
        _match("2000-02-01T12:00:00", "United", "Ciudad", 1, 1, league="PD"),
        _match("2000-03-01T12:00:00", "United", "Rovers", 1, 0, league="ELC"),
    ]

    frame = build_feature_frame(matches)

    # Spain's identically named club starts clean; Championship retains the
    # English club's Premier League state after relegation/promotion.
    assert frame.iloc[1]["home_overall_matches_5"] == 0
    assert frame.iloc[2]["home_overall_matches_5"] == 1
    assert frame.iloc[2]["home_overall_goals_for_5"] == pytest.approx(2.0)


def test_quote_bases_are_complete_and_never_mix_sources():
    extra = {
        # Incomplete Bet365 sets must be represented as unavailable.
        "b365_home": 1.80,
        "b365_draw": 3.50,
        "avg_home_odds": 1.85,
        "avg_draw_odds": 3.55,
        "avg_away_odds": 4.60,
        "max_home_odds": 1.90,
        "max_draw_odds": 3.65,
        "max_away_odds": 4.80,
        "b365_close_home": 1.75,
        "b365_close_draw": 3.60,
        "pinnacle_close_home": 1.77,
        "pinnacle_close_draw": 3.61,
        "pinnacle_close_away": 4.88,
        "avg_close_home_odds": 1.78,
        "avg_close_draw_odds": 3.62,
        "avg_close_away_odds": 4.90,
        "b365_over25": 1.92,
        "pinnacle_over25": 1.96,
        "pinnacle_under25": 1.94,
        "avg_over25": 1.95,
        "avg_under25": 1.91,
        "max_over25": 2.00,
        "max_under25": 1.98,
        "avg_close_over25": 1.90,
        "avg_close_under25": 1.96,
        "pinnacle_close_over25": 1.91,
        "pinnacle_close_under25": 1.95,
    }
    match = _match(
        "2024-08-01T19:00:00",
        "Alpha",
        "Beta",
        2,
        2,
        season=2024,
        odds=(1.82, 3.60, 4.70),
        extra=extra,
    )

    row = build_feature_frame([match]).iloc[0]

    assert row["odds_1x2_b365_available"] == 0
    assert pd.isna(row["odds_1x2_b365_home"])
    assert pd.isna(row["odds_1x2_b365_draw"])
    assert pd.isna(row["odds_1x2_b365_away"])
    assert row["odds_1x2_avg_available"] == 1
    assert row["odds_1x2_close_source"] == "pinnacle_close"
    assert row["odds_ou25_b365_available"] == 0
    assert row["odds_ou25_pinnacle_available"] == 1
    assert row["odds_ou25_pinnacle_source"] == "pinnacle_open"
    assert row["odds_ou25_primary_source"] == "market_average_open"
    assert row["odds_ou25_close_source"] == "pinnacle_close"
    assert sum(
        row[f"market_1x2_primary_{outcome}_prob"]
        for outcome in ("home", "draw", "away")
    ) == pytest.approx(1.0)
    assert row["market_ou25_primary_over25_prob"] + row[
        "market_ou25_primary_under25_prob"
    ] == pytest.approx(1.0)
    assert row["poisson_home_prob"] + row["poisson_draw_prob"] + row[
        "poisson_away_prob"
    ] == pytest.approx(1.0)
    assert row["poisson_over25_prob"] + row["poisson_under25_prob"] == pytest.approx(1.0)
