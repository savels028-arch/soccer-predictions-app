import hashlib
import json

import pytest

from research.edge_atlas import (
    STRATEGY_IDS,
    build_edge_atlas,
    validate_public_source,
    write_edge_atlas,
)


def _match(
    date,
    home,
    away,
    home_score,
    away_score,
    *,
    season=2020,
    league="PL",
    one_x_two=(2.0, 3.5, 4.0),
    totals=(1.9, 1.9),
    generic=(99.0, 99.0, 99.0),
):
    extra = {"max_home_odds": 250.0, "max_draw_odds": 250.0, "max_away_odds": 250.0}
    if one_x_two is not None:
        extra.update(
            {
                "b365_home": one_x_two[0],
                "b365_draw": one_x_two[1],
                "b365_away": one_x_two[2],
            }
        )
    if totals is not None:
        extra.update({"b365_over25": totals[0], "b365_under25": totals[1]})
    return {
        "match_date": date,
        "season": season,
        "league_code": league,
        "league_name": {"PL": "Premier League", "SA": "Serie A"}.get(league, league),
        "home_team_name": home,
        "away_team_name": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_odds": generic[0],
        "draw_odds": generic[1],
        "away_odds": generic[2],
        "extra_data": extra,
    }


def _league(payload, code):
    return next(league for league in payload["leagues"] if league["code"] == code)


def _season_row(scope, season):
    return next(row for row in scope["seasons"] if row["season"] == season)


def _ranked(row, strategy_id):
    return next(
        strategy
        for strategy in row["hindsight"]["ranking"]
        if strategy["strategyId"] == strategy_id
    )


def test_descriptive_map_and_roi_use_only_named_b365_with_profit_haircut():
    matches = [
        _match("2020-08-01T12:00:00", "Alpha", "Beta", 2, 1),
        _match("2020-08-02T12:00:00", "Gamma", "Delta", 0, 1),
        _match(
            "2020-08-03T12:00:00",
            "Epsilon",
            "Zeta",
            1,
            1,
            one_x_two=(2.4, 2.4, 4.0),  # tied minimum is not a unique favourite
        ),
    ]

    payload = build_edge_atlas(matches, {"dataset_id": "fixture"})
    row = _season_row(_league(payload, "PL"), 2020)
    descriptive = row["descriptive"]

    assert payload["dataset"] == {
        "id": "fixture",
        "sourceId": "fixture",
        "sourceRows": 3,
        "rejectedSourceRows": 0,
        "leagueMismatchRows": 0,
        "duplicateSourceRows": 0,
        "matches": 3,
        "startSeason": 2020,
        "endSeason": 2020,
        "leagueCount": 1,
    }
    assert descriptive["results"]["home"]["count"] == 1
    assert descriptive["results"]["draw"]["count"] == 1
    assert descriptive["results"]["away"]["count"] == 1
    assert descriptive["goals"]["over"]["0.5"]["ratePct"] == 100.0
    assert descriptive["goals"]["over"]["2.5"]["count"] == 1
    assert descriptive["goals"]["over"]["5.5"] == {"count": 0, "ratePct": 0.0}
    assert descriptive["goals"]["bothTeamsToScore"] == {
        "yes": {"count": 2, "ratePct": 66.67},
        "no": {"count": 1, "ratePct": 33.33},
    }
    assert descriptive["favourite"]["completeB365Quotes"] == 3
    assert descriptive["favourite"]["tiedPriceQuotesExcluded"] == 1

    # One Bet365 2.00 win returns +0.99 after the 1% haircut and two losses
    # return -1 each. Generic and Max odds are deliberately enormous and
    # would produce a visibly different result if either leaked into P&L.
    home = _ranked(row, "home_win")
    assert home["bets"] == 3
    assert home["profitUnits"] == pytest.approx(-1.01)
    assert home["roiPct"] == pytest.approx(-33.67)
    assert home["label"] == "hindsight"
    assert home["profitClaimAllowed"] is False
    assert payload["methodology"]["genericOddsFallback"] is False
    assert payload["methodology"]["maxOddsExecution"] is False


def test_missing_b365_quote_never_falls_back_to_generic_prices():
    match = _match(
        "2020-08-01T12:00:00",
        "Alpha",
        "Beta",
        1,
        0,
        one_x_two=None,
        totals=None,
    )
    payload = build_edge_atlas([match])
    row = _season_row(payload["global"], 2020)

    for strategy_id in STRATEGY_IDS:
        assert _ranked(row, strategy_id)["bets"] == 0
    assert row["descriptive"]["results"]["home"]["count"] == 1
    assert row["descriptive"]["results"]["home"]["b365"]["roiPct"] is None


def test_cod_uses_exact_prior_season_distribution_and_minimum_six_matches():
    matches = []
    # With B365 prices (4, 4, 2), Alpha's normalized probability of losing is
    # 0.5. Six actual losses therefore put zero points at the extreme low end
    # of the exact points distribution. No earlier match is COD-eligible.
    for index in range(6):
        matches.append(
            _match(
                f"2020-08-{index + 1:02d}T12:00:00",
                "Alpha",
                f"Opponent {index}",
                0,
                1,
                one_x_two=(4.0, 4.0, 2.0),
                totals=None,
            )
        )
    matches.append(
        _match(
            "2020-08-07T12:00:00",
            "Alpha",
            "Final Opponent",
            2,
            0,
            one_x_two=(4.0, 4.0, 2.0),
            totals=None,
        )
    )

    payload = build_edge_atlas(matches)
    row = _season_row(payload["global"], 2020)
    strict = _ranked(row, "cod_home_lte_0_125")

    assert strict["bets"] == 1
    assert strict["wins"] == 1
    assert strict["profitUnits"] == pytest.approx(2.97)
    assert payload["methodology"]["cod"]["distribution"] == "exact point-total convolution"
    assert payload["methodology"]["cod"]["minimumPriorMatches"] == 6


def test_inverse_cod_can_back_the_opponent_without_future_leakage():
    matches = [
        _match(
            f"2020-08-{index + 1:02d}T12:00:00",
            f"Opponent {index}",
            "Alpha",
            0,
            1,
            one_x_two=(2.0, 4.0, 4.0),
            totals=None,
        )
        for index in range(6)
    ]
    matches.append(
        _match(
            "2020-08-07T12:00:00",
            "Final Opponent",
            "Alpha",
            2,
            0,
            one_x_two=(2.0, 4.0, 4.0),
            totals=None,
        )
    )

    payload = build_edge_atlas(matches)
    row = _season_row(payload["global"], 2020)
    inverse = _ranked(row, "cod_away_gte_0_875_back_home")

    assert inverse["bets"] == 1
    assert inverse["wins"] == 1
    assert inverse["profitUnits"] == pytest.approx(0.99)


def test_cod_does_not_leak_results_between_matches_at_the_same_kickoff():
    matches = [
        _match(
            f"2020-08-{index + 1:02d}T12:00:00",
            "Alpha",
            f"Opponent {index}",
            0,
            1,
            one_x_two=(4.0, 4.0, 2.0),
            totals=None,
        )
        for index in range(5)
    ]
    # Both matches see only five prior quoted Alpha matches. The first result
    # must not make the second match eligible at the same kickoff.
    matches.extend(
        [
            _match(
                "2020-08-06T12:00:00",
                "Alpha",
                "Sixth Opponent",
                0,
                1,
                one_x_two=(4.0, 4.0, 2.0),
                totals=None,
            ),
            _match(
                "2020-08-06T12:00:00",
                "Alpha",
                "Seventh Opponent",
                2,
                0,
                one_x_two=(4.0, 4.0, 2.0),
                totals=None,
            ),
        ]
    )

    row = _season_row(build_edge_atlas(matches)["global"], 2020)

    assert _ranked(row, "cod_home_lte_0_375")["bets"] == 0


def test_direct_h2h_favourite_agreement_uses_prior_direction_and_fixed_thresholds():
    matches = []
    # Six home wins and four away wins produce a unique H mode at exactly 60%.
    for index in range(10):
        home_won = index < 6
        matches.append(
            _match(
                f"{2010 + index}-08-01T12:00:00",
                "Alpha",
                "Beta",
                1 if home_won else 0,
                0 if home_won else 1,
                season=2010 + index,
                one_x_two=(1.8, 3.5, 4.5),
                totals=None,
            )
        )
    matches.append(
        _match(
            "2020-08-01T12:00:00",
            "Alpha",
            "Beta",
            1,
            0,
            season=2020,
            one_x_two=(1.8, 3.5, 4.5),
            totals=None,
        )
    )

    row = _season_row(build_edge_atlas(matches)["global"], 2020)
    threshold_60 = _ranked(row, "favorite_direct_h2h_agree_0_60")
    threshold_67 = _ranked(row, "favorite_direct_h2h_agree_0_67")

    assert threshold_60["bets"] == 1
    assert threshold_60["wins"] == 1
    assert threshold_60["profitUnits"] == pytest.approx(0.79)
    assert threshold_67["bets"] == 0


def test_direct_h2h_isolates_same_kickoff_and_never_uses_reverse_direction():
    matches = []
    # Reverse-direction meetings are deliberately plentiful but irrelevant.
    for index in range(12):
        matches.append(
            _match(
                f"{1995 + index}-07-01T12:00:00",
                "Beta",
                "Alpha",
                0,
                1,
                season=1995 + index,
                one_x_two=(4.5, 3.5, 1.8),
                totals=None,
            )
        )
    for index in range(9):
        matches.append(
            _match(
                f"{2007 + index}-08-01T12:00:00",
                "Alpha",
                "Beta",
                1,
                0,
                season=2007 + index,
                one_x_two=(1.8, 3.5, 4.5),
                totals=None,
            )
        )
    # Both simultaneous fixtures see only nine prior directed meetings.
    matches.extend(
        [
            _match(
                "2016-08-01T12:00:00",
                "Alpha",
                "Beta",
                1,
                0,
                season=2016,
                one_x_two=(1.8, 3.5, 4.5),
                totals=None,
            ),
            _match(
                "2016-08-01T12:00:00",
                "Alpha",
                "Beta",
                1,
                0,
                season=2016,
                one_x_two=(1.8, 3.5, 4.5),
                totals=None,
            ),
            _match(
                "2017-08-01T12:00:00",
                "Alpha",
                "Beta",
                1,
                0,
                season=2017,
                one_x_two=(1.8, 3.5, 4.5),
                totals=None,
            ),
        ]
    )

    payload = build_edge_atlas(matches)
    row_2016 = _season_row(payload["global"], 2016)
    row_2017 = _season_row(payload["global"], 2017)

    assert _ranked(row_2016, "favorite_direct_h2h_agree_0_67")["bets"] == 0
    assert _ranked(row_2017, "favorite_direct_h2h_agree_0_67")["bets"] == 1
    assert payload["methodology"]["directH2hFavouriteAgreement"]["sameKickoffIsolation"] is True


def test_walk_forward_selection_uses_prior_seasons_then_reports_losing_holdout():
    matches = []
    for season in (2020, 2021):
        for index in range(3):
            matches.append(
                _match(
                    f"{season}-08-{index + 1:02d}T12:00:00",
                    f"Home {season} {index}",
                    f"Away {season} {index}",
                    1,
                    0,
                    season=season,
                    one_x_two=(3.0, 3.5, 1.5),
                    totals=None,
                )
            )
    for index in range(3):
        matches.append(
            _match(
                f"2022-08-{index + 1:02d}T12:00:00",
                f"Holdout Home {index}",
                f"Holdout Away {index}",
                0,
                1,
                season=2022,
                one_x_two=(3.0, 3.5, 1.5),
                totals=None,
            )
        )

    payload = build_edge_atlas(
        matches,
        min_training_seasons=2,
        min_training_bets=4,
    )
    rows = payload["global"]["walkForward"]["rows"]
    holdout = next(row for row in rows if row["season"] == 2022)

    assert holdout["selectedStrategyId"] == "home_win"
    assert holdout["label"] == "walk_forward_candidate"
    assert holdout["training"]["bets"] == 6
    assert holdout["training"]["roiPct"] == pytest.approx(198.0)
    assert holdout["training"]["ci95Pct"]["lower"] > 0
    assert holdout["test"]["bets"] == 3
    assert holdout["test"]["roiPct"] == -100.0
    # The held-out loss is visible and cannot alter the already-recorded
    # training evidence or the strategy chosen for 2022.
    assert holdout["test"]["positiveLowerCi"] is False


def test_outputs_are_isolated_per_league_and_also_aggregated_globally():
    matches = [
        _match("2020-08-01T12:00:00", "Alpha", "Beta", 1, 0, league="PL"),
        _match("2020-08-01T13:00:00", "Gamma", "Delta", 0, 1, league="SA"),
    ]
    payload = build_edge_atlas(matches)

    assert [league["code"] for league in payload["leagues"]] == ["PL", "SA"]
    assert _season_row(_league(payload, "PL"), 2020)["descriptive"]["results"]["home"]["ratePct"] == 100.0
    assert _season_row(_league(payload, "SA"), 2020)["descriptive"]["results"]["away"]["ratePct"] == 100.0
    global_row = _season_row(payload["global"], 2020)
    assert global_row["descriptive"]["matches"] == 2
    assert global_row["descriptive"]["results"]["home"]["ratePct"] == 50.0
    assert global_row["descriptive"]["results"]["away"]["ratePct"] == 50.0


def test_parameter_validation_is_explicit():
    with pytest.raises(ValueError, match="min_cod_matches"):
        build_edge_atlas([], min_cod_matches=0)
    with pytest.raises(ValueError, match="min_training_bets"):
        build_edge_atlas([], min_training_bets=1)


def test_public_source_guard_rejects_partial_cache():
    matches = [_match("2020-08-01T12:00:00", "Alpha", "Beta", 1, 0)]
    with pytest.raises(RuntimeError, match="incomplete, mislabelled, or has unaccounted"):
        validate_public_source(
            matches,
            {
                "source": "data/cache/football_data_csv",
                "start_season": 2020,
                "end_season": 2020,
                "raw_rows": 1,
                "invalid_rows": 0,
                "duplicates": 0,
                "file_hashes": {},
            },
        )


def test_atomic_writer_emits_canonical_json_and_matching_checksum(tmp_path):
    payload = {"z": [2, 1], "a": "ærlig"}
    output = tmp_path / "atlas.json"

    result = write_edge_atlas(payload, output)
    content = output.read_bytes()

    assert json.loads(content) == payload
    assert content.endswith(b"\n")
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert (tmp_path / "atlas.json.sha256").read_text(encoding="ascii") == (
        f"{result['sha256']}  atlas.json\n"
    )
    assert not list(tmp_path.glob("*.tmp"))
