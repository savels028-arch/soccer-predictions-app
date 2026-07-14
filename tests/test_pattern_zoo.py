import copy
from datetime import datetime
import hashlib
import json

import pytest

from research import run_pattern_zoo as pattern_zoo_runner
from research.dataset import LATEST_COMPLETE_SEASON
from research.pattern_zoo import (
    StrategyZooValidationError,
    _Event,
    _summary,
    build_strategy_zoo,
    load_strategy_zoo,
    validate_strategy_zoo,
)
from research.run_pattern_zoo import (
    _guard_against_regression,
    _missing_canonical_files,
    build_parser,
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
    one_x_two=(1.8, 3.5, 4.5),
    totals=(1.9, 1.9),
):
    extra = {}
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
        "league_code": league,
        "league_name": "Test league",
        "season": season,
        "home_team_name": home,
        "away_team_name": away,
        "home_score": home_score,
        "away_score": away_score,
        "home_odds": one_x_two[0] if one_x_two else None,
        "draw_odds": one_x_two[1] if one_x_two else None,
        "away_odds": one_x_two[2] if one_x_two else None,
        "extra_data": extra,
    }


def _strategy(payload, identifier):
    return next(strategy for strategy in payload["strategies"] if strategy["id"] == identifier)


def _build(matches, **kwargs):
    source_seasons = [int(match["season"]) for match in matches]
    complete_through_season = kwargs.pop("complete_through_season", max(source_seasons))
    display_through_season = kwargs.pop("display_through_season", max(source_seasons))
    return build_strategy_zoo(
        matches,
        {"dataset_id": "fixture"},
        generated_at="2026-07-14T00:00:00Z",
        bootstrap_resamples=10,
        complete_through_season=complete_through_season,
        display_through_season=display_through_season,
        **kwargs,
    )


def test_directed_h2h_uses_only_prior_matches_and_isolates_same_kickoff():
    matches = [
        _match(f"2020-0{month}-01T12:00:00Z", "Alpha", "Beta", 2, 0)
        for month in range(1, 5)
    ]
    # The first simultaneous match loses and must not alter the second match's
    # pre-kickoff four-match, 100% home-win snapshot.
    matches.extend(
        [
            _match("2020-05-01T12:00:00Z", "Alpha", "Beta", 0, 1),
            _match("2020-05-01T12:00:00Z", "Alpha", "Beta", 1, 0),
        ]
    )

    payload = _build(matches)
    strategy = _strategy(payload, "directed_h2h_dominance")

    assert strategy["overall"]["opportunities"] == 2
    assert strategy["overall"]["hits"] == 1
    assert strategy["overall"]["bets"] == 2
    assert strategy["overall"]["profitUnits"] == pytest.approx(-0.21)
    assert payload["methodology"]["sameKickoffIsolation"] is True


def test_missing_or_incomplete_opening_quote_never_fabricates_profit():
    matches = [
        _match(f"2020-0{month}-01T12:00:00Z", "Alpha", "Beta", 2, 0, one_x_two=None)
        for month in range(1, 6)
    ]
    incomplete = _match("2020-06-01T12:00:00Z", "Alpha", "Beta", 1, 0)
    incomplete["extra_data"].pop("b365_away")
    matches.append(incomplete)

    strategy = _strategy(_build(matches), "directed_h2h_dominance")

    assert strategy["overall"]["opportunities"] == 2
    assert strategy["overall"]["bets"] == 0
    assert strategy["overall"]["profitUnits"] is None
    assert strategy["overall"]["roiPct"] is None
    assert strategy["status"] == "descriptive_only"


def test_team_dominance_never_relabels_a_dominant_draw_as_a_team_pick():
    matches = [
        _match(f"2020-0{month}-01T12:00:00Z", "Alpha", "Beta", 1, 1)
        for month in range(1, 7)
    ]

    strategy = _strategy(_build(matches), "pair_h2h_team_dominance")

    assert strategy["overall"]["opportunities"] == 0
    assert strategy["status"] == "unavailable"


def test_incomplete_season_is_explicitly_quarantined_from_every_metric():
    matches = [
        _match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024),
        _match("2025-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2025),
    ]

    payload = _build(
        matches,
        complete_through_season=2024,
        display_through_season=2025,
    )

    assert payload["dataset"]["evaluatedMatches"] == 1
    assert payload["dataset"]["quarantinedMatches"] == 1
    assert payload["dataset"]["completeThroughSeason"] == 2024
    favourite = _strategy(payload, "favourite_1_80_2_20")
    assert favourite["overall"]["bets"] == 1
    row_2025 = next(row for row in favourite["yearly"] if row["season"] == 2025)
    assert row_2025["available"] is False
    assert row_2025["quarantined"] is True
    assert row_2025["quarantineReason"] == "incomplete_local_snapshot"
    assert row_2025["bets"] == 0
    assert row_2025["profitUnits"] is None


def test_h2h_pairs_expose_concrete_record_without_calling_every_pair_a_derby():
    matches = []
    for index in range(10):
        alpha_home = index % 2 == 0
        matches.append(
            _match(
                f"2020-{index + 1:02d}-01T12:00:00Z",
                "Alpha" if alpha_home else "Beta",
                "Beta" if alpha_home else "Alpha",
                2 if alpha_home else 0,
                0 if alpha_home else 2,
            )
        )

    payload = _build(matches)
    pattern = payload["rivalryPatterns"][0]

    assert pattern["team"] == "Alpha"
    assert pattern["opponent"] == "Beta"
    assert pattern["dominantTeam"] == "Alpha"
    assert pattern["meetings"] == 10
    assert pattern["record"] == {"wins": 10, "draws": 0, "losses": 0}
    assert pattern["winRatePct"] == 100.0
    assert pattern["unbeatenRatePct"] == 100.0
    assert pattern["perfectWinRecord"] is True
    assert pattern["guaranteed"] is False
    assert pattern["relationshipLabel"] == "head_to_head_pair_not_verified_derby"
    assert payload["findings"]["rivalryScreen"]["perfectWinRecordCount"] == 1
    assert "ikke fremtidige garantier" in payload["findings"]["rivalryScreen"]["perfectRecordConclusion"]


def test_exact_score_is_accuracy_only_without_exact_score_odds():
    matches = [
        _match(
            f"{2000 + index // 12}-{index % 12 + 1:02d}-01T12:00:00Z",
            f"Home {index}",
            f"Away {index}",
            1,
            1 if index < 200 else 0,
            season=2000 + index // 12,
        )
        for index in range(205)
    ]

    strategy = _strategy(_build(matches), "league_exact_score_mode")

    assert strategy["overall"]["opportunities"] == 5
    assert strategy["overall"]["hits"] == 0
    assert strategy["overall"]["bets"] == 0
    assert strategy["overall"]["roiPct"] is None


def test_validator_fails_closed_on_guarantees_and_synthetic_pnl(tmp_path):
    payload = _build([_match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024)])
    guarantee = copy.deepcopy(payload)
    guarantee["strategies"][0]["guaranteed"] = True
    with pytest.raises(StrategyZooValidationError, match="must not claim"):
        validate_strategy_zoo(guarantee)

    synthetic = copy.deepcopy(payload)
    exact = next(item for item in synthetic["strategies"] if item["id"] == "league_exact_score_mode")
    exact["overall"]["profitUnits"] = 10.0
    with pytest.raises(StrategyZooValidationError, match="fabricates P&L"):
        validate_strategy_zoo(synthetic)

    path = tmp_path / "strategy-zoo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_strategy_zoo(path)["schemaVersion"] == 1

    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    assert load_strategy_zoo(path, require_checksum=True)["schemaVersion"] == 1
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(StrategyZooValidationError, match="checksum"):
        load_strategy_zoo(path, require_checksum=True)


def test_publication_rebuild_rejects_coherent_checksum_valid_forgery(tmp_path, monkeypatch):
    original = _build([
        _match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024)
    ])
    forged = copy.deepcopy(original)
    forged["dataset"]["datasetId"] = "coherent-forgery"
    validate_strategy_zoo(forged)

    path = tmp_path / "strategy-zoo.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    assert load_strategy_zoo(path, require_checksum=True)["dataset"]["datasetId"] == "coherent-forgery"

    monkeypatch.setattr(pattern_zoo_runner, "MIN_SEASON", 2024)
    monkeypatch.setattr(pattern_zoo_runner, "MAX_SEASON", 2024)
    monkeypatch.setattr(pattern_zoo_runner, "LATEST_COMPLETE_SEASON", 2024)
    monkeypatch.setattr(
        pattern_zoo_runner,
        "build_from_canonical_cache",
        lambda **_kwargs: original,
    )
    with pytest.raises(StrategyZooValidationError, match="not reproducible"):
        pattern_zoo_runner.verify_artifact_against_canonical(path)


def test_publication_rejects_a_reproducible_but_truncated_public_range(tmp_path):
    payload = _build([
        _match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024)
    ])
    path = tmp_path / "strategy-zoo.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )

    with pytest.raises(StrategyZooValidationError, match="complete public canonical range"):
        pattern_zoo_runner.verify_artifact_against_canonical(path)


def test_validator_distinguishes_win_rate_patterns_from_betting_edges():
    matches = []
    for index in range(12):
        season = 2014 + index // 2
        matches.append(
            _match(f"{season}-{index % 2 + 1:02d}-01T12:00:00Z", "Alpha", "Beta", 2, 0, season=season)
        )
    for index in range(10):
        season = 2020 + index // 2
        matches.append(
            _match(f"{season}-{index % 2 + 1:02d}-01T12:00:00Z", "Alpha", "Beta", 2, 0, season=season)
        )
    payload = _build(matches)
    h2h = payload["findings"]["h2hValidation"]

    validated = validate_strategy_zoo(payload)
    assert validated["findings"]["h2hValidation"]["confirmedWinRatePatterns"] == 1
    assert validated["findings"]["h2hValidation"]["confirmedEdges"] == 0
    assert validated["findings"]["h2hValidation"]["confirmedBettingEdges"] == 0

    unsafe = copy.deepcopy(payload)
    unsafe["findings"]["h2hValidation"]["confirmedBettingEdges"] = 1
    with pytest.raises(StrategyZooValidationError, match="betting edge"):
        validate_strategy_zoo(unsafe)

    forged_significance = copy.deepcopy(payload)
    forged_significance["findings"]["h2hValidation"]["candidateTests"][0]["pValue"] = 0.0
    forged_significance["findings"]["h2hValidation"]["candidateTests"][0]["qValue"] = 0.0
    with pytest.raises(StrategyZooValidationError, match="p-value"):
        validate_strategy_zoo(forged_significance)


def test_validator_rejects_inconsistent_status_and_yearly_pnl():
    payload = _build([_match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024)])
    status = copy.deepcopy(payload)
    status["strategies"][0]["status"] = "confirmed_guaranteed_profit"
    with pytest.raises(StrategyZooValidationError, match="unsupported status"):
        validate_strategy_zoo(status)

    synthetic = copy.deepcopy(payload)
    strategy = next(item for item in synthetic["strategies"] if item["id"] == "favourite_1_80_2_20")
    row = next(item for item in strategy["yearly"] if item["season"] == 2024)
    row["profitUnits"] = 999.0
    with pytest.raises(StrategyZooValidationError, match="profit"):
        validate_strategy_zoo(synthetic)

    interval = copy.deepcopy(payload)
    strategy = next(item for item in interval["strategies"] if item["id"] == "favourite_1_80_2_20")
    strategy["overall"]["roiCi95Pct"] = [-20.0, -10.0]
    with pytest.raises(StrategyZooValidationError, match="ROI interval"):
        validate_strategy_zoo(interval)

    seasons = copy.deepcopy(payload)
    strategy = next(item for item in seasons["strategies"] if item["id"] == "favourite_1_80_2_20")
    strategy["overall"]["activeSeasons"] = 2
    with pytest.raises(StrategyZooValidationError, match="season counts"):
        validate_strategy_zoo(seasons)


def test_drawdown_batches_simultaneous_settlements():
    kickoff = "2024-08-01T12:00:00+00:00"

    events = [
        _Event(datetime.fromisoformat(kickoff), 2024, False, "A", "H", 2.0, 2.0),
        _Event(datetime.fromisoformat(kickoff), 2024, True, "H", "H", 2.02, 1.9998),
    ]

    summary = _summary(events, "same-kickoff", bootstrap_resamples=10)

    assert summary["profitUnits"] == 0.0
    assert summary["maxDrawdownUnits"] == 0.0


def test_cli_defaults_cover_the_latest_complete_season():
    args = build_parser().parse_args([])

    assert args.complete_through_season == LATEST_COMPLETE_SEASON == 2025
    assert args.source_end_season == 2025
    assert args.display_through_season == 2025


def test_builder_fails_closed_on_empty_or_mismatched_source_ranges():
    with pytest.raises(ValueError, match="at least one valid source match"):
        build_strategy_zoo([], {"dataset_id": "empty"})

    matches = [_match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024)]
    with pytest.raises(ValueError, match="has no source matches"):
        build_strategy_zoo(matches, {"dataset_id": "fixture"}, complete_through_season=2025)
    with pytest.raises(ValueError, match="latest loaded source season"):
        build_strategy_zoo(
            matches,
            {"dataset_id": "fixture"},
            complete_through_season=2024,
            display_through_season=2025,
        )


def test_atomic_publication_guard_rejects_a_shrinking_dataset(tmp_path):
    previous = _build([
        _match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024),
        _match("2024-08-02T12:00:00Z", "Gamma", "Delta", 1, 0, season=2024),
    ])
    path = tmp_path / "strategy-zoo.json"
    path.write_text(json.dumps(previous), encoding="utf-8")
    smaller = _build([
        _match("2024-08-01T12:00:00Z", "Alpha", "Beta", 1, 0, season=2024),
    ])

    with pytest.raises(RuntimeError, match="smaller evaluated dataset"):
        _guard_against_regression(path, smaller)


def test_canonical_coverage_guard_detects_earliest_and_intermediate_gaps():
    complete = {
        f"{season % 100:02d}{(season + 1) % 100:02d}_{league}.csv": "hash"
        for season in range(1993, 1996)
        for league in ("E0", "E1", "SP1", "D1", "D2", "I1", "F1", "N1", "P1", "B1")
    }
    complete.pop("9394_B1.csv")
    complete.pop("9495_B1.csv")
    assert not _missing_canonical_files(complete, start_season=1993, end_season=1995)

    complete.pop("9394_E0.csv")
    complete.pop("9495_D1.csv")
    assert _missing_canonical_files(complete, start_season=1993, end_season=1995) == {
        "9394_E0.csv",
        "9495_D1.csv",
    }
