import json
import sys

import pytest

from research import run_asian_handicap_benchmark
from research.asian_handicap import (
    asian_handicap_metrics,
    fixed_blind_asian_benchmark,
    settle_asian_handicap,
    split_asian_handicap_line,
    valid_asian_handicap_line,
)
from research.dataset import _coverage_flags
from research.features import build_feature_frame, extract_asian_handicap_quotes
from research.models import infer_feature_columns
from src.api.csv_football_client import FootballDataCSVClient


def _match(*, home_score=1, away_score=0, extra=None):
    return {
        "match_date": "2024-08-01T19:00:00Z",
        "league_code": "PL",
        "league_name": "Premier League",
        "season": 2024,
        "home_team_name": "Alpha",
        "away_team_name": "Beta",
        "home_score": home_score,
        "away_score": away_score,
        "home_odds": 2.0,
        "draw_odds": 3.5,
        "away_odds": 4.0,
        "extra_data": extra or {},
    }


@pytest.mark.parametrize(
    ("line", "legs"),
    [
        (0, (0.0,)),
        (-0.5, (-0.5,)),
        (-0.25, (-0.5, 0.0)),
        (0.25, (0.0, 0.5)),
        (-0.75, (-1.0, -0.5)),
        (0.75, (0.5, 1.0)),
        (-1.75, (-2.0, -1.5)),
    ],
)
def test_asian_line_split_uses_adjacent_half_goal_lines(line, legs):
    assert split_asian_handicap_line(line) == legs


@pytest.mark.parametrize("bad", [None, "", float("nan"), float("inf"), -1.2, 0.758, -225])
def test_malformed_source_lines_are_rejected_not_rounded(bad):
    assert valid_asian_handicap_line(bad) is None


@pytest.mark.parametrize(
    ("home", "away", "line", "side", "odds", "outcome", "profit"),
    [
        (1, 1, -0.5, "home", 1.90, "loss", -1.0),
        (1, 1, 0.5, "home", 1.90, "win", 0.90),
        (1, 1, 0.0, "home", 1.90, "push", 0.0),
        (1, 1, -0.25, "home", 2.00, "half_loss", -0.5),
        (1, 1, 0.25, "home", 2.00, "half_win", 0.5),
        (2, 1, -0.75, "home", 2.00, "half_win", 0.5),
        (2, 1, -1.25, "home", 2.00, "half_loss", -0.5),
        # AHh belongs to home; away automatically receives the inverse line.
        (1, 1, -0.25, "away", 2.00, "half_win", 0.5),
        (2, 1, -0.75, "away", 2.00, "half_loss", -0.5),
    ],
)
def test_settlement_preserves_pushes_and_half_outcomes(
    home, away, line, side, odds, outcome, profit
):
    settled = settle_asian_handicap(home, away, line, side, odds)

    assert settled.outcome == outcome
    assert settled.profit == pytest.approx(profit)
    assert settled.returned == pytest.approx(1.0 + profit)


def test_home_and_away_settlements_are_symmetric_at_even_odds():
    for quarter_units in range(-15, 13):
        line = quarter_units / 4.0
        for home_goals in range(5):
            for away_goals in range(5):
                home = settle_asian_handicap(home_goals, away_goals, line, "home", 2.0)
                away = settle_asian_handicap(home_goals, away_goals, line, "away", 2.0)
                assert home.profit + away.profit == pytest.approx(0.0)
                assert home.returned + away.returned == pytest.approx(2.0)


def test_settlement_validates_scores_odds_side_and_line():
    with pytest.raises(ValueError, match="home_goals"):
        settle_asian_handicap(1.5, 0, 0, "home", 1.9)
    with pytest.raises(ValueError, match="side"):
        settle_asian_handicap(1, 0, 0, "draw", 1.9)
    with pytest.raises(ValueError, match="decimal_odds"):
        settle_asian_handicap(1, 0, 0, "home", 1.0)
    with pytest.raises(ValueError, match="quarter-goal"):
        settle_asian_handicap(1, 0, -0.3, "home", 1.9)


def test_metrics_use_actual_return_instead_of_binary_win_coercion():
    settlements = [
        settle_asian_handicap(1, 1, 0.25, "home", 2.0),  # +0.5
        settle_asian_handicap(1, 1, -0.25, "home", 2.0),  # -0.5
        settle_asian_handicap(1, 1, 0.0, "home", 2.0),  # push
    ]

    result = asian_handicap_metrics(settlements)

    assert result["bets"] == 3
    assert result["profit"] == pytest.approx(0.0)
    assert result["roi"] == pytest.approx(0.0)
    assert result["outcomes"] == {
        "win": 0,
        "half_win": 1,
        "push": 1,
        "half_loss": 1,
        "loss": 0,
    }


def test_quote_extraction_never_mixes_price_sources_or_open_and_close_lines():
    match = _match(
        extra={
            "asian_handicap_line": -0.25,
            "b365_asian_home": 1.92,  # incomplete pair: unavailable
            "pinnacle_asian_home": 1.97,
            "pinnacle_asian_away": 1.95,
            "avg_asian_home": 1.94,
            "avg_asian_away": 1.93,
            "asian_handicap_close_line": -0.5,
            "b365_close_asian_home": 1.91,
            "b365_close_asian_away": 1.99,
        }
    )

    quotes = extract_asian_handicap_quotes(match)

    assert quotes["b365"] is None
    assert quotes["pinnacle"] == {
        "source": "pinnacle_open",
        "home": 1.97,
        "away": 1.95,
        "home_line": -0.25,
        "away_line": 0.25,
    }
    assert quotes["close"]["source"] == "bet365_close"
    assert quotes["close"]["home_line"] == -0.5
    assert quotes["close"]["away_line"] == 0.5


def test_bad_line_makes_every_open_quote_unavailable():
    quotes = extract_asian_handicap_quotes(
        _match(
            extra={
                "asian_handicap_line": -1.2,
                "b365_asian_home": 1.90,
                "b365_asian_away": 2.00,
                "avg_asian_home": 1.95,
                "avg_asian_away": 1.95,
            }
        )
    )

    assert quotes["b365"] is None
    assert quotes["avg"] is None


def test_legacy_bet365_line_stays_attached_to_its_own_prices():
    parser = FootballDataCSVClient.__new__(FootballDataCSVClient)
    normalized = parser._normalize_csv_row(
        {
            "Date": "16/08/03",
            "HomeTeam": "Alpha",
            "AwayTeam": "Beta",
            "FTHG": "2",
            "FTAG": "1",
            "B365AH": "-0.75",
            "B365AHH": "1.90",
            "B365AHA": "2.00",
        },
        "PL",
        {"name": "Premier League", "country": "England"},
        2003,
    )

    assert normalized["extra_data"]["asian_handicap_line"] is None
    assert normalized["extra_data"]["b365_asian_line"] == -0.75
    assert extract_asian_handicap_quotes(normalized)["b365"] == {
        "source": "bet365_open",
        "home": 1.90,
        "away": 2.00,
        "home_line": -0.75,
        "away_line": 0.75,
    }
    assert _coverage_flags(normalized)["asian_handicap_open"] is True


def test_bet365_specific_line_takes_precedence_over_market_line():
    quotes = extract_asian_handicap_quotes(
        _match(
            extra={
                "asian_handicap_line": -0.5,
                "b365_asian_line": -0.75,
                "b365_asian_home": 1.90,
                "b365_asian_away": 2.00,
                "avg_asian_home": 1.95,
                "avg_asian_away": 1.95,
            }
        )
    )

    assert quotes["b365"]["home_line"] == -0.75
    assert quotes["avg"]["home_line"] == -0.5


def test_feature_frame_attaches_ah_quotes_as_prices_not_model_features():
    frame = build_feature_frame(
        [
            _match(
                extra={
                    "asian_handicap_line": -0.25,
                    "pinnacle_asian_home": 1.97,
                    "pinnacle_asian_away": 1.95,
                }
            )
        ]
    )
    row = frame.iloc[0]

    assert row["odds_ah_pinnacle_available"] == 1
    assert row["odds_ah_pinnacle_home_line"] == -0.25
    assert row["odds_ah_pinnacle_away_line"] == 0.25
    assert row["market_ah_pinnacle_home_prob"] + row[
        "market_ah_pinnacle_away_prob"
    ] == pytest.approx(1.0)
    model_columns = infer_feature_columns(frame, "ou25")
    assert not any("odds_ah" in column or "market_ah" in column for column in model_columns.market_numeric)


def test_fixed_benchmark_labels_executable_and_proxy_tracks():
    matches = [
        _match(
            home_score=1,
            away_score=1,
            extra={
                "asian_handicap_line": 0.25,
                "b365_asian_home": 2.0,
                "b365_asian_away": 2.0,
                "avg_asian_home": 2.0,
                "avg_asian_away": 2.0,
            },
        )
    ]

    rows = fixed_blind_asian_benchmark(matches, bases=("b365", "avg"))

    by_key = {(row["odds_basis"], row["side"]): row for row in rows}
    assert by_key[("b365", "home")]["track"] == "executable"
    assert by_key[("b365", "home")]["profit"] == pytest.approx(0.5)
    assert by_key[("b365", "away")]["profit"] == pytest.approx(-0.5)
    assert by_key[("avg", "home")]["track"] == "proxy"


def test_fixed_benchmark_extracts_each_match_once(monkeypatch):
    import research.features as features

    matches = [
        _match(
            extra={
                "asian_handicap_line": 0.25,
                "b365_asian_home": 2.0,
                "b365_asian_away": 2.0,
                "avg_asian_home": 2.0,
                "avg_asian_away": 2.0,
            }
        )
        for _ in range(3)
    ]
    original = features.extract_asian_handicap_quotes
    calls = 0

    def counted(match):
        nonlocal calls
        calls += 1
        return original(match)

    monkeypatch.setattr(features, "extract_asian_handicap_quotes", counted)
    fixed_blind_asian_benchmark(matches, bases=("b365", "avg"))

    assert calls == len(matches)


def test_fixed_benchmark_with_no_bases_does_not_consume_matches():
    def matches():
        raise AssertionError("empty benchmark should not read the dataset")
        yield {}

    assert fixed_blind_asian_benchmark(matches(), bases=()) == []


def test_benchmark_cli_uses_manifest_row_count_and_writes_artifact(
    monkeypatch, capsys, tmp_path
):
    artifact = tmp_path / "asian.json"
    monkeypatch.setattr(
        run_asian_handicap_benchmark,
        "load_canonical_matches",
        lambda **_kwargs: ([], {"dataset_id": "fixture-dataset", "rows": 17}),
    )
    monkeypatch.setattr(
        run_asian_handicap_benchmark,
        "fixed_blind_asian_benchmark",
        lambda _matches, *, bases: [],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["asian-benchmark", "--start-season", "2003", "--output", str(artifact)],
    )

    run_asian_handicap_benchmark.main()

    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(artifact.read_text(encoding="utf-8"))
    assert printed == saved
    assert saved["source_matches"] == 17
    assert saved["season_start"] == 2003
