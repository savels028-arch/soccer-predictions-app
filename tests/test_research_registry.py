import json

import pytest

from research.registry import (
    RegistryValidationError,
    load_registry,
    register_research_result,
)


def _summary(*, passing: bool) -> dict:
    return {
        "bets": 400,
        "wins": 220,
        "profit": 40.0 if passing else -10.0,
        "roi": 0.10 if passing else -0.025,
        "roi_pct": 10.0 if passing else -2.5,
        "bootstrap": {
            "ci_lower": 0.02 if passing else -0.08,
            "ci_upper": 0.18 if passing else 0.03,
            "probability_roi_positive": 0.99 if passing else 0.20,
        },
        "stability": {
            "n_seasons": 5,
            "positive_season_rate": 0.80 if passing else 0.40,
        },
        "closing_line": {
            "bets_with_close": 250,
            "coverage_rate": 0.625,
            "mean_clv": 0.01,
        },
    }


def _result(*, passing: bool, odds_basis: str = "b365") -> dict:
    reasons = [] if passing else [
        "non_positive_outer_test_roi",
        "bootstrap_roi_lower_bound_not_positive",
        "probability_positive_roi_below_95pct",
        "positive_season_rate_below_60pct",
    ]
    return {
        "method": "nested_expanding_walk_forward_train_calibrate_select_test",
        "config": {"markets": ["ou25"], "policy_lock_season": 2023},
        "summaries": {"ou25": {"locked_executable": _summary(passing=passing)}},
        "promotion_gates": {
            "ou25": {"locked_executable": {"passed": passing, "reasons": reasons}}
        },
        "locked_strategies": {
            "ou25": {
                "executable": {
                    "selected": {
                        "eligible": True,
                        "spec": {
                            "market": "ou25",
                            "family": "market__raw",
                            "odds_basis": odds_basis,
                            "side": "over",
                            "min_edge": 0.03,
                            "min_confidence": 0.55,
                            "min_odds": 1.5,
                            "max_odds": 2.5,
                        },
                        "bets": 900,
                        "roi": 0.04,
                        "seasons": 10,
                        "positive_season_rate": 0.70,
                    }
                }
            }
        },
        "champion_candidate": {
            "status": "PROMOTABLE_TO_SHADOW" if passing else "NO_PROMOTION",
            "markets": ["ou25"] if passing else [],
        },
    }


def test_failed_gate_is_audited_but_never_registered(tmp_path):
    registry_path = tmp_path / "registry.json"

    registry, event = register_research_result(
        registry_path,
        _result(passing=False),
        run_id="failed-run",
        dataset_id="dataset-1",
        evaluated_at="2026-07-14T12:00:00+00:00",
    )

    assert event["status"] == "NO_PROMOTION"
    assert event["registered_markets"] == []
    assert registry["shadow_challengers"] == {}
    assert registry["automatic_live_activation"] is False
    assert "non_positive_outer_test_roi" in event["rejected_markets"]["ou25"]


def test_only_passing_locked_executable_strategy_enters_shadow_registry(tmp_path):
    registry_path = tmp_path / "registry.json"

    registry, event = register_research_result(
        registry_path,
        _result(passing=True),
        run_id="passing-run",
        dataset_id="dataset-1",
        evaluated_at="2026-07-14T12:00:00+00:00",
        git_sha="abc123",
    )

    candidate = registry["shadow_challengers"]["ou25"]["passing-run"]
    assert event["status"] == "REGISTERED_TO_SHADOW"
    assert candidate["track"] == "locked_executable"
    assert candidate["mode"] == "shadow_only"
    assert candidate["strategy"]["odds_basis"] == "b365"
    assert candidate["promotion_gate"] == {"passed": True, "reasons": []}
    assert "live_strategy" not in registry
    assert load_registry(registry_path) == registry


def test_tampered_reported_gate_fails_closed(tmp_path):
    result = _result(passing=False)
    result["promotion_gates"]["ou25"]["locked_executable"] = {
        "passed": True,
        "reasons": [],
    }
    result["champion_candidate"] = {
        "status": "PROMOTABLE_TO_SHADOW",
        "markets": ["ou25"],
    }

    registry, event = register_research_result(
        tmp_path / "registry.json",
        result,
        run_id="tampered-run",
        dataset_id="dataset-1",
    )

    assert registry["shadow_challengers"] == {}
    assert event["status"] == "NO_PROMOTION"
    assert any("reported_gate_does_not_match" in reason for reason in event["rejected_markets"]["ou25"])


def test_proxy_or_non_executable_odds_cannot_be_registered(tmp_path):
    registry, event = register_research_result(
        tmp_path / "registry.json",
        _result(passing=True, odds_basis="max"),
        run_id="proxy-run",
        dataset_id="dataset-1",
    )

    assert registry["shadow_challengers"] == {}
    assert event["status"] == "NO_PROMOTION"
    assert "locked_strategy_does_not_use_executable_odds" in event["rejected_markets"]["ou25"]


def test_invalid_strategy_spec_cannot_enter_shadow_registry(tmp_path):
    result = _result(passing=True)
    result["locked_strategies"]["ou25"]["executable"]["selected"]["spec"]["side"] = "draw"

    registry, event = register_research_result(
        tmp_path / "registry.json",
        result,
        run_id="invalid-spec-run",
        dataset_id="dataset-1",
    )

    assert registry["shadow_challengers"] == {}
    assert event["status"] == "NO_PROMOTION"
    assert "locked_strategy_has_invalid_side" in event["rejected_markets"]["ou25"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("family", "invented_market50__raw", "locked_strategy_has_invalid_family"),
        ("family", "market", "locked_strategy_has_invalid_family"),
        ("min_edge", 0.02, "locked_strategy_has_invalid_min_edge"),
        ("min_odds", 1.33, "locked_strategy_is_not_in_fixed_odds_grid"),
    ],
)
def test_strategy_outside_predeclared_fixed_grid_cannot_be_registered(
    tmp_path, field, value, reason
):
    result = _result(passing=True)
    result["locked_strategies"]["ou25"]["executable"]["selected"]["spec"][field] = value

    registry, event = register_research_result(
        tmp_path / "registry.json",
        result,
        run_id=f"invalid-{field}-{value}",
        dataset_id="dataset-1",
    )

    assert registry["shadow_challengers"] == {}
    assert event["status"] == "NO_PROMOTION"
    assert reason in event["rejected_markets"]["ou25"]


def test_registry_refuses_live_activation_and_conflicting_run_ids(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "research_shadow_only",
                "automatic_live_activation": True,
                "shadow_challengers": {},
                "evaluations": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RegistryValidationError, match="automatic live activation"):
        load_registry(registry_path)

    registry_path.unlink()
    register_research_result(
        registry_path,
        _result(passing=False),
        run_id="same-run",
        dataset_id="dataset-1",
    )
    changed = _result(passing=False)
    changed["summaries"]["ou25"]["locked_executable"]["bets"] = 401
    with pytest.raises(RegistryValidationError, match="different evidence"):
        register_research_result(
            registry_path,
            changed,
            run_id="same-run",
            dataset_id="dataset-1",
        )
