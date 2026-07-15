"""Nested walk-forward strategy-zoo engine for AIBets.

This module is intentionally isolated from the legacy ``backtest.py``.  Every
outer test season is evaluated only after model fitting, calibration and policy
selection have finished on strictly earlier data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.calibration import (
    apply_temperature,
    fit_isotonic_ovr,
    fit_temperature,
    normalize_probabilities,
)
from research.dataset import LATEST_COMPLETE_SEASON
from research.metrics import (
    block_bootstrap_roi,
    flat_stake_metrics,
    probability_metrics,
    season_stability,
    settle_flat_stake,
)
from research.models import DixonColesModel, fit_probability_families, odds_matrices
from research.selection import (
    DEFAULT_CONFIDENCE_THRESHOLDS,
    DEFAULT_EDGE_THRESHOLDS,
    StrategySpec,
    _strategy_tie_break,
    apply_selected_strategy,
    candidate_mask,
    select_strategy,
)
from research.splits import NestedSeasonFold, nested_season_folds


DEFAULT_ODDS_BANDS = ((1.20, 5.00), (1.20, 2.00), (1.50, 2.50), (1.80, 3.50))
DEFAULT_EXECUTABLE_BASES = {
    "1x2": ("primary", "b365"),
    "ou25": ("b365", "pinnacle"),
}
DEFAULT_PROXY_BASES = {
    "1x2": ("avg", "max"),
    "ou25": ("primary", "avg", "max"),
}
_QUOTE_SOURCE_FAMILIES = {
    "bet365_open": "bet365",
    "bet365_close": "bet365",
    "pinnacle_open": "pinnacle",
    "pinnacle_close": "pinnacle",
    "market_average_open": "market_average",
    "market_average_close": "market_average",
    "market_max_open": "market_max",
    "market_max_close": "market_max",
}


@dataclass(frozen=True)
class ResearchConfig:
    first_test_season: int = 2012
    last_test_season: int = LATEST_COMPLETE_SEASON
    markets: tuple[str, ...] = ("1x2", "ou25")
    min_train_seasons: int = 5
    min_selection_bets: int = 40
    odds_haircut: float = 0.01
    random_state: int = 20260714
    include_boosting: bool = True
    include_isotonic: bool = False
    bootstrap_resamples: int = 2_000
    policy_lock_season: int = 2023


@dataclass
class _CandidateAggregate:
    bets: int = 0
    wins: int = 0
    profit: float = 0.0
    profit_squares: float = 0.0
    season_profit: Dict[int, float] = None  # type: ignore[assignment]
    season_bets: Dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.season_profit is None:
            self.season_profit = {}
        if self.season_bets is None:
            self.season_bets = {}


def _target(frame: pd.DataFrame, market: str) -> np.ndarray:
    column = "target_1x2_index" if market == "1x2" else "target_over25"
    return frame[column].to_numpy(dtype=int)


def haircut_odds(values: np.ndarray, haircut: float) -> np.ndarray:
    """Reduce the profit portion of decimal odds by an execution haircut."""

    odds = np.asarray(values, dtype=float).copy()
    if not 0.0 <= haircut < 1.0:
        raise ValueError("odds haircut must be in [0, 1)")
    valid = np.isfinite(odds) & (odds > 1.0)
    odds[valid] = 1.0 + (odds[valid] - 1.0) * (1.0 - haircut)
    return odds


def _calibrated_prediction_slices(
    raw: Mapping[str, Mapping[str, np.ndarray]],
    calibration_outcomes: np.ndarray,
    *,
    include_isotonic: bool,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Fit calibration on one slice and apply it to selection and test."""

    output: Dict[str, Dict[str, np.ndarray]] = {"selection": {}, "test": {}}
    calibration = raw["calibration"]
    for family, calibration_probabilities in calibration.items():
        calibration_probabilities = normalize_probabilities(calibration_probabilities)
        temperature = fit_temperature(calibration_probabilities, calibration_outcomes)
        isotonic = (
            fit_isotonic_ovr(calibration_probabilities, calibration_outcomes)
            if include_isotonic
            else None
        )
        for slice_name in ("selection", "test"):
            values = normalize_probabilities(raw[slice_name][family])
            output[slice_name][f"{family}__raw"] = values
            output[slice_name][f"{family}__temperature"] = apply_temperature(values, temperature)
            if isotonic is not None:
                output[slice_name][f"{family}__isotonic"] = isotonic.transform(values)
    return output


def _slice_odds(
    frame: pd.DataFrame,
    positions: np.ndarray,
    market: str,
    haircut: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    raw = odds_matrices(frame.iloc[positions], market)
    adjusted = {name: haircut_odds(values, haircut) for name, values in raw.items()}
    return raw, adjusted


def _same_source_closing_price(
    opening_source: object,
    closing_source: object,
    closing_price: object,
) -> Optional[float]:
    """Return a close only when it belongs to the opening quote's source.

    A generic closing matrix can contain Bet365, Pinnacle, market-average or
    market-maximum prices on different rows.  Comparing one family with another
    creates fake CLV, so unknown and cross-family pairs deliberately return
    ``None``.
    """

    opening_family = _QUOTE_SOURCE_FAMILIES.get(str(opening_source or ""))
    closing_family = _QUOTE_SOURCE_FAMILIES.get(str(closing_source or ""))
    try:
        price = float(closing_price)
    except (TypeError, ValueError):
        return None
    if (
        opening_family is None
        or closing_family is None
        or opening_family != closing_family
        or not np.isfinite(price)
        or price <= 1.0
    ):
        return None
    return price


def _materialize_bets(
    selected: Optional[Mapping[str, object]],
    probabilities: Mapping[str, np.ndarray],
    raw_odds: Mapping[str, np.ndarray],
    adjusted_odds: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    test_frame: pd.DataFrame,
    *,
    market: str,
    track: str,
    test_season: int,
) -> List[Dict[str, object]]:
    if selected is None:
        return []
    spec = StrategySpec(**dict(selected["spec"]))
    values = probabilities[spec.family]
    mask, labels, confidence, edge = candidate_mask(spec, values, adjusted_odds[spec.odds_basis])
    selected_positions = np.flatnonzero(mask)
    close = raw_odds.get("close")
    bets: List[Dict[str, object]] = []
    for position in selected_positions:
        label = int(labels[position])
        won = int(label == int(outcomes[position]))
        raw_price = float(raw_odds[spec.odds_basis][position, label])
        price = float(adjusted_odds[spec.odds_basis][position, label])
        source_row = test_frame.iloc[position]
        opening_source = source_row.get(f"odds_{market}_{spec.odds_basis}_source")
        closing_source = source_row.get(f"odds_{market}_close_source")
        observed_close = close[position, label] if close is not None else None
        close_price = _same_source_closing_price(
            opening_source,
            closing_source,
            observed_close,
        )
        profit = price - 1.0 if won else -1.0
        bets.append(
            {
                "match_id": str(source_row["match_id"]),
                "match_date": pd.Timestamp(source_row["match_date"]).isoformat(),
                "season": int(test_season),
                "league": str(source_row["league_code"]),
                "home_team": str(source_row["home_team"]),
                "away_team": str(source_row["away_team"]),
                "market": market,
                "track": track,
                "family": spec.family,
                "odds_basis": spec.odds_basis,
                "side_filter": spec.side,
                "selection": label,
                "outcome": int(outcomes[position]),
                "won": won,
                "probability": float(confidence[position]),
                "model_edge": float(edge[position]),
                "raw_odds": raw_price,
                "decimal_odds": price,
                "opening_odds_source": opening_source,
                "observed_closing_odds_source": closing_source,
                "closing_odds": close_price,
                "clv": (raw_price / close_price - 1.0)
                if close_price is not None
                else None,
                "profit": profit,
            }
        )
    return bets


def _fixed_family_allowed(family: str) -> bool:
    base = family.split("__", 1)[0]
    return base in {
        "market",
        "poisson",
        "dixon_coles",
        "league_prior",
        "elo",
        "logistic_market",
        "boosting_market",
    } or base.endswith("market50")


def _fixed_specs(
    market: str,
    probability_families: Mapping[str, np.ndarray],
    allowed_bases: Iterable[str],
) -> Iterable[StrategySpec]:
    sides = ("all", "no_draw", "home", "draw", "away") if market == "1x2" else ("all", "under", "over")
    for family in sorted(probability_families):
        if not _fixed_family_allowed(family):
            continue
        for basis in allowed_bases:
            for side in sides:
                for edge in (None, 0.0, 0.03, 0.06, 0.09):
                    for confidence in (None, 0.55, 0.65):
                        for min_odds, max_odds in (
                            (1.20, 5.00),
                            (1.20, 2.00),
                            (1.50, 2.50),
                            (1.80, 3.50),
                        ):
                            yield StrategySpec(
                                market=market,
                                family=family,
                                odds_basis=basis,
                                side=side,
                                min_edge=edge,
                                min_confidence=confidence,
                                min_odds=min_odds,
                                max_odds=max_odds,
                            )


def _accumulate_fixed_candidates(
    accumulator: Dict[StrategySpec, _CandidateAggregate],
    market: str,
    allowed_bases: Iterable[str],
    probabilities: Mapping[str, np.ndarray],
    odds: Mapping[str, np.ndarray],
    outcomes: np.ndarray,
    season: int,
) -> int:
    evaluated = 0
    for spec in _fixed_specs(market, probabilities, allowed_bases):
        if spec.odds_basis not in odds:
            continue
        mask, labels, _, _ = candidate_mask(spec, probabilities[spec.family], odds[spec.odds_basis])
        selected = np.flatnonzero(mask)
        evaluated += 1
        if not len(selected):
            continue
        selected_labels = labels[selected]
        prices = odds[spec.odds_basis][selected, selected_labels]
        wins = (selected_labels == outcomes[selected]).astype(int)
        profits = np.where(wins == 1, prices - 1.0, -1.0)
        aggregate = accumulator.setdefault(spec, _CandidateAggregate())
        aggregate.bets += int(len(selected))
        aggregate.wins += int(wins.sum())
        aggregate.profit += float(profits.sum())
        aggregate.profit_squares += float(np.square(profits).sum())
        aggregate.season_profit[season] = aggregate.season_profit.get(season, 0.0) + float(profits.sum())
        aggregate.season_bets[season] = aggregate.season_bets.get(season, 0) + int(len(selected))
    return evaluated


def _aggregate_candidate_result(spec: StrategySpec, aggregate: _CandidateAggregate) -> Dict[str, object]:
    roi = aggregate.profit / aggregate.bets if aggregate.bets else 0.0
    if aggregate.bets > 1:
        variance = max(
            0.0,
            (aggregate.profit_squares - aggregate.profit**2 / aggregate.bets) / (aggregate.bets - 1),
        )
        standard_error = float(np.sqrt(variance / aggregate.bets))
    else:
        standard_error = float("inf")
    seasons = [season for season, bets in aggregate.season_bets.items() if bets > 0]
    positive_seasons = sum(aggregate.season_profit.get(season, 0.0) > 0.0 for season in seasons)
    positive_rate = positive_seasons / len(seasons) if seasons else 0.0
    return {
        "spec": asdict(spec),
        "bets": aggregate.bets,
        "wins": aggregate.wins,
        "hit_rate": aggregate.wins / aggregate.bets if aggregate.bets else 0.0,
        "profit": aggregate.profit,
        "roi": roi,
        "roi_pct": roi * 100.0,
        "roi_standard_error": standard_error,
        "roi_lcb_90": roi - 1.645 * standard_error if np.isfinite(standard_error) else float("-inf"),
        "seasons": len(seasons),
        "positive_season_rate": positive_rate,
        "by_season_profit": dict(sorted(aggregate.season_profit.items())),
    }


def _lock_fixed_strategy(
    accumulator: Mapping[StrategySpec, _CandidateAggregate],
    *,
    min_bets: int = 300,
    min_seasons: int = 5,
    diagnostic_min_bets: int = 150,
    diagnostic_min_seasons: int = 3,
) -> Dict[str, object]:
    ranked = []
    diagnostic_ranked = []
    for spec, aggregate in accumulator.items():
        result = _aggregate_candidate_result(spec, aggregate)
        eligibility_reasons = []
        if int(result["bets"]) < min_bets:
            eligibility_reasons.append(f"fewer_than_{min_bets}_development_bets")
        if int(result["seasons"]) < min_seasons:
            eligibility_reasons.append(f"fewer_than_{min_seasons}_development_seasons")
        if float(result["roi"]) <= 0.0:
            eligibility_reasons.append("non_positive_development_roi")
        if float(result["positive_season_rate"]) < 0.55:
            eligibility_reasons.append("development_positive_season_rate_below_55pct")
        result["eligible"] = not eligibility_reasons
        result["eligibility_reasons"] = eligibility_reasons
        diagnostic_history = (
            int(result["bets"]) >= diagnostic_min_bets
            and int(result["seasons"]) >= diagnostic_min_seasons
        )
        if diagnostic_history:
            diagnostic_ranked.append(result)
        if result["eligible"]:
            ranked.append(result)
    rank_key = lambda result: (
        float(result["roi_lcb_90"]),
        float(result["roi"]),
        int(result["bets"]),
        *_strategy_tie_break(result),
    )
    ranked.sort(
        key=rank_key,
        reverse=True,
    )
    diagnostic_ranked.sort(key=rank_key, reverse=True)
    return {
        "candidate_specs_with_development_bets": len(accumulator),
        "eligible_strategy_specs": len(ranked),
        "selected": ranked[0] if ranked else None,
        "diagnostic_selected": diagnostic_ranked[0] if diagnostic_ranked else None,
        "diagnostic_min_bets": diagnostic_min_bets,
        "diagnostic_min_seasons": diagnostic_min_seasons,
        "top_development_candidates": ranked[:10],
        "top_observed_candidates": diagnostic_ranked[:10],
    }


def _fold_market(
    frame: pd.DataFrame,
    fold: NestedSeasonFold,
    market: str,
    config: ResearchConfig,
    fixed_development: Dict[str, Dict[StrategySpec, _CandidateAggregate]],
    locked_strategies: Dict[str, Optional[Dict[str, object]]],
    dixon_coles_model: DixonColesModel,
) -> Dict[str, object]:
    positions = {
        "calibration": np.flatnonzero(fold.calibration_mask),
        "selection": np.flatnonzero(fold.selection_mask),
        "test": np.flatnonzero(fold.test_mask),
    }
    training = np.flatnonzero(fold.train_mask)
    raw_predictions = fit_probability_families(
        frame,
        market,
        training,
        positions,
        random_state=config.random_state + fold.test_season,
        include_boosting=config.include_boosting,
        dixon_coles_model=dixon_coles_model,
    )
    calibration_y = _target(frame.iloc[positions["calibration"]], market)
    selection_y = _target(frame.iloc[positions["selection"]], market)
    test_y = _target(frame.iloc[positions["test"]], market)
    predictions = _calibrated_prediction_slices(
        raw_predictions,
        calibration_y,
        include_isotonic=config.include_isotonic,
    )
    selection_raw_odds, selection_odds = _slice_odds(
        frame, positions["selection"], market, config.odds_haircut
    )
    test_raw_odds, test_odds = _slice_odds(frame, positions["test"], market, config.odds_haircut)

    compact_edges = tuple(value for value in DEFAULT_EDGE_THRESHOLDS if value in (None, 0.0, 0.02, 0.04, 0.06, 0.08))
    compact_confidences = tuple(
        value for value in DEFAULT_CONFIDENCE_THRESHOLDS if value in (None, 0.50, 0.55, 0.60, 0.65)
    )
    tracks = {
        "executable": DEFAULT_EXECUTABLE_BASES[market],
        "proxy_upper_bound": DEFAULT_PROXY_BASES[market],
    }
    fold_tracks: Dict[str, object] = {}
    bets: List[Dict[str, object]] = []
    test_frame = frame.iloc[positions["test"]].reset_index(drop=True)

    for track, allowed_bases in tracks.items():
        selection = select_strategy(
            market,
            predictions["selection"],
            selection_odds,
            selection_y,
            min_bets=config.min_selection_bets,
            edge_thresholds=compact_edges,
            confidence_thresholds=compact_confidences,
            odds_bands=DEFAULT_ODDS_BANDS,
            allowed_odds_bases=allowed_bases,
        )
        selected = selection["selected"]
        outer = apply_selected_strategy(
            selected,
            predictions["test"],
            test_odds,
            test_y,
        )
        if selected is not None:
            selected_family = str(selected["spec"]["family"])
            outer["probability_metrics"] = probability_metrics(
                predictions["test"][selected_family].tolist(), test_y.tolist()
            )
        fold_tracks[track] = {
            "selection": selection,
            "outer_test": outer,
        }
        bets.extend(
            _materialize_bets(
                selected,
                predictions["test"],
                test_raw_odds,
                test_odds,
                test_y,
                test_frame,
                market=market,
                track=track,
                test_season=fold.test_season,
            )
        )

        if fold.test_season < config.policy_lock_season:
            evaluated_fixed = _accumulate_fixed_candidates(
                fixed_development[track],
                market,
                allowed_bases,
                predictions["test"],
                test_odds,
                test_y,
                fold.test_season,
            )
            fold_tracks[track]["fixed_policy_development_spec_evaluations"] = evaluated_fixed
        else:
            if track not in locked_strategies:
                locked_strategies[track] = _lock_fixed_strategy(fixed_development[track])
            lock_result = locked_strategies[track]
            fixed_selected = lock_result.get("selected") if lock_result else None
            fixed_outer = apply_selected_strategy(
                fixed_selected,
                predictions["test"],
                test_odds,
                test_y,
            )
            fold_tracks[track]["locked_policy"] = {
                "development": lock_result,
                "outer_test": fixed_outer,
            }
            bets.extend(
                _materialize_bets(
                    fixed_selected,
                    predictions["test"],
                    test_raw_odds,
                    test_odds,
                    test_y,
                    test_frame,
                    market=market,
                    track=f"locked_{track}",
                    test_season=fold.test_season,
                )
            )
            diagnostic_selected = lock_result.get("diagnostic_selected") if lock_result else None
            if fixed_selected is None and diagnostic_selected is not None:
                diagnostic_outer = apply_selected_strategy(
                    diagnostic_selected,
                    predictions["test"],
                    test_odds,
                    test_y,
                )
                fold_tracks[track]["locked_policy"]["diagnostic_outer_test"] = diagnostic_outer
                bets.extend(
                    _materialize_bets(
                        diagnostic_selected,
                        predictions["test"],
                        test_raw_odds,
                        test_odds,
                        test_y,
                        test_frame,
                        market=market,
                        track=f"locked_diagnostic_{track}",
                        test_season=fold.test_season,
                    )
                )

    return {
        "market": market,
        "train_rows": int(fold.train_mask.sum()),
        "calibration_rows": int(fold.calibration_mask.sum()),
        "selection_rows": int(fold.selection_mask.sum()),
        "test_rows": int(fold.test_mask.sum()),
        "tracks": fold_tracks,
        "bets": bets,
    }


def summarize_bets(
    bets: Sequence[Mapping[str, object]],
    *,
    bootstrap_resamples: int = 2_000,
    seed: int = 20260714,
) -> Dict[str, object]:
    if not bets:
        return {
            "bets": 0,
            "wins": 0,
            "profit": 0.0,
            "roi": 0.0,
            "roi_pct": 0.0,
            "status": "no_bets",
        }
    ordered = sorted(bets, key=lambda bet: str(bet["match_date"]))
    wins = [int(bet["won"]) for bet in ordered]
    odds = [float(bet["decimal_odds"]) for bet in ordered]
    profits = settle_flat_stake(wins, odds)
    seasons = [bet["season"] for bet in ordered]
    metrics = flat_stake_metrics(wins, odds)
    stability = season_stability(profits, seasons)
    bootstrap = block_bootstrap_roi(
        profits,
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    clv = [float(bet["clv"]) for bet in ordered if bet.get("clv") is not None]
    scenario_roi = {}
    raw_odds = np.asarray([float(bet["raw_odds"]) for bet in ordered], dtype=float)
    for haircut in (0.0, 0.01, 0.02):
        scenario = flat_stake_metrics(wins, haircut_odds(raw_odds[:, None], haircut)[:, 0].tolist())
        scenario_roi[f"haircut_{int(haircut * 100)}pct"] = scenario["roi"]
    return {
        **metrics,
        "status": "tested",
        "bootstrap": bootstrap,
        "stability": stability,
        "closing_line": {
            "bets_with_close": len(clv),
            "coverage_rate": len(clv) / len(ordered),
            "mean_clv": float(np.mean(clv)) if clv else None,
            "positive_clv_rate": float(np.mean(np.asarray(clv) > 0.0)) if clv else None,
        },
        "odds_haircut_scenarios": scenario_roi,
    }


def _promotion_gate(summary: Mapping[str, object]) -> Dict[str, object]:
    if int(summary.get("bets", 0)) == 0:
        return {"passed": False, "reasons": ["no_outer_test_bets"]}
    bootstrap = summary.get("bootstrap", {})
    stability = summary.get("stability", {})
    closing = summary.get("closing_line", {})
    reasons = []
    if int(summary.get("bets", 0)) < 300:
        reasons.append("fewer_than_300_outer_test_bets")
    if float(summary.get("roi", 0.0)) <= 0.0:
        reasons.append("non_positive_outer_test_roi")
    if float(bootstrap.get("ci_lower", -1.0)) <= 0.0:
        reasons.append("bootstrap_roi_lower_bound_not_positive")
    if float(bootstrap.get("probability_roi_positive", 0.0)) < 0.95:
        reasons.append("probability_positive_roi_below_95pct")
    if float(stability.get("positive_season_rate", 0.0)) < 0.60:
        reasons.append("positive_season_rate_below_60pct")
    if int(stability.get("n_seasons", 0) or 0) < 3:
        reasons.append("fewer_than_3_outer_test_seasons_with_bets")
    comparable_closes = int(closing.get("bets_with_close", 0) or 0)
    closing_coverage = float(closing.get("coverage_rate", 0.0) or 0.0)
    if comparable_closes < 100:
        reasons.append("fewer_than_100_same_source_closing_observations")
    if closing_coverage < 0.50:
        reasons.append("same_source_closing_coverage_below_50pct")
    if (
        comparable_closes >= 100
        and closing_coverage >= 0.50
        and float(closing.get("mean_clv") or 0.0) <= 0.0
    ):
        reasons.append("non_positive_mean_clv")
    return {"passed": not reasons, "reasons": reasons}


def run_nested_strategy_zoo(
    frame: pd.DataFrame,
    config: ResearchConfig,
    *,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run all configured model/policy families across nested season folds."""

    if config.last_test_season - config.policy_lock_season + 1 < 3:
        raise ValueError(
            "policy_lock_season must leave at least three configured outer holdout seasons"
        )
    folds = nested_season_folds(
        frame,
        first_test_season=config.first_test_season,
        last_test_season=config.last_test_season,
        min_train_seasons=config.min_train_seasons,
    )
    holdout_seasons = {
        fold.test_season for fold in folds if fold.test_season >= config.policy_lock_season
    }
    if len(holdout_seasons) < 3:
        raise ValueError(
            "policy_lock_season must leave at least three available outer holdout folds"
        )
    fold_results: List[Dict[str, object]] = []
    all_bets: List[Dict[str, object]] = []
    fixed_development = {
        market: {"executable": {}, "proxy_upper_bound": {}}
        for market in config.markets
    }
    locked_strategies: Dict[str, Dict[str, Optional[Dict[str, object]]]] = {
        market: {} for market in config.markets
    }
    for fold in folds:
        if progress:
            progress(f"outer fold {fold.test_season}: train through {fold.validation_season - 1}")
        fold_result: Dict[str, object] = {
            "test_season": fold.test_season,
            "validation_season": fold.validation_season,
            "train_seasons": list(fold.train_seasons),
            "calibration_end": fold.calibration_end,
            "markets": {},
        }
        training = np.flatnonzero(fold.train_mask)
        dixon_coles_model = DixonColesModel().fit(frame.iloc[training])
        for market in config.markets:
            if progress:
                progress(f"outer fold {fold.test_season}: fitting and selecting {market}")
            market_result = _fold_market(
                frame,
                fold,
                market,
                config,
                fixed_development[market],
                locked_strategies[market],
                dixon_coles_model,
            )
            fold_result["markets"][market] = {key: value for key, value in market_result.items() if key != "bets"}
            all_bets.extend(market_result["bets"])
            if progress:
                tested = market_result["tracks"]["executable"]["outer_test"]
                progress(
                    f"outer fold {fold.test_season}: {market} test bets={tested.get('bets', 0)} "
                    f"roi={float(tested.get('roi_pct', 0.0)):+.2f}%"
                )
        fold_results.append(fold_result)

    summaries: Dict[str, Dict[str, object]] = {}
    gates: Dict[str, Dict[str, object]] = {}
    for market in config.markets:
        summaries[market] = {}
        gates[market] = {}
        for track in ("executable", "proxy_upper_bound"):
            subset = [bet for bet in all_bets if bet["market"] == market and bet["track"] == track]
            summary = summarize_bets(
                subset,
                bootstrap_resamples=config.bootstrap_resamples,
                seed=config.random_state,
            )
            summaries[market][track] = summary
            gates[market][track] = _promotion_gate(summary) if track == "executable" else {
                "passed": False,
                "reasons": ["proxy_or_max_prices_are_not_live_promotable"],
            }

        for track in ("locked_executable", "locked_proxy_upper_bound"):
            subset = [bet for bet in all_bets if bet["market"] == market and bet["track"] == track]
            summary = summarize_bets(
                subset,
                bootstrap_resamples=config.bootstrap_resamples,
                seed=config.random_state,
            )
            summaries[market][track] = summary
            gates[market][track] = (
                _promotion_gate(summary)
                if track == "locked_executable"
                else {"passed": False, "reasons": ["proxy_or_max_prices_are_not_live_promotable"]}
            )
        for track in ("locked_diagnostic_executable", "locked_diagnostic_proxy_upper_bound"):
            subset = [bet for bet in all_bets if bet["market"] == market and bet["track"] == track]
            summary = summarize_bets(
                subset,
                bootstrap_resamples=config.bootstrap_resamples,
                seed=config.random_state,
            )
            summaries[market][track] = summary
            gates[market][track] = {
                "passed": False,
                "reasons": ["development_strategy_failed_predeclared_eligibility_gate"],
            }

    promotable = [
        market
        for market in config.markets
        if gates[market]["locked_executable"]["passed"]
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "nested_expanding_walk_forward_train_calibrate_select_test",
        "config": asdict(config),
        "folds": fold_results,
        "bets": all_bets,
        "summaries": summaries,
        "promotion_gates": gates,
        "locked_strategies": locked_strategies,
        "champion_candidate": {
            "status": "PROMOTABLE_TO_SHADOW" if promotable else "NO_PROMOTION",
            "markets": promotable,
            "notice": "Historical evidence creates a shadow candidate, never an automatic live strategy.",
        },
    }


__all__ = [
    "ResearchConfig",
    "haircut_odds",
    "run_nested_strategy_zoo",
    "summarize_bets",
]
