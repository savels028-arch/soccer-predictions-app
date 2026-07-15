"""Point-in-time league and season betting evidence atlas.

The atlas deliberately separates three different questions:

* ``descriptive``: what happened in the matches;
* ``hindsight``: which fixed strategy happened to rank best in that season;
* ``walk_forward_candidate`` / ``rejected``: what could have been selected
  using only seasons that had already finished.

Only complete, explicitly named Bet365 pre-closing quotes are executable.  The
generic normalized odds fields and ``Max*`` prices are never used for P&L.
This makes the output conservative, reproducible, and suitable for a public
research product without implying that hindsight profit was achievable.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


COD_THRESHOLDS: Tuple[float, ...] = (0.125, 0.25, 0.375)
DIRECT_H2H_THRESHOLDS: Tuple[float, ...] = (0.60, 0.67)
DIRECT_H2H_MIN_MEETINGS = 10
GOAL_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
DEFAULT_MIN_COD_MATCHES = 6
DEFAULT_MIN_TRAINING_SEASONS = 3
DEFAULT_MIN_TRAINING_BETS = 100
PROFIT_HAIRCUT = 0.01
_CI_Z = 1.959963984540054
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "edge_atlas_public.json"


def _threshold_id(value: float) -> str:
    return format(value, "g").replace(".", "_")


BASE_STRATEGY_IDS: Tuple[str, ...] = (
    "home_win",
    "draw",
    "away_win",
    "favourite",
    "over_2_5",
    "under_2_5",
)
COD_STRATEGY_IDS: Tuple[str, ...] = tuple(
    f"cod_{side}_lte_{_threshold_id(threshold)}"
    for side in ("home", "away")
    for threshold in COD_THRESHOLDS
)
INVERSE_COD_STRATEGY_IDS: Tuple[str, ...] = tuple(
    strategy_id
    for threshold in COD_THRESHOLDS
    for strategy_id in (
        f"cod_home_gte_{_threshold_id(1.0 - threshold)}_back_away",
        f"cod_away_gte_{_threshold_id(1.0 - threshold)}_back_home",
    )
)
DIRECT_H2H_STRATEGY_IDS: Tuple[str, ...] = tuple(
    f"favorite_direct_h2h_agree_0_{round(threshold * 100):02d}"
    for threshold in DIRECT_H2H_THRESHOLDS
)
STRATEGY_IDS: Tuple[str, ...] = (
    BASE_STRATEGY_IDS
    + COD_STRATEGY_IDS
    + INVERSE_COD_STRATEGY_IDS
    + DIRECT_H2H_STRATEGY_IDS
)


@dataclass
class _CodState:
    """Exact distribution of a team's points in quoted season matches."""

    matches: int = 0
    actual_points: int = 0
    distribution: Dict[int, float] = field(default_factory=lambda: {0: 1.0})

    def value(self, min_matches: int) -> Optional[float]:
        if self.matches < min_matches:
            return None
        below = math.fsum(
            probability
            for points, probability in self.distribution.items()
            if points < self.actual_points
        )
        equal = self.distribution.get(self.actual_points, 0.0)
        # Wheatcroft's COD is the mid-quantile: P(S < actual) + 0.5 P(S = actual).
        return min(1.0, max(0.0, below + 0.5 * equal))

    def add(self, probabilities: Tuple[float, float, float], points: int) -> None:
        win_probability, draw_probability, loss_probability = probabilities
        updated: Dict[int, float] = {}
        for total, probability in self.distribution.items():
            updated[total] = updated.get(total, 0.0) + probability * loss_probability
            updated[total + 1] = updated.get(total + 1, 0.0) + probability * draw_probability
            updated[total + 3] = updated.get(total + 3, 0.0) + probability * win_probability
        self.distribution = updated
        self.actual_points += points
        self.matches += 1


@dataclass(frozen=True)
class _PreparedMatch:
    order: int
    kickoff: str
    season: int
    league_code: str
    league_name: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    quote_1x2: Optional[Tuple[float, float, float]]
    quote_ou25: Optional[Tuple[float, float]]
    home_cod: Optional[float]
    away_cod: Optional[float]
    direct_h2h_meetings: int
    direct_h2h_mode: Optional[str]
    direct_h2h_hit_rate: Optional[float]


@dataclass(frozen=True)
class _Bet:
    order: int
    season: int
    won: bool
    odds: float
    profit: float


@dataclass
class _EvidenceAccumulator:
    """Incremental evidence used by walk-forward selection."""

    count: int = 0
    wins: int = 0
    profit: float = 0.0
    profit_squared: float = 0.0
    equity: float = 0.0
    peak: float = 0.0
    drawdown: float = 0.0
    by_season: Dict[int, float] = field(default_factory=dict)
    bets_by_season: Dict[int, int] = field(default_factory=dict)

    def add(self, bets: Sequence[_Bet]) -> None:
        for bet in bets:
            self.count += 1
            self.wins += int(bet.won)
            self.profit += bet.profit
            self.profit_squared += bet.profit * bet.profit
            self.equity += bet.profit
            self.peak = max(self.peak, self.equity)
            self.drawdown = max(self.drawdown, self.peak - self.equity)
            self.by_season[bet.season] = self.by_season.get(bet.season, 0.0) + bet.profit
            self.bets_by_season[bet.season] = self.bets_by_season.get(bet.season, 0) + 1

    def evidence(self) -> Dict[str, Any]:
        return _evidence_from_totals(
            count=self.count,
            wins=self.wins,
            profit=self.profit,
            profit_squared=self.profit_squared,
            drawdown=self.drawdown,
            by_season=self.by_season,
            bets_by_season=self.bets_by_season,
        )


def _finite_price(value: Any) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 1.0:
        return None
    return price


def _score(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0.0 or not number.is_integer():
        return None
    return int(number)


def _season(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        season = int(value)
    except (TypeError, ValueError):
        return None
    return season if 1800 <= season <= 2200 else None


def _complete_quote(extra: Mapping[str, Any], keys: Sequence[str]) -> Optional[Tuple[float, ...]]:
    prices = tuple(_finite_price(extra.get(key)) for key in keys)
    if any(price is None for price in prices):
        return None
    return tuple(float(price) for price in prices if price is not None)


def _normalized_1x2(quote: Tuple[float, float, float]) -> Tuple[float, float, float]:
    inverse = tuple(1.0 / price for price in quote)
    total = math.fsum(inverse)
    return tuple(probability / total for probability in inverse)  # type: ignore[return-value]


def _points(home_score: int, away_score: int) -> Tuple[int, int]:
    if home_score > away_score:
        return 3, 0
    if home_score < away_score:
        return 0, 3
    return 1, 1


def _sort_key(match: Mapping[str, Any], original_order: int) -> Tuple[str, str, str, str, int]:
    return (
        str(match.get("match_date") or ""),
        str(match.get("league_code") or ""),
        str(match.get("home_team_name") or "").casefold(),
        str(match.get("away_team_name") or "").casefold(),
        original_order,
    )


def _direct_h2h_snapshot(counts: Mapping[str, int]) -> Tuple[int, Optional[str], Optional[float]]:
    meetings = sum(counts.values())
    if meetings < DIRECT_H2H_MIN_MEETINGS:
        return meetings, None, None
    maximum = max(counts.values(), default=0)
    modes = [selection for selection in ("H", "D", "A") if counts.get(selection, 0) == maximum]
    if len(modes) != 1:
        return meetings, None, None
    return meetings, modes[0], maximum / meetings


def _prepare_matches(
    matches: Iterable[Mapping[str, Any]],
    *,
    min_cod_matches: int,
) -> List[_PreparedMatch]:
    valid: List[Tuple[Mapping[str, Any], int]] = []
    for original_order, match in enumerate(matches):
        if not isinstance(match, Mapping):
            continue
        home_score = _score(match.get("home_score"))
        away_score = _score(match.get("away_score"))
        match_season = _season(match.get("season"))
        if home_score is None or away_score is None or match_season is None:
            continue
        if not str(match.get("league_code") or "").strip():
            continue
        if not str(match.get("home_team_name") or "").strip() or not str(
            match.get("away_team_name") or ""
        ).strip():
            continue
        valid.append((match, original_order))

    valid.sort(key=lambda item: _sort_key(item[0], item[1]))
    states: MutableMapping[Tuple[str, int, str], _CodState] = {}
    direct_h2h: MutableMapping[Tuple[str, str, str], Dict[str, int]] = {}
    prepared: List[_PreparedMatch] = []
    cursor = 0
    output_order = 0
    while cursor < len(valid):
        kickoff = str(valid[cursor][0].get("match_date") or "")
        end = cursor + 1
        while end < len(valid) and str(valid[end][0].get("match_date") or "") == kickoff:
            end += 1

        # Every snapshot in a kickoff batch is created before any result from
        # that batch is incorporated, preventing same-kickoff leakage.
        batch_updates: List[
            Tuple[
                Tuple[str, int, str],
                Tuple[str, int, str],
                Tuple[float, float, float],
                Tuple[int, int],
            ]
        ] = []
        direct_h2h_updates: List[Tuple[Tuple[str, str, str], str]] = []
        for match, _original_order in valid[cursor:end]:
            match_season = int(match["season"])
            league_code = str(match.get("league_code") or "").strip()
            home_team = str(match.get("home_team_name") or "").strip()
            away_team = str(match.get("away_team_name") or "").strip()
            home_key = (league_code, match_season, home_team.casefold())
            away_key = (league_code, match_season, away_team.casefold())
            direct_h2h_key = (league_code, home_team.casefold(), away_team.casefold())
            home_state = states.setdefault(home_key, _CodState())
            away_state = states.setdefault(away_key, _CodState())
            h2h_counts = direct_h2h.setdefault(
                direct_h2h_key,
                {"H": 0, "D": 0, "A": 0},
            )
            h2h_meetings, h2h_mode, h2h_hit_rate = _direct_h2h_snapshot(h2h_counts)

            extra_value = match.get("extra_data")
            extra: Mapping[str, Any] = extra_value if isinstance(extra_value, Mapping) else {}
            quote_1x2_raw = _complete_quote(
                extra,
                ("b365_home", "b365_draw", "b365_away"),
            )
            quote_ou25_raw = _complete_quote(extra, ("b365_over25", "b365_under25"))
            quote_1x2 = (
                tuple(quote_1x2_raw) if quote_1x2_raw is not None else None
            )
            quote_ou25 = tuple(quote_ou25_raw) if quote_ou25_raw is not None else None
            home_score = int(match["home_score"])
            away_score = int(match["away_score"])

            prepared.append(
                _PreparedMatch(
                    order=output_order,
                    kickoff=kickoff,
                    season=match_season,
                    league_code=league_code,
                    league_name=str(match.get("league_name") or league_code),
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    quote_1x2=quote_1x2,  # type: ignore[arg-type]
                    quote_ou25=quote_ou25,  # type: ignore[arg-type]
                    home_cod=home_state.value(min_cod_matches),
                    away_cod=away_state.value(min_cod_matches),
                    direct_h2h_meetings=h2h_meetings,
                    direct_h2h_mode=h2h_mode,
                    direct_h2h_hit_rate=h2h_hit_rate,
                )
            )
            output_order += 1
            direct_h2h_updates.append((direct_h2h_key, _selection_result(prepared[-1])))

            if quote_1x2 is not None:
                normalized = _normalized_1x2(quote_1x2)  # type: ignore[arg-type]
                batch_updates.append(
                    (home_key, away_key, normalized, _points(home_score, away_score))
                )

        for home_key, away_key, normalized, actual_points in batch_updates:
            home_probability = normalized
            away_probability = (normalized[2], normalized[1], normalized[0])
            states[home_key].add(home_probability, actual_points[0])
            states[away_key].add(away_probability, actual_points[1])
        for direct_h2h_key, result in direct_h2h_updates:
            direct_h2h[direct_h2h_key][result] += 1
        cursor = end

    return prepared


def _profit(won: bool, odds: float) -> float:
    return (odds - 1.0) * (1.0 - PROFIT_HAIRCUT) if won else -1.0


def _selection_result(match: _PreparedMatch) -> str:
    if match.home_score > match.away_score:
        return "H"
    if match.home_score < match.away_score:
        return "A"
    return "D"


def _unique_favourite(quote: Tuple[float, float, float]) -> Optional[str]:
    minimum = min(quote)
    if sum(math.isclose(price, minimum, rel_tol=0.0, abs_tol=1e-12) for price in quote) != 1:
        return None
    return ("H", "D", "A")[quote.index(minimum)]


def _strategy_bet(match: _PreparedMatch, strategy_id: str) -> Optional[_Bet]:
    result = _selection_result(match)
    selection: Optional[str] = None
    odds: Optional[float] = None

    if strategy_id in {"home_win", "draw", "away_win", "favourite"}:
        if match.quote_1x2 is None:
            return None
        if strategy_id == "home_win":
            selection, odds = "H", match.quote_1x2[0]
        elif strategy_id == "draw":
            selection, odds = "D", match.quote_1x2[1]
        elif strategy_id == "away_win":
            selection, odds = "A", match.quote_1x2[2]
        else:
            selection = _unique_favourite(match.quote_1x2)
            if selection is None:
                return None
            odds = match.quote_1x2[("H", "D", "A").index(selection)]
    elif strategy_id in {"over_2_5", "under_2_5"}:
        if match.quote_ou25 is None:
            return None
        is_over = match.home_score + match.away_score > 2.5
        selection = "OVER" if strategy_id == "over_2_5" else "UNDER"
        result = "OVER" if is_over else "UNDER"
        odds = match.quote_ou25[0 if selection == "OVER" else 1]
    elif strategy_id.startswith("cod_home_lte_"):
        if match.quote_1x2 is None or match.home_cod is None:
            return None
        threshold = float(strategy_id.removeprefix("cod_home_lte_").replace("_", "."))
        if match.home_cod > threshold:
            return None
        selection, odds = "H", match.quote_1x2[0]
    elif strategy_id.startswith("cod_away_lte_"):
        if match.quote_1x2 is None or match.away_cod is None:
            return None
        threshold = float(strategy_id.removeprefix("cod_away_lte_").replace("_", "."))
        if match.away_cod > threshold:
            return None
        selection, odds = "A", match.quote_1x2[2]
    elif strategy_id.startswith("cod_home_gte_") and strategy_id.endswith("_back_away"):
        if match.quote_1x2 is None or match.home_cod is None:
            return None
        threshold = float(
            strategy_id.removeprefix("cod_home_gte_")
            .removesuffix("_back_away")
            .replace("_", ".")
        )
        if match.home_cod < threshold:
            return None
        selection, odds = "A", match.quote_1x2[2]
    elif strategy_id.startswith("cod_away_gte_") and strategy_id.endswith("_back_home"):
        if match.quote_1x2 is None or match.away_cod is None:
            return None
        threshold = float(
            strategy_id.removeprefix("cod_away_gte_")
            .removesuffix("_back_home")
            .replace("_", ".")
        )
        if match.away_cod < threshold:
            return None
        selection, odds = "H", match.quote_1x2[0]
    elif strategy_id.startswith("favorite_direct_h2h_agree_0_"):
        if (
            match.quote_1x2 is None
            or match.direct_h2h_mode is None
            or match.direct_h2h_hit_rate is None
            or match.direct_h2h_meetings < DIRECT_H2H_MIN_MEETINGS
        ):
            return None
        threshold = int(strategy_id.rsplit("_", 1)[1]) / 100.0
        favourite = _unique_favourite(match.quote_1x2)
        if favourite is None or favourite != match.direct_h2h_mode:
            return None
        if match.direct_h2h_hit_rate < threshold:
            return None
        selection = favourite
        odds = match.quote_1x2[("H", "D", "A").index(selection)]
    else:  # pragma: no cover - internal candidate registry is closed
        raise ValueError(f"unknown strategy: {strategy_id}")

    won = selection == result
    return _Bet(
        order=match.order,
        season=match.season,
        won=won,
        odds=float(odds),
        profit=_profit(won, float(odds)),
    )


def _rounded(value: float, digits: int = 2) -> float:
    rounded = round(value, digits)
    return 0.0 if rounded == -0.0 else rounded


def _evidence_from_totals(
    *,
    count: int,
    wins: int,
    profit: float,
    profit_squared: float,
    drawdown: float,
    by_season: Mapping[int, float],
    bets_by_season: Mapping[int, int],
) -> Dict[str, Any]:
    if count == 0:
        return {
            "bets": 0,
            "wins": 0,
            "winRatePct": None,
            "profitUnits": None,
            "roiPct": None,
            "ci95Pct": {"lower": None, "upper": None},
            "positiveSeasons": 0,
            "seasons": 0,
            "positiveSeasonRatePct": None,
            "maxDrawdownUnits": None,
            "positiveLowerCi": False,
        }

    mean = profit / count
    season_count = len(by_season)
    if season_count > 1:
        # Season-clustered sandwich standard error. Results within a football
        # season need not be independent, so the promotion gate should not
        # get the much narrower IID interval that a per-bet calculation can
        # produce on a long history.
        cluster_sum = math.fsum(
            (season_profit - mean * bets_by_season[season]) ** 2
            for season, season_profit in by_season.items()
        )
        standard_error = math.sqrt(
            (season_count / (season_count - 1)) * cluster_sum / (count * count)
        )
        lower = mean - _CI_Z * standard_error
        upper = mean + _CI_Z * standard_error
    elif count > 1:
        # A one-season hindsight row has no independent season clusters. Keep
        # the familiar per-bet descriptive interval, but it can never be
        # promoted by the walk-forward gate which requires prior seasons.
        variance_numerator = profit_squared - (profit * profit / count)
        sample_variance = max(0.0, variance_numerator / (count - 1))
        standard_error = math.sqrt(sample_variance / count)
        lower = mean - _CI_Z * standard_error
        upper = mean + _CI_Z * standard_error
    else:
        lower = upper = None

    positive_seasons = sum(value > 0.0 for value in by_season.values())
    lower_positive = lower is not None and lower > 0.0

    return {
        "bets": count,
        "wins": wins,
        "winRatePct": _rounded(wins / count * 100.0),
        "profitUnits": _rounded(profit),
        "roiPct": _rounded(mean * 100.0),
        "ci95Pct": {
            "lower": _rounded(lower * 100.0) if lower is not None else None,
            "upper": _rounded(upper * 100.0) if upper is not None else None,
        },
        "positiveSeasons": positive_seasons,
        "seasons": season_count,
        "positiveSeasonRatePct": _rounded(positive_seasons / season_count * 100.0),
        "maxDrawdownUnits": _rounded(drawdown),
        "positiveLowerCi": lower_positive,
    }


def _evidence(bets: Sequence[_Bet]) -> Dict[str, Any]:
    accumulator = _EvidenceAccumulator()
    accumulator.add(sorted(bets, key=lambda bet: bet.order))
    return accumulator.evidence()


def _pct(numerator: int, denominator: int) -> Optional[float]:
    return _rounded(numerator / denominator * 100.0) if denominator else None


def _descriptive(
    matches: Sequence[_PreparedMatch],
    bets_by_strategy: Mapping[str, Sequence[_Bet]],
) -> Dict[str, Any]:
    match_count = len(matches)
    home_wins = sum(match.home_score > match.away_score for match in matches)
    draws = sum(match.home_score == match.away_score for match in matches)
    away_wins = match_count - home_wins - draws

    results: Dict[str, Any] = {}
    for key, count, strategy_id in (
        ("home", home_wins, "home_win"),
        ("draw", draws, "draw"),
        ("away", away_wins, "away_win"),
    ):
        results[key] = {
            "count": count,
            "ratePct": _pct(count, match_count),
            "b365": _evidence(bets_by_strategy[strategy_id]),
        }

    total_goals = [match.home_score + match.away_score for match in matches]
    btts_yes = sum(match.home_score > 0 and match.away_score > 0 for match in matches)
    goals = {
        "average": _rounded(math.fsum(total_goals) / match_count) if match_count else None,
        "over": {
            format(line, "g"): {
                "count": sum(total > line for total in total_goals),
                "ratePct": _pct(sum(total > line for total in total_goals), match_count),
            }
            for line in GOAL_LINES
        },
        "bothTeamsToScore": {
            "yes": {"count": btts_yes, "ratePct": _pct(btts_yes, match_count)},
            "no": {
                "count": match_count - btts_yes,
                "ratePct": _pct(match_count - btts_yes, match_count),
            },
        },
    }

    complete_1x2 = sum(match.quote_1x2 is not None for match in matches)
    tied_favourites = sum(
        match.quote_1x2 is not None and _unique_favourite(match.quote_1x2) is None
        for match in matches
    )
    complete_ou25 = sum(match.quote_ou25 is not None for match in matches)
    return {
        "label": "descriptive",
        "matches": match_count,
        "results": results,
        "goals": goals,
        "favourite": {
            "completeB365Quotes": complete_1x2,
            "tiedPriceQuotesExcluded": tied_favourites,
            "b365": _evidence(bets_by_strategy["favourite"]),
        },
        "overUnder25": {
            "completeB365Quotes": complete_ou25,
            "over": _evidence(bets_by_strategy["over_2_5"]),
            "under": _evidence(bets_by_strategy["under_2_5"]),
        },
    }


def _rank_hindsight(bets_by_strategy: Mapping[str, Sequence[_Bet]]) -> List[Dict[str, Any]]:
    rows = [
        {"strategyId": strategy_id, "label": "hindsight", **_evidence(bets_by_strategy[strategy_id])}
        for strategy_id in STRATEGY_IDS
    ]
    rows.sort(
        key=lambda row: (
            row["roiPct"] is None,
            -(row["roiPct"] if row["roiPct"] is not None else -math.inf),
            -row["bets"],
            row["strategyId"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        # A same-season ranking is never a deployable profit claim, even when
        # its unadjusted interval happens to exclude zero.
        row["profitClaimAllowed"] = False
    return rows


def _partition_bets(matches: Sequence[_PreparedMatch]) -> Dict[str, List[_Bet]]:
    partitioned: Dict[str, List[_Bet]] = {strategy_id: [] for strategy_id in STRATEGY_IDS}
    for match in matches:
        for strategy_id in STRATEGY_IDS:
            bet = _strategy_bet(match, strategy_id)
            if bet is not None:
                partitioned[strategy_id].append(bet)
    return partitioned


def _walk_forward(
    seasons: Sequence[int],
    bets_by_strategy: Mapping[str, Sequence[_Bet]],
    *,
    min_training_seasons: int,
    min_training_bets: int,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    tested: List[_Bet] = []
    qualified: List[_Bet] = []
    historical: Dict[str, _EvidenceAccumulator] = {
        strategy_id: _EvidenceAccumulator() for strategy_id in STRATEGY_IDS
    }
    by_season: Dict[str, Dict[int, List[_Bet]]] = {
        strategy_id: {} for strategy_id in STRATEGY_IDS
    }
    for strategy_id in STRATEGY_IDS:
        for bet in bets_by_strategy[strategy_id]:
            by_season[strategy_id].setdefault(bet.season, []).append(bet)

    for target_season in seasons:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        for strategy_id in STRATEGY_IDS:
            evidence = historical[strategy_id].evidence()
            if (
                evidence["bets"] >= min_training_bets
                and evidence["seasons"] >= min_training_seasons
                and evidence["ci95Pct"]["lower"] is not None
            ):
                candidates.append((strategy_id, evidence))

        if not candidates:
            rows.append(
                {
                    "season": target_season,
                    "label": "rejected",
                    "selectedStrategyId": None,
                    "reason": "insufficient_prior_seasons_or_bets",
                    "training": None,
                    "test": None,
                }
            )
        else:
            candidates.sort(
                key=lambda item: (
                    -item[1]["ci95Pct"]["lower"],
                    -item[1]["roiPct"],
                    -item[1]["bets"],
                    item[0],
                )
            )
            strategy_id, training_evidence = candidates[0]
            test_bets = by_season[strategy_id].get(target_season, [])
            label = (
                "walk_forward_candidate"
                if training_evidence["positiveLowerCi"]
                else "rejected"
            )
            test_evidence = _evidence(test_bets)
            tested.extend(test_bets)
            if label == "walk_forward_candidate":
                qualified.extend(test_bets)
            rows.append(
                {
                    "season": target_season,
                    "label": label,
                    "selectedStrategyId": strategy_id,
                    "reason": (
                        "positive_training_lower_ci"
                        if label == "walk_forward_candidate"
                        else "training_lower_ci_not_positive"
                    ),
                    "training": training_evidence,
                    "test": test_evidence,
                }
            )

        # Update after selection and evaluation so the current season cannot
        # influence its own choice.
        for strategy_id in STRATEGY_IDS:
            historical[strategy_id].add(by_season[strategy_id].get(target_season, []))

    tested_evidence = _evidence(tested)
    qualified_evidence = _evidence(qualified)
    return {
        "rows": rows,
        "testedPortfolio": {
            "label": (
                "walk_forward_candidate"
                if tested_evidence["positiveLowerCi"]
                else "rejected"
            ),
            **tested_evidence,
        },
        "qualifiedPortfolio": {
            "label": (
                "walk_forward_candidate"
                if qualified_evidence["positiveLowerCi"]
                else "rejected"
            ),
            **qualified_evidence,
        },
    }


def _build_scope(
    matches: Sequence[_PreparedMatch],
    *,
    min_training_seasons: int,
    min_training_bets: int,
) -> Dict[str, Any]:
    bets_by_strategy = _partition_bets(matches)
    matches_by_season: Dict[int, List[_PreparedMatch]] = {}
    for match in matches:
        matches_by_season.setdefault(match.season, []).append(match)
    seasons = sorted(matches_by_season)
    bets_by_season: Dict[str, Dict[int, List[_Bet]]] = {
        strategy_id: {} for strategy_id in STRATEGY_IDS
    }
    for strategy_id in STRATEGY_IDS:
        for bet in bets_by_strategy[strategy_id]:
            bets_by_season[strategy_id].setdefault(bet.season, []).append(bet)
    season_rows: List[Dict[str, Any]] = []
    for season in seasons:
        season_matches = matches_by_season[season]
        season_bets = {
            strategy_id: bets_by_season[strategy_id].get(season, [])
            for strategy_id in STRATEGY_IDS
        }
        season_rows.append(
            {
                "season": season,
                "descriptive": _descriptive(season_matches, season_bets),
                "hindsight": {
                    "label": "hindsight",
                    "selectionBasis": "same_season_results_not_available_pre_match",
                    "ranking": _rank_hindsight(season_bets),
                },
            }
        )

    return {
        "matches": len(matches),
        "seasonCount": len(seasons),
        "seasons": season_rows,
        "allSeasons": {
            "descriptive": _descriptive(matches, bets_by_strategy),
            "strategies": [
                {
                    "strategyId": strategy_id,
                    "label": "descriptive",
                    **_evidence(bets_by_strategy[strategy_id]),
                }
                for strategy_id in STRATEGY_IDS
            ],
        },
        "walkForward": _walk_forward(
            seasons,
            bets_by_strategy,
            min_training_seasons=min_training_seasons,
            min_training_bets=min_training_bets,
        ),
    }


def build_edge_atlas(
    matches: Iterable[Mapping[str, Any]],
    manifest: Optional[Mapping[str, Any]] = None,
    *,
    min_cod_matches: int = DEFAULT_MIN_COD_MATCHES,
    min_training_seasons: int = DEFAULT_MIN_TRAINING_SEASONS,
    min_training_bets: int = DEFAULT_MIN_TRAINING_BETS,
) -> Dict[str, Any]:
    """Build a deterministic global and per-league season evidence atlas.

    ``matches`` should be the rows returned by
    :func:`research.dataset.load_canonical_matches`.  Invalid unfinished rows
    are ignored.  Selection for season ``S`` is fitted exclusively on bets
    from seasons earlier than ``S``.
    """

    for name, value, minimum in (
        ("min_cod_matches", min_cod_matches, 1),
        ("min_training_seasons", min_training_seasons, 1),
        ("min_training_bets", min_training_bets, 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")

    prepared = _prepare_matches(matches, min_cod_matches=min_cod_matches)
    global_scope = _build_scope(
        prepared,
        min_training_seasons=min_training_seasons,
        min_training_bets=min_training_bets,
    )

    leagues: List[Dict[str, Any]] = []
    for league_code in sorted({match.league_code for match in prepared}):
        league_matches = [match for match in prepared if match.league_code == league_code]
        league_name = next(
            (match.league_name for match in league_matches if match.league_name),
            league_code,
        )
        leagues.append(
            {
                "code": league_code,
                "name": league_name,
                **_build_scope(
                    league_matches,
                    min_training_seasons=min_training_seasons,
                    min_training_bets=min_training_bets,
                ),
            }
        )

    source = dict(manifest or {})
    return {
        "schemaVersion": 1,
        "dataset": {
            "id": source.get("dataset_id"),
            "sourceId": source.get("source_dataset_id", source.get("dataset_id")),
            "sourceRows": source.get("raw_rows", len(prepared)),
            "rejectedSourceRows": source.get("invalid_rows", 0),
            "leagueMismatchRows": source.get("league_mismatch_rows", 0),
            "duplicateSourceRows": source.get("duplicates", 0),
            "matches": len(prepared),
            "startSeason": min((match.season for match in prepared), default=None),
            "endSeason": max((match.season for match in prepared), default=None),
            "leagueCount": len(leagues),
        },
        "methodology": {
            "executionOdds": "Bet365 named pre-closing quote only",
            "closingOddsExecution": False,
            "closingOddsRole": "benchmark only; not executed in this atlas",
            "genericOddsFallback": False,
            "maxOddsExecution": False,
            "profitHaircutPct": PROFIT_HAIRCUT * 100.0,
            "roiCi": "two-sided 95% season-clustered interval; per-bet descriptive fallback within one season",
            "cod": {
                "definition": "P(simulated_points < actual_points) + 0.5 * P(equal)",
                "distribution": "exact point-total convolution",
                "history": "prior complete Bet365-quoted matches in the same team season",
                "minimumPriorMatches": min_cod_matches,
                "sameKickoffIsolation": True,
                "thresholds": list(COD_THRESHOLDS),
            },
            "directH2hFavouriteAgreement": {
                "history": "prior same-league meetings in the identical home/away direction",
                "minimumPriorMeetings": DIRECT_H2H_MIN_MEETINGS,
                "uniqueModeRequired": True,
                "uniqueB365FavouriteRequired": True,
                "favouriteMustEqualMode": True,
                "hitRateThresholds": list(DIRECT_H2H_THRESHOLDS),
                "sameKickoffIsolation": True,
            },
            "walkForward": {
                "selection": "highest prior-season lower 95% ROI bound among fixed candidates",
                "minimumPriorSeasons": min_training_seasons,
                "minimumPriorBets": min_training_bets,
                "fixedCandidates": list(STRATEGY_IDS),
                "positiveClaimRule": "never promote unless the lower training CI is above zero",
            },
            "labels": [
                "descriptive",
                "hindsight",
                "walk_forward_candidate",
                "rejected",
            ],
        },
        "global": global_scope,
        "leagues": leagues,
    }


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def write_edge_atlas(payload: Mapping[str, Any], output: str | Path) -> Dict[str, str]:
    """Atomically write canonical JSON and a matching SHA-256 sidecar."""

    output_path = Path(output).expanduser().resolve()
    content = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    checksum = hashlib.sha256(content).hexdigest()
    checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
    checksum_content = f"{checksum}  {output_path.name}\n".encode("ascii")

    _atomic_write(output_path, content)
    _atomic_write(checksum_path, checksum_content)
    return {
        "output": str(output_path),
        "checksumFile": str(checksum_path),
        "sha256": checksum,
    }


def validate_public_source(
    matches: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    """Fail closed unless a public build uses the complete canonical cache."""

    from research.dataset import MAX_SEASON, MIN_SEASON, assert_public_canonical_coverage

    assert_public_canonical_coverage(
        matches,
        manifest,
        start_season=MIN_SEASON,
        end_season=MAX_SEASON,
    )


def build_public_edge_atlas() -> Dict[str, Any]:
    """Build the public atlas only from the fully accounted canonical range."""

    from research.dataset import CSV_LEAGUES, MAX_SEASON, MIN_SEASON, load_canonical_matches

    matches, manifest = load_canonical_matches(start=MIN_SEASON, end=MAX_SEASON)
    validate_public_source(matches, manifest)
    payload = build_edge_atlas(matches, manifest)
    expected_leagues = {item["code"] for item in CSV_LEAGUES.values()}
    observed_leagues = {league["code"] for league in payload["leagues"]}
    observed_seasons = {row["season"] for row in payload["global"]["seasons"]}
    if (
        payload["dataset"]["matches"] != manifest.get("rows")
        or payload["dataset"]["startSeason"] != MIN_SEASON
        or payload["dataset"]["endSeason"] != MAX_SEASON
        or observed_leagues != expected_leagues
        or observed_seasons != set(range(MIN_SEASON, MAX_SEASON + 1))
    ):
        raise RuntimeError("public Edge Atlas failed its post-build coverage contract")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic league-by-season AIBets edge atlas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"JSON output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    # Import lazily so importing this research module never reads files or
    # initializes the production CSV client.
    payload = build_public_edge_atlas()
    result = write_edge_atlas(payload, args.output)
    print(json.dumps(result, sort_keys=True))
    return 0


__all__ = [
    "COD_THRESHOLDS",
    "DIRECT_H2H_THRESHOLDS",
    "INVERSE_COD_STRATEGY_IDS",
    "STRATEGY_IDS",
    "build_edge_atlas",
    "build_public_edge_atlas",
    "main",
    "validate_public_source",
    "write_edge_atlas",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
