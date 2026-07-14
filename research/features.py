"""Leakage-free point-in-time football features for the research pipeline.

The builder in this module deliberately separates *feature evaluation* from
*state updates*.  Every scored fixture updates football state, including old
fixtures without betting prices, while fixtures sharing the same kickoff are
all evaluated against the same pre-kickoff snapshot.

The input contract is the normalized match dictionary produced by
``FootballDataCSVClient``.  A few harmless aliases are accepted so that saved
canonical exports can be replayed without reshaping them first.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
from itertools import groupby
import math
from typing import Deque, Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from research.asian_handicap import valid_asian_handicap_line


DEFAULT_WINDOWS = (5, 10, 20)
ONE_X_TWO_BASES = ("primary", "b365", "avg", "max", "close")
OU25_BASES = ("primary", "b365", "pinnacle", "avg", "max", "close")
ASIAN_HANDICAP_BASES = ("b365", "pinnacle", "avg", "max", "close")

_COMPETITION_GROUPS = {
    "PL": "ENG",
    "ELC": "ENG",
    "BL1": "GER",
    "BL2": "GER",
}

_DEFAULT_LEAGUE_PRIORS = {
    "home_goals": 1.45,
    "away_goals": 1.15,
    "home_win_rate": 0.45,
    "draw_rate": 0.27,
    "away_win_rate": 0.28,
    "over25_rate": 0.50,
    "btts_rate": 0.50,
}


@dataclass(frozen=True)
class _TeamObservation:
    goals_for: int
    goals_against: int
    points: int
    over25: int
    btts: int


@dataclass(frozen=True)
class _LeagueObservation:
    home_goals: int
    away_goals: int
    home_win: int
    draw: int
    away_win: int
    over25: int
    btts: int


@dataclass
class _LeagueState:
    max_history: int
    history: Deque[_LeagueObservation] = field(init=False)
    home_goals: float = 0.0
    away_goals: float = 0.0
    home_win: float = 0.0
    draw: float = 0.0
    away_win: float = 0.0
    over25: float = 0.0
    btts: float = 0.0

    def __post_init__(self) -> None:
        self.history = deque()

    def append(self, observation: _LeagueObservation) -> None:
        if len(self.history) >= self.max_history:
            self._apply(self.history.popleft(), -1.0)
        self.history.append(observation)
        self._apply(observation, 1.0)

    def _apply(self, observation: _LeagueObservation, direction: float) -> None:
        self.home_goals += direction * observation.home_goals
        self.away_goals += direction * observation.away_goals
        self.home_win += direction * observation.home_win
        self.draw += direction * observation.draw
        self.away_win += direction * observation.away_win
        self.over25 += direction * observation.over25
        self.btts += direction * observation.btts


@dataclass
class _TeamState:
    max_history: int
    overall: Deque[_TeamObservation] = field(init=False)
    home: Deque[_TeamObservation] = field(init=False)
    away: Deque[_TeamObservation] = field(init=False)
    last_kickoff: pd.Timestamp | None = None

    def __post_init__(self) -> None:
        self.overall = deque(maxlen=self.max_history)
        self.home = deque(maxlen=self.max_history)
        self.away = deque(maxlen=self.max_history)


def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _valid_price(value: object) -> float | None:
    number = _as_float(value)
    return number if number is not None and number > 1.0 else None


def _complete_quote(
    source: str,
    values: Sequence[object],
    outcomes: Sequence[str],
) -> Dict[str, object] | None:
    prices = [_valid_price(value) for value in values]
    if any(price is None for price in prices):
        return None
    return {
        "source": source,
        **{outcome: price for outcome, price in zip(outcomes, prices)},
    }


def _first_complete_quote(
    candidates: Iterable[tuple[str, Sequence[object]]],
    outcomes: Sequence[str],
) -> Dict[str, object] | None:
    for source, values in candidates:
        quote = _complete_quote(source, values, outcomes)
        if quote is not None:
            return quote
    return None


def extract_1x2_quotes(match: Mapping[str, object]) -> Dict[str, Dict[str, object] | None]:
    """Return complete 1X2 quote sets without cross-source price mixing."""

    extra = match.get("extra_data")
    extra = extra if isinstance(extra, Mapping) else {}
    outcomes = ("home", "draw", "away")

    b365 = _complete_quote(
        "bet365_open",
        (extra.get("b365_home"), extra.get("b365_draw"), extra.get("b365_away")),
        outcomes,
    )
    avg = _complete_quote(
        "market_average_open",
        (
            extra.get("avg_home_odds"),
            extra.get("avg_draw_odds"),
            extra.get("avg_away_odds"),
        ),
        outcomes,
    )
    maximum = _complete_quote(
        "market_max_open",
        (
            extra.get("max_home_odds"),
            extra.get("max_draw_odds"),
            extra.get("max_away_odds"),
        ),
        outcomes,
    )
    close = _first_complete_quote(
        (
            (
                "bet365_close",
                (
                    extra.get("b365_close_home"),
                    extra.get("b365_close_draw"),
                    extra.get("b365_close_away"),
                ),
            ),
            (
                "pinnacle_close",
                (
                    extra.get("pinnacle_close_home"),
                    extra.get("pinnacle_close_draw"),
                    extra.get("pinnacle_close_away"),
                ),
            ),
            (
                "market_average_close",
                (
                    extra.get("avg_close_home_odds"),
                    extra.get("avg_close_draw_odds"),
                    extra.get("avg_close_away_odds"),
                ),
            ),
            (
                "market_max_close",
                (
                    extra.get("max_close_home_odds"),
                    extra.get("max_close_draw_odds"),
                    extra.get("max_close_away_odds"),
                ),
            ),
        ),
        outcomes,
    )

    return {
        "primary": _complete_quote(
            "normalized_primary",
            (match.get("home_odds"), match.get("draw_odds"), match.get("away_odds")),
            outcomes,
        ),
        "b365": b365,
        "avg": avg,
        "max": maximum,
        "close": close,
    }


def extract_ou25_quotes(match: Mapping[str, object]) -> Dict[str, Dict[str, object] | None]:
    """Return complete over/under 2.5 quote sets.

    ``primary`` follows the project's historical convention: average market
    open, then Bet365, then Pinnacle.  Each fallback is an entire two-way
    quote; an over price is never paired with an under price from elsewhere.
    """

    extra = match.get("extra_data")
    extra = extra if isinstance(extra, Mapping) else {}
    outcomes = ("over25", "under25")

    b365 = _complete_quote(
        "bet365_open",
        (extra.get("b365_over25"), extra.get("b365_under25")),
        outcomes,
    )
    pinnacle = _complete_quote(
        "pinnacle_open",
        (extra.get("pinnacle_over25"), extra.get("pinnacle_under25")),
        outcomes,
    )
    avg = _complete_quote(
        "market_average_open",
        (extra.get("avg_over25"), extra.get("avg_under25")),
        outcomes,
    )
    maximum = _complete_quote(
        "market_max_open",
        (extra.get("max_over25"), extra.get("max_under25")),
        outcomes,
    )
    close = _first_complete_quote(
        (
            (
                "bet365_close",
                (extra.get("b365_close_over25"), extra.get("b365_close_under25")),
            ),
            (
                "pinnacle_close",
                (
                    extra.get("pinnacle_close_over25"),
                    extra.get("pinnacle_close_under25"),
                ),
            ),
            (
                "market_average_close",
                (extra.get("avg_close_over25"), extra.get("avg_close_under25")),
            ),
            (
                "market_max_close",
                (extra.get("max_close_over25"), extra.get("max_close_under25")),
            ),
        ),
        outcomes,
    )

    return {
        "primary": _first_complete_quote(
            (
                ("market_average_open", (extra.get("avg_over25"), extra.get("avg_under25"))),
                ("bet365_open", (extra.get("b365_over25"), extra.get("b365_under25"))),
                ("pinnacle_open", (extra.get("pinnacle_over25"), extra.get("pinnacle_under25"))),
            ),
            outcomes,
        ),
        "b365": b365,
        "avg": avg,
        "max": maximum,
        "close": close,
        "pinnacle": pinnacle,
    }


def _complete_asian_quote(
    source: str,
    home_line: object,
    home_price: object,
    away_price: object,
) -> Dict[str, object] | None:
    """Build one coherent two-way quote without repairing malformed lines."""

    line = valid_asian_handicap_line(home_line)
    quote = _complete_quote(source, (home_price, away_price), ("home", "away"))
    if line is None or quote is None:
        return None
    quote["home_line"] = line
    quote["away_line"] = -line
    return quote


def extract_asian_handicap_quotes(
    match: Mapping[str, object],
) -> Dict[str, Dict[str, object] | None]:
    """Return coherent Asian Handicap quotes with the source's home line.

    Football-Data defines ``AHh``/``BbAHh`` as the home-team handicap.  The
    2003/04-2004/05 files instead pair Bet365 prices with its own ``B365AH``
    line. Closing prices are paired only with the separate ``AHCh`` line.
    Market averages/maxima are research proxies, never executable quotes.
    """

    extra = match.get("extra_data")
    extra = extra if isinstance(extra, Mapping) else {}
    open_line = extra.get("asian_handicap_line")
    close_line = extra.get("asian_handicap_close_line")

    b365_line = extra.get("b365_asian_line")
    if valid_asian_handicap_line(b365_line) is None:
        b365_line = open_line

    b365 = _complete_asian_quote(
        "bet365_open",
        b365_line,
        extra.get("b365_asian_home"),
        extra.get("b365_asian_away"),
    )
    pinnacle = _complete_asian_quote(
        "pinnacle_open",
        open_line,
        extra.get("pinnacle_asian_home"),
        extra.get("pinnacle_asian_away"),
    )
    average = _complete_asian_quote(
        "market_average_open",
        open_line,
        extra.get("avg_asian_home"),
        extra.get("avg_asian_away"),
    )
    maximum = _complete_asian_quote(
        "market_max_open",
        open_line,
        extra.get("max_asian_home"),
        extra.get("max_asian_away"),
    )
    closing_candidates = (
        _complete_asian_quote(
            "bet365_close",
            close_line,
            extra.get("b365_close_asian_home"),
            extra.get("b365_close_asian_away"),
        ),
        _complete_asian_quote(
            "pinnacle_close",
            close_line,
            extra.get("pinnacle_close_asian_home"),
            extra.get("pinnacle_close_asian_away"),
        ),
        _complete_asian_quote(
            "market_average_close",
            close_line,
            extra.get("avg_close_asian_home"),
            extra.get("avg_close_asian_away"),
        ),
        _complete_asian_quote(
            "market_max_close",
            close_line,
            extra.get("max_close_asian_home"),
            extra.get("max_close_asian_away"),
        ),
    )
    return {
        "b365": b365,
        "pinnacle": pinnacle,
        "avg": average,
        "max": maximum,
        "close": next((quote for quote in closing_candidates if quote is not None), None),
    }


def _no_vig(prices: Sequence[float]) -> list[float]:
    inverses = [1.0 / price for price in prices]
    total = sum(inverses)
    return [value / total for value in inverses]


def _flatten_quotes(
    row: MutableMapping[str, object],
    market: str,
    quotes: Mapping[str, Dict[str, object] | None],
    bases: Sequence[str],
    outcomes: Sequence[str],
) -> None:
    for basis in bases:
        prefix = f"odds_{market}_{basis}"
        probability_prefix = f"market_{market}_{basis}"
        quote = quotes.get(basis)
        row[f"{prefix}_available"] = int(quote is not None)
        row[f"{prefix}_source"] = quote.get("source") if quote else None
        if quote is None:
            for outcome in outcomes:
                row[f"{prefix}_{outcome}"] = math.nan
                row[f"{probability_prefix}_{outcome}_prob"] = math.nan
            continue

        prices = [float(quote[outcome]) for outcome in outcomes]
        probabilities = _no_vig(prices)
        for outcome, price, probability in zip(outcomes, prices, probabilities):
            row[f"{prefix}_{outcome}"] = price
            row[f"{probability_prefix}_{outcome}_prob"] = probability


def _flatten_asian_handicap_quotes(
    row: MutableMapping[str, object],
    quotes: Mapping[str, Dict[str, object] | None],
) -> None:
    """Attach lines, complete prices and no-vig probabilities to one row."""

    for basis in ASIAN_HANDICAP_BASES:
        prefix = f"odds_ah_{basis}"
        probability_prefix = f"market_ah_{basis}"
        quote = quotes.get(basis)
        row[f"{prefix}_available"] = int(quote is not None)
        row[f"{prefix}_source"] = quote.get("source") if quote else None
        if quote is None:
            row[f"{prefix}_home_line"] = math.nan
            row[f"{prefix}_away_line"] = math.nan
            for outcome in ("home", "away"):
                row[f"{prefix}_{outcome}"] = math.nan
                row[f"{probability_prefix}_{outcome}_prob"] = math.nan
            continue
        row[f"{prefix}_home_line"] = float(quote["home_line"])
        row[f"{prefix}_away_line"] = float(quote["away_line"])
        prices = [float(quote[outcome]) for outcome in ("home", "away")]
        probabilities = _no_vig(prices)
        for outcome, price, probability in zip(("home", "away"), prices, probabilities):
            row[f"{prefix}_{outcome}"] = price
            row[f"{probability_prefix}_{outcome}_prob"] = probability


def _mean_or_nan(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _shrunk_mean(values: Sequence[float], prior: float, strength: float) -> float:
    return (sum(values) + strength * prior) / (len(values) + strength)


def _history_features(
    history: Sequence[_TeamObservation],
    window: int,
    priors: Mapping[str, float],
    strength: float,
) -> Dict[str, float | int]:
    sample = list(history)[-window:]
    vectors = {
        "form_ppg": [float(item.points) for item in sample],
        "goals_for": [float(item.goals_for) for item in sample],
        "goals_against": [float(item.goals_against) for item in sample],
        "over25_rate": [float(item.over25) for item in sample],
        "btts_rate": [float(item.btts) for item in sample],
    }
    result: Dict[str, float | int] = {"matches": len(sample)}
    for name, values in vectors.items():
        result[name] = _mean_or_nan(values)
        result[f"{name}_shrunk"] = _shrunk_mean(values, priors[name], strength)
    return result


def _history_features_for_windows(
    history: Sequence[_TeamObservation],
    windows: Sequence[int],
    priors: Mapping[str, float],
    strength: float,
    *,
    include_unshrunk: bool,
) -> Dict[int, Dict[str, float | int]]:
    """Compute all rolling windows from one suffix accumulation.

    The original implementation rebuilt five Python lists for every window.
    Across 115k matches that dominated runtime.  One reverse pass over the
    (at most 20-match) deque preserves identical values with far less work.
    """

    observations = list(history)
    max_count = min(len(observations), max(windows))
    recent = list(reversed(observations[-max_count:]))
    cumulative = {
        "form_ppg": [0.0],
        "goals_for": [0.0],
        "goals_against": [0.0],
        "over25_rate": [0.0],
        "btts_rate": [0.0],
    }
    for item in recent:
        values = {
            "form_ppg": float(item.points),
            "goals_for": float(item.goals_for),
            "goals_against": float(item.goals_against),
            "over25_rate": float(item.over25),
            "btts_rate": float(item.btts),
        }
        for name, value in values.items():
            cumulative[name].append(cumulative[name][-1] + value)

    result: Dict[int, Dict[str, float | int]] = {}
    for window in windows:
        count = min(max_count, window)
        values: Dict[str, float | int] = {"matches": count}
        for name, sums in cumulative.items():
            total = sums[count]
            if include_unshrunk:
                values[name] = total / count if count else math.nan
            values[f"{name}_shrunk"] = (total + strength * priors[name]) / (count + strength)
        result[window] = values
    return result


def _add_team_history_features(
    row: MutableMapping[str, object],
    side: str,
    scope: str,
    history: Sequence[_TeamObservation],
    windows: Sequence[int],
    priors: Mapping[str, float],
    strength: float,
    *,
    include_unshrunk: bool,
) -> Dict[int, Dict[str, float | int]]:
    all_values = _history_features_for_windows(
        history,
        windows,
        priors,
        strength,
        include_unshrunk=include_unshrunk,
    )
    for window, values in all_values.items():
        for feature, value in values.items():
            row[f"{side}_{scope}_{feature}_{window}"] = value
    return all_values


def _league_priors(
    state: _LeagueState,
    strength: float,
) -> Dict[str, float]:
    n = len(state.history)

    def estimate(total: float, prior_name: str) -> float:
        return (total + strength * _DEFAULT_LEAGUE_PRIORS[prior_name]) / (n + strength)

    return {
        "history_matches": float(n),
        "home_goals": estimate(state.home_goals, "home_goals"),
        "away_goals": estimate(state.away_goals, "away_goals"),
        "home_win_rate": estimate(state.home_win, "home_win_rate"),
        "draw_rate": estimate(state.draw, "draw_rate"),
        "away_win_rate": estimate(state.away_win, "away_win_rate"),
        "over25_rate": estimate(state.over25, "over25_rate"),
        "btts_rate": estimate(state.btts, "btts_rate"),
    }


def _team_priors(league: Mapping[str, float], venue: str) -> Dict[str, float]:
    home_ppg = 3.0 * league["home_win_rate"] + league["draw_rate"]
    away_ppg = 3.0 * league["away_win_rate"] + league["draw_rate"]
    if venue == "home":
        goals_for = league["home_goals"]
        goals_against = league["away_goals"]
        form_ppg = home_ppg
    elif venue == "away":
        goals_for = league["away_goals"]
        goals_against = league["home_goals"]
        form_ppg = away_ppg
    else:
        goals_for = goals_against = (league["home_goals"] + league["away_goals"]) / 2.0
        form_ppg = (home_ppg + away_ppg) / 2.0
    return {
        "form_ppg": form_ppg,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "over25_rate": league["over25_rate"],
        "btts_rate": league["btts_rate"],
    }


def _poisson_probabilities(home_lambda: float, away_lambda: float) -> Dict[str, float]:
    max_goals = 10
    home_mass = [math.exp(-home_lambda)]
    away_mass = [math.exp(-away_lambda)]
    for goals in range(1, max_goals + 1):
        home_mass.append(home_mass[-1] * home_lambda / goals)
        away_mass.append(away_mass[-1] * away_lambda / goals)

    # O(max_goals) cumulative calculation instead of the previous 11x11
    # score-grid loop.  It is mathematically identical for the truncated grid.
    home = draw = 0.0
    away_cumulative = 0.0
    for goals, home_probability in enumerate(home_mass):
        home += home_probability * away_cumulative
        draw += home_probability * away_mass[goals]
        away_cumulative += away_mass[goals]
    captured = sum(home_mass) * sum(away_mass)
    away = max(0.0, captured - home - draw)
    if captured > 0:
        home, draw, away = home / captured, draw / captured, away / captured

    total_lambda = home_lambda + away_lambda
    under25 = math.exp(-total_lambda) * (
        1.0 + total_lambda + total_lambda**2 / 2.0
    )
    return {
        "poisson_home_prob": home,
        "poisson_draw_prob": draw,
        "poisson_away_prob": away,
        "poisson_over25_prob": 1.0 - under25,
        "poisson_under25_prob": under25,
        "poisson_btts_prob": (1.0 - math.exp(-home_lambda))
        * (1.0 - math.exp(-away_lambda)),
    }


def _poisson_features(
    home_stats: Mapping[str, float | int],
    away_stats: Mapping[str, float | int],
    league: Mapping[str, float],
) -> Dict[str, float]:
    league_home = max(0.20, league["home_goals"])
    league_away = max(0.20, league["away_goals"])
    home_attack = float(home_stats["goals_for_shrunk"])
    home_defence = float(home_stats["goals_against_shrunk"])
    away_attack = float(away_stats["goals_for_shrunk"])
    away_defence = float(away_stats["goals_against_shrunk"])

    home_lambda = max(0.10, min(5.0, home_attack * away_defence / league_home))
    away_lambda = max(0.10, min(5.0, away_attack * home_defence / league_away))
    return {
        "poisson_home_lambda": home_lambda,
        "poisson_away_lambda": away_lambda,
        **_poisson_probabilities(home_lambda, away_lambda),
    }


def _team_key(league: str, name: object) -> str:
    """Scope names by a stable competition group while retaining promotions."""

    competition_group = _COMPETITION_GROUPS.get(league, league or "UNKNOWN")
    normalized_name = " ".join(str(name or "").casefold().split())
    return f"{competition_group}|{normalized_name}"


def _match_identity(
    match: Mapping[str, object],
    kickoff: pd.Timestamp,
    league: str,
    home: str,
    away: str,
) -> tuple[str, str]:
    natural_key = f"{league}|{kickoff.isoformat()}|{home}|{away}"
    api_id = match.get("api_id")
    if api_id is not None and str(api_id).strip():
        return str(api_id), natural_key
    digest = hashlib.sha256(natural_key.encode("utf-8")).hexdigest()[:20]
    return digest, natural_key


def _rest_days(state: _TeamState, kickoff: pd.Timestamp) -> float:
    if state.last_kickoff is None:
        return math.nan
    return max(0.0, (kickoff - state.last_kickoff).total_seconds() / 86_400.0)


def _score(value: object) -> int | None:
    number = _as_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _kickoff(value: object) -> pd.Timestamp | None:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def build_feature_frame(
    matches: Iterable[Mapping[str, object]],
    *,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    team_prior_strength: float = 5.0,
    league_prior_strength: float = 40.0,
    league_window: int = 500,
    elo_initial: float = 1500.0,
    elo_home_advantage: float = 65.0,
    elo_k_factor: float = 20.0,
    include_unshrunk_history: bool = True,
) -> pd.DataFrame:
    """Build one pre-match feature row for every dated, scored fixture.

    Parameters are intentionally explicit so research runs can record their
    feature-state assumptions.  Outcomes and prices are attached to the row
    for later evaluation, but neither is read when producing its features.
    Closing prices are data columns only and must not be used by a strategy
    claiming to place bets before those prices were observable.
    """

    clean_windows = tuple(sorted({int(window) for window in windows if int(window) > 0}))
    if not clean_windows:
        raise ValueError("windows must contain at least one positive integer")
    if team_prior_strength <= 0 or league_prior_strength <= 0:
        raise ValueError("shrinkage strengths must be positive")
    if league_window <= 0:
        raise ValueError("league_window must be positive")

    prepared = []
    source_count = 0
    skipped_unscored = 0
    skipped_invalid_date = 0
    for ordinal, match in enumerate(matches):
        source_count += 1
        home_score = _score(match.get("home_score"))
        away_score = _score(match.get("away_score"))
        if home_score is None or away_score is None:
            skipped_unscored += 1
            continue
        kickoff = _kickoff(match.get("match_date") or match.get("kickoff"))
        if kickoff is None:
            skipped_invalid_date += 1
            continue
        prepared.append((kickoff, ordinal, match, home_score, away_score))
    prepared.sort(key=lambda item: (item[0].value, item[1]))

    max_history = max(clean_windows)
    teams: Dict[str, _TeamState] = {}
    leagues: Dict[str, _LeagueState] = defaultdict(lambda: _LeagueState(league_window))
    elo: Dict[str, float] = {}
    # Column-oriented accumulation avoids retaining 115k large Python dicts
    # (roughly 190 keys each) before pandas can compact them into arrays.
    columns: Dict[str, list[object]] = {}

    def team_state(key: str) -> _TeamState:
        if key not in teams:
            teams[key] = _TeamState(max_history=max_history)
        return teams[key]

    for _, kickoff_group in groupby(prepared, key=lambda item: item[0].value):
        group = list(kickoff_group)
        team_updates = []
        league_updates = []
        elo_updates: Dict[str, float] = defaultdict(float)

        for kickoff, _, match, home_score, away_score in group:
            league = str(match.get("league_code") or match.get("league") or "")
            home = str(match.get("home_team_name") or match.get("home") or "")
            away = str(match.get("away_team_name") or match.get("away") or "")
            if not home or not away:
                continue
            home_key = _team_key(league, home)
            away_key = _team_key(league, away)
            home_state = team_state(home_key)
            away_state = team_state(away_key)
            league_history = leagues[league]
            league_values = _league_priors(league_history, league_prior_strength)
            overall_priors = _team_priors(league_values, "overall")
            home_priors = _team_priors(league_values, "home")
            away_priors = _team_priors(league_values, "away")

            match_id, natural_key = _match_identity(match, kickoff, league, home, away)
            total_goals = home_score + away_score
            target_1x2 = "H" if home_score > away_score else "A" if away_score > home_score else "D"
            target_index = {"H": 0, "D": 1, "A": 2}[target_1x2]
            row: Dict[str, object] = {
                "match_id": match_id,
                "api_id": match.get("api_id"),
                "natural_key": natural_key,
                "match_date": kickoff,
                "league_code": league,
                "league_name": match.get("league_name"),
                "season": match.get("season"),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "target_1x2": target_1x2,
                "target_1x2_index": target_index,
                "target_total_goals": total_goals,
                "target_over25": int(total_goals > 2.5),
                "target_btts": int(home_score > 0 and away_score > 0),
                "league_history_matches": int(league_values["history_matches"]),
                "league_home_goals": league_values["home_goals"],
                "league_away_goals": league_values["away_goals"],
                "league_home_win_rate": league_values["home_win_rate"],
                "league_draw_rate": league_values["draw_rate"],
                "league_away_win_rate": league_values["away_win_rate"],
                "league_over25_rate": league_values["over25_rate"],
                "league_btts_rate": league_values["btts_rate"],
            }

            _add_team_history_features(
                row,
                "home",
                "overall",
                home_state.overall,
                clean_windows,
                overall_priors,
                team_prior_strength,
                include_unshrunk=include_unshrunk_history,
            )
            _add_team_history_features(
                row,
                "away",
                "overall",
                away_state.overall,
                clean_windows,
                overall_priors,
                team_prior_strength,
                include_unshrunk=include_unshrunk_history,
            )
            home_venue_features = _add_team_history_features(
                row,
                "home",
                "venue",
                home_state.home,
                clean_windows,
                home_priors,
                team_prior_strength,
                include_unshrunk=include_unshrunk_history,
            )
            away_venue_features = _add_team_history_features(
                row,
                "away",
                "venue",
                away_state.away,
                clean_windows,
                away_priors,
                team_prior_strength,
                include_unshrunk=include_unshrunk_history,
            )

            home_rest = _rest_days(home_state, kickoff)
            away_rest = _rest_days(away_state, kickoff)
            row.update(
                {
                    "home_rest_days": home_rest,
                    "away_rest_days": away_rest,
                    "home_rest_days_capped": min(home_rest, 30.0)
                    if math.isfinite(home_rest)
                    else math.nan,
                    "away_rest_days_capped": min(away_rest, 30.0)
                    if math.isfinite(away_rest)
                    else math.nan,
                    "rest_days_difference": home_rest - away_rest
                    if math.isfinite(home_rest) and math.isfinite(away_rest)
                    else math.nan,
                }
            )

            home_elo = elo.get(home_key, elo_initial)
            away_elo = elo.get(away_key, elo_initial)
            expected_home = 1.0 / (
                1.0 + 10.0 ** ((away_elo - home_elo - elo_home_advantage) / 400.0)
            )
            row.update(
                {
                    "home_elo": home_elo,
                    "away_elo": away_elo,
                    "elo_difference": home_elo - away_elo,
                    "elo_difference_with_home_advantage": home_elo
                    + elo_home_advantage
                    - away_elo,
                    "elo_expected_home_score": expected_home,
                }
            )
            row.update(
                _poisson_features(
                    home_venue_features[max_history],
                    away_venue_features[max_history],
                    league_values,
                )
            )
            _flatten_quotes(
                row,
                "1x2",
                extract_1x2_quotes(match),
                ONE_X_TWO_BASES,
                ("home", "draw", "away"),
            )
            _flatten_quotes(
                row,
                "ou25",
                extract_ou25_quotes(match),
                OU25_BASES,
                ("over25", "under25"),
            )
            _flatten_asian_handicap_quotes(row, extract_asian_handicap_quotes(match))
            if not columns:
                columns = {name: [] for name in row}
            if row.keys() != columns.keys():
                missing = set(columns).difference(row)
                unexpected = set(row).difference(columns)
                raise RuntimeError(
                    f"feature schema changed while building frame; missing={missing}, unexpected={unexpected}"
                )
            for name, value in row.items():
                columns[name].append(value)

            home_points = 3 if home_score > away_score else 1 if home_score == away_score else 0
            away_points = 3 if away_score > home_score else 1 if home_score == away_score else 0
            over25 = int(total_goals > 2.5)
            btts = int(home_score > 0 and away_score > 0)
            team_updates.append(
                (
                    home_state,
                    away_state,
                    kickoff,
                    _TeamObservation(home_score, away_score, home_points, over25, btts),
                    _TeamObservation(away_score, home_score, away_points, over25, btts),
                )
            )
            league_updates.append(
                (
                    league_history,
                    _LeagueObservation(
                        home_score,
                        away_score,
                        int(home_score > away_score),
                        int(home_score == away_score),
                        int(away_score > home_score),
                        over25,
                        btts,
                    ),
                )
            )

            actual_home_score = (
                1.0
                if home_score > away_score
                else 0.5
                if home_score == away_score
                else 0.0
            )
            margin_multiplier = 1.0 + math.log1p(abs(home_score - away_score)) / 2.0
            elo_delta = elo_k_factor * margin_multiplier * (actual_home_score - expected_home)
            elo_updates[home_key] += elo_delta
            elo_updates[away_key] -= elo_delta

        # The whole kickoff group is committed only after every row is built.
        for home_state, away_state, kickoff, home_obs, away_obs in team_updates:
            home_state.overall.append(home_obs)
            home_state.home.append(home_obs)
            away_state.overall.append(away_obs)
            away_state.away.append(away_obs)
            home_state.last_kickoff = kickoff
            away_state.last_kickoff = kickoff
        for history, observation in league_updates:
            history.append(observation)
        for key, delta in elo_updates.items():
            elo[key] = elo.get(key, elo_initial) + delta

    frame = pd.DataFrame(columns)
    frame.attrs.update(
        {
            "point_in_time": True,
            "source_matches": source_count,
            "scored_dated_matches": len(prepared),
            "skipped_unscored": skipped_unscored,
            "skipped_invalid_date": skipped_invalid_date,
            "windows": clean_windows,
            "include_unshrunk_history": include_unshrunk_history,
        }
    )
    return frame


__all__ = [
    "ASIAN_HANDICAP_BASES",
    "DEFAULT_WINDOWS",
    "ONE_X_TWO_BASES",
    "OU25_BASES",
    "build_feature_frame",
    "extract_1x2_quotes",
    "extract_asian_handicap_quotes",
    "extract_ou25_quotes",
]
