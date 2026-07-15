"""Point-in-time historical pattern zoo for the public AIBets site.

The nested model research engine answers whether a learned betting policy can
survive walk-forward evaluation.  This module answers a different, simpler
product question: what happened to fixed, understandable football patterns
over time?  It deliberately keeps those two concerns separate.

Every signal is computed from matches strictly before the fixture being
evaluated.  Matches with the same kickoff are evaluated against one shared
snapshot and committed afterwards.  Betting returns are only materialised
from a complete Bet365 opening quote; outcome-only patterns remain visible but
cannot acquire synthetic profit.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata
from typing import Any, Deque, Dict, Iterable, Mapping, MutableMapping, Sequence

from research.dataset import LATEST_COMPLETE_SEASON
from research.features import extract_1x2_quotes, extract_ou25_quotes
from research.metrics import max_drawdown


SCHEMA_VERSION = 2
DEFAULT_PUBLIC_PATH = Path(__file__).resolve().parents[1] / "data" / "strategy_zoo_public.json"
DEFAULT_PUBLIC_CHECKSUM_PATH = DEFAULT_PUBLIC_PATH.with_suffix(".sha256")
# Schema v2 adds audited annual rankings and descriptive season profiles.  The
# compact canonical payload is ~1.05 MB, still far below Cloudflare KV's value
# limit; retain a tight regression guard with modest headroom.
MAX_PUBLIC_BYTES = 1_250_000
ODDS_HAIRCUT = 0.01
ROI_BOOTSTRAP_RESAMPLES = 2_000
ROI_BOOTSTRAP_SEED = 20260714
HINDSIGHT_MINIMUM_SEASON_BETS = 200
WALK_FORWARD_MINIMUM_PRIOR_BETS = 1_000
WALK_FORWARD_MINIMUM_PRIOR_PRICED_SEASONS = 5
WALK_FORWARD_ACTIVATION_THRESHOLD_ROI_PCT = 0.0

_COMPETITION_GROUPS = {
    "PL": "ENG",
    "ELC": "ENG",
    "BL1": "GER",
    "BL2": "GER",
}
_OUTCOME_LABELS = ("H", "D", "A")
_GOAL_BUCKETS = ("0-1", "2", "3", "4+")

FAVOURITE_BANDS = (
    (1.01, 1.30),
    (1.30, 1.50),
    (1.50, 1.80),
    (1.80, 2.20),
    (2.20, 3.01),
)
DRAW_BANDS = (
    (2.00, 3.00),
    (3.00, 3.50),
    (3.50, 4.00),
    (4.00, 5.00),
    (5.00, 10.01),
)
OUTSIDER_BANDS = (
    (3.00, 4.00),
    (4.00, 6.00),
    (6.00, 10.00),
    (10.00, 51.00),
)
TOTALS_BANDS = (
    (1.20, 1.60),
    (1.60, 1.80),
    (1.80, 2.00),
    (2.00, 2.30),
    (2.30, 4.01),
)


class StrategyZooValidationError(ValueError):
    """Raised when a public strategy-zoo artifact violates its contract."""


@dataclass(frozen=True)
class StrategyDefinition:
    id: str
    title: str
    family: str
    market: str
    rule: Mapping[str, Any]
    comparison: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _PreparedMatch:
    kickoff: datetime
    season: int
    league: str
    home: str
    away: str
    home_key: str
    away_key: str
    home_score: int
    away_score: int
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class _Event:
    kickoff: datetime
    season: int
    correct: bool
    selection: str
    actual: str
    raw_odds: float | None = None
    decimal_odds: float | None = None

    @property
    def profit(self) -> float | None:
        if self.decimal_odds is None:
            return None
        return self.decimal_odds - 1.0 if self.correct else -1.0


def _normalize_team(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.casefold().split())


def _as_score(value: object) -> int | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _as_kickoff(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_season(value: object, kickoff: datetime) -> int:
    try:
        season = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        season = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
    return season


def _prepare(matches: Iterable[Mapping[str, Any]]) -> list[_PreparedMatch]:
    prepared: list[_PreparedMatch] = []
    for match in matches:
        kickoff = _as_kickoff(match.get("match_date") or match.get("kickoff"))
        home_score = _as_score(match.get("home_score"))
        away_score = _as_score(match.get("away_score"))
        league = str(match.get("league_code") or match.get("league") or "").strip()
        home = str(match.get("home_team_name") or match.get("home") or "").strip()
        away = str(match.get("away_team_name") or match.get("away") or "").strip()
        if kickoff is None or home_score is None or away_score is None or not league or not home or not away:
            continue
        prepared.append(
            _PreparedMatch(
                kickoff=kickoff,
                season=_as_season(match.get("season"), kickoff),
                league=league,
                home=home,
                away=away,
                home_key=_normalize_team(home),
                away_key=_normalize_team(away),
                home_score=home_score,
                away_score=away_score,
                raw=match,
            )
        )
    prepared.sort(key=lambda item: (item.kickoff, item.league, item.home_key, item.away_key))
    return prepared


def _slug_number(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def _band_id(prefix: str, lower: float, upper: float) -> str:
    return f"{prefix}_{_slug_number(lower)}_{_slug_number(upper)}"


def _band_rule(selection: str, lower: float, upper: float) -> Dict[str, Any]:
    return {
        "selection": selection,
        "minimumDecimalOddsInclusive": lower,
        "maximumDecimalOddsExclusive": upper,
        "quote": "bet365_open_complete_market",
    }


def _comparison(
    group_id: str | None = None,
    role: str | None = None,
    opposite_strategy_id: str | None = None,
    *,
    kind: str = "none",
    same_opportunity_set: bool = False,
) -> Dict[str, Any]:
    return {
        "groupId": group_id,
        "role": role,
        "oppositeStrategyId": opposite_strategy_id,
        "kind": kind,
        "sameOpportunitySet": same_opportunity_set,
    }


def _definition_comparison(definition: StrategyDefinition) -> Dict[str, Any]:
    return dict(definition.comparison or _comparison())


def strategy_definitions() -> tuple[StrategyDefinition, ...]:
    """Return the fixed strategy catalogue in stable display order."""

    definitions = [
        StrategyDefinition(
            "all_unique_favourites",
            "Alle entydige holdfavoritter",
            "baselines",
            "1x2",
            {
                "selection": "unique_lower_priced_home_or_away_team",
                "tiePolicy": "skip",
                "quote": "bet365_open_complete_market",
            },
            _comparison(
                "all_unique_price_extremes",
                "favourite",
                "all_unique_outsiders",
                kind="contrast",
                same_opportunity_set=False,
            ),
        ),
        StrategyDefinition(
            "all_unique_outsiders",
            "Alle entydige 1X2-outsidere",
            "baselines",
            "1x2",
            {
                "selection": "unique_highest_priced_1x2_outcome",
                "tiePolicy": "skip",
                "quote": "bet365_open_complete_market",
            },
            _comparison(
                "all_unique_price_extremes",
                "outsider",
                "all_unique_favourites",
                kind="contrast",
                same_opportunity_set=False,
            ),
        ),
        StrategyDefinition(
            "all_home_wins",
            "Hjemmesejr i alle kampe",
            "baselines",
            "1x2",
            {"selection": "home", "quote": "bet365_open_complete_market"},
            _comparison(
                "all_1x2_outcomes",
                "home",
                kind="contrast",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "all_draws",
            "Uafgjort i alle kampe",
            "baselines",
            "1x2",
            {"selection": "draw", "quote": "bet365_open_complete_market"},
            _comparison(
                "all_1x2_outcomes",
                "draw",
                kind="contrast",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "all_away_wins",
            "Udesejr i alle kampe",
            "baselines",
            "1x2",
            {"selection": "away", "quote": "bet365_open_complete_market"},
            _comparison(
                "all_1x2_outcomes",
                "away",
                kind="contrast",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "all_over15",
            "Over 1,5 mål i alle kampe",
            "baselines",
            "over_under_1_5",
            {"selection": "over15", "quote": None, "outcomeOnly": True},
            _comparison(
                "all_ou15",
                "over15",
                "all_under15",
                kind="binary_complement",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "all_under15",
            "Under 1,5 mål i alle kampe",
            "baselines",
            "over_under_1_5",
            {"selection": "under15", "quote": None, "outcomeOnly": True},
            _comparison(
                "all_ou15",
                "under15",
                "all_over15",
                kind="binary_complement",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "all_over25",
            "Over 2,5 mål i alle kampe",
            "baselines",
            "over_under_2_5",
            {"selection": "over25", "quote": "bet365_open_complete_market"},
            _comparison(
                "all_ou25",
                "over25",
                "all_under25",
                kind="binary_complement",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "all_under25",
            "Under 2,5 mål i alle kampe",
            "baselines",
            "over_under_2_5",
            {"selection": "under25", "quote": "bet365_open_complete_market"},
            _comparison(
                "all_ou25",
                "under25",
                "all_over25",
                kind="binary_complement",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "directed_h2h_dominance",
            "Samme hjemmehold dominerer H2H",
            "rivalry",
            "1x2",
            {
                "history": "same_home_team_vs_same_away_team",
                "minimumPriorMeetings": 4,
                "minimumDominantOutcomeRatePct": 70.0,
                "uniqueModeRequired": True,
            },
        ),
        StrategyDefinition(
            "pair_h2h_team_dominance",
            "Hold dominerer indbyrdes uanset bane",
            "rivalry",
            "1x2",
            {
                "history": "same_team_pair_any_venue",
                "minimumPriorMeetings": 5,
                "minimumDominantWinnerRatePct": 65.0,
                "uniqueModeRequired": True,
            },
        ),
        StrategyDefinition(
            "h2h_over25_dominance",
            "H2H: over 2,5 mål gentager sig",
            "goals",
            "over_under_2_5",
            {"minimumPriorMeetings": 5, "minimumOverRatePct": 70.0},
            _comparison(
                "h2h_over25_signal",
                "follow",
                "fade_h2h_over25_dominance",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "fade_h2h_over25_dominance",
            "Modspil: H2H over 2,5-signal",
            "goals",
            "over_under_2_5",
            {
                "signalStrategyId": "h2h_over25_dominance",
                "selection": "under25",
                "quote": "bet365_open_complete_market",
            },
            _comparison(
                "h2h_over25_signal",
                "fade",
                "h2h_over25_dominance",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "h2h_under25_dominance",
            "H2H: under 2,5 mål gentager sig",
            "goals",
            "over_under_2_5",
            {"minimumPriorMeetings": 5, "minimumUnderRatePct": 70.0},
            _comparison(
                "h2h_under25_signal",
                "follow",
                "fade_h2h_under25_dominance",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "fade_h2h_under25_dominance",
            "Modspil: H2H under 2,5-signal",
            "goals",
            "over_under_2_5",
            {
                "signalStrategyId": "h2h_under25_dominance",
                "selection": "over25",
                "quote": "bet365_open_complete_market",
            },
            _comparison(
                "h2h_under25_signal",
                "fade",
                "h2h_under25_dominance",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "league_over25_extreme",
            "Ligaens rullende over 2,5-signal",
            "goals",
            "over_under_2_5",
            {"rollingLeagueMatches": 500, "minimumHistory": 200, "minimumOverRatePct": 60.0},
            _comparison(
                "league_over25_signal",
                "follow",
                "fade_league_over25_extreme",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "fade_league_over25_extreme",
            "Modspil: ligaens over 2,5-signal",
            "goals",
            "over_under_2_5",
            {
                "signalStrategyId": "league_over25_extreme",
                "selection": "under25",
                "quote": "bet365_open_complete_market",
            },
            _comparison(
                "league_over25_signal",
                "fade",
                "league_over25_extreme",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "league_under25_extreme",
            "Ligaens rullende under 2,5-signal",
            "goals",
            "over_under_2_5",
            {"rollingLeagueMatches": 500, "minimumHistory": 200, "minimumUnderRatePct": 60.0},
            _comparison(
                "league_under25_signal",
                "follow",
                "fade_league_under25_extreme",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "fade_league_under25_extreme",
            "Modspil: ligaens under 2,5-signal",
            "goals",
            "over_under_2_5",
            {
                "signalStrategyId": "league_under25_extreme",
                "selection": "over25",
                "quote": "bet365_open_complete_market",
            },
            _comparison(
                "league_under25_signal",
                "fade",
                "league_under25_extreme",
                kind="mirrored_signal",
                same_opportunity_set=True,
            ),
        ),
        StrategyDefinition(
            "league_goal_bucket_mode",
            "Ligaens mest typiske målinterval",
            "score",
            "goal_bucket",
            {
                "buckets": list(_GOAL_BUCKETS),
                "rollingLeagueMatches": 500,
                "minimumHistory": 200,
                "uniqueModeRequired": True,
            },
        ),
        StrategyDefinition(
            "league_exact_score_mode",
            "Ligaens hyppigste præcise score",
            "score",
            "exact_score",
            {"rollingLeagueMatches": 1_000, "minimumHistory": 200, "uniqueModeRequired": True},
        ),
        StrategyDefinition(
            "directed_h2h_exact_score_mode",
            "H2H's hyppigste præcise score",
            "score",
            "exact_score",
            {
                "history": "same_home_team_vs_same_away_team",
                "minimumPriorMeetings": 5,
                "minimumModeRatePct": 40.0,
                "uniqueModeRequired": True,
            },
        ),
    ]

    for lower, upper in FAVOURITE_BANDS:
        definitions.append(
            StrategyDefinition(
                _band_id("favourite", lower, upper),
                f"Holdfavorit @ {lower:.2f}–<{upper:.2f}",
                "odds",
                "1x2",
                _band_rule("lower_priced_home_or_away_team", lower, upper),
            )
        )
    for lower, upper in DRAW_BANDS:
        definitions.append(
            StrategyDefinition(
                _band_id("draw", lower, upper),
                f"Uafgjort @ {lower:.2f}–<{upper:.2f}",
                "draws",
                "1x2",
                _band_rule("draw", lower, upper),
            )
        )
    for lower, upper in OUTSIDER_BANDS:
        definitions.append(
            StrategyDefinition(
                _band_id("outsider", lower, upper),
                f"1X2-outsider @ {lower:.2f}–<{upper:.2f}",
                "odds",
                "1x2",
                _band_rule("highest_priced_1x2_outcome", lower, upper),
            )
        )
    for side, title in (("over25", "Over 2,5"), ("under25", "Under 2,5")):
        for lower, upper in TOTALS_BANDS:
            definitions.append(
                StrategyDefinition(
                    _band_id(side, lower, upper),
                    f"{title} @ {lower:.2f}–<{upper:.2f}",
                    "goals",
                    "over_under_2_5",
                    _band_rule(side, lower, upper),
                )
            )
    return tuple(definitions)


def _outcome(match: _PreparedMatch) -> str:
    if match.home_score > match.away_score:
        return "H"
    if match.away_score > match.home_score:
        return "A"
    return "D"


def _goal_bucket(total: int) -> str:
    if total <= 1:
        return "0-1"
    if total == 2:
        return "2"
    if total == 3:
        return "3"
    return "4+"


def _competition_group(league: str) -> str:
    return _COMPETITION_GROUPS.get(league, league)


def _directed_key(match: _PreparedMatch) -> tuple[str, str, str]:
    return (_competition_group(match.league), match.home_key, match.away_key)


def _pair_key(match: _PreparedMatch) -> tuple[str, str, str]:
    first, second = sorted((match.home_key, match.away_key))
    return (_competition_group(match.league), first, second)


def _unique_mode(counter: Mapping[Any, int]) -> tuple[Any | None, int, float]:
    total = sum(counter.values())
    if not total:
        return None, 0, 0.0
    ranked = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, total, ranked[0][1] / total
    return ranked[0][0], total, ranked[0][1] / total


def _verified_prices(match: Mapping[str, Any], market: str) -> Mapping[str, float] | None:
    quotes = extract_1x2_quotes(match) if market == "1x2" else extract_ou25_quotes(match)
    quote = quotes.get("b365")
    if not isinstance(quote, Mapping) or quote.get("source") != "bet365_open":
        return None
    labels = ("home", "draw", "away") if market == "1x2" else ("over25", "under25")
    prices: Dict[str, float] = {}
    for label in labels:
        try:
            price = float(quote[label])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(price) or price <= 1.0:
            return None
        prices[label] = price
    return prices


def _adjust_price(raw_odds: float) -> float:
    return 1.0 + (raw_odds - 1.0) * (1.0 - ODDS_HAIRCUT)


def _event(
    match: _PreparedMatch,
    selection: str,
    actual: str,
    raw_odds: float | None = None,
) -> _Event:
    return _Event(
        kickoff=match.kickoff,
        season=match.season,
        correct=selection == actual,
        selection=selection,
        actual=actual,
        raw_odds=raw_odds,
        decimal_odds=_adjust_price(raw_odds) if raw_odds is not None else None,
    )


def _within_band(price: float, band: tuple[float, float]) -> bool:
    return band[0] <= price < band[1]


def _unique_price_extreme(prices: Mapping[str, float], *, highest: bool) -> str | None:
    ranked = sorted(prices.items(), key=lambda item: item[1], reverse=highest)
    if len(ranked) > 1 and math.isclose(ranked[0][1], ranked[1][1], abs_tol=1e-12):
        return None
    return ranked[0][0]


def _team_favourite(prices: Mapping[str, float]) -> str | None:
    """Return the uniquely shorter-priced team, excluding the draw quote."""

    home_price = float(prices["home"])
    away_price = float(prices["away"])
    if math.isclose(home_price, away_price, abs_tol=1e-12):
        return None
    return "home" if home_price < away_price else "away"


def _wilson_interval(hits: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = hits / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return [round(max(0.0, centre - radius) * 100.0, 2), round(min(1.0, centre + radius) * 100.0, 2)]


def _season_cluster_roi_ci(
    priced_events: Sequence[_Event],
    strategy_id: str,
    *,
    resamples: int,
) -> list[float] | None:
    grouped: MutableMapping[int, list[float]] = defaultdict(list)
    for event in priced_events:
        profit = event.profit
        if profit is not None:
            grouped[event.season].append(profit)
    seasons = sorted(grouped)
    if len(seasons) < 2 or resamples < 1:
        return None
    digest = hashlib.sha256(strategy_id.encode("utf-8")).digest()
    seed = ROI_BOOTSTRAP_SEED ^ int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(resamples):
        profit = 0.0
        bets = 0
        for _season_position in seasons:
            selected = seasons[rng.randrange(len(seasons))]
            values = grouped[selected]
            profit += math.fsum(values)
            bets += len(values)
        samples.append(profit / bets if bets else 0.0)
    samples.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(samples) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return samples[lower]
        weight = position - lower
        return samples[lower] * (1.0 - weight) + samples[upper] * weight

    return [round(quantile(0.025) * 100.0, 2), round(quantile(0.975) * 100.0, 2)]


def _round_optional(value: float | None, digits: int = 2) -> float | None:
    return None if value is None else round(value, digits)


def _summary(
    events: Sequence[_Event],
    strategy_id: str,
    *,
    bootstrap_resamples: int,
    include_roi_ci: bool = True,
) -> Dict[str, Any]:
    ordered = sorted(events, key=lambda item: (item.kickoff, item.selection))
    opportunities = len(ordered)
    hits = sum(event.correct for event in ordered)
    priced = [event for event in ordered if event.decimal_odds is not None]
    wins = sum(event.correct for event in priced)
    profits = [event.profit for event in priced]
    clean_profits = [float(value) for value in profits if value is not None]
    profit = math.fsum(clean_profits)
    roi = profit / len(priced) if priced else None
    hit_rate = hits / opportunities if opportunities else None
    average_odds = (
        math.fsum(float(event.raw_odds) for event in priced if event.raw_odds is not None) / len(priced)
        if priced
        else None
    )
    seasons = sorted({event.season for event in ordered})
    priced_by_season: MutableMapping[int, list[float]] = defaultdict(list)
    for event in priced:
        if event.profit is not None:
            priced_by_season[event.season].append(event.profit)
    profit_by_kickoff: MutableMapping[datetime, list[float]] = defaultdict(list)
    for event in priced:
        if event.profit is not None:
            profit_by_kickoff[event.kickoff].append(event.profit)
    settlement_profits = [
        math.fsum(profit_by_kickoff[kickoff])
        for kickoff in sorted(profit_by_kickoff)
    ]
    positive_seasons = sum(math.fsum(values) > 0.0 for values in priced_by_season.values())
    pnl_available = bool(priced)
    return {
        "opportunities": opportunities,
        "hits": hits,
        "hitRatePct": _round_optional(hit_rate * 100.0 if hit_rate is not None else None),
        "hitRateCi95Pct": _wilson_interval(hits, opportunities),
        "bets": len(priced),
        "wins": wins,
        "stakeUnits": float(len(priced)),
        "pnlAvailable": pnl_available,
        "pnlAvailabilityReason": (
            None
            if pnl_available
            else "no_verified_pre_match_odds_for_market"
            if opportunities
            else "no_qualifying_opportunities"
        ),
        "oddsCoveragePct": round(len(priced) / opportunities * 100.0, 2) if opportunities else 0.0,
        "averageOpeningOdds": _round_optional(average_odds, 3),
        "profitUnits": round(profit, 2) if priced else None,
        "roiPct": _round_optional(roi * 100.0 if roi is not None else None),
        "roiCi95Pct": (
            _season_cluster_roi_ci(priced, strategy_id, resamples=bootstrap_resamples)
            if priced and include_roi_ci
            else None
        ),
        # Simultaneous fixtures settle as one batch.  Ordering them by team or
        # selection would invent an intra-kickoff equity path and drawdown.
        "maxDrawdownUnits": round(max_drawdown(settlement_profits), 2) if priced else None,
        "activeSeasons": len(seasons),
        "pricedSeasons": len(priced_by_season),
        "positivePricedSeasons": positive_seasons,
        "positivePricedSeasonRatePct": (
            round(positive_seasons / len(priced_by_season) * 100.0, 2)
            if priced_by_season
            else None
        ),
    }


def _status(overall: Mapping[str, Any]) -> tuple[str, list[str]]:
    opportunities = int(overall["opportunities"])
    bets = int(overall["bets"])
    priced_seasons = int(overall["pricedSeasons"])
    roi = overall["roiPct"]
    roi_ci = overall["roiCi95Pct"]
    positive_rate = overall["positivePricedSeasonRatePct"]
    if not opportunities:
        return "unavailable", ["no_qualifying_point_in_time_opportunities"]
    if not bets:
        return "descriptive_only", ["no_verified_pre_match_odds_for_this_market"]
    if bets < 200 or priced_seasons < 5:
        return "insufficient", ["fewer_than_200_bets_or_5_priced_seasons"]
    if (
        roi is not None
        and roi > 0.0
        and roi_ci is not None
        and roi_ci[0] > 0.0
        and positive_rate is not None
        and positive_rate >= 60.0
    ):
        return "historical_positive_unconfirmed", [
            "season_cluster_roi_ci_above_zero",
            "not_confirmed_by_untouched_holdout_or_multiple_testing_control",
        ]
    if roi_ci is not None and roi_ci[1] < 0.0:
        return "historical_negative", ["season_cluster_roi_ci_below_zero"]
    if roi is not None and roi > 0.0:
        return "historical_positive_unconfirmed", ["positive_point_estimate_but_confidence_interval_crosses_zero"]
    return "historical_negative_unconfirmed", ["non_positive_point_estimate_without_conclusive_negative_interval"]


def _strategy_payload(
    definition: StrategyDefinition,
    events: Sequence[_Event],
    seasons: Sequence[int],
    *,
    bootstrap_resamples: int,
    complete_through_season: int,
) -> Dict[str, Any]:
    overall = _summary(events, definition.id, bootstrap_resamples=bootstrap_resamples)
    status, reasons = _status(overall)
    by_season: MutableMapping[int, list[_Event]] = defaultdict(list)
    for event in events:
        by_season[event.season].append(event)
    yearly = []
    for season in seasons:
        season_events = by_season.get(season, [])
        metrics = _summary(
            season_events,
            f"{definition.id}:{season}",
            bootstrap_resamples=bootstrap_resamples,
            include_roi_ci=False,
        )
        metrics.update(
            {
                "season": season,
                "label": f"{season}/{str(season + 1)[-2:]}",
                "available": bool(season_events) and season <= complete_through_season,
                "quarantined": season > complete_through_season,
                "quarantineReason": (
                    "incomplete_local_snapshot" if season > complete_through_season else None
                ),
                "availabilityReason": (
                    "incomplete_local_snapshot"
                    if season > complete_through_season
                    else None
                    if season_events
                    else "no_qualifying_opportunities"
                ),
            }
        )
        if season > complete_through_season:
            metrics["pnlAvailabilityReason"] = "incomplete_local_snapshot"
        yearly.append(metrics)
    active = [row["season"] for row in yearly if row["available"]]
    return {
        "id": definition.id,
        "title": definition.title,
        "family": definition.family,
        "market": definition.market,
        "rule": dict(definition.rule),
        "comparison": _definition_comparison(definition),
        "status": status,
        "statusReasons": reasons,
        "guaranteed": False,
        "firstActiveSeason": min(active) if active else None,
        "lastActiveSeason": max(active) if active else None,
        "overall": overall,
        "yearly": yearly,
    }


def _rivalry_payload(
    rivalry_events: Mapping[tuple[str, str, str], Sequence[_Event]],
    pair_records: Mapping[tuple[str, str, str], Mapping[str, int]],
    pair_seasons: Mapping[tuple[str, str, str], Sequence[int]],
    display_names: Mapping[tuple[str, str, str], Mapping[str, str]],
    *,
    bootstrap_resamples: int,
    limit: int = 40,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for key, record in pair_records.items():
        meetings = sum(record.values())
        if meetings < 10:
            continue
        identifier = "|".join(key)
        events = list(rivalry_events.get(key, ()))
        metrics = _summary(events, f"rivalry:{identifier}", bootstrap_resamples=bootstrap_resamples)
        selections = Counter(event.selection for event in events)
        names = display_names.get(key, {})
        teams = [names.get(key[1], key[1]), names.get(key[2], key[2])]
        first_wins = int(record.get(key[1], 0))
        second_wins = int(record.get(key[2], 0))
        draws = int(record.get("__draw__", 0))
        if first_wins == second_wins:
            continue
        else:
            dominant_key = key[1] if first_wins > second_wins else key[2]
        if dominant_key == key[1]:
            wins, losses = first_wins, second_wins
            team, opponent = teams[0], teams[1]
        elif dominant_key == key[2]:
            wins, losses = second_wins, first_wins
            team, opponent = teams[1], teams[0]
        observed_seasons = list(pair_seasons.get(key, ()))
        rows.append(
            {
                "id": hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:16],
                "competitionGroup": key[0],
                "teams": teams,
                "relationshipLabel": "head_to_head_pair_not_verified_derby",
                "team": team,
                "opponent": opponent,
                "dominantTeam": team,
                "meetings": meetings,
                "record": {"wins": wins, "draws": draws, "losses": losses},
                "winRatePct": round(wins / meetings * 100.0, 2),
                "winRateCi95Pct": _wilson_interval(wins, meetings),
                "unbeatenRatePct": round((wins + draws) / meetings * 100.0, 2),
                "perfectWinRecord": wins == meetings,
                "perfectUnbeatenRecord": losses == 0,
                "mostFrequentPointInTimePick": selections.most_common(1)[0][0] if selections else None,
                "status": "descriptive_only",
                "guaranteed": False,
                "pointInTimeSignal": metrics,
                "firstSeason": min(observed_seasons) if observed_seasons else None,
                "lastSeason": max(observed_seasons) if observed_seasons else None,
            }
        )
    rows.sort(
        key=lambda row: (
            (row["winRateCi95Pct"] or [0.0])[0],
            row["meetings"],
            row["winRatePct"],
        ),
        reverse=True,
    )
    return rows[:limit]


def _one_sided_binomial_p_value(wins: int, trials: int, null_probability: float = 0.5) -> float:
    """Exact P(X >= wins) under a binomial null."""

    if trials <= 0 or wins < 0 or wins > trials:
        raise ValueError("invalid binomial counts")
    return math.fsum(
        math.comb(trials, successes)
        * null_probability**successes
        * (1.0 - null_probability) ** (trials - successes)
        for successes in range(wins, trials + 1)
    )


def _h2h_validation_audit(
    matches: Sequence[_PreparedMatch],
    *,
    validation_end_season: int,
) -> Dict[str, Any]:
    """Reproduce the discovery/validation H2H multiple-testing audit.

    Discovery fixes a dominant team using seasons through 2019/20.  Only then
    is its 2020/21-through-latest-complete record tested.  Benjamini-Hochberg is applied over
    every candidate with enough validation meetings; this tests repeatable
    win-rate evidence, not betting profitability.
    """

    discovery: MutableMapping[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    validation: MutableMapping[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    display: MutableMapping[tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    for match in matches:
        key = _pair_key(match)
        display[key][match.home_key] = match.home
        display[key][match.away_key] = match.away
        outcome = _outcome(match)
        winner = (
            match.home_key
            if outcome == "H"
            else match.away_key
            if outcome == "A"
            else "__draw__"
        )
        if match.season <= 2019:
            discovery[key][winner] += 1
        elif 2020 <= match.season <= validation_end_season:
            validation[key][winner] += 1

    candidates: list[Dict[str, Any]] = []
    for key, discovery_record in discovery.items():
        discovery_meetings = sum(discovery_record.values())
        if discovery_meetings < 12:
            continue
        first_wins = int(discovery_record.get(key[1], 0))
        second_wins = int(discovery_record.get(key[2], 0))
        if first_wins == second_wins:
            continue
        dominant = key[1] if first_wins > second_wins else key[2]
        discovery_wins = max(first_wins, second_wins)
        if discovery_wins / discovery_meetings < 0.65:
            continue
        validation_record = validation.get(key, Counter())
        validation_meetings = sum(validation_record.values())
        if validation_meetings < 6:
            continue
        validation_wins = int(validation_record.get(dominant, 0))
        other = key[2] if dominant == key[1] else key[1]
        validation_draws = int(validation_record.get("__draw__", 0))
        validation_losses = int(validation_record.get(other, 0))
        names = display[key]
        candidates.append(
            {
                "team": names.get(dominant, dominant),
                "opponent": names.get(other, other),
                "competitionGroup": key[0],
                "discovery": {
                    "meetings": discovery_meetings,
                    "wins": discovery_wins,
                    "winRatePct": round(discovery_wins / discovery_meetings * 100.0, 2),
                },
                "validation": {
                    "meetings": validation_meetings,
                    "wins": validation_wins,
                    "draws": validation_draws,
                    "losses": validation_losses,
                    "winRatePct": round(validation_wins / validation_meetings * 100.0, 2),
                },
                "pValue": _one_sided_binomial_p_value(validation_wins, validation_meetings),
            }
        )

    ordered = sorted(enumerate(candidates), key=lambda item: (item[1]["pValue"], item[0]))
    running_q = 1.0
    for reverse_position in range(len(ordered) - 1, -1, -1):
        original_index, candidate = ordered[reverse_position]
        rank = reverse_position + 1
        running_q = min(running_q, candidate["pValue"] * len(ordered) / rank)
        candidates[original_index]["qValue"] = min(1.0, running_q)
    for candidate in candidates:
        candidate["confirmedAtQ05"] = candidate.get("qValue", 1.0) <= 0.05
        candidate["pValue"] = round(candidate["pValue"], 8)
        candidate["qValue"] = round(candidate.get("qValue", 1.0), 8)
    ranked = sorted(candidates, key=lambda item: (item["qValue"], item["pValue"], -item["validation"]["meetings"]))
    confirmed = [candidate for candidate in ranked if candidate["confirmedAtQ05"]]
    candidate_tests = [
        {
            "competitionGroup": candidate["competitionGroup"],
            "team": candidate["team"],
            "opponent": candidate["opponent"],
            "validationMeetings": candidate["validation"]["meetings"],
            "validationWins": candidate["validation"]["wins"],
            "pValue": candidate["pValue"],
            "qValue": candidate["qValue"],
            "confirmedAtQ05": candidate["confirmedAtQ05"],
        }
        for candidate in ranked
    ]
    return {
        "multipleTestingCorrectionApplied": True,
        "protocol": {
            "discoverySeasons": "<=2019/20",
            "validationSeasons": (
                f"2020/21-{validation_end_season}/{str(validation_end_season + 1)[-2:]}"
            ),
            "minimumDiscoveryMeetings": 12,
            "minimumDiscoveryWinRatePct": 65.0,
            "minimumValidationMeetings": 6,
            "nullHypothesis": "dominant team win probability <= 0.50",
            "test": "exact one-sided binomial",
            "correction": "Benjamini-Hochberg FDR q<=0.05",
        },
        "candidateCount": len(candidates),
        "candidateTests": candidate_tests,
        "confirmedWinRatePatterns": len(confirmed),
        # A repeatable win rate is not a betting edge: the bookmaker price may
        # already encode the mismatch, and profitability is not tested here.
        "confirmedEdges": 0,
        "status": (
            "NO_CONFIRMED_WIN_RATE_PATTERN"
            if not confirmed
            else "CONFIRMED_WIN_RATE_PATTERNS_NO_BETTING_EDGE"
        ),
        "bettingEdgeTested": False,
        "confirmedBettingEdges": 0,
        "topValidationCandidates": ranked[:10],
        "reason": (
            "Ingen H2H-kandidat overlevede korrektion for de mange samtidige tests."
            if not confirmed
            else (
                f"{len(confirmed)} H2H-mønstre overlevede win-rate-testen, men en "
                "gentagelig sejrsrate er ikke det samme som profit; odds/P&L "
                "kræver en separat validering."
            )
        ),
    }


def _extreme(
    strategies: Sequence[Mapping[str, Any]],
    families: set[str],
    metric: str,
    *,
    highest: bool,
    minimum_bets: int = 200,
) -> Dict[str, Any] | None:
    eligible = [
        strategy
        for strategy in strategies
        if strategy["family"] in families
        and int(strategy["overall"]["bets"]) >= minimum_bets
        and strategy["overall"].get(metric) is not None
    ]
    if not eligible:
        return None
    selected = sorted(
        eligible,
        key=lambda strategy: (strategy["overall"][metric], strategy["overall"]["bets"]),
        reverse=highest,
    )[0]
    return {
        "strategyId": selected["id"],
        "title": selected["title"],
        "bets": selected["overall"]["bets"],
        metric: selected["overall"][metric],
        "status": selected["status"],
    }


def _build_findings(
    strategies: Sequence[Mapping[str, Any]],
    rivalries: Sequence[Mapping[str, Any]],
    score_distribution: Mapping[str, int],
    h2h_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    exact = next(strategy for strategy in strategies if strategy["id"] == "league_exact_score_mode")
    h2h_exact = next(strategy for strategy in strategies if strategy["id"] == "directed_h2h_exact_score_mode")
    perfect = [
        {
            "team": row["team"],
            "opponent": row["opponent"],
            "meetings": row["meetings"],
            "winRateCi95Pct": row["winRateCi95Pct"],
        }
        for row in rivalries
        if row["perfectWinRecord"]
    ]
    score_total = sum(score_distribution.values())
    top_scores = [
        {
            "score": score,
            "matches": count,
            "ratePct": round(count / score_total * 100.0, 2) if score_total else 0.0,
        }
        for score, count in sorted(
            score_distribution.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]
    ]
    return {
        "globalStatus": "NO_CONFIRMED_BETTING_EDGE",
        "h2hValidation": dict(h2h_validation),
        "researchVerdict": "NO_CONFIRMED_BETTING_EDGE",
        "guarantees": {
            "alwaysWinsFound": False,
            "neverWinsFound": False,
            "message": (
                "Ingen historisk stikprøve gør et fremtidigt udfald sikkert; "
                "100% eller 0% observeret er ikke en garanti."
            ),
        },
        "bestOddsRoi": _extreme(strategies, {"odds"}, "roiPct", highest=True),
        "worstOddsRoi": _extreme(strategies, {"odds"}, "roiPct", highest=False),
        "bestDrawRoi": _extreme(strategies, {"draws"}, "roiPct", highest=True),
        "worstDrawRoi": _extreme(strategies, {"draws"}, "roiPct", highest=False),
        "bestGoalsRoi": _extreme(strategies, {"goals"}, "roiPct", highest=True),
        "worstGoalsRoi": _extreme(strategies, {"goals"}, "roiPct", highest=False),
        "exactScoreReliability": {
            "topObservedScores": top_scores,
            "leagueMode": {
                "opportunities": exact["overall"]["opportunities"],
                "hitRatePct": exact["overall"]["hitRatePct"],
                "hitRateCi95Pct": exact["overall"]["hitRateCi95Pct"],
            },
            "directedH2hMode": {
                "opportunities": h2h_exact["overall"]["opportunities"],
                "hitRatePct": h2h_exact["overall"]["hitRatePct"],
                "hitRateCi95Pct": h2h_exact["overall"]["hitRateCi95Pct"],
            },
            "message": (
                "1-1 er historisk den hyppigste score, men selv den rammer kun omtrent "
                "hver ottende kamp. Præcis score er et høj-varians forecast uden "
                "verificerede exact-score-odds i datasættet; der vises derfor ingen P&L."
            ),
        },
        "rivalryScreen": {
            "reportedPairs": len(rivalries),
            "relationshipDefinition": "historical head-to-head pair; not necessarily a derby or cultural rivalry",
            "minimumHistoricalMeetings": 10,
            "minimumOutOfSampleSignals": 3,
            "perfectWinRecordPairs": perfect[:10],
            "perfectWinRecordCount": len(perfect),
            "perfectRecordConclusion": (
                "Ingen holdpar i den rapporterede stikprøve vandt alle mindst 10 møder."
                if not perfect
                else (
                    "Observerede perfekte serier findes, men deres konfidensinterval "
                    "er ikke 100%, og de er ikke fremtidige garantier."
                )
            ),
            "multipleTestingWarning": (
                "Mange holdpar er screenet. Ranglisten er beskrivende og må ikke "
                "læses som en garanti eller et automatisk spil."
            ),
        },
    }


_TOTAL_GOAL_THRESHOLDS = ("0.5", "1.5", "2.5", "3.5", "4.5", "5.5")
_EXACT_TOTAL_GOAL_BUCKETS = ("0", "1", "2", "3", "4", "5", "6+")


def _count_rate(count: int, total: int) -> Dict[str, Any]:
    return {
        "count": count,
        "ratePct": round(count / total * 100.0, 2) if total else None,
    }


def _market_profile_row(
    matches: Sequence[_PreparedMatch],
    *,
    scope: str,
    season: int | None,
    label: str,
) -> Dict[str, Any]:
    scored = len(matches)
    outcomes = Counter(_outcome(match) for match in matches)
    total_goals = [match.home_score + match.away_score for match in matches]
    totals: Dict[str, Any] = {}
    for threshold_label in _TOTAL_GOAL_THRESHOLDS:
        threshold = float(threshold_label)
        over = sum(total > threshold for total in total_goals)
        totals[threshold_label] = {
            "over": _count_rate(over, scored),
            "under": _count_rate(scored - over, scored),
        }

    exact_totals = Counter(str(total) if total < 6 else "6+" for total in total_goals)
    complete_priced = 0
    unique_team_selections = 0
    ties_skipped = 0
    favourite_wins = 0
    favourite_draws = 0
    for match in matches:
        prices = _verified_prices(match.raw, "1x2")
        if prices is None:
            continue
        complete_priced += 1
        favourite = _team_favourite(prices)
        if favourite is None:
            ties_skipped += 1
            continue
        unique_team_selections += 1
        actual = _outcome(match)
        favourite_wins += int(
            (favourite == "home" and actual == "H")
            or (favourite == "away" and actual == "A")
        )
        favourite_draws += int(actual == "D")
    favourite_losses = unique_team_selections - favourite_wins - favourite_draws
    return {
        "scope": scope,
        "season": season,
        "label": label,
        "scoredMatches": scored,
        "oneXTwo": {
            "home": _count_rate(outcomes["H"], scored),
            "draw": _count_rate(outcomes["D"], scored),
            "away": _count_rate(outcomes["A"], scored),
        },
        "totalGoals": totals,
        "exactTotalGoals": {
            bucket: _count_rate(exact_totals[bucket], scored)
            for bucket in _EXACT_TOTAL_GOAL_BUCKETS
        },
        "teamFavourites": {
            "completePricedMatches": complete_priced,
            "uniqueTeamSelections": unique_team_selections,
            "tiesSkipped": ties_skipped,
            "won": favourite_wins,
            "drawn": favourite_draws,
            "lost": favourite_losses,
            "winRatePct": (
                round(favourite_wins / unique_team_selections * 100.0, 2)
                if unique_team_selections
                else None
            ),
            "drawRatePct": (
                round(favourite_draws / unique_team_selections * 100.0, 2)
                if unique_team_selections
                else None
            ),
            "lossRatePct": (
                round(favourite_losses / unique_team_selections * 100.0, 2)
                if unique_team_selections
                else None
            ),
        },
    }


def _market_profile_stability(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evaluated = [row for row in rows if int(row["scoredMatches"]) > 0]

    def metric_stats(values: Sequence[float | None]) -> Dict[str, Any]:
        finite = [float(value) for value in values if value is not None]
        if not finite:
            return {
                "seasonsObserved": 0,
                "meanPct": None,
                "stdDevPctPoints": None,
                "minPct": None,
                "maxPct": None,
                "majoritySeasons": 0,
                "last5MeanPct": None,
                "priorMeanPct": None,
                "deltaPctPoints": None,
                "directionChanges": 0,
                "trendSlopePctPointsPerSeason": None,
            }
        mean = math.fsum(finite) / len(finite)
        variance = math.fsum((value - mean) ** 2 for value in finite) / len(finite)
        last_five = finite[-5:]
        prior = finite[:-5]
        last_five_mean = math.fsum(last_five) / len(last_five)
        prior_mean = math.fsum(prior) / len(prior) if prior else None
        differences = [current - previous for previous, current in zip(finite, finite[1:])]
        directions = [1 if difference > 0 else -1 for difference in differences if not math.isclose(difference, 0.0)]
        direction_changes = sum(first != second for first, second in zip(directions, directions[1:]))
        x_mean = (len(finite) - 1) / 2.0
        denominator = math.fsum((index - x_mean) ** 2 for index in range(len(finite)))
        slope = (
            math.fsum((index - x_mean) * (value - mean) for index, value in enumerate(finite))
            / denominator
            if denominator
            else None
        )
        return {
            "seasonsObserved": len(finite),
            "meanPct": round(mean, 2),
            "stdDevPctPoints": round(math.sqrt(variance), 2),
            "minPct": round(min(finite), 2),
            "maxPct": round(max(finite), 2),
            "majoritySeasons": sum(value > 50.0 for value in finite),
            "last5MeanPct": round(last_five_mean, 2),
            "priorMeanPct": round(prior_mean, 2) if prior_mean is not None else None,
            "deltaPctPoints": (
                round(last_five_mean - prior_mean, 2) if prior_mean is not None else None
            ),
            "directionChanges": direction_changes,
            "trendSlopePctPointsPerSeason": round(slope, 3) if slope is not None else None,
        }

    most_common = {"home": 0, "draw": 0, "away": 0, "tied": 0}
    for row in evaluated:
        counts = {side: int(row["oneXTwo"][side]["count"]) for side in ("home", "draw", "away")}
        maximum = max(counts.values())
        leaders = [side for side, count in counts.items() if count == maximum]
        most_common[leaders[0] if len(leaders) == 1 else "tied"] += 1
    return {
        "evaluatedSeasons": len(evaluated),
        "metrics": {
            "home": metric_stats([row["oneXTwo"]["home"]["ratePct"] for row in evaluated]),
            "draw": metric_stats([row["oneXTwo"]["draw"]["ratePct"] for row in evaluated]),
            "away": metric_stats([row["oneXTwo"]["away"]["ratePct"] for row in evaluated]),
            **{
                f"over{threshold.replace('.', '')}": metric_stats(
                    [row["totalGoals"][threshold]["over"]["ratePct"] for row in evaluated]
                )
                for threshold in _TOTAL_GOAL_THRESHOLDS
            },
            "teamFavouriteWin": metric_stats(
                [row["teamFavourites"]["winRatePct"] for row in evaluated]
            ),
        },
        "recurrence": {
            "mostCommon1x2OutcomeSeasons": most_common,
            "overMajoritySeasons": {
                threshold: sum(
                    row["totalGoals"][threshold]["over"]["ratePct"] is not None
                    and float(row["totalGoals"][threshold]["over"]["ratePct"]) > 50.0
                    for row in evaluated
                )
                for threshold in _TOTAL_GOAL_THRESHOLDS
            },
            "teamFavouriteMajorityWinSeasons": sum(
                row["teamFavourites"]["winRatePct"] is not None
                and float(row["teamFavourites"]["winRatePct"]) > 50.0
                for row in evaluated
            ),
        },
    }


def _build_season_market_profiles(
    matches: Sequence[_PreparedMatch],
    seasons: Sequence[int],
) -> Dict[str, Any]:
    by_season_matches: MutableMapping[int, list[_PreparedMatch]] = defaultdict(list)
    for match in matches:
        by_season_matches[match.season].append(match)
    rows = [
        _market_profile_row(
            by_season_matches.get(season, []),
            scope="season",
            season=season,
            label=f"{season}/{str(season + 1)[-2:]}",
        )
        for season in seasons
    ]
    return {
        "methodology": {
            "descriptiveOnly": True,
            "scoredMatchDenominator": "all_scored_canonical_matches",
            "favouriteDefinition": "unique_lower_home_or_away_bet365_open_price_in_complete_1x2_market",
            "favouriteTiePolicy": "skip",
            "guaranteesOrEdgeClaims": False,
        },
        "allTime": _market_profile_row(
            matches,
            scope="all_time",
            season=None,
            label="Alle sæsoner",
        ),
        "bySeason": rows,
        "stability": _market_profile_stability(rows),
    }


def _selection_policy() -> Dict[str, Any]:
    return {
        "hindsightMinimumSeasonBets": HINDSIGHT_MINIMUM_SEASON_BETS,
        "walkForwardMinimumPriorBets": WALK_FORWARD_MINIMUM_PRIOR_BETS,
        "walkForwardMinimumPriorPricedSeasons": WALK_FORWARD_MINIMUM_PRIOR_PRICED_SEASONS,
        "walkForwardActivationThresholdRoiPct": WALK_FORWARD_ACTIVATION_THRESHOLD_ROI_PCT,
        "walkForwardUsesOnlyPriorSeasons": True,
        "walkForwardTieBreak": "roi_desc_then_strategy_id_asc",
        "cashWhenNoPositiveCandidate": True,
    }


def _build_season_audits(
    strategies: Sequence[Mapping[str, Any]],
    seasons: Sequence[int],
) -> list[Dict[str, Any]]:
    """Build descriptive annual rankings and a causal retrospective selector.

    The hindsight ranking is explicitly descriptive and may use season-S
    results.  The walk-forward choice for S is made solely from rows before S,
    is frozen for the season, and stays in cash unless the best eligible prior
    pooled ROI is strictly positive.
    """

    yearly_by_strategy = {
        str(strategy["id"]): {
            int(row["season"]): row for row in strategy["yearly"]
        }
        for strategy in strategies
    }
    audits: list[Dict[str, Any]] = []
    first_season = min(seasons)
    for season in seasons:
        hindsight_candidates: list[tuple[float, str, Mapping[str, Any]]] = []
        for strategy in strategies:
            identifier = str(strategy["id"])
            row = yearly_by_strategy[identifier][season]
            if int(row["bets"]) < HINDSIGHT_MINIMUM_SEASON_BETS or row["roiPct"] is None:
                continue
            hindsight_candidates.append((float(row["roiPct"]), identifier, row))
        hindsight_candidates.sort(key=lambda item: (-item[0], item[1]))
        hindsight_ranking = [
            {
                "rank": rank,
                "strategyId": identifier,
                "bets": int(row["bets"]),
                "stakeUnits": float(row["stakeUnits"]),
                "profitUnits": float(row["profitUnits"]),
                "roiPct": float(row["roiPct"]),
            }
            for rank, (_, identifier, row) in enumerate(hindsight_candidates, start=1)
        ]

        eligible: list[tuple[float, str, int, int, float]] = []
        for strategy in strategies:
            identifier = str(strategy["id"])
            prior_rows = [
                row
                for prior_season, row in yearly_by_strategy[identifier].items()
                if prior_season < season and int(row["bets"]) > 0
            ]
            prior_bets = sum(int(row["bets"]) for row in prior_rows)
            prior_priced_seasons = len(prior_rows)
            if (
                prior_bets < WALK_FORWARD_MINIMUM_PRIOR_BETS
                or prior_priced_seasons < WALK_FORWARD_MINIMUM_PRIOR_PRICED_SEASONS
            ):
                continue
            prior_profit = math.fsum(float(row["profitUnits"]) for row in prior_rows)
            prior_roi = prior_profit / prior_bets * 100.0
            eligible.append(
                (prior_roi, identifier, prior_bets, prior_priced_seasons, prior_profit)
            )
        eligible.sort(key=lambda item: (-item[0], item[1]))

        selected_strategy_id: str | None = None
        selected_prior_bets = 0
        selected_prior_priced_seasons = 0
        selected_prior_profit: float | None = None
        selected_prior_roi: float | None = None
        activated = False
        if not eligible:
            activation_reason = "no_eligible_strategy"
        elif eligible[0][0] <= WALK_FORWARD_ACTIVATION_THRESHOLD_ROI_PCT:
            activation_reason = "best_prior_roi_not_positive"
        else:
            prior_roi, selected_strategy_id, selected_prior_bets, selected_prior_priced_seasons, prior_profit = eligible[0]
            selected_prior_profit = round(prior_profit, 2)
            selected_prior_roi = round(prior_roi, 2)
            activated = True
            activation_reason = "positive_prior_roi"

        selected_row = (
            yearly_by_strategy[selected_strategy_id][season]
            if selected_strategy_id is not None
            else None
        )
        selected_bets = int(selected_row["bets"]) if selected_row is not None else 0
        selected_stake = float(selected_row["stakeUnits"]) if selected_row is not None else 0.0
        selected_profit = (
            float(selected_row["profitUnits"])
            if selected_row is not None and selected_row["profitUnits"] is not None
            else 0.0
        )
        selected_roi = (
            float(selected_row["roiPct"])
            if selected_row is not None and selected_row["roiPct"] is not None
            else None
        )
        audits.append(
            {
                "season": season,
                "label": f"{season}/{str(season + 1)[-2:]}",
                "hindsightRanking": hindsight_ranking,
                "walkForward": {
                    "basedThroughSeason": season - 1 if season > first_season else None,
                    "eligibleStrategyCount": len(eligible),
                    "selectedStrategyId": selected_strategy_id,
                    "selectedPriorBets": selected_prior_bets,
                    "selectedPriorPricedSeasons": selected_prior_priced_seasons,
                    "selectedPriorProfitUnits": selected_prior_profit,
                    "selectedPriorRoiPct": selected_prior_roi,
                    "activated": activated,
                    "activationReason": activation_reason,
                    "bets": selected_bets,
                    "stakeUnits": selected_stake,
                    "profitUnits": selected_profit,
                    "roiPct": selected_roi,
                },
            }
        )
    return audits


def build_strategy_zoo(
    matches: Iterable[Mapping[str, Any]],
    dataset_manifest: Mapping[str, Any] | None = None,
    *,
    generated_at: str | None = None,
    bootstrap_resamples: int = ROI_BOOTSTRAP_RESAMPLES,
    complete_through_season: int = LATEST_COMPLETE_SEASON,
    display_through_season: int | None = None,
) -> Dict[str, Any]:
    """Build the compact public strategy-zoo artifact.

    Strategy rules are fixed in :func:`strategy_definitions`; no strategy is
    selected, tuned or relabelled using the year it is evaluated on.
    """

    if isinstance(bootstrap_resamples, bool) or not isinstance(bootstrap_resamples, int) or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    source_prepared = _prepare(matches)
    if not source_prepared:
        raise ValueError("strategy zoo requires at least one valid source match")
    source_seasons = {match.season for match in source_prepared}
    source_end_season = max(source_seasons)
    if complete_through_season not in source_seasons:
        raise ValueError("complete_through_season has no source matches")
    if display_through_season is None:
        display_through_season = source_end_season
    if display_through_season < complete_through_season:
        raise ValueError("display_through_season cannot precede complete_through_season")
    if display_through_season != source_end_season:
        raise ValueError("display_through_season must equal the latest loaded source season")
    prepared = [match for match in source_prepared if match.season <= complete_through_season]
    score_distribution = Counter(f"{match.home_score}-{match.away_score}" for match in prepared)
    definitions = strategy_definitions()
    events: Dict[str, list[_Event]] = {definition.id: [] for definition in definitions}

    directed_results: MutableMapping[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    directed_scores: MutableMapping[tuple[str, str, str], Counter[tuple[int, int]]] = defaultdict(Counter)
    pair_winners: MutableMapping[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    pair_totals: MutableMapping[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    league_over_history: MutableMapping[str, Deque[int]] = defaultdict(lambda: deque(maxlen=500))
    league_bucket_history: MutableMapping[str, Deque[str]] = defaultdict(lambda: deque(maxlen=500))
    league_score_history: MutableMapping[str, Deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=1_000))
    rivalry_events: MutableMapping[tuple[str, str, str], list[_Event]] = defaultdict(list)
    pair_seasons: MutableMapping[tuple[str, str, str], list[int]] = defaultdict(list)
    pair_display_names: MutableMapping[tuple[str, str, str], Dict[str, str]] = defaultdict(dict)
    coverage: MutableMapping[int, Dict[str, int]] = defaultdict(
        lambda: {
            "matches": 0,
            "evaluatedMatches": 0,
            "quarantinedMatches": 0,
            "b3651x2Matches": 0,
            "b365Ou25Matches": 0,
        }
    )

    for match in source_prepared:
        if match.season <= complete_through_season:
            continue
        coverage[match.season]["matches"] += 1
        coverage[match.season]["quarantinedMatches"] += 1
        coverage[match.season]["b3651x2Matches"] += int(_verified_prices(match.raw, "1x2") is not None)
        coverage[match.season]["b365Ou25Matches"] += int(_verified_prices(match.raw, "ou25") is not None)

    position = 0
    while position < len(prepared):
        kickoff = prepared[position].kickoff
        end = position + 1
        while end < len(prepared) and prepared[end].kickoff == kickoff:
            end += 1
        group = prepared[position:end]

        for match in group:
            actual_1x2 = _outcome(match)
            actual_total15 = "over15" if match.home_score + match.away_score > 1 else "under15"
            actual_total = "over25" if match.home_score + match.away_score > 2 else "under25"
            actual_bucket = _goal_bucket(match.home_score + match.away_score)
            actual_score = f"{match.home_score}-{match.away_score}"
            directed_key = _directed_key(match)
            pair_key = _pair_key(match)
            pair_display_names[pair_key][match.home_key] = match.home
            pair_display_names[pair_key][match.away_key] = match.away
            coverage[match.season]["matches"] += 1
            coverage[match.season]["evaluatedMatches"] += 1

            prices_1x2 = _verified_prices(match.raw, "1x2")
            prices_ou = _verified_prices(match.raw, "ou25")
            if prices_1x2 is not None:
                coverage[match.season]["b3651x2Matches"] += 1
            if prices_ou is not None:
                coverage[match.season]["b365Ou25Matches"] += 1

            # Outcome-only 1.5 baselines deliberately remain unpriced: the
            # canonical Football-Data archive has scores but no coherent
            # historical Bet365 O/U 1.5 quote pair.
            events["all_over15"].append(_event(match, "over15", actual_total15))
            events["all_under15"].append(_event(match, "under15", actual_total15))

            directed_label, directed_count, directed_rate = _unique_mode(directed_results[directed_key])
            if directed_label in _OUTCOME_LABELS and directed_count >= 4 and directed_rate >= 0.70:
                price_label = {"H": "home", "D": "draw", "A": "away"}[directed_label]
                price = prices_1x2.get(price_label) if prices_1x2 else None
                events["directed_h2h_dominance"].append(_event(match, directed_label, actual_1x2, price))

            dominant_winner, pair_count, pair_rate = _unique_mode(pair_winners[pair_key])
            pair_selection: str | None = None
            pair_price_label: str | None = None
            if dominant_winner == match.home_key:
                pair_selection, pair_price_label = "H", "home"
            elif dominant_winner == match.away_key:
                pair_selection, pair_price_label = "A", "away"
            if pair_selection is not None and pair_count >= 5 and pair_rate >= 0.65:
                price = prices_1x2.get(pair_price_label) if prices_1x2 and pair_price_label else None
                signal = _event(match, pair_selection, actual_1x2, price)
                events["pair_h2h_team_dominance"].append(signal)
                display_pick = (
                    "Uafgjort"
                    if pair_selection == "D"
                    else match.home
                    if pair_selection == "H"
                    else match.away
                )
                rivalry_events[pair_key].append(
                    _Event(
                        kickoff=signal.kickoff,
                        season=signal.season,
                        correct=signal.correct,
                        selection=display_pick,
                        actual=signal.actual,
                        raw_odds=signal.raw_odds,
                        decimal_odds=signal.decimal_odds,
                    )
                )

            total_history = pair_totals[pair_key]
            total_count = sum(total_history.values())
            if total_count >= 5:
                over_rate = total_history["over25"] / total_count
                if over_rate >= 0.70:
                    over_price = prices_ou.get("over25") if prices_ou else None
                    under_price = prices_ou.get("under25") if prices_ou else None
                    events["h2h_over25_dominance"].append(
                        _event(match, "over25", actual_total, over_price)
                    )
                    events["fade_h2h_over25_dominance"].append(
                        _event(match, "under25", actual_total, under_price)
                    )
                if over_rate <= 0.30:
                    under_price = prices_ou.get("under25") if prices_ou else None
                    over_price = prices_ou.get("over25") if prices_ou else None
                    events["h2h_under25_dominance"].append(
                        _event(match, "under25", actual_total, under_price)
                    )
                    events["fade_h2h_under25_dominance"].append(
                        _event(match, "over25", actual_total, over_price)
                    )

            league_over = league_over_history[match.league]
            if len(league_over) >= 200:
                league_rate = sum(league_over) / len(league_over)
                if league_rate >= 0.60:
                    over_price = prices_ou.get("over25") if prices_ou else None
                    under_price = prices_ou.get("under25") if prices_ou else None
                    events["league_over25_extreme"].append(
                        _event(match, "over25", actual_total, over_price)
                    )
                    events["fade_league_over25_extreme"].append(
                        _event(match, "under25", actual_total, under_price)
                    )
                if league_rate <= 0.40:
                    under_price = prices_ou.get("under25") if prices_ou else None
                    over_price = prices_ou.get("over25") if prices_ou else None
                    events["league_under25_extreme"].append(
                        _event(match, "under25", actual_total, under_price)
                    )
                    events["fade_league_under25_extreme"].append(
                        _event(match, "over25", actual_total, over_price)
                    )

            bucket_history = league_bucket_history[match.league]
            if len(bucket_history) >= 200:
                bucket_mode, _, _ = _unique_mode(Counter(bucket_history))
                if bucket_mode in _GOAL_BUCKETS:
                    events["league_goal_bucket_mode"].append(_event(match, bucket_mode, actual_bucket))

            score_history = league_score_history[match.league]
            if len(score_history) >= 200:
                score_mode, _, _ = _unique_mode(Counter(score_history))
                if isinstance(score_mode, tuple):
                    events["league_exact_score_mode"].append(
                        _event(match, f"{score_mode[0]}-{score_mode[1]}", actual_score)
                    )

            h2h_score_mode, score_count, score_rate = _unique_mode(directed_scores[directed_key])
            if isinstance(h2h_score_mode, tuple) and score_count >= 5 and score_rate >= 0.40:
                events["directed_h2h_exact_score_mode"].append(
                    _event(match, f"{h2h_score_mode[0]}-{h2h_score_mode[1]}", actual_score)
                )

            if prices_1x2 is not None:
                price_outcome = {"home": "H", "draw": "D", "away": "A"}
                favourite = _team_favourite(prices_1x2)
                outsider = _unique_price_extreme(prices_1x2, highest=True)
                if favourite is not None:
                    price = prices_1x2[favourite]
                    events["all_unique_favourites"].append(
                        _event(match, price_outcome[favourite], actual_1x2, price)
                    )
                    for band in FAVOURITE_BANDS:
                        if _within_band(price, band):
                            identifier = _band_id("favourite", *band)
                            events[identifier].append(_event(match, price_outcome[favourite], actual_1x2, price))
                            break
                draw_price = prices_1x2["draw"]
                events["all_home_wins"].append(
                    _event(match, "H", actual_1x2, prices_1x2["home"])
                )
                events["all_draws"].append(_event(match, "D", actual_1x2, draw_price))
                events["all_away_wins"].append(
                    _event(match, "A", actual_1x2, prices_1x2["away"])
                )
                for band in DRAW_BANDS:
                    if _within_band(draw_price, band):
                        identifier = _band_id("draw", *band)
                        events[identifier].append(_event(match, "D", actual_1x2, draw_price))
                        break
                if outsider is not None:
                    price = prices_1x2[outsider]
                    events["all_unique_outsiders"].append(
                        _event(match, price_outcome[outsider], actual_1x2, price)
                    )
                    for band in OUTSIDER_BANDS:
                        if _within_band(price, band):
                            identifier = _band_id("outsider", *band)
                            events[identifier].append(_event(match, price_outcome[outsider], actual_1x2, price))
                            break

            if prices_ou is not None:
                events["all_over25"].append(
                    _event(match, "over25", actual_total, prices_ou["over25"])
                )
                events["all_under25"].append(
                    _event(match, "under25", actual_total, prices_ou["under25"])
                )
                for side in ("over25", "under25"):
                    price = prices_ou[side]
                    for band in TOTALS_BANDS:
                        if _within_band(price, band):
                            identifier = _band_id(side, *band)
                            events[identifier].append(_event(match, side, actual_total, price))
                            break

        # Commit outcomes only after every fixture at this kickoff was scored.
        for match in group:
            actual_1x2 = _outcome(match)
            pair_key = _pair_key(match)
            directed_key = _directed_key(match)
            winner = (
                match.home_key
                if actual_1x2 == "H"
                else match.away_key
                if actual_1x2 == "A"
                else "__draw__"
            )
            total = match.home_score + match.away_score
            directed_results[directed_key][actual_1x2] += 1
            directed_scores[directed_key][(match.home_score, match.away_score)] += 1
            pair_winners[pair_key][winner] += 1
            pair_seasons[pair_key].append(match.season)
            pair_totals[pair_key]["over25" if total > 2 else "under25"] += 1
            league_over_history[match.league].append(int(total > 2))
            league_bucket_history[match.league].append(_goal_bucket(total))
            league_score_history[match.league].append((match.home_score, match.away_score))
        position = end

    minimum_season = min((match.season for match in source_prepared), default=complete_through_season)
    seasons = list(range(minimum_season, display_through_season + 1))
    strategies = [
        _strategy_payload(
            definition,
            events[definition.id],
            seasons,
            bootstrap_resamples=bootstrap_resamples,
            complete_through_season=complete_through_season,
        )
        for definition in definitions
    ]
    rivalries = _rivalry_payload(
        rivalry_events,
        pair_winners,
        pair_seasons,
        pair_display_names,
        bootstrap_resamples=bootstrap_resamples,
    )
    h2h_validation = _h2h_validation_audit(
        prepared,
        validation_end_season=complete_through_season,
    )
    manifest = dataset_manifest or {}
    total_matches = len(source_prepared)
    coverage_by_season = []
    for season in seasons:
        values = coverage[season]
        coverage_by_season.append(
            {
                "season": season,
                **values,
                "b3651x2CoveragePct": round(values["b3651x2Matches"] / values["matches"] * 100.0, 2)
                if values["matches"]
                else 0.0,
                "b365Ou25CoveragePct": round(values["b365Ou25Matches"] / values["matches"] * 100.0, 2)
                if values["matches"]
                else 0.0,
            }
        )
    output = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedAt": generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "researchVerdict": "NO_CONFIRMED_BETTING_EDGE",
        "dataset": {
            "datasetId": manifest.get("dataset_id"),
            "source": manifest.get("source", "data/cache/football_data_csv"),
            "matches": total_matches,
            "evaluatedMatches": len(prepared),
            "quarantinedMatches": len(source_prepared) - len(prepared),
            "startDate": source_prepared[0].kickoff.date().isoformat() if source_prepared else None,
            "endDate": source_prepared[-1].kickoff.date().isoformat() if source_prepared else None,
            "leagues": sorted({match.league for match in source_prepared}),
            "completeThroughSeason": complete_through_season,
            "quarantinedSeasons": [
                season for season in seasons if season > complete_through_season
            ],
        },
        "methodology": {
            "pointInTime": True,
            "sameKickoffIsolation": True,
            "fixedRulesNoInYearTuning": True,
            "completeThroughSeason": complete_through_season,
            "quarantinePolicy": "incomplete seasons appear in the UI but contribute no metrics or P&L",
            "unitStake": 1.0,
            "profitQuote": "complete Bet365 opening market only",
            "executionHaircutPctOnProfitPortion": ODDS_HAIRCUT * 100.0,
            "missingOddsPolicy": "count_accuracy_opportunity_but_never_impute_profit",
            "hitRateCi": "95% Wilson score interval",
            "roiCi": f"95% season-cluster bootstrap, {bootstrap_resamples} deterministic resamples",
            "multipleTesting": "descriptive strategy screen; no automatic live activation",
            "guarantees": False,
        },
        "seasons": seasons,
        "coverage": {"bySeason": coverage_by_season},
        "seasonMarketProfiles": _build_season_market_profiles(prepared, seasons),
        "strategies": strategies,
        "selectionPolicy": _selection_policy(),
        "seasonAudits": _build_season_audits(strategies, seasons),
        "rivalryPatterns": rivalries,
        "findings": _build_findings(strategies, rivalries, score_distribution, h2h_validation),
    }
    return validate_strategy_zoo(output)


_ALLOWED_STRATEGY_STATUSES = {
    "unavailable",
    "descriptive_only",
    "insufficient",
    "historical_positive_unconfirmed",
    "historical_negative",
    "historical_negative_unconfirmed",
}


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_metric(metric: Any, context: str) -> None:
    if not isinstance(metric, Mapping):
        raise StrategyZooValidationError(f"{context} must be an object")
    counts: Dict[str, int] = {}
    for field in ("opportunities", "hits", "bets", "wins"):
        value = metric.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StrategyZooValidationError(f"{context}.{field} must be a non-negative integer")
        counts[field] = value
    if counts["hits"] > counts["opportunities"]:
        raise StrategyZooValidationError(f"{context} has more hits than opportunities")
    if counts["bets"] > counts["opportunities"] or counts["wins"] > counts["bets"]:
        raise StrategyZooValidationError(f"{context} has invalid priced-bet counts")

    stake = metric.get("stakeUnits")
    if not _finite_number(stake) or abs(float(stake) - counts["bets"]) > 1e-9:
        raise StrategyZooValidationError(f"{context}.stakeUnits must equal one unit per priced bet")
    pnl_available = metric.get("pnlAvailable")
    pnl_reason = metric.get("pnlAvailabilityReason")
    expected_pnl_available = counts["bets"] > 0
    if pnl_available is not expected_pnl_available:
        raise StrategyZooValidationError(f"{context} has an inconsistent P&L availability flag")
    if expected_pnl_available:
        if pnl_reason is not None:
            raise StrategyZooValidationError(f"{context} gives a reason for available P&L")
    else:
        allowed_reasons = (
            {"no_verified_pre_match_odds_for_market"}
            if counts["opportunities"]
            else {"no_qualifying_opportunities", "incomplete_local_snapshot"}
        )
        if pnl_reason not in allowed_reasons:
            raise StrategyZooValidationError(f"{context} hides why P&L is unavailable")

    hit_rate = metric.get("hitRatePct")
    expected_hit_rate = (
        counts["hits"] / counts["opportunities"] * 100.0
        if counts["opportunities"]
        else None
    )
    if expected_hit_rate is None:
        if hit_rate is not None:
            raise StrategyZooValidationError(f"{context} fabricates a hit rate without opportunities")
    elif not _finite_number(hit_rate) or not 0.0 <= float(hit_rate) <= 100.0 or abs(float(hit_rate) - expected_hit_rate) > 0.011:
        raise StrategyZooValidationError(f"{context} has an inconsistent hit rate")

    hit_ci = metric.get("hitRateCi95Pct")
    if expected_hit_rate is None:
        if hit_ci is not None:
            raise StrategyZooValidationError(f"{context} fabricates a hit-rate interval")
    elif (
        not isinstance(hit_ci, list)
        or len(hit_ci) != 2
        or any(not _finite_number(value) for value in hit_ci)
        or not 0.0 <= float(hit_ci[0]) <= expected_hit_rate <= float(hit_ci[1]) <= 100.0
    ):
        raise StrategyZooValidationError(f"{context} has an invalid hit-rate interval")

    odds_coverage = metric.get("oddsCoveragePct")
    expected_coverage = (
        counts["bets"] / counts["opportunities"] * 100.0
        if counts["opportunities"]
        else 0.0
    )
    if not _finite_number(odds_coverage) or abs(float(odds_coverage) - expected_coverage) > 0.011:
        raise StrategyZooValidationError(f"{context} has inconsistent odds coverage")

    pnl_fields = ("averageOpeningOdds", "profitUnits", "roiPct", "maxDrawdownUnits")
    if not counts["bets"]:
        if any(metric.get(field) is not None for field in pnl_fields + ("roiCi95Pct",)):
            raise StrategyZooValidationError(f"{context} fabricates P&L without verified odds")
    else:
        if any(not _finite_number(metric.get(field)) for field in pnl_fields):
            raise StrategyZooValidationError(f"{context} has incomplete or non-finite P&L")
        average_odds = float(metric["averageOpeningOdds"])
        profit = float(metric["profitUnits"])
        roi = float(metric["roiPct"])
        drawdown = float(metric["maxDrawdownUnits"])
        if not 1.0 < average_odds <= 1_000.0:
            raise StrategyZooValidationError(f"{context} has invalid opening odds")
        losses = counts["bets"] - counts["wins"]
        rounding_tolerance = max(0.02, counts["bets"] * 0.001)
        maximum_profit = (1.0 - ODDS_HAIRCUT) * counts["bets"] * (average_odds - 1.0)
        if profit < -losses - rounding_tolerance or profit > maximum_profit + rounding_tolerance:
            raise StrategyZooValidationError(f"{context} has impossible profit for its odds and results")
        if abs(roi - profit / counts["bets"] * 100.0) > 0.51:
            raise StrategyZooValidationError(f"{context} has ROI inconsistent with profit")
        if not 0.0 <= drawdown <= counts["bets"] + 0.01:
            raise StrategyZooValidationError(f"{context} has invalid drawdown")
        roi_ci = metric.get("roiCi95Pct")
        if roi_ci is not None and (
            not isinstance(roi_ci, list)
            or len(roi_ci) != 2
            or any(not _finite_number(value) for value in roi_ci)
            or not float(roi_ci[0]) <= roi <= float(roi_ci[1])
        ):
            raise StrategyZooValidationError(f"{context} has an invalid ROI interval")

    for field in ("activeSeasons", "pricedSeasons", "positivePricedSeasons"):
        value = metric.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StrategyZooValidationError(f"{context}.{field} must be a non-negative integer")
    if not 0 <= metric["positivePricedSeasons"] <= metric["pricedSeasons"] <= metric["activeSeasons"]:
        raise StrategyZooValidationError(f"{context} has inconsistent season counts")
    positive_rate = metric.get("positivePricedSeasonRatePct")
    expected_positive_rate = (
        metric["positivePricedSeasons"] / metric["pricedSeasons"] * 100.0
        if metric["pricedSeasons"]
        else None
    )
    if expected_positive_rate is None:
        if positive_rate is not None:
            raise StrategyZooValidationError(f"{context} fabricates a positive-season rate")
    elif not _finite_number(positive_rate) or abs(float(positive_rate) - expected_positive_rate) > 0.011:
        raise StrategyZooValidationError(f"{context} has an inconsistent positive-season rate")


def _validate_rivalry_pattern(row: Any, index: int, seasons: Sequence[int]) -> None:
    context = f"rivalryPatterns[{index}]"
    expected_fields = {
        "id",
        "competitionGroup",
        "teams",
        "relationshipLabel",
        "team",
        "opponent",
        "dominantTeam",
        "meetings",
        "record",
        "winRatePct",
        "winRateCi95Pct",
        "unbeatenRatePct",
        "perfectWinRecord",
        "perfectUnbeatenRecord",
        "mostFrequentPointInTimePick",
        "status",
        "guaranteed",
        "pointInTimeSignal",
        "firstSeason",
        "lastSeason",
    }
    if not isinstance(row, Mapping) or set(row) != expected_fields:
        raise StrategyZooValidationError(f"{context} has an unsupported schema")
    identifier = row.get("id")
    if not isinstance(identifier, str) or len(identifier) != 16 or any(
        character not in "0123456789abcdef" for character in identifier
    ):
        raise StrategyZooValidationError(f"{context} has an invalid id")
    teams = row.get("teams")
    if (
        not isinstance(teams, list)
        or len(teams) != 2
        or any(not isinstance(team, str) or not team.strip() for team in teams)
        or teams[0] == teams[1]
    ):
        raise StrategyZooValidationError(f"{context} has invalid teams")
    team = row.get("team")
    opponent = row.get("opponent")
    if team not in teams or opponent not in teams or team == opponent or row.get("dominantTeam") != team:
        raise StrategyZooValidationError(f"{context} has an invalid dominant team")
    if row.get("relationshipLabel") != "head_to_head_pair_not_verified_derby":
        raise StrategyZooValidationError(f"{context} mislabels a head-to-head pair")
    if not isinstance(row.get("competitionGroup"), str) or not row["competitionGroup"]:
        raise StrategyZooValidationError(f"{context} has an invalid competition group")
    meetings = row.get("meetings")
    record = row.get("record")
    if (
        isinstance(meetings, bool)
        or not isinstance(meetings, int)
        or meetings < 10
        or not isinstance(record, Mapping)
        or set(record) != {"wins", "draws", "losses"}
        or any(isinstance(record.get(field), bool) or not isinstance(record.get(field), int) or record[field] < 0 for field in record)
        or sum(record.values()) != meetings
        or record["wins"] < record["losses"]
    ):
        raise StrategyZooValidationError(f"{context} has an invalid historical record")
    expected_win_rate = record["wins"] / meetings * 100.0
    expected_unbeaten_rate = (record["wins"] + record["draws"]) / meetings * 100.0
    if not _finite_number(row.get("winRatePct")) or abs(float(row["winRatePct"]) - expected_win_rate) > 0.011:
        raise StrategyZooValidationError(f"{context} has an inconsistent win rate")
    if not _finite_number(row.get("unbeatenRatePct")) or abs(float(row["unbeatenRatePct"]) - expected_unbeaten_rate) > 0.011:
        raise StrategyZooValidationError(f"{context} has an inconsistent unbeaten rate")
    win_ci = row.get("winRateCi95Pct")
    if (
        not isinstance(win_ci, list)
        or len(win_ci) != 2
        or any(not _finite_number(value) for value in win_ci)
        or not 0.0 <= float(win_ci[0]) <= expected_win_rate <= float(win_ci[1]) <= 100.0
    ):
        raise StrategyZooValidationError(f"{context} has an invalid win-rate interval")
    if row.get("perfectWinRecord") is not (record["wins"] == meetings):
        raise StrategyZooValidationError(f"{context} has an inconsistent perfect-win flag")
    if row.get("perfectUnbeatenRecord") is not (record["losses"] == 0):
        raise StrategyZooValidationError(f"{context} has an inconsistent unbeaten flag")
    if row.get("status") != "descriptive_only" or row.get("guaranteed") is not False:
        raise StrategyZooValidationError(f"{context} must remain descriptive and non-guaranteed")
    _validate_metric(row.get("pointInTimeSignal"), f"{context}.pointInTimeSignal")
    point_in_time_pick = row.get("mostFrequentPointInTimePick")
    if row["pointInTimeSignal"]["opportunities"]:
        if point_in_time_pick not in teams:
            raise StrategyZooValidationError(f"{context} has an invalid point-in-time pick")
    elif point_in_time_pick is not None:
        raise StrategyZooValidationError(f"{context} fabricates a point-in-time pick")
    if row["pointInTimeSignal"]["opportunities"] > meetings:
        raise StrategyZooValidationError(f"{context} has too many point-in-time signals")
    first_season = row.get("firstSeason")
    last_season = row.get("lastSeason")
    if (
        isinstance(first_season, bool)
        or not isinstance(first_season, int)
        or isinstance(last_season, bool)
        or not isinstance(last_season, int)
        or first_season not in seasons
        or last_season not in seasons
        or first_season > last_season
    ):
        raise StrategyZooValidationError(f"{context} has an invalid season range")


def _validate_h2h_validation(value: Any, complete_through_season: int) -> None:
    context = "findings.h2hValidation"
    expected_fields = {
        "multipleTestingCorrectionApplied",
        "protocol",
        "candidateCount",
        "candidateTests",
        "confirmedWinRatePatterns",
        "confirmedEdges",
        "status",
        "bettingEdgeTested",
        "confirmedBettingEdges",
        "topValidationCandidates",
        "reason",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise StrategyZooValidationError(f"{context} has an unsupported schema")
    if value.get("multipleTestingCorrectionApplied") is not True:
        raise StrategyZooValidationError(f"{context} must apply multiple-testing correction")
    expected_protocol = {
        "discoverySeasons": "<=2019/20",
        "validationSeasons": f"2020/21-{complete_through_season}/{str(complete_through_season + 1)[-2:]}",
        "minimumDiscoveryMeetings": 12,
        "minimumDiscoveryWinRatePct": 65.0,
        "minimumValidationMeetings": 6,
        "nullHypothesis": "dominant team win probability <= 0.50",
        "test": "exact one-sided binomial",
        "correction": "Benjamini-Hochberg FDR q<=0.05",
    }
    if value.get("protocol") != expected_protocol:
        raise StrategyZooValidationError(f"{context} protocol is inconsistent")
    candidate_count = value.get("candidateCount")
    confirmed_count = value.get("confirmedWinRatePatterns")
    if (
        isinstance(candidate_count, bool)
        or not isinstance(candidate_count, int)
        or candidate_count < 0
        or isinstance(confirmed_count, bool)
        or not isinstance(confirmed_count, int)
        or not 0 <= confirmed_count <= candidate_count
        or value.get("confirmedEdges") != 0
        or value.get("confirmedBettingEdges") != 0
        or value.get("bettingEdgeTested") is not False
    ):
        raise StrategyZooValidationError(f"{context} cannot claim a confirmed betting edge")
    candidate_tests = value.get("candidateTests")
    if not isinstance(candidate_tests, list) or len(candidate_tests) != candidate_count:
        raise StrategyZooValidationError(f"{context}.candidateTests must include every candidate")
    test_pair_ids: set[tuple[str, str, str]] = set()
    raw_p_values: list[float] = []
    for index, candidate_test in enumerate(candidate_tests):
        test_context = f"{context}.candidateTests[{index}]"
        if not isinstance(candidate_test, Mapping) or set(candidate_test) != {
            "competitionGroup",
            "team",
            "opponent",
            "validationMeetings",
            "validationWins",
            "pValue",
            "qValue",
            "confirmedAtQ05",
        }:
            raise StrategyZooValidationError(f"{test_context} has an unsupported schema")
        team = candidate_test.get("team")
        opponent = candidate_test.get("opponent")
        competition = candidate_test.get("competitionGroup")
        meetings = candidate_test.get("validationMeetings")
        wins = candidate_test.get("validationWins")
        pair_id = (competition, team, opponent)
        if (
            any(not isinstance(item, str) or not item for item in pair_id)
            or team == opponent
            or pair_id in test_pair_ids
            or isinstance(meetings, bool)
            or not isinstance(meetings, int)
            or meetings < 6
            or isinstance(wins, bool)
            or not isinstance(wins, int)
            or not 0 <= wins <= meetings
        ):
            raise StrategyZooValidationError(f"{test_context} has invalid identifiers or counts")
        test_pair_ids.add(pair_id)
        raw_p_value = _one_sided_binomial_p_value(wins, meetings)
        raw_p_values.append(raw_p_value)
        if candidate_test.get("pValue") != round(raw_p_value, 8):
            raise StrategyZooValidationError(f"{test_context} p-value disagrees with its record")

    expected_q_values = [1.0] * candidate_count
    ordered_tests = sorted(range(candidate_count), key=lambda index: (raw_p_values[index], index))
    running_q = 1.0
    for reverse_position in range(candidate_count - 1, -1, -1):
        original_index = ordered_tests[reverse_position]
        rank = reverse_position + 1
        running_q = min(running_q, raw_p_values[original_index] * candidate_count / rank)
        expected_q_values[original_index] = min(1.0, running_q)
    for index, candidate_test in enumerate(candidate_tests):
        test_context = f"{context}.candidateTests[{index}]"
        if candidate_test.get("qValue") != round(expected_q_values[index], 8):
            raise StrategyZooValidationError(f"{test_context} q-value disagrees with Benjamini-Hochberg")
        expected_confirmed = expected_q_values[index] <= 0.05
        if candidate_test.get("confirmedAtQ05") is not expected_confirmed:
            raise StrategyZooValidationError(f"{test_context} has an inconsistent significance flag")
    expected_test_order = sorted(
        candidate_tests,
        key=lambda item: (item["qValue"], item["pValue"], -item["validationMeetings"]),
    )
    if candidate_tests != expected_test_order:
        raise StrategyZooValidationError(f"{context}.candidateTests are not deterministically ranked")
    confirmed_in_tests = sum(candidate["confirmedAtQ05"] is True for candidate in candidate_tests)
    if confirmed_count != confirmed_in_tests:
        raise StrategyZooValidationError(f"{context} confirmed count disagrees with candidate tests")
    candidates = value.get("topValidationCandidates")
    if not isinstance(candidates, list) or len(candidates) > 10 or len(candidates) > candidate_count:
        raise StrategyZooValidationError(f"{context} has invalid candidate rows")
    pair_ids: set[tuple[str, str, str]] = set()
    for index, candidate in enumerate(candidates):
        candidate_context = f"{context}.topValidationCandidates[{index}]"
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "team",
            "opponent",
            "competitionGroup",
            "discovery",
            "validation",
            "pValue",
            "qValue",
            "confirmedAtQ05",
        }:
            raise StrategyZooValidationError(f"{candidate_context} has an unsupported schema")
        team = candidate.get("team")
        opponent = candidate.get("opponent")
        competition = candidate.get("competitionGroup")
        if any(not isinstance(item, str) or not item for item in (team, opponent, competition)) or team == opponent:
            raise StrategyZooValidationError(f"{candidate_context} has invalid teams")
        pair_id = (competition, team, opponent)
        if pair_id in pair_ids:
            raise StrategyZooValidationError(f"{candidate_context} is duplicated")
        pair_ids.add(pair_id)
        discovery = candidate.get("discovery")
        validation = candidate.get("validation")
        if not isinstance(discovery, Mapping) or set(discovery) != {"meetings", "wins", "winRatePct"}:
            raise StrategyZooValidationError(f"{candidate_context}.discovery is invalid")
        if not isinstance(validation, Mapping) or set(validation) != {"meetings", "wins", "draws", "losses", "winRatePct"}:
            raise StrategyZooValidationError(f"{candidate_context}.validation is invalid")
        for phase, record in (("discovery", discovery), ("validation", validation)):
            count_fields = ("meetings", "wins") if phase == "discovery" else ("meetings", "wins", "draws", "losses")
            if any(isinstance(record.get(field), bool) or not isinstance(record.get(field), int) or record[field] < 0 for field in count_fields):
                raise StrategyZooValidationError(f"{candidate_context}.{phase} has invalid counts")
            if record["wins"] > record["meetings"] or (
                phase == "validation" and record["wins"] + record["draws"] + record["losses"] != record["meetings"]
            ):
                raise StrategyZooValidationError(f"{candidate_context}.{phase} counts disagree")
            expected_rate = record["wins"] / record["meetings"] * 100.0 if record["meetings"] else 0.0
            if (
                phase == "discovery"
                and (record["meetings"] < 12 or expected_rate < 65.0)
            ) or (phase == "validation" and record["meetings"] < 6):
                raise StrategyZooValidationError(f"{candidate_context}.{phase} misses the audit threshold")
            if not _finite_number(record.get("winRatePct")) or abs(float(record["winRatePct"]) - expected_rate) > 0.011:
                raise StrategyZooValidationError(f"{candidate_context}.{phase} rate disagrees")
        p_value = candidate.get("pValue")
        q_value = candidate.get("qValue")
        if (
            not _finite_number(p_value)
            or not _finite_number(q_value)
            or not 0.0 <= float(p_value) <= float(q_value) <= 1.0
        ):
            raise StrategyZooValidationError(f"{candidate_context} has invalid p/q values")
        expected_confirmed = float(q_value) <= 0.05
        if candidate.get("confirmedAtQ05") is not expected_confirmed:
            raise StrategyZooValidationError(f"{candidate_context} has an inconsistent significance flag")
        expected_test = candidate_tests[index]
        if (
            expected_test["competitionGroup"] != competition
            or expected_test["team"] != team
            or expected_test["opponent"] != opponent
            or expected_test["validationMeetings"] != validation["meetings"]
            or expected_test["validationWins"] != validation["wins"]
            or expected_test["pValue"] != p_value
            or expected_test["qValue"] != q_value
            or expected_test["confirmedAtQ05"] is not candidate["confirmedAtQ05"]
        ):
            raise StrategyZooValidationError(f"{candidate_context} disagrees with candidateTests")
    if len(candidates) != min(10, candidate_count):
        raise StrategyZooValidationError(f"{context} must expose the ten highest-ranked candidates")
    expected_status = (
        "CONFIRMED_WIN_RATE_PATTERNS_NO_BETTING_EDGE"
        if confirmed_count
        else "NO_CONFIRMED_WIN_RATE_PATTERN"
    )
    if value.get("status") != expected_status or not isinstance(value.get("reason"), str) or not value["reason"]:
        raise StrategyZooValidationError(f"{context} has an inconsistent status")


def _validate_count_rate(value: Any, denominator: int, context: str) -> int:
    if not isinstance(value, Mapping) or set(value) != {"count", "ratePct"}:
        raise StrategyZooValidationError(f"{context} has an unsupported schema")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= denominator:
        raise StrategyZooValidationError(f"{context}.count is invalid")
    expected_rate = count / denominator * 100.0 if denominator else None
    rate = value.get("ratePct")
    if expected_rate is None:
        if rate is not None:
            raise StrategyZooValidationError(f"{context} fabricates a rate")
    elif not _finite_number(rate) or abs(float(rate) - expected_rate) > 0.011:
        raise StrategyZooValidationError(f"{context}.ratePct disagrees with its denominator")
    return count


def _validate_market_profile_row(value: Any, context: str) -> None:
    expected_fields = {
        "scope",
        "season",
        "label",
        "scoredMatches",
        "oneXTwo",
        "totalGoals",
        "exactTotalGoals",
        "teamFavourites",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise StrategyZooValidationError(f"{context} has an unsupported schema")
    scored = value.get("scoredMatches")
    if isinstance(scored, bool) or not isinstance(scored, int) or scored < 0:
        raise StrategyZooValidationError(f"{context}.scoredMatches is invalid")
    one_x_two = value.get("oneXTwo")
    if not isinstance(one_x_two, Mapping) or set(one_x_two) != {"home", "draw", "away"}:
        raise StrategyZooValidationError(f"{context}.oneXTwo has an unsupported schema")
    outcome_counts = [
        _validate_count_rate(one_x_two[side], scored, f"{context}.oneXTwo.{side}")
        for side in ("home", "draw", "away")
    ]
    if sum(outcome_counts) != scored:
        raise StrategyZooValidationError(f"{context}.oneXTwo does not partition scored matches")

    total_goals = value.get("totalGoals")
    if not isinstance(total_goals, Mapping) or tuple(total_goals) != _TOTAL_GOAL_THRESHOLDS:
        raise StrategyZooValidationError(f"{context}.totalGoals has an unsupported schema")
    previous_over = scored + 1
    for threshold in _TOTAL_GOAL_THRESHOLDS:
        threshold_row = total_goals[threshold]
        if not isinstance(threshold_row, Mapping) or set(threshold_row) != {"over", "under"}:
            raise StrategyZooValidationError(f"{context}.totalGoals[{threshold}] is invalid")
        over = _validate_count_rate(
            threshold_row["over"], scored, f"{context}.totalGoals[{threshold}].over"
        )
        under = _validate_count_rate(
            threshold_row["under"], scored, f"{context}.totalGoals[{threshold}].under"
        )
        if over + under != scored:
            raise StrategyZooValidationError(f"{context}.totalGoals[{threshold}] is not complementary")
        if over > previous_over:
            raise StrategyZooValidationError(f"{context}.totalGoals over counts are not monotonic")
        previous_over = over

    exact = value.get("exactTotalGoals")
    exact_buckets = _EXACT_TOTAL_GOAL_BUCKETS
    if not isinstance(exact, Mapping) or tuple(exact) != exact_buckets:
        raise StrategyZooValidationError(f"{context}.exactTotalGoals has an unsupported schema")
    if sum(
        _validate_count_rate(exact[bucket], scored, f"{context}.exactTotalGoals[{bucket}]")
        for bucket in exact_buckets
    ) != scored:
        raise StrategyZooValidationError(f"{context}.exactTotalGoals does not partition scored matches")
    cumulative_exact = 0
    for threshold, bucket in zip(_TOTAL_GOAL_THRESHOLDS, exact_buckets[:-1]):
        cumulative_exact += int(exact[bucket]["count"])
        if int(total_goals[threshold]["over"]["count"]) != scored - cumulative_exact:
            raise StrategyZooValidationError(
                f"{context}.exactTotalGoals disagrees with totalGoals[{threshold}]"
            )

    favourites = value.get("teamFavourites")
    favourite_fields = {
        "completePricedMatches",
        "uniqueTeamSelections",
        "tiesSkipped",
        "won",
        "drawn",
        "lost",
        "winRatePct",
        "drawRatePct",
        "lossRatePct",
    }
    if not isinstance(favourites, Mapping) or set(favourites) != favourite_fields:
        raise StrategyZooValidationError(f"{context}.teamFavourites has an unsupported schema")
    count_fields = (
        "completePricedMatches",
        "uniqueTeamSelections",
        "tiesSkipped",
        "won",
        "drawn",
        "lost",
    )
    if any(
        isinstance(favourites.get(field), bool)
        or not isinstance(favourites.get(field), int)
        or favourites[field] < 0
        for field in count_fields
    ):
        raise StrategyZooValidationError(f"{context}.teamFavourites has invalid counts")
    if (
        favourites["completePricedMatches"] > scored
        or favourites["uniqueTeamSelections"] + favourites["tiesSkipped"]
        != favourites["completePricedMatches"]
        or favourites["won"] + favourites["drawn"] + favourites["lost"]
        != favourites["uniqueTeamSelections"]
    ):
        raise StrategyZooValidationError(f"{context}.teamFavourites counts disagree")
    denominator = favourites["uniqueTeamSelections"]
    for count_field, rate_field in (
        ("won", "winRatePct"),
        ("drawn", "drawRatePct"),
        ("lost", "lossRatePct"),
    ):
        expected_rate = favourites[count_field] / denominator * 100.0 if denominator else None
        rate = favourites[rate_field]
        if expected_rate is None:
            if rate is not None:
                raise StrategyZooValidationError(f"{context}.teamFavourites fabricates a rate")
        elif not _finite_number(rate) or abs(float(rate) - expected_rate) > 0.011:
            raise StrategyZooValidationError(f"{context}.teamFavourites.{rate_field} disagrees")


def _validate_season_market_profiles(
    value: Any,
    seasons: Sequence[int],
    coverage_rows: Sequence[Mapping[str, Any]],
    evaluated_matches: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"methodology", "allTime", "bySeason", "stability"}:
        raise StrategyZooValidationError("seasonMarketProfiles has an unsupported schema")
    if value.get("methodology") != {
        "descriptiveOnly": True,
        "scoredMatchDenominator": "all_scored_canonical_matches",
        "favouriteDefinition": "unique_lower_home_or_away_bet365_open_price_in_complete_1x2_market",
        "favouriteTiePolicy": "skip",
        "guaranteesOrEdgeClaims": False,
    }:
        raise StrategyZooValidationError("seasonMarketProfiles methodology is inconsistent")
    rows = value.get("bySeason")
    if not isinstance(rows, list) or [row.get("season") for row in rows if isinstance(row, Mapping)] != list(seasons):
        raise StrategyZooValidationError("seasonMarketProfiles must contain every season")
    for row, coverage in zip(rows, coverage_rows):
        _validate_market_profile_row(row, f"seasonMarketProfiles.bySeason[{coverage['season']}]")
        season = coverage["season"]
        if (
            row["scope"] != "season"
            or row["label"] != f"{season}/{str(season + 1)[-2:]}"
            or row["scoredMatches"] != coverage["evaluatedMatches"]
        ):
            raise StrategyZooValidationError("seasonMarketProfiles season metadata disagrees")
        if (
            coverage["quarantinedMatches"] == 0
            and row["teamFavourites"]["completePricedMatches"] != coverage["b3651x2Matches"]
        ):
            raise StrategyZooValidationError(
                "seasonMarketProfiles favourite quote coverage disagrees"
            )

    all_time = value.get("allTime")
    _validate_market_profile_row(all_time, "seasonMarketProfiles.allTime")
    if (
        all_time["scope"] != "all_time"
        or all_time["season"] is not None
        or all_time["label"] != "Alle sæsoner"
        or all_time["scoredMatches"] != evaluated_matches
    ):
        raise StrategyZooValidationError("seasonMarketProfiles all-time metadata disagrees")
    for side in ("home", "draw", "away"):
        if all_time["oneXTwo"][side]["count"] != sum(row["oneXTwo"][side]["count"] for row in rows):
            raise StrategyZooValidationError("seasonMarketProfiles all-time 1X2 counts disagree")
    for threshold in _TOTAL_GOAL_THRESHOLDS:
        for side in ("over", "under"):
            if all_time["totalGoals"][threshold][side]["count"] != sum(
                row["totalGoals"][threshold][side]["count"] for row in rows
            ):
                raise StrategyZooValidationError("seasonMarketProfiles all-time goal counts disagree")
    for bucket in _EXACT_TOTAL_GOAL_BUCKETS:
        if all_time["exactTotalGoals"][bucket]["count"] != sum(
            row["exactTotalGoals"][bucket]["count"] for row in rows
        ):
            raise StrategyZooValidationError("seasonMarketProfiles all-time exact totals disagree")
    for field in ("completePricedMatches", "uniqueTeamSelections", "tiesSkipped", "won", "drawn", "lost"):
        if all_time["teamFavourites"][field] != sum(row["teamFavourites"][field] for row in rows):
            raise StrategyZooValidationError("seasonMarketProfiles all-time favourite counts disagree")
    if value.get("stability") != _market_profile_stability(rows):
        raise StrategyZooValidationError("seasonMarketProfiles stability disagrees with season rows")


def validate_strategy_zoo(payload: Any) -> Dict[str, Any]:
    """Validate and return an isolated JSON-safe strategy-zoo payload."""

    if not isinstance(payload, Mapping):
        raise StrategyZooValidationError("strategy zoo root must be an object")
    if set(payload) != {
        "schemaVersion",
        "generatedAt",
        "researchVerdict",
        "dataset",
        "methodology",
        "seasons",
        "coverage",
        "seasonMarketProfiles",
        "strategies",
        "selectionPolicy",
        "seasonAudits",
        "rivalryPatterns",
        "findings",
    }:
        raise StrategyZooValidationError("strategy zoo root has an unsupported schema")
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise StrategyZooValidationError("unsupported strategy zoo schema version")
    if payload.get("researchVerdict") != "NO_CONFIRMED_BETTING_EDGE":
        raise StrategyZooValidationError("strategy zoo cannot publish an unconfirmed betting edge")
    generated_at = payload.get("generatedAt")
    try:
        generated_timestamp = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise StrategyZooValidationError("generatedAt must be an ISO-8601 timestamp") from exc
    if not isinstance(generated_at, str) or generated_timestamp.tzinfo is None:
        raise StrategyZooValidationError("generatedAt must include a timezone")
    methodology = payload.get("methodology")
    expected_methodology_fields = {
        "pointInTime",
        "sameKickoffIsolation",
        "fixedRulesNoInYearTuning",
        "completeThroughSeason",
        "quarantinePolicy",
        "unitStake",
        "profitQuote",
        "executionHaircutPctOnProfitPortion",
        "missingOddsPolicy",
        "hitRateCi",
        "roiCi",
        "multipleTesting",
        "guarantees",
    }
    if not isinstance(methodology, Mapping) or set(methodology) != expected_methodology_fields:
        raise StrategyZooValidationError("methodology has an unsupported schema")
    complete_through_season = methodology.get("completeThroughSeason")
    if isinstance(complete_through_season, bool) or not isinstance(complete_through_season, int):
        raise StrategyZooValidationError("methodology.completeThroughSeason must be an integer")
    expected_methodology = {
        "pointInTime": True,
        "sameKickoffIsolation": True,
        "fixedRulesNoInYearTuning": True,
        "completeThroughSeason": complete_through_season,
        "quarantinePolicy": "incomplete seasons appear in the UI but contribute no metrics or P&L",
        "unitStake": 1.0,
        "profitQuote": "complete Bet365 opening market only",
        "executionHaircutPctOnProfitPortion": ODDS_HAIRCUT * 100.0,
        "missingOddsPolicy": "count_accuracy_opportunity_but_never_impute_profit",
        "hitRateCi": "95% Wilson score interval",
        "multipleTesting": "descriptive strategy screen; no automatic live activation",
        "guarantees": False,
    }
    if any(methodology.get(key) != value for key, value in expected_methodology.items()):
        raise StrategyZooValidationError("methodology safety policy is inconsistent")
    roi_ci_description = methodology.get("roiCi")
    roi_ci_prefix = "95% season-cluster bootstrap, "
    roi_ci_suffix = " deterministic resamples"
    if (
        not isinstance(roi_ci_description, str)
        or not roi_ci_description.startswith(roi_ci_prefix)
        or not roi_ci_description.endswith(roi_ci_suffix)
        or not roi_ci_description[len(roi_ci_prefix) : -len(roi_ci_suffix)].isdigit()
        or int(roi_ci_description[len(roi_ci_prefix) : -len(roi_ci_suffix)]) < 1
    ):
        raise StrategyZooValidationError("methodology ROI confidence interval is invalid")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "datasetId",
        "source",
        "matches",
        "evaluatedMatches",
        "quarantinedMatches",
        "startDate",
        "endDate",
        "leagues",
        "completeThroughSeason",
        "quarantinedSeasons",
    }:
        raise StrategyZooValidationError("dataset has an unsupported schema")
    if not isinstance(dataset.get("datasetId"), str) or not dataset["datasetId"]:
        raise StrategyZooValidationError("datasetId must be a non-empty string")
    if dataset.get("source") != "data/cache/football_data_csv":
        raise StrategyZooValidationError("dataset source is not canonical")
    if dataset.get("completeThroughSeason") != complete_through_season:
        raise StrategyZooValidationError("dataset and methodology season cutoffs disagree")
    match_count = dataset.get("matches")
    evaluated_matches = dataset.get("evaluatedMatches")
    quarantined_matches = dataset.get("quarantinedMatches")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (match_count, evaluated_matches, quarantined_matches)
    ) or evaluated_matches + quarantined_matches != match_count:
        raise StrategyZooValidationError("dataset match counts are inconsistent")
    if not match_count or not evaluated_matches:
        raise StrategyZooValidationError("dataset cannot be empty")
    try:
        start_date = datetime.fromisoformat(str(dataset.get("startDate"))).date()
        end_date = datetime.fromisoformat(str(dataset.get("endDate"))).date()
    except (TypeError, ValueError) as exc:
        raise StrategyZooValidationError("dataset dates must be ISO-8601 dates") from exc
    if start_date > end_date:
        raise StrategyZooValidationError("dataset dates are reversed")
    leagues = dataset.get("leagues")
    if (
        not isinstance(leagues, list)
        or not leagues
        or any(not isinstance(league, str) or not league for league in leagues)
        or leagues != sorted(set(leagues))
    ):
        raise StrategyZooValidationError("dataset leagues must be unique and sorted")
    seasons = payload.get("seasons")
    if not isinstance(seasons, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in seasons
    ):
        raise StrategyZooValidationError("seasons must be an integer array")
    if (
        seasons != sorted(set(seasons))
        or not seasons
        or seasons != list(range(seasons[0], seasons[-1] + 1))
        or complete_through_season not in seasons
    ):
        raise StrategyZooValidationError("seasons must be a complete, unique and sorted range")
    expected_quarantined_seasons = [season for season in seasons if season > complete_through_season]
    if dataset.get("quarantinedSeasons") != expected_quarantined_seasons:
        raise StrategyZooValidationError("dataset quarantined seasons disagree with cutoff")
    coverage = payload.get("coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != {"bySeason"}:
        raise StrategyZooValidationError("coverage has an unsupported schema")
    coverage_rows = coverage.get("bySeason")
    if not isinstance(coverage_rows, list) or [
        row.get("season") for row in coverage_rows if isinstance(row, Mapping)
    ] != seasons:
        raise StrategyZooValidationError("coverage must contain one row per season")
    coverage_totals = {
        "matches": 0,
        "evaluatedMatches": 0,
        "quarantinedMatches": 0,
    }
    for row in coverage_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "season",
            "matches",
            "evaluatedMatches",
            "quarantinedMatches",
            "b3651x2Matches",
            "b365Ou25Matches",
            "b3651x2CoveragePct",
            "b365Ou25CoveragePct",
        }:
            raise StrategyZooValidationError("coverage row has an unsupported schema")
        count_fields = (
            "matches",
            "evaluatedMatches",
            "quarantinedMatches",
            "b3651x2Matches",
            "b365Ou25Matches",
        )
        if any(isinstance(row.get(field), bool) or not isinstance(row.get(field), int) or row[field] < 0 for field in count_fields):
            raise StrategyZooValidationError("coverage row has invalid counts")
        if (
            row["evaluatedMatches"] + row["quarantinedMatches"] != row["matches"]
            or row["b3651x2Matches"] > row["matches"]
            or row["b365Ou25Matches"] > row["matches"]
            or (row["season"] <= complete_through_season and row["quarantinedMatches"] != 0)
            or (row["season"] > complete_through_season and row["evaluatedMatches"] != 0)
        ):
            raise StrategyZooValidationError("coverage row counts disagree")
        for count_field, rate_field in (
            ("b3651x2Matches", "b3651x2CoveragePct"),
            ("b365Ou25Matches", "b365Ou25CoveragePct"),
        ):
            expected_rate = row[count_field] / row["matches"] * 100.0 if row["matches"] else 0.0
            if not _finite_number(row.get(rate_field)) or abs(float(row[rate_field]) - expected_rate) > 0.011:
                raise StrategyZooValidationError("coverage percentage disagrees with counts")
        for field in coverage_totals:
            coverage_totals[field] += row[field]
    if coverage_totals != {
        "matches": match_count,
        "evaluatedMatches": evaluated_matches,
        "quarantinedMatches": quarantined_matches,
    }:
        raise StrategyZooValidationError("coverage totals disagree with dataset")
    _validate_season_market_profiles(
        payload.get("seasonMarketProfiles"),
        seasons,
        coverage_rows,
        evaluated_matches,
    )
    strategies = payload.get("strategies")
    definitions = strategy_definitions()
    if (
        not isinstance(strategies, list)
        or [strategy.get("id") for strategy in strategies if isinstance(strategy, Mapping)]
        != [definition.id for definition in definitions]
    ):
        raise StrategyZooValidationError("strategies must match the fixed strategy catalog")
    identifiers: set[str] = set()
    for strategy, definition in zip(strategies, definitions):
        if not isinstance(strategy, Mapping) or set(strategy) != {
            "id",
            "title",
            "family",
            "market",
            "rule",
            "comparison",
            "status",
            "statusReasons",
            "guaranteed",
            "firstActiveSeason",
            "lastActiveSeason",
            "overall",
            "yearly",
        }:
            raise StrategyZooValidationError("each strategy must use the fixed public schema")
        identifier = strategy.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise StrategyZooValidationError("strategy ids must be unique non-empty strings")
        identifiers.add(identifier)
        if (
            strategy.get("title") != definition.title
            or strategy.get("family") != definition.family
            or strategy.get("market") != definition.market
            or strategy.get("rule") != dict(definition.rule)
            or strategy.get("comparison") != _definition_comparison(definition)
        ):
            raise StrategyZooValidationError(f"{identifier} metadata differs from the fixed catalog")
        if strategy.get("guaranteed") is not False:
            raise StrategyZooValidationError(f"{identifier} must not claim a guarantee")
        if strategy.get("status") not in _ALLOWED_STRATEGY_STATUSES:
            raise StrategyZooValidationError(f"{identifier} has an unsupported status")
        yearly = strategy.get("yearly")
        if not isinstance(yearly, list) or [
            row.get("season") for row in yearly if isinstance(row, Mapping)
        ] != seasons:
            raise StrategyZooValidationError(f"{identifier} must contain one row for every season")
        for row in yearly:
            if not isinstance(row, Mapping) or set(row) != {
                "opportunities",
                "hits",
                "hitRatePct",
                "hitRateCi95Pct",
                "bets",
                "wins",
                "stakeUnits",
                "pnlAvailable",
                "pnlAvailabilityReason",
                "oddsCoveragePct",
                "averageOpeningOdds",
                "profitUnits",
                "roiPct",
                "roiCi95Pct",
                "maxDrawdownUnits",
                "activeSeasons",
                "pricedSeasons",
                "positivePricedSeasons",
                "positivePricedSeasonRatePct",
                "season",
                "label",
                "available",
                "quarantined",
                "quarantineReason",
                "availabilityReason",
            }:
                raise StrategyZooValidationError(f"{identifier}.yearly rows have an unsupported schema")
            season = row.get("season")
            if isinstance(season, bool) or not isinstance(season, int):
                raise StrategyZooValidationError(f"{identifier}.yearly has an invalid season")
            if row.get("label") != f"{season}/{str(season + 1)[-2:]}":
                raise StrategyZooValidationError(f"{identifier}.yearly[{season}] has an invalid label")
            _validate_metric(row, f"{identifier}.yearly[{season}]")
            expected_active_seasons = int(row["opportunities"] > 0)
            expected_priced_seasons = int(row["bets"] > 0)
            expected_positive_seasons = int(
                row.get("profitUnits") is not None and float(row["profitUnits"]) > 0.0
            )
            if (
                row["activeSeasons"] != expected_active_seasons
                or row["pricedSeasons"] != expected_priced_seasons
                or row["positivePricedSeasons"] != expected_positive_seasons
            ):
                raise StrategyZooValidationError(f"{identifier}.yearly[{season}] has inconsistent season counts")
            expected_quarantine = season > complete_through_season
            if row.get("quarantined") is not expected_quarantine:
                raise StrategyZooValidationError(f"{identifier} has an inconsistent quarantine flag")
            if row.get("quarantineReason") != (
                "incomplete_local_snapshot" if expected_quarantine else None
            ):
                raise StrategyZooValidationError(f"{identifier} has an inconsistent quarantine reason")
            expected_available = not expected_quarantine and row.get("opportunities", 0) > 0
            if row.get("available") is not expected_available:
                raise StrategyZooValidationError(f"{identifier} has an inconsistent availability flag")
            if expected_quarantine:
                if row.get("availabilityReason") != "incomplete_local_snapshot":
                    raise StrategyZooValidationError(f"{identifier} exposes a quarantined season")
                if int(row.get("opportunities", 0)) or int(row.get("bets", 0)):
                    raise StrategyZooValidationError(f"{identifier} counts quarantined results")
            elif not expected_available and row.get("availabilityReason") != "no_qualifying_opportunities":
                raise StrategyZooValidationError(f"{identifier} hides why a season is unavailable")
            elif expected_available and row.get("availabilityReason") is not None:
                raise StrategyZooValidationError(f"{identifier} marks an available season unavailable")
        overall = strategy.get("overall")
        if not isinstance(overall, Mapping) or set(overall) != {
            "opportunities",
            "hits",
            "hitRatePct",
            "hitRateCi95Pct",
                "bets",
                "wins",
                "stakeUnits",
                "pnlAvailable",
                "pnlAvailabilityReason",
            "oddsCoveragePct",
            "averageOpeningOdds",
            "profitUnits",
            "roiPct",
            "roiCi95Pct",
            "maxDrawdownUnits",
            "activeSeasons",
            "pricedSeasons",
            "positivePricedSeasons",
            "positivePricedSeasonRatePct",
        }:
            raise StrategyZooValidationError(f"{identifier}.overall has an unsupported schema")
        _validate_metric(overall, f"{identifier}.overall")
        for field in ("opportunities", "hits", "bets", "wins", "stakeUnits"):
            if overall[field] != sum(row[field] for row in yearly):
                raise StrategyZooValidationError(f"{identifier}.overall.{field} disagrees with yearly rows")
        yearly_profit = math.fsum(
            float(row["profitUnits"])
            for row in yearly
            if row.get("profitUnits") is not None
        )
        if overall["bets"] and abs(float(overall["profitUnits"]) - yearly_profit) > 0.2:
            raise StrategyZooValidationError(f"{identifier}.overall profit disagrees with yearly rows")
        expected_active_count = sum(row["opportunities"] > 0 for row in yearly)
        expected_priced_count = sum(row["bets"] > 0 for row in yearly)
        expected_positive_count = sum(
            row.get("profitUnits") is not None and float(row["profitUnits"]) > 0.0
            for row in yearly
        )
        if (
            overall["activeSeasons"] != expected_active_count
            or overall["pricedSeasons"] != expected_priced_count
            or overall["positivePricedSeasons"] != expected_positive_count
        ):
            raise StrategyZooValidationError(f"{identifier}.overall season counts disagree with yearly rows")
        expected_status, expected_reasons = _status(overall)
        if strategy.get("status") != expected_status or strategy.get("statusReasons") != expected_reasons:
            raise StrategyZooValidationError(f"{identifier} status does not match its evidence")
        active_seasons = [row["season"] for row in yearly if row["opportunities"] > 0]
        if strategy.get("firstActiveSeason") != (min(active_seasons) if active_seasons else None):
            raise StrategyZooValidationError(f"{identifier} has an inconsistent first active season")
        if strategy.get("lastActiveSeason") != (max(active_seasons) if active_seasons else None):
            raise StrategyZooValidationError(f"{identifier} has an inconsistent last active season")
        if strategy.get("market") == "exact_score" and overall["bets"] != 0:
            raise StrategyZooValidationError(f"{identifier} fabricates exact-score P&L")
        if strategy["rule"].get("outcomeOnly") is True and overall["bets"] != 0:
            raise StrategyZooValidationError(f"{identifier} fabricates outcome-only P&L")

    strategies_by_id = {strategy["id"]: strategy for strategy in strategies}
    favourite_strategy = strategies_by_id["all_unique_favourites"]
    favourite_profiles = payload["seasonMarketProfiles"]["bySeason"]
    for profile, metric in zip(favourite_profiles, favourite_strategy["yearly"]):
        if not metric["available"]:
            continue
        favourites = profile["teamFavourites"]
        if (
            favourites["uniqueTeamSelections"] != metric["opportunities"]
            or favourites["uniqueTeamSelections"] != metric["bets"]
            or favourites["won"] != metric["hits"]
            or favourites["won"] != metric["wins"]
        ):
            raise StrategyZooValidationError(
                "season market profiles and hold-favourite strategy disagree"
            )
    available_favourite_profiles = [
        profile
        for profile, metric in zip(favourite_profiles, favourite_strategy["yearly"])
        if metric["available"]
    ]
    if (
        favourite_strategy["overall"]["opportunities"]
        != sum(profile["teamFavourites"]["uniqueTeamSelections"] for profile in available_favourite_profiles)
        or favourite_strategy["overall"]["hits"]
        != sum(profile["teamFavourites"]["won"] for profile in available_favourite_profiles)
    ):
        raise StrategyZooValidationError(
            "all-time market profile and hold-favourite strategy disagree"
        )
    for strategy in strategies:
        comparison = strategy["comparison"]
        opposite_id = comparison["oppositeStrategyId"]
        if opposite_id is None:
            continue
        opposite = strategies_by_id.get(opposite_id)
        if opposite is None or opposite["comparison"]["oppositeStrategyId"] != strategy["id"]:
            raise StrategyZooValidationError(f"{strategy['id']} has a non-reciprocal comparison")
        for field in ("groupId", "kind", "sameOpportunitySet"):
            if comparison[field] != opposite["comparison"][field]:
                raise StrategyZooValidationError(f"{strategy['id']} comparison metadata disagrees")
        if comparison["sameOpportunitySet"]:
            for row, opposite_row in zip(strategy["yearly"], opposite["yearly"]):
                if (
                    row["opportunities"] != opposite_row["opportunities"]
                    or row["bets"] != opposite_row["bets"]
                    or row["hits"] + opposite_row["hits"] != row["opportunities"]
                    or row["wins"] + opposite_row["wins"] != row["bets"]
                ):
                    raise StrategyZooValidationError(
                        f"{strategy['id']} and {opposite_id} do not share complementary opportunities"
                    )

    if payload.get("selectionPolicy") != _selection_policy():
        raise StrategyZooValidationError("selectionPolicy is inconsistent")
    if payload.get("seasonAudits") != _build_season_audits(strategies, seasons):
        raise StrategyZooValidationError("seasonAudits disagree with causal strategy metrics")
    rivalries = payload.get("rivalryPatterns")
    if not isinstance(rivalries, list) or len(rivalries) > 40:
        raise StrategyZooValidationError("rivalryPatterns must be an array of at most 40 rows")
    rivalry_ids: set[str] = set()
    for index, rivalry in enumerate(rivalries):
        _validate_rivalry_pattern(rivalry, index, seasons)
        if rivalry["id"] in rivalry_ids:
            raise StrategyZooValidationError("rivalry pattern ids must be unique")
        rivalry_ids.add(rivalry["id"])
    findings = payload.get("findings")
    if (
        not isinstance(findings, Mapping)
        or set(findings) != {
            "globalStatus",
            "h2hValidation",
            "researchVerdict",
            "guarantees",
            "bestOddsRoi",
            "worstOddsRoi",
            "bestDrawRoi",
            "worstDrawRoi",
            "bestGoalsRoi",
            "worstGoalsRoi",
            "exactScoreReliability",
            "rivalryScreen",
        }
        or findings.get("globalStatus") != "NO_CONFIRMED_BETTING_EDGE"
        or findings.get("researchVerdict") != payload.get("researchVerdict")
    ):
        raise StrategyZooValidationError("findings must publish NO_CONFIRMED_BETTING_EDGE")
    h2h_validation = findings.get("h2hValidation")
    _validate_h2h_validation(h2h_validation, complete_through_season)
    guarantees = findings.get("guarantees")
    if (
        not isinstance(guarantees, Mapping)
        or set(guarantees) != {"alwaysWinsFound", "neverWinsFound", "message"}
        or guarantees.get("alwaysWinsFound") is not False
        or guarantees.get("neverWinsFound") is not False
        or guarantees.get("message") != (
            "Ingen historisk stikprøve gør et fremtidigt udfald sikkert; "
            "100% eller 0% observeret er ikke en garanti."
        )
    ):
        raise StrategyZooValidationError("findings must reject always/never guarantees")
    expected_extremes = {
        "bestOddsRoi": _extreme(strategies, {"odds"}, "roiPct", highest=True),
        "worstOddsRoi": _extreme(strategies, {"odds"}, "roiPct", highest=False),
        "bestDrawRoi": _extreme(strategies, {"draws"}, "roiPct", highest=True),
        "worstDrawRoi": _extreme(strategies, {"draws"}, "roiPct", highest=False),
        "bestGoalsRoi": _extreme(strategies, {"goals"}, "roiPct", highest=True),
        "worstGoalsRoi": _extreme(strategies, {"goals"}, "roiPct", highest=False),
    }
    for key, expected in expected_extremes.items():
        if findings.get(key) != expected:
            raise StrategyZooValidationError(f"findings.{key} disagrees with validated strategies")

    exact_score = findings.get("exactScoreReliability")
    exact_strategy = next(item for item in strategies if item["id"] == "league_exact_score_mode")
    h2h_exact_strategy = next(item for item in strategies if item["id"] == "directed_h2h_exact_score_mode")
    if not isinstance(exact_score, Mapping) or set(exact_score) != {
        "topObservedScores",
        "leagueMode",
        "directedH2hMode",
        "message",
    }:
        raise StrategyZooValidationError("findings.exactScoreReliability has an unsupported schema")
    if exact_score.get("leagueMode") != {
        "opportunities": exact_strategy["overall"]["opportunities"],
        "hitRatePct": exact_strategy["overall"]["hitRatePct"],
        "hitRateCi95Pct": exact_strategy["overall"]["hitRateCi95Pct"],
    } or exact_score.get("directedH2hMode") != {
        "opportunities": h2h_exact_strategy["overall"]["opportunities"],
        "hitRatePct": h2h_exact_strategy["overall"]["hitRatePct"],
        "hitRateCi95Pct": h2h_exact_strategy["overall"]["hitRateCi95Pct"],
    }:
        raise StrategyZooValidationError("exact-score findings disagree with validated strategies")
    top_scores = exact_score.get("topObservedScores")
    if not isinstance(top_scores, list) or not 1 <= len(top_scores) <= 5:
        raise StrategyZooValidationError("exact-score findings require one to five score rows")
    previous_count: int | None = None
    observed_scores: set[str] = set()
    for row in top_scores:
        if not isinstance(row, Mapping) or set(row) != {"score", "matches", "ratePct"}:
            raise StrategyZooValidationError("exact-score finding row has an unsupported schema")
        score = row.get("score")
        count = row.get("matches")
        if (
            not isinstance(score, str)
            or score in observed_scores
            or len(score.split("-")) != 2
            or any(not part.isdigit() for part in score.split("-"))
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or (previous_count is not None and count > previous_count)
        ):
            raise StrategyZooValidationError("exact-score finding row is invalid")
        expected_rate = count / evaluated_matches * 100.0
        if not _finite_number(row.get("ratePct")) or abs(float(row["ratePct"]) - expected_rate) > 0.011:
            raise StrategyZooValidationError("exact-score finding rate disagrees with the dataset")
        observed_scores.add(score)
        previous_count = count
    if exact_score.get("message") != (
        "1-1 er historisk den hyppigste score, men selv den rammer kun omtrent "
        "hver ottende kamp. Præcis score er et høj-varians forecast uden "
        "verificerede exact-score-odds i datasættet; der vises derfor ingen P&L."
    ):
        raise StrategyZooValidationError("exact-score finding message is inconsistent")

    perfect = [
        {
            "team": row["team"],
            "opponent": row["opponent"],
            "meetings": row["meetings"],
            "winRateCi95Pct": row["winRateCi95Pct"],
        }
        for row in rivalries
        if row["perfectWinRecord"]
    ]
    expected_rivalry_screen = {
        "reportedPairs": len(rivalries),
        "relationshipDefinition": "historical head-to-head pair; not necessarily a derby or cultural rivalry",
        "minimumHistoricalMeetings": 10,
        "minimumOutOfSampleSignals": 3,
        "perfectWinRecordPairs": perfect[:10],
        "perfectWinRecordCount": len(perfect),
        "perfectRecordConclusion": (
            "Ingen holdpar i den rapporterede stikprøve vandt alle mindst 10 møder."
            if not perfect
            else (
                "Observerede perfekte serier findes, men deres konfidensinterval "
                "er ikke 100%, og de er ikke fremtidige garantier."
            )
        ),
        "multipleTestingWarning": (
            "Mange holdpar er screenet. Ranglisten er beskrivende og må ikke "
            "læses som en garanti eller et automatisk spil."
        ),
    }
    if findings.get("rivalryScreen") != expected_rivalry_screen:
        raise StrategyZooValidationError("findings.rivalryScreen disagrees with rivalry patterns")
    try:
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StrategyZooValidationError(f"strategy zoo is not finite JSON: {exc}") from exc
    if len(encoded) > MAX_PUBLIC_BYTES:
        raise StrategyZooValidationError(
            f"strategy zoo is {len(encoded)} bytes; public limit is {MAX_PUBLIC_BYTES}"
        )
    return json.loads(encoded.decode("utf-8"))


def load_strategy_zoo(
    path: Path | str = DEFAULT_PUBLIC_PATH,
    *,
    require_checksum: bool = False,
) -> Dict[str, Any]:
    """Load an artifact, optionally checking its SHA-256 integrity sidecar.

    The sidecar detects accidental corruption.  It is deliberately not called
    an attestation or signature: a publisher must still deterministically
    compare the artifact with the canonical source data before publication.
    """

    artifact = Path(path)
    try:
        encoded = artifact.read_bytes()
        payload = json.loads(encoded.decode("utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrategyZooValidationError(f"cannot load strategy zoo: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise StrategyZooValidationError(f"cannot decode strategy zoo: {exc}") from exc
    if require_checksum:
        checksum = artifact.with_suffix(".sha256")
        try:
            expected_digest = checksum.read_text(encoding="ascii").strip()
        except OSError as exc:
            raise StrategyZooValidationError(f"cannot load strategy zoo checksum: {exc}") from exc
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise StrategyZooValidationError("strategy zoo checksum is malformed")
        actual_digest = hashlib.sha256(encoded).hexdigest()
        if actual_digest != expected_digest:
            raise StrategyZooValidationError("strategy zoo artifact does not match its checksum")
    return validate_strategy_zoo(payload)


__all__ = [
    "DEFAULT_PUBLIC_PATH",
    "DEFAULT_PUBLIC_CHECKSUM_PATH",
    "MAX_PUBLIC_BYTES",
    "SCHEMA_VERSION",
    "StrategyZooValidationError",
    "build_strategy_zoo",
    "load_strategy_zoo",
    "strategy_definitions",
    "validate_strategy_zoo",
]
