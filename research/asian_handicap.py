"""Canonical Asian Handicap extraction, settlement, and fixed benchmarks.

Football-Data's ``AHh``/``BbAHh`` value is the handicap applied to the home
team.  The away line is therefore its additive inverse.  Quarter-goal lines
split a stake equally across the adjacent half-goal lines; this module keeps
those half wins/losses and pushes as profit values instead of coercing them
into a binary target.

This module is research-only.  It deliberately does not alter live prediction
or staking code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Mapping, Sequence

ASIAN_OUTCOMES = ("win", "half_win", "push", "half_loss", "loss")
EXECUTABLE_ASIAN_BASES = ("b365", "pinnacle")
PROXY_ASIAN_BASES = ("avg", "max")


def valid_asian_handicap_line(value: object, *, max_abs_line: float = 10.0) -> float | None:
    """Return a canonical quarter-goal line, or ``None`` for bad source data.

    Football-Data contains a handful of malformed historical values such as
    ``-1.2``, ``0.758`` and ``-225``.  They are rejected, never rounded or
    silently repaired. Ten goals is only a corruption guard; the valid source
    lines in the canonical archive range from -3.75 to +3 goals.
    """

    try:
        line = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(line) or abs(line) > max_abs_line:
        return None
    quarter_units = line * 4.0
    if not math.isclose(quarter_units, round(quarter_units), abs_tol=1e-9):
        return None
    # Normalize negative zero and floating-point noise from CSV parsing.
    normalized = round(quarter_units) / 4.0
    return 0.0 if normalized == 0.0 else normalized


def split_asian_handicap_line(line: object) -> tuple[float, ...]:
    """Split a whole/half/quarter handicap into its settlement legs."""

    normalized = valid_asian_handicap_line(line, max_abs_line=float("inf"))
    if normalized is None:
        raise ValueError("Asian handicap line must be a finite quarter-goal value")
    half_units = normalized * 2.0
    if math.isclose(half_units, round(half_units), abs_tol=1e-9):
        return (normalized,)
    lower = math.floor(half_units) / 2.0
    return (lower, lower + 0.5)


def _score(value: object, name: str) -> int:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(number) or number < 0.0 or not number.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(number)


@dataclass(frozen=True)
class AsianHandicapSettlement:
    side: str
    home_handicap: float
    decimal_odds: float
    stake: float
    legs: tuple[float, ...]
    leg_results: tuple[int, ...]
    outcome: str
    profit: float
    returned: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _AsianMetricsAccumulator:
    """Constant-memory chronological metrics for benchmark streams."""

    bets: int = 0
    staked: float = 0.0
    profit: float = 0.0
    equity: float = 0.0
    peak: float = 0.0
    drawdown: float = 0.0
    outcomes: dict[str, int] = field(
        default_factory=lambda: {outcome: 0 for outcome in ASIAN_OUTCOMES}
    )

    def add(self, settlement: AsianHandicapSettlement) -> None:
        self.bets += 1
        self.staked += settlement.stake
        self.profit += settlement.profit
        self.outcomes[settlement.outcome] += 1
        self.equity += settlement.profit
        self.peak = max(self.peak, self.equity)
        self.drawdown = max(self.drawdown, self.peak - self.equity)

    def metrics(self) -> dict[str, object]:
        return {
            "bets": self.bets,
            "outcomes": dict(self.outcomes),
            "staked": self.staked,
            "returned": self.staked + self.profit,
            "profit": self.profit,
            "roi": self.profit / self.staked if self.staked else 0.0,
            "roi_pct": self.profit / self.staked * 100.0 if self.staked else 0.0,
            "max_drawdown": self.drawdown,
        }


def settle_asian_handicap(
    home_goals: object,
    away_goals: object,
    home_handicap: object,
    side: str,
    decimal_odds: object,
    *,
    stake: float = 1.0,
) -> AsianHandicapSettlement:
    """Settle one Asian Handicap selection at decimal odds.

    ``leg_results`` uses ``1`` for win, ``0`` for push and ``-1`` for loss.
    Profit is net of the original stake, so a push returns zero profit.
    """

    home = _score(home_goals, "home_goals")
    away = _score(away_goals, "away_goals")
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'")
    line = valid_asian_handicap_line(home_handicap, max_abs_line=float("inf"))
    if line is None:
        raise ValueError("home_handicap must be a finite quarter-goal value")
    try:
        odds = float(decimal_odds)  # type: ignore[arg-type]
        stake_value = float(stake)
    except (TypeError, ValueError) as exc:
        raise ValueError("decimal_odds and stake must be numeric") from exc
    if not math.isfinite(odds) or odds <= 1.0:
        raise ValueError("decimal_odds must be finite and greater than 1")
    if not math.isfinite(stake_value) or stake_value <= 0.0:
        raise ValueError("stake must be a positive finite number")

    home_legs = split_asian_handicap_line(line)
    # The stored line always belongs to the home side.  Mirroring both the
    # score margin and line produces the corresponding away selection.
    selection_legs = home_legs if side == "home" else tuple(-leg for leg in reversed(home_legs))
    score_margin = home - away if side == "home" else away - home
    leg_results = tuple(
        1 if score_margin + leg > 0.0 else -1 if score_margin + leg < 0.0 else 0
        for leg in selection_legs
    )
    leg_stake = stake_value / len(leg_results)
    profit = math.fsum(
        leg_stake * (odds - 1.0) if result > 0 else -leg_stake if result < 0 else 0.0
        for result in leg_results
    )
    result_key = {
        (1,): "win",
        (0,): "push",
        (-1,): "loss",
        (0, 1): "half_win",
        (1, 0): "half_win",
        (-1, 0): "half_loss",
        (0, -1): "half_loss",
        (1, 1): "win",
        (-1, -1): "loss",
    }.get(leg_results)
    if result_key is None:
        raise AssertionError(f"impossible adjacent Asian Handicap results: {leg_results}")
    return AsianHandicapSettlement(
        side=side,
        home_handicap=line,
        decimal_odds=odds,
        stake=stake_value,
        legs=selection_legs,
        leg_results=leg_results,
        outcome=result_key,
        profit=profit,
        returned=stake_value + profit,
    )


def asian_handicap_metrics(
    settlements: Sequence[AsianHandicapSettlement],
) -> dict[str, object]:
    """Aggregate already-settled selections without losing partial outcomes."""

    accumulator = _AsianMetricsAccumulator()
    for settlement in settlements:
        accumulator.add(settlement)
    return accumulator.metrics()


def fixed_blind_asian_benchmark(
    matches: Iterable[Mapping[str, object]],
    *,
    bases: Sequence[str] = EXECUTABLE_ASIAN_BASES + PROXY_ASIAN_BASES,
) -> list[dict[str, object]]:
    """Evaluate pre-declared blind home/away baselines in chronological order.

    This is a benchmark, not a selected betting strategy.  Bet365 and Pinnacle
    rows are labelled executable historical quotes; market average and maximum
    rows are explicitly labelled non-executable proxies.
    """

    # Local import avoids coupling the settlement primitive to pandas/features.
    from research.features import extract_asian_handicap_quotes

    selected_bases = tuple(dict.fromkeys(bases))
    unsupported = [
        basis
        for basis in selected_bases
        if basis not in EXECUTABLE_ASIAN_BASES + PROXY_ASIAN_BASES
    ]
    if unsupported:
        raise ValueError(f"unsupported Asian Handicap quote basis {unsupported[0]!r}")
    if not selected_bases:
        return []

    ordered = sorted(
        matches,
        key=lambda match: str(match.get("match_date") or match.get("kickoff") or ""),
    )
    accumulators = {
        (basis, side): _AsianMetricsAccumulator()
        for basis in selected_bases
        for side in ("home", "away")
    }
    for match in ordered:
        quotes = extract_asian_handicap_quotes(match)
        for basis in selected_bases:
            quote = quotes.get(basis)
            if quote is None:
                continue
            for side in ("home", "away"):
                try:
                    accumulators[(basis, side)].add(
                        settle_asian_handicap(
                            match.get("home_score"),
                            match.get("away_score"),
                            quote["home_line"],
                            side,
                            quote[side],
                        )
                    )
                except ValueError:
                    continue

    output: list[dict[str, object]] = []
    for basis in selected_bases:
        for side in ("home", "away"):
            output.append(
                {
                    "market": "asian_handicap",
                    "strategy": f"blind_{side}_all_available",
                    "side": side,
                    "odds_basis": basis,
                    "track": "executable" if basis in EXECUTABLE_ASIAN_BASES else "proxy",
                    **accumulators[(basis, side)].metrics(),
                }
            )
    return output


__all__ = [
    "ASIAN_OUTCOMES",
    "EXECUTABLE_ASIAN_BASES",
    "PROXY_ASIAN_BASES",
    "AsianHandicapSettlement",
    "asian_handicap_metrics",
    "fixed_blind_asian_benchmark",
    "settle_asian_handicap",
    "split_asian_handicap_line",
    "valid_asian_handicap_line",
]
