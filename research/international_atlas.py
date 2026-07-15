"""Deterministic international-tournament evidence atlas.

The checked-in international results snapshot is useful for describing what
happened, but its score fields are not guaranteed to stop at 90 minutes in
knockout matches.  Those rows therefore never produce a betting P&L claim.

An optional Football-Data World Cup workbook can add historical 90-minute
1X2 P&L.  It is parsed with the Python standard library, settles only the
``HGFT``/``AGFT`` fields, and records extra-time and penalty annotations.  The
output is deliberately labelled as hindsight evidence, not as a validated
forward betting edge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ElementTree
from zipfile import BadZipFile, ZipFile


SCHEMA = "aibets.international-atlas.v1"
WORLD_CUP_WORKBOOK_SOURCE_URL = (
    "https://www.football-data.co.uk/WorldCup2026.xlsx"
)
DEFAULT_RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "international"
    / "results_1990_plus.csv.gz"
)
DEFAULT_MANIFEST_PATH = DEFAULT_RESULTS_PATH.with_name("manifest.json")
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "international_atlas_public.json"
)
PINNED_WORLD_CUP_WORKBOOK_SHA256 = (
    "b14a24e218f25ffaa0037718471187bced6b835e2293b6e94cb2ce2a76ad544b"
)
PINNED_RESULTS_SNAPSHOT_SHA256 = (
    "a1046082e3d4eef99ea6b1561f4305f337c0c691d11d2355fdf32c84d18b144d"
)
PINNED_RESULTS_SNAPSHOT_ROWS = 32_387
PINNED_RESULTS_SNAPSHOT_START = "1990-01-12"
PINNED_RESULTS_SNAPSHOT_END = "2026-07-14"
GOAL_LINES: Tuple[float, ...] = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)
WORLD_CUP_EDITIONS: Tuple[int, ...] = tuple(range(1990, 2027, 4))
EURO_EDITIONS: Tuple[int, ...] = (
    1992,
    1996,
    2000,
    2004,
    2008,
    2012,
    2016,
    2020,
    2024,
)
SUPPORTED_WORLD_CUP_ODDS_EDITIONS: Tuple[int, ...] = (2014, 2018, 2022, 2026)
EXPECTED_WORLD_CUP_MATCHES: Mapping[int, int] = {
    2014: 64,
    2018: 64,
    2022: 64,
    # The expanded 48-team 2026 finals contain 104 matches.
    2026: 104,
}
PROFIT_HAIRCUT = 0.01
REGULATION_CAVEAT = (
    "The international results snapshot may include extra time in knockout "
    "scores; these figures are descriptive and are not used for 90-minute ROI."
)

_SPREADSHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CELL_REFERENCE = re.compile(r"^([A-Z]+)[0-9]+$")
_WORLD_CUP_SHEET = re.compile(r"^WorldCup(\d{4})$", re.IGNORECASE)

# Each quote must be a complete three-way quote from one named source.  The
# ordering is intentional and is reported in the output.
ODDS_SOURCE_PREFERENCE: Tuple[Tuple[str, Tuple[Tuple[str, str, str], ...]], ...] = (
    (
        "bet365",
        (
            ("bet365-H", "bet365-D", "bet365-A"),
            ("B365H", "B365D", "B365A"),
        ),
    ),
    (
        "pinnacle",
        (
            ("Pinny-H", "Pinny-D", "Pinny-A"),
            ("PSH", "PSD", "PSA"),
            ("PH", "PD", "PA"),
        ),
    ),
    (
        "betfair_exchange",
        (
            ("Betfair_Exch-H", "Betfair_Exch-D", "Betfair_Exch-A"),
            ("BFH", "BFD", "BFA"),
        ),
    ),
)

_ROI_STRATEGIES: Tuple[Tuple[str, str], ...] = (
    ("home_win", "Back every listed home team"),
    ("draw", "Back the 90-minute draw"),
    ("away_win", "Back every listed away team"),
    ("favourite", "Back the unique shortest-priced 1X2 selection"),
    ("longshot", "Back the unique longest-priced 1X2 selection"),
    ("odds_on_favourite", "Back the unique favourite only below decimal 2.00"),
)


@dataclass(frozen=True)
class InternationalResult:
    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool


@dataclass(frozen=True)
class _CompetitionSpec:
    competition_id: str
    label: str
    finals_name: str
    qualification_name: str
    editions: Tuple[int, ...]


_COMPETITIONS: Tuple[_CompetitionSpec, ...] = (
    _CompetitionSpec(
        "fifa_world_cup",
        "FIFA World Cup",
        "FIFA World Cup",
        "FIFA World Cup qualification",
        WORLD_CUP_EDITIONS,
    ),
    _CompetitionSpec(
        "uefa_euro",
        "UEFA Euro",
        "UEFA Euro",
        "UEFA Euro qualification",
        EURO_EDITIONS,
    ),
)


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        return None
    return int(number)


def _price(value: Any) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 1.0:
        return None
    return price


def _truth(value: Any) -> Optional[bool]:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def load_international_results(
    path: Path | str = DEFAULT_RESULTS_PATH,
) -> Tuple[InternationalResult, ...]:
    """Load and validate the normalized international results snapshot."""

    source_path = Path(path)
    opener = gzip.open if source_path.suffix.casefold() == ".gz" else open
    results: List[InternationalResult] = []
    with opener(source_path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        }
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError("international results snapshot has an unsupported schema")
        for row in reader:
            try:
                match_date = date.fromisoformat(str(row.get("date") or ""))
            except ValueError:
                continue
            home_score = _integer(row.get("home_score"))
            away_score = _integer(row.get("away_score"))
            neutral = _truth(row.get("neutral"))
            home_team = str(row.get("home_team") or "").strip()
            away_team = str(row.get("away_team") or "").strip()
            tournament = str(row.get("tournament") or "").strip()
            if (
                home_score is None
                or away_score is None
                or neutral is None
                or not home_team
                or not away_team
                or not tournament
            ):
                continue
            results.append(
                InternationalResult(
                    match_date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    tournament=tournament,
                    neutral=neutral,
                )
            )
    return tuple(
        sorted(
            results,
            key=lambda row: (
                row.match_date,
                row.tournament.casefold(),
                row.home_team.casefold(),
                row.away_team.casefold(),
            ),
        )
    )


def _coerce_result(value: InternationalResult | Mapping[str, Any]) -> Optional[InternationalResult]:
    if isinstance(value, InternationalResult):
        return value
    if not isinstance(value, Mapping):
        return None
    raw_date = value.get("match_date", value.get("date"))
    if isinstance(raw_date, datetime):
        match_date = raw_date.date()
    elif isinstance(raw_date, date):
        match_date = raw_date
    else:
        try:
            match_date = date.fromisoformat(str(raw_date or "")[:10])
        except ValueError:
            return None
    home_score = _integer(value.get("home_score"))
    away_score = _integer(value.get("away_score"))
    neutral = _truth(value.get("neutral"))
    home_team = str(value.get("home_team") or "").strip()
    away_team = str(value.get("away_team") or "").strip()
    tournament = str(value.get("tournament") or "").strip()
    if (
        home_score is None
        or away_score is None
        or neutral is None
        or not home_team
        or not away_team
        or not tournament
    ):
        return None
    return InternationalResult(
        match_date,
        home_team,
        away_team,
        home_score,
        away_score,
        tournament,
        neutral,
    )


def _rate(count: int, total: int) -> Dict[str, Any]:
    return {
        "count": count,
        "ratePct": round(100.0 * count / total, 2) if total else None,
    }


def _line_key(line: float) -> str:
    return format(line, "g")


def _strategy_hit(
    strategy_id: str,
    label: str,
    hits: int,
    opportunities: int,
) -> Dict[str, Any]:
    return {
        "strategyId": strategy_id,
        "label": label,
        "opportunities": opportunities,
        "hits": hits,
        "hitRatePct": round(100.0 * hits / opportunities, 2) if opportunities else None,
        "roiPct": None,
        "evidenceStatus": "descriptive_no_odds",
    }


def _summarize_results(
    rows: Sequence[InternationalResult],
    *,
    edition: int,
) -> Dict[str, Any]:
    matches = len(rows)
    home_wins = sum(row.home_score > row.away_score for row in rows)
    draws = sum(row.home_score == row.away_score for row in rows)
    away_wins = matches - home_wins - draws
    neutral = sum(row.neutral for row in rows)
    totals = [row.home_score + row.away_score for row in rows]
    btts = sum(row.home_score > 0 and row.away_score > 0 for row in rows)
    scores = Counter(f"{row.home_score}-{row.away_score}" for row in rows)
    over = {
        _line_key(line): _rate(sum(total > line for total in totals), matches)
        for line in GOAL_LINES
    }
    under = {
        _line_key(line): _rate(sum(total < line for total in totals), matches)
        for line in GOAL_LINES
    }
    strategies = [
        _strategy_hit("home_win", "Listed home team wins", home_wins, matches),
        _strategy_hit("draw", "Match is level at the source score", draws, matches),
        _strategy_hit("away_win", "Listed away team wins", away_wins, matches),
        _strategy_hit("btts_yes", "Both teams score", btts, matches),
        _strategy_hit("btts_no", "At least one team does not score", matches - btts, matches),
    ]
    for line in GOAL_LINES:
        key = _line_key(line)
        strategies.extend(
            (
                _strategy_hit(
                    f"over_{key.replace('.', '_')}",
                    f"Over {key} total goals",
                    int(over[key]["count"]),
                    matches,
                ),
                _strategy_hit(
                    f"under_{key.replace('.', '_')}",
                    f"Under {key} total goals",
                    int(under[key]["count"]),
                    matches,
                ),
            )
        )
    return {
        "edition": edition,
        "calendarYears": sorted({row.match_date.year for row in rows}),
        "matches": matches,
        "scoreScope": "source_final_score_not_guaranteed_90_minutes",
        "regulationCaveat": REGULATION_CAVEAT,
        "neutral": _rate(neutral, matches),
        "results": {
            "home": _rate(home_wins, matches),
            "draw": _rate(draws, matches),
            "away": _rate(away_wins, matches),
        },
        "goals": {
            "average": round(math.fsum(totals) / matches, 2) if matches else None,
            "over": over,
            "under": under,
        },
        "btts": {
            "yes": _rate(btts, matches),
            "no": _rate(matches - btts, matches),
        },
        "commonScores": [
            {"score": score, **_rate(count, matches)}
            for score, count in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:5]
        ],
        "strategyHitRates": strategies,
    }


def _finals_edition(match: InternationalResult, spec: _CompetitionSpec) -> Optional[int]:
    if spec.competition_id == "uefa_euro" and match.match_date.year == 2021:
        return 2020
    year = match.match_date.year
    return year if year in spec.editions else None


def _qualification_edition(
    match: InternationalResult,
    spec: _CompetitionSpec,
) -> Optional[int]:
    return next((edition for edition in spec.editions if edition >= match.match_date.year), None)


def _competition_payload(
    results: Sequence[InternationalResult],
    spec: _CompetitionSpec,
) -> Dict[str, Any]:
    finals: Dict[int, List[InternationalResult]] = {}
    qualification: Dict[int, List[InternationalResult]] = {}
    unassigned_qualification = 0
    for match in results:
        if match.tournament == spec.finals_name:
            edition = _finals_edition(match, spec)
            if edition is not None:
                finals.setdefault(edition, []).append(match)
        elif match.tournament == spec.qualification_name:
            edition = _qualification_edition(match, spec)
            if edition is None:
                unassigned_qualification += 1
            else:
                qualification.setdefault(edition, []).append(match)
    finals_rows = [
        _summarize_results(finals[edition], edition=edition)
        for edition in sorted(finals)
    ]
    qualification_rows = [
        _summarize_results(qualification[edition], edition=edition)
        for edition in sorted(qualification)
    ]
    return {
        "id": spec.competition_id,
        "label": spec.label,
        "finals": {
            "evidenceStatus": "observed_descriptive_only",
            "matches": sum(row["matches"] for row in finals_rows),
            "editions": finals_rows,
        },
        "qualification": {
            "evidenceStatus": "observed_descriptive_only",
            "matches": sum(row["matches"] for row in qualification_rows),
            "unassignedMatches": unassigned_qualification,
            "editions": qualification_rows,
        },
    }


def _cell_column(reference: str) -> Optional[str]:
    match = _CELL_REFERENCE.match(reference)
    return match.group(1) if match else None


def _shared_strings(archive: ZipFile) -> List[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{{{_SPREADSHEET_NS}}}t"))
        for item in root.findall(f"{{{_SPREADSHEET_NS}}}si")
    ]


def _cell_value(cell: ElementTree.Element, shared: Sequence[str]) -> Any:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{{{_SPREADSHEET_NS}}}t")
        )
    value = cell.find(f"{{{_SPREADSHEET_NS}}}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        try:
            return shared[int(value.text)]
        except (IndexError, ValueError):
            return None
    return value.text


def _workbook_sheets(archive: ZipFile) -> List[Tuple[str, str]]:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    targets = {
        relationship.attrib.get("Id"): relationship.attrib.get("Target")
        for relationship in relationships.findall(f"{{{_PACKAGE_REL_NS}}}Relationship")
    }
    sheets: List[Tuple[str, str]] = []
    sheet_container = workbook.find(f"{{{_SPREADSHEET_NS}}}sheets")
    if sheet_container is None:
        return sheets
    for sheet in sheet_container:
        relationship_id = sheet.attrib.get(f"{{{_OFFICE_REL_NS}}}id")
        target = targets.get(relationship_id)
        if not target:
            continue
        if target.startswith("/"):
            normalized = target.lstrip("/")
        else:
            normalized = str(PurePosixPath("xl") / target)
        if ".." in PurePosixPath(normalized).parts:
            continue
        sheets.append((str(sheet.attrib.get("name") or ""), normalized))
    return sheets


def _worksheet_rows(
    archive: ZipFile,
    target: str,
    shared: Sequence[str],
) -> List[Dict[str, Any]]:
    root = ElementTree.fromstring(archive.read(target))
    raw_rows: List[Dict[str, Any]] = []
    for row in root.findall(
        f".//{{{_SPREADSHEET_NS}}}sheetData/{{{_SPREADSHEET_NS}}}row"
    ):
        values: Dict[str, Any] = {}
        for cell in row.findall(f"{{{_SPREADSHEET_NS}}}c"):
            column = _cell_column(str(cell.attrib.get("r") or ""))
            if column is not None:
                values[column] = _cell_value(cell, shared)
        raw_rows.append(values)
    if not raw_rows:
        return []
    headers = {
        column: str(value).strip()
        for column, value in raw_rows[0].items()
        if value is not None and str(value).strip()
    }
    return [
        {
            header: values.get(column)
            for column, header in headers.items()
        }
        for values in raw_rows[1:]
    ]


def _excel_date(value: Any) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip()
    try:
        serial = float(text)
    except ValueError:
        for parser in (
            lambda: date.fromisoformat(text[:10]),
            lambda: datetime.strptime(text, "%d/%m/%Y").date(),
        ):
            try:
                return parser().isoformat()
            except ValueError:
                continue
        return None
    if not math.isfinite(serial) or serial < 1:
        return None
    return (date(1899, 12, 30) + timedelta(days=int(serial))).isoformat()


def _pick_quote(row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    for source, aliases in ODDS_SOURCE_PREFERENCE:
        for home_key, draw_key, away_key in aliases:
            quote = (
                _price(row.get(home_key)),
                _price(row.get(draw_key)),
                _price(row.get(away_key)),
            )
            if all(price is not None for price in quote):
                return {
                    "source": source,
                    "home": float(quote[0]),
                    "draw": float(quote[1]),
                    "away": float(quote[2]),
                }
    return None


def parse_world_cup_xlsx(path: Path | str) -> List[Dict[str, Any]]:
    """Parse supported Football-Data World Cup finals sheets without pandas.

    Only rows containing valid ``HGFT`` and ``AGFT`` scores are returned.  A
    missing quote remains explicit and can never fall back to Max/Avg prices.
    """

    parsed: List[Dict[str, Any]] = []
    try:
        with ZipFile(Path(path)) as archive:
            shared = _shared_strings(archive)
            for sheet_name, target in _workbook_sheets(archive):
                sheet_match = _WORLD_CUP_SHEET.match(sheet_name)
                if sheet_match is None:
                    continue
                edition = int(sheet_match.group(1))
                if edition not in SUPPORTED_WORLD_CUP_ODDS_EDITIONS:
                    continue
                for row in _worksheet_rows(archive, target, shared):
                    home_score = _integer(row.get("HGFT"))
                    away_score = _integer(row.get("AGFT"))
                    home_team = str(row.get("Home") or "").strip()
                    away_team = str(row.get("Away") or "").strip()
                    if (
                        home_score is None
                        or away_score is None
                        or not home_team
                        or not away_team
                    ):
                        continue
                    finished = str(row.get("Finished") or "").strip()
                    normalized_finished = finished.casefold()
                    parsed.append(
                        {
                            "edition": edition,
                            "sheet": sheet_name,
                            "date": _excel_date(row.get("Date")),
                            "homeTeam": home_team,
                            "awayTeam": away_team,
                            "regulationHomeGoals": home_score,
                            "regulationAwayGoals": away_score,
                            "finished": finished or None,
                            "extraTime": "extra" in normalized_finished,
                            "penalties": "penalt" in normalized_finished,
                            "settlementMarket": "90_minute_1x2",
                            "settlementScoreFields": ["HGFT", "AGFT"],
                            "odds": _pick_quote(row),
                        }
                    )
    except (BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise ValueError("unsupported or corrupt World Cup XLSX workbook") from error
    return sorted(
        parsed,
        key=lambda row: (
            row["edition"],
            row["date"] or "",
            row["homeTeam"].casefold(),
            row["awayTeam"].casefold(),
        ),
    )


def _result_side(row: Mapping[str, Any]) -> str:
    home = int(row["regulationHomeGoals"])
    away = int(row["regulationAwayGoals"])
    if home > away:
        return "home"
    if home < away:
        return "away"
    return "draw"


def _strategy_selection(strategy_id: str, row: Mapping[str, Any]) -> Optional[str]:
    if strategy_id in {"home_win", "draw", "away_win"}:
        return {"home_win": "home", "draw": "draw", "away_win": "away"}[strategy_id]
    odds = row.get("odds")
    if not isinstance(odds, Mapping):
        return None
    prices = {side: float(odds[side]) for side in ("home", "draw", "away")}
    target = (
        min(prices.values())
        if strategy_id in {"favourite", "odds_on_favourite"}
        else max(prices.values())
    )
    selections = [side for side, price in prices.items() if price == target]
    if len(selections) != 1:
        return None
    if strategy_id == "odds_on_favourite" and target >= 2.0:
        return None
    return selections[0]


def _roi_evidence(
    strategy_id: str,
    label: str,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    returns: List[float] = []
    wins = 0
    odds_sum = 0.0
    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for row in rows:
        odds = row.get("odds")
        if not isinstance(odds, Mapping):
            continue
        selection = _strategy_selection(strategy_id, row)
        if selection is None:
            continue
        price = float(odds[selection])
        won = selection == _result_side(row)
        profit = (
            (price - 1.0) * (1.0 - PROFIT_HAIRCUT)
            if won
            else -1.0
        )
        returns.append(profit)
        wins += int(won)
        odds_sum += price
        equity += profit
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    bets = len(returns)
    profit = math.fsum(returns)
    roi = profit / bets if bets else None
    if bets >= 2:
        mean = profit / bets
        variance = math.fsum((value - mean) ** 2 for value in returns) / (bets - 1)
        margin = 1.959963984540054 * math.sqrt(variance / bets)
        ci = {
            "lower": round(100.0 * (mean - margin), 2),
            "upper": round(100.0 * (mean + margin), 2),
        }
    else:
        ci = {"lower": None, "upper": None}
    return {
        "strategyId": strategy_id,
        "label": label,
        "bets": bets,
        "wins": wins,
        "hitRatePct": round(100.0 * wins / bets, 2) if bets else None,
        "averageOdds": round(odds_sum / bets, 3) if bets else None,
        "profitUnits": round(profit, 3),
        "roiPct": round(100.0 * roi, 2) if roi is not None else None,
        "ci95Pct": ci,
        "maxDrawdownUnits": round(max_drawdown, 3),
        "evidenceStatus": "historical_hindsight_only",
        "confirmedEdge": False,
        "profitClaimAllowed": False,
    }


def _world_cup_odds_payload(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_sha256: Optional[str],
) -> Dict[str, Any]:
    provenance = {
        "source": "Football-Data World Cup workbook",
        "sourceUrl": WORLD_CUP_WORKBOOK_SOURCE_URL,
        "sourceSha256": source_sha256,
    }
    editions: List[Dict[str, Any]] = []
    for edition in SUPPORTED_WORLD_CUP_ODDS_EDITIONS:
        edition_rows = [row for row in rows if int(row["edition"]) == edition]
        if not edition_rows:
            continue
        quoted = [row for row in edition_rows if isinstance(row.get("odds"), Mapping)]
        sources = Counter(str(row["odds"]["source"]) for row in quoted)
        dates = sorted(str(row["date"]) for row in edition_rows if row.get("date"))
        expected_matches = EXPECTED_WORLD_CUP_MATCHES.get(edition)
        editions.append(
            {
                "edition": edition,
                "workbookMatches": len(edition_rows),
                "expectedMatches": expected_matches,
                "complete": (
                    len(edition_rows) == expected_matches
                    if expected_matches is not None
                    else None
                ),
                "startDate": dates[0] if dates else None,
                "endDate": dates[-1] if dates else None,
                "quotedMatches": len(quoted),
                "unquotedMatches": len(edition_rows) - len(quoted),
                "extraTimeRows": sum(bool(row["extraTime"]) for row in edition_rows),
                "penaltyRows": sum(bool(row["penalties"]) for row in edition_rows),
                "quoteSources": dict(sorted(sources.items())),
                "settlement": {
                    "market": "90_minute_1x2",
                    "scoreFields": ["HGFT", "AGFT"],
                    "extraTimeAndPenaltiesIgnored": True,
                    "tournamentWinnerUsed": False,
                },
                "strategies": [
                    _roi_evidence(strategy_id, label, edition_rows)
                    for strategy_id, label in _ROI_STRATEGIES
                ],
            }
        )
    if not editions:
        return {
            **provenance,
            "evidenceStatus": "unavailable_no_verified_odds",
            "startDate": None,
            "endDate": None,
            "editions": [],
        }
    all_dates = sorted(str(row["date"]) for row in rows if row.get("date"))
    return {
        **provenance,
        "evidenceStatus": "historical_hindsight_only",
        "market": "90_minute_1x2",
        "priceTiming": "pre-closing",
        "startDate": all_dates[0] if all_dates else None,
        "endDate": all_dates[-1] if all_dates else None,
        "oddsSourcePreference": [source for source, _aliases in ODDS_SOURCE_PREFERENCE],
        "profitHaircutPct": PROFIT_HAIRCUT * 100.0,
        "profitHaircutBasis": "winning_gross_profit_only",
        "maxOrAverageOddsFallback": False,
        "editions": editions,
    }


def build_international_atlas(
    results: Optional[
        Iterable[InternationalResult | Mapping[str, Any]]
    ] = None,
    *,
    results_path: Path | str = DEFAULT_RESULTS_PATH,
    world_cup_xlsx: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Build a JSON-ready atlas for World Cup and Euro finals/qualification."""

    source_results = load_international_results(results_path) if results is None else results
    normalized = [
        result
        for raw in source_results
        if (result := _coerce_result(raw)) is not None
    ]
    normalized.sort(
        key=lambda row: (
            row.match_date,
            row.tournament.casefold(),
            row.home_team.casefold(),
            row.away_team.casefold(),
        )
    )
    if world_cup_xlsx:
        workbook_path = Path(world_cup_xlsx)
        workbook_sha256 = hashlib.sha256(workbook_path.read_bytes()).hexdigest()
        odds_rows = parse_world_cup_xlsx(workbook_path)
    else:
        workbook_sha256 = None
        odds_rows = []
    competitions = [
        _competition_payload(normalized, spec)
        for spec in _COMPETITIONS
    ]
    world_cup_odds = _world_cup_odds_payload(
        odds_rows,
        source_sha256=workbook_sha256,
    )
    return {
        "schema": SCHEMA,
        "dataset": {
            "matches": len(normalized),
            "startDate": normalized[0].match_date.isoformat() if normalized else None,
            "endDate": normalized[-1].match_date.isoformat() if normalized else None,
        },
        "methodology": {
            "finalsAndQualificationSeparated": True,
            "goalLines": [_line_key(line) for line in GOAL_LINES],
            "internationalScoreScope": "descriptive_source_final_score",
            "regulationCaveat": REGULATION_CAVEAT,
            "worldCupRoiSettlement": "HGFT/AGFT 90-minute 1X2 only",
            "tournamentWinnerMarketsUsed": False,
            "hindsightIsNotEdge": True,
        },
        "competitions": competitions,
        "worldCupOdds": world_cup_odds,
        "claims": [
            {
                "claimId": "international_results_patterns",
                "evidenceStatus": "observed_descriptive_only",
                "allowed": True,
                "roiAllowed": False,
                "reason": REGULATION_CAVEAT,
            },
            {
                "claimId": "world_cup_1x2_roi",
                "evidenceStatus": world_cup_odds["evidenceStatus"],
                "allowed": bool(world_cup_odds["editions"]),
                "confirmedEdge": False,
                "reason": (
                    "Named historical strategies settled from workbook HGFT/AGFT; "
                    "results are hindsight evidence, not a forward edge."
                    if world_cup_odds["editions"]
                    else "No verified World Cup 1X2 workbook was supplied."
                ),
            },
            {
                "claimId": "uefa_euro_roi",
                "evidenceStatus": "unavailable_no_verified_odds",
                "allowed": False,
                "confirmedEdge": False,
                "reason": "No verified historical UEFA Euro 90-minute odds were supplied.",
            },
        ],
    }


def verify_public_results_snapshot(
    results_path: Path | str = DEFAULT_RESULTS_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> Mapping[str, Any]:
    """Verify the public snapshot bytes and metadata against its manifest."""

    results_file = Path(results_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("international snapshot manifest is unavailable or invalid") from exc
    snapshot = manifest.get("snapshot")
    source = manifest.get("source")
    if manifest.get("schema") != "aibets.international_manifest.v1" or not isinstance(
        snapshot, Mapping
    ) or not isinstance(source, Mapping):
        raise RuntimeError("international snapshot manifest has an unsupported schema")
    from src.predictions.international_model import (
        PINNED_SOURCE_COMMIT,
        PINNED_SOURCE_SHA256,
        PINNED_SOURCE_URL,
    )

    if (
        source.get("commit") != PINNED_SOURCE_COMMIT
        or source.get("sha256") != PINNED_SOURCE_SHA256
        or source.get("url") != PINNED_SOURCE_URL
    ):
        raise RuntimeError("international snapshot source does not match the reviewed pin")
    declared_path = snapshot.get("path")
    if not isinstance(declared_path, str) or not declared_path:
        raise RuntimeError("international snapshot manifest path is invalid")
    manifested_file = (manifest_file.parent / declared_path).resolve()
    if manifested_file != results_file or not results_file.is_file():
        raise RuntimeError("international public build must use the manifested snapshot")
    digest = hashlib.sha256(results_file.read_bytes()).hexdigest()
    if digest != snapshot.get("sha256") or digest != PINNED_RESULTS_SNAPSHOT_SHA256:
        raise RuntimeError("international snapshot checksum does not match its manifest")
    rows = load_international_results(results_file)
    if (
        isinstance(snapshot.get("rows"), bool)
        or snapshot.get("rows") != len(rows)
        or len(rows) != PINNED_RESULTS_SNAPSHOT_ROWS
        or not rows
        or snapshot.get("start") != rows[0].match_date.isoformat()
        or snapshot.get("end") != rows[-1].match_date.isoformat()
        or snapshot.get("start") != PINNED_RESULTS_SNAPSHOT_START
        or snapshot.get("end") != PINNED_RESULTS_SNAPSHOT_END
    ):
        raise RuntimeError("international snapshot coverage does not match its manifest")
    return manifest


def verify_public_world_cup_workbook(path: Path | str) -> None:
    """Pin the exact reviewed World Cup workbook used for public ROI."""

    workbook = Path(path)
    try:
        digest = hashlib.sha256(workbook.read_bytes()).hexdigest()
    except OSError as exc:
        raise RuntimeError("the pinned World Cup workbook is unavailable") from exc
    if digest != PINNED_WORLD_CUP_WORKBOOK_SHA256:
        raise RuntimeError("World Cup workbook checksum is not the reviewed public pin")
    edition_counts = Counter(row["edition"] for row in parse_world_cup_xlsx(workbook))
    expected_counts = {2014: 64, 2018: 64, 2022: 64, 2026: 100}
    if edition_counts != expected_counts:
        raise RuntimeError("World Cup workbook edition coverage is not the reviewed public pin")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_international_atlas(
    payload: Mapping[str, Any],
    output_path: Path | str,
) -> Dict[str, Any]:
    """Atomically write canonical JSON plus a SHA-256 sidecar."""

    output = Path(output_path)
    canonical = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    checksum_path = output.with_name(f"{output.name}.sha256")
    checksum = f"{digest}  {output.name}\n".encode("ascii")
    _atomic_write(output, canonical)
    _atomic_write(checksum_path, checksum)
    return {
        "output": str(output),
        "sha256Path": str(checksum_path),
        "sha256": digest,
        "bytes": len(canonical),
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic AIbets international evidence atlas."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Normalized international results CSV or CSV.GZ snapshot.",
    )
    parser.add_argument(
        "--world-cup-xlsx",
        type=Path,
        default=None,
        help="Optional Football-Data World Cup XLSX with named 1X2 odds.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest that pins the normalized international results snapshot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON path; a .sha256 sidecar is written alongside it.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_cli_parser().parse_args(argv)
    if arguments.output.resolve() == DEFAULT_OUTPUT_PATH.resolve():
        verify_public_results_snapshot(arguments.results, arguments.manifest)
        if arguments.world_cup_xlsx is None:
            raise RuntimeError(
                "refusing to publish International Atlas without the pinned World Cup workbook"
            )
        verify_public_world_cup_workbook(arguments.world_cup_xlsx)
    payload = build_international_atlas(
        results_path=arguments.results,
        world_cup_xlsx=arguments.world_cup_xlsx,
    )
    receipt = write_international_atlas(payload, arguments.output)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "DEFAULT_RESULTS_PATH",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_OUTPUT_PATH",
    "EXPECTED_WORLD_CUP_MATCHES",
    "GOAL_LINES",
    "InternationalResult",
    "ODDS_SOURCE_PREFERENCE",
    "PINNED_WORLD_CUP_WORKBOOK_SHA256",
    "PINNED_RESULTS_SNAPSHOT_SHA256",
    "PINNED_RESULTS_SNAPSHOT_ROWS",
    "PINNED_RESULTS_SNAPSHOT_START",
    "PINNED_RESULTS_SNAPSHOT_END",
    "SCHEMA",
    "WORLD_CUP_WORKBOOK_SOURCE_URL",
    "build_cli_parser",
    "build_international_atlas",
    "load_international_results",
    "main",
    "parse_world_cup_xlsx",
    "verify_public_results_snapshot",
    "verify_public_world_cup_workbook",
    "write_international_atlas",
]


if __name__ == "__main__":
    raise SystemExit(main())
