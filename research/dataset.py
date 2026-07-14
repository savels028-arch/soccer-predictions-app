"""Canonical, network-free loader for Football-Data research matches.

The research dataset intentionally has no configurable input directory.  It
only reads the repository's immutable raw CSV cache and uses the production
CSV row normalizer without calling any download methods.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from research.asian_handicap import valid_asian_handicap_line
from src.api.csv_football_client import FootballDataCSVClient


MIN_SEASON = 1993
MAX_SEASON = 2025
# The local 2025/26 snapshot is incomplete and must not be selected by
# default.  It remains within MAX_SEASON so explicit audits can quarantine or
# inspect it, while production research stops at the last complete season.
LATEST_COMPLETE_SEASON = 2024
CANONICAL_CACHE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "cache" / "football_data_csv"
)

# Football-Data file code -> stable research league identity.  In particular,
# B1 is Belgium; the production BSA code denotes Brazil and must never be used
# for these files.
CSV_LEAGUES: Dict[str, Dict[str, str]] = {
    "E0": {"code": "PL", "name": "Premier League", "country": "England"},
    "E1": {"code": "ELC", "name": "Championship", "country": "England"},
    "SP1": {"code": "PD", "name": "La Liga", "country": "Spain"},
    "D1": {"code": "BL1", "name": "Bundesliga", "country": "Germany"},
    "D2": {"code": "BL2", "name": "2. Bundesliga", "country": "Germany"},
    "I1": {"code": "SA", "name": "Serie A", "country": "Italy"},
    "F1": {"code": "FL1", "name": "Ligue 1", "country": "France"},
    "N1": {"code": "DED", "name": "Eredivisie", "country": "Netherlands"},
    "P1": {"code": "PPL", "name": "Primeira Liga", "country": "Portugal"},
    "B1": {"code": "BEL1", "name": "Belgian First Division A", "country": "Belgium"},
}

_FILE_RE = re.compile(r"^(?P<season>\d{4})_(?P<league>[A-Z0-9]+)\.csv$")


def decode_season_code(code: str) -> Optional[int]:
    """Decode a Football-Data code such as ``9394`` or ``0001``.

    Only the canonical 1993/94 through 2025/26 range is accepted.  Malformed
    and non-consecutive year pairs return ``None`` instead of being guessed.
    """

    value = str(code)
    if not re.fullmatch(r"\d{4}", value):
        return None
    first = int(value[:2])
    second = int(value[2:])
    start = (1900 if first >= 93 else 2000) + first
    if not MIN_SEASON <= start <= MAX_SEASON:
        return None
    if second != (start + 1) % 100:
        return None
    return start


def _selected_leagues(leagues: Optional[Iterable[str]]) -> set[str]:
    research_codes = {info["code"] for info in CSV_LEAGUES.values()}
    if leagues is None:
        return research_codes
    if isinstance(leagues, str):
        values: Sequence[str] = [leagues]
    else:
        values = list(leagues)
    selected: set[str] = set()
    for value in values:
        code = str(value).strip().upper()
        if code in CSV_LEAGUES:
            code = CSV_LEAGUES[code]["code"]
        if code not in research_codes:
            allowed = ", ".join(sorted(research_codes))
            raise ValueError(f"unsupported research league {value!r}; expected one of: {allowed}")
        selected.add(code)
    return selected


def _validate_season_range(start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    start_year = MIN_SEASON if start is None else start
    end_year = MAX_SEASON if end is None else end
    if isinstance(start_year, bool) or not isinstance(start_year, int):
        raise ValueError("start must be a season start year")
    if isinstance(end_year, bool) or not isinstance(end_year, int):
        raise ValueError("end must be a season start year")
    if start_year < MIN_SEASON or start_year > MAX_SEASON:
        raise ValueError(f"start must be between {MIN_SEASON} and {MAX_SEASON}")
    if end_year < MIN_SEASON or end_year > MAX_SEASON:
        raise ValueError(f"end must be between {MIN_SEASON} and {MAX_SEASON}")
    if start_year > end_year:
        raise ValueError("start must not be after end")
    return start_year, end_year


def _parsed_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _canonical_team(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(normalized.casefold().split())


def _natural_key(match: Dict[str, Any]) -> Tuple[str, str, str, str]:
    match_datetime = _parsed_datetime(match.get("match_date"))
    match_day = match_datetime.date().isoformat() if match_datetime else ""
    return (
        str(match.get("league_code", "")),
        match_day,
        _canonical_team(match.get("home_team_name")),
        _canonical_team(match.get("away_team_name")),
    )


def _information_score(match: Dict[str, Any]) -> int:
    """Prefer the richer copy when duplicate natural keys are encountered."""

    score = sum(
        value not in (None, "")
        for key, value in match.items()
        if key != "extra_data"
    )
    score += sum(value not in (None, "") for value in match.get("extra_data", {}).values())
    return score


def _valid_price(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 1.0


def _complete_price_set(values: Iterable[Any]) -> bool:
    return all(_valid_price(value) for value in values)


def _has_any_complete(extra: Dict[str, Any], key_sets: Sequence[Tuple[str, ...]]) -> bool:
    return any(_complete_price_set(extra.get(key) for key in keys) for keys in key_sets)


def _coverage_flags(match: Dict[str, Any]) -> Dict[str, bool]:
    extra = match.get("extra_data", {})
    open_1x2 = _complete_price_set(
        (match.get("home_odds"), match.get("draw_odds"), match.get("away_odds"))
    ) or _has_any_complete(
        extra,
        (
            ("avg_home_odds", "avg_draw_odds", "avg_away_odds"),
            ("max_home_odds", "max_draw_odds", "max_away_odds"),
        ),
    )
    open_over_under = _has_any_complete(
        extra,
        (
            ("b365_over25", "b365_under25"),
            ("pinnacle_over25", "pinnacle_under25"),
            ("avg_over25", "avg_under25"),
            ("max_over25", "max_under25"),
        ),
    )
    market_asian_line = valid_asian_handicap_line(extra.get("asian_handicap_line"))
    b365_asian_line = valid_asian_handicap_line(extra.get("b365_asian_line"))
    if b365_asian_line is None:
        b365_asian_line = market_asian_line
    open_asian = (
        b365_asian_line is not None
        and _complete_price_set(
            (extra.get("b365_asian_home"), extra.get("b365_asian_away"))
        )
    ) or (
        market_asian_line is not None
        and _has_any_complete(
            extra,
            (
                ("pinnacle_asian_home", "pinnacle_asian_away"),
                ("avg_asian_home", "avg_asian_away"),
                ("max_asian_home", "max_asian_away"),
            ),
        )
    )
    closing_asian = valid_asian_handicap_line(extra.get("asian_handicap_close_line")) is not None and _has_any_complete(
        extra,
        (
            ("b365_close_asian_home", "b365_close_asian_away"),
            ("pinnacle_close_asian_home", "pinnacle_close_asian_away"),
            ("avg_close_asian_home", "avg_close_asian_away"),
            ("max_close_asian_home", "max_close_asian_away"),
        ),
    )
    closing_1x2 = _has_any_complete(
        extra,
        (
            ("b365_close_home", "b365_close_draw", "b365_close_away"),
            ("pinnacle_close_home", "pinnacle_close_draw", "pinnacle_close_away"),
            ("avg_close_home_odds", "avg_close_draw_odds", "avg_close_away_odds"),
            ("max_close_home_odds", "max_close_draw_odds", "max_close_away_odds"),
        ),
    )
    closing_over_under = _has_any_complete(
        extra,
        (
            ("b365_close_over25", "b365_close_under25"),
            ("pinnacle_close_over25", "pinnacle_close_under25"),
            ("avg_close_over25", "avg_close_under25"),
            ("max_close_over25", "max_close_under25"),
        ),
    )
    return {
        "1x2_open": open_1x2,
        "over_under_2_5_open": open_over_under,
        "asian_handicap_open": open_asian,
        "asian_handicap_closing": closing_asian,
        "1x2_closing": closing_1x2,
        "over_under_2_5_closing": closing_over_under,
    }


def _odds_coverage(matches: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, float | int]]:
    counts: Dict[str, int] = {}
    for match in matches:
        for market, covered in _coverage_flags(match).items():
            counts[market] = counts.get(market, 0) + int(covered)
    total = len(matches)
    return {
        market: {
            "rows": count,
            "rate": count / total if total else 0.0,
            "pct": (count / total * 100.0) if total else 0.0,
        }
        for market, count in counts.items()
    } | {
        market: {"rows": 0, "rate": 0.0, "pct": 0.0}
        for market in (
            "1x2_open",
            "over_under_2_5_open",
            "asian_handicap_open",
            "asian_handicap_closing",
            "1x2_closing",
            "over_under_2_5_closing",
        )
        if market not in counts
    }


def load_canonical_matches(
    *,
    leagues: Optional[Iterable[str]] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and deduplicate cached matches for inclusive season filters.

    ``start`` and ``end`` are season start years, so ``end=2025`` includes the
    complete 2025/26 file.  League filters use research codes; Football-Data
    file codes are also accepted as convenience aliases (``B1`` -> ``BEL1``).
    """

    selected_leagues = _selected_leagues(leagues)
    start_year, end_year = _validate_season_range(start, end)
    cache_dir = CANONICAL_CACHE_DIR.resolve()
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"canonical Football-Data cache not found: {cache_dir}")

    # Bypass the network-owning constructor.  The row normalizer and its safe
    # conversion helpers are stateless and do not require a requests session.
    parser = FootballDataCSVClient.__new__(FootballDataCSVClient)
    selected_files: List[Tuple[Path, int, str, Dict[str, str]]] = []
    ignored_files = 0
    for path in sorted(cache_dir.iterdir(), key=lambda candidate: candidate.name):
        if path.suffix.lower() != ".csv":
            continue
        match = _FILE_RE.fullmatch(path.name)
        if not match or path.is_symlink() or path.resolve().parent != cache_dir:
            ignored_files += 1
            continue
        season = decode_season_code(match.group("season"))
        league_info = CSV_LEAGUES.get(match.group("league"))
        if season is None or league_info is None:
            ignored_files += 1
            continue
        if not start_year <= season <= end_year or league_info["code"] not in selected_leagues:
            continue
        selected_files.append((path, season, match.group("league"), league_info))

    file_hashes: Dict[str, str] = {}
    dataset_digest = hashlib.sha256()
    for path, _season, _csv_league, _league_info in selected_files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        file_hash = digest.hexdigest()
        file_hashes[path.name] = file_hash
        dataset_digest.update(path.name.encode("utf-8"))
        dataset_digest.update(file_hash.encode("ascii"))

    raw_rows = 0
    normalized_rows = 0
    invalid_rows = 0
    duplicates = 0
    deduplicated: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for path, season, _csv_league, league_info in selected_files:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                raw_rows += 1
                normalized = parser._normalize_csv_row(
                    row,
                    league_info["code"],
                    league_info,
                    season,
                )
                if normalized is None:
                    invalid_rows += 1
                    continue
                normalized_rows += 1
                # The shared parser supplies the common normalized schema.
                # Preserve Pinnacle closing prices that are available in the
                # modern raw files but are not yet part of that shared schema.
                extra = normalized["extra_data"]
                extra.update(
                    {
                        "pinnacle_close_home": parser._safe_float(row.get("PSCH")),
                        "pinnacle_close_draw": parser._safe_float(row.get("PSCD")),
                        "pinnacle_close_away": parser._safe_float(row.get("PSCA")),
                        "pinnacle_close_over25": parser._safe_float(row.get("PC>2.5")),
                        "pinnacle_close_under25": parser._safe_float(row.get("PC<2.5")),
                    }
                )
                if (
                    _parsed_datetime(normalized.get("match_date")) is None
                    or normalized.get("home_score") is None
                    or normalized.get("away_score") is None
                ):
                    invalid_rows += 1
                    continue
                normalized["source_file"] = path.name
                natural_key = _natural_key(normalized)
                existing = deduplicated.get(natural_key)
                if existing is not None:
                    duplicates += 1
                    if _information_score(normalized) > _information_score(existing):
                        deduplicated[natural_key] = normalized
                else:
                    deduplicated[natural_key] = normalized

    matches = sorted(
        deduplicated.values(),
        key=lambda match: (
            match["match_date"],
            match["league_code"],
            _canonical_team(match["home_team_name"]),
            _canonical_team(match["away_team_name"]),
        ),
    )
    match_dates = [_parsed_datetime(match["match_date"]) for match in matches]
    valid_dates = [value for value in match_dates if value is not None]
    manifest: Dict[str, Any] = {
        "dataset_id": dataset_digest.hexdigest()[:20],
        "source": "data/cache/football_data_csv",
        "file_hashes": file_hashes,
        "files": len(selected_files),
        "ignored_files": ignored_files,
        "raw_rows": raw_rows,
        "normalized_rows": normalized_rows,
        "invalid_rows": invalid_rows,
        "duplicates": duplicates,
        "rows": len(matches),
        "start_date": min(valid_dates).date().isoformat() if valid_dates else None,
        "end_date": max(valid_dates).date().isoformat() if valid_dates else None,
        "start_season": start_year,
        "end_season": end_year,
        "leagues": sorted(selected_leagues),
        "odds_coverage": _odds_coverage(matches),
    }
    return matches, manifest


__all__ = [
    "CANONICAL_CACHE_DIR",
    "CSV_LEAGUES",
    "MAX_SEASON",
    "MIN_SEASON",
    "decode_season_code",
    "load_canonical_matches",
]
