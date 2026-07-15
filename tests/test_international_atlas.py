from __future__ import annotations

from datetime import date, timedelta
import hashlib
from html import escape
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from research.international_atlas import (
    DEFAULT_OUTPUT_PATH,
    InternationalResult,
    PINNED_WORLD_CUP_WORKBOOK_SHA256,
    WORLD_CUP_WORKBOOK_SOURCE_URL,
    build_cli_parser,
    build_international_atlas,
    load_international_results,
    parse_world_cup_xlsx,
    verify_public_results_snapshot,
    verify_public_world_cup_workbook,
    write_international_atlas,
)
from src.predictions.international_model import (
    PINNED_SOURCE_COMMIT,
    PINNED_SOURCE_SHA256,
    PINNED_SOURCE_URL,
)


def _competition(payload, competition_id):
    return next(row for row in payload["competitions"] if row["id"] == competition_id)


def _edition(section, edition):
    return next(row for row in section["editions"] if row["edition"] == edition)


def _strategy(edition, strategy_id):
    return next(
        row for row in edition["strategies"] if row["strategyId"] == strategy_id
    )


def _column(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(reference: str, value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f'<c r="{reference}"><v>{value}</v></c>'
    return (
        f'<c r="{reference}" t="inlineStr"><is><t>'
        f"{escape(str(value))}</t></is></c>"
    )


def _write_world_cup_workbook(tmp_path: Path, rows, *, edition=2014) -> Path:
    headers = [
        "Competition",
        "Home",
        "Away",
        "Date",
        "HGFT",
        "AGFT",
        "HGET",
        "AGET",
        "HGP",
        "AGP",
        "Finished",
        "bet365-H",
        "bet365-D",
        "bet365-A",
        "Pinny-H",
        "Pinny-D",
        "Pinny-A",
        "Betfair_Exch-H",
        "Betfair_Exch-D",
        "Betfair_Exch-A",
        "H-Max",
        "D-Max",
        "A-Max",
    ]
    all_rows = [headers] + [[row.get(header) for header in headers] for row in rows]
    row_xml = []
    for row_number, values in enumerate(all_rows, start=1):
        cells = "".join(
            _xlsx_cell(f"{_column(column_number)}{row_number}", value)
            for column_number, value in enumerate(values, start=1)
        )
        row_xml.append(f'<row r="{row_number}">{cells}</row>')
    worksheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(row_xml)}</sheetData></worksheet>"
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets><sheet name="WorldCup{edition}" sheetId="1" r:id="rId1"/>'
        '<sheet name="WorldCup2026Qualifiers" sheetId="2" r:id="rId2"/>'
        "</sheets></workbook>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '<Relationship Id="rId2" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet2.xml"/>'
        "</Relationships>"
    )
    path = tmp_path / "world-cup.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData/></worksheet>",
        )
    return path


def test_checked_in_snapshot_is_split_by_tournament_and_edition():
    results = load_international_results()
    payload = build_international_atlas(results)
    world_cup = _competition(payload, "fifa_world_cup")
    euro = _competition(payload, "uefa_euro")

    assert len(results) == 32_387
    assert world_cup["finals"]["matches"] == 653
    assert world_cup["qualification"]["matches"] == 6_977
    assert euro["finals"]["matches"] == 323
    assert euro["qualification"]["matches"] == 2_114
    assert _edition(world_cup["finals"], 2026)["matches"] == 101
    assert _edition(euro["finals"], 2020)["calendarYears"] == [2021]
    assert _edition(world_cup["qualification"], 1994)["matches"] == 495
    assert world_cup["qualification"]["unassignedMatches"] == 0
    assert euro["qualification"]["unassignedMatches"] == 0


def test_descriptive_goal_map_is_complete_but_never_claims_roi():
    rows = [
        InternationalResult(
            match_date=date(2022, 11, 20),
            home_team="alpha",
            away_team="beta",
            home_score=2,
            away_score=1,
            tournament="FIFA World Cup",
            neutral=True,
        ),
        InternationalResult(
            match_date=date(2022, 11, 21),
            home_team="gamma",
            away_team="delta",
            home_score=0,
            away_score=0,
            tournament="FIFA World Cup",
            neutral=False,
        ),
        # Exact tournament matching prevents CONIFA from leaking into FIFA.
        InternationalResult(
            match_date=date(2022, 11, 22),
            home_team="other",
            away_team="other two",
            home_score=9,
            away_score=9,
            tournament="CONIFA World Football Cup",
            neutral=True,
        ),
    ]
    payload = build_international_atlas(rows)
    edition = _edition(_competition(payload, "fifa_world_cup")["finals"], 2022)

    assert edition["matches"] == 2
    assert edition["results"]["home"] == {"count": 1, "ratePct": 50.0}
    assert edition["results"]["draw"] == {"count": 1, "ratePct": 50.0}
    assert edition["neutral"] == {"count": 1, "ratePct": 50.0}
    for line in ("0.5", "1.5", "2.5", "3.5", "4.5", "5.5"):
        assert (
            edition["goals"]["over"][line]["count"]
            + edition["goals"]["under"][line]["count"]
            == 2
        )
    assert edition["btts"]["yes"] == {"count": 1, "ratePct": 50.0}
    assert all(row["roiPct"] is None for row in edition["strategyHitRates"])
    assert payload["worldCupOdds"]["evidenceStatus"] == "unavailable_no_verified_odds"
    assert payload["worldCupOdds"]["sourceUrl"] == WORLD_CUP_WORKBOOK_SOURCE_URL
    assert payload["worldCupOdds"]["sourceSha256"] is None
    assert payload["claims"][2]["claimId"] == "uefa_euro_roi"
    assert payload["claims"][2]["allowed"] is False


def test_world_cup_xlsx_uses_named_quote_preference_and_90_minute_scores(tmp_path):
    workbook = _write_world_cup_workbook(
        tmp_path,
        [
            {
                "Competition": "World Cup 2014",
                "Home": "Alpha",
                "Away": "Beta",
                "Date": 41800,
                "HGFT": 2,
                "AGFT": 0,
                "Finished": "90 minutes",
                "bet365-H": 2.0,
                "bet365-D": 3.0,
                "bet365-A": 4.0,
                "Pinny-H": 50.0,
                "Pinny-D": 50.0,
                "Pinny-A": 50.0,
                "Betfair_Exch-H": 60.0,
                "Betfair_Exch-D": 60.0,
                "Betfair_Exch-A": 60.0,
            },
            {
                "Competition": "World Cup 2014",
                "Home": "Gamma",
                "Away": "Delta",
                "Date": 41801,
                "HGFT": 1,
                "AGFT": 1,
                "HGET": 2,
                "AGET": 1,
                "Finished": "Extra time",
                "Pinny-H": 2.5,
                "Pinny-D": 3.2,
                "Pinny-A": 3.1,
            },
            {
                "Competition": "World Cup 2014",
                "Home": "Epsilon",
                "Away": "Zeta",
                "Date": 41802,
                "HGFT": 0,
                "AGFT": 0,
                "HGP": 4,
                "AGP": 5,
                "Finished": "Penalties",
                "Betfair_Exch-H": 1.5,
                "Betfair_Exch-D": 3.5,
                "Betfair_Exch-A": 8.0,
            },
        ],
    )

    rows = parse_world_cup_xlsx(workbook)
    assert [row["odds"]["source"] for row in rows] == [
        "bet365",
        "pinnacle",
        "betfair_exchange",
    ]
    assert rows[1]["extraTime"] is True
    assert rows[2]["penalties"] is True
    assert rows[1]["regulationHomeGoals"] == rows[1]["regulationAwayGoals"] == 1
    assert rows[2]["regulationHomeGoals"] == rows[2]["regulationAwayGoals"] == 0
    assert all(row["settlementScoreFields"] == ["HGFT", "AGFT"] for row in rows)

    payload = build_international_atlas([], world_cup_xlsx=workbook)
    edition = payload["worldCupOdds"]["editions"][0]
    assert edition["quoteSources"] == {
        "bet365": 1,
        "betfair_exchange": 1,
        "pinnacle": 1,
    }
    assert edition["extraTimeRows"] == 1
    assert edition["penaltyRows"] == 1
    assert edition["settlement"]["tournamentWinnerUsed"] is False
    assert payload["worldCupOdds"]["priceTiming"] == "pre-closing"
    assert payload["worldCupOdds"]["sourceUrl"] == WORLD_CUP_WORKBOOK_SOURCE_URL
    assert payload["worldCupOdds"]["sourceSha256"] == hashlib.sha256(
        workbook.read_bytes()
    ).hexdigest()
    assert payload["worldCupOdds"]["profitHaircutBasis"] == (
        "winning_gross_profit_only"
    )
    draw = _strategy(edition, "draw")
    assert draw["bets"] == 3
    assert draw["wins"] == 2
    assert draw["profitUnits"] == pytest.approx(3.653)
    assert draw["roiPct"] == pytest.approx(121.77)
    assert draw["confirmedEdge"] is False
    assert draw["profitClaimAllowed"] is False


def test_max_and_average_prices_are_never_an_execution_fallback(tmp_path):
    workbook = _write_world_cup_workbook(
        tmp_path,
        [
            {
                "Competition": "World Cup 2014",
                "Home": "Alpha",
                "Away": "Beta",
                "Date": 41800,
                "HGFT": 3,
                "AGFT": 0,
                "Finished": "90 minutes",
                "H-Max": 100.0,
                "D-Max": 100.0,
                "A-Max": 100.0,
            }
        ],
    )
    parsed = parse_world_cup_xlsx(workbook)
    assert parsed[0]["odds"] is None

    payload = build_international_atlas([], world_cup_xlsx=workbook)
    edition = payload["worldCupOdds"]["editions"][0]
    assert edition["quotedMatches"] == 0
    assert edition["unquotedMatches"] == 1
    assert all(row["bets"] == 0 for row in edition["strategies"])
    assert payload["worldCupOdds"]["maxOrAverageOddsFallback"] is False


def test_output_is_deterministic_and_corrupt_workbook_fails_closed(tmp_path):
    rows = [
        {
            "date": "2024-06-15",
            "home_team": "A",
            "away_team": "B",
            "home_score": 1,
            "away_score": 0,
            "tournament": "UEFA Euro",
            "neutral": True,
        }
    ]
    assert build_international_atlas(rows) == build_international_atlas(reversed(rows))

    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not an xlsx")
    with pytest.raises(ValueError, match="corrupt World Cup XLSX"):
        parse_world_cup_xlsx(broken)


def test_writer_is_canonical_atomic_and_records_sha256(tmp_path):
    output = tmp_path / "nested" / "international_atlas_public.json"
    payload = {"z": [2, 1], "a": "Mål", "nested": {"b": False, "a": None}}

    first = write_international_atlas(payload, output)
    first_bytes = output.read_bytes()
    second = write_international_atlas(payload, output)

    assert first_bytes == output.read_bytes()
    assert first == second
    assert first_bytes == (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(first_bytes).hexdigest()
    assert first["sha256"] == digest
    assert output.with_name(f"{output.name}.sha256").read_text() == (
        f"{digest}  {output.name}\n"
    )
    assert list(output.parent.glob("*.tmp")) == []


def test_cli_parser_supports_requested_export_command(tmp_path):
    output = tmp_path / "atlas.json"
    arguments = build_cli_parser().parse_args(
        [
            "--world-cup-xlsx",
            "/tmp/WorldCup2026.xlsx",
            "--output",
            str(output),
        ]
    )

    assert arguments.world_cup_xlsx == Path("/tmp/WorldCup2026.xlsx")
    assert arguments.output == output
    assert arguments.results.name == "results_1990_plus.csv.gz"
    assert arguments.manifest.name == "manifest.json"


def test_public_snapshot_guard_rejects_mismatched_manifest(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2024-01-01,A,B,1,0,Friendly,FALSE\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aibets.international_manifest.v1",
                "source": {
                    "commit": PINNED_SOURCE_COMMIT,
                    "sha256": PINNED_SOURCE_SHA256,
                    "url": PINNED_SOURCE_URL,
                },
                "snapshot": {
                    "path": results.name,
                    "sha256": "wrong",
                    "rows": 1,
                    "start": "2024-01-01",
                    "end": "2024-01-01",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="checksum"):
        verify_public_results_snapshot(results, manifest)


def test_public_snapshot_guard_rejects_self_consistent_but_unpinned_snapshot(tmp_path):
    results = tmp_path / "results.csv"
    results.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,neutral\n"
        "2024-01-01,A,B,1,0,Friendly,FALSE\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(results.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "aibets.international_manifest.v1",
                "source": {
                    "commit": PINNED_SOURCE_COMMIT,
                    "sha256": PINNED_SOURCE_SHA256,
                    "url": PINNED_SOURCE_URL,
                },
                "snapshot": {
                    "path": results.name,
                    "sha256": digest,
                    "rows": 1,
                    "start": "2024-01-01",
                    "end": "2024-01-01",
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="checksum"):
        verify_public_results_snapshot(results, manifest)


def test_public_world_cup_guard_is_pinned(tmp_path):
    workbook = tmp_path / "world-cup.xlsx"
    workbook.write_bytes(b"not the reviewed workbook")
    assert len(PINNED_WORLD_CUP_WORKBOOK_SHA256) == 64
    with pytest.raises(RuntimeError, match="reviewed public pin"):
        verify_public_world_cup_workbook(workbook)


def test_2026_workbook_contract_is_100_of_104_through_july_12(tmp_path):
    first_match = date(2026, 6, 11)
    rows = []
    for index in range(100):
        # The fixture mirrors the currently available workbook boundary while
        # keeping the test independent of a machine-local /tmp download.
        match_date = first_match + timedelta(days=min(index, 31))
        rows.append(
            {
                "Competition": "World Cup 2026",
                "Home": f"Home {index}",
                "Away": f"Away {index}",
                "Date": match_date.isoformat(),
                "HGFT": index % 3,
                "AGFT": (index + 1) % 3,
                "Finished": "90 minutes",
                "bet365-H": 2.0,
                "bet365-D": 3.2,
                "bet365-A": 4.1,
            }
        )
    workbook = _write_world_cup_workbook(tmp_path, rows, edition=2026)

    payload = build_international_atlas([], world_cup_xlsx=workbook)
    odds = payload["worldCupOdds"]
    edition = odds["editions"][0]

    assert odds["startDate"] == "2026-06-11"
    assert odds["endDate"] == "2026-07-12"
    assert edition["startDate"] == "2026-06-11"
    assert edition["endDate"] == "2026-07-12"
    assert edition["workbookMatches"] == 100
    assert edition["expectedMatches"] == 104
    assert edition["complete"] is False
