import csv

import pytest

from research import dataset
from src.api.csv_football_client import FootballDataCSVClient


def _write_csv(path, rows):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(date, home, away, *, home_score="1", away_score="0", **odds):
    return {
        "Date": date,
        "Time": "15:00",
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": home_score,
        "FTAG": away_score,
        "FTR": "H",
        **odds,
    }


@pytest.fixture
def canonical_cache(tmp_path, monkeypatch):
    cache = tmp_path / "data" / "cache" / "football_data_csv"
    cache.mkdir(parents=True)
    monkeypatch.setattr(dataset, "CANONICAL_CACHE_DIR", cache)
    return cache


def test_season_codes_decode_across_century_boundary():
    assert dataset.decode_season_code("9394") == 1993
    assert dataset.decode_season_code("9900") == 1999
    assert dataset.decode_season_code("0001") == 2000
    assert dataset.decode_season_code("2526") == 2025
    assert dataset.decode_season_code("9293") is None
    assert dataset.decode_season_code("2527") is None
    assert dataset.decode_season_code("2627") is None


def test_loader_deduplicates_natural_keys_and_reports_manifest(canonical_cache, monkeypatch):
    duplicate = _row(
        "12/08/00",
        "Alpha FC",
        "Beta FC",
        B365H="2.00",
        B365D="3.20",
        B365A="4.00",
        **{"BbAv>2.5": "1.90", "BbAv<2.5": "1.95"},
    )
    _write_csv(canonical_cache / "0001_E0.csv", [duplicate, duplicate])
    _write_csv(
        canonical_cache / "9394_E0.csv",
        [_row("14/08/93", "Old Alpha", "Old Beta", home_score="2", away_score="2")],
    )
    (canonical_cache / "not_canonical.csv").write_text("ignored", encoding="utf-8")

    def fail_if_networked(*_args, **_kwargs):
        raise AssertionError("canonical loader must not call downloader")

    monkeypatch.setattr(FootballDataCSVClient, "get_season_matches", fail_if_networked)
    matches, manifest = dataset.load_canonical_matches(leagues=["PL"])

    assert len(matches) == 2
    assert [match["season"] for match in matches] == [1993, 2000]
    assert manifest["source"] == "data/cache/football_data_csv"
    assert manifest["files"] == 2
    assert manifest["raw_rows"] == 3
    assert manifest["normalized_rows"] == 3
    assert manifest["duplicates"] == 1
    assert manifest["rows"] == 2
    assert manifest["start_date"] == "1993-08-14"
    assert manifest["end_date"] == "2000-08-12"
    assert manifest["odds_coverage"]["1x2_open"]["rows"] == 1
    assert manifest["odds_coverage"]["over_under_2_5_open"]["rows"] == 1
    assert manifest["odds_coverage"]["1x2_open"]["rate"] == 0.5


def test_belgian_file_maps_to_bel1_and_filters_are_inclusive(canonical_cache):
    _write_csv(
        canonical_cache / "0001_B1.csv",
        [_row("20/08/00", "Anderlecht", "Brugge", B365H="1.8", B365D="3.4", B365A="4.5")],
    )
    _write_csv(
        canonical_cache / "0102_B1.csv",
        [_row("19/08/01", "Brugge", "Anderlecht")],
    )
    _write_csv(canonical_cache / "0001_E0.csv", [_row("19/08/00", "Alpha", "Beta")])

    matches, manifest = dataset.load_canonical_matches(leagues=["B1"], start=2000, end=2000)

    assert len(matches) == 1
    assert matches[0]["league_code"] == "BEL1"
    assert matches[0]["league_code"] != "BSA"
    assert matches[0]["league_name"] == "Belgian First Division A"
    assert matches[0]["source_file"] == "0001_B1.csv"
    assert manifest["files"] == 1
    assert manifest["leagues"] == ["BEL1"]
    assert manifest["start_season"] == 2000
    assert manifest["end_season"] == 2000


def test_loader_rejects_non_research_leagues_and_invalid_ranges(canonical_cache):
    with pytest.raises(ValueError, match="unsupported research league"):
        dataset.load_canonical_matches(leagues=["BSA"])
    with pytest.raises(ValueError, match="must not be after"):
        dataset.load_canonical_matches(start=2020, end=2019)
