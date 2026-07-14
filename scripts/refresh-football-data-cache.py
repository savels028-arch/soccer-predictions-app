#!/usr/bin/env python3
"""Atomically refresh cached Football-Data CSV files for one season.

Historical research runs are network-free.  This explicit command is the only
place that updates their raw input cache, and it refuses malformed or shrinking
downloads so a transient provider error cannot silently corrupt the dataset.
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import io
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "football_data_csv"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
LEAGUES = ("E0", "E1", "SP1", "D1", "D2", "I1", "F1", "N1", "P1", "B1")
REQUIRED_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def current_season_start(today: date | None = None) -> int:
    current = today or date.today()
    return current.year if current.month >= 8 else current.year - 1


def season_code(start_year: int) -> str:
    if not 1993 <= start_year <= 2030:
        raise ValueError("season must be a start year between 1993 and 2030")
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def parse_csv(payload: bytes) -> tuple[str, int, int]:
    text = payload.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text))
    columns = set(reader.fieldnames or ())
    missing = REQUIRED_COLUMNS.difference(columns)
    if missing:
        raise ValueError(f"download is not a valid Football-Data CSV; missing {sorted(missing)}")
    rows = list(reader)
    scored = sum(
        bool(str(row.get("FTHG") or "").strip())
        and bool(str(row.get("FTAG") or "").strip())
        for row in rows
    )
    if not rows or not scored:
        raise ValueError("download contains no scored fixtures")
    return text, len(rows), scored


def existing_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        _text, rows, _scored = parse_csv(path.read_bytes())
        return rows
    except (OSError, UnicodeError, ValueError):
        return 0


def download(url: str, *, attempts: int = 3) -> bytes:
    request = Request(url, headers={"User-Agent": "AIBets-research-cache/1.0"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}")
                return response.read()
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    tmp_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        default=current_season_start(),
        help="season start year (defaults to the current European season)",
    )
    parser.add_argument(
        "--allow-shrink",
        action="store_true",
        help="allow a remote file with fewer rows than the current cache",
    )
    args = parser.parse_args()

    code = season_code(args.season)
    validated = []
    for league in LEAGUES:
        path = CACHE_DIR / f"{code}_{league}.csv"
        url = f"{BASE_URL}/{code}/{league}.csv"
        payload = download(url)
        text, rows, scored = parse_csv(payload)
        previous_rows = existing_row_count(path)
        if rows < previous_rows and not args.allow_shrink:
            raise RuntimeError(
                f"refusing to shrink {path.name}: remote={rows}, cached={previous_rows}"
            )
        new_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        validated.append((path, text, rows, scored, new_hash, old_hash))
        print(f"{path.name}: validated, rows={rows}, scored={scored}, sha256={new_hash[:12]}")
        time.sleep(0.25)

    # Do not modify any file until every configured league has downloaded and
    # passed validation. A provider outage therefore leaves one coherent,
    # last-known-good season cache rather than a partially refreshed dataset.
    changed = 0
    for path, text, rows, scored, new_hash, old_hash in validated:
        if old_hash != new_hash:
            atomic_write(path, text)
            changed += 1
        status = "updated" if old_hash != new_hash else "unchanged"
        print(f"{path.name}: {status}, rows={rows}, scored={scored}, sha256={new_hash[:12]}")

    print(f"refreshed season {args.season}/{args.season + 1}: {changed}/{len(LEAGUES)} files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
