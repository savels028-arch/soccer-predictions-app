"""Deterministic nested chronological splits for football research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NestedSeasonFold:
    test_season: int
    train_seasons: tuple[int, ...]
    validation_season: int
    calibration_end: str
    train_mask: np.ndarray
    calibration_mask: np.ndarray
    selection_mask: np.ndarray
    test_mask: np.ndarray


def _season_values(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["season"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("season must be present and numeric for every feature row")
    return values.astype(int)


def nested_season_folds(
    frame: pd.DataFrame,
    *,
    first_test_season: int = 2012,
    last_test_season: int | None = None,
    min_train_seasons: int = 5,
    min_calibration_rows: int = 100,
    min_selection_rows: int = 100,
) -> List[NestedSeasonFold]:
    """Create train/calibrate/select/test folds without temporal overlap.

    For outer season ``S`` the model trains through ``S-2``.  Season ``S-1``
    is split by unique kickoff days: its first half calibrates probabilities
    and its second half selects the betting policy.  ``S`` is untouched until
    that fixed policy is evaluated.
    """

    if frame.empty:
        return []
    if "match_date" not in frame or "season" not in frame:
        raise ValueError("frame must contain match_date and season")
    if min_train_seasons < 1:
        raise ValueError("min_train_seasons must be positive")

    seasons = _season_values(frame)
    dates = pd.to_datetime(frame["match_date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("match_date must be parseable for every feature row")
    available = sorted(set(seasons.tolist()))
    if not available:
        return []
    last = max(available) if last_test_season is None else int(last_test_season)
    folds: List[NestedSeasonFold] = []

    for test_season in available:
        if test_season < first_test_season or test_season > last:
            continue
        validation_season = test_season - 1
        train_seasons = tuple(season for season in available if season < validation_season)
        if len(train_seasons) < min_train_seasons or validation_season not in available:
            continue

        validation_positions = np.flatnonzero(seasons == validation_season)
        validation_days = np.array(
            sorted({dates.iloc[position].normalize().value for position in validation_positions}),
            dtype=np.int64,
        )
        if len(validation_days) < 2:
            continue
        split_at = len(validation_days) // 2
        calibration_days = set(validation_days[:split_at].tolist())
        selection_days = set(validation_days[split_at:].tolist())
        # ``astype('int64')`` follows pandas' backing resolution (often
        # microseconds in pandas 3), while ``Timestamp.value`` is always
        # nanoseconds.  Use the latter on both sides of the membership test.
        normalized_dates = dates.dt.normalize().map(lambda value: value.value).to_numpy(dtype=np.int64)

        train_mask = seasons < validation_season
        calibration_mask = (seasons == validation_season) & np.isin(normalized_dates, list(calibration_days))
        selection_mask = (seasons == validation_season) & np.isin(normalized_dates, list(selection_days))
        test_mask = seasons == test_season
        if calibration_mask.sum() < min_calibration_rows or selection_mask.sum() < min_selection_rows:
            continue
        if not test_mask.any():
            continue

        train_end = dates[train_mask].max()
        calibration_start = dates[calibration_mask].min()
        calibration_end_date = dates[calibration_mask].max()
        selection_start = dates[selection_mask].min()
        selection_end = dates[selection_mask].max()
        test_start = dates[test_mask].min()
        if not (
            train_end < calibration_start
            and calibration_end_date < selection_start
            and selection_end < test_start
        ):
            raise ValueError(
                f"season labels violate chronological train/calibrate/select/test order for {test_season}"
            )

        calibration_end = calibration_end_date.isoformat()
        folds.append(
            NestedSeasonFold(
                test_season=test_season,
                train_seasons=train_seasons,
                validation_season=validation_season,
                calibration_end=calibration_end,
                train_mask=train_mask,
                calibration_mask=calibration_mask,
                selection_mask=selection_mask,
                test_mask=test_mask,
            )
        )
    return folds


def positions(mask: Iterable[bool]) -> np.ndarray:
    """Return integer row positions for a boolean fold mask."""

    return np.flatnonzero(np.asarray(list(mask), dtype=bool))


__all__ = ["NestedSeasonFold", "nested_season_folds", "positions"]
