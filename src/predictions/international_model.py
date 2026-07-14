"""Fail-closed national-team forecasts for international fixtures.

This module is intentionally independent from the domestic club feature stack.
It contains a small Elo-style model, deterministic point-in-time updating, and
strict artifact/provenance validation.  The model produces probabilities only;
without historical pre-match odds it must not be treated as a betting strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MODEL_SCHEMA = "aibets.international_elo.v1"
MANIFEST_SCHEMA = "aibets.international_manifest.v1"
VALIDATED_STATUS = "VALIDATED_FORECAST_ONLY"
PINNED_SOURCE_COMMIT = "f73286079f8c6b48a59f8a16e895d757119dca71"
PINNED_SOURCE_SHA256 = "096184efc2d705b2acd6f5aebec3887a42019f24e2f8c11f60b76fa4b38a6a7c"
PINNED_SOURCE_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/"
    f"{PINNED_SOURCE_COMMIT}/results.csv"
)
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "international"
    / "manifest.json"
)


class InternationalModelUnavailable(RuntimeError):
    """Raised when a national-team forecast cannot be produced safely."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _ascii_key(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    ascii_value = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.casefold()).strip()


_TEAM_ALIASES = {
    "usa": "united states",
    "us": "united states",
    "u s a": "united states",
    "united states of america": "united states",
    "korea republic": "south korea",
    "republic of korea": "south korea",
    "korea south": "south korea",
    "ir iran": "iran",
    "iran islamic republic": "iran",
    "turkiye": "turkey",
    "cabo verde": "cape verde",
    "cote d ivoire": "ivory coast",
    "cote divoire": "ivory coast",
    "congo dr": "dr congo",
    "d r congo": "dr congo",
    "democratic republic of congo": "dr congo",
    "democratic republic of the congo": "dr congo",
    "czechia": "czech republic",
    "bosnia herzegovina": "bosnia and herzegovina",
    "curacao": "curacao",
}


def normalize_team_name(value: str) -> str:
    """Return a stable national-team key compatible with ESPN/OpenFootball."""
    key = _ascii_key(value)
    return _TEAM_ALIASES.get(key, key)


def _parse_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("missing_date")
    # Fixture timestamps are ISO-8601; the dataset is YYYY-MM-DD.
    return date.fromisoformat(raw[:10])


@dataclass(frozen=True)
class InternationalMatch:
    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    neutral: bool

    @property
    def outcome(self) -> str:
        if self.home_score > self.away_score:
            return "home"
        if self.home_score < self.away_score:
            return "away"
        return "draw"


@dataclass(frozen=True)
class EloParameters:
    k_factor: float = 30.0
    home_advantage: float = 70.0
    temperature: float = 1.0
    draw_base: float = 0.27
    draw_decay: float = 0.6
    min_team_matches: int = 8

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EloParameters":
        return cls(
            k_factor=float(value["k_factor"]),
            home_advantage=float(value["home_advantage"]),
            temperature=float(value["temperature"]),
            draw_base=float(value["draw_base"]),
            draw_decay=float(value["draw_decay"]),
            min_team_matches=int(value.get("min_team_matches", 8)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "temperature": self.temperature,
            "draw_base": self.draw_base,
            "draw_decay": self.draw_decay,
            "min_team_matches": self.min_team_matches,
        }


def tournament_importance(tournament: str) -> float:
    """Small, documented importance weighting inspired by FIFA's Elo SUM model."""
    name = str(tournament or "").casefold()
    if "fifa world cup" in name and "qualification" not in name:
        return 1.0
    if "qualification" in name or "qualifier" in name:
        return 0.75
    if any(
        token in name
        for token in (
            "uefa euro",
            "copa am",
            "african cup of nations",
            "afc asian cup",
            "gold cup",
            "nations league",
        )
    ):
        return 0.75
    if name == "friendly" or "friendly" in name:
        return 0.35
    return 0.5


def _elo_expectation(rating_difference: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / 400.0))


def probabilities_from_ratings(
    home_rating: float,
    away_rating: float,
    neutral: bool,
    params: EloParameters,
) -> Dict[str, float]:
    """Convert pre-match ratings into calibrated three-way probabilities."""
    advantage = 0.0 if neutral else params.home_advantage
    difference = home_rating + advantage - away_rating
    decisive_home = 1.0 / (
        1.0 + 10.0 ** (-difference / (400.0 * params.temperature))
    )
    draw = params.draw_base * math.exp(
        -params.draw_decay * abs(difference) / 400.0
    )
    draw = min(0.38, max(0.10, draw))
    home = (1.0 - draw) * decisive_home
    away = (1.0 - draw) * (1.0 - decisive_home)
    total = home + draw + away
    return {
        "home": home / total,
        "draw": draw / total,
        "away": away / total,
    }


class InternationalEloState:
    """Mutable national-team Elo state with same-date batch updates."""

    def __init__(
        self,
        params: EloParameters,
        ratings: Optional[Mapping[str, float]] = None,
        match_counts: Optional[Mapping[str, int]] = None,
    ):
        self.params = params
        self.ratings: Dict[str, float] = {
            normalize_team_name(k): float(v) for k, v in (ratings or {}).items()
        }
        self.match_counts: Dict[str, int] = {
            normalize_team_name(k): int(v) for k, v in (match_counts or {}).items()
        }

    def rating(self, team: str) -> float:
        return self.ratings.get(normalize_team_name(team), 1500.0)

    def predict(self, home_team: str, away_team: str, neutral: bool) -> Dict[str, float]:
        return probabilities_from_ratings(
            self.rating(home_team), self.rating(away_team), neutral, self.params
        )

    def update_batch(self, matches: Sequence[InternationalMatch]) -> None:
        """Update a kickoff/date batch from one frozen pre-match state.

        Accumulating deltas before applying them makes the result deterministic
        and prevents a match earlier in the input list leaking into another
        fixture with the same date.
        """
        deltas: Dict[str, float] = {}
        count_deltas: Dict[str, int] = {}
        frozen = dict(self.ratings)
        for match in matches:
            home = normalize_team_name(match.home_team)
            away = normalize_team_name(match.away_team)
            home_rating = frozen.get(home, 1500.0)
            away_rating = frozen.get(away, 1500.0)
            advantage = 0.0 if match.neutral else self.params.home_advantage
            expected = _elo_expectation(home_rating + advantage - away_rating)
            actual = 1.0 if match.home_score > match.away_score else (
                0.0 if match.home_score < match.away_score else 0.5
            )
            delta = (
                self.params.k_factor
                * tournament_importance(match.tournament)
                * (actual - expected)
            )
            deltas[home] = deltas.get(home, 0.0) + delta
            deltas[away] = deltas.get(away, 0.0) - delta
            count_deltas[home] = count_deltas.get(home, 0) + 1
            count_deltas[away] = count_deltas.get(away, 0) + 1

        for team, delta in deltas.items():
            self.ratings[team] = frozen.get(team, 1500.0) + delta
        for team, count in count_deltas.items():
            self.match_counts[team] = self.match_counts.get(team, 0) + count


def walk_forward_predictions(
    matches: Iterable[InternationalMatch],
    params: EloParameters,
    score_from: Optional[date] = None,
    score_through: Optional[date] = None,
) -> Tuple[List[Dict[str, Any]], InternationalEloState]:
    """Predict then update chronologically, batching all matches on one date."""
    ordered = sorted(
        matches,
        key=lambda m: (
            m.match_date,
            normalize_team_name(m.home_team),
            normalize_team_name(m.away_team),
            m.tournament,
        ),
    )
    state = InternationalEloState(params)
    predictions: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(ordered):
        batch_date = ordered[cursor].match_date
        end = cursor + 1
        while end < len(ordered) and ordered[end].match_date == batch_date:
            end += 1
        batch = ordered[cursor:end]
        should_score = (
            (score_from is None or batch_date >= score_from)
            and (score_through is None or batch_date <= score_through)
        )
        if should_score:
            for match in batch:
                predictions.append(
                    {
                        "match_date": match.match_date.isoformat(),
                        "home_team": normalize_team_name(match.home_team),
                        "away_team": normalize_team_name(match.away_team),
                        "tournament": match.tournament,
                        "outcome": match.outcome,
                        "probabilities": state.predict(
                            match.home_team, match.away_team, match.neutral
                        ),
                        "home_rating": state.rating(match.home_team),
                        "away_rating": state.rating(match.away_team),
                    }
                )
        state.update_batch(batch)
        cursor = end
    return predictions, state


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_child(parent: Path, relative: Any) -> Path:
    raw = str(relative or "")
    if not raw or Path(raw).is_absolute():
        raise InternationalModelUnavailable("invalid_artifact_path")
    result = (parent / raw).resolve()
    try:
        result.relative_to(parent.resolve())
    except ValueError as exc:
        raise InternationalModelUnavailable("invalid_artifact_path") from exc
    return result


class ValidatedInternationalModel:
    """Read-only inference wrapper loaded from a verified artifact bundle."""

    def __init__(self, artifact: Mapping[str, Any]):
        self.artifact = dict(artifact)
        self.params = EloParameters.from_mapping(self.artifact["parameters"])
        self.training_cutoff = _parse_date(self.artifact["training_cutoff"])
        self.state = InternationalEloState(
            self.params,
            ratings=self.artifact["ratings"],
            match_counts=self.artifact["match_counts"],
        )
        self.hosts = {
            normalize_team_name(team) for team in self.artifact.get("world_cup_hosts", [])
        }

    @property
    def validation(self) -> Mapping[str, Any]:
        return self.artifact["validation"]

    def predict_fixture(
        self,
        home_team: str,
        away_team: str,
        fixture_date: Any,
        neutral: Optional[bool] = None,
    ) -> Dict[str, Any]:
        match_date = _parse_date(fixture_date)
        if match_date <= self.training_cutoff:
            raise InternationalModelUnavailable("fixture_not_after_training_cutoff")

        home = normalize_team_name(home_team)
        away = normalize_team_name(away_team)
        if not home or not away or home == away:
            raise InternationalModelUnavailable("invalid_international_fixture")
        min_matches = self.params.min_team_matches
        if self.state.match_counts.get(home, 0) < min_matches:
            raise InternationalModelUnavailable("insufficient_home_team_history")
        if self.state.match_counts.get(away, 0) < min_matches:
            raise InternationalModelUnavailable("insufficient_away_team_history")

        # World Cup feeds do not always expose a neutral flag. A host listed as
        # the home side gets home advantage; all other fixtures default neutral.
        inferred_neutral = home not in self.hosts
        is_neutral = inferred_neutral if neutral is None else bool(neutral)
        probabilities = self.state.predict(home, away, is_neutral)
        outcome = max(("home", "draw", "away"), key=probabilities.__getitem__)
        return {
            "probabilities": probabilities,
            "forecast_outcome": outcome,
            "confidence": probabilities[outcome],
            "home_team_key": home,
            "away_team_key": away,
            "neutral": is_neutral,
            "model_version": self.artifact["model_version"],
            "training_cutoff": self.training_cutoff.isoformat(),
            "decision_scope": "forecast_only_no_historical_odds",
        }


def load_validated_international_model(
    manifest_path: Path | str = DEFAULT_MANIFEST,
) -> ValidatedInternationalModel:
    """Load a model only when manifest, snapshot, artifact and gates all verify."""
    manifest_file = Path(manifest_path).resolve()
    if not manifest_file.is_file():
        raise InternationalModelUnavailable("international_manifest_missing")
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InternationalModelUnavailable("international_manifest_invalid") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise InternationalModelUnavailable("international_manifest_schema_mismatch")
    source_meta = manifest.get("source") or {}
    if (
        source_meta.get("commit") != PINNED_SOURCE_COMMIT
        or source_meta.get("sha256") != PINNED_SOURCE_SHA256
        or source_meta.get("url") != PINNED_SOURCE_URL
    ):
        raise InternationalModelUnavailable("international_source_pin_mismatch")

    parent = manifest_file.parent
    snapshot_meta = manifest.get("snapshot") or {}
    artifact_meta = manifest.get("artifact") or {}
    snapshot_path = _validated_child(parent, snapshot_meta.get("path"))
    artifact_path = _validated_child(parent, artifact_meta.get("path"))
    if not snapshot_path.is_file() or not artifact_path.is_file():
        raise InternationalModelUnavailable("international_bundle_incomplete")
    if _sha256_file(snapshot_path) != snapshot_meta.get("sha256"):
        raise InternationalModelUnavailable("international_snapshot_checksum_mismatch")
    if _sha256_file(artifact_path) != artifact_meta.get("sha256"):
        raise InternationalModelUnavailable("international_artifact_checksum_mismatch")

    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InternationalModelUnavailable("international_artifact_invalid") from exc
    if artifact.get("schema") != MODEL_SCHEMA:
        raise InternationalModelUnavailable("international_model_schema_mismatch")
    if artifact.get("status") != VALIDATED_STATUS:
        raise InternationalModelUnavailable("international_model_not_validated")
    if artifact.get("normalized_snapshot_sha256") != snapshot_meta.get("sha256"):
        raise InternationalModelUnavailable("international_provenance_mismatch")
    if artifact.get("source_sha256") != source_meta.get("sha256"):
        raise InternationalModelUnavailable("international_source_mismatch")

    validation = artifact.get("validation") or {}
    gates = validation.get("gates") or {}
    required_gates = {
        "enough_holdout_matches",
        "enough_world_cup_matches",
        "beats_prior_accuracy",
        "beats_prior_brier",
        "beats_prior_log_loss",
        "calibration_within_limit",
        "world_cup_beats_prior_accuracy",
        "world_cup_beats_prior_brier",
        "world_cup_beats_prior_log_loss",
        "world_cup_calibration_within_limit",
        "point_in_time_batching",
    }
    if not required_gates.issubset(gates) or not all(gates[name] for name in required_gates):
        raise InternationalModelUnavailable("international_validation_gates_failed")

    cutoff = _parse_date(artifact.get("training_cutoff"))
    if cutoff > date.today():
        raise InternationalModelUnavailable("international_training_cutoff_in_future")
    if not artifact.get("ratings") or not artifact.get("match_counts"):
        raise InternationalModelUnavailable("international_model_state_missing")
    return ValidatedInternationalModel(artifact)


def try_load_default_international_model() -> Tuple[Optional[ValidatedInternationalModel], str]:
    try:
        return load_validated_international_model(DEFAULT_MANIFEST), ""
    except InternationalModelUnavailable as exc:
        return None, exc.reason
