"""Fail-closed registry for research-only shadow strategy candidates.

This module deliberately has no dependency on the live prediction pipeline and
exposes no activation operation.  A historical research result can only be
recorded as a shadow challenger after the engine's locked, executable policy
passes its promotion gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple

from research.engine import DEFAULT_EXECUTABLE_BASES, _promotion_gate


SCHEMA_VERSION = 1
REGISTRY_MODE = "research_shadow_only"
_FIXED_DIRECT_FAMILIES = {
    "1x2": {
        "market",
        "poisson",
        "dixon_coles",
        "league_prior",
        "elo",
        "logistic_market",
        "boosting_market",
    },
    "ou25": {
        "market",
        "poisson",
        "dixon_coles",
        "league_prior",
        "logistic_market",
        "boosting_market",
    },
}
_FIXED_BLEND_FAMILIES = {
    market: {
        f"blend_{family}_market50"
        for family in (
            {"poisson", "dixon_coles", "logistic_market", "boosting_market"}
            | ({"elo"} if market == "1x2" else set())
        )
    }
    for market in ("1x2", "ou25")
}
_FIXED_EDGE_THRESHOLDS = {None, 0.0, 0.03, 0.06, 0.09}
_FIXED_CONFIDENCE_THRESHOLDS = {None, 0.55, 0.65}
_FIXED_ODDS_BANDS = {(1.20, 5.00), (1.20, 2.00), (1.50, 2.50), (1.80, 3.50)}


class RegistryValidationError(ValueError):
    """Raised when registry state or research evidence is not trustworthy."""


def _json_default(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(f"research evidence is not valid JSON: {exc}") from exc


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _empty_registry() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": REGISTRY_MODE,
        "automatic_live_activation": False,
        "updated_at": None,
        "shadow_challengers": {},
        "evaluations": [],
    }


def _locked_strategy_reasons(market: str, selected: Mapping[str, Any]) -> list[str]:
    reasons = []
    spec = _mapping(selected.get("spec"))
    if selected.get("eligible") is not True:
        reasons.append("locked_development_strategy_is_not_eligible")
    if int(selected.get("bets", 0) or 0) < 300:
        reasons.append("locked_development_has_fewer_than_300_bets")
    if int(selected.get("seasons", 0) or 0) < 5:
        reasons.append("locked_development_has_fewer_than_5_seasons")
    if float(selected.get("roi", 0.0) or 0.0) <= 0.0:
        reasons.append("locked_development_roi_is_not_positive")
    if float(selected.get("positive_season_rate", 0.0) or 0.0) < 0.55:
        reasons.append("locked_development_positive_season_rate_below_55pct")
    if spec.get("market") != market:
        reasons.append("locked_strategy_market_mismatch")
    if spec.get("odds_basis") not in DEFAULT_EXECUTABLE_BASES.get(market, ()):
        reasons.append("locked_strategy_does_not_use_executable_odds")
    allowed_sides = {
        "1x2": {"all", "no_draw", "home", "draw", "away"},
        "ou25": {"all", "under", "over"},
    }
    if spec.get("side") not in allowed_sides.get(market, set()):
        reasons.append("locked_strategy_has_invalid_side")
    family = spec.get("family")
    if not isinstance(family, str) or not family.strip():
        reasons.append("locked_strategy_has_invalid_family")
    else:
        parts = family.rsplit("__", 1)
        base = parts[0]
        calibration = parts[1] if len(parts) == 2 else ""
        allowed_bases = _FIXED_DIRECT_FAMILIES.get(market, set()) | _FIXED_BLEND_FAMILIES.get(
            market, set()
        )
        if base not in allowed_bases or calibration not in {"raw", "temperature", "isotonic"}:
            reasons.append("locked_strategy_has_invalid_family")
    try:
        min_odds = float(spec.get("min_odds"))
        max_odds = float(spec.get("max_odds"))
        if not (
            math.isfinite(min_odds)
            and math.isfinite(max_odds)
            and 1.0 < min_odds <= max_odds
        ):
            reasons.append("locked_strategy_has_invalid_odds_band")
    except (TypeError, ValueError):
        reasons.append("locked_strategy_has_invalid_odds_band")
    for field, allowed in (
        ("min_edge", _FIXED_EDGE_THRESHOLDS),
        ("min_confidence", _FIXED_CONFIDENCE_THRESHOLDS),
    ):
        value = spec.get(field)
        try:
            normalized = None if value is None else float(value)
        except (TypeError, ValueError):
            normalized = "invalid"
        if normalized not in allowed:
            reasons.append(f"locked_strategy_has_invalid_{field}")
    try:
        normalized_band = (float(spec.get("min_odds")), float(spec.get("max_odds")))
    except (TypeError, ValueError):
        normalized_band = None
    if normalized_band not in _FIXED_ODDS_BANDS:
        reasons.append("locked_strategy_is_not_in_fixed_odds_grid")
    return reasons


def _validate_registry(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise RegistryValidationError("registry root must be an object")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise RegistryValidationError("unsupported registry schema version")
    if value.get("mode") != REGISTRY_MODE:
        raise RegistryValidationError("registry is not shadow-only")
    if value.get("automatic_live_activation") is not False:
        raise RegistryValidationError("automatic live activation must remain disabled")
    if not isinstance(value.get("shadow_challengers"), dict):
        raise RegistryValidationError("shadow_challengers must be an object")
    if not isinstance(value.get("evaluations"), list):
        raise RegistryValidationError("evaluations must be an array")

    event_ids = set()
    for event in value["evaluations"]:
        if not isinstance(event, dict) or not isinstance(event.get("run_id"), str):
            raise RegistryValidationError("each evaluation must have a run_id")
        if event["run_id"] in event_ids:
            raise RegistryValidationError("registry contains duplicate run_id evaluations")
        if event.get("automatic_live_activation") is not False:
            raise RegistryValidationError("evaluation attempts automatic live activation")
        event_ids.add(event["run_id"])

    for market, market_candidates in value["shadow_challengers"].items():
        if not isinstance(market_candidates, dict):
            raise RegistryValidationError(f"shadow_challengers.{market} must be an object")
        for run_id, candidate in market_candidates.items():
            candidate = _mapping(candidate)
            if run_id not in event_ids or candidate.get("run_id") != run_id:
                raise RegistryValidationError("shadow candidate has no matching evaluation")
            if candidate.get("track") != "locked_executable" or candidate.get("mode") != "shadow_only":
                raise RegistryValidationError("registry contains a non-shadow executable candidate")
            summary = _mapping(candidate.get("outer_test_evidence"))
            recomputed_gate = _promotion_gate(summary)
            if recomputed_gate != candidate.get("promotion_gate") or not recomputed_gate["passed"]:
                raise RegistryValidationError("shadow candidate does not pass its recomputed gate")
            development = _mapping(candidate.get("development_evidence"))
            if _locked_strategy_reasons(str(market), development):
                raise RegistryValidationError("shadow candidate fails locked development eligibility")
            if candidate.get("strategy") != development.get("spec"):
                raise RegistryValidationError("shadow candidate strategy does not match its evidence")
    return value


def load_registry(path: Path | str) -> Dict[str, Any]:
    """Load a registry, returning a new empty shadow registry if absent."""

    registry_path = Path(path)
    if not registry_path.exists():
        return _empty_registry()
    try:
        with registry_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryValidationError(f"cannot load registry: {exc}") from exc
    return _validate_registry(value)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _market_names(result: Mapping[str, Any]) -> Tuple[str, ...]:
    configured = _mapping(result.get("config")).get("markets")
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        return tuple(str(market) for market in configured)
    summaries = result.get("summaries")
    return tuple(str(market) for market in summaries) if isinstance(summaries, Mapping) else ()


def _evidence_digest(result: Mapping[str, Any]) -> str:
    evidence = {
        "method": result.get("method"),
        "config": result.get("config"),
        "summaries": result.get("summaries"),
        "promotion_gates": result.get("promotion_gates"),
        "locked_strategies": result.get("locked_strategies"),
        "champion_candidate": result.get("champion_candidate"),
    }
    return hashlib.sha256(_canonical_json(evidence).encode("utf-8")).hexdigest()


def evaluate_shadow_candidates(
    result: Mapping[str, Any],
    *,
    run_id: str,
    dataset_id: str,
    evaluated_at: str | None = None,
    git_sha: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Validate a research result and return its audit event and candidates.

    Gates are recomputed from the locked executable summaries and must exactly
    agree with the engine output.  Any integrity mismatch rejects every
    candidate in the run.
    """

    if not run_id.strip() or not dataset_id.strip():
        raise RegistryValidationError("run_id and dataset_id are required")
    if result.get("method") != "nested_expanding_walk_forward_train_calibrate_select_test":
        raise RegistryValidationError("unsupported research method")
    timestamp = evaluated_at or datetime.now(timezone.utc).isoformat()

    markets = _market_names(result)
    if not markets:
        raise RegistryValidationError("research result contains no markets")

    summaries = _mapping(result.get("summaries"))
    reported_gates = _mapping(result.get("promotion_gates"))
    locked_strategies = _mapping(result.get("locked_strategies"))
    champion = _mapping(result.get("champion_candidate"))
    expected_promotable = []
    market_checks: Dict[str, Dict[str, Any]] = {}
    integrity_errors = []

    for market in markets:
        summary = _mapping(_mapping(summaries.get(market)).get("locked_executable"))
        reported = _mapping(_mapping(reported_gates.get(market)).get("locked_executable"))
        recomputed = _promotion_gate(summary)
        reported_normalized = {
            "passed": reported.get("passed"),
            "reasons": list(reported.get("reasons", []))
            if isinstance(reported.get("reasons", []), list)
            else reported.get("reasons"),
        }
        if reported_normalized != recomputed:
            integrity_errors.append(f"{market}:reported_gate_does_not_match_recomputed_gate")
        if recomputed["passed"]:
            expected_promotable.append(market)
        market_checks[market] = {
            "summary": summary,
            "gate": recomputed,
            "locked": _mapping(_mapping(locked_strategies.get(market)).get("executable")),
        }

    expected_status = "PROMOTABLE_TO_SHADOW" if expected_promotable else "NO_PROMOTION"
    champion_markets = champion.get("markets", [])
    if not isinstance(champion_markets, list):
        integrity_errors.append("champion_candidate.markets_is_not_an_array")
        champion_markets = []
    if champion.get("status") != expected_status:
        integrity_errors.append("champion_candidate.status_does_not_match_gates")
    if sorted(str(market) for market in champion_markets) != sorted(expected_promotable):
        integrity_errors.append("champion_candidate.markets_do_not_match_gates")

    candidates: Dict[str, Dict[str, Any]] = {}
    rejected: Dict[str, list[str]] = {}
    for market, check in market_checks.items():
        reasons = list(check["gate"]["reasons"])
        locked = check["locked"]
        selected = _mapping(locked.get("selected"))
        spec = _mapping(selected.get("spec"))
        if check["gate"]["passed"]:
            reasons.extend(_locked_strategy_reasons(market, selected))
        if integrity_errors:
            reasons.extend(integrity_errors)

        if not reasons and check["gate"]["passed"]:
            candidates[market] = {
                "run_id": run_id,
                "dataset_id": dataset_id,
                "registered_at": timestamp,
                "track": "locked_executable",
                "mode": "shadow_only",
                "strategy": _json_copy(spec),
                "development_evidence": _json_copy(selected),
                "outer_test_evidence": _json_copy(check["summary"]),
                "promotion_gate": _json_copy(check["gate"]),
                "git_sha": git_sha,
            }
        else:
            rejected[market] = list(dict.fromkeys(str(reason) for reason in reasons))

    event = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "evaluated_at": timestamp,
        "evidence_sha256": _evidence_digest(result),
        "status": "REGISTERED_TO_SHADOW" if candidates else "NO_PROMOTION",
        "registered_markets": sorted(candidates),
        "rejected_markets": rejected,
        "automatic_live_activation": False,
        "git_sha": git_sha,
    }
    return event, candidates


def register_research_result(
    path: Path | str,
    result: Mapping[str, Any],
    *,
    run_id: str,
    dataset_id: str,
    evaluated_at: str | None = None,
    git_sha: str | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Atomically audit a result and register only passing shadow challengers."""

    registry_path = Path(path)
    registry = load_registry(registry_path)
    event, candidates = evaluate_shadow_candidates(
        result,
        run_id=run_id,
        dataset_id=dataset_id,
        evaluated_at=evaluated_at,
        git_sha=git_sha,
    )

    existing = [entry for entry in registry["evaluations"] if entry.get("run_id") == run_id]
    if existing:
        if (
            len(existing) != 1
            or existing[0].get("dataset_id") != dataset_id
            or existing[0].get("evidence_sha256") != event["evidence_sha256"]
        ):
            raise RegistryValidationError("run_id already exists with different evidence")
        return registry, existing[0]

    mutable_candidates: MutableMapping[str, Any] = registry["shadow_challengers"]
    for market, candidate in candidates.items():
        market_candidates = mutable_candidates.setdefault(market, {})
        if not isinstance(market_candidates, dict):
            raise RegistryValidationError(f"shadow_challengers.{market} must be an object")
        market_candidates[run_id] = candidate

    registry["evaluations"].append(event)
    registry["updated_at"] = event["evaluated_at"]
    _validate_registry(registry)
    _atomic_write(registry_path, registry)
    return registry, event


__all__ = [
    "REGISTRY_MODE",
    "RegistryValidationError",
    "evaluate_shadow_candidates",
    "load_registry",
    "register_research_result",
]
