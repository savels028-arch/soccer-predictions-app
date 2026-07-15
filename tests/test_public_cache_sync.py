import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pipeline import PredictionPipeline, build_model_breakdown
from src.firestore_writer import FirestoreWriter
from src.firestore_writer import build_source_weights_payload
from src.public_cache_sync import (
    load_public_cache_contract,
    serialize_bulk_request,
    sync_public_cache,
)


EXPECTED_CACHE_IDS = {
    "matches",
    "ai_predictions",
    "ml_predictions",
    "model_breakdown",
    "source_weights",
    "coupon_history",
    "prediction_history",
    "paper_trading",
    "forecast_history",
    "strategy_zoo",
}


class _Response:
    def __init__(self, status=200):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self):
        return self.status


def _envelope(data):
    return {"data": data, "updatedAt": "2026-07-14T12:00:00Z"}


def test_public_cache_contract_has_only_active_producer_consumer_ids():
    contract = load_public_cache_contract()

    assert set(contract["documents"]) == EXPECTED_CACHE_IDS
    assert "showcase_results" not in contract["documents"]
    for cache_id in (
        "source_weights",
        "coupon_history",
        "prediction_history",
        "paper_trading",
        "forecast_history",
        "strategy_zoo",
    ):
        assert contract["documents"][cache_id]["maxAgeSeconds"] is None


def test_root_and_deploy_contracts_match_when_frontend_checkout_is_present():
    root = Path(__file__).resolve().parents[1]
    deploy_contract = root / "deploy" / "public-cache-contract.json"
    if not deploy_contract.exists():
        pytest.skip("deploy repository is not included in this checkout")

    assert json.loads(deploy_contract.read_text(encoding="utf-8")) == load_public_cache_contract()


def test_bulk_request_serializes_all_envelopes_and_utc_timestamps():
    caches = {
        "matches": _envelope([{"kickoff": datetime(2026, 7, 15, tzinfo=timezone.utc)}]),
        "prediction_history": _envelope({"results": []}),
    }

    payload = json.loads(serialize_bulk_request(caches))

    assert payload["contractVersion"] == 1
    assert set(payload["caches"]) == set(caches)
    assert payload["caches"]["matches"]["data"][0]["kickoff"] == "2026-07-15T00:00:00Z"


def test_bulk_request_rejects_unknown_cache_and_oversize_payload():
    with pytest.raises(RuntimeError, match="cache_contract_violation"):
        serialize_bulk_request({"private_data": _envelope({})})

    tiny_contract = {
        "version": 1,
        "maxRequestBytes": 20,
        "documents": {"matches": {}},
    }
    with pytest.raises(RuntimeError, match="request_too_large"):
        serialize_bulk_request({"matches": _envelope(["too large"])}, contract=tiny_contract)


def test_bulk_request_requires_every_contract_document_for_the_run_mode():
    with pytest.raises(RuntimeError, match="missing_required_cache_envelopes"):
        serialize_bulk_request({"prediction_history": _envelope({})}, mode="evaluate")

    evaluate_ids = {
        cache_id
        for cache_id, settings in load_public_cache_contract()["documents"].items()
        if "evaluate" in settings["requiredInModes"]
    }
    payload = json.loads(
        serialize_bulk_request(
            {cache_id: _envelope({}) for cache_id in evaluate_ids},
            mode="evaluate",
        )
    )
    assert set(payload["caches"]) == evaluate_ids

    with pytest.raises(RuntimeError, match="invalid_cache_mode"):
        serialize_bulk_request({"matches": _envelope({})}, mode="fulll")


def test_sync_uses_bearer_auth_and_retries_only_transient_failure(monkeypatch):
    requests = []
    responses = [
        HTTPError("https://aibets.dk", 503, "unavailable", {}, None),
        _Response(204),
    ]

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    sleeps = []
    monkeypatch.setattr("src.public_cache_sync.urlopen", fake_urlopen)
    result = sync_public_cache(
        {"matches": _envelope([])},
        secret="test-secret",
        attempts=3,
        sleep=sleeps.append,
    )

    assert result.synced is True
    assert result.attempts == 2
    assert sleeps == [0.5]
    assert len(requests) == 2
    assert requests[0][0].get_header("Authorization") == "Bearer test-secret"
    assert json.loads(requests[0][0].data)["caches"] == {"matches": _envelope([])}


def test_sync_missing_secret_and_auth_failure_are_non_throwing(monkeypatch):
    monkeypatch.delenv("AIBETS_CACHE_SYNC_SECRET", raising=False)
    monkeypatch.setattr(
        "src.public_cache_sync.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network must not be called without a secret")
        ),
    )
    missing = sync_public_cache({"matches": _envelope([])}, secret=None)
    assert missing.synced is False
    assert missing.reason == "missing_secret"
    assert missing.attempts == 0

    calls = []

    def unauthorized(*_args, **_kwargs):
        calls.append(True)
        raise HTTPError("https://aibets.dk", 401, "unauthorized", {}, None)

    monkeypatch.setattr("src.public_cache_sync.urlopen", unauthorized)
    denied = sync_public_cache({"matches": _envelope([])}, secret="wrong", attempts=3)
    assert denied.synced is False
    assert denied.reason == "http_401"
    assert denied.attempts == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://aibets.dk/api/internal/cache-sync",
        "https://attacker.example/api/internal/cache-sync",
        "https://aibets.dk:444/api/internal/cache-sync",
        "https://aibets.dk/api/internal/other",
        "https://user:pass@aibets.dk/api/internal/cache-sync",
        "https://aibets.dk/api/internal/cache-sync?redirect=attacker.example",
    ],
)
def test_sync_rejects_urls_that_could_exfiltrate_the_secret(monkeypatch, unsafe_url):
    monkeypatch.setattr(
        "src.public_cache_sync.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe URL must be rejected before any network call")
        ),
    )

    result = sync_public_cache(
        {"matches": _envelope([])},
        secret="test-secret",
        url=unsafe_url,
    )

    assert result.synced is False
    assert result.reason == "invalid_sync_url"
    assert result.attempts == 0


def test_source_weights_are_derived_from_real_evaluations_without_default_weights():
    payload = build_source_weights_payload(
        {
            "ml_xgboost": {
                "weight": 0.42,
                "brierScore": 0.51,
                "totalPredictions": 8,
                "roi": 0.9,
            },
        },
        [
            {"source": "ML Ensemble v1", "isCorrect": True},
            {"source": "ML Ensemble v1", "isCorrect": False},
            {"source": "ML Ensemble v2", "isCorrect": True},
        ],
    )

    assert payload["ml_xgboost"]["weight"] == 0.42
    assert payload["ml_xgboost"]["roi"] is None
    assert payload["ml_xgboost"]["roiBasis"] == "unavailable_no_verified_odds"
    assert payload["ML Ensemble"]["totalPredictions"] == 3
    assert payload["ML Ensemble"]["correct"] == 2
    assert payload["ML Ensemble"]["accuracy"] == pytest.approx(2 / 3, abs=0.0001)
    assert "weight" not in payload["ML Ensemble"]
    assert payload["ML Ensemble v1"]["accuracy"] == 0.5


def test_model_breakdown_uses_only_valid_current_model_probabilities():
    result = build_model_breakdown({
        "fixture-1": {
            "models": {
                "xgboost": {"home": 0.5, "draw": 0.3, "away": 0.2},
                "broken": {"home": 0.5},
            },
        },
    })

    assert result == {
        "fixture-1": [{
            "source": "ml_xgboost",
            "predicted_outcome": "HOME",
            "confidence": 50,
            "probabilities": {"home": 0.5, "draw": 0.3, "away": 0.2},
        }],
    }


def test_strategy_zoo_cache_revalidates_and_stages_the_public_artifact(monkeypatch):
    payload = {
        "schemaVersion": 2,
        "dataset": {"completeThroughSeason": 2024},
        "strategies": [{"id": "fixed-rule"}],
    }
    monkeypatch.setattr(
        "research.run_pattern_zoo.verify_artifact_against_canonical",
        lambda: payload,
    )
    writer = object.__new__(FirestoreWriter)
    writer._public_cache_envelopes = {}

    class FirestoreMustNotBeUsed:
        def collection(self, _name):
            raise AssertionError("strategy zoo must bypass the Firestore document limit")

    writer.db = FirestoreMustNotBeUsed()

    result = writer.refresh_strategy_zoo_cache()

    assert result is payload
    assert writer._public_cache_envelopes["strategy_zoo"]["data"] is payload
    assert isinstance(writer._public_cache_envelopes["strategy_zoo"]["updatedAt"], str)


def test_full_run_refreshes_performance_after_evaluation_then_syncs():
    pipeline = object.__new__(PredictionPipeline)
    events = []
    pipeline.engine = type("Engine", (), {"is_trained": True})()
    pipeline._should_retrain = lambda: False
    pipeline.fetch_matches = lambda: events.append("matches")
    pipeline.enrich_match_context = lambda: events.append("context")
    pipeline.fetch_odds = lambda: events.append("odds")
    pipeline.scrape_ai_predictions = lambda: events.append("scrape")
    pipeline.run_ml_predictions = lambda: events.append("predict")
    pipeline.compute_meta_features = lambda: events.append("features")
    pipeline.build_daily_coupon = lambda: events.append("coupon")
    pipeline.write_legacy_cache = lambda: events.append("early_cache")
    pipeline.evaluate_finished = lambda: events.append("evaluate")
    pipeline.update_source_performance = lambda: events.append("sources")
    pipeline.refresh_public_performance_caches = lambda: events.append("final_cache")
    pipeline.sync_public_cache = lambda **kwargs: events.append(f"sync:{kwargs['mode']}")
    pipeline._stats = {
        key: 0
        for key in (
            "matches_fetched",
            "odds_fetched",
            "ai_predictions",
            "ml_predictions",
            "results_saved",
            "coupons_evaluated",
            "sources_updated",
            "odds_snapshots",
            "pick_snapshots",
            "match_contexts",
        )
    }

    pipeline.run_full()

    assert events.index("early_cache") < events.index("evaluate")
    assert events[-4:] == ["evaluate", "sources", "final_cache", "sync:full"]


def test_evaluate_only_refreshes_after_source_update_then_syncs():
    pipeline = object.__new__(PredictionPipeline)
    events = []
    pipeline.fetch_matches = lambda: events.append("matches")
    pipeline.evaluate_finished = lambda: events.append("evaluate")
    pipeline.update_source_performance = lambda: events.append("sources")
    pipeline.refresh_public_performance_caches = lambda: events.append("final_cache")
    pipeline.sync_public_cache = lambda **kwargs: events.append(f"sync:{kwargs['mode']}")

    pipeline.run_evaluate_only()

    assert events == ["matches", "evaluate", "sources", "final_cache", "sync:evaluate"]
