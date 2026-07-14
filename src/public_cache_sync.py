"""Authenticated bulk sync for the public Cloudflare cache.

The ML pipeline remains the system of record in Firestore.  This module only
publishes an explicit allow-list of already-built frontend cache envelopes to
Cloudflare KV through the protected Worker ingest endpoint.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_SYNC_URL = "https://aibets.dk/api/internal/cache-sync"
DEFAULT_TIMEOUT_SECONDS = 12.0
DEFAULT_ATTEMPTS = 3
DEFAULT_MAX_REQUEST_BYTES = 5 * 1024 * 1024
ALLOWED_SYNC_HOSTS = frozenset({"aibets.dk", "www.aibets.dk"})
CONTRACT_PATH = Path(__file__).resolve().parents[1] / "config" / "public_cache_contract.json"


class PublicCacheSyncError(RuntimeError):
    """A sanitized cache-sync error that never contains credentials."""


@dataclass(frozen=True)
class PublicCacheSyncResult:
    synced: bool
    cache_count: int
    byte_count: int
    attempts: int
    reason: str
    attempted_at: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_public_cache_contract(path: Path = CONTRACT_PATH) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    documents = contract.get("documents")
    if contract.get("version") != 1 or not isinstance(documents, dict) or not documents:
        raise ValueError("Invalid public-cache contract")
    if any(not isinstance(name, str) or not name for name in documents):
        raise ValueError("Invalid public-cache document id")
    max_bytes = contract.get("maxRequestBytes", DEFAULT_MAX_REQUEST_BYTES)
    if not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("Invalid public-cache request limit")
    return contract


def public_cache_document_ids(path: Path = CONTRACT_PATH) -> frozenset[str]:
    return frozenset(load_public_cache_contract(path)["documents"])


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    raise TypeError(f"Unsupported public-cache value type: {type(value).__name__}")


def serialize_bulk_request(
    caches: Mapping[str, Mapping[str, Any]],
    *,
    contract: Optional[Mapping[str, Any]] = None,
) -> bytes:
    resolved_contract = dict(contract or load_public_cache_contract())
    allowed = frozenset(resolved_contract["documents"])
    unknown = sorted(set(caches) - allowed)
    if unknown:
        raise PublicCacheSyncError("cache_contract_violation")
    if not caches:
        raise PublicCacheSyncError("no_cache_envelopes")

    for cache_id, envelope in caches.items():
        if not isinstance(envelope, Mapping):
            raise PublicCacheSyncError("invalid_cache_envelope")
        if set(envelope) != {"data", "updatedAt"}:
            raise PublicCacheSyncError("invalid_cache_envelope")
        if not isinstance(envelope.get("updatedAt"), str):
            raise PublicCacheSyncError("invalid_cache_timestamp")

    encoded = json.dumps(
        {"contractVersion": resolved_contract["version"], "caches": caches},
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    max_bytes = int(resolved_contract.get("maxRequestBytes", DEFAULT_MAX_REQUEST_BYTES))
    if len(encoded) > max_bytes:
        raise PublicCacheSyncError("request_too_large")
    return encoded


def sync_public_cache(
    caches: Mapping[str, Mapping[str, Any]],
    *,
    secret: Optional[str] = None,
    url: Optional[str] = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep=time.sleep,
) -> PublicCacheSyncResult:
    """POST all staged envelopes once, retrying only transient failures.

    The returned reason is intentionally a short code.  Neither exceptions nor
    response bodies are surfaced, so a credential cannot leak through logs.
    """

    attempted_at = utc_now_iso()
    resolved_secret = secret if secret is not None else os.environ.get(
        "AIBETS_CACHE_SYNC_SECRET", ""
    )
    if not resolved_secret:
        return PublicCacheSyncResult(
            False, len(caches), 0, 0, "missing_secret", attempted_at
        )

    resolved_url = url or os.environ.get("AIBETS_PUBLIC_CACHE_SYNC_URL") or os.environ.get(
        "AIBETS_CACHE_SYNC_URL", DEFAULT_SYNC_URL
    )
    parsed_url = urlsplit(resolved_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname not in ALLOWED_SYNC_HOSTS
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.port not in (None, 443)
        or parsed_url.path != "/api/internal/cache-sync"
        or parsed_url.query
        or parsed_url.fragment
    ):
        return PublicCacheSyncResult(
            False, len(caches), 0, 0, "invalid_sync_url", attempted_at
        )
    try:
        payload = serialize_bulk_request(caches)
    except PublicCacheSyncError as exc:
        return PublicCacheSyncResult(
            False, len(caches), 0, 0, str(exc), attempted_at
        )

    max_attempts = max(1, int(attempts))
    reason = "network_error"
    used_attempts = 0
    for attempt in range(1, max_attempts + 1):
        used_attempts = attempt
        request = Request(
            resolved_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {resolved_secret}",
                "Content-Type": "application/json",
                "User-Agent": "AIBets-local-pipeline/1",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                status = int(response.getcode())
            if 200 <= status < 300:
                return PublicCacheSyncResult(
                    True, len(caches), len(payload), attempt, "ok", attempted_at
                )
            reason = f"http_{status}"
            transient = status == 429 or status >= 500
        except HTTPError as exc:
            status = int(exc.code)
            reason = f"http_{status}"
            transient = status == 429 or status >= 500
        except (URLError, TimeoutError, OSError):
            reason = "network_error"
            transient = True

        if not transient or attempt >= max_attempts:
            break
        sleep(min(0.5 * (2 ** (attempt - 1)), 2.0))

    return PublicCacheSyncResult(
        False, len(caches), len(payload), used_attempts, reason, attempted_at
    )
