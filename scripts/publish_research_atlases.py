#!/usr/bin/env python3
"""Build guarded research atlases and publish verified static deploy copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.edge_atlas import (
    DEFAULT_OUTPUT_PATH as EDGE_OUTPUT_PATH,
    build_public_edge_atlas,
    write_edge_atlas,
)
from research.international_atlas import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as INTERNATIONAL_OUTPUT_PATH,
    DEFAULT_RESULTS_PATH,
    build_international_atlas,
    verify_public_results_snapshot,
    verify_public_world_cup_workbook,
    write_international_atlas,
)


DEFAULT_DEPLOY_PUBLIC = ROOT / "deploy" / "public"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _verified_artifact_bytes(path: Path) -> bytes:
    content = path.read_bytes()
    sidecar = path.with_name(f"{path.name}.sha256")
    try:
        declared = sidecar.read_text(encoding="ascii").split()[0]
    except (OSError, IndexError, UnicodeError) as exc:
        raise RuntimeError(f"missing or invalid checksum sidecar for {path.name}") from exc
    actual = hashlib.sha256(content).hexdigest()
    if declared != actual:
        raise RuntimeError(f"checksum mismatch for {path.name}")
    return content


def _copy_static_artifact(source: Path, destination: Path) -> Mapping[str, Any]:
    content = _verified_artifact_bytes(source)
    digest = hashlib.sha256(content).hexdigest()
    checksum = f"{digest}  {destination.name}\n".encode("ascii")
    _atomic_write(destination, content)
    _atomic_write(destination.with_name(f"{destination.name}.sha256"), checksum)
    if destination.read_bytes() != content:
        raise RuntimeError(f"static publication verification failed for {destination.name}")
    return {"path": str(destination), "bytes": len(content), "sha256": digest}


def publish(
    *,
    world_cup_xlsx: Path,
    results: Path = DEFAULT_RESULTS_PATH,
    manifest: Path = DEFAULT_MANIFEST_PATH,
    deploy_public: Path = DEFAULT_DEPLOY_PUBLIC,
) -> Mapping[str, Any]:
    """Build every guarded source before mutating either deploy asset."""

    edge_payload = build_public_edge_atlas()
    edge_receipt = write_edge_atlas(edge_payload, EDGE_OUTPUT_PATH)

    verify_public_results_snapshot(results, manifest)
    verify_public_world_cup_workbook(world_cup_xlsx)
    international_payload = build_international_atlas(
        results_path=results,
        world_cup_xlsx=world_cup_xlsx,
    )
    international_receipt = write_international_atlas(
        international_payload,
        INTERNATIONAL_OUTPUT_PATH,
    )

    # Read and validate both source pairs before the first deploy write.
    _verified_artifact_bytes(EDGE_OUTPUT_PATH)
    _verified_artifact_bytes(INTERNATIONAL_OUTPUT_PATH)
    deployed = [
        _copy_static_artifact(EDGE_OUTPUT_PATH, deploy_public / "edge-atlas.json"),
        _copy_static_artifact(
            INTERNATIONAL_OUTPUT_PATH,
            deploy_public / "international-atlas.json",
        ),
    ]
    return {
        "edge": edge_receipt,
        "international": international_receipt,
        "deployed": deployed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-cup-xlsx", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--deploy-public", type=Path, default=DEFAULT_DEPLOY_PUBLIC)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    receipt = publish(
        world_cup_xlsx=arguments.world_cup_xlsx,
        results=arguments.results,
        manifest=arguments.manifest,
        deploy_public=arguments.deploy_public,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
