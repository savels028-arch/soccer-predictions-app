from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.publish_research_atlases import _copy_static_artifact


def _write_source(path: Path, content: bytes, digest: str | None = None) -> None:
    path.write_bytes(content)
    checksum = digest or hashlib.sha256(content).hexdigest()
    path.with_name(f"{path.name}.sha256").write_text(
        f"{checksum}  {path.name}\n",
        encoding="ascii",
    )


def test_static_publication_is_byte_identical_and_rewrites_valid_sidecar(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "public" / "renamed.json"
    _write_source(source, b'{"verified":true}\n')

    receipt = _copy_static_artifact(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert destination.with_name("renamed.json.sha256").read_text(encoding="ascii") == (
        f"{receipt['sha256']}  renamed.json\n"
    )


def test_static_publication_rejects_bad_source_before_destination_write(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "public" / "atlas.json"
    _write_source(source, b"{}\n", digest="0" * 64)

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _copy_static_artifact(source, destination)

    assert not destination.exists()
