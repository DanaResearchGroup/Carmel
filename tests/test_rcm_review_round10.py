"""ChemKED manifest refusals name the invalid path or hash invariant."""

import json
from pathlib import Path

import pytest

from carmel.services.chemked_archive import ManifestError, load_manifest


def _manifest_file(tmp_path: Path, path: str, sha256: str) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "license": "CC-BY-4.0",
                "repository": "example/source",
                "commit": "a" * 40,
                "files": [{"path": path, "sha256": sha256}],
            }
        )
    )
    return manifest


@pytest.mark.parametrize(
    "path",
    [
        "",
        "../escape.yaml",
        "dir/../escape.yaml",
        "C:relative.yaml",
        "dir\\member.yaml",
        "/absolute.yaml",
        "C:/absolute.yaml",
        "//server/share/member.yaml",
    ],
)
def test_manifest_path_refusal_names_the_containment_invariant(tmp_path: Path, path: str) -> None:
    manifest = _manifest_file(tmp_path, path, "b" * 64)

    with pytest.raises(ManifestError) as refusal:
        load_manifest(manifest)

    assert str(refusal.value) == (
        "invalid ChemKED manifest: files[0].path must be a non-empty contained relative POSIX path "
        "(no backslash, absolute or drive name, or ..)"
    )


@pytest.mark.parametrize("sha256", ["", "A" * 64, "a" * 63, "g" * 64, "a" * 65, "../" + "a" * 61])
def test_manifest_sha256_refusal_names_the_hash_invariant(tmp_path: Path, sha256: str) -> None:
    manifest = _manifest_file(tmp_path, "contained/member.yaml", sha256)

    with pytest.raises(ManifestError) as refusal:
        load_manifest(manifest)

    assert str(refusal.value) == "invalid ChemKED manifest: files[0].sha256 must be 64 lowercase-hex characters"
