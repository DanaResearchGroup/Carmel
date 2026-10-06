"""Pinned individual ChemKED YAML fetches using Carmel's content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from carmel.services.archive_unpack import _is_absolute_member_name
from carmel.services.respecth_archive import (
    ArchiveFetchError,
    ArchiveIntegrityError,
    ManifestError,
    cached_archive_path,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

# The largest file in the pinned commit is 488,490 bytes
# (methyl-pentanoate/phi=2.0/15-bar/Tc_772K_P0_0.5589_T0_373K_chemked.yaml).
# Four MiB leaves more than eight-fold headroom while bounding both cache and HTTP reads.
_MAX_FILE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class ChemkedFile:
    path: str
    sha256: str


@dataclass(frozen=True)
class ChemkedManifest:
    repository: str
    commit: str
    files: tuple[ChemkedFile, ...]

    def raw_url(self, item: ChemkedFile) -> str:
        encoded_path = quote(item.path, safe="/")
        return f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{encoded_path}"


def _require(mapping: dict[str, Any], key: str, kind: type, where: str) -> Any:
    value = mapping.get(key)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ManifestError(f"invalid ChemKED manifest: {where}: {key!r} must be a {kind.__name__}, got {value!r}")
    return value


def load_manifest(path: Path | None = None) -> ChemkedManifest:
    try:
        source = path if path is not None else resources.files("carmel.data").joinpath("chemked_manifest.json")
        with source.open("rb") as handle:
            raw = handle.read(_MAX_FILE_BYTES + 1)
        if len(raw) > _MAX_FILE_BYTES:
            raise ManifestError("ChemKED manifest exceeds 4 MiB")
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"invalid ChemKED manifest: cannot read it: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError("invalid ChemKED manifest: expected a JSON object")
    manifest_version = _require(data, "manifest_version", int, "manifest")
    license_name = _require(data, "license", str, "manifest")
    if manifest_version != 1 or license_name != "CC-BY-4.0":
        raise ManifestError("invalid ChemKED manifest: expected manifest_version 1 and license CC-BY-4.0")
    repository = _require(data, "repository", str, "manifest")
    commit = _require(data, "commit", str, "manifest")
    if not repository or not _COMMIT_RE.fullmatch(commit):
        raise ManifestError("invalid ChemKED manifest: repository must be non-empty and commit 40 lowercase hex")
    entries = data.get("files")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("invalid ChemKED manifest: files must be a non-empty list")
    files: list[ChemkedFile] = []
    for index, entry in enumerate(entries):
        where = f"files[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"invalid ChemKED manifest: {where} must be an object")
        item = ChemkedFile(
            path=_require(entry, "path", str, where),
            sha256=_require(entry, "sha256", str, where),
        )
        if (
            not item.path
            or "\\" in item.path
            or _is_absolute_member_name(item.path)
            or ".." in PurePosixPath(item.path).parts
        ):
            raise ManifestError(
                f"invalid ChemKED manifest: {where}.path must be a non-empty contained relative POSIX path "
                "(no backslash, absolute or drive name, or ..)"
            )
        if not _SHA256_RE.fullmatch(item.sha256):
            raise ManifestError(f"invalid ChemKED manifest: {where}.sha256 must be 64 lowercase-hex characters")
        files.append(item)
    if len({item.sha256 for item in files}) != len(files):
        raise ManifestError("invalid ChemKED manifest: the same sha256 is pinned more than once")
    return ChemkedManifest(repository=repository, commit=commit, files=tuple(files))


def fetch_file(item: ChemkedFile, manifest: ChemkedManifest, cache_root: Path, *, download: bool = True) -> bytes:
    target = cached_archive_path(cache_root, item.sha256)
    if target.is_file():
        size = target.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise ArchiveIntegrityError(
                f"cached ChemKED file {target} is {size} bytes, above the {_MAX_FILE_BYTES}-byte maximum; "
                "refused unread"
            )
        with target.open("rb") as handle:
            data = handle.read(_MAX_FILE_BYTES + 1)
    elif not download:
        raise ArchiveFetchError(f"{item.path!r} is not in the cache")
    else:
        try:
            with urllib.request.urlopen(manifest.raw_url(item), timeout=120) as response:  # noqa: S310 -- immutable pin
                data = response.read(_MAX_FILE_BYTES + 1)
        except (urllib.error.URLError, OSError) as exc:
            raise ArchiveFetchError(f"cannot fetch {item.path}: {exc}") from exc
        if len(data) > _MAX_FILE_BYTES:
            raise ArchiveIntegrityError(
                f"ChemKED file {item.path!r} exceeds the {_MAX_FILE_BYTES}-byte maximum; refused"
            )
        if hashlib.sha256(data).hexdigest() == item.sha256:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                handle.write(data)
                temporary = Path(handle.name)
            temporary.replace(target)
    if len(data) > _MAX_FILE_BYTES:  # defensive if a file grows between stat and read
        raise ArchiveIntegrityError(f"ChemKED file {item.path!r} exceeds the {_MAX_FILE_BYTES}-byte maximum; refused")
    if hashlib.sha256(data).hexdigest() != item.sha256:
        raise ArchiveIntegrityError(f"ChemKED file {item.path!r} does not match pinned sha256; refused")
    return data
