# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Pinned ReSpecTh archives: the committed manifest, a content-addressed cache, and
fail-closed member reads.

The ReSpecTh OSF mirror (https://osf.io/nbmzv/, DOI 10.17605/OSF.IO/NBMZV, CC BY 4.0)
publishes its ReSpecTh Kinetics Data (RKD) records only as one zip per fuel family, so the
ZIP is the unit Carmel pins: ``carmel/data/respecth_manifest.json`` names each archive by
OSF file id, OSF version and sha256, and nothing else is trusted. The database is never
vendored; an archive is downloaded on first use into a content-addressed cache
(:func:`default_cache_root`) and re-verified on every read.

Every failure is a typed :class:`RespecthError` subclass, never a partial result: a cached
file whose bytes no longer hash to its pin is refused (and left in place for the operator
to inspect) rather than silently re-downloaded, and a download that does not hash to its pin
is never written into the cache at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from io import BytesIO
from pathlib import Path
from typing import Any

__all__ = [
    "DATA_CACHE_ENV_VAR",
    "DEFAULT_DATA_CACHE_SUBPATH",
    "ArchiveFetchError",
    "ArchiveIntegrityError",
    "ManifestError",
    "PinnedArchive",
    "RespecthError",
    "RespecthManifest",
    "cached_archive_path",
    "default_cache_root",
    "fetch_archive",
    "iter_xml_members",
    "load_manifest",
    "read_member",
]

#: Default cache location, under the same per-user ``~/.carmel`` directory as the daily
#: ledger (see :mod:`carmel.paths`). Deliberately outside every campaign workspace: a pinned
#: archive is the same bytes for every campaign, so there is nothing to gain from a copy each.
DEFAULT_DATA_CACHE_SUBPATH: tuple[str, ...] = (".carmel", "data_cache")

#: Environment variable that overrides :func:`default_cache_root`.
DATA_CACHE_ENV_VAR = "CARMEL_DATA_CACHE"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OSF_FILE_ID_RE = re.compile(r"[0-9a-f]{24}")

#: Bounds on what a member read will inflate. The largest RKD member in either pinned archive
#: is ~16 KB; these leave two orders of magnitude of headroom while keeping a hostile or
#: corrupt archive (a zip bomb) from exhausting memory.
_MAX_MEMBER_BYTES = 4 * 1024 * 1024
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

_MANIFEST_PACKAGE = "carmel.data"
_MANIFEST_RESOURCE = "respecth_manifest.json"


class RespecthError(ValueError):
    """Base class for every error the ReSpecTh lane raises."""


class ManifestError(RespecthError):
    """The pinned manifest is missing, malformed, or pins something it cannot pin."""


class ArchiveIntegrityError(RespecthError):
    """Archive or member bytes do not match their pinned sha256 (or size, or zip shape)."""


class ArchiveFetchError(RespecthError):
    """An archive could not be downloaded at all (network, HTTP status, or timeout)."""


@dataclass(frozen=True)
class PinnedArchive:
    """One archive the manifest pins -- the unit Carmel downloads and verifies."""

    name: str
    osf_path: str
    osf_file_id: str
    osf_version: int
    size: int
    sha256: str


@dataclass(frozen=True)
class RespecthManifest:
    """The committed pin set: every archive the lane may read, and how to fetch it."""

    source: str
    osf_node: str
    doi: str
    license: str
    download_url_template: str
    archives: tuple[PinnedArchive, ...]

    def download_url(self, archive: PinnedArchive) -> str:
        """The OSF download URL for ``archive`` at its PINNED version (never "latest")."""
        return self.download_url_template.format(osf_file_id=archive.osf_file_id, osf_version=archive.osf_version)


def _require(mapping: dict[str, Any], key: str, kind: type, where: str) -> Any:
    value = mapping.get(key)
    # bool is an int subclass: a JSON `true` must never pass as a version or size.
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ManifestError(f"{where}: {key!r} must be a {kind.__name__}, got {value!r}")
    return value


def load_manifest(path: Path | None = None) -> RespecthManifest:
    """Load and validate the pinned manifest (the packaged one unless ``path`` is given).

    Raises:
        ManifestError: The file is unreadable, not JSON, or any pin is malformed -- a sha256
            that is not 64 lowercase hex, an OSF id of the wrong shape, a non-positive
            version or size, or a licence other than CC-BY-4.0.
    """
    try:
        raw = (
            path.read_bytes()
            if path is not None
            else resources.files(_MANIFEST_PACKAGE).joinpath(_MANIFEST_RESOURCE).read_bytes()
        )
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read the ReSpecTh manifest: {exc}") from exc
    if not isinstance(data, dict) or data.get("manifest_version") != 1:
        raise ManifestError("the ReSpecTh manifest must be a JSON object with manifest_version 1")
    if data.get("license") != "CC-BY-4.0":
        raise ManifestError(f"the ReSpecTh manifest pins licence {data.get('license')!r}; only CC-BY-4.0 is admitted")
    entries = data.get("archives")
    if not isinstance(entries, list) or not entries:
        raise ManifestError("the ReSpecTh manifest must pin at least one archive")
    archives: list[PinnedArchive] = []
    for index, entry in enumerate(entries):
        where = f"archives[{index}]"
        if not isinstance(entry, dict):
            raise ManifestError(f"{where} must be an object")
        archive = PinnedArchive(
            name=_require(entry, "name", str, where),
            osf_path=_require(entry, "osf_path", str, where),
            osf_file_id=_require(entry, "osf_file_id", str, where),
            osf_version=_require(entry, "osf_version", int, where),
            size=_require(entry, "size", int, where),
            sha256=_require(entry, "sha256", str, where),
        )
        if not _SHA256_RE.fullmatch(archive.sha256):
            raise ManifestError(f"{where}: sha256 {archive.sha256!r} is not 64 lowercase hex characters")
        if not _OSF_FILE_ID_RE.fullmatch(archive.osf_file_id):
            raise ManifestError(f"{where}: osf_file_id {archive.osf_file_id!r} is not an OSF file id")
        if archive.osf_version < 1 or archive.size < 1:
            raise ManifestError(f"{where}: osf_version and size must be positive")
        archives.append(archive)
    if len({archive.sha256 for archive in archives}) != len(archives):
        raise ManifestError("the ReSpecTh manifest pins the same sha256 twice")
    return RespecthManifest(
        source=_require(data, "source", str, "manifest"),
        osf_node=_require(data, "osf_node", str, "manifest"),
        doi=_require(data, "doi", str, "manifest"),
        license=data["license"],
        download_url_template=_require(data, "download_url_template", str, "manifest"),
        archives=tuple(archives),
    )


def default_cache_root() -> Path:
    """``$CARMEL_DATA_CACHE`` if set, else ``~/.carmel/data_cache``."""
    override = os.environ.get(DATA_CACHE_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return Path.home().joinpath(*DEFAULT_DATA_CACHE_SUBPATH)


def cached_archive_path(cache_root: Path, sha256: str) -> Path:
    """Where the archive with this content address lives in the cache."""
    return cache_root / "sha256" / sha256


def _verify(data: bytes, archive: PinnedArchive, *, what: str) -> None:
    actual = hashlib.sha256(data).hexdigest()
    if len(data) != archive.size or actual != archive.sha256:
        raise ArchiveIntegrityError(
            f"{what} for {archive.name!r} does not match its pin: expected {archive.size} bytes with sha256 "
            f"{archive.sha256}, got {len(data)} bytes with sha256 {actual}; refused"
        )


def fetch_archive(
    archive: PinnedArchive,
    *,
    manifest: RespecthManifest,
    cache_root: Path,
    download: bool = True,
    timeout: float = 120.0,
) -> bytes:
    """Return ``archive``'s verified bytes, from the cache or by downloading them once.

    A cached copy is size-checked before it is read, then read (bounded) and re-hashed on every
    call. A download is read to at most one byte past the pinned size, verified, and only then
    moved into the cache atomically, so a truncated or substituted transfer never lands there.

    Raises:
        ArchiveIntegrityError: The cached file, or the downloaded bytes, do not match the pin.
            A mismatched CACHED file is refused and left in place -- silently replacing it
            would hide whatever corrupted it.
        ArchiveFetchError: The archive is not cached and ``download`` is false, or the
            download itself failed.
    """
    path = cached_archive_path(cache_root, archive.sha256)
    if path.is_file():
        on_disk = path.stat().st_size
        if on_disk != archive.size:
            raise ArchiveIntegrityError(
                f"cached file {path} for {archive.name!r} does not match its pin: {on_disk} bytes on disk, "
                f"expected {archive.size}; refused unread"
            )
        with path.open("rb") as handle:
            data = handle.read(archive.size + 1)
        _verify(data, archive, what=f"cached file {path}")
        return data
    if not download:
        raise ArchiveFetchError(f"{archive.name!r} is not in the cache at {path} and downloading is disabled")
    url = manifest.download_url(archive)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - pinned https URL
            downloaded: bytes = response.read(archive.size + 1)
    except (urllib.error.URLError, OSError) as exc:
        raise ArchiveFetchError(f"downloading {archive.name!r} from {url} failed: {exc}") from exc
    _verify(downloaded, archive, what=f"download from {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(downloaded)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)
    return downloaded


def _open_zip(archive_bytes: bytes) -> zipfile.ZipFile:
    try:
        bundle = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ArchiveIntegrityError(f"archive is not a readable zip: {exc}") from exc
    if sum(info.file_size for info in bundle.infolist()) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ArchiveIntegrityError("archive declares more uncompressed bytes than the lane will inflate; refused")
    return bundle


def _read_info(bundle: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    if info.file_size > _MAX_MEMBER_BYTES:
        raise ArchiveIntegrityError(f"member {info.filename!r} declares {info.file_size} bytes; refused")
    try:
        with bundle.open(info) as handle:
            data = handle.read(_MAX_MEMBER_BYTES + 1)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArchiveIntegrityError(f"member {info.filename!r} cannot be read: {exc}") from exc
    # Unreachable while zipfile holds: ZipExtFile never yields past the header's file_size (it
    # counts it down), and a file_size over the cap is refused above. Kept against a zipfile change.
    if len(data) > _MAX_MEMBER_BYTES:  # pragma: no cover
        raise ArchiveIntegrityError(f"member {info.filename!r} inflates past its declared size; refused")
    return data


def iter_xml_members(archive_bytes: bytes) -> Iterator[tuple[str, bytes]]:
    """Yield ``(member_path, member_bytes)`` for every ``.xml`` member, in archive order."""
    bundle = _open_zip(archive_bytes)
    for info in bundle.infolist():
        if info.is_dir() or not info.filename.endswith(".xml"):
            continue
        yield info.filename, _read_info(bundle, info)


def read_member(archive_bytes: bytes, member_path: str, member_sha256: str) -> bytes:
    """Read one member and verify it hashes to ``member_sha256``.

    The path is only a lookup hint; the sha256 is the identity. A member whose bytes do not
    match is refused, whatever its name says.

    Raises:
        ArchiveIntegrityError: No such member, or its bytes do not hash to ``member_sha256``.
    """
    bundle = _open_zip(archive_bytes)
    try:
        info = bundle.getinfo(member_path)
    except KeyError:
        raise ArchiveIntegrityError(f"archive has no member {member_path!r}") from None
    data = _read_info(bundle, info)
    actual = hashlib.sha256(data).hexdigest()
    if actual != member_sha256:
        raise ArchiveIntegrityError(
            f"member {member_path!r} hashes to {actual}, not the pinned {member_sha256}; refused"
        )
    return data
