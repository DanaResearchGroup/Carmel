"""Pinned individual ChemKED YAML fetches using Carmel's content-addressed cache."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from carmel.services.respecth_archive import ArchiveFetchError, ArchiveIntegrityError, cached_archive_path


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
        return f"https://raw.githubusercontent.com/{self.repository}/{self.commit}/{item.path}"


def load_manifest(path: Path | None = None) -> ChemkedManifest:
    raw = path.read_bytes() if path else resources.files("carmel.data").joinpath("chemked_manifest.json").read_bytes()
    data = json.loads(raw)
    if data.get("manifest_version") != 1 or data.get("license") != "CC-BY-4.0":
        raise ValueError("invalid ChemKED manifest")
    files = tuple(ChemkedFile(**entry) for entry in data["files"])
    if not files or any(len(item.sha256) != 64 for item in files):
        raise ValueError("invalid ChemKED file pins")
    return ChemkedManifest(repository=data["repository"], commit=data["commit"], files=files)


def fetch_file(item: ChemkedFile, manifest: ChemkedManifest, cache_root: Path, *, download: bool = True) -> bytes:
    target = cached_archive_path(cache_root, item.sha256)
    if target.exists():
        data = target.read_bytes()
    elif not download:
        raise ArchiveFetchError(f"{item.path!r} is not in the cache")
    else:
        try:
            with urllib.request.urlopen(manifest.raw_url(item), timeout=120) as response:  # noqa: S310 -- immutable pin
                data = response.read()
        except (urllib.error.URLError, OSError) as exc:
            raise ArchiveFetchError(f"cannot fetch {item.path}: {exc}") from exc
        if hashlib.sha256(data).hexdigest() == item.sha256:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                handle.write(data)
                temporary = Path(handle.name)
            temporary.replace(target)
    if hashlib.sha256(data).hexdigest() != item.sha256:
        raise ArchiveIntegrityError(f"ChemKED file {item.path!r} does not match pinned sha256; refused")
    return data
