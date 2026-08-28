"""Fail-closed unpacking of a received supplementary archive.

Unpacking bytes that arrived from the internet is the most dangerous thing this
codebase does, so this module refuses rather than sanitises. A member is written
to disk ONLY once it has been shown to be a plain file whose resolved path stays
inside the extraction root and whose size stays within the caller's bounds. Every
other member is REFUSED, and a refusal is a recorded outcome -- an entry in
:attr:`ArchiveUnpackResult.refusals` naming what was refused and why -- never a
silent skip.

The four structural attacks this closes, each its own :class:`ArchiveUnpackRefusalReason`:

* a member whose resolved path escapes the extraction root (``PATH_ESCAPE``);
* a symlink member (``SYMLINK``) -- a link never becomes a file on disk here, so it
  cannot be dereferenced later to reach outside the root;
* an absolute-path member (``ABSOLUTE_PATH``), whether POSIX (``/etc/...``) or
  Windows (``C:\\...``, ``\\...``);
* a decompression bomb: member count, per-member size and total size are all bounded
  BEFORE any bytes are written (``MEMBER_COUNT_EXCEEDED``, ``MEMBER_SIZE_EXCEEDED``,
  ``TOTAL_SIZE_EXCEEDED``), and the write itself is capped so a header that UNDERSTATES
  a member's size cannot overrun the bound either (``DECLARED_SIZE_MISMATCH``).

This module does not read, ground, or admit anything as evidence; it only turns an
archive's bytes into files on disk, content-addressed by sha256, that a later pass
(:mod:`carmel.services.member_tables`) may read.
"""

from __future__ import annotations

import hashlib
import io
import ntpath
import stat
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

from carmel.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "MAX_MEMBER_COUNT",
    "MAX_MEMBER_UNCOMPRESSED_BYTES",
    "MAX_TOTAL_UNCOMPRESSED_BYTES",
    "ArchiveRefusal",
    "ArchiveUnpackRefusalReason",
    "ArchiveUnpackResult",
    "UnpackedMember",
    "unpack_archive",
]

#: The most members an archive may contain before the whole archive is refused,
#: written nothing. Bounded because member COUNT is itself an exhaustion vector: a
#: zip with millions of empty entries costs nothing to compress and would otherwise
#: open millions of file handles and inodes. Checked against the central directory
#: before a single member is written.
MAX_MEMBER_COUNT = 4096

#: The most uncompressed bytes any single member may declare (or actually produce)
#: before it is refused. A member over this is refused individually; the rest of the
#: archive is still processed.
MAX_MEMBER_UNCOMPRESSED_BYTES = 256 * 1024 * 1024

#: The most uncompressed bytes the whole archive may expand to across all members.
#: Enforced as a RUNNING total, checked per member BEFORE writing: a member whose
#: declared size would push the running total past this is refused individually
#: (``TOTAL_SIZE_EXCEEDED``) and the archive keeps processing, so a later member that
#: still fits under the cap is admitted. The bound itself is never crossed -- every
#: member is tested against it -- which is what stops a small archive expanding into
#: an unbounded one. Refusing the REST of the archive after the first over-cap member
#: would buy no extra safety (the running check already holds the bound) and would
#: discard legitimate small members because of one oversized neighbour.
MAX_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024

#: Bytes read per chunk while writing a member. Small enough that a lying header is
#: caught within one chunk of overrun, large enough not to dominate real files.
_CHUNK_BYTES = 1024 * 1024


class ArchiveUnpackRefusalReason(StrEnum):
    """Why a member (or a whole archive) was refused. Each value is a distinct,
    recorded fact -- collapsing them would hide which structural attack fired."""

    UNREADABLE_ARCHIVE = "unreadable_archive"
    """The bytes are not a readable zip archive at all. Archive-level: nothing was
    written and no member was inspected."""

    MEMBER_COUNT_EXCEEDED = "member_count_exceeded"
    """The archive's central directory lists more members than
    :data:`MAX_MEMBER_COUNT`. Archive-level, checked before any write, so the whole
    archive is refused and nothing is written."""

    ABSOLUTE_PATH = "absolute_path"
    """The member's stored name is an absolute path (POSIX ``/...`` or Windows
    ``C:\\...`` / ``\\...``). An absolute path names a destination of the archive's
    choosing, not one under the extraction root."""

    SYMLINK = "symlink"
    """The member is a symbolic link. Materialising it would place a link on disk
    that a later read could follow out of the root, so it is never written."""

    PATH_ESCAPE = "path_escape"
    """The member's name resolves to a path outside the extraction root (``..``
    traversal). Refused rather than clamped: a clamped path silently retargets the
    member, and a retarget is exactly what an attacker wants."""

    MEMBER_SIZE_EXCEEDED = "member_size_exceeded"
    """The member's declared uncompressed size exceeds
    :data:`MAX_MEMBER_UNCOMPRESSED_BYTES`. Checked before writing."""

    TOTAL_SIZE_EXCEEDED = "total_size_exceeded"
    """Admitting this member would push the running uncompressed total past
    :data:`MAX_TOTAL_UNCOMPRESSED_BYTES`. Checked before writing."""

    DECLARED_SIZE_MISMATCH = "declared_size_mismatch"
    """The member produced more bytes than its header declared -- a header that
    understates its size to slip past the size gate, or a member whose compressed
    bytes are corrupt. The partial write is discarded and the member refused, so a
    lying header cannot overrun the bound either."""


@dataclass(frozen=True)
class ArchiveRefusal:
    """One recorded refusal. ``member_name`` is the archive's own stored name for the
    member verbatim (display only, never trusted as a path), or ``""`` for an
    archive-level refusal that inspected no single member."""

    member_name: str
    reason: ArchiveUnpackRefusalReason
    detail: str


@dataclass(frozen=True)
class UnpackedMember:
    """One member that passed every check and was written to disk."""

    member_display_path: str
    """The name the archive stored this member under, verbatim. Display only -- it is
    never used to address the bytes; ``sha256`` is. Mirrors
    :attr:`carmel.schemas.datasets.ArchiveOrigin.member_display_path`."""

    extracted_path: Path
    """Where the member's bytes were written, always inside the extraction root."""

    sha256: str
    """Content address of the written bytes: 64 lowercase hex characters."""

    size_bytes: int
    """Number of bytes written."""


@dataclass(frozen=True)
class ArchiveUnpackResult:
    """Everything unpacking an archive established: the members written, and every
    refusal, in the order they were decided."""

    members: tuple[UnpackedMember, ...]
    refusals: tuple[ArchiveRefusal, ...]


def _is_absolute_member_name(name: str) -> bool:
    """Whether ``name`` is an absolute path under POSIX or Windows rules.

    Both are checked because a zip written on Windows stores ``\\``-separated and
    possibly drive-qualified names, and a POSIX extractor that only tested for a
    leading ``/`` would treat ``C:\\Windows\\...`` as relative and happily join it
    under the root on one platform while another honoured the drive.
    """
    return PurePosixPath(name).is_absolute() or ntpath.isabs(name)


def _resolves_within(root: Path, name: str) -> Path | None:
    """The absolute path ``name`` would extract to, if it stays within ``root``;
    ``None`` if it escapes or names ``root`` itself.

    ``root`` is a freshly created directory the caller owns, so ``resolve`` cannot be
    tricked by a pre-existing symlink inside it. Backslashes are folded to forward
    slashes first so a Windows-style ``..\\..\\`` traversal is seen as traversal
    rather than a single odd filename.
    """
    normalized = name.replace("\\", "/")
    candidate = (root / normalized).resolve()
    if candidate != root and candidate.is_relative_to(root):
        return candidate
    return None


def unpack_archive(
    archive_bytes: bytes,
    extraction_root: Path,
    *,
    max_member_count: int = MAX_MEMBER_COUNT,
    max_member_bytes: int = MAX_MEMBER_UNCOMPRESSED_BYTES,
    max_total_bytes: int = MAX_TOTAL_UNCOMPRESSED_BYTES,
) -> ArchiveUnpackResult:
    """Unpack ``archive_bytes`` into ``extraction_root``, fail-closed.

    Every member that survives the structural and size checks is written under
    ``extraction_root`` and returned in :attr:`ArchiveUnpackResult.members`; every
    member (or whole archive) that does not is refused and returned in
    :attr:`ArchiveUnpackResult.refusals`. Nothing is ever written outside the root,
    and no refusal is silent.

    ``extraction_root`` is created if absent. The caller owns it and is responsible
    for it being empty and disposable -- this routine never deletes it, only writes
    accepted members and discards its own partial writes.

    The bounds default to the module constants but are parameters so a caller (and a
    test) can drive tight limits without a giant fixture.
    """
    extraction_root.mkdir(parents=True, exist_ok=True)
    root = extraction_root.resolve()

    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        logger.warning("supplementary archive is not a readable zip: %s", exc)
        return ArchiveUnpackResult(
            members=(),
            refusals=(
                ArchiveRefusal(
                    member_name="",
                    reason=ArchiveUnpackRefusalReason.UNREADABLE_ARCHIVE,
                    detail=f"bytes are not a readable zip archive: {exc}",
                ),
            ),
        )

    members: list[UnpackedMember] = []
    refusals: list[ArchiveRefusal] = []

    with archive:
        infos = archive.infolist()
        if len(infos) > max_member_count:
            # Checked before any write: the whole archive is refused, so a
            # member-count bomb costs one central-directory read and nothing on disk.
            logger.warning("supplementary archive lists %d members, over the %d cap", len(infos), max_member_count)
            return ArchiveUnpackResult(
                members=(),
                refusals=(
                    ArchiveRefusal(
                        member_name="",
                        reason=ArchiveUnpackRefusalReason.MEMBER_COUNT_EXCEEDED,
                        detail=f"archive lists {len(infos)} members, over the {max_member_count} cap",
                    ),
                ),
            )

        total_written = 0
        for info in infos:
            name = info.filename
            if info.is_dir():
                # A directory entry carries no bytes to admit; the files that need it
                # create it via mkdir(parents=True). Not a member, not a refusal.
                continue

            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                refusals.append(
                    ArchiveRefusal(
                        member_name=name,
                        reason=ArchiveUnpackRefusalReason.SYMLINK,
                        detail=f"member {name!r} is a symlink; a link is never materialised on disk",
                    )
                )
                continue

            if _is_absolute_member_name(name):
                refusals.append(
                    ArchiveRefusal(
                        member_name=name,
                        reason=ArchiveUnpackRefusalReason.ABSOLUTE_PATH,
                        detail=(
                            f"member {name!r} is an absolute path; only paths under the extraction root are written"
                        ),
                    )
                )
                continue

            target = _resolves_within(root, name)
            if target is None:
                refusals.append(
                    ArchiveRefusal(
                        member_name=name,
                        reason=ArchiveUnpackRefusalReason.PATH_ESCAPE,
                        detail=f"member {name!r} resolves outside the extraction root",
                    )
                )
                continue

            declared = info.file_size
            if declared > max_member_bytes:
                refusals.append(
                    ArchiveRefusal(
                        member_name=name,
                        reason=ArchiveUnpackRefusalReason.MEMBER_SIZE_EXCEEDED,
                        detail=(
                            f"member {name!r} declares {declared} bytes, over the {max_member_bytes} per-member cap"
                        ),
                    )
                )
                continue
            if total_written + declared > max_total_bytes:
                refusals.append(
                    ArchiveRefusal(
                        member_name=name,
                        reason=ArchiveUnpackRefusalReason.TOTAL_SIZE_EXCEEDED,
                        detail=(
                            f"member {name!r} would push the uncompressed total to {total_written + declared} "
                            f"bytes, over the {max_total_bytes} cap"
                        ),
                    )
                )
                continue

            written = _write_member(archive, info, target, declared_size=declared)
            if written is None:
                # The header understated the size and the write overran it, or the
                # member's bytes were corrupt; the partial file has already been
                # removed. Refused, and the rest of the archive still processes.
                refusals.append(
                    ArchiveRefusal(
                        member_name=name,
                        reason=ArchiveUnpackRefusalReason.DECLARED_SIZE_MISMATCH,
                        detail=(
                            f"member {name!r} produced more than its declared {declared} bytes, or could not "
                            "be read; a header that understates its size cannot overrun the bound"
                        ),
                    )
                )
                continue

            digest, size = written
            total_written += size
            members.append(
                UnpackedMember(
                    member_display_path=name,
                    extracted_path=target,
                    sha256=digest,
                    size_bytes=size,
                )
            )

    return ArchiveUnpackResult(members=tuple(members), refusals=tuple(refusals))


def _write_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    target: Path,
    *,
    declared_size: int,
) -> tuple[str, int] | None:
    """Write one already-vetted member to ``target``, capped at ``declared_size``.

    Returns ``(sha256, size)`` on success, or ``None`` if the member produced more
    bytes than it declared (or could not be read) -- in which case the partial write
    is removed before returning, so a lying header leaves nothing on disk.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    # One byte over the declared size is enough to prove the header lied; reading that
    # far and no further keeps the write bounded even for a member that claims to be
    # empty.
    limit = declared_size + 1
    try:
        with archive.open(info) as source, target.open("wb") as sink:
            while size < limit:
                chunk = source.read(min(_CHUNK_BYTES, limit - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > declared_size:
                    sink.close()
                    target.unlink(missing_ok=True)
                    return None
                sink.write(chunk)
                hasher.update(chunk)
    except (OSError, zipfile.BadZipFile) as exc:
        # A member whose compressed bytes are corrupt raises mid-read. Treat it as an
        # overrun-style refusal rather than crashing the whole unpack: discard the
        # partial file, so the rest of the archive still processes.
        logger.warning("member %r could not be read from the archive: %s", info.filename, exc)
        target.unlink(missing_ok=True)
        return None
    return hasher.hexdigest(), size
