"""Tests for fail-closed archive unpacking.

Every hostile archive is built PROGRAMMATICALLY here -- no malicious archive is committed
to the repository. The property under test is that each structural attack is REFUSED for
its own stated reason and leaves nothing on disk, and that a refusal is a recorded
outcome rather than a silent skip.
"""

from __future__ import annotations

import hashlib
import io
import stat
import struct
import zipfile
from pathlib import Path

from carmel.services.archive_unpack import (
    ArchiveUnpackRefusalReason,
    unpack_archive,
)


def _zip(*members: tuple[str, bytes]) -> bytes:
    """A zip whose members are the given ``(arcname, data)`` pairs, all deflated."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, data in members:
            archive.writestr(arcname, data)
    return buffer.getvalue()


def _zip_with_symlink(arcname: str, target: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo(arcname)
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, target)
    return buffer.getvalue()


def test_plain_member_is_written_and_content_addressed(tmp_path: Path) -> None:
    payload = b"t_ms,CH4\n0.0,0.21\n1.0,0.02\n"
    root = tmp_path / "extract"
    result = unpack_archive(_zip(("data/species.csv", payload)), root)

    assert result.refusals == ()
    assert len(result.members) == 1
    member = result.members[0]
    assert member.member_display_path == "data/species.csv"
    assert member.extracted_path.read_bytes() == payload
    assert member.extracted_path.is_relative_to(root.resolve())
    assert member.sha256 == hashlib.sha256(payload).hexdigest()
    assert member.size_bytes == len(payload)


def test_directory_entry_is_neither_a_member_nor_a_refusal(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("subdir/", b"")
        archive.writestr("subdir/a.csv", b"a,b\n1,2\n")
    result = unpack_archive(buffer.getvalue(), tmp_path / "extract")

    assert result.refusals == ()
    assert [m.member_display_path for m in result.members] == ["subdir/a.csv"]


def test_bytes_that_are_not_a_zip_are_refused_as_unreadable(tmp_path: Path) -> None:
    result = unpack_archive(b"this is not a zip archive", tmp_path / "extract")

    assert result.members == ()
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.UNREADABLE_ARCHIVE
    assert result.refusals[0].member_name == ""


def test_traversal_member_is_refused_and_nothing_escapes(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    result = unpack_archive(_zip(("../escape.csv", b"pwned")), root)

    assert result.members == ()
    assert len(result.refusals) == 1
    refusal = result.refusals[0]
    assert refusal.reason is ArchiveUnpackRefusalReason.PATH_ESCAPE
    assert refusal.member_name == "../escape.csv"
    assert not (tmp_path / "escape.csv").exists()


def test_deep_traversal_member_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "a" / "b" / "extract"
    result = unpack_archive(_zip(("../../../../../../tmp/evil.csv", b"x")), root)

    assert result.members == ()
    assert [r.reason for r in result.refusals] == [ArchiveUnpackRefusalReason.PATH_ESCAPE]


def test_absolute_posix_member_is_refused(tmp_path: Path) -> None:
    result = unpack_archive(_zip(("/abs.csv", b"x")), tmp_path / "extract")

    assert result.members == ()
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.ABSOLUTE_PATH
    assert result.refusals[0].member_name == "/abs.csv"
    assert not Path("/abs.csv").exists()


def test_absolute_windows_member_is_refused(tmp_path: Path) -> None:
    result = unpack_archive(_zip(("C:\\windows\\evil.csv", b"x")), tmp_path / "extract")

    assert result.members == ()
    assert [r.reason for r in result.refusals] == [ArchiveUnpackRefusalReason.ABSOLUTE_PATH]


def test_symlink_member_is_refused_and_never_materialised(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    result = unpack_archive(_zip_with_symlink("link.csv", "/etc/passwd"), root)

    assert result.members == ()
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.SYMLINK
    assert result.refusals[0].member_name == "link.csv"
    assert not (root / "link.csv").exists()
    assert not (root / "link.csv").is_symlink()


def test_member_count_is_bounded_before_any_write(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    result = unpack_archive(
        _zip(("a.csv", b"1"), ("b.csv", b"2"), ("c.csv", b"3")),
        root,
        max_member_count=2,
    )

    assert result.members == ()
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.MEMBER_COUNT_EXCEEDED
    assert result.refusals[0].member_name == ""
    # Nothing was written: the whole archive was refused before any member.
    assert not any(root.glob("*.csv"))


def test_oversized_member_is_refused_but_a_small_sibling_still_lands(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    result = unpack_archive(
        _zip(("big.csv", b"x" * 100), ("small.csv", b"ok")),
        root,
        max_member_bytes=10,
    )

    assert [m.member_display_path for m in result.members] == ["small.csv"]
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.MEMBER_SIZE_EXCEEDED
    assert result.refusals[0].member_name == "big.csv"
    assert not (root / "big.csv").exists()


def test_total_size_is_bounded_as_a_running_total(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    # First member (8 bytes) fits under a 10-byte total; the second would cross it.
    result = unpack_archive(
        _zip(("first.csv", b"12345678"), ("second.csv", b"12345")),
        root,
        max_total_bytes=10,
    )

    assert [m.member_display_path for m in result.members] == ["first.csv"]
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.TOTAL_SIZE_EXCEEDED
    assert result.refusals[0].member_name == "second.csv"
    assert not (root / "second.csv").exists()


def test_a_member_after_an_over_cap_one_is_still_admitted_if_it_fits(tmp_path: Path) -> None:
    """The cap is a per-member test against the RUNNING total, not a latch.

    Pinning this because the docstring on ``MAX_TOTAL_UNCOMPRESSED_BYTES`` used to
    claim the over-cap member "and every subsequent one" is refused, which the code
    has never done. The bound itself is what matters and it holds either way -- every
    member is tested against the running total before it is written -- so refusing the
    remainder would buy no safety and would discard a legitimate small member because
    of one oversized neighbour. If this test ever goes red, the docstring is the thing
    to re-read: the behaviour and the prose have to move together.
    """
    root = tmp_path / "extract"
    # 8 fits under 10; 5 would cross it and is refused; 2 still fits and must land.
    result = unpack_archive(
        _zip(("first.csv", b"12345678"), ("big.csv", b"12345"), ("last.csv", b"ab")),
        root,
        max_total_bytes=10,
    )

    assert [m.member_display_path for m in result.members] == ["first.csv", "last.csv"]
    assert [r.member_name for r in result.refusals] == ["big.csv"]
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.TOTAL_SIZE_EXCEEDED
    assert (root / "last.csv").exists()
    assert not (root / "big.csv").exists()


def _zip_with_understated_size(arcname: str, data: bytes, fake_size: int) -> bytes:
    """A zip whose central directory understates ``arcname``'s uncompressed size.

    The member's real (deflated) content is ``data``; only the central-directory
    ``uncompressed size`` field is rewritten to ``fake_size``, so it slips past the size
    gate while its bytes actually decompress to more. This is the lying-header attack.
    """
    raw = _zip((arcname, data))
    signature = b"PK\x01\x02"  # central directory file header
    index = raw.index(signature)
    # Uncompressed size is a 4-byte little-endian field at offset 24 of the record.
    offset = index + 24
    return raw[:offset] + struct.pack("<I", fake_size) + raw[offset + 4 :]


def test_understated_header_cannot_overrun_the_bound(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    # Real content is 5000 compressible bytes; the header claims 8, passing an 8-byte cap.
    archive = _zip_with_understated_size("bomb.csv", b"A" * 5000, fake_size=8)
    result = unpack_archive(archive, root, max_member_bytes=8)

    assert result.members == ()
    assert len(result.refusals) == 1
    assert result.refusals[0].reason is ArchiveUnpackRefusalReason.DECLARED_SIZE_MISMATCH
    assert result.refusals[0].member_name == "bomb.csv"
    assert not (root / "bomb.csv").exists()


def test_hostile_and_benign_members_are_reported_together(tmp_path: Path) -> None:
    root = tmp_path / "extract"
    result = unpack_archive(
        _zip(
            ("../escape.csv", b"x"),
            ("good.csv", b"a,b\n1,2\n"),
            ("/abs.csv", b"y"),
        ),
        root,
    )

    assert [m.member_display_path for m in result.members] == ["good.csv"]
    reasons = {r.reason for r in result.refusals}
    assert reasons == {
        ArchiveUnpackRefusalReason.PATH_ESCAPE,
        ArchiveUnpackRefusalReason.ABSOLUTE_PATH,
    }
