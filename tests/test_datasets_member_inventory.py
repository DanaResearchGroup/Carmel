"""Tests for the EmbeddedMemberTableInventory schema model (T1 self-coherence).

The model embeds a member inventory record verbatim and proves, holding no member bytes,
that the record is self-addressing, of a readable version, of the exact record shape, and
cell-addressable. It proves NOTHING about whether the grid is a real member -- that is
verify_member_inventory_record's job -- so these tests exercise only the self-coherence
checks.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import EmbeddedMemberTableInventory
from carmel.services.member_table_record import (
    compute_member_inventory_sha,
    member_inventory_record_bytes,
    member_inventory_record_payload,
    read_delimited_member,
)

_CSV = b"t_ms,CH4\n0.0,0.21\n0.5,0.10\n"


def _canonical(sheet_name: str = "species.csv") -> tuple[str, str, str]:
    """Return ``(canonical_json_str, inventory_sha256, member_sha256)`` for a valid record."""
    member_sha256 = hashlib.sha256(_CSV).hexdigest()
    inventory = read_delimited_member(_CSV, sheet_name=sheet_name)
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    canonical = member_inventory_record_bytes(payload).decode("utf-8")
    return canonical, compute_member_inventory_sha(payload), member_sha256


def test_valid_record_constructs_and_is_cell_addressable() -> None:
    canonical, inventory_sha, member_sha = _canonical()
    model = EmbeddedMemberTableInventory(
        inventory_sha256=inventory_sha,
        member_sha256=member_sha,
        canonical_json=canonical,
    )
    assert model.has_cell(row=0, col=0)
    assert model.cell_text(row=1, col=1) == "0.21"
    assert not model.has_cell(row=99, col=99)
    assert model.cell_text(row=99, col=99) is None


def test_wrong_address_is_refused() -> None:
    canonical, _inventory_sha, member_sha = _canonical()
    with pytest.raises(ValidationError, match="does not live at the address it claims"):
        EmbeddedMemberTableInventory(
            inventory_sha256="a" * 64,
            member_sha256=member_sha,
            canonical_json=canonical,
        )


def test_member_sha_disagreeing_with_payload_is_refused() -> None:
    canonical, inventory_sha, _member_sha = _canonical()
    with pytest.raises(ValidationError, match="the record names member"):
        EmbeddedMemberTableInventory(
            inventory_sha256=inventory_sha,
            member_sha256="b" * 64,
            canonical_json=canonical,
        )


def test_non_canonical_bytes_are_refused() -> None:
    canonical, inventory_sha, member_sha = _canonical()
    # Re-address a payload with a stray space so it is no longer canonical bytes.
    tampered = canonical.replace('"cells":', '"cells" :', 1)
    address = hashlib.sha256(tampered.encode("utf-8")).hexdigest()
    with pytest.raises(ValidationError, match="not the canonical rendering"):
        EmbeddedMemberTableInventory(
            inventory_sha256=address,
            member_sha256=member_sha,
            canonical_json=tampered,
        )


def test_unknown_payload_version_is_refused() -> None:
    member_sha256 = hashlib.sha256(_CSV).hexdigest()
    inventory = read_delimited_member(_CSV, sheet_name="species.csv")
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    payload["payload_version"] = 999
    canonical = member_inventory_record_bytes(payload).decode("utf-8")
    address = compute_member_inventory_sha(payload)
    with pytest.raises(ValidationError, match="not the readable version"):
        EmbeddedMemberTableInventory(
            inventory_sha256=address,
            member_sha256=member_sha256,
            canonical_json=canonical,
        )


def test_extra_key_is_refused() -> None:
    member_sha256 = hashlib.sha256(_CSV).hexdigest()
    inventory = read_delimited_member(_CSV, sheet_name="species.csv")
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    payload["stray"] = 1
    canonical = member_inventory_record_bytes(payload).decode("utf-8")
    address = compute_member_inventory_sha(payload)
    with pytest.raises(ValidationError, match="not the shape of a version"):
        EmbeddedMemberTableInventory(
            inventory_sha256=address,
            member_sha256=member_sha256,
            canonical_json=canonical,
        )


def test_duplicate_cell_coordinate_is_refused() -> None:
    member_sha256 = hashlib.sha256(_CSV).hexdigest()
    inventory = read_delimited_member(_CSV, sheet_name="species.csv")
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    payload["cells"].append({"col": 0, "row": 0, "text": "dupe"})
    canonical = member_inventory_record_bytes(payload).decode("utf-8")
    address = compute_member_inventory_sha(payload)
    with pytest.raises(ValidationError, match="repeats the coordinate"):
        EmbeddedMemberTableInventory(
            inventory_sha256=address,
            member_sha256=member_sha256,
            canonical_json=canonical,
        )


def test_boolean_ordinal_is_refused() -> None:
    member_sha256 = hashlib.sha256(_CSV).hexdigest()
    inventory = read_delimited_member(_CSV, sheet_name="species.csv")
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    payload["cells"] = [{"col": False, "row": True, "text": "x"}]
    canonical = member_inventory_record_bytes(payload).decode("utf-8")
    address = compute_member_inventory_sha(payload)
    with pytest.raises(ValidationError, match="not an integer ordinal"):
        EmbeddedMemberTableInventory(
            inventory_sha256=address,
            member_sha256=member_sha256,
            canonical_json=canonical,
        )


def test_empty_sheet_name_is_refused() -> None:
    member_sha256 = hashlib.sha256(_CSV).hexdigest()
    inventory = read_delimited_member(_CSV, sheet_name="species.csv")
    payload = member_inventory_record_payload(inventory, member_sha256=member_sha256)
    payload["sheet_name"] = ""
    canonical = member_inventory_record_bytes(payload).decode("utf-8")
    address = compute_member_inventory_sha(payload)
    with pytest.raises(ValidationError, match="names no sheet"):
        EmbeddedMemberTableInventory(
            inventory_sha256=address,
            member_sha256=member_sha256,
            canonical_json=canonical,
        )


def test_malformed_sha_shapes_are_refused() -> None:
    canonical, inventory_sha, member_sha = _canonical()
    with pytest.raises(ValidationError, match="not 64 lowercase hex"):
        EmbeddedMemberTableInventory(
            inventory_sha256="XYZ",
            member_sha256=member_sha,
            canonical_json=canonical,
        )
