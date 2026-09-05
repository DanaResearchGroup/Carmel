"""Tests for the EmbeddedOoxmlTableInventory schema model (T1 self-coherence).

The model embeds an OOXML (``.docx``) inventory record verbatim and proves, holding no
document bytes, that the record is self-addressing, of a readable version, of the exact
record shape, and cell-addressable. It proves NOTHING about whether the grid is a real
document -- that is verify_ooxml_inventory_record's job -- so these tests exercise only
the self-coherence checks. Every ``.docx`` is built synthetically in-process; no corpus
document enters the repository.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from carmel.schemas.datasets import EmbeddedOoxmlTableInventory
from carmel.services.ooxml_table_record import (
    compute_ooxml_inventory_sha,
    ooxml_inventory_record_bytes,
    ooxml_inventory_record_payload,
    read_ooxml_table,
)
from tests.ooxml_fixtures import docx_bytes

_TABLE = [
    ["Equivalence ratio", "Laminar burning velocity (cm/s)"],
    ["0.40", "9.08"],
    ["0.45", "13.73"],
]


def _canonical() -> tuple[str, str, str]:
    """Return ``(canonical_json_str, inventory_sha256, source_sha256)`` for a valid record."""
    docx = docx_bytes([_TABLE])
    source_sha256 = hashlib.sha256(docx).hexdigest()
    inventory = read_ooxml_table(docx, table_index=0)
    payload = ooxml_inventory_record_payload(inventory, source_sha256=source_sha256)
    canonical = ooxml_inventory_record_bytes(payload).decode("utf-8")
    return canonical, compute_ooxml_inventory_sha(payload), source_sha256


def test_valid_record_constructs_and_is_cell_addressable() -> None:
    canonical, inventory_sha, source_sha = _canonical()
    model = EmbeddedOoxmlTableInventory(
        inventory_sha256=inventory_sha,
        source_sha256=source_sha,
        canonical_json=canonical,
    )
    assert model.has_cell(row=0, col=0)
    assert model.cell_text(row=1, col=1) == "9.08"
    assert not model.has_cell(row=99, col=99)
    assert model.cell_text(row=99, col=99) is None
    assert (1, 1, "9.08") in model.grid_cells()


def test_wrong_address_is_refused() -> None:
    canonical, _inventory_sha, source_sha = _canonical()
    with pytest.raises(ValidationError, match="does not live at the address it claims"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256="a" * 64,
            source_sha256=source_sha,
            canonical_json=canonical,
        )


def test_source_sha_disagreeing_with_payload_is_refused() -> None:
    canonical, inventory_sha, _source_sha = _canonical()
    with pytest.raises(ValidationError, match="the record names document"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=inventory_sha,
            source_sha256="b" * 64,
            canonical_json=canonical,
        )


def test_non_canonical_bytes_are_refused() -> None:
    canonical, _inventory_sha, source_sha = _canonical()
    payload = json.loads(canonical)
    reordered = {k: payload[k] for k in reversed(list(payload))}
    non_canonical = json.dumps(reordered, separators=(", ", ": "))
    with pytest.raises(ValidationError, match="canonical rendering"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=hashlib.sha256(non_canonical.encode()).hexdigest(),
            source_sha256=source_sha,
            canonical_json=non_canonical,
        )


def test_unknown_payload_version_is_refused() -> None:
    canonical_json, _inventory_sha, source_sha = _canonical()
    payload = json.loads(canonical_json)
    payload["payload_version"] = 999
    canonical = ooxml_inventory_record_bytes(payload).decode("utf-8")
    with pytest.raises(ValidationError, match="is not the readable version"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            source_sha256=source_sha,
            canonical_json=canonical,
        )


def test_extra_key_is_refused() -> None:
    canonical_json, _inventory_sha, source_sha = _canonical()
    payload = json.loads(canonical_json)
    payload["surprise"] = 1
    canonical = ooxml_inventory_record_bytes(payload).decode("utf-8")
    with pytest.raises(ValidationError, match="is not the shape of a version-"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            source_sha256=source_sha,
            canonical_json=canonical,
        )


def test_duplicate_cell_coordinate_is_refused() -> None:
    canonical_json, _inventory_sha, source_sha = _canonical()
    payload = json.loads(canonical_json)
    payload["cells"].append(dict(payload["cells"][0]))
    canonical = ooxml_inventory_record_bytes(payload).decode("utf-8")
    with pytest.raises(ValidationError, match="repeats the coordinate"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            source_sha256=source_sha,
            canonical_json=canonical,
        )


def test_boolean_ordinal_is_refused() -> None:
    canonical_json, _inventory_sha, source_sha = _canonical()
    payload = json.loads(canonical_json)
    payload["cells"][0]["row"] = True
    canonical = ooxml_inventory_record_bytes(payload).decode("utf-8")
    with pytest.raises(ValidationError, match="is not an integer ordinal"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
            source_sha256=source_sha,
            canonical_json=canonical,
        )


def test_malformed_sha_shapes_are_refused() -> None:
    canonical, inventory_sha, _source_sha = _canonical()
    with pytest.raises(ValidationError, match="is not 64 lowercase hex characters"):
        EmbeddedOoxmlTableInventory(
            inventory_sha256=inventory_sha,
            source_sha256="XYZ",
            canonical_json=canonical,
        )
