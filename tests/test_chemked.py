"""ChemKED's pinned YAML fixture replays without network access."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from carmel.schemas.datasets import SourceRef, YamlPathLocator
from carmel.services.chemked import ChemkedRefusal, parse_idt_record, replay_idt_record
from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest, fetch_file
from carmel.services.respecth_archive import ArchiveIntegrityError

FIXTURE = Path(__file__).parent / "fixtures" / "chemked" / "Bec_2014_2-b_20atm.yaml"


def test_real_fixture_maps_thirteen_points_and_replays_every_value() -> None:
    raw = FIXTURE.read_bytes()
    record = parse_idt_record(raw, "2-butanol/Bec_2014_2-b_20atm.yaml", hashlib.sha256(raw).hexdigest())
    assert len(record.envelope.series[0].points) == 13
    first = record.envelope.series[0].points[0]
    assert first.coordinates[0].value.raw_text == "15.6"
    assert first.coordinates[1].value.raw_text == "828"
    assert first.observations[0].value.raw_text == "13797"
    replay_idt_record(record, raw)


def test_unknown_experiment_type_refuses_without_output() -> None:
    with pytest.raises(ChemkedRefusal, match="unknown_experiment_type"):
        parse_idt_record(FIXTURE.read_bytes().replace(b"ignition delay", b"flame speed", 1), "bad.yaml")


def test_schema_invalid_yaml_refuses_without_output() -> None:
    with pytest.raises(ChemkedRefusal, match="schema_rejected"):
        parse_idt_record(FIXTURE.read_bytes().replace(b"file-authors:", b"file-authors-missing:", 1), "bad.yaml")


def test_missing_yaml_path_refuses_replay() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    value = record.envelope.series[0].points[0].coordinates[0].value
    replacement = value.model_copy(
        update={"value_ref": SourceRef(node_id="record", locator=YamlPathLocator(path="gone"))}
    )
    first_point = record.envelope.series[0].points[0]
    point = first_point.model_copy(
        update={
            "coordinates": (
                first_point.coordinates[0].model_copy(update={"value": replacement}),
                first_point.coordinates[1],
            )
        }
    )
    series = record.envelope.series[0].model_copy(update={"points": (point,) + record.envelope.series[0].points[1:]})
    with pytest.raises(ChemkedRefusal, match="unresolvable_yaml_path"):
        replay_idt_record(
            replace(record, envelope=record.envelope.model_copy(update={"series": (series,)})), FIXTURE.read_bytes()
        )


def test_precompression_rcm_history_refuses() -> None:
    raw = FIXTURE.read_bytes().replace(b"kind: shock tube", b"kind: rapid compression machine", 1)
    raw = raw.replace(
        b"  - temperature:",
        b"  - volume-history:\n      values:\n        - [0.0, 1.0]\n        - [1.0, 0.2]\n    temperature:",
        1,
    )
    with pytest.raises(ChemkedRefusal, match="rcm_pre_compression_conditions"):
        parse_idt_record(raw, "precompression.yaml")


def test_cache_sha_mismatch_refuses(tmp_path: Path) -> None:
    item = ChemkedFile("fixture.yaml", "0" * 64)
    cache = tmp_path / "sha256" / item.sha256
    cache.parent.mkdir()
    cache.write_bytes(b"wrong")
    with pytest.raises(ArchiveIntegrityError):
        fetch_file(item, ChemkedManifest("example.invalid/x", "a" * 40, (item,)), tmp_path, download=False)
