"""ChemKED's pinned YAML fixture replays without network access."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import Counter
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from Carmel import main
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    ComponentRole,
    Composition,
    DatasetEnvelope,
    SourceRef,
    UnitProvenance,
    XPathLocator,
    YamlPathLocator,
)
from carmel.services import chemked_archive, chemked_query, respecth_archive, respecth_query
from carmel.services.chemked import (
    ChemkedIdtRecord,
    ChemkedRefusal,
    ChemkedRefusalReason,
    parse_idt_record,
    replay_idt_record,
    yaml_value,
)
from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest, fetch_file, load_manifest
from carmel.services.chemked_query import find_idt, load_idt_records, point_conditions
from carmel.services.respecth import IgnitionCriterion, IgnitionTarget
from carmel.services.respecth import parse_idt_record as parse_respecth_idt_record
from carmel.services.respecth_archive import ArchiveFetchError, ArchiveIntegrityError, cached_archive_path
from carmel.services.respecth_query import ConditionWindow, LoadResult
from carmel.services.units import QuantityKind

FIXTURE = Path(__file__).parent / "fixtures" / "chemked" / "Bec_2014_2-b_20atm.yaml"


def _document() -> dict[str, object]:
    document = yaml.safe_load(FIXTURE.read_bytes())
    assert isinstance(document, dict)
    return document


def _yaml(document: object) -> bytes:
    return yaml.safe_dump(document, sort_keys=False).encode()


def _with_ignition(target: str, criterion: str) -> bytes:
    document = _document()
    common = document["common-properties"]
    assert isinstance(common, dict)
    ignition = common["ignition-type"]
    assert isinstance(ignition, dict)
    ignition.update(target=target, type=criterion)
    return _yaml(document)


def _with_first_pressure_value(record: ChemkedIdtRecord, **updates: object) -> ChemkedIdtRecord:
    series = record.envelope.series[0]
    point = series.points[0]
    coordinate = point.coordinates[0]
    value = coordinate.value.model_copy(update=updates)
    coordinates = (coordinate.model_copy(update={"value": value}),) + point.coordinates[1:]
    points = (point.model_copy(update={"coordinates": coordinates}),) + series.points[1:]
    envelope = record.envelope.model_copy(update={"series": (series.model_copy(update={"points": points}),)})
    return replace(record, envelope=envelope)


def _with_first_composition_component(record: ChemkedIdtRecord, **updates: object) -> ChemkedIdtRecord:
    series = record.envelope.series[0]
    point = series.points[0]
    assert isinstance(point.composition, Composition)
    component = point.composition.components[0].model_copy(update=updates)
    composition = point.composition.model_copy(update={"components": (component,) + point.composition.components[1:]})
    points = (point.model_copy(update={"composition": composition}),) + series.points[1:]
    envelope = record.envelope.model_copy(update={"series": (series.model_copy(update={"points": points}),)})
    return replace(record, envelope=envelope)


def _load_good_and_bad(tmp_path: Path, bad: bytes, bad_path: str) -> chemked_query.ChemkedLoadResult:
    good = FIXTURE.read_bytes()
    files = (
        ChemkedFile("good.yaml", hashlib.sha256(good).hexdigest()),
        ChemkedFile(bad_path, hashlib.sha256(bad).hexdigest()),
    )
    for item, data in zip(files, (good, bad), strict=True):
        target = cached_archive_path(tmp_path, item.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return load_idt_records(
        ChemkedManifest("example.invalid/source", "a" * 40, files), cache_root=tmp_path, download=False
    )


def test_real_fixture_maps_thirteen_points_and_replays_every_value() -> None:
    raw = FIXTURE.read_bytes()
    record = parse_idt_record(raw, "2-butanol/Bec_2014_2-b_20atm.yaml", hashlib.sha256(raw).hexdigest())
    assert len(record.envelope.series[0].points) == 13
    first = record.envelope.series[0].points[0]
    assert first.coordinates[0].value.raw_text == "15.6"
    assert first.coordinates[1].value.raw_text == "828"
    assert first.observations[0].value.raw_text == "13797"
    assert record.ignition.target is IgnitionTarget.OH_STAR
    assert record.ignition.target_raw.raw == "OH*"
    assert record.ignition.criterion is IgnitionCriterion.MAX_SLOPE
    assert record.ignition.criterion_raw.raw == "d/dt max"
    round_tripped = DatasetEnvelope.from_identity_payload(record.envelope.identity_payload())
    assert round_tripped.identity_payload() == record.envelope.identity_payload()
    replay_idt_record(record, raw)


def test_variable_composition_is_carried_per_point_and_replays() -> None:
    raw = FIXTURE.read_bytes()
    record = parse_idt_record(raw, "fixture.yaml")
    assert isinstance(record.envelope.composition, Absent)
    assert record.envelope.composition.reason is AbsenceReason.NOT_APPLICABLE
    first, second = record.envelope.series[0].points[:2]
    assert isinstance(first.composition, Composition)
    assert isinstance(second.composition, Composition)
    first_fuel = next(item for item in first.composition.components if item.species_raw_name == "2-butanol")
    second_fuel = next(item for item in second.composition.components if item.species_raw_name == "2-butanol")
    assert first_fuel.amount.canonical_decimal_value == "0.03"
    assert second_fuel.amount.canonical_decimal_value == "0.031"
    assert first_fuel.amount.value_ref.locator.path == "datapoints[0].composition.species[0].amount[0]"
    assert second_fuel.amount.value_ref.locator.path == "datapoints[1].composition.species[0].amount[0]"
    replay_idt_record(record, raw)


def test_stated_equivalence_ratios_are_grounded_per_row_and_replay() -> None:
    raw = FIXTURE.read_bytes()
    record = parse_idt_record(raw, "fixture.yaml")
    first, second = record.envelope.series[0].points[:2]
    assert isinstance(first.composition, Composition)
    assert isinstance(second.composition, Composition)
    first_ratio = first.composition.equivalence_ratio
    second_ratio = second.composition.equivalence_ratio
    assert not isinstance(first_ratio, Absent)
    assert not isinstance(second_ratio, Absent)
    assert (first_ratio.canonical_decimal_value, second_ratio.canonical_decimal_value) == ("1.54", "1.68")
    assert first_ratio.quantity_kind is QuantityKind.EQUIVALENCE_RATIO
    assert first_ratio.unit_provenance is UnitProvenance.NOT_PRINTED_IN_SOURCE
    assert isinstance(first_ratio.unit_raw, Absent)
    assert isinstance(first_ratio.unit_ref, Absent)
    assert first_ratio.value_ref.locator.path == "datapoints[0].equivalence-ratio"
    assert second_ratio.value_ref.locator.path == "datapoints[1].equivalence-ratio"
    replay_idt_record(record, raw)

    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    row = rows[0]
    del row["equivalence-ratio"]
    document["datapoints"] = [row]
    no_ratio_raw = _yaml(document)
    no_ratio_record = parse_idt_record(no_ratio_raw, "no-ratio.yaml")
    assert isinstance(no_ratio_record.envelope.composition, Composition)
    assert isinstance(no_ratio_record.envelope.composition.equivalence_ratio, Absent)
    replay_idt_record(no_ratio_record, no_ratio_raw)


def test_equivalence_ratio_participates_in_composition_inheritance() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    first_composition = rows[0]["composition"]
    for row in rows[1:]:
        assert isinstance(row, dict)
        row["composition"] = first_composition
    record = parse_idt_record(_yaml(document), "varying-ratio.yaml")
    assert isinstance(record.envelope.composition, Absent)
    assert all(isinstance(point.composition, Composition) for point in record.envelope.series[0].points)


def test_tampered_equivalence_ratio_fails_replay_and_names_the_field() -> None:
    raw = FIXTURE.read_bytes()
    record = parse_idt_record(raw, "fixture.yaml")
    series = record.envelope.series[0]
    first = series.points[0]
    assert isinstance(first.composition, Composition)
    ratio = first.composition.equivalence_ratio
    assert not isinstance(ratio, Absent)
    composition = first.composition.model_copy(
        update={"equivalence_ratio": ratio.model_copy(update={"canonical_decimal_value": "9"})}
    )
    points = (first.model_copy(update={"composition": composition}),) + series.points[1:]
    envelope = record.envelope.model_copy(update={"series": (series.model_copy(update={"points": points}),)})
    with pytest.raises(ChemkedRefusal, match=r"composition\.equivalence_ratio\.canonical_decimal_value"):
        replay_idt_record(replace(record, envelope=envelope), raw)


@pytest.mark.parametrize("malformed", ["not-a-number", float("nan")])
def test_malformed_equivalence_ratio_is_a_typed_refusal_counted_by_loader(tmp_path: Path, malformed: object) -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["equivalence-ratio"] = malformed
    loaded = _load_good_and_bad(tmp_path, _yaml(document), "bad-ratio.yaml")
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"schema_rejected": 1}


def test_constant_composition_uses_dataset_composition_and_explicit_inheritance() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    first_composition = rows[0]["composition"]
    for row in rows[1:]:
        assert isinstance(row, dict)
        row["composition"] = first_composition
        row["equivalence-ratio"] = rows[0]["equivalence-ratio"]
    record = parse_idt_record(_yaml(document), "constant.yaml")
    assert isinstance(record.envelope.composition, Composition)
    assert all(
        isinstance(point.composition, Absent) and point.composition.reason is AbsenceReason.SAME_AS_DATASET
        for point in record.envelope.series[0].points
    )


@pytest.mark.parametrize(
    ("target_raw", "criterion_raw", "target", "criterion"),
    [
        ("pressure", "d/dt max", IgnitionTarget.PRESSURE, IgnitionCriterion.MAX_SLOPE),
        ("OH*", "d/dt max", IgnitionTarget.OH_STAR, IgnitionCriterion.MAX_SLOPE),
        ("CH*", "d/dt max", IgnitionTarget.CH_STAR, IgnitionCriterion.MAX_SLOPE),
        ("OH*", "1/2 max", IgnitionTarget.OH_STAR, IgnitionCriterion.HALF_MAX),
        (
            "OH*",
            "d/dt max extrapolated",
            IgnitionTarget.OH_STAR,
            IgnitionCriterion.EXTRAPOLATED_MAX_SLOPE,
        ),
        ("OH*", "max", IgnitionTarget.OH_STAR, IgnitionCriterion.PEAK),
        ("CH*", "max", IgnitionTarget.CH_STAR, IgnitionCriterion.PEAK),
        ("OH", "1/2 max", IgnitionTarget.OH, IgnitionCriterion.HALF_MAX),
        ("CH", "d/dt max", IgnitionTarget.CH, IgnitionCriterion.MAX_SLOPE),
    ],
)
def test_schema_ignition_vocabulary_maps_without_merging_spellings(
    target_raw: str,
    criterion_raw: str,
    target: IgnitionTarget,
    criterion: IgnitionCriterion,
) -> None:
    raw = _with_ignition(target_raw, criterion_raw)
    record = parse_idt_record(raw, "fixture.yaml")
    assert (record.ignition.target, record.ignition.target_raw.raw) == (target, target_raw)
    assert (record.ignition.criterion, record.ignition.criterion_raw.raw) == (criterion, criterion_raw)
    replay_idt_record(record, raw)


@pytest.mark.parametrize(
    ("document", "path", "expected"),
    [
        ({"rows": [{"value": "12 bar"}]}, "rows[0].value", "12 bar"),
        ({"rows": [{"value": "12 bar"}]}, "rows[0].value#key", "value"),
        ({"rows": [{"value": "12 bar"}]}, "rows[0].value#value", "12"),
        ({"rows": [{"value": "12 bar"}]}, "rows[0].value#unit", "bar"),
    ],
)
def test_yaml_value_follows_positional_paths_and_markers(document: dict[str, object], path: str, expected: str) -> None:
    assert yaml_value(document, path) == expected


def test_yaml_value_rejects_non_mapping_and_out_of_range_list_access() -> None:
    with pytest.raises(KeyError):
        yaml_value({"rows": "not-a-list"}, "rows[0]")
    with pytest.raises(KeyError):
        yaml_value({"rows": []}, "rows[0]")
    with pytest.raises(KeyError):
        yaml_value({"rows": [{"value": "bare"}]}, "rows[0].value#unit")


def test_unknown_experiment_type_refuses_without_output() -> None:
    with pytest.raises(ChemkedRefusal, match="unknown_experiment_type"):
        parse_idt_record(FIXTURE.read_bytes().replace(b"ignition delay", b"flame speed", 1), "bad.yaml")


@pytest.mark.parametrize(("target", "criterion"), [("temperature", "d/dt max"), ("OH*", "min")])
def test_schema_vocabulary_outside_carmel_mapping_is_a_typed_refusal(target: str, criterion: str) -> None:
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_with_ignition(target, criterion), "bad.yaml")
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_IGNITION_DEFINITION


def test_unmapped_unit_refuses_without_output() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["temperature"] = ["828 fortnight"]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "bad-unit.yaml")
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_UNIT


def test_malformed_composition_amount_is_a_chemked_refusal_counted_by_loader(tmp_path: Path) -> None:
    good = FIXTURE.read_bytes()
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    composition = rows[0]["composition"]
    assert isinstance(composition, dict)
    species = composition["species"]
    assert isinstance(species, list) and isinstance(species[0], dict)
    species[0]["amount"] = ["not-a-number"]
    bad = _yaml(document)
    files = (
        ChemkedFile("good.yaml", hashlib.sha256(good).hexdigest()),
        ChemkedFile("bad-amount.yaml", hashlib.sha256(bad).hexdigest()),
    )
    for item, data in zip(files, (good, bad), strict=True):
        target = cached_archive_path(tmp_path, item.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    loaded = load_idt_records(
        ChemkedManifest("example.invalid/source", "a" * 40, files), cache_root=tmp_path, download=False
    )
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"unmapped_unit": 1}


def test_non_numeric_rcm_volume_history_is_a_chemked_refusal_counted_by_loader(tmp_path: Path) -> None:
    good = FIXTURE.read_bytes()
    document = _document()
    apparatus = document["apparatus"]
    rows = document["datapoints"]
    assert isinstance(apparatus, dict)
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    apparatus["kind"] = "rapid compression machine"
    rows[0]["volume-history"] = {"values": [[0.0, "not-a-number"], [1.0, 0.2]]}
    bad = _yaml(document)
    files = (
        ChemkedFile("good.yaml", hashlib.sha256(good).hexdigest()),
        ChemkedFile("bad-volume-history.yaml", hashlib.sha256(bad).hexdigest()),
    )
    for item, data in zip(files, (good, bad), strict=True):
        target = cached_archive_path(tmp_path, item.sha256)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    loaded = load_idt_records(
        ChemkedManifest("example.invalid/source", "a" * 40, files), cache_root=tmp_path, download=False
    )
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"schema_rejected": 1}


def test_empty_species_is_a_typed_refusal_counted_by_loader(tmp_path: Path) -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    composition = rows[0]["composition"]
    assert isinstance(composition, dict)
    composition["species"] = []
    loaded = _load_good_and_bad(tmp_path, _yaml(document), "empty-species.yaml")
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"incomplete_record": 1}


@pytest.mark.parametrize("entry", [[], 1, [0.0], [0.0, 1.0, 2.0]], ids=["empty", "scalar", "one", "three"])
def test_malformed_rcm_history_entry_is_a_typed_refusal_counted_by_loader(tmp_path: Path, entry: object) -> None:
    document = _document()
    apparatus = document["apparatus"]
    rows = document["datapoints"]
    assert isinstance(apparatus, dict)
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    apparatus["kind"] = "rapid compression machine"
    rows[0]["volume-history"] = {"values": [entry]}
    loaded = _load_good_and_bad(tmp_path, _yaml(document), "malformed-history.yaml")
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"schema_rejected": 1}


@pytest.mark.parametrize("field", ["pressure", "temperature", "ignition-delay"])
def test_multi_valued_scalar_is_a_typed_refusal_counted_by_loader(tmp_path: Path, field: str) -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    values = rows[0][field]
    assert isinstance(values, list)
    rows[0][field] = [values[0], values[0]]
    loaded = _load_good_and_bad(tmp_path, _yaml(document), f"multi-{field}.yaml")
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"schema_rejected": 1}


def test_multi_valued_species_amount_is_a_typed_refusal_counted_by_loader(tmp_path: Path) -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    composition = rows[0]["composition"]
    assert isinstance(composition, dict)
    species = composition["species"]
    assert isinstance(species, list) and isinstance(species[0], dict)
    species[0]["amount"] = [0.03, 0.04]
    loaded = _load_good_and_bad(tmp_path, _yaml(document), "multi-amount.yaml")
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"incomplete_record": 1}


def test_rcm_volume_history_without_values_is_accepted() -> None:
    document = _document()
    apparatus = document["apparatus"]
    rows = document["datapoints"]
    assert isinstance(apparatus, dict)
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    apparatus["kind"] = "rapid compression machine"
    rows[0]["volume-history"] = {"values": []}
    record = parse_idt_record(_yaml(document), "empty-volume-history.yaml")
    assert len(record.envelope.series[0].points) == 13


def test_non_mapping_datapoint_is_a_typed_refusal() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list)
    rows[0] = "not-a-mapping"
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "bad-point.yaml")
    assert caught.value.reason is ChemkedRefusalReason.INCOMPLETE_RECORD


def test_only_explicitly_known_species_roles_are_emitted() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        composition = row["composition"]
        assert isinstance(composition, dict)
        species = composition["species"]
        assert isinstance(species, list) and isinstance(species[-1], dict)
        species[-1]["species-name"] = "CO2"
    record = parse_idt_record(_yaml(document), "roles.yaml")
    assert record.fuels == ("2-butanol",)
    first = record.envelope.series[0].points[0].composition
    assert isinstance(first, Composition)
    roles = {item.species_raw_name: item.role for item in first.components}
    assert roles["2-butanol"] is ComponentRole.FUEL
    assert roles["O2"] is ComponentRole.OXIDIZER
    assert isinstance(roles["CO2"], Absent)


def test_value_without_a_unit_refuses_without_output() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    rows[0]["temperature"] = [828]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "bad-unit.yaml")
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_UNIT


def test_schema_invalid_yaml_refuses_without_output() -> None:
    with pytest.raises(ChemkedRefusal, match="schema_rejected"):
        parse_idt_record(FIXTURE.read_bytes().replace(b"file-authors:", b"file-authors-missing:", 1), "bad.yaml")


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b"[unterminated", ChemkedRefusalReason.MALFORMED_YAML),
        (b"- not\n- a\n- mapping\n", ChemkedRefusalReason.MALFORMED_YAML),
    ],
)
def test_malformed_yaml_refuses_without_output(raw: bytes, reason: ChemkedRefusalReason) -> None:
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(raw, "bad.yaml")
    assert caught.value.reason is reason


def test_unmapped_apparatus_refuses_without_output() -> None:
    document = _document()
    apparatus = document["apparatus"]
    assert isinstance(apparatus, dict)
    apparatus["kind"] = "flow reactor"
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "bad-apparatus.yaml")
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_APPARATUS


def test_incomplete_reference_refuses_without_output() -> None:
    document = _document()
    reference = document["reference"]
    assert isinstance(reference, dict)
    reference["doi"] = ""
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "missing-doi.yaml")
    assert caught.value.reason is ChemkedRefusalReason.INCOMPLETE_RECORD


@pytest.mark.parametrize("key", ["reference", "datapoints"])
def test_missing_required_schema_section_refuses_without_output(key: str) -> None:
    document = _document()
    del document[key]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "schema-invalid.yaml")
    assert caught.value.reason is ChemkedRefusalReason.SCHEMA_REJECTED


def test_missing_required_point_field_refuses_without_output() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    del rows[0]["pressure"]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "schema-invalid.yaml")
    assert caught.value.reason is ChemkedRefusalReason.SCHEMA_REJECTED


@pytest.mark.parametrize(
    "edit",
    [
        lambda document: document["datapoints"][0].update(composition={}),
        lambda document: document["datapoints"][0]["composition"].update(species=[{}]),
        lambda document: document["datapoints"][0]["composition"]["species"][0].update(amount=[]),
    ],
)
def test_incomplete_composition_refuses_without_output(edit: object) -> None:
    document = _document()
    edit(document)  # type: ignore[operator]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "bad-composition.yaml")
    assert caught.value.reason is ChemkedRefusalReason.INCOMPLETE_RECORD


def test_points_with_different_ignition_definitions_refuse_without_output() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list) and isinstance(rows[1], dict)
    rows[1]["ignition-type"] = {"target": "CH", "type": "d/dt max"}
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "mixed.yaml")
    assert caught.value.reason is ChemkedRefusalReason.UNMAPPED_IGNITION_DEFINITION


def test_non_mapping_point_refuses_without_output() -> None:
    document = _document()
    document["datapoints"] = ["not a mapping"]
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(_yaml(document), "bad-point.yaml")
    assert caught.value.reason is ChemkedRefusalReason.INCOMPLETE_RECORD


def test_common_ignition_definition_is_used_when_a_point_does_not_repeat_it() -> None:
    document = _document()
    rows = document["datapoints"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        del row["ignition-type"]
    raw = _yaml(document)
    record = parse_idt_record(raw, "common.yaml")
    assert record.ignition.target is IgnitionTarget.OH_STAR
    assert record.ignition.target_raw.ref.locator.path == "common-properties.ignition-type.target"
    replay_idt_record(record, raw)


def test_parse_sha_mismatch_refuses_without_output() -> None:
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml", "0" * 64)
    assert caught.value.reason is ChemkedRefusalReason.INCOMPLETE_RECORD


@pytest.mark.parametrize(
    "path",
    ["datapoints[99].pressure", "datapoints[0].pressure#value", "datapoints[0].pressure#other"],
)
def test_invalid_yaml_path_shapes_are_refused(path: str) -> None:
    with pytest.raises(KeyError):
        yaml_value(_document(), path)


def test_excessively_long_yaml_index_is_refused() -> None:
    with pytest.raises(KeyError):
        yaml_value({"rows": []}, f"rows[{'9' * 5000}]")


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


def test_changed_yaml_value_refuses_replay() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = replace(
        record,
        ignition=record.ignition.model_copy(
            update={"target_raw": record.ignition.target_raw.model_copy(update={"raw": "OH"})}
        ),
    )
    with pytest.raises(ChemkedRefusal) as caught:
        replay_idt_record(forged, FIXTURE.read_bytes())
    assert caught.value.reason is ChemkedRefusalReason.UNRESOLVABLE_PATH


def test_replay_rederives_ignition_target() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = replace(record, ignition=record.ignition.model_copy(update={"target": IgnitionTarget.PRESSURE}))
    with pytest.raises(ChemkedRefusal, match=r"ignition\.target"):
        replay_idt_record(forged, FIXTURE.read_bytes())


def test_replay_rederives_ignition_criterion() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = replace(record, ignition=record.ignition.model_copy(update={"criterion": IgnitionCriterion.PEAK}))
    with pytest.raises(ChemkedRefusal, match=r"ignition\.criterion"):
        replay_idt_record(forged, FIXTURE.read_bytes())


def test_replay_rederives_normalized_unit() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = _with_first_pressure_value(record, unit_normalized="bar")
    with pytest.raises(ChemkedRefusal, match=r"value\.unit_normalized"):
        replay_idt_record(forged, FIXTURE.read_bytes())


def test_replay_rederives_canonical_decimal_value() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = _with_first_pressure_value(record, canonical_decimal_value="1")
    with pytest.raises(ChemkedRefusal, match=r"value\.canonical_decimal_value"):
        replay_idt_record(forged, FIXTURE.read_bytes())


def test_replay_rederives_per_point_composition_amount() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    component = record.envelope.series[0].points[0].composition.components[0]
    forged = _with_first_composition_component(
        record, amount=component.amount.model_copy(update={"canonical_decimal_value": "0.04"})
    )
    with pytest.raises(ChemkedRefusal, match=r"composition\.components\[0\]\.amount\.canonical_decimal_value"):
        replay_idt_record(forged, FIXTURE.read_bytes())


def test_replay_rederives_per_point_composition_role() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = _with_first_composition_component(record, role=ComponentRole.DILUENT)
    with pytest.raises(ChemkedRefusal, match=r"composition\.components\[0\]\.role"):
        replay_idt_record(forged, FIXTURE.read_bytes())


def test_non_yaml_locator_refuses_replay() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    forged = replace(
        record,
        ignition=record.ignition.model_copy(
            update={
                "target_raw": record.ignition.target_raw.model_copy(
                    update={"ref": SourceRef(node_id="record", locator=XPathLocator(xpath="/record"))}
                )
            }
        ),
    )
    with pytest.raises(ChemkedRefusal) as caught:
        replay_idt_record(forged, FIXTURE.read_bytes())
    assert caught.value.reason is ChemkedRefusalReason.UNRESOLVABLE_PATH


def test_sha_mismatch_refuses_replay_before_any_partial_check() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    with pytest.raises(ChemkedRefusal) as caught:
        replay_idt_record(record, FIXTURE.read_bytes() + b"\n")
    assert caught.value.reason is ChemkedRefusalReason.INCOMPLETE_RECORD


def test_precompression_rcm_history_refuses() -> None:
    raw = FIXTURE.read_bytes().replace(b"kind: shock tube", b"kind: rapid compression machine", 1)
    raw = raw.replace(
        b"  - temperature:",
        b"  - volume-history:\n      values:\n        - [0.0, 1.0]\n        - [1.0, 0.2]\n    temperature:",
        1,
    )
    with pytest.raises(ChemkedRefusal, match="rcm_pre_compression_conditions"):
        parse_idt_record(raw, "precompression.yaml")


def test_yaml_path_locator_is_available_from_star_import() -> None:
    namespace: dict[str, object] = {}
    exec("from carmel.schemas.datasets import *", namespace)
    assert namespace["YamlPathLocator"] is YamlPathLocator


def test_packaged_manifest_pins_the_chemked_commit() -> None:
    manifest = load_manifest()
    assert manifest.repository == "pr-omethe-us/ChemKED-database"
    assert manifest.commit == "606005bfc8f5214b3f0b5ca7300a96a82815c2ae"
    assert len(manifest.files) == 351
    assert manifest.raw_url(manifest.files[0]).startswith(
        "https://raw.githubusercontent.com/pr-omethe-us/ChemKED-database/606005bfc8f5214b3f0b5ca7300a96a82815c2ae/"
    )


def test_fetch_quotes_spaces_in_pinned_path_before_using_http(
    http_double: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hasattr(http_double, "body") and hasattr(http_double, "server_address")
    data = b"fixture"
    item = ChemkedFile("n-heptane/Burcat 1981/fixture.yaml", hashlib.sha256(data).hexdigest())
    manifest = ChemkedManifest("example/source", "a" * 40, (item,))
    http_double.body = data
    real_urlopen = urllib.request.urlopen
    seen: list[str] = []

    def route_to_double(url: str, timeout: float) -> object:
        seen.append(url)
        assert " " not in url
        assert "Burcat%201981" in url
        local_url = f"http://127.0.0.1:{http_double.server_address[1]}/fixture.yaml"
        return real_urlopen(local_url, timeout=timeout)

    monkeypatch.setattr(urllib.request, "urlopen", route_to_double)
    assert fetch_file(item, manifest, tmp_path) == data
    assert len(seen) == 1


@pytest.mark.parametrize(
    "edit",
    [
        lambda payload: payload.update(license="CC-BY-NC-4.0"),
        lambda payload: payload.update(manifest_version=2),
        lambda payload: payload.update(files=[]),
        lambda payload: payload["files"][0].update(sha256="short"),
    ],
)
def test_malformed_manifest_is_refused(tmp_path: Path, edit: object) -> None:
    payload = json.loads((Path("carmel/data/chemked_manifest.json")).read_text())
    edit(payload)  # type: ignore[operator]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid ChemKED"):
        load_manifest(path)


@pytest.mark.parametrize(
    "raw",
    [b"not json", b"[]", json.dumps({"manifest_version": 1, "license": "CC-BY-4.0"}).encode()],
)
def test_manifest_read_and_top_level_shape_fail_closed(tmp_path: Path, raw: bytes) -> None:
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="invalid ChemKED"):
        load_manifest(path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda payload: payload.update(repository=""),
        lambda payload: payload["files"].__setitem__(0, "not-an-object"),
        lambda payload: payload["files"].append(payload["files"][0].copy()),
    ],
)
def test_manifest_rejects_empty_repository_non_object_and_duplicate_pins(tmp_path: Path, edit: object) -> None:
    payload = json.loads(Path("carmel/data/chemked_manifest.json").read_text())
    edit(payload)  # type: ignore[operator]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid ChemKED"):
        load_manifest(path)


@pytest.mark.parametrize(
    "edit",
    [
        lambda payload: payload.update(manifest_version=True),
        lambda payload: payload.update(license=7),
        lambda payload: payload["files"][0].update(sha256="A" * 64),
        lambda payload: payload["files"][0].update(sha256="../" + "a" * 61),
        lambda payload: payload["files"][0].update(path=7),
        lambda payload: payload.update(repository=7),
    ],
)
def test_manifest_rejects_non_lowercase_hashes_traversal_and_wrong_field_types(
    http_double: object, tmp_path: Path, edit: object
) -> None:
    assert hasattr(http_double, "requests")
    payload = json.loads(Path("carmel/data/chemked_manifest.json").read_text())
    edit(payload)  # type: ignore[operator]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="invalid ChemKED"):
        load_manifest(path)
    assert http_double.requests == 0


def _point_manifest_at(http_double: object, item: ChemkedFile, monkeypatch: pytest.MonkeyPatch) -> ChemkedManifest:
    assert hasattr(http_double, "server_address")
    url = f"http://127.0.0.1:{http_double.server_address[1]}/fixture.yaml"
    monkeypatch.setattr(ChemkedManifest, "raw_url", lambda self, selected: url)
    return ChemkedManifest("example.invalid/source", "a" * 40, (item,))


def test_download_verifies_caches_and_is_not_repeated(
    http_double: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hasattr(http_double, "body") and hasattr(http_double, "requests")
    data = FIXTURE.read_bytes()
    item = ChemkedFile("fixture.yaml", hashlib.sha256(data).hexdigest())
    manifest = _point_manifest_at(http_double, item, monkeypatch)
    http_double.body = data
    assert fetch_file(item, manifest, tmp_path) == data
    assert cached_archive_path(tmp_path, item.sha256).read_bytes() == data
    assert fetch_file(item, manifest, tmp_path, download=False) == data
    assert http_double.requests == 1


def test_download_sha_mismatch_is_refused_and_not_cached(
    http_double: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hasattr(http_double, "body")
    item = ChemkedFile("fixture.yaml", hashlib.sha256(b"expected").hexdigest())
    manifest = _point_manifest_at(http_double, item, monkeypatch)
    http_double.body = b"wrong"
    with pytest.raises(ArchiveIntegrityError, match="does not match pinned sha256"):
        fetch_file(item, manifest, tmp_path)
    assert not cached_archive_path(tmp_path, item.sha256).exists()


def test_oversized_download_is_a_typed_refusal_and_is_not_cached(
    http_double: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hasattr(http_double, "body")
    data = b"x" * (4 * 1024 * 1024 + 1)
    item = ChemkedFile("oversized.yaml", hashlib.sha256(data).hexdigest())
    manifest = _point_manifest_at(http_double, item, monkeypatch)
    http_double.body = data
    with pytest.raises(ArchiveIntegrityError, match="maximum"):
        fetch_file(item, manifest, tmp_path)
    assert not cached_archive_path(tmp_path, item.sha256).exists()


def test_oversized_cached_file_is_refused_before_it_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    item = ChemkedFile("oversized.yaml", "a" * 64)
    manifest = ChemkedManifest("example/source", "b" * 40, (item,))
    target = cached_archive_path(tmp_path, item.sha256)
    target.parent.mkdir(parents=True)
    with target.open("wb") as handle:
        handle.truncate(4 * 1024 * 1024 + 1)
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self} without a bound"))
    with pytest.raises(ArchiveIntegrityError, match="maximum"):
        fetch_file(item, manifest, tmp_path, download=False)


def test_cached_file_that_grows_after_stat_is_refused_without_cache_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = b"x" * (4 * 1024 * 1024 + 1)
    item = ChemkedFile("growing.yaml", hashlib.sha256(data).hexdigest())
    manifest = ChemkedManifest("example/source", "b" * 40, (item,))
    target = cached_archive_path(tmp_path, item.sha256)
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    original_stat = Path.stat

    def report_small_size(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = original_stat(path, *args, **kwargs)
        if path == target:
            values = list(result)
            values[6] = 0
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", report_small_size)
    with pytest.raises(ArchiveIntegrityError, match="maximum"):
        fetch_file(item, manifest, tmp_path, download=False)
    assert target.read_bytes() == data


def test_cached_sha_mismatch_refuses_before_contacting_http_double(
    http_double: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert hasattr(http_double, "body") and hasattr(http_double, "requests")
    item = ChemkedFile("fixture.yaml", hashlib.sha256(b"expected").hexdigest())
    manifest = _point_manifest_at(http_double, item, monkeypatch)
    http_double.body = b"expected"
    cache = cached_archive_path(tmp_path, item.sha256)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"corrupted cache")
    with pytest.raises(ArchiveIntegrityError, match="does not match pinned sha256"):
        fetch_file(item, manifest, tmp_path)
    assert http_double.requests == 0
    assert cache.read_bytes() == b"corrupted cache"


def test_http_failure_and_offline_miss_are_fetch_errors(
    http_double: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = ChemkedFile("fixture.yaml", hashlib.sha256(b"expected").hexdigest())
    manifest = _point_manifest_at(http_double, item, monkeypatch)
    with pytest.raises(ArchiveFetchError, match="cannot fetch"):
        fetch_file(item, manifest, tmp_path)
    with pytest.raises(ArchiveFetchError, match="not in the cache"):
        fetch_file(item, manifest, tmp_path, download=False)


def test_loader_counts_typed_refusals_without_partial_records(tmp_path: Path) -> None:
    good = FIXTURE.read_bytes()
    bad = b"[unterminated"
    files = (
        ChemkedFile("good.yaml", hashlib.sha256(good).hexdigest()),
        ChemkedFile("bad.yaml", hashlib.sha256(bad).hexdigest()),
    )
    for item, data in zip(files, (good, bad), strict=True):
        path = cached_archive_path(tmp_path, item.sha256)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    loaded = load_idt_records(
        ChemkedManifest("example.invalid/source", "a" * 40, files), cache_root=tmp_path, download=False
    )
    assert [record.path for record in loaded.records] == ["good.yaml"]
    assert dict(loaded.refusals) == {"malformed_yaml": 1}


def test_query_reports_exact_condition_ranges_and_matched_points() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    assert point_conditions(record)[0] == (Decimal("828"), Decimal("15.8067"))
    (match,) = find_idt(
        (record,),
        fuel="2-butanol",
        temperature_k=ConditionWindow(Decimal("800"), Decimal("900")),
        pressure_bar=ConditionWindow(Decimal("15"), Decimal("21")),
    )
    assert match.record is record
    assert match.matched_points == 4
    assert match.total_points == 13
    assert match.temperature_k == (Decimal("828"), Decimal("1062"))
    assert match.pressure_bar == (Decimal("15.097425"), Decimal("20.974275"))
    assert (
        find_idt(
            (record,),
            fuel="H2",
            temperature_k=None,
            pressure_bar=None,
        )
        == []
    )


def test_query_returns_no_match_when_windows_exclude_all_points() -> None:
    record = parse_idt_record(FIXTURE.read_bytes(), "fixture.yaml")
    assert find_idt((record,), temperature_k=ConditionWindow(Decimal("100"), Decimal("200"))) == []


def test_data_find_lists_chemked_and_deduplicates_only_matching_doi_and_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    respecth_manifest = respecth_archive.load_manifest()
    archive = next(item for item in respecth_manifest.archives if item.name == "syngas_indirect_v2_3.zip")
    member_path = "x40001058_19.xml"
    respecth_record = parse_respecth_idt_record(
        (Path("tests/fixtures/respecth") / member_path).read_bytes(), archive, member_path
    )
    assert respecth_record.paper_doi is not None
    ((temperature, pressure),) = respecth_query.point_conditions(respecth_record)

    duplicate_doc = _document()
    duplicate_reference = duplicate_doc["reference"]
    duplicate_rows = duplicate_doc["datapoints"]
    assert isinstance(duplicate_reference, dict) and isinstance(duplicate_rows, list)
    duplicate_reference["doi"] = respecth_record.paper_doi
    duplicate_row = duplicate_rows[0]
    assert isinstance(duplicate_row, dict)
    duplicate_row["temperature"] = [f"{temperature} K"]
    duplicate_row["pressure"] = [f"{pressure * Decimal(100000)} Pa"]
    duplicate_doc["datapoints"] = [duplicate_row]
    duplicate = parse_idt_record(_yaml(duplicate_doc), "duplicate.yaml")

    distinct_doc = _document()
    distinct_reference = distinct_doc["reference"]
    assert isinstance(distinct_reference, dict)
    distinct_reference["doi"] = respecth_record.paper_doi
    distinct = parse_idt_record(_yaml(distinct_doc), "distinct.yaml")

    monkeypatch.setattr(respecth_archive, "load_manifest", lambda path=None: object())
    monkeypatch.setattr(
        respecth_query,
        "load_records",
        lambda manifest, kind, cache_root, download: LoadResult((respecth_record,), Counter()),
    )
    monkeypatch.setattr(chemked_archive, "load_manifest", lambda: object())
    monkeypatch.setattr(
        chemked_query,
        "load_idt_records",
        lambda manifest, cache_root, download: chemked_query.ChemkedLoadResult((duplicate, distinct), Counter()),
    )

    assert main(["data", "find", "--kind", "idt", "--cache", str(tmp_path), "--offline"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "source\trecord_doi\tpaper_doi\tdevice\tfuels\tT_K\tP_bar\tpoints",
        "ReSpecTh\t10.24388/x40001058_19\t10.1016/j.combustflame.2014.03.001\trcm\tCO+H2\t1039-1039\t11.04-11.04\t1/1",
        "ChemKED\t10.1016/j.combustflame.2014.03.001\t10.1016/j.combustflame.2014.03.001\t-\t2-butanol\t828-1062\t15.1-20.97\t13/13",
        "2 matching dataset(s) of 1 ReSpecTh and 2 ChemKED mapped ignition-delay records; 0 refused (none)",
    ]


def test_data_find_chemked_integrity_failure_lists_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(respecth_archive, "load_manifest", lambda path=None: object())
    monkeypatch.setattr(
        respecth_query,
        "load_records",
        lambda manifest, kind, cache_root, download: LoadResult((), Counter()),
    )
    monkeypatch.setattr(
        chemked_archive,
        "load_manifest",
        lambda: (_ for _ in ()).throw(ArchiveIntegrityError("bad ChemKED pin")),
    )
    assert main(["data", "find", "--kind", "idt", "--cache", str(tmp_path), "--offline"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Refusing to list ChemKED records: bad ChemKED pin" in captured.err
