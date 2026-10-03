"""Curated history mapping, evidence replay, and T3 export acceptance checks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from Carmel import main
from carmel.services import chem, t3_export
from carmel.services.chemked import ChemkedRefusal, parse_idt_record, replay_idt_record, yaml_value
from carmel.services.rcm_history import in_si
from carmel.services.respecth import IgnitionCriterion, IgnitionTarget, RespecthRefusal
from carmel.services.respecth import parse_idt_record as parse_respecth
from carmel.services.respecth import replay_idt_record as replay_respecth
from carmel.services.respecth_archive import load_manifest
from carmel.services.respecth_query import point_conditions
from carmel.services.t3_export import export_idt, write_export

FIXTURES = Path(__file__).parent / "fixtures"


def chemked_history(*, explicit: bool = True) -> bytes:
    doc = yaml.safe_load((FIXTURES / "chemked/Bec_2014_2-b_20atm.yaml").read_bytes())
    doc["apparatus"]["kind"] = "rapid compression machine"
    doc["datapoints"] = doc["datapoints"][:1]
    row = doc["datapoints"][0]
    row.update(
        temperature=["350 kelvin"],
        pressure=["1 bar"],
        **{
            "compressed-temperature": ["800 kelvin"],
            "compressed-pressure": ["10 bar"],
            "volume-history": {
                "time": {"units": "ms", "column": 1},
                "volume": {"units": "cm3", "column": 0},
                "values": [[10, 5], [2, 7], [3, 8]],
            },
        },
    )
    if explicit:
        row["compression-time"] = ["7.1 ms"]
    return yaml.safe_dump(doc, sort_keys=False).encode()


def respecth_record(name: str = "x40001039.xml"):
    raw = (FIXTURES / "respecth" / name).read_bytes()
    archive = next(a for a in load_manifest().archives if a.name.startswith("syngas"))
    return parse_respecth(raw, archive, name), raw


def test_stated_history_state_samples_columns_and_time_axis_replay() -> None:
    raw = chemked_history()
    record = parse_idt_record(raw, "study/history.yaml")
    history, state = record.rcm_histories[0], record.rcm_states[0]
    assert history is not None and state is not None
    assert history.times == (0.005, 0.007, 0.008)
    assert history.volumes == (1e-5, 2e-6, 3e-6)
    assert history.compression_time == 0.0071
    assert not history.compression_time_derived
    assert history.volume_ratio == pytest.approx(10 / 2.1)
    assert in_si(state.initial_temperature) == 350
    assert in_si(state.temperature) == 800
    assert in_si(state.initial_pressure) == 100000
    assert in_si(state.pressure) == 1000000
    assert state.eoc_basis == "stated"
    replay_idt_record(record, raw)
    doc = yaml.safe_load(raw)
    for index, value in enumerate(history.volume):
        assert yaml_value(doc, value.value_ref.locator.path) == [10, 2, 3][index]
    with pytest.raises(KeyError):
        yaml_value(doc, "datapoints[0].volume-history.values[0][999]")


def test_minimum_volume_compression_time_is_explicitly_derived() -> None:
    record = parse_idt_record(chemked_history(explicit=False), "history.yaml")
    history = record.rcm_histories[0]
    assert history.compression_time == 0.007
    assert history.compression_time_derived
    assert history.volume_ratio == pytest.approx(5)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("reversed_time", "history_nonmonotone_time"),
        ("duplicate_time", "history_nonmonotone_time"),
        ("zero_volume", "history_nonpositive_volume"),
        ("negative_volume", "history_nonpositive_volume"),
        ("flat", "history_no_compression"),
        ("missing_state", "history_missing_initial_state"),
        ("compression_outside", "history_invalid"),
        ("same_column", "history_invalid"),
        ("bool_column", "history_invalid"),
        ("list_column", "history_invalid"),
    ],
)
def test_chemked_history_failures_have_precise_reasons(change: str, reason: str) -> None:
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    history = row["volume-history"]
    if change == "reversed_time":
        history["values"][1][1] = 4
    elif change == "duplicate_time":
        history["values"][1][1] = 5
    elif change == "zero_volume":
        history["values"][1][0] = 0
    elif change == "negative_volume":
        history["values"][1][0] = -1
    elif change == "flat":
        history["values"] = [[10, 5], [10, 7], [10, 8]]
    elif change == "missing_state":
        row.pop("temperature")
    elif change == "compression_outside":
        row["compression-time"] = ["9 ms"]
    elif change == "same_column":
        history["volume"]["column"] = 1
    elif change == "bool_column":
        history["volume"]["column"] = False
    elif change == "list_column":
        history["volume"]["column"] = []
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "bad.yaml")
    assert caught.value.reason.value == reason


def test_new_history_and_state_facts_cannot_be_forged_while_replaying() -> None:
    raw = chemked_history()
    record = parse_idt_record(raw, "history.yaml")
    state = record.rcm_states[0]
    forged = replace(record, rcm_states=(state.model_copy(update={"eoc_basis": "derived-isentropic"}),))
    with pytest.raises(ChemkedRefusal, match="rcm_states"):
        replay_idt_record(forged, raw)
    history = record.rcm_histories[0]
    forged = replace(record, rcm_histories=(history.model_copy(update={"compression_time_stated": None}),))
    with pytest.raises(ChemkedRefusal, match="rcm_histories"):
        replay_idt_record(forged, raw)
    with pytest.raises(ValueError, match="align"):
        replace(record, rcm_states=())
    with pytest.raises(ValueError, match="paired"):
        replace(record, rcm_states=(None,))
    with pytest.raises(ValueError, match="apparatus"):
        replace(record, apparatus="shock tube")


def test_respecth_carries_grounded_initial_state_and_derivation_basis() -> None:
    record, raw = respecth_record()
    assert record.end_of_compression_basis == "derived-isentropic"
    assert record.rcm_conditions.state == "pre_compression"
    assert record.skipped_data_groups == ()
    assert len(record.rcm_states) == len(record.envelope.series[0].points)
    history = record.rcm_conditions.histories[0].history
    assert history.compression_time == history.times[history.volumes.index(min(history.volumes))]
    assert history.compression_time_derived
    assert replay_respecth(record, raw).verified
    assert point_conditions(record) == []
    forged = record.model_copy(update={"rcm_states": tuple(reversed(record.rcm_states))})
    # Distinct point states must stay aligned with the point links.
    if forged.rcm_states != record.rcm_states:
        assert not replay_respecth(forged, raw).verified
    with pytest.raises(ValueError, match="derived-isentropic"):
        type(record).model_validate({**record.model_dump(), "end_of_compression_basis": "stated"})


def test_respecth_torr_uses_the_rational_table_and_maps_compression() -> None:
    record, raw = respecth_record()
    edited = raw.replace(b'units="mbar"', b'units="Torr"')
    mapped = parse_respecth(
        edited, next(a for a in load_manifest().archives if a.name.startswith("syngas")), record.archive.member_path
    )
    assert mapped.end_of_compression_basis == "derived-isentropic"
    assert replay_respecth(mapped, edited).verified
    initial_pressure = mapped.rcm_states[0].initial_pressure
    assert in_si(initial_pressure) == pytest.approx(586 * 101325 / 760)


def test_export_round_trip_of_three_apparatus_state_forms(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    shock = parse_idt_record((FIXTURES / "chemked/Bec_2014_2-b_20atm.yaml").read_bytes(), "shock.yaml")
    rcm, _ = respecth_record("x40001058_19.xml")
    history = parse_idt_record(chemked_history(), "history.yaml")
    payload, report = export_idt([shock, rcm, history])
    assert report["exported"] == 15
    assert report["stated"] == 15 and report["derived"] == 0
    path = tmp_path / "points.yaml"
    report_path = write_export(payload, report, path)
    loaded = yaml.safe_load(path.read_bytes())
    assert loaded == payload
    histories = [p for p in loaded["points"] if p.get("volume_history")]
    assert len(histories) == 1
    point = histories[0]
    assert point["temperature"] == {"value": 800.0, "units": "K"}
    assert point["pressure"] == {"value": 1000000.0, "units": "Pa"}
    assert point["initial_temperature"]["value"] == 350
    assert point["volume_history"]["compression_time"]["value"] == 0.0071
    assert point["ignition_definition"] == {"target": "OH*", "type": "d/dt max"}
    assert all(p["idt"]["units"] == "s" and p["source"]["doi"] for p in loaded["points"])
    assert report_path.exists()


def test_export_selection_is_exact_and_unknown_studies_fail_closed() -> None:
    pytest.importorskip("rdkit")
    a = parse_idt_record(chemked_history(), "a/history.yaml")
    b = replace(a, path="b/history.yaml")
    payload, report = export_idt([b, a], studies=("a/history.yaml",), history_only=True)
    assert report["exported"] == 1
    assert "a/history.yaml" in payload["points"][0]["source"]["record"]
    assert export_idt([a], fuel="absent")[1]["exported"] == 0
    with pytest.raises(ValueError, match="not mapped"):
        export_idt([a], studies=("unknown.yaml",))


def test_derived_label_emission_and_switch(tmp_path: Path) -> None:
    pytest.importorskip("rdkit")
    record, _ = respecth_record()
    payload, report = export_idt([record], history_only=True, include_derived_labels=True)
    assert report["derived"] == len(record.rcm_states)
    assert report["stated"] == 0
    assert not report["refused"]
    assert all("eoc=derived-isentropic" in p["source"]["record"] for p in payload["points"])
    assert all(p["temperature"]["value"] > p["initial_temperature"]["value"] for p in payload["points"])
    unlabelled, report = export_idt([record])
    assert report["unlabelled"] == len(record.rcm_states)
    assert all("temperature" not in p and "pressure" not in p for p in unlabelled["points"])
    assert yaml.safe_load(yaml.safe_dump(payload)) == payload


@pytest.mark.parametrize(
    "target,criterion",
    [(IgnitionTarget.OHEX, IgnitionCriterion.PEAK), (IgnitionTarget.OH, IgnitionCriterion.RELATIVE_CONCENTRATION)],
)
def test_inexpressible_ignition_definitions_are_reported(target, criterion) -> None:
    record, _ = respecth_record()
    record = record.model_copy(
        update={"ignition": record.ignition.model_copy(update={"target": target, "criterion": criterion})}
    )
    _, report = export_idt([record])
    assert report["refused"] == {"inexpressible_ignition_definition": len(record.rcm_states)}


def test_missing_smiles_and_no_rdkit_are_explicit_refusals(monkeypatch) -> None:
    record = parse_idt_record(chemked_history(), "history.yaml")
    record = replace(record, species_identifiers=())
    _, report = export_idt([record])
    assert report["refused"] == {"no_confident_smiles": 1}
    monkeypatch.setattr(chem, "canonical_smiles", lambda _: None)
    monkeypatch.setattr(chem, "smiles_from_inchi", lambda _: None)
    _, report = export_idt([parse_idt_record(chemked_history(), "history.yaml")])
    assert report["refused"] == {"no_confident_smiles": 1}


@pytest.mark.parametrize(
    "path",
    [
        "../escape.yaml",
        "/absolute.yaml",
        "C:/absolute.yaml",
        "C:relative.yaml",
        "dir\\member.yaml",
        "//server/share/member.yaml",
    ],
)
def test_source_member_locators_reject_cross_platform_escape(path: str) -> None:
    record = parse_idt_record(chemked_history(), path)
    _, report = export_idt([record])
    assert report["refused"] == {"unsafe_source_locator": 1}


def test_cli_writes_report_and_handles_load_errors(tmp_path: Path, monkeypatch, capsys) -> None:
    record = parse_idt_record(chemked_history(), "history.yaml")
    monkeypatch.setattr(t3_export, "load_records", lambda **_: ((record,), {}))
    output = tmp_path / "output.yaml"
    assert (
        main(
            [
                "data",
                "export-t3",
                "--source",
                "chemked",
                "--study",
                "history.yaml",
                "--output",
                str(output),
                "--offline",
            ]
        )
        == 0
    )
    assert output.exists() and output.with_suffix(".report.json").exists()
    assert '"output"' in capsys.readouterr().out
    assert main(["data", "export-t3", "--study", "absent.yaml", "--output", str(output)]) == 1
    assert "Refusing T3 export" in capsys.readouterr().err
    with pytest.raises(ValueError, match="suffix"):
        write_export({}, {}, tmp_path / "same.report.json")


def test_record_size_limits_precede_hashing_and_parsing() -> None:
    with pytest.raises(ChemkedRefusal, match="4 MiB"):
        parse_idt_record(b"x" * (4 * 1024 * 1024 + 1), "large.yaml")
    with pytest.raises(RespecthRefusal, match="4 MiB"):
        parse_respecth(b"x" * (4 * 1024 * 1024 + 1), load_manifest().archives[0], "large.xml")


def test_source_compressed_labels_must_be_paired() -> None:
    doc = yaml.safe_load(chemked_history())
    del doc["datapoints"][0]["compressed-pressure"]
    with pytest.raises(ChemkedRefusal, match="paired"):
        parse_idt_record(yaml.safe_dump(doc).encode(), "bad.yaml")


def test_source_identifiers_and_history_collections_are_immutable() -> None:
    record = parse_idt_record(chemked_history(), "history.yaml")
    for changes in (
        {"species_identifiers": list(record.species_identifiers)},
        {"species_identifiers": (list(record.species_identifiers[0]),)},
        {"rcm_histories": list(record.rcm_histories)},
        {"species_identifiers": (("", "smiles", record.species_identifiers[0][2]),)},
    ):
        with pytest.raises(ValueError):
            replace(record, **changes)


def test_history_and_state_reject_quantity_role_corruption() -> None:
    from carmel.services.rcm_history import RcmHistory, RcmState

    record = parse_idt_record(chemked_history(), "history.yaml")
    history, state = record.rcm_histories[0], record.rcm_states[0]
    invalid = [
        {"time": history.volume, "volume": history.volume},
        {"time": history.time, "volume": history.volume[:2]},
        {"time": history.time, "volume": history.volume, "compression_time_stated": state.initial_pressure},
    ]
    for fields in invalid:
        with pytest.raises(ValueError):
            RcmHistory(**fields)
    for updates in (
        {"initial_temperature": state.initial_pressure},
        {"initial_pressure": state.initial_pressure.model_copy(update={"canonical_decimal_value": "0"})},
        {"temperature": None},
        {"eoc_basis": "derived-isentropic"},
        {"temperature": None, "pressure": state.pressure, "eoc_basis": "derived-isentropic"},
    ):
        with pytest.raises(ValueError):
            RcmState.model_validate({**state.model_dump(), **updates})


def test_manifest_bounds_precede_parsing(tmp_path: Path) -> None:
    from carmel.services import chemked_archive, respecth_archive

    path = tmp_path / "large.json"
    path.write_bytes(b"x" * (4 * 1024 * 1024 + 1))
    for lane in (chemked_archive, respecth_archive):
        with pytest.raises(lane.ManifestError, match="4 MiB"):
            lane.load_manifest(path)


def test_coverage_recognizes_rcm_and_refuses_unbounded_or_malformed_input() -> None:
    from carmel.services.data_coverage import _is_rcm

    assert _is_rcm(chemked_history(), "ChemKED")
    assert not _is_rcm(b"[", "ChemKED")
    assert not _is_rcm(b"t: " + b"9" * 5000, "ChemKED")
    assert not _is_rcm(b"not xml", "ReSpecTh")
    assert not _is_rcm(b"x" * (4 * 1024 * 1024 + 1), "ChemKED")


def test_source_inchi_conversion_fails_closed(monkeypatch) -> None:
    if __import__("importlib.util").util.find_spec("rdkit") is None:
        assert chem.smiles_from_inchi("1S/H2/h1H") is None
        return
    from rdkit import Chem

    assert chem.smiles_from_inchi("1S/H2O/h1H2") == "O"
    assert chem.smiles_from_inchi("not an InChI") is None
    monkeypatch.setattr(Chem, "MolFromInchi", lambda _: (_ for _ in ()).throw(ValueError("bad library")))
    assert chem.smiles_from_inchi("1S/H2/h1H") is None


def test_loader_replays_pinned_sources_and_exact_fuel_prefilter(tmp_path: Path, monkeypatch) -> None:
    import hashlib
    from types import SimpleNamespace

    from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest
    from carmel.services.respecth import RespecthRefusalReason

    good = chemked_history()
    items = (
        ChemkedFile("chosen/good.yaml", hashlib.sha256(good).hexdigest()),
        ChemkedFile("other/skipped.yaml", hashlib.sha256(good).hexdigest()),
        ChemkedFile("chosen/bad.yaml", hashlib.sha256(b"bad").hexdigest()),
    )
    manifest = ChemkedManifest("fixture/repo", "0" * 40, items)
    monkeypatch.setattr(t3_export, "load_manifest", lambda: manifest)
    fetched = []

    def fetch(item, *args, **kwargs):
        fetched.append(item.path)
        return b"bad" if item.path.endswith("bad.yaml") else good

    monkeypatch.setattr(t3_export, "fetch_file", fetch)
    records, refused = t3_export.load_records(source="chemked", cache_root=tmp_path, download=False, fuel="chosen")
    assert len(records) == 1 and set(fetched) == {"chosen/good.yaml", "chosen/bad.yaml"}
    assert sum(refused.values()) == 1
    mapped, raw = respecth_record()
    monkeypatch.setattr(t3_export, "respecth_manifest", lambda: load_manifest())
    monkeypatch.setattr(t3_export, "fetch_archive", lambda *args, **kwargs: b"archive")
    monkeypatch.setattr(
        t3_export,
        "iter_xml_members",
        lambda _: iter([("good.xml", raw), ("bad.xml", b"bad"), ("non-idt.xml", b"non-idt")]),
    )

    def parse(raw, archive, member):
        if raw == b"bad":
            raise RespecthRefusal(RespecthRefusalReason.MALFORMED_XML, "bad")
        if raw == b"non-idt":
            raise RespecthRefusal(RespecthRefusalReason.NOT_IGNITION_DELAY, "other observable")
        return parse_respecth(raw, archive, member)

    monkeypatch.setattr(t3_export, "parse_respecth", parse)
    records, refused = t3_export.load_records(source="all", cache_root=tmp_path, download=False)
    assert len(records) == 4 and refused["respecth:malformed_xml"] == 2
    monkeypatch.setattr(t3_export, "replay_respecth", lambda *args: SimpleNamespace(verified=False))
    with pytest.raises(ValueError, match="replay failed"):
        t3_export.load_records(source="respecth", cache_root=tmp_path, download=False)
    with pytest.raises(ValueError, match="source must"):
        t3_export.load_records(source="unknown", cache_root=tmp_path, download=False)


def test_export_refuses_invalid_point_values_and_history_duration(monkeypatch) -> None:
    from carmel.schemas.datasets import AbsenceReason, Absent

    pytest.importorskip("rdkit")
    record = parse_idt_record(chemked_history(), "history.yaml")
    series = record.envelope.series[0]
    point = series.points[0]
    observation = point.observations[0]
    for value in (
        Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
        observation.value.model_copy(update={"canonical_decimal_value": "0"}),
        observation.value.model_copy(update={"canonical_decimal_value": "11", "unit_normalized": "s"}),
    ):
        altered = point.model_copy(update={"observations": (observation.model_copy(update={"value": value}),)})
        changed = replace(
            record,
            envelope=record.envelope.model_copy(update={"series": (series.model_copy(update={"points": (altered,)}),)}),
        )
        assert export_idt([changed])[1]["refused"] == {"t3_constraint": 1}
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"][0]["volume-history"]["values"][2][1] = 20000
    long = parse_idt_record(yaml.safe_dump(doc).encode(), "history.yaml")
    assert export_idt([long])[1]["refused"] == {"t3_constraint": 1}
    derived, _ = respecth_record()
    monkeypatch.setattr(
        t3_export, "isentropic_eoc", lambda *args: (_ for _ in ()).throw(ValueError("unsupported thermo"))
    )
    assert export_idt([derived], include_derived_labels=True)[1]["refused"] == {
        "rcm_thermo_unavailable": len(derived.rcm_states)
    }


def test_export_does_not_repair_source_composition() -> None:
    from carmel.schemas.datasets import AbsenceReason, Absent

    pytest.importorskip("rdkit")
    record = parse_idt_record(chemked_history(), "history.yaml")
    composition = record.envelope.composition
    for fractions in (("0", "0", "0"), ("0.5", "0.5", "0.5"), ("-0.1", "0.5", "0.6"), ("2", "0", "0")):
        components = tuple(
            c.model_copy(update={"amount": c.amount.model_copy(update={"canonical_decimal_value": n})})
            for c, n in zip(composition.components, fractions, strict=True)
        )
        changed = replace(
            record,
            envelope=record.envelope.model_copy(
                update={"composition": composition.model_copy(update={"components": components})}
            ),
        )
        assert export_idt([changed])[1]["refused"] == {"invalid_composition": 1}
    absent = Absent(reason=AbsenceReason.NOT_REPORTED_HERE)
    changed = replace(record, envelope=record.envelope.model_copy(update={"composition": absent}))
    assert export_idt([changed])[1]["refused"] == {"invalid_composition": 1}
    first = record.species_identifiers[0]
    changed = replace(
        record,
        species_identifiers=record.species_identifiers
        + ((first[0], "smiles", first[2].model_copy(update={"raw": "C"})),),
    )
    assert export_idt([changed])[1]["refused"] == {"no_confident_smiles": 1}


@pytest.mark.parametrize("fraction", ["1E-999", "-1E-999", "1E-324"])
def test_nonzero_source_fractions_that_underflow_are_refused(fraction: str) -> None:
    pytest.importorskip("rdkit")
    record = parse_idt_record(chemked_history(), "history.yaml")
    composition = record.envelope.composition
    components = tuple(
        c.model_copy(update={"amount": c.amount.model_copy(update={"canonical_decimal_value": n})})
        for c, n in zip(composition.components, (fraction, "0", "1"), strict=True)
    )
    altered = replace(
        record,
        envelope=record.envelope.model_copy(
            update={"composition": composition.model_copy(update={"components": components})}
        ),
    )
    payload, report = export_idt([altered])
    assert payload["points"] == []
    assert report["refused"] == {"invalid_composition": 1}


@pytest.mark.parametrize("column", ["time", "volume"])
@pytest.mark.parametrize("same_id", [True, False])
def test_duplicate_history_columns_are_refused_before_mapping(column: str, same_id: bool) -> None:
    import copy
    import xml.etree.ElementTree as ET

    record, raw = respecth_record()
    root = ET.fromstring(raw)
    group = root.findall("dataGroup")[1]
    duplicate = copy.deepcopy(next(p for p in group.findall("property") if p.get("name") == column))
    if not same_id:
        duplicate.set("id", "x99")
    group.append(duplicate)
    with pytest.raises(RespecthRefusal) as caught:
        parse_respecth(ET.tostring(root), load_manifest().archives[0], record.archive.member_path)
    assert caught.value.reason.value == "history_invalid"


@pytest.mark.parametrize("column", ["time", "volume"])
def test_duplicate_history_columns_cannot_replay(column: str) -> None:
    import copy
    import hashlib
    import xml.etree.ElementTree as ET

    record, raw = respecth_record()
    root = ET.fromstring(raw)
    group = root.findall("dataGroup")[1]
    group.append(copy.deepcopy(next(p for p in group.findall("property") if p.get("name") == column)))
    altered = ET.tostring(root)
    graph = record.envelope.source_graph
    nodes = tuple(node.model_copy(update={"sha256": hashlib.sha256(altered).hexdigest()}) for node in graph.nodes)
    envelope = record.envelope.model_copy(update={"source_graph": graph.model_copy(update={"nodes": nodes})})
    report = replay_respecth(record.model_copy(update={"envelope": envelope}), altered)
    assert not report.verified
    assert any("history_invalid" in finding for finding in report.findings)


@pytest.mark.parametrize("derived", [False, True])
def test_cli_derived_labels_are_opt_in(tmp_path: Path, monkeypatch, derived: bool) -> None:
    pytest.importorskip("rdkit")
    record, _ = respecth_record()
    monkeypatch.setattr(t3_export, "load_records", lambda **_: ((record,), {}))
    output = tmp_path / "output.yaml"
    args = ["data", "export-t3", "--history-only", "--offline", "--output", str(output)]
    if derived:
        args.append("--derived-labels")
    assert main(args) == 0
    points = yaml.safe_load(output.read_text())["points"]
    assert points
    assert all(("temperature" in point and "pressure" in point) == derived for point in points)


def test_respecth_drive_relative_member_lookup_and_iteration_are_refused() -> None:
    import io
    import zipfile

    from carmel.services import respecth_archive

    for name in ("C:relative.yaml", "C:relative.xml"):
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr(name, b"<experiment/>")
        with pytest.raises(respecth_archive.ArchiveIntegrityError, match="unsafe"):
            respecth_archive.read_member(bundle.getvalue(), name, "0" * 64)
        if name.endswith(".xml"):
            with pytest.raises(respecth_archive.ArchiveIntegrityError, match="unsafe"):
                list(respecth_archive.iter_xml_members(bundle.getvalue()))


def test_export_uncertainty_preserves_symmetric_relative_and_absolute_but_refuses_asymmetry() -> None:
    from carmel.schemas.datasets import AbsenceReason, Absent, UncertaintyBasis

    pytest.importorskip("rdkit")
    record, _ = respecth_record()
    series = record.envelope.series[0]
    point = series.points[0]
    observation = point.observations[0]
    original = observation.uncertainty

    def export_with(uncertainty):
        p = point.model_copy(update={"observations": (observation.model_copy(update={"uncertainty": uncertainty}),)})
        envelope = record.envelope.model_copy(update={"series": (series.model_copy(update={"points": (p,)}),)})
        # Restrict a copy to the first grounded point; keep its linked history.
        r = record.model_copy(update={"envelope": envelope, "rcm_states": record.rcm_states[:1]})
        return export_idt([r])

    payload, report = export_with(original)
    assert not report["refused"]
    assert payload["points"][0]["uncertainty"]["value"] == pytest.approx(
        in_si(original.upper) * in_si(observation.value)
    )
    absolute = original.model_copy(
        update={"basis": UncertaintyBasis.ABSOLUTE, "upper": observation.value, "lower": observation.value}
    )
    assert export_with(absolute)[0]["points"][0]["uncertainty"]["value"] == in_si(observation.value)
    for updates in (
        {"basis": Absent(reason=AbsenceReason.NOT_REPORTED_HERE)},
        {"upper": Absent(reason=AbsenceReason.NOT_REPORTED_HERE)},
        {"lower": original.lower.model_copy(update={"canonical_decimal_value": "0.001"})},
    ):
        assert export_with(original.model_copy(update=updates))[1]["refused"] == {"t3_constraint": 1}


@pytest.mark.parametrize(
    "edit,reason",
    [
        ("time", "history_nonmonotone_time"),
        ("volume", "history_nonpositive_volume"),
        ("state", "history_missing_initial_state"),
        ("range", "rcm_thermo_unavailable"),
        ("species", "rcm_thermo_unavailable"),
    ],
)
def test_respecth_history_failures_are_typed(edit, reason) -> None:
    import xml.etree.ElementTree as ET

    record, raw = respecth_record()
    root = ET.fromstring(raw)
    group = root.findall("dataGroup")[1]
    if edit == "time":
        group.findall("dataPoint")[1].find("x4").text = "0"
    if edit == "volume":
        group.findall("dataPoint")[1].find("x5").text = "0"
    if edit in {"state", "range"}:
        root.findall("dataGroup")[0].find("dataPoint/x2").text = "0" if edit == "state" else "100"
    if edit == "species":
        root.find("commonProperties/property/component/speciesLink").set("preferredKey", "CH4")
    with pytest.raises(RespecthRefusal) as caught:
        parse_respecth(ET.tostring(root), load_manifest().archives[0], record.archive.member_path)
    assert caught.value.reason.value == reason


def test_respecth_owner_validates_history_links_and_initial_states() -> None:
    record, _ = respecth_record()
    data = record.model_dump()
    for change in ("states", "missing_history", "overlap", "uncovered", "stated_state", "apparatus"):
        import copy

        edited = copy.deepcopy(data)
        if change == "states":
            edited["rcm_states"] = []
        elif change == "missing_history":
            edited["rcm_conditions"]["histories"][0]["history"] = None
        elif change == "overlap":
            edited["rcm_conditions"]["histories"] *= 2
        elif change == "uncovered":
            edited["rcm_conditions"]["histories"][0]["point_link"]["raw"] = "1"
        elif change == "stated_state":
            edited["end_of_compression_basis"] = "stated"
        elif change == "apparatus":
            edited["apparatus"]["device_class"] = "shock_tube"
        with pytest.raises(ValueError):
            type(record).model_validate(edited)


def test_replay_refuses_identity_association_and_missing_initial_state_forgery() -> None:
    record, raw = respecth_record()
    assert not replay_respecth(record, b"x" * (4 * 1024 * 1024 + 1)).verified
    for fields in (
        {"species_identifiers": ()},
        {"species_identifiers": tuple(reversed(record.species_identifiers))},
        {"rcm_states": ()},
        {"end_of_compression_basis": "stated"},
    ):
        assert not replay_respecth(record.model_copy(update=fields), raw).verified
    # Older postcompression records can omit the defaulted identifier extension.
    old, raw = respecth_record("x40001058_19.xml")
    assert replay_respecth(old.model_copy(update={"species_identifiers": ()}), raw).verified


def test_chemked_queries_use_stated_compressed_labels_and_skip_derived() -> None:
    from carmel.services.chemked_query import point_conditions

    record = parse_idt_record(chemked_history(), "history.yaml")
    assert point_conditions(record) == ((800, 10),)
    doc = yaml.safe_load(chemked_history())
    del doc["datapoints"][0]["compressed-temperature"]
    del doc["datapoints"][0]["compressed-pressure"]
    derived = parse_idt_record(yaml.safe_dump(doc).encode(), "history.yaml")
    assert point_conditions(derived) == ()


@pytest.mark.parametrize(
    "path",
    [
        "../escape.xml",
        "/absolute.xml",
        "C:/absolute.xml",
        "C:relative.yaml",
        "C:relative.xml",
        "dir\\member.xml",
        "//server/share/member.xml",
    ],
)
def test_pinned_source_paths_refuse_cross_platform_escape(tmp_path, path) -> None:
    import io
    import json
    import zipfile

    from carmel.services import chemked_archive, respecth_archive

    payload = json.loads(Path("carmel/data/chemked_manifest.json").read_text())
    payload["files"][0]["path"] = path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(chemked_archive.ManifestError):
        chemked_archive.load_manifest(manifest)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(path, b"<experiment/>")
    if path.endswith(".xml"):
        with pytest.raises(respecth_archive.ArchiveIntegrityError, match="unsafe"):
            list(respecth_archive.iter_xml_members(bundle.getvalue()))
    with pytest.raises(respecth_archive.ArchiveIntegrityError, match="unsafe"):
        respecth_archive.read_member(bundle.getvalue(), path, "0" * 64)


def test_respecth_flat_history_is_precisely_refused() -> None:
    import xml.etree.ElementTree as ET

    _, raw = respecth_record()
    root = ET.fromstring(raw)
    for cell in root.findall("dataGroup")[1].findall("dataPoint/x5"):
        cell.text = "1"
    with pytest.raises(RespecthRefusal) as caught:
        parse_respecth(ET.tostring(root), load_manifest().archives[0], "flat.xml")
    assert caught.value.reason.value == "history_no_compression"


@pytest.mark.parametrize("field", ["compression-time", "compressed-temperature", "compressed-pressure"])
@pytest.mark.parametrize("invalid", [{}, [], ["1 s", "unmodelled metadata"], [False]])
def test_optional_history_quantities_fail_closed(field, invalid) -> None:
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"][0][field] = invalid
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "bad.yaml")
    assert caught.value.reason.value == "history_invalid"


def test_record_rejects_unknown_apparatus() -> None:
    record = parse_idt_record(chemked_history(), "history.yaml")
    with pytest.raises(ValueError, match="apparatus"):
        replace(record, apparatus="unmapped")
