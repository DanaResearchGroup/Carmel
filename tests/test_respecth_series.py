# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The ReSpecTh flame-speed and speciation lane: real pinned members -> DatasetEnvelope ->
replay, the typed refusals, and ``carmel data find --kind {lbv,jsr,outlet,profile}``.

Every fixture under ``tests/fixtures/respecth/`` is an unmodified member of a pinned archive
(see that directory's README for each member's archive and sha256). Every value asserted below
was read by hand off the fixture XML."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from decimal import Decimal
from pathlib import Path

import pytest

import Carmel
from Carmel import main
from carmel.schemas.campaign import ReactorType
from carmel.schemas.datasets import (
    Absent,
    AxisRole,
    ComponentRole,
    Composition,
    Coordinate,
    MeasuredValue,
    Observation,
    UncertaintyBasis,
    UncertaintyKind,
)
from carmel.services import units
from carmel.services.respecth import RespecthRefusal, RespecthRefusalReason, evaluate_xpath
from carmel.services.respecth_archive import PinnedArchive, cached_archive_path, load_manifest
from carmel.services.respecth_query import QUERY_KINDS, find_records, load_records
from carmel.services.respecth_series import (
    APPARATUS_DEVICE_CLASSES,
    EXPERIMENT_KINDS,
    RespecthExperimentKind,
    RespecthSeriesRecord,
    SpeciesIdentity,
    TimeshiftType,
    parse_series_record,
    read_experiment_type,
    replay_series_record,
)
from carmel.services.units import QuantityKind

FIXTURES = Path(__file__).parent / "fixtures" / "respecth"
_H2 = "H2_indirect_v2_3.zip"
_SYNGAS = "syngas_indirect_v2_3.zip"

#: Member -> (archive, sha256), read off the pinned archives and recorded in the fixture README.
MEMBERS = {
    "x20000040d.xml": (_H2, "fd76c40663e962eeb88356da833cbec786ec0038a86340315887a606aec36e07"),
    "x20000054.xml": (_H2, "9aba880466a22f406e74b91b4d16014a2f4a18c5a2de632ecc7f1d581fc50bc5"),
    "x20001222.xml": (_SYNGAS, "c4e7cfd32f2b6f209f2315d6e466bb67d7517f214ebae5c8c6e36c02760ba9db"),
    "g00000005psr.xml": (_H2, "02df814ca8a5a799b2bce47b0f0e27be768c4cc25b22815026639b6a07df5961"),
    "x30000017.xml": (_H2, "5e4db2c4f389a3989578831d0d008c44b3a5c22d1b4817a2a810ae6407c122be"),
    "x30000029.xml": (_H2, "eba547816918758ed516a84bffd04c9b86fc47c2a75a6f089e9e59047862a9cf"),
    "x50001004.xml": (_SYNGAS, "65b89665f1410c440a51f4b06be9fc480741841622188def78adbc97e0d27e19"),
    "x20000074burn.xml": (_H2, "54a7492582f58a57b37bad1740d187c1fb0fcce4e909641c3505c8bc1dd04acd"),
}
_MAPPED = [name for name in MEMBERS if name != "x20000074burn.xml"]
_LBV_PER_POINT = "x20000040d.xml"
_LBV_COMMON = "x20000054.xml"
_LBV_BARS = "x20001222.xml"
_JSR = "g00000005psr.xml"
_PROFILE = "x30000017.xml"
_OUTLET = "x30000029.xml"
_OUTLET_ST = "x50001004.xml"
_BURNER = "x20000074burn.xml"
_R = RespecthRefusalReason


def _member(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _archive(name: str) -> PinnedArchive:
    by_name = {archive.name: archive for archive in load_manifest().archives}
    return by_name[MEMBERS[name][0]]


def _parse(name: str, data: bytes | None = None) -> RespecthSeriesRecord:
    return parse_series_record(_member(name) if data is None else data, _archive(name), name)


def _values(record: RespecthSeriesRecord, axis_id: str) -> list[MeasuredValue]:
    """Every point's value on ``axis_id`` (a constant repeats), in point order."""
    series = record.envelope.series[0]
    constant = next((c.value for c in series.constants if c.axis_id == axis_id), None)
    out: list[MeasuredValue] = []
    for point in series.points:
        cells: list[Coordinate | Observation] = [*point.coordinates, *point.observations]
        value = constant if constant is not None else next(c.value for c in cells if c.axis_id == axis_id)
        assert isinstance(value, MeasuredValue)
        out.append(value)
    return out


def _raw(record: RespecthSeriesRecord, axis_id: str) -> list[tuple[str, str]]:
    return [(value.raw_text, str(value.unit_raw)) for value in _values(record, axis_id)]


def _roles(record: RespecthSeriesRecord) -> dict[str, AxisRole]:
    return {axis.axis_id: axis.role for axis in record.envelope.series[0].axes}


def _replays(name: str) -> None:
    record = _parse(name)
    report = replay_series_record(record, _member(name))
    assert report.verified, report.findings
    assert report.checked > len(record.envelope.series[0].points)


class TestFixturesArePinnedMembers:
    @pytest.mark.parametrize("name", sorted(MEMBERS))
    def test_fixture_bytes_hash_to_the_recorded_member_sha256(self, name: str) -> None:
        assert hashlib.sha256(_member(name)).hexdigest() == MEMBERS[name][1]

    @pytest.mark.parametrize("name", _MAPPED)
    def test_every_value_replays_to_the_fixture_bytes(self, name: str) -> None:
        _replays(name)

    @pytest.mark.parametrize("name", _MAPPED)
    def test_the_envelope_binds_table_v3_and_pins_the_member(self, name: str) -> None:
        record = _parse(name)
        assert [table.sha256 for table in record.envelope.conversion_tables] == [units.TABLE_V3.sha256]
        assert record.member_sha256 == MEMBERS[name][1]
        assert record.archive.archive_name == MEMBERS[name][0]
        assert record.citation_doi == f"10.24388/{name.removesuffix('.xml')}"


class TestLaminarBurningVelocity:
    def test_per_point_mixture_in_mm_per_second(self) -> None:
        record = _parse(_LBV_PER_POINT)
        assert record.kind is RespecthExperimentKind.LAMINAR_BURNING_VELOCITY
        assert len(record.envelope.series[0].points) == 4
        assert _raw(record, "laminar_burning_velocity") == [
            ("1640", "mm/s"),
            ("1810", "mm/s"),
            ("1970", "mm/s"),
            ("2210", "mm/s"),
        ]
        assert _raw(record, "x_h2")[0] == ("6.539792e-001", "mole fraction")
        assert _raw(record, "x_ar")[3] == ("1.891892e-001", "mole fraction")
        assert _raw(record, "temperature") == [("298", "K")] * 4
        velocity = _values(record, "laminar_burning_velocity")[0]
        converted = units.convert(
            velocity.canonical_decimal_value,
            quantity=QuantityKind.VELOCITY,
            from_unit=velocity.unit_normalized,
            to_unit="m/s",
            table=units.TABLE_V3,
        )
        assert Decimal(converted.exact) == Decimal("1.64")

    def test_the_mixture_is_a_coordinate_and_each_points_composition(self) -> None:
        record = _parse(_LBV_PER_POINT)
        assert _roles(record) == {
            "laminar_burning_velocity": AxisRole.OBSERVATION,
            "pressure": AxisRole.CONSTANT,
            "temperature": AxisRole.CONSTANT,
            "x_ar": AxisRole.COORDINATE,
            "x_h2": AxisRole.COORDINATE,
            "x_o2": AxisRole.COORDINATE,
        }
        assert isinstance(record.envelope.composition, Absent)
        composition = record.envelope.series[0].points[1].composition
        assert isinstance(composition, Composition)
        assert [(c.species_raw_name, c.amount.raw_text, c.role) for c in composition.components] == [
            ("Ar", "2.405063e-001", ComponentRole.DILUENT),
            ("H2", "6.835443e-001", ComponentRole.FUEL),
            ("O2", "7.594937e-002", ComponentRole.OXIDIZER),
        ]
        assert isinstance(composition.equivalence_ratio, Absent)
        assert record.species("x_h2").key == ("H2", "1S/H2/h1H", "1333-74-0", "[HH]")

    def test_per_point_inchi_only_mixture_components_use_their_inchi_names(self) -> None:
        data = _member(_LBV_PER_POINT)
        data = data.replace(
            b'<speciesLink preferredKey="H2"  CAS="1333-74-0"  InChI="1S/H2/h1H"   '
            b'SMILES="[HH]"  chemName="hydrogen" />',
            b'<speciesLink InChI="1S/H2/h1H" />',
        )
        data = data.replace(
            b'<speciesLink preferredKey="O2"  CAS="7782-44-7"  InChI="1S/O2/c1-2"  '
            b'SMILES="O=O"   chemName="oxygen"   />',
            b'<speciesLink InChI="1S/O2/c1-2" />',
        )
        data = data.replace(
            b'<speciesLink preferredKey="Ar"  CAS="7440-37-1"  InChI="1S/Ar"       '
            b'SMILES="[Ar]"  chemName="argon"    />',
            b'<speciesLink InChI="1S/Ar" />',
        )
        record = _parse(_LBV_PER_POINT, data)
        composition = record.envelope.series[0].points[0].composition
        assert isinstance(composition, Composition)
        assert {component.species_raw_name for component in composition.components} == {
            "1S/H2/h1H",
            "1S/O2/c1-2",
            "1S/Ar",
        }

    def test_modes_uncertainties_and_labels(self) -> None:
        record = _parse(_LBV_PER_POINT)
        assert record.apparatus.device_class is ReactorType.FLAME
        assert [mode.raw for mode in record.apparatus.modes_raw] == ["premixed", "laminar"]
        (observation,) = record.envelope.series[0].points[0].observations
        assert not isinstance(observation.uncertainty, Absent)
        assert observation.uncertainty.kind is UncertaintyKind.STD_DEV
        assert observation.uncertainty.basis is UncertaintyBasis.ABSOLUTE
        assert isinstance(observation.uncertainty.upper, MeasuredValue)
        assert (observation.uncertainty.upper.raw_text, observation.uncertainty.upper.unit_raw) == ("30.3", "cm/s")
        temperature = next(c for c in record.envelope.series[0].constants if c.axis_id == "temperature")
        assert not isinstance(temperature.uncertainty, Absent)
        assert temperature.uncertainty.kind is UncertaintyKind.UNKNOWN
        assert isinstance(temperature.uncertainty.upper, MeasuredValue)
        assert temperature.uncertainty.upper.raw_text == "3"
        assert [(s.axis_id, s.name_raw.raw) for s in record.uncertainty_statements] == [
            ("laminar_burning_velocity", "evaluated standard deviation"),
            ("temperature", "uncertainty"),
        ]
        labels = {axis.axis_id: axis.label_raw for axis in record.envelope.series[0].axes}
        assert labels["x_h2"] == "[H2]"
        assert labels["laminar_burning_velocity"] == "laminar burning velocity"
        assert isinstance(record.timeshift, Absent)

    def test_the_common_mixture_encoding(self) -> None:
        """The pressure-sweep records state the mixture once, in commonProperties."""
        record = _parse(_LBV_COMMON)
        assert len(record.envelope.series[0].points) == 1
        assert _raw(record, "laminar_burning_velocity") == [("58.2", "cm/s")]
        assert _raw(record, "pressure") == [("5.0", "atm")]
        assert _raw(record, "temperature") == [("295", "K")]
        composition = record.envelope.composition
        assert isinstance(composition, Composition)
        assert [(c.species_raw_name, c.amount.raw_text) for c in composition.components] == [
            ("H2", "0.1305"),
            ("He", "0.6519"),
            ("O2", "0.2175"),
        ]
        assert isinstance(record.envelope.series[0].points[0].composition, Absent)
        assert record.species_columns == ()
        assert record.paper_doi == "10.1016/j.proci.2010.05.021"

    def test_a_per_point_reported_uncertainty_column_is_kept_beside_the_envelope(self) -> None:
        record = _parse(_LBV_BARS)
        assert len(record.envelope.series[0].points) == 4
        assert _raw(record, "laminar_burning_velocity") == [
            ("153.1", "cm/s"),
            ("55.0", "cm/s"),
            ("33.2", "cm/s"),
            ("25.4", "cm/s"),
        ]
        assert _raw(record, "pressure")[1] == ("15", "atm")
        (column,) = record.reported_uncertainties
        assert column.axis_id == "laminar_burning_velocity"
        assert column.reference_raw.raw == "Sl"
        assert [(value.point_id, value.value.raw_text) for value in column.values] == [
            ("p0001", "7.7"),
            ("p0002", "2.8"),
            ("p0003", "1.7"),
            ("p0004", "1.3"),
        ]
        observation = record.envelope.series[0].points[0].observations[0]
        assert not isinstance(observation.uncertainty, Absent)
        assert isinstance(observation.uncertainty.upper, MeasuredValue)
        assert observation.uncertainty.upper.raw_text == "3.38"
        assert [mode.raw for mode in record.apparatus.modes_raw] == ["premixed", "laminar", "OPF"]
        composition = record.envelope.composition
        assert isinstance(composition, Composition)
        co2 = next(c for c in composition.components if c.species_raw_name == "CO2")
        assert isinstance(co2.role, Absent)


class TestSpeciation:
    def test_jet_stirred_reactor_several_species_share_one_axis(self) -> None:
        record = _parse(_JSR)
        assert record.kind is RespecthExperimentKind.JET_STIRRED_REACTOR
        assert record.apparatus.device_class is ReactorType.JSR
        assert len(record.envelope.series[0].points) == 6
        assert _roles(record) == {
            "pressure": AxisRole.CONSTANT,
            "residence_time": AxisRole.CONSTANT,
            "temperature": AxisRole.COORDINATE,
            "volume": AxisRole.CONSTANT,
            "x_h2": AxisRole.OBSERVATION,
            "x_h2o": AxisRole.OBSERVATION,
        }
        assert _raw(record, "temperature")[0] == ("850", "K")
        assert _raw(record, "x_h2")[0] == ("0.00907", "mole fraction")
        assert _raw(record, "x_h2o")[5] == ("0.00609", "mole fraction")
        assert _raw(record, "residence_time")[0] == ("1000", "ms")
        assert _raw(record, "volume")[0] == ("30", "cm3")
        assert record.species("x_h2o").key == ("H2O", "1S/H2O/h1H2", "7732-18-5", "O")
        water = next(o for o in record.envelope.series[0].points[0].observations if o.axis_id == "x_h2o")
        assert not isinstance(water.uncertainty, Absent)
        assert isinstance(water.uncertainty.upper, MeasuredValue)
        assert water.uncertainty.upper.raw_text == "9.1e-5"
        statement = next(s for s in record.uncertainty_statements if s.axis_id == "x_h2o")
        assert isinstance(statement.species, SpeciesIdentity)
        assert statement.species.display == "H2O"

    def test_a_time_resolved_profile_is_a_series_whose_coordinate_is_time(self) -> None:
        record = _parse(_PROFILE)
        assert record.kind is RespecthExperimentKind.CONCENTRATION_TIME_PROFILE
        assert record.apparatus.device_class is ReactorType.PFR
        assert len(record.envelope.series[0].points) == 6
        assert _roles(record)["time"] is AxisRole.COORDINATE
        assert _raw(record, "time")[3] == ("2.622000e-002", "s")
        assert _raw(record, "x_h2")[3] == ("6.480000e-005", "mole fraction")
        assert _raw(record, "temperature")[0] == ("935", "K")
        assert _raw(record, "pressure")[0] == ("2.55", "atm")
        assert not isinstance(record.timeshift, Absent)
        assert (record.timeshift.target_raw.raw, record.timeshift.type) == ("H2", TimeshiftType.HALF_DECREASE)
        assert isinstance(record.timeshift.amount, Absent)

    def test_outlet_flow_reactor(self) -> None:
        record = _parse(_OUTLET)
        assert record.kind is RespecthExperimentKind.OUTLET_CONCENTRATION
        assert record.apparatus.device_class is ReactorType.PFR
        assert len(record.envelope.series[0].points) == 9
        assert _raw(record, "temperature")[0] == ("702.5", "K")
        assert _raw(record, "residence_time")[0] == ("7.964", "s")
        assert _raw(record, "x_h2")[8] == ("4.550000e-004", "mole fraction")
        assert _raw(record, "pressure")[0] == ("50", "bar")

    def test_outlet_shock_tube_sweeps_pressure_too(self) -> None:
        record = _parse(_OUTLET_ST)
        assert record.apparatus.device_class is ReactorType.SHOCK_TUBE
        assert len(record.envelope.series[0].points) == 16
        assert {axis for axis, role in _roles(record).items() if role is AxisRole.COORDINATE} == {
            "pressure",
            "residence_time",
            "temperature",
        }
        assert _raw(record, "temperature")[0] == ("1128", "K")
        assert _raw(record, "x_co")[0] == ("0.0004767", "mole fraction")
        assert _raw(record, "pressure")[0] == ("499.5", "atm")
        assert [c.species.display for c in record.species_columns] == ["CO", "CO2", "O2"]


# --------------------------------------------------------------------------- refusals


def _refused(name: str, data: bytes | None = None) -> RespecthRefusal:
    with pytest.raises(RespecthRefusal) as caught:
        _parse(name, data)
    return caught.value


def _mutated(name: str, old: bytes, new: bytes) -> bytes:
    data = _member(name)
    assert old in data, old
    return data.replace(old, new, 1)


class TestTicketRefusals:
    """The three refusals the ticket names -- each typed, none with partial output."""

    def test_an_unidentifiable_species_column(self) -> None:
        data = _mutated(
            _JSR,
            b'preferredKey="H2O"  CAS="7732-18-5"  InChI="1S/H2O/h1H2"  SMILES="O"     chemName="water"    />\n'
            b"        </property>\n\n",
            b'chemName="water"    />\n        </property>\n\n',
        )
        refusal = _refused(_JSR, data)
        assert refusal.reason is _R.UNIDENTIFIED_SPECIES
        assert "names no species identifier" in refusal.detail

    def test_an_unknown_experiment_type(self) -> None:
        data = _mutated(_JSR, b"jet stirred reactor measurement", b"plasma reactor measurement")
        refusal = _refused(_JSR, data)
        assert refusal.reason is _R.UNMAPPED_EXPERIMENT_TYPE
        assert "plasma reactor measurement" in refusal.detail

    def test_a_unit_with_no_exact_si_scale(self) -> None:
        data = _mutated(_JSR, b'units="atm"', b'units="Torr"')
        refusal = _refused(_JSR, data)
        assert refusal.reason is _R.UNMAPPED_UNIT
        assert "Torr" in refusal.detail

    def test_the_real_burner_member_is_refused_by_type(self) -> None:
        refusal = _refused(_BURNER)
        assert refusal.reason is _R.UNMAPPED_EXPERIMENT_TYPE
        assert "mass flux" in refusal.detail

    def test_an_ignition_delay_member_is_not_this_parsers(self) -> None:
        refusal = parse_refusal(FIXTURES / "x10000001.xml")
        assert refusal.reason is _R.UNMAPPED_EXPERIMENT_TYPE


def parse_refusal(path: Path) -> RespecthRefusal:
    with pytest.raises(RespecthRefusal) as caught:
        parse_series_record(path.read_bytes(), _archive(_JSR), path.name)
    return caught.value


_JSR_H2O_LINK = (
    b'<speciesLink preferredKey="H2O"  CAS="7732-18-5"  InChI="1S/H2O/h1H2"  SMILES="O"     chemName="water"    />'
)


@pytest.mark.parametrize(
    ("name", "old", "new", "reason", "fragment"),
    [
        (
            _JSR,
            b"<major>2</major>\n        <minor>3</minor>",
            b"<major>3</major>\n        <minor>0</minor>",
            _R.UNSUPPORTED_FORMAT_VERSION,
            "only 2.x",
        ),
        (_JSR, b"<fileAuthor>", b"<surprise/><fileAuthor>", _R.UNMAPPED_PROPERTY, "top-level"),
        (
            _JSR,
            b"<kind>stirred reactor</kind>",
            b"<kind>stirred reactor</kind><mode>perfectly stirred</mode>",
            _R.UNMAPPED_APPARATUS,
            "perfectly stirred",
        ),
        (_LBV_PER_POINT, b"<mode>laminar</mode>", b"<mode>premixed</mode>", _R.UNMAPPED_APPARATUS, "premixed"),
        (_JSR, b'<property name="volume"', b'<property name="density"', _R.UNMAPPED_PROPERTY, "density"),
        (
            _JSR,
            b'<property name="residence time"                label="tau"',
            b'<property name="pressure"                label="tau"',
            _R.UNMAPPED_PROPERTY,
            "stated twice",
        ),
        (_JSR, b"<commonProperties>", b"<commonProperties><note/>", _R.UNMAPPED_PROPERTY, "<note>"),
        (
            _JSR,
            b'<property name="temperature"  id="x1"',
            b'<property name="velocity"  id="x1"',
            _R.UNMAPPED_PROPERTY,
            "velocity",
        ),
        (
            _JSR,
            b'<property name="temperature"  id="x1"',
            b'<property name="pressure"  id="x1"',
            _R.UNMAPPED_PROPERTY,
            "both as a common constant",
        ),
        (
            _JSR,
            b'label="[H2O]"  sourcetype="digitized"  units="mole fraction" >\n            ' + _JSR_H2O_LINK,
            b'label="[H2O]"  sourcetype="digitized"  units="mole fraction" >\n            '
            + _JSR_H2O_LINK.replace(b"H2O", b"H2", 1).replace(b"1S/H2O/h1H2", b"1S/H2/h1H"),
            _R.UNIDENTIFIED_SPECIES,
            "both map to axis 'x_h2'",
        ),
        (
            _JSR,
            _JSR_H2O_LINK + b"\n            <value>9.1e-5</value>",
            _JSR_H2O_LINK.replace(b"1S/H2O/h1H2", b"1S/H2O/h1H3") + b"\n            <value>9.1e-5</value>",
            _R.UNIDENTIFIED_SPECIES,
            "match no column exactly",
        ),
        (
            _JSR,
            _JSR_H2O_LINK + b"\n        </property>\n\n",
            _JSR_H2O_LINK + _JSR_H2O_LINK + b"\n        </property>\n\n",
            _R.UNIDENTIFIED_SPECIES,
            "2 speciesLink",
        ),
        (
            _JSR,
            _JSR_H2O_LINK + b"\n        </property>\n\n",
            _JSR_H2O_LINK.replace(b"/>", b'color="blue" />') + b"\n        </property>\n\n",
            _R.UNIDENTIFIED_SPECIES,
            "color",
        ),
        (
            _JSR,
            b'<property name="pressure"                      label="P"',
            b'<property name="altitude"                      label="P"',
            _R.UNMAPPED_PROPERTY,
            "altitude",
        ),
        (
            _OUTLET,
            b'<property name="pressure"                      label="P"  sourcetype="reported"   units="bar"'
            b"           ><value>50</value></property>",
            b"",
            _R.INCOMPLETE_RECORD,
            "no pressure",
        ),
        (
            _LBV_COMMON,
            b'<property name="laminar burning velocity"  id="x1"',
            b'<property name="temperature"  id="x1"',
            _R.UNMAPPED_PROPERTY,
            "both as a common constant",
        ),
        (
            _LBV_COMMON,
            b'<property name="pressure"                  id="x2"',
            b'<property name="time"                  id="x2"',
            _R.INCOMPLETE_RECORD,
            "no pressure",
        ),
        (
            _LBV_PER_POINT,
            b'<property name="pressure"',
            b'<property name="initial composition"><component>'
            b'<speciesLink preferredKey="H2" /><amount units="mole fraction">1</amount></component></property>'
            b'<property name="pressure"',
            _R.INCOMPLETE_RECORD,
            "exactly once",
        ),
        (
            _OUTLET,
            b'<property name="initial composition"',
            b'<property name="initial mixture"',
            _R.UNMAPPED_PROPERTY,
            "initial mixture",
        ),
        (
            _PROFILE,
            b'<property name="time"         id="x1"',
            b'<property name="residence time"         id="x1"',
            _R.INCOMPLETE_RECORD,
            "no time column",
        ),
        (_PROFILE, b'type="half decrease"', b'type="first rise"', _R.UNMAPPED_TIMESHIFT, "first rise"),
        (
            _PROFILE,
            b'type="half decrease"',
            b'type="half decrease" amount="0.5"',
            _R.UNMAPPED_TIMESHIFT,
            "does not define",
        ),
        (_PROFILE, b'type="half decrease"', b'type="relative decrease"', _R.INCOMPLETE_RECORD, "@amount"),
        (
            _PROFILE,
            b'type="half decrease"',
            b'type="relative decrease" amount="x"',
            _R.UNMAPPED_TIMESHIFT,
            "not a decimal",
        ),
        (_PROFILE, b'type="half decrease"', b'type="relative decrease" amount="1.5"', _R.UNMAPPED_TIMESHIFT, "(0, 1]"),
        (
            _JSR,
            b"</dataGroup>",
            b'</dataGroup>\n    <timeshift target="H2" type="half decrease"/>',
            _R.UNMAPPED_TIMESHIFT,
            "jsr record",
        ),
        (_JSR, b"</dataGroup>", b'</dataGroup>\n    <dataGroup id="dg2"/>', _R.UNMAPPED_DATA_GROUP, "found 2"),
        (
            _JSR,
            b'kind="absolute"  method="statistical scatter" >\n            <speciesLink preferredKey="H2" ',
            b'kind="guessed"  method="statistical scatter" >\n            <speciesLink preferredKey="H2" ',
            _R.UNMAPPED_PROPERTY,
            "guessed",
        ),
        (_JSR, b'reference="composition"', b'reference="temperature"', _R.UNMAPPED_PROPERTY, "reference='temperature'"),
        (_LBV_PER_POINT, b'reference="temperature"', b'reference="flow rate"', _R.UNMAPPED_PROPERTY, "flow rate"),
        (_LBV_PER_POINT, b'bound="plusminus"', b'bound="upper"', _R.UNMAPPED_PROPERTY, "bound 'upper'"),
        (
            _LBV_PER_POINT,
            b'reference="temperature"',
            b'reference="laminar burning velocity"',
            _R.UNMAPPED_PROPERTY,
            "names no condition",
        ),
        (_LBV_PER_POINT, b'reference="temperature"', b'reference="pressure"', _R.UNMAPPED_UNIT, "unit 'K'"),
        (_LBV_BARS, b'reference="Sl"', b'reference="Tu"', _R.UNMAPPED_PROPERTY, "reference 'Tu'"),
        (_LBV_BARS, b"<x3>7.7</x3>", b"", _R.INCOMPLETE_RECORD, "not exactly the columns"),
        (_LBV_BARS, b'units="cm/s"  reference="Sl"', b'units="Torr"  reference="Sl"', _R.UNMAPPED_UNIT, "Torr"),
        (
            _LBV_PER_POINT,
            b'<speciesLink preferredKey="Ar"  CAS="7440-37-1"  InChI="1S/Ar"       '
            b'SMILES="[Ar]"  chemName="argon"    />',
            b"<speciesLink />",
            _R.UNIDENTIFIED_SPECIES,
            "names no species identifier",
        ),
        (
            _JSR,
            b"<dataPoint> <x1>850</x1>",
            b"<dataPoint> <x1>850</x1><x1>851</x1>",
            _R.INCOMPLETE_RECORD,
            "not exactly",
        ),
        (_JSR, b"<x2>0.00907</x2>", b"<x2>0.009O7</x2>", _R.UNREADABLE_VALUE, "0.009O7"),
        (_JSR, b"<value>9.1e-5</value>", b"<value>-9.1e-5</value>", _R.SCHEMA_REJECTED, "strictly positive"),
        (
            _LBV_PER_POINT,
            b"<kind>flame</kind>",
            b"<kind>flame</kind><vendor>x</vendor>",
            _R.UNMAPPED_APPARATUS,
            "vendor",
        ),
        (_JSR, b"</experiment>", b"", _R.MALFORMED_XML, "well-formed"),
    ],
)
def test_each_refusal_branch(name: str, old: bytes, new: bytes, reason: RespecthRefusalReason, fragment: str) -> None:
    refusal = _refused(name, _mutated(name, old, new))
    assert refusal.reason is reason, refusal
    assert fragment in str(refusal)


def test_a_root_other_than_experiment_is_refused() -> None:
    refusal = _refused(_JSR, _member(_JSR).replace(b"experiment>", b"record>"))
    assert refusal.reason is _R.MALFORMED_XML
    assert "<record>" in refusal.detail


def test_two_deviations_for_one_species_are_refused() -> None:
    hydrogen = (
        b'<speciesLink preferredKey="H2"   CAS="1333-74-0"  InChI="1S/H2/h1H"    SMILES="[HH]"  chemName="hydrogen" />'
    )
    data = _mutated(
        _JSR, _JSR_H2O_LINK + b"\n            <value>9.1e-5</value>", hydrogen + b"\n            <value>9.1e-5</value>"
    )
    refusal = _refused(_JSR, data)
    assert refusal.reason is _R.UNMAPPED_PROPERTY
    assert "two uncertainties are stated for 'x_h2'" in refusal.detail


def test_a_group_without_rows_is_refused() -> None:
    data = _member(_JSR)
    start = data.index(b"<dataPoint>")
    end = data.index(b"</dataGroup>")
    refusal = _refused(_JSR, data[:start] + data[end:])
    assert refusal.reason is _R.INCOMPLETE_RECORD
    assert "no data points" in refusal.detail


def test_duplicate_data_column_ids_are_refused_without_partial_output() -> None:
    data = _member(_LBV_COMMON)
    data = data.replace(
        b'<property name="pressure"                  id="x2"',
        b'<property name="pressure"                  id="x1"',
    )
    data = data.replace(b"<x2>5.0</x2>", b"")
    data = data.replace(b"<x2>10.0</x2>", b"")
    refusal = _refused(_LBV_COMMON, data)
    assert refusal.reason is _R.INCOMPLETE_RECORD
    assert "x1" in refusal.detail


def test_a_speciation_record_needs_a_species_column_and_a_mixture() -> None:
    data = _member(_OUTLET)
    start = data.index(b'<property name="composition"     id="x3"')
    end = data.index(b"</property>", start) + len(b"</property>")
    stripped = data[:start] + data[end:]
    stripped = b"".join(stripped.split(b"<x3>")[0:1]) + b"".join(
        part.split(b"</x3>", 1)[1] for part in stripped.split(b"<x3>")[1:]
    )
    assert _refused(_OUTLET, stripped).reason is _R.INCOMPLETE_RECORD
    lbv = _mutated(
        _LBV_COMMON, b'<property name="laminar burning velocity"  id="x1"', b'<property name="temperature"  id="x1"'
    )
    assert _refused(_LBV_COMMON, lbv).reason is _R.UNMAPPED_PROPERTY


def test_no_lbv_column_and_no_mixture_are_incomplete() -> None:
    data = _member(_LBV_COMMON)
    start = data.index(b'<property name="laminar burning velocity"  id="x1"')
    end = data.index(b"/>", start) + 2
    no_column = data[:start] + data[end:]
    no_column = no_column.replace(b"<x1>58.2</x1>", b"")
    refusal = _refused(_LBV_COMMON, no_column)
    assert refusal.reason is _R.INCOMPLETE_RECORD
    assert "no laminar burning velocity column" in refusal.detail
    start = data.index(b'<property name="initial composition"')
    end = data.index(b"</property>", start) + len(b"</property>")
    refusal = _refused(_LBV_COMMON, data[:start] + data[end:])
    assert "exactly once" in refusal.detail
    speciation = _member(_OUTLET)
    start = speciation.index(b'<property name="initial composition"')
    end = speciation.index(b"</property>", start) + len(b"</property>")
    refusal = _refused(_OUTLET, speciation[:start] + speciation[end:])
    assert "no initial composition" in refusal.detail


def test_a_profile_without_a_timeshift_maps_with_it_absent() -> None:
    data = _mutated(_PROFILE, b'    <timeshift target="H2" type="half decrease"/>\n', b"")
    record = _parse(_PROFILE, data)
    assert isinstance(record.timeshift, Absent)
    relative = _parse(_PROFILE, _mutated(_PROFILE, b'type="half decrease"', b'type="relative decrease" amount="0.82"'))
    assert not isinstance(relative.timeshift, Absent)
    assert relative.timeshift.type is TimeshiftType.RELATIVE_DECREASE
    assert not isinstance(relative.timeshift.amount, Absent)
    assert relative.timeshift.amount.raw == "0.82"


def test_a_species_named_only_by_inchi_gets_a_column_axis_id() -> None:
    data = _mutated(
        _JSR,
        b'label="[H2O]"  sourcetype="digitized"  units="mole fraction" >\n            '
        b'<speciesLink preferredKey="H2O"  ',
        b'label="[H2O]"  sourcetype="digitized"  units="mole fraction" >\n            <speciesLink ',
    )
    data = data.replace(b'<speciesLink preferredKey="H2O"  CAS="7732-18-5"', b'<speciesLink CAS="7732-18-5"', 1)
    record = _parse(_JSR, data)
    assert record.species("x_col_x3").key == (None, "1S/H2O/h1H2", "7732-18-5", "O")
    assert record.species("x_col_x3").display == "1S/H2O/h1H2"
    assert replay_series_record(record, data).verified


def test_a_relative_standard_deviation_binds_as_a_relative_uncertainty() -> None:
    data = _mutated(
        _JSR,
        b'units="mole fraction"  reference="composition"  kind="absolute"  method="statistical scatter" >\n'
        b'            <speciesLink preferredKey="H2" ',
        b'units="unitless"  reference="composition"  kind="relative"  method="statistical scatter" >\n'
        b'            <speciesLink preferredKey="H2" ',
    )
    record = _parse(_JSR, data)
    hydrogen = next(o for o in record.envelope.series[0].points[0].observations if o.axis_id == "x_h2")
    assert not isinstance(hydrogen.uncertainty, Absent)
    assert hydrogen.uncertainty.basis is UncertaintyBasis.RELATIVE
    assert isinstance(hydrogen.uncertainty.upper, MeasuredValue)
    assert hydrogen.uncertainty.upper.quantity_kind is QuantityKind.RELATIVE_UNCERTAINTY


def test_the_apparatus_table_covers_only_known_kinds() -> None:
    assert {kind for kind, _, _ in APPARATUS_DEVICE_CLASSES} == set(RespecthExperimentKind)
    assert set(EXPERIMENT_KINDS.values()) == set(RespecthExperimentKind)
    with pytest.raises(ValueError, match="preferred key, InChI, CAS or SMILES"):
        SpeciesIdentity(
            preferred_key=Absent(reason="not_reported_here"),  # type: ignore[arg-type]
            inchi=Absent(reason="not_reported_here"),  # type: ignore[arg-type]
            cas=Absent(reason="not_reported_here"),  # type: ignore[arg-type]
            smiles=Absent(reason="not_reported_here"),  # type: ignore[arg-type]
            chem_name=Absent(reason="not_reported_here"),  # type: ignore[arg-type]
        )


def test_read_experiment_type() -> None:
    assert read_experiment_type(_member(_JSR)) == "jet stirred reactor measurement"
    with pytest.raises(RespecthRefusal):
        read_experiment_type(b"<experiment/>")


# --------------------------------------------------------------------------- replay


class TestReplayCatchesForgery:
    def test_bytes_that_do_not_hash_to_the_node(self) -> None:
        record = _parse(_JSR)
        report = replay_series_record(record, _member(_JSR) + b" ")
        assert not report.verified
        assert "hash to" in report.findings[0]

    def test_a_recorded_value_the_bytes_do_not_say(self) -> None:
        record = _parse(_JSR)
        forged = _parse(_JSR, _mutated(_JSR, b"<x2>0.00907</x2>", b"<x2>0.00999</x2>"))
        forged = forged.model_copy(
            update={"envelope": forged.envelope.model_copy(update={"source_graph": record.envelope.source_graph})}
        )
        report = replay_series_record(forged, _member(_JSR))
        assert not report.verified
        assert any("reads '0.00907', recorded '0.00999'" in finding for finding in report.findings)

    def test_a_mapped_fact_its_text_does_not_support(self) -> None:
        record = _parse(_JSR)
        forged = record.model_copy(
            update={"apparatus": record.apparatus.model_copy(update={"device_class": ReactorType.PFR})}
        )
        report = replay_series_record(forged, _member(_JSR))
        assert not report.verified
        assert report.findings == ("apparatus does not re-derive from the member",)

    def test_bytes_that_hash_right_but_no_longer_map(self) -> None:
        data = _mutated(_JSR, b'units="atm"', b'units="Torr"')
        record = _parse(_JSR)
        node = record.envelope.source_graph.nodes[0].model_copy(update={"sha256": hashlib.sha256(data).hexdigest()})
        graph = record.envelope.source_graph.model_copy(update={"nodes": (node,)})
        forged = record.model_copy(update={"envelope": record.envelope.model_copy(update={"source_graph": graph})})
        report = replay_series_record(forged, data)
        assert not report.verified
        assert "does not re-map" in report.findings[0]

    def test_locators_address_the_hand_read_cells(self) -> None:
        from xml.etree import ElementTree

        record = _parse(_PROFILE)
        root = ElementTree.fromstring(_member(_PROFILE))  # noqa: S314 - committed fixture
        value = _values(record, "x_h2")[3]
        assert value.value_ref.locator.kind.value == "xpath"
        assert evaluate_xpath(root, value.value_ref.locator.xpath) == "6.480000e-005"  # type: ignore[union-attr]


def test_a_placeholder_reference_doi_is_never_the_paper_doi() -> None:
    record = _parse(_JSR)
    assert record.paper_doi == "10.1021/ef800832q"
    placeholder = record.model_copy(
        update={"paper": record.paper.model_copy(update={"trust": "placeholder"})}  # type: ignore[union-attr]
    )
    assert placeholder.paper_doi is None
    assert record.model_copy(update={"paper": Absent(reason="unknown")}).paper_doi is None  # type: ignore[arg-type]


# --------------------------------------------------------------------------- query + CLI


def _fixture_zip(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            bundle.writestr(zipfile.ZipInfo(name, date_time=(2024, 8, 8, 13, 31, 0)), _member(name))
        bundle.writestr(zipfile.ZipInfo("broken.xml", date_time=(2024, 8, 8, 13, 31, 0)), b"<experiment>")
    return buffer.getvalue()


@pytest.fixture
def fixture_cache(tmp_path: Path) -> tuple[Path, Path]:
    """A manifest pinning one zip of every fixture member (plus the IDT ones), already cached."""
    data = _fixture_zip([*MEMBERS, "x10000001.xml"])
    sha = hashlib.sha256(data).hexdigest()
    path = cached_archive_path(tmp_path / "cache", sha)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "source": "respecth",
                "osf_node": "test",
                "doi": "10.0/test",
                "license": "CC-BY-4.0",
                "download_url_template": "http://127.0.0.1:9/{osf_file_id}",
                "archives": [
                    {
                        "name": "fixtures.zip",
                        "osf_path": "/test/fixtures.zip",
                        "osf_file_id": "0123456789abcdef01234567",
                        "osf_version": 1,
                        "size": len(data),
                        "sha256": sha,
                    }
                ],
            }
        )
    )
    return manifest, tmp_path / "cache"


def _find(fixture_cache: tuple[Path, Path], kind: str, *extra: str) -> int:
    manifest, cache = fixture_cache
    return main(
        ["data", "find", "--kind", kind, "--manifest", str(manifest), "--cache", str(cache), "--offline", *extra]
    )


_HEADER = "record_doi\tpaper_doi\tdevice\tfuels\tT_K\tP_bar\tpoints"


class TestDataFind:
    def test_lbv(self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
        assert _find(fixture_cache, "lbv") == 0
        assert capsys.readouterr().out.splitlines() == [
            _HEADER,
            "10.24388/x20000040d\t10.1016/S0010-2180(00)00229-7\tflame\tH2\t298-298\t1.013-1.013\t4/4",
            "10.24388/x20000054\t10.1016/j.proci.2010.05.021\tflame\tH2\t295-295\t5.066-5.066\t1/1",
            "10.24388/x20001222\t10.1016/j.proci.2014.07.047\tflame\tH2\t353-353\t5.066-25.33\t4/4",
            "3 matching dataset(s) of 3 mapped lbv records; 1 refused (malformed_xml 1)",
        ]

    def test_lbv_filters(self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
        assert _find(fixture_cache, "lbv", "--P", "10:30", "--T", "300:400", "--fuel", "H2") == 0
        assert capsys.readouterr().out.splitlines()[1:] == [
            "10.24388/x20001222\t10.1016/j.proci.2014.07.047\tflame\tH2\t353-353\t5.066-25.33\t3/4",
            "1 matching dataset(s) of 3 mapped lbv records; 1 refused (malformed_xml 1)",
        ]

    @pytest.mark.parametrize(
        ("kind", "rows"),
        [
            ("jsr", ["10.24388/g00000005psr\t10.1021/ef800832q\tjsr\tH2\t850-1150\t10.13-10.13\t6/6"]),
            (
                "outlet",
                [
                    "10.24388/x30000029\t10.1016/j.proci.2014.05.101\tpfr\tH2\t702.5-899.4\t50-50\t9/9",
                    "10.24388/x50001004\t10.1016/j.proci.2006.08.057\tshock_tube\tCO+H2\t1128-1414\t406.1-506.1\t16/16",
                ],
            ),
            (
                "profile",
                [
                    "10.24388/x30000017\t10.1002/(SICI)1097-4601(1999)31:2%3C113::AID-KIN5%3E3.0.CO;2-0\tpfr\tH2\t935-935\t2.584-2.584\t6/6"
                ],
            ),
        ],
    )
    def test_speciation_kinds(
        self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str], kind: str, rows: list[str]
    ) -> None:
        assert _find(fixture_cache, kind) == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == _HEADER
        assert lines[1:-1] == rows
        assert lines[-1] == (
            f"{len(rows)} matching dataset(s) of {len(rows)} mapped {kind} records; 1 refused (malformed_xml 1)"
        )

    @pytest.mark.parametrize("kind", ["lbv", "jsr", "outlet", "profile"])
    def test_a_filter_excluding_everything_is_empty_and_exit_zero(
        self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str], kind: str
    ) -> None:
        assert _find(fixture_cache, kind, "--T", "5000:6000") == 0
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == _HEADER
        assert len(lines) == 2
        assert lines[1].startswith("0 matching dataset(s) of ")

    def test_the_idt_kind_still_skips_every_other_type(
        self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _find(fixture_cache, "idt") == 0
        assert capsys.readouterr().out.splitlines()[-1] == (
            "1 matching dataset(s) of 1 mapped ignition-delay records; 1 refused (malformed_xml 1)"
        )

    def test_the_cli_kinds_are_the_query_kinds(self) -> None:
        parser = Carmel.create_parser()
        data = next(action for action in parser._subparsers._group_actions if action.dest == "command")  # type: ignore[union-attr]
        find = data.choices["data"]._subparsers._group_actions[0].choices["find"]  # type: ignore[attr-defined]
        (kind,) = (action for action in find._actions if action.dest == "kind")
        assert tuple(kind.choices) == QUERY_KINDS

    def test_load_records_refuses_an_unknown_kind(self, fixture_cache: tuple[Path, Path]) -> None:
        manifest, cache = fixture_cache
        with pytest.raises(ValueError):
            load_records(load_manifest(manifest), "flame-speed", cache_root=cache, download=False)

    def test_the_fuel_filter_reads_per_point_mixtures(self) -> None:
        record = _parse(_LBV_PER_POINT)
        (match,) = find_records((record,), fuel="H2")
        assert match.fuels == ("H2",)
        assert find_records((record,), fuel="CO") == []
