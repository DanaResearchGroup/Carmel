# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The ReSpecTh ignition-delay lane: real pinned members -> DatasetEnvelope -> replay, the
typed refusals, the pinned cache (against a local HTTP double -- no real network), and
``carmel data find``.

Every fixture under ``tests/fixtures/respecth/`` is an unmodified member of a pinned archive
(see that directory's README for each member's archive and sha256)."""

from __future__ import annotations

import hashlib
import http.server
import io
import json
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from Carmel import main
from carmel.schemas.campaign import ReactorType
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisRole,
    ComponentRole,
    DatasetEnvelope,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    UncertaintyBasis,
    UncertaintyKind,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.dataset_store import canonical_json_bytes
from carmel.services.respecth import (
    APPARATUS_DEVICE_CLASSES,
    ASSUMED_APPARATUS_MODES,
    ApparatusModeBasis,
    IgnitionCriterion,
    IgnitionTarget,
    ReferenceDoiTrust,
    RespecthIdtRecord,
    RespecthRefusal,
    RespecthRefusalReason,
    evaluate_xpath,
    parse_idt_record,
    replay_idt_record,
)
from carmel.services.respecth_archive import (
    ArchiveFetchError,
    ArchiveIntegrityError,
    ManifestError,
    PinnedArchive,
    RespecthManifest,
    cached_archive_path,
    fetch_archive,
    iter_xml_members,
    load_manifest,
    read_member,
)
from carmel.services.respecth_query import parse_window
from carmel.services.units import QuantityKind

FIXTURES = Path(__file__).parent / "fixtures" / "respecth"

#: Member sha256s, read off the pinned archives and recorded in the fixture README.
MEMBER_SHA256 = {
    "x00000070_p.xml": "e5fbc013960dec9e76b28c6476d7f30a3516f300ac7a11e3dc68280d94967314",
    "x10000001.xml": "f9a7f6703bff4a298d9fa99ff0e3db5d2550ce84946e3e37d3108ecf278af901",
    "x10000030_x.xml": "11576b4dd2559738ffc5dea6cf7456672a44896f36e569146ee05ef55bb32f86",
    "x40001039.xml": "4c0066cb7e54a98326f3d85f3e0dd1e7fbad4786444b988ab33d2ffd4cb00c21",
    "x40001058_19.xml": "b505031a16926bc1bb22746c50636696c5c0513621779e54e78769af4017f85d",
}


def _member(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _archive(name: str) -> PinnedArchive:
    by_name = {archive.name: archive for archive in load_manifest().archives}
    return by_name["syngas_indirect_v2_3.zip" if name.startswith("x4") else "H2_indirect_v2_3.zip"]


def _parse(name: str, data: bytes | None = None) -> RespecthIdtRecord:
    return parse_idt_record(_member(name) if data is None else data, _archive(name), name)


def _triples(record: RespecthIdtRecord) -> list[tuple[str, str, str, str, str, str]]:
    """``(T, T unit, P, P unit, IDT, IDT unit)`` verbatim per point, constants filled in."""
    series = record.envelope.series[0]
    constants = {c.axis_id: c.value for c in series.constants}
    rows = []
    for point in series.points:
        values = {**constants, **{c.axis_id: c.value for c in point.coordinates}}
        (observation,) = point.observations
        assert not isinstance(observation.value, Absent)
        rows.append(
            (
                values["temperature"].raw_text,
                str(values["temperature"].unit_raw),
                values["pressure"].raw_text,
                str(values["pressure"].unit_raw),
                observation.value.raw_text,
                str(observation.value.unit_raw),
            )
        )
    return rows


class TestFixturesArePinnedMembers:
    @pytest.mark.parametrize("name", sorted(MEMBER_SHA256))
    def test_fixture_bytes_hash_to_the_recorded_member_sha256(self, name: str) -> None:
        assert hashlib.sha256(_member(name)).hexdigest() == MEMBER_SHA256[name]

    def test_packaged_manifest_pins_both_archives_under_cc_by(self) -> None:
        manifest = load_manifest()
        assert manifest.license == "CC-BY-4.0"
        assert manifest.doi == "10.17605/OSF.IO/NBMZV"
        assert {(a.name, a.osf_file_id, a.osf_version, a.sha256) for a in manifest.archives} == {
            (
                "H2_indirect_v2_3.zip",
                "6716900181a7f78563a5e911",
                1,
                "7f7247f0c95dfe65cf784807b6fde5539d34f3fce43b47a779d4ad288b96ddda",
            ),
            (
                "syngas_indirect_v2_3.zip",
                "671691044a236f2bf52ecb24",
                1,
                "a14628e5300a7922f50bb354b95f46f1ae853eb6b31e2a33555ffab17812321b",
            ),
        }
        assert manifest.download_url(manifest.archives[0]) == (
            "https://files.osf.io/v1/resources/nbmzv/providers/osfstorage/6716900181a7f78563a5e911?version=1"
        )


class TestShockTubeRoundTrip:
    """``x00000070_p.xml``: reflected-shock tube, 3 points, P common at 64 atm."""

    def test_exact_points_and_hand_read_triples(self) -> None:
        record = _parse("x00000070_p.xml")
        assert len(record.envelope.series[0].points) == 3
        # Read by hand from the member's <dataGroup> and <commonProperties>.
        assert _triples(record) == [
            ("1279", "K", "64", "atm", "265.1", "us"),
            ("1314", "K", "64", "atm", "152.4", "us"),
            ("1344", "K", "64", "atm", "68.9", "us"),
        ]

    def test_every_value_replays_to_the_fixture_bytes(self) -> None:
        record = _parse("x00000070_p.xml")
        report = replay_idt_record(record, _member("x00000070_p.xml"))
        assert report.findings == ()
        assert report.verified
        assert report.checked == len(list(iter_source_refs(record)))

    def test_the_node_pins_the_member_inside_its_archive(self) -> None:
        record = _parse("x00000070_p.xml")
        (node,) = record.envelope.source_graph.nodes
        assert node.kind is SourceNodeKind.DATABASE_RECORD
        assert node.sha256 == MEMBER_SHA256["x00000070_p.xml"]
        assert not isinstance(node.origin, Absent)
        assert node.origin.archive_sha256 == _archive("x00000070_p.xml").sha256
        assert record.archive.osf_file_id == "6716900181a7f78563a5e911"
        assert record.envelope.series[0].source_form is SourceForm.STRUCTURED_RECORD

    def test_record_level_facts(self) -> None:
        record = _parse("x00000070_p.xml")
        assert record.citation_doi == "10.24388/x00000070"
        assert record.apparatus.device_class is ReactorType.SHOCK_TUBE
        assert (record.ignition.target, record.ignition.criterion) == (IgnitionTarget.OH, IgnitionCriterion.MAX_SLOPE)
        assert not isinstance(record.uncertainty, Absent)
        assert not isinstance(record.uncertainty.method_raw, Absent)
        assert record.uncertainty.method_raw.raw == "statistical scatter"
        assert dict(record.column_source_types)["ignition_delay"].raw == "digitized"

    def test_uncertainty_and_composition(self) -> None:
        record = _parse("x00000070_p.xml")
        for point in record.envelope.series[0].points:
            (observation,) = point.observations
            uncertainty = observation.uncertainty
            assert not isinstance(uncertainty, Absent)
            assert (uncertainty.kind, uncertainty.basis) == (UncertaintyKind.STD_DEV, UncertaintyBasis.RELATIVE)
            assert not isinstance(uncertainty.upper, Absent)
            assert uncertainty.upper.canonical_decimal_value == "0.18"
        composition = record.envelope.composition
        assert not isinstance(composition, Absent)
        assert [(c.species_raw_name, c.amount.raw_text, c.role) for c in composition.components] == [
            ("Ar", "0.9925", ComponentRole.DILUENT),
            ("H2", "0.0050", ComponentRole.FUEL),
            ("O2", "0.0025", ComponentRole.OXIDIZER),
        ]

    def test_placeholder_reference_doi_is_kept_but_never_cited(self) -> None:
        record = _parse("x00000070_p.xml")
        assert not isinstance(record.paper, Absent)
        assert record.paper.trust is ReferenceDoiTrust.PLACEHOLDER
        assert record.paper.doi.raw == "10.1016/s0082-0784(96)80289-x"
        assert not isinstance(record.paper.placeholder_evidence, Absent)
        assert "just for sorting purposes" in record.paper.placeholder_evidence.raw
        assert record.paper_doi is None

    def test_the_envelope_round_trips_through_its_identity_payload(self) -> None:
        # Byte-compared, not model-compared: the projection deliberately drops the
        # display-only ArchiveOrigin.member_display_path (the member sha256 is the identity,
        # and the record's ArchivePin keeps the path).
        envelope = _parse("x00000070_p.xml").envelope
        parsed = DatasetEnvelope.from_identity_payload(envelope.identity_payload())
        assert canonical_json_bytes(parsed.identity_payload()) == canonical_json_bytes(envelope.identity_payload())


class TestAssumedApparatusMode:
    """A shock tube that states no mode is mapped as reflected shock, and says so everywhere."""

    def test_real_member_maps_with_the_mode_assumed_not_stated(self) -> None:
        record = _parse("x10000030_x.xml")
        apparatus = record.apparatus
        assert apparatus.device_class is ReactorType.SHOCK_TUBE
        assert apparatus.mode_basis is ApparatusModeBasis.ASSUMED
        assert apparatus.assumed_mode == "reflected shock"
        assert isinstance(apparatus.mode_raw, Absent)
        assert apparatus.mode_raw.reason is AbsenceReason.NOT_REPORTED_HERE
        assert _triples(record)[0] == ("1.194743e+003", "K", "5", "bar", "2.232698e-002", "ms")
        assert len(record.envelope.series[0].points) == 7

    def test_replay_verifies_and_shows_the_assumption(self) -> None:
        report = replay_idt_record(_parse("x10000030_x.xml"), _member("x10000030_x.xml"))
        assert report.verified
        assert report.assumptions == (
            "apparatus mode 'reflected shock' is ASSUMED: /experiment/apparatus[1] states no mode",
        )

    def test_a_stated_mode_carries_no_assumption(self) -> None:
        record = _parse("x00000070_p.xml")
        assert record.apparatus.mode_basis is ApparatusModeBasis.STATED
        assert isinstance(record.apparatus.assumed_mode, Absent)
        assert replay_idt_record(record, _member("x00000070_p.xml")).assumptions == ()

    def test_an_empty_mode_element_is_no_stated_mode(self) -> None:
        data = _member("x00000070_p.xml").replace(b"reflected shock", b"")
        assert _parse("x00000070_p.xml", data).apparatus.mode_basis is ApparatusModeBasis.ASSUMED

    def test_an_assumed_mode_cannot_be_dressed_as_stated(self) -> None:
        apparatus = _parse("x10000030_x.xml").apparatus
        with pytest.raises(ValueError, match="mode_basis"):
            type(apparatus).model_validate({**apparatus.model_dump(), "mode_basis": ApparatusModeBasis.STATED})
        with pytest.raises(ValueError, match="mode_basis"):
            type(apparatus).model_validate(
                {**apparatus.model_dump(), "assumed_mode": Absent(reason=AbsenceReason.NOT_APPLICABLE)}
            )

    def test_replay_refuses_an_assumption_the_member_contradicts(self) -> None:
        # A stated mode relabelled as assumed: every ref still replays, but the member states one.
        record = _parse("x00000070_p.xml")
        forged = record.model_copy(
            update={
                "apparatus": record.apparatus.model_copy(
                    update={
                        "mode_raw": Absent(reason=AbsenceReason.NOT_REPORTED_HERE),
                        "mode_basis": ApparatusModeBasis.ASSUMED,
                        "assumed_mode": "reflected shock",
                    }
                )
            }
        )
        report = replay_idt_record(forged, _member("x00000070_p.xml"))
        assert not report.verified
        assert report.findings == ("apparatus mode is recorded as assumed, but the member states 'reflected shock'",)

    def test_every_assumed_key_is_a_mapped_modeless_key(self) -> None:
        for key in ASSUMED_APPARATUS_MODES:
            assert key[1] is None
            assert key in APPARATUS_DEVICE_CLASSES
            assert (key[0], ASSUMED_APPARATUS_MODES[key]) in APPARATUS_DEVICE_CLASSES


class TestRcmConditions:
    """RKD v2.5: an RCM record's P/T are the state at the START of its volume-time history --
    before compression or at its end. Only a history with no compression phase identifies them
    as end-of-compression conditions; anything else is refused, never mapped as the ignition state."""

    def test_end_of_compression_member_maps_with_grounded_evidence(self) -> None:
        record = _parse("x40001058_19.xml")
        conditions = record.rcm_conditions
        assert not isinstance(conditions, Absent)
        assert conditions.state == "end_of_compression"
        (history,) = conditions.histories
        assert (history.group_id.raw, history.point_link.raw) == ("dg2", "1")
        assert (history.first_volume.raw, history.minimum_volume.raw) == ("1.0000000", "1.0000000")
        assert history.first_volume.ref.locator.xpath == "/experiment/dataGroup[2]/dataPoint[1]/x9[1]"
        assert replay_idt_record(record, _member("x40001058_19.xml")).verified

    def test_shock_tubes_carry_no_rcm_evidence(self) -> None:
        assert isinstance(_parse("x00000070_p.xml").rcm_conditions, Absent)

    def test_real_mbar_member_is_refused_as_pre_compression(self) -> None:
        # x40001039: 586 mbar / 354 K, and a history that compresses from 1.0 to ~0.1 --
        # the pressure unit binds first (no unmapped_unit), then the conditions are refused.
        with pytest.raises(RespecthRefusal) as caught:
            _parse("x40001039.xml")
        assert caught.value.reason is RespecthRefusalReason.RCM_PRE_COMPRESSION_CONDITIONS
        assert "dg2" in caught.value.detail

    def test_a_compressing_history_is_refused(self) -> None:
        data = _member("x40001058_19.xml").replace(b"<x9>1.0034797</x9>", b"<x9>0.9034797</x9>")
        with pytest.raises(RespecthRefusal) as caught:
            _parse("x40001058_19.xml", data)
        assert caught.value.reason is RespecthRefusalReason.RCM_PRE_COMPRESSION_CONDITIONS

    def test_no_history_cannot_identify_the_conditions(self) -> None:
        data = _member("x40001058_19.xml")
        data = (
            data[: data.index(b'    <dataGroup id="dg2"')]
            + data[data.index(b"</dataGroup>", data.index(b'id="dg2"')) + 13 :]
        )
        with pytest.raises(RespecthRefusal) as caught:
            _parse("x40001058_19.xml", data)
        assert caught.value.reason is RespecthRefusalReason.RCM_CONDITIONS_UNIDENTIFIED

    def test_a_history_linked_to_another_point_leaves_this_one_unidentified(self) -> None:
        data = _member("x40001058_19.xml").replace(b'dataPointLink="1"', b'dataPointLink="2"')
        with pytest.raises(RespecthRefusal) as caught:
            _parse("x40001058_19.xml", data)
        assert caught.value.reason is RespecthRefusalReason.RCM_CONDITIONS_UNIDENTIFIED

    def test_replay_re_derives_the_verdict(self) -> None:
        # The evidence names the first and minimum cells; a member whose history compresses
        # elsewhere must not replay as end-of-compression even though those two cells still match.
        record = _parse("x40001058_19.xml")
        data = _member("x40001058_19.xml").replace(b"<x9>1.0046817</x9>", b"<x9>0.5046817</x9>")
        forged = record.model_copy(
            update={
                "envelope": record.envelope.model_copy(
                    update={
                        "source_graph": record.envelope.source_graph.model_copy(
                            update={
                                "nodes": (
                                    record.envelope.source_graph.nodes[0].model_copy(
                                        update={"sha256": hashlib.sha256(data).hexdigest()}
                                    ),
                                )
                            }
                        )
                    }
                )
            }
        )
        report = replay_idt_record(forged, data)
        assert not report.verified
        assert any("rcm_pre_compression_conditions" in finding for finding in report.findings)


class TestImplausibleTemperature:
    def test_a_condition_below_500_k_is_refused(self) -> None:
        data = _member("x00000070_p.xml").replace(b"<x1>1279</x1>", b"<x1>450</x1>")
        with pytest.raises(RespecthRefusal) as caught:
            _parse("x00000070_p.xml", data)
        assert caught.value.reason is RespecthRefusalReason.IMPLAUSIBLE_IGNITION_TEMPERATURE
        assert "450" in caught.value.detail

    def test_500_k_itself_is_kept(self) -> None:
        data = _member("x00000070_p.xml").replace(b"<x1>1279</x1>", b"<x1>500</x1>")
        assert _parse("x00000070_p.xml", data).envelope.series[0].points


class TestOtherRealMembers:
    def test_cited_reference_doi(self) -> None:
        record = _parse("x10000001.xml")
        assert not isinstance(record.paper, Absent)
        assert record.paper.trust is ReferenceDoiTrust.CITED
        assert record.paper_doi == "10.1063/1.1696266"
        assert len(record.envelope.series[0].points) == 7
        assert _triples(record)[0] == ("964", "K", "5", "atm", "15000", "us")
        assert replay_idt_record(record, _member("x10000001.xml")).verified

    def test_rapid_compression_machine_with_volume_history(self) -> None:
        record = _parse("x40001058_19.xml")
        assert record.apparatus.device_class is ReactorType.RCM
        assert isinstance(record.apparatus.mode_raw, Absent)
        assert record.apparatus.mode_basis is ApparatusModeBasis.NOT_APPLICABLE
        assert record.apparatus.assumption is None
        assert _triples(record) == [("1039", "K", "10.9", "atm", "14.0", "ms")]
        axes = {axis.axis_id: axis.role for axis in record.envelope.series[0].axes}
        assert axes == {
            "ignition_delay": AxisRole.OBSERVATION,
            "pressure": AxisRole.COORDINATE,
            "temperature": AxisRole.COORDINATE,
        }
        assert [(g.group_id, g.column_names) for g in record.skipped_data_groups] == [("dg2", ("time", "volume"))]
        assert (record.ignition.target, record.ignition.criterion) == (
            IgnitionTarget.PRESSURE,
            IgnitionCriterion.MAX_SLOPE,
        )
        composition = record.envelope.composition
        assert not isinstance(composition, Absent)
        roles = {c.species_raw_name: c.role for c in composition.components}
        assert roles["CO"] is ComponentRole.FUEL
        assert isinstance(roles["CO2"], Absent)
        assert replay_idt_record(record, _member("x40001058_19.xml")).verified

    def test_mbar_pressure_binds_exactly_to_pascal(self) -> None:
        # Every real mbar member states pre-compression conditions and is refused for that
        # (TestRcmConditions), so the end-of-compression RCM fixture's pressure is restated in mbar.
        data = (
            _member("x40001058_19.xml")
            .replace(b'units="atm"', b'units="mbar"')
            .replace(b"<x2>10.9</x2>", b"<x2>11044</x2>")
        )
        record = _parse("x40001058_19.xml", data)
        (pressure,) = (c.value for c in record.envelope.series[0].points[0].coordinates if c.axis_id == "pressure")
        assert (pressure.raw_text, pressure.unit_normalized) == ("11044", "mbar")
        assert pressure.conversion_table_sha256 == units.TABLE_V2.sha256
        converted = units.convert(
            pressure.canonical_decimal_value,
            quantity=QuantityKind.PRESSURE,
            from_unit="mbar",
            to_unit="Pa",
            table=units.TABLE_V2,
        )
        assert converted.exact == "1104400"
        assert replay_idt_record(record, data).verified

    def test_relative_concentration_carries_its_amount(self) -> None:
        # No mapped real member uses this criterion (the three that do are refused for a common
        # `uncertainty` in kPa), so the fixture's ignitionType is rewritten to the corpus's own spelling.
        data = _member("x00000070_p.xml").replace(
            b'<ignitionType target="OH;" type="d/dt max"/>',
            b'<ignitionType target="OHEX;" type="relative concentration" amount="0.5" units="unitless"/>',
        )
        record = _parse("x00000070_p.xml", data)
        assert (record.ignition.target, record.ignition.criterion) == (
            IgnitionTarget.OHEX,
            IgnitionCriterion.RELATIVE_CONCENTRATION,
        )
        assert not isinstance(record.ignition.amount, Absent)
        assert record.ignition.amount.raw == "0.5"
        assert replay_idt_record(record, data).verified


class TestReplayRefuses:
    def test_bytes_that_do_not_hash_to_the_node(self) -> None:
        record = _parse("x00000070_p.xml")
        tampered = _member("x00000070_p.xml").replace(b"<x2>265.1</x2>", b"<x2>265.2</x2>")
        report = replay_idt_record(record, tampered)
        assert not report.verified
        assert report.checked == 0

    def test_a_recorded_value_the_bytes_do_not_say(self) -> None:
        record = _parse("x00000070_p.xml")
        forged = record.model_copy(update={"record_doi": record.record_doi.model_copy(update={"raw": "10.0/forged"})})
        report = replay_idt_record(forged, _member("x00000070_p.xml"))
        assert not report.verified
        assert any("10.0/forged" in finding for finding in report.findings)

    def test_a_trust_marker_the_comments_do_not_support(self) -> None:
        record = _parse("x10000001.xml")
        assert not isinstance(record.paper, Absent)
        forged = record.model_copy(
            update={"paper": record.paper.model_copy(update={"trust": ReferenceDoiTrust.PLACEHOLDER})}
        )
        assert not replay_idt_record(forged, _member("x10000001.xml")).verified

    def test_the_evidence_store_replayer_never_reports_it_verified(self, tmp_path: Path) -> None:
        # The paper-lane replayer has no XPath gate: it must fail closed, never pass.
        from carmel.services.dataset_replay import ReplayOutcome, replay_envelope

        report = replay_envelope(tmp_path, _parse("x00000070_p.xml").envelope)
        assert report.overall_outcome is not ReplayOutcome.VERIFIED


class TestRefusals:
    """Each refusal is a typed RespecthRefusal: no record is returned at all."""

    def _refused(self, name: str, data: bytes | None = None) -> RespecthRefusal:
        with pytest.raises(RespecthRefusal) as caught:
            _parse(name, data)
        return caught.value

    @pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be", "utf-32"])
    def test_a_non_utf8_member_is_refused_before_any_entity_can_expand(self, codec: str) -> None:
        # Spelled in UTF-16/32 the byte scan for "<!DOCTYPE"/"<!ENTITY" sees nothing, and expat
        # would honour the declared encoding and expand the internal entity into the value.
        text = (
            f'<?xml version="1.0" encoding="{codec.upper()}"?>'
            '<!DOCTYPE experiment [<!ENTITY boom "1279">]>'
            + _member("x00000070_p.xml").decode("utf-8").replace("<x1>1279</x1>", "<x1>&boom;</x1>", 1)
        )
        refusal = self._refused("x00000070_p.xml", text.encode(codec))
        assert refusal.reason is RespecthRefusalReason.MALFORMED_XML

    @pytest.mark.parametrize(
        "prolog", [b'<?xml version="1.0" encoding="ISO-8859-1"?>', b"<?xml version=\"1.0\" encoding='latin-1'?>"]
    )
    def test_a_non_utf8_encoding_declaration_is_refused(self, prolog: bytes) -> None:
        refusal = self._refused("x00000070_p.xml", prolog + _member("x00000070_p.xml"))
        assert refusal.reason is RespecthRefusalReason.MALFORMED_XML
        assert "encoding" in refusal.detail

    def test_a_utf8_declaration_and_bom_still_parse(self) -> None:
        data = b'\xef\xbb\xbf<?xml version="1.0" encoding="UTF-8"?>' + _member("x00000070_p.xml")
        assert len(_parse("x00000070_p.xml", data).envelope.series[0].points) == 3

    def test_unmapped_apparatus_incident_shock(self) -> None:
        data = _member("x00000070_p.xml").replace(b"reflected shock", b"incident shock")
        assert self._refused("x00000070_p.xml", data).reason is RespecthRefusalReason.UNMAPPED_APPARATUS

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            (b'type="d/dt max"', b'type="concentration"'),
            (b'target="OH;"', b'target="CH*;"'),
            (b'target="OH;"', b'target="OH;p;"'),
        ],
    )
    def test_unknown_ignition_definition(self, old: bytes, new: bytes) -> None:
        data = _member("x00000070_p.xml").replace(old, new)
        assert self._refused("x00000070_p.xml", data).reason is RespecthRefusalReason.UNMAPPED_IGNITION_DEFINITION

    def test_unmapped_unit(self) -> None:
        data = _member("x00000070_p.xml").replace(b'units="atm"', b'units="Torr"')
        assert self._refused("x00000070_p.xml", data).reason is RespecthRefusalReason.UNMAPPED_UNIT

    def test_unmapped_common_property(self) -> None:
        data = _member("x10000001.xml").replace(b'name="pressure"', b'name="pressure rise"')
        assert self._refused("x10000001.xml", data).reason is RespecthRefusalReason.UNMAPPED_PROPERTY

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda data: data[: len(data) // 2],
            lambda data: b"\x00\xff not xml",
            lambda data: b"",
            lambda data: b'<!DOCTYPE x [<!ENTITY a "b">]>' + data,
        ],
        ids=["truncated", "garbage", "empty", "doctype"],
    )
    def test_truncated_or_corrupt_xml(self, mutate: object) -> None:
        assert callable(mutate)
        data = mutate(_member("x00000070_p.xml"))
        assert self._refused("x00000070_p.xml", data).reason is RespecthRefusalReason.MALFORMED_XML

    def test_not_an_ignition_delay_record(self) -> None:
        data = _member("x00000070_p.xml").replace(
            b"ignition delay measurement", b"laminar burning velocity measurement"
        )
        assert self._refused("x00000070_p.xml", data).reason is RespecthRefusalReason.NOT_IGNITION_DELAY


class TestXPath:
    def test_positional_paths_address_one_node(self) -> None:
        from xml.etree import ElementTree

        root = ElementTree.fromstring(_member("x00000070_p.xml"))  # noqa: S314 - trusted fixture
        assert evaluate_xpath(root, "/experiment/fileDOI[1]") == "10.24388/x00000070"
        assert evaluate_xpath(root, "/experiment/dataGroup[1]/dataPoint[3]/x2[1]") == "68.9"
        assert evaluate_xpath(root, "/experiment/dataGroup[1]/property[2]/@units") == "us"
        assert evaluate_xpath(root, "/experiment/dataGroup[1]/dataPoint[4]/x2[1]") is None
        assert evaluate_xpath(root, "//x2") is None
        assert evaluate_xpath(root, "/other/fileDOI[1]") is None


class TestSchemaRules:
    """The additive schema members hold their own lines (V4, I4)."""

    def test_structured_record_series_cannot_claim_textual(self) -> None:
        envelope = _parse("x00000070_p.xml").envelope
        textual = Series(**{**dict(envelope.series[0]), "source_form": SourceForm.TEXTUAL})
        with pytest.raises(ValueError, match="DATABASE_RECORD"):
            DatasetEnvelope(**{**dict(envelope), "series": (textual,)})

    def test_database_record_is_a_parentless_root(self) -> None:
        (node,) = _parse("x00000070_p.xml").envelope.source_graph.nodes
        paper = SourceNode(
            node_id="paper",
            kind=SourceNodeKind.PAPER_PDF,
            sha256="a" * 64,
            parent_node_id=None,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            glyph_health=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            verification=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            crop_region=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            document_kind=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )
        child = SourceNode(**{**dict(node), "parent_node_id": "paper"})
        with pytest.raises(ValueError, match="parentless root"):
            SourceGraph(nodes=(child, paper))


# --------------------------------------------------------------------------- cache


def _zip_of(names: list[str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in names:
            info = zipfile.ZipInfo(name, date_time=(2024, 8, 8, 13, 31, 0))
            bundle.writestr(info, _member(name))
    return buffer.getvalue()


def _pin(name: str, data: bytes) -> PinnedArchive:
    return PinnedArchive(
        name=name,
        osf_path=f"/test/{name}",
        osf_file_id="0123456789abcdef01234567",
        osf_version=1,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _write_manifest(path: Path, archives: list[PinnedArchive], url_template: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "manifest_version": 1,
                "source": "respecth",
                "osf_node": "test",
                "doi": "10.0/test",
                "license": "CC-BY-4.0",
                "download_url_template": url_template,
                "archives": [archive.__dict__ for archive in archives],
            }
        )
    )
    return path


class _Double(http.server.BaseHTTPRequestHandler):
    """Serves ``server.body`` for any GET, counting requests."""

    server: _DoubleServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.server.requests += 1
        if self.server.body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        self.wfile.write(self.server.body)

    def log_message(self, *args: object) -> None:  # noqa: D102 - silence test noise
        pass


class _DoubleServer(http.server.ThreadingHTTPServer):
    body: bytes | None = None
    requests: int = 0


@pytest.fixture
def http_double(monkeypatch: pytest.MonkeyPatch) -> Iterator[_DoubleServer]:
    # Loopback must bypass any configured proxy so the double is reached directly.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    server = _DoubleServer(("127.0.0.1", 0), _Double)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _manifest_for(server: _DoubleServer, archive: PinnedArchive, tmp_path: Path) -> RespecthManifest:
    url = f"http://127.0.0.1:{server.server_address[1]}/{{osf_file_id}}?version={{osf_version}}"
    return load_manifest(_write_manifest(tmp_path / "manifest.json", [archive], url))


class TestCache:
    def test_download_verifies_caches_and_is_not_repeated(self, http_double: _DoubleServer, tmp_path: Path) -> None:
        data = _zip_of(["x00000070_p.xml"])
        archive = _pin("h2.zip", data)
        manifest = _manifest_for(http_double, archive, tmp_path)
        http_double.body = data
        cache = tmp_path / "cache"
        assert fetch_archive(archive, manifest=manifest, cache_root=cache) == data
        assert cached_archive_path(cache, archive.sha256).read_bytes() == data
        assert fetch_archive(archive, manifest=manifest, cache_root=cache, download=False) == data
        assert http_double.requests == 1

    def test_a_download_that_does_not_match_its_pin_is_never_cached(
        self, http_double: _DoubleServer, tmp_path: Path
    ) -> None:
        archive = _pin("h2.zip", _zip_of(["x00000070_p.xml"]))
        manifest = _manifest_for(http_double, archive, tmp_path)
        http_double.body = _zip_of(["x10000001.xml"])
        cache = tmp_path / "cache"
        with pytest.raises(ArchiveIntegrityError, match="does not match its pin"):
            fetch_archive(archive, manifest=manifest, cache_root=cache)
        assert not cached_archive_path(cache, archive.sha256).exists()

    def test_a_cached_file_with_the_wrong_sha256_is_refused_and_left(self, tmp_path: Path) -> None:
        data = _zip_of(["x00000070_p.xml"])
        archive = _pin("h2.zip", data)
        manifest = load_manifest(_write_manifest(tmp_path / "m.json", [archive], "http://127.0.0.1:9/{osf_file_id}"))
        path = cached_archive_path(tmp_path / "cache", archive.sha256)
        path.parent.mkdir(parents=True)
        corrupted = data[:-1] + bytes([data[-1] ^ 1])
        path.write_bytes(corrupted)
        with pytest.raises(ArchiveIntegrityError, match="cached file"):
            fetch_archive(archive, manifest=manifest, cache_root=tmp_path / "cache")
        assert path.read_bytes() == corrupted

    def test_an_oversized_cached_file_is_refused_before_it_is_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        data = _zip_of(["x00000070_p.xml"])
        archive = _pin("h2.zip", data)
        manifest = load_manifest(_write_manifest(tmp_path / "m.json", [archive], "http://127.0.0.1:9/{osf_file_id}"))
        path = cached_archive_path(tmp_path / "cache", archive.sha256)
        path.parent.mkdir(parents=True)
        with path.open("wb") as handle:
            handle.truncate(archive.size + 10**9)  # sparse: a 1 GB entry that costs no disk
        reads: list[int] = []
        real_open = Path.open

        def spying_open(self: Path, *args: object, **kwargs: object) -> object:
            handle = real_open(self, *args, **kwargs)  # type: ignore[call-overload]
            if self == path:
                real_read = handle.read
                handle.read = lambda size=-1: reads.append(size) or real_read(size)  # type: ignore[method-assign]
            return handle

        monkeypatch.setattr(Path, "open", spying_open)
        monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self} in full"))
        with pytest.raises(ArchiveIntegrityError, match="bytes on disk"):
            fetch_archive(archive, manifest=manifest, cache_root=tmp_path / "cache")
        assert reads == []
        assert path.stat().st_size == archive.size + 10**9

    def test_http_failure_and_offline_miss_are_fetch_errors(self, http_double: _DoubleServer, tmp_path: Path) -> None:
        archive = _pin("h2.zip", _zip_of(["x00000070_p.xml"]))
        manifest = _manifest_for(http_double, archive, tmp_path)
        with pytest.raises(ArchiveFetchError):
            fetch_archive(archive, manifest=manifest, cache_root=tmp_path / "cache")
        with pytest.raises(ArchiveFetchError, match="downloading is disabled"):
            fetch_archive(archive, manifest=manifest, cache_root=tmp_path / "cache", download=False)

    def test_member_reads_are_verified_by_sha256(self) -> None:
        data = _zip_of(["x00000070_p.xml", "x10000001.xml"])
        member = read_member(data, "x00000070_p.xml", MEMBER_SHA256["x00000070_p.xml"])
        assert member == _member("x00000070_p.xml")
        with pytest.raises(ArchiveIntegrityError, match="not the pinned"):
            read_member(data, "x00000070_p.xml", MEMBER_SHA256["x10000001.xml"])

    def test_member_listing_and_unreadable_archives(self) -> None:
        data = _zip_of(["x00000070_p.xml", "x10000001.xml"])
        assert [name for name, _ in iter_xml_members(data)] == ["x00000070_p.xml", "x10000001.xml"]
        with pytest.raises(ArchiveIntegrityError, match="not a readable zip"):
            list(iter_xml_members(b"not a zip"))
        with pytest.raises(ArchiveIntegrityError, match="no member"):
            read_member(data, "missing.xml", "0" * 64)

    @pytest.mark.parametrize("text", ["900", "a:b", "1300:900", "nan:1", "1:2:3"])
    def test_a_malformed_window_is_refused(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_window(text)

    @pytest.mark.parametrize(
        ("field", "value"),
        [("license", "CC-BY-NC-4.0"), ("sha256", "ABC"), ("osf_version", True)],
    )
    def test_a_malformed_manifest_is_refused(self, tmp_path: Path, field: str, value: object) -> None:
        archive = _pin("h2.zip", b"x")
        path = _write_manifest(tmp_path / "m.json", [archive], "http://x/{osf_file_id}")
        payload = json.loads(path.read_text())
        if field == "license":
            payload["license"] = value
        else:
            payload["archives"][0][field] = value
        path.write_text(json.dumps(payload))
        with pytest.raises(ManifestError):
            load_manifest(path)


# --------------------------------------------------------------------------- CLI

_FIXTURE_MEMBERS = ["x00000070_p.xml", "x10000001.xml", "x10000030_x.xml", "x40001058_19.xml"]
_HEADER = "record_doi\tpaper_doi\tdevice\tfuels\tT_K\tP_bar\tpoints"


@pytest.fixture
def fixture_cache(tmp_path: Path) -> tuple[Path, Path]:
    """A manifest pinning one zip of every fixture member, already in the cache."""
    data = _zip_of(_FIXTURE_MEMBERS)
    archive = _pin("fixtures.zip", data)
    path = cached_archive_path(tmp_path / "cache", archive.sha256)
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    manifest = _write_manifest(tmp_path / "manifest.json", [archive], "http://127.0.0.1:9/{osf_file_id}")
    return manifest, tmp_path / "cache"


def _find(fixture_cache: tuple[Path, Path], *extra: str) -> int:
    manifest, cache = fixture_cache
    return main(
        ["data", "find", "--kind", "idt", "--manifest", str(manifest), "--cache", str(cache), "--offline", *extra]
    )


class TestDataFind:
    def test_rows_for_the_ticket_query(
        self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _find(fixture_cache, "--fuel", "H2", "--T", "900:1300", "--P", "10:40") == 0
        assert capsys.readouterr().out.splitlines() == [
            _HEADER,
            "10.24388/x40001058_19\t10.1016/j.combustflame.2014.03.001\trcm\tCO+H2\t1039-1039\t11.04-11.04\t1/1",
            "1 matching dataset(s) of 4 mapped ignition-delay records; 0 refused (none)",
        ]

    def test_every_h2_record(self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]) -> None:
        assert _find(fixture_cache, "--fuel", "H2") == 0
        assert capsys.readouterr().out.splitlines()[1:5] == [
            "10.24388/x00000070\tplaceholder\tshock_tube\tH2\t1279-1344\t64.85-64.85\t3/3",
            "10.24388/x10000001\t10.1063/1.1696266\tshock_tube\tH2\t964-1075\t5.066-5.066\t7/7",
            "10.24388/x10000030\t10.1016/j.combustflame.2011.09.010\tshock_tube (mode assumed)"
            "\tH2\t1024-1195\t5-5\t7/7",
            "10.24388/x40001058_19\t10.1016/j.combustflame.2014.03.001\trcm\tCO+H2\t1039-1039\t11.04-11.04\t1/1",
        ]

    def test_a_filter_excluding_everything_is_empty_and_exit_zero(
        self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert _find(fixture_cache, "--T", "5000:6000") == 0
        assert capsys.readouterr().out.splitlines() == [
            _HEADER,
            "0 matching dataset(s) of 4 mapped ignition-delay records; 0 refused (none)",
        ]

    def test_a_cache_integrity_failure_lists_nothing(
        self, fixture_cache: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
    ) -> None:
        manifest, cache = fixture_cache
        (cached,) = (cache / "sha256").iterdir()
        cached.write_bytes(b"not the pinned archive")
        assert _find(fixture_cache, "--fuel", "H2") == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Refusing to list ReSpecTh records" in captured.err

    def test_a_reversed_window_is_a_usage_error(self, fixture_cache: tuple[Path, Path]) -> None:
        with pytest.raises(SystemExit) as caught:
            _find(fixture_cache, "--T", "1300:900")
        assert caught.value.code == 2


# --------------------------------------------------------------------------- refusal branches


_ST = "x00000070_p.xml"
_RCM = "x40001058_19.xml"
_R = RespecthRefusalReason
_COMPOSITION = _member(_ST)[
    _member(_ST).index(b'        <property name="initial composition"') : _member(_ST).index(
        b"</property>", _member(_ST).index(b'name="initial composition"')
    )
    + len(b"</property>\n")
]

#: (id, fixture, old bytes, new bytes, reason, detail fragment) -- one small edit of a real
#: member per refusal branch of the parser.
_MUTATIONS = [
    (
        "empty text",
        _ST,
        b"<fileDOI>10.24388/x00000070</fileDOI>",
        b"<fileDOI> </fileDOI>",
        _R.INCOMPLETE_RECORD,
        "is empty",
    ),
    ("empty attribute", _ST, b'units="us"', b'units=" "', _R.INCOMPLETE_RECORD, "has no @units"),
    (
        "missing element",
        _ST,
        b'<ignitionType target="OH;" type="d/dt max"/>',
        b"",
        _R.INCOMPLETE_RECORD,
        "exactly one <ignitionType>",
    ),
    (
        "repeated optional element",
        _ST,
        b"<mode>reflected shock</mode>",
        b"<mode>reflected shock</mode><mode>reflected shock</mode>",
        _R.INCOMPLETE_RECORD,
        "at most one <mode>",
    ),
    ("unclean numeral", _ST, b"<x2>265.1</x2>", b"<x2>2.6.5</x2>", _R.UNREADABLE_VALUE, "not a clean numeral"),
    ("unrepresentable numeral", _ST, b"<x2>265.1</x2>", b"<x2>1e999999999</x2>", _R.UNREADABLE_VALUE, "1e999999999"),
    ("wrong root", _ST, b"experiment>", b"record>", _R.MALFORMED_XML, "root element is <record>"),
    (
        "format major",
        _ST,
        b"<ReSpecThVersion>\n        <major>2</major>",
        b"<ReSpecThVersion>\n        <major>3</major>",
        _R.UNSUPPORTED_FORMAT_VERSION,
        "major '3'",
    ),
    (
        "extra ignition attribute",
        _ST,
        b'type="d/dt max"/>',
        b'type="d/dt max" onset="1"/>',
        _R.UNMAPPED_IGNITION_DEFINITION,
        "unmapped attributes ['onset']",
    ),
    (
        "amount on a criterion without one",
        _ST,
        b'type="d/dt max"/>',
        b'type="d/dt max" amount="0.5"/>',
        _R.UNMAPPED_IGNITION_DEFINITION,
        "carries an amount",
    ),
    (
        "non-decimal amount",
        _ST,
        b'type="d/dt max"/>',
        b'type="relative concentration" amount="half" units="unitless"/>',
        _R.UNMAPPED_IGNITION_DEFINITION,
        "is not a decimal",
    ),
    (
        "amount outside (0, 1]",
        _ST,
        b'type="d/dt max"/>',
        b'type="relative concentration" amount="1.5" units="unitless"/>',
        _R.UNMAPPED_IGNITION_DEFINITION,
        "not a unitless fraction",
    ),
    (
        "amount with a unit",
        _ST,
        b'type="d/dt max"/>',
        b'type="relative concentration" amount="0.5" units="percent"/>',
        _R.UNMAPPED_IGNITION_DEFINITION,
        "not a unitless fraction",
    ),
    (
        "stray child in composition",
        _ST,
        b'<component><speciesLink preferredKey="Ar"',
        b'<note>x</note><component><speciesLink preferredKey="Ar"',
        _R.UNMAPPED_PROPERTY,
        "unexpected <note>",
    ),
    (
        "amount not a mole fraction",
        _ST,
        b'<amount units="mole fraction" >0.0050',
        b'<amount units="ppm" >5000',
        _R.UNMAPPED_UNIT,
        "'ppm'",
    ),
    (
        "empty composition",
        _ST,
        _COMPOSITION,
        b'        <property name="initial composition"></property>\n',
        _R.INCOMPLETE_RECORD,
        "lists no components",
    ),
    ("no composition", _ST, _COMPOSITION, b"", _R.INCOMPLETE_RECORD, "no initial composition"),
    ("duplicate species", _ST, b'preferredKey="O2"', b'preferredKey="H2"', _R.SCHEMA_REJECTED, "duplicate"),
    (
        "uncertainty of another property",
        _ST,
        b'reference="ignition delay"',
        b'reference="temperature"',
        _R.UNMAPPED_PROPERTY,
        "reference='temperature'",
    ),
    (
        "repeated common property",
        _ST,
        b"<commonProperties>",
        b'<commonProperties><property name="pressure" units="atm"><value>1</value></property>',
        _R.UNMAPPED_PROPERTY,
        "repeated common property 'pressure'",
    ),
    ("no pressure", _ST, b'<property name="pressure"', b'<property name="density"', _R.UNMAPPED_PROPERTY, "'density'"),
    (
        "unmapped data group",
        _ST,
        b"<ignitionType",
        b'<dataGroup id="dg9"><property name="time" id="x9" units="s"/></dataGroup><ignitionType',
        _R.UNMAPPED_DATA_GROUP,
        "('time',)",
    ),
    (
        "two ignition-delay groups",
        _ST,
        b"<ignitionType",
        b'<dataGroup id="dg9"><property name="ignition delay" id="x9" units="us"/></dataGroup><ignitionType',
        _R.INCOMPLETE_RECORD,
        "found 2",
    ),
    (
        "unmapped data column",
        _ST,
        b'<property name="ignition delay"  id="x2"',
        b'<property name="density" id="x3" units="K"/><property name="ignition delay"  id="x2"',
        _R.UNMAPPED_PROPERTY,
        "data column 'density'",
    ),
    (
        "condition stated twice",
        _ST,
        b'<property name="ignition delay"  id="x2"',
        b'<property name="pressure" id="x3" units="atm"/><property name="ignition delay"  id="x2"',
        _R.UNMAPPED_PROPERTY,
        "'pressure' is stated more than once",
    ),
    (
        "no varying condition",
        _ST,
        b'<property name="temperature"     id="x1"   label="T"    sourcetype="digitized"  units="K"  />',
        b"",
        _R.INCOMPLETE_RECORD,
        "no condition varies",
    ),
    (
        "row missing a cell",
        _ST,
        b"<x1>1279</x1><x2>265.1</x2>",
        b"<x1>1279</x1>",
        _R.INCOMPLETE_RECORD,
        "not exactly the columns",
    ),
    ("unreadable volume", _RCM, b"<x9>1.0034797</x9>", b"<x9>n/a</x9>", _R.UNREADABLE_VALUE, "reads 'n/a'"),
    ("unreadable point link", _RCM, b'dataPointLink="1"', b'dataPointLink="first"', _R.UNREADABLE_VALUE, "'first'"),
    (
        "negative uncertainty",
        _ST,
        b"><value>0.18</value>",
        b"><value>-0.18</value>",
        _R.SCHEMA_REJECTED,
        "strictly posit",
    ),
]


@pytest.mark.parametrize(
    ("name", "old", "new", "reason", "fragment"),
    [pytest.param(name, old, new, reason, fragment, id=case) for case, name, old, new, reason, fragment in _MUTATIONS],
)
def test_each_parser_refusal_branch(
    name: str, old: bytes, new: bytes, reason: RespecthRefusalReason, fragment: str
) -> None:
    data = _member(name)
    assert old in data
    edited = data.replace(old, new)
    with pytest.raises(RespecthRefusal) as caught:
        _parse(name, edited)
    assert caught.value.reason is reason
    assert fragment in caught.value.detail


@pytest.mark.parametrize(
    ("name", "row", "fragment"),
    [
        (_ST, rb"<dataPoint><x1>.*?</dataPoint>", "has no data points"),
        (_RCM, rb"<dataPoint><x8>.*?</dataPoint>", "holds no data points"),
    ],
)
def test_a_group_without_rows_is_refused(name: str, row: bytes, fragment: str) -> None:
    import re

    edited, count = re.subn(row, b"", _member(name))
    assert count > 0
    with pytest.raises(RespecthRefusal) as caught:
        _parse(name, edited)
    assert caught.value.reason is RespecthRefusalReason.INCOMPLETE_RECORD
    assert fragment in caught.value.detail


class TestOptionalFactsAbsent:
    def test_no_reference_doi_maps_with_the_paper_absent(self) -> None:
        data = _member("x10000001.xml")
        start = data.index(b"<referenceDOI>")
        data = data[:start] + data[data.index(b"</referenceDOI>") + len(b"</referenceDOI>") :]
        record = _parse("x10000001.xml", data)
        assert isinstance(record.paper, Absent)
        assert record.paper.reason is AbsenceReason.NOT_REPORTED_HERE
        assert record.paper_doi is None
        assert replay_idt_record(record, data).verified

    def test_a_property_without_sourcetype_records_none(self) -> None:
        data = (
            _member(_ST)
            .replace(b'label="P"  sourcetype="reported"', b'label="P"')
            .replace(b'label="T"    sourcetype="digitized"', b'label="T"')
        )
        record = _parse(_ST, data)
        assert dict(record.column_source_types).keys() == {"ignition_delay"}
        assert replay_idt_record(record, data).verified

    def test_a_history_linked_to_all_points_covers_them(self) -> None:
        data = _member(_RCM).replace(b'dataPointLink="1"', b'dataPointLink="all"')
        record = _parse(_RCM, data)
        assert not isinstance(record.rcm_conditions, Absent)
        assert record.rcm_conditions.histories[0].point_link.raw == "all"


class TestRecordValidators:
    def test_placeholder_trust_needs_its_evidence(self) -> None:
        paper = _parse(_ST).paper
        assert not isinstance(paper, Absent)
        with pytest.raises(ValueError, match="placeholder_evidence must be present"):
            type(paper).model_validate(
                {**paper.model_dump(), "placeholder_evidence": Absent(reason=AbsenceReason.NOT_APPLICABLE)}
            )

    def test_the_record_node_must_match_the_archive_pin(self) -> None:
        record = _parse(_ST)
        moved = record.archive.model_copy(update={"member_path": "elsewhere.xml"})
        with pytest.raises(ValueError, match="disagrees with the archive pin"):
            RespecthIdtRecord.model_validate({**dict(record), "archive": moved})

    def test_the_record_node_must_be_a_database_record(self) -> None:
        record = _parse(_ST)
        node = record.envelope.source_graph.nodes[0]
        graph = record.envelope.source_graph.model_copy(
            update={"nodes": (node.model_copy(update={"origin": Absent(reason=AbsenceReason.NOT_APPLICABLE)}),)}
        )
        envelope = record.envelope.model_copy(update={"source_graph": graph})
        with pytest.raises(ValueError, match="must be a DATABASE_RECORD with an ArchiveOrigin"):
            RespecthIdtRecord.model_validate({**dict(record), "envelope": envelope})


def _with(record: RespecthIdtRecord, **update: object) -> RespecthIdtRecord:
    return record.model_copy(update=update)


class TestReplayCatchesForgery:
    """Each forged record keeps every grounded string true, so only the re-derivation catches it."""

    def test_a_ref_to_another_node(self) -> None:
        record = _parse(_ST)
        forged = _with(
            record,
            record_doi=record.record_doi.model_copy(
                update={"ref": record.record_doi.ref.model_copy(update={"node_id": "elsewhere"})}
            ),
        )
        report = replay_idt_record(forged, _member(_ST))
        assert report.findings == ("record_doi.ref: does not address the record node by XPath",)

    def test_a_device_class_the_apparatus_does_not_map_to(self) -> None:
        record = _parse("x10000001.xml")
        forged = _with(record, apparatus=record.apparatus.model_copy(update={"device_class": ReactorType.RCM}))
        findings = replay_idt_record(forged, _member("x10000001.xml")).findings
        assert "apparatus ('shock tube', 'reflected shock') does not re-map to 'rcm'" in findings

    def test_a_dropped_assumption(self) -> None:
        record = _parse("x10000030_x.xml")
        forged = _with(
            record,
            apparatus=record.apparatus.model_copy(
                update={
                    "mode_basis": ApparatusModeBasis.NOT_APPLICABLE,
                    "assumed_mode": Absent(reason=AbsenceReason.NOT_APPLICABLE),
                }
            ),
        )
        report = replay_idt_record(forged, _member("x10000030_x.xml"))
        assert report.findings == (
            "apparatus mode assumption None does not re-map (the table gives 'reflected shock')",
        )
        assert report.assumptions == ()

    def test_dropped_rcm_evidence(self) -> None:
        record = _parse(_RCM)
        forged = _with(record, rcm_conditions=Absent(reason=AbsenceReason.NOT_APPLICABLE))
        (finding,) = replay_idt_record(forged, _member(_RCM)).findings
        assert finding.startswith("rcm_conditions None do not re-derive")

    def test_a_relabelled_ignition_target(self) -> None:
        record = _parse(_ST)
        forged = _with(record, ignition=record.ignition.model_copy(update={"target": IgnitionTarget.PRESSURE}))
        assert replay_idt_record(forged, _member(_ST)).findings == (
            "the ignition definition does not re-map from its raw text",
        )

    def test_an_uncertainty_definition_that_is_not_the_delays(self) -> None:
        record = _parse(_ST)
        uncertainty = record.uncertainty
        assert not isinstance(uncertainty, Absent)
        forged = _with(
            record,
            uncertainty=uncertainty.model_copy(
                update={"reference_raw": uncertainty.reference_raw.model_copy(update={"raw": "temperature"})}
            ),
        )
        findings = replay_idt_record(forged, _member(_ST)).findings
        assert "the uncertainty definition does not re-map to a relative standard deviation of the delay" in findings

    def test_bytes_that_hash_right_but_do_not_parse(self) -> None:
        garbage = b"<!DOCTYPE x>"
        record = _parse(_ST)
        node = record.envelope.source_graph.nodes[0].model_copy(update={"sha256": hashlib.sha256(garbage).hexdigest()})
        forged = _with(
            record,
            envelope=record.envelope.model_copy(
                update={"source_graph": record.envelope.source_graph.model_copy(update={"nodes": (node,)})}
            ),
        )
        report = replay_idt_record(forged, garbage)
        assert (report.verified, report.checked) == (False, 0)
        assert report.findings[0].startswith("malformed_xml:")


class TestArchiveEdges:
    def _stored_zip(self, entries: list[tuple[str, bytes]]) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as bundle:
            for name, data in entries:
                bundle.writestr(zipfile.ZipInfo(name, date_time=(2024, 8, 8, 13, 31, 0)), data)
        return buffer.getvalue()

    def test_directories_and_non_xml_members_are_skipped(self) -> None:
        data = self._stored_zip([("indirect/", b""), ("notes.txt", b"cite us"), (_ST, _member(_ST))])
        assert [name for name, _ in iter_xml_members(data)] == [_ST]

    def test_a_member_whose_bytes_fail_their_crc_is_refused(self) -> None:
        data = bytearray(self._stored_zip([(_ST, _member(_ST))]))
        at = bytes(data).index(b"<fileDOI>")
        data[at + 1] ^= 1
        with pytest.raises(ArchiveIntegrityError, match="cannot be read"):
            list(iter_xml_members(bytes(data)))

    def test_declared_sizes_past_the_caps_are_refused_unread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from carmel.services import respecth_archive

        data = self._stored_zip([(_ST, _member(_ST))])
        monkeypatch.setattr(respecth_archive, "_MAX_MEMBER_BYTES", 100)
        with pytest.raises(ArchiveIntegrityError, match="declares"):
            list(iter_xml_members(data))
        monkeypatch.setattr(respecth_archive, "_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 100)
        with pytest.raises(ArchiveIntegrityError, match="more uncompressed bytes"):
            list(iter_xml_members(data))

    def test_default_cache_root(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from carmel.services.respecth_archive import DATA_CACHE_ENV_VAR, default_cache_root

        monkeypatch.setenv(DATA_CACHE_ENV_VAR, "~/elsewhere")
        monkeypatch.setenv("HOME", str(tmp_path))
        assert default_cache_root() == tmp_path / "elsewhere"
        monkeypatch.delenv(DATA_CACHE_ENV_VAR)
        assert default_cache_root() == tmp_path / ".carmel" / "data_cache"


class TestManifestShape:
    def _payload(self, tmp_path: Path) -> dict[str, object]:
        path = _write_manifest(
            tmp_path / "m.json", [_pin("a.zip", b"a"), _pin("b.zip", b"b")], "http://x/{osf_file_id}"
        )
        payload: dict[str, object] = json.loads(path.read_text())
        return payload

    def _refused(self, tmp_path: Path, text: str, match: str) -> None:
        path = tmp_path / "bad.json"
        path.write_text(text)
        with pytest.raises(ManifestError, match=match):
            load_manifest(path)

    def test_unreadable_and_non_json(self, tmp_path: Path) -> None:
        with pytest.raises(ManifestError, match="cannot read"):
            load_manifest(tmp_path / "missing.json")
        self._refused(tmp_path, "{not json", "cannot read")

    @pytest.mark.parametrize("top", ["[]", '{"manifest_version": 2}'])
    def test_not_a_version_1_object(self, tmp_path: Path, top: str) -> None:
        self._refused(tmp_path, top, "manifest_version 1")

    @pytest.mark.parametrize(
        ("edit", "match"),
        [
            (lambda p: p.update(archives=[]), "at least one archive"),
            (lambda p: p["archives"].__setitem__(0, "a.zip"), r"archives\[0\] must be an object"),
            (lambda p: p["archives"][0].update(osf_file_id="nope"), "is not an OSF file id"),
            (lambda p: p["archives"][0].update(size=0), "must be positive"),
            (lambda p: p["archives"][1].update(sha256=p["archives"][0]["sha256"]), "same sha256 twice"),
        ],
    )
    def test_malformed_pins(self, tmp_path: Path, edit: object, match: str) -> None:
        payload = self._payload(tmp_path)
        edit(payload)  # type: ignore[operator]
        self._refused(tmp_path, json.dumps(payload), match)


class TestQueryEdges:
    def test_refusals_are_counted_and_other_experiment_types_skipped(self, tmp_path: Path) -> None:
        from carmel.services.respecth_query import load_idt_records

        not_idt = _member(_ST).replace(b"ignition delay measurement", b"laminar burning velocity measurement")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("x40001039.xml", _member("x40001039.xml"))
            bundle.writestr("flame.xml", not_idt)
            bundle.writestr(_RCM, _member(_RCM))
        data = buffer.getvalue()
        archive = _pin("mix.zip", data)
        path = cached_archive_path(tmp_path / "cache", archive.sha256)
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        manifest = load_manifest(_write_manifest(tmp_path / "m.json", [archive], "http://127.0.0.1:9/{osf_file_id}"))
        loaded = load_idt_records(manifest, cache_root=tmp_path / "cache", download=False)
        assert [record.record_doi.raw for record in loaded.records] == ["10.24388/x40001058_19"]
        assert dict(loaded.refusals) == {"rcm_pre_compression_conditions": 1}

    def test_fuel_filter_and_a_record_without_composition(self) -> None:
        from carmel.services.respecth_query import find_records

        record = _parse(_ST)
        bare = _with(
            record, envelope=record.envelope.model_copy(update={"composition": Absent(reason=AbsenceReason.UNKNOWN)})
        )
        assert find_records((record,), fuel="CH4") == []
        (match,) = find_records((bare,))
        assert match.fuels == ()
        assert find_records((bare,), fuel="H2") == []
