"""Public source/history arithmetic regressions for PR review round 7."""

import math
import xml.etree.ElementTree as ET

import pytest
import yaml

from carmel.services import respecth
from carmel.services.chemked import ChemkedRefusal, ChemkedRefusalReason, parse_idt_record, replay_idt_record
from carmel.services.rcm_history import HistoryReason, HistoryRefusal, RcmHistory
from carmel.services.t3_export import export_idt
from tests.test_rcm_export import chemked_history
from tests.test_rcm_review_round4 import precompression_source


def history_source(samples, compression_time):
    doc = yaml.safe_load(chemked_history())
    row = doc["datapoints"][0]
    row["volume-history"]["time"]["units"] = "s"
    row["volume-history"]["volume"]["units"] = "m3"
    row["volume-history"]["values"] = [[volume, time] for time, volume in samples]
    row["compression-time"] = [f"{compression_time} s"]
    return yaml.safe_dump(doc).encode()


def test_extreme_endpoint_history_refuses_without_zero_division():
    raw = history_source([(0, "1E+308"), (1, "1E-308")], 1)
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(raw, "extreme-endpoint.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


def test_chemked_replay_translates_invalid_history_arithmetic(monkeypatch):
    raw = history_source([(0, "1E+308"), (1, "1E-308")], 1)
    with monkeypatch.context() as bypass:
        bypass.setattr(RcmHistory, "volume_ratio", property(lambda self: 2.0))
        forged = parse_idt_record(raw, "extreme-replay.yaml")
    with pytest.raises(ChemkedRefusal) as caught:
        replay_idt_record(forged, raw)
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


@pytest.mark.parametrize("history", [{"values": [[0, 1], [1, 0.2]]}, None, {}, {"values": []}])
def test_any_present_history_on_a_shock_tube_is_refused(history):
    doc = yaml.safe_load(chemked_history())
    doc["apparatus"]["kind"] = "shock tube"
    doc["datapoints"][0]["volume-history"] = history
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "shock-tube-history.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


@pytest.mark.parametrize(
    "pair",
    [[], 1, [0], [0, 1, 2], [False, 1], [0, True], ["not-numeric", 1], [0, None], [0, "NaN"], ["Infinity", 1]],
)
def test_malformed_rcm_sample_pairs_are_history_invalid(pair):
    doc = yaml.safe_load(chemked_history())
    doc["datapoints"][0]["volume-history"]["values"][0] = pair
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(yaml.safe_dump(doc).encode(), "malformed-pair.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID


@pytest.mark.parametrize("interior", [False, True])
def test_exact_sample_volume_avoids_cancellation(interior):
    samples = [(0, "1E-307"), (1, "1E+308"), (2, "1E-308")]
    if interior:
        samples.append((3, "1E-307"))
    record = parse_idt_record(history_source(samples, 2), "exact-sample.yaml")
    assert record.rcm_histories[0].volume_ratio == pytest.approx(10, rel=1e-15)


@pytest.mark.parametrize("eoc", [0.125, 0.3, 1.25, 1.9])
def test_weighted_interpolation_matches_ordinary_old_formula_within_one_ulp(eoc):
    samples = [(0, 10), (1, 2), (2, 3)]
    record = parse_idt_record(history_source(samples, eoc), "between-samples.yaml")
    left = 0 if eoc < 1 else 1
    fraction = eoc - samples[left][0]
    old_end_volume = samples[left][1] + fraction * (samples[left + 1][1] - samples[left][1])
    old_ratio = 10 / old_end_volume
    assert abs(record.rcm_histories[0].volume_ratio - old_ratio) <= math.ulp(old_ratio)


def test_interpolated_volume_with_overflowing_ratio_is_history_invalid():
    raw = history_source([(0, "1E+308"), (1, "1E-308"), (2, "1E-310")], 1.5)
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(raw, "overflow-ratio.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID
    assert "volume ratio must be finite and positive" in caught.value.detail


def test_interpolation_across_finite_extreme_times_does_not_overflow_the_interval():
    raw = history_source([("-1E+308", 10), ("1E+308", 2), ("1.5E+308", 3)], 0)
    record = parse_idt_record(raw, "wide-time-interval.yaml")
    assert record.rcm_histories[0].volume_ratio == 10 / 6


def test_interpolation_with_overflowing_offset_and_interval_remains_finite():
    raw = history_source([("-1.5E+308", 10), ("1.5E+308", 2), ("1.6E+308", 3)], "1E+308")
    record = parse_idt_record(raw, "overflowing-offset.yaml")
    assert record.rcm_histories[0].volume_ratio == pytest.approx(3, rel=1e-15)


def test_exact_weighted_end_volume_remains_representable():
    raw = history_source([(0, "1E-323"), (1, "5E-324"), (2, "5E-324")], 1.5)
    record = parse_idt_record(raw, "small-end-volume.yaml")
    assert record.rcm_histories[0].volume_ratio == 2


def test_underflowed_ratio_is_a_typed_history_refusal():
    raw = history_source([(0, "1E-307"), (1, "1E-308"), (2, "1E+308")], 2)
    with pytest.raises(ChemkedRefusal) as caught:
        parse_idt_record(raw, "underflow-ratio.yaml")
    assert caught.value.reason is ChemkedRefusalReason.HISTORY_INVALID
    assert "compression time must describe compression" in caught.value.detail


def respecth_extreme_source():
    raw, pin = precompression_source(minimum_volume="0.2")
    root = ET.fromstring(raw)
    group = root.findall("dataGroup")[1]
    group.find("property[@name='volume']").set("units", "m3")
    for point in group.findall("dataPoint"):
        group.remove(point)
    for time, volume in (("0", "1E+308"), ("1", "1E-308")):
        point = ET.SubElement(group, "dataPoint")
        ET.SubElement(point, "x4").text = time
        ET.SubElement(point, "x5").text = volume
    return ET.tostring(root), pin


def test_respecth_parse_and_replay_translate_invalid_history_arithmetic(monkeypatch):
    raw, pin = respecth_extreme_source()
    # Source-grounded forged record: bypass only admission during construction.
    # Real parse/replay must reject the same bytes without leaking arithmetic.
    with monkeypatch.context() as bypass:
        bypass.setattr(RcmHistory, "volume_ratio", property(lambda self: 2.0))
        forged = respecth.parse_idt_record(raw, pin, "extreme.xml")
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, pin, "extreme.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.HISTORY_INVALID
    replay = respecth.replay_idt_record(forged, raw)
    assert not replay.verified
    assert any("history_invalid" in finding for finding in replay.findings)


def test_derived_export_translates_history_refusal_instead_of_thermo_refusal(monkeypatch):
    pytest.importorskip("rdkit")
    raw, pin = respecth_extreme_source()
    with monkeypatch.context() as bypass:
        bypass.setattr(RcmHistory, "volume_ratio", property(lambda self: 2.0))
        forged = respecth.parse_idt_record(raw, pin, "extreme.xml")
    payload, report = export_idt([forged], include_derived_labels=True)
    assert payload["points"] == []
    assert report["refused"] == {"t3_constraint": 1}
    assert "history_invalid" in report["refusals"][0]["detail"]


def test_respecth_derived_caller_and_replay_preserve_history_refusal(monkeypatch):
    raw, pin = precompression_source(minimum_volume="0.2")
    record = respecth.parse_idt_record(raw, pin, "caller.xml")

    def unavailable_ratio(*args):
        raise HistoryRefusal(HistoryReason.INVALID_HISTORY, "unrepresentable volume ratio")

    monkeypatch.setattr(respecth, "isentropic_eoc", unavailable_ratio)
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, pin, "caller.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.HISTORY_INVALID
    replay = respecth.replay_idt_record(record, raw)
    assert not replay.verified
    assert any("history_invalid" in finding for finding in replay.findings)


def test_derived_pressure_overflow_remains_a_typed_thermo_refusal(monkeypatch):
    raw, pin = precompression_source(minimum_volume="0.2")
    root = ET.fromstring(raw)
    root.find("dataGroup/property[@name='pressure']").set("units", "Pa")
    root.find("dataGroup/dataPoint/x1").text = "1E+308"
    raw = ET.tostring(root)
    with monkeypatch.context() as bypass:
        bypass.setattr(respecth, "isentropic_eoc", lambda *args: (800.0, 100000.0))
        forged = respecth.parse_idt_record(raw, pin, "pressure-overflow.xml")
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, pin, "pressure-overflow.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE
    assert "derived pressure is not finite" in caught.value.detail
    replay = respecth.replay_idt_record(forged, raw)
    assert not replay.verified
    assert any("rcm_thermo_unavailable" in finding for finding in replay.findings)
    pytest.importorskip("rdkit")
    payload, report = export_idt([forged], include_derived_labels=True)
    assert payload["points"] == []
    assert report["refused"] == {"rcm_thermo_unavailable": 1}
    assert "derived pressure is not finite" in report["refusals"][0]["detail"]
