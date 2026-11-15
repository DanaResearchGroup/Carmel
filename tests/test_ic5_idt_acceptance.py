"""Iso-pentane shock admission and the acceptance-only case-3 revision."""

import hashlib
import json
from decimal import ROUND_DOWN, ROUND_UP, localcontext
from pathlib import Path

import pytest

from carmel.schemas.datasets import Absent
from carmel.services.respecth import RespecthRefusal, parse_idt_record, replay_idt_record
from carmel.services.respecth_archive import load_manifest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/respecth/x10011001-idt-excerpt.xml"
ARCHIVE = next(a for a in load_manifest().archives if a.name == "pentane_indirect_v2.5.zip")


@pytest.mark.parametrize("precision,rounding", [(6, ROUND_DOWN), (28, ROUND_UP)])
def test_ic5_stated_phi_admits_and_replays(precision, rounding):
    """The published rounded composition admits phi 0.3 in either caller context."""
    raw = FIXTURE.read_bytes()
    with localcontext() as context:
        context.prec = precision
        context.rounding = rounding
        record = parse_idt_record(raw, ARCHIVE, "x10011001.xml")
        assert replay_idt_record(record, raw).verified
    assert record.envelope.composition.equivalence_ratio.raw_text == "0.3"
    assert len(record.envelope.series[0].points) == 1
    component = next(c for c in record.envelope.composition.components if c.species_raw_name == "IC5H12")
    assert isinstance(component.role, Absent)


@pytest.mark.parametrize(
    "old,new,reason",
    [
        (b"<value>0.3</value>", b"<value>0.300002</value>", "unmapped_property"),
        (b"<value>0.3</value>", b"<value>0.299998</value>", "unmapped_property"),
        (b'preferredKey="IC5H12"', b'preferredKey="UNKNOWN"', "unmapped_property"),
        (b'type="d/dt max"', b'type="d/dt second max"', "unmapped_ignition_definition"),
    ],
)
def test_ic5_refusing_twins(old, new, reason):
    """Disagreeing phi, unknown demand, and second maxima retain typed refusals."""
    raw = FIXTURE.read_bytes()
    assert raw.count(old) == 1
    with pytest.raises(RespecthRefusal) as caught:
        parse_idt_record(raw.replace(old, new), ARCHIVE, "x10011001.xml")
    assert caught.value.reason.value == reason


def test_ic5_phi_inside_existing_tolerance_is_accepted():
    """The unchanged absolute 1e-6 tolerance accepts a smaller discrepancy."""
    raw = FIXTURE.read_bytes().replace(b"<value>0.3</value>", b"<value>0.3000005</value>")
    record = parse_idt_record(raw, ARCHIVE, "x10011001.xml")
    assert replay_idt_record(record, raw).verified


def test_case3_a1_changes_only_the_three_shock_acceptances():
    """Undoing the six allowed leaves recovers the entire previous frozen digest."""
    document = json.loads((ROOT / "benchmarks/iso-pentane-low-t/contract.json").read_bytes())
    changed = []
    refused = {}
    for study in document["split"]["studies"]:
        if study["set"] != "development":
            continue
        for item in study["files"]:
            if item["acceptance"] == "supported":
                assert item["refusal_reason"] is None
                assert item["low_temperature_rows"] == item["low_temperature_observations"] == 0
                changed.append((item["path"], item["rows"], item["observations"]))
                item["acceptance"] = "refused"
                item["refusal_reason"] = "unmapped_property"
            else:
                reason = item["refusal_reason"]
                refused[reason] = refused.get(reason, 0) + item["rows"]
    assert changed == [("x10011001.xml", 9, 9), ("x10011002.xml", 12, 12), ("x10011003.xml", 7, 7)]
    assert refused == {"rcm_thermo_unavailable": 75, "unmapped_ignition_definition": 18}
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == "8f2d7b4f7cb021ebbdb2819f8b05fa9b359b5e2ab2ed195e0a5ca0017aac0a5e"
