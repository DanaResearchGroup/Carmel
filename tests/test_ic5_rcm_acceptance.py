"""Pinned iso-pentane thermo admits RCM histories without relaxing source bounds."""

import copy
import hashlib
import json
import os
import runpy
import zipfile
from decimal import ROUND_DOWN, ROUND_UP, Decimal, localcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from xml.etree import ElementTree

import pytest

from carmel.benchmarks import iso_pentane, n_pentane
from carmel.benchmarks.schema import validate_contract_json
from carmel.schemas.datasets import AbsenceReason, Absent
from carmel.services import respecth
from carmel.services.rcm_history import in_si_decimal
from carmel.services.rcm_thermo import NASA7, check_initial_temperature, isentropic_eoc, solver_composition
from carmel.services.respecth_archive import load_manifest
from carmel.services.t3_export import point_history
from tests.ic5_acceptance import A2_LEAVES, reverse_a2

FIXTURES = Path(__file__).parent / "fixtures"
ARCHIVE = next(a for a in load_manifest().archives if a.name == "pentane_indirect_v2.5.zip")
ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "benchmarks/iso-pentane-low-t/contract.json"


def rcm_excerpt(name="x40011001"):
    """Read one archived observation and its complete source-linked volume history."""
    return (FIXTURES / "respecth" / f"{name}-rcm-excerpt.xml").read_bytes()


@pytest.mark.parametrize("precision,rounding", [(6, ROUND_DOWN), (28, ROUND_UP)])
def test_ic5_rcm_admits_and_replays_without_rewriting_initial_state(precision, rounding):
    """The pinned mixture supports the source history under either caller context."""
    raw = rcm_excerpt()
    with localcontext() as context:
        context.prec = precision
        context.rounding = rounding
        record = respecth.parse_idt_record(raw, ARCHIVE, "x40011001.xml")
        assert respecth.replay_idt_record(record, raw).verified
    assert len(record.envelope.series[0].points) == len(record.rcm_states) == 1
    history, state = point_history(record, 0)
    assert state.initial_temperature.raw_text == "378.00"
    assert state.temperature is None
    assert state.eoc_basis == "derived-isentropic"
    before = record.model_dump_json()
    mixture = {c.species_raw_name: in_si_decimal(c.amount) for c in record.envelope.composition.components}
    temperature, pressure = isentropic_eoc(
        float(in_si_decimal(state.initial_temperature)),
        float(in_si_decimal(state.initial_pressure)),
        history.volume_ratio,
        solver_composition(mixture),
    )
    assert temperature == pytest.approx(886.4981027121144, abs=1e-6)
    assert pressure == pytest.approx(1004790.6627165674, abs=1e-4)
    assert record.model_dump_json() == before


def test_ic5_292_k_point_remains_refused():
    """The real x40011008 first point remains below the nitrogen/argon allowance."""
    raw = rcm_excerpt("x40011008")
    assert b"292.00" in raw
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, ARCHIVE, "x40011008.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE
    assert "295.0..3500.0" in caught.value.detail


def test_admission_depends_only_on_the_added_pinned_thermo(monkeypatch):
    """Removing the new capability restores the original typed reader refusal."""
    monkeypatch.setattr(respecth, "NASA7", MappingProxyType({k: v for k, v in NASA7.items() if k != "IC5H12"}))
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(rcm_excerpt(), ARCHIVE, "x40011001.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.RCM_THERMO_UNAVAILABLE
    assert "without pinned NASA7" in caught.value.detail


def test_ic5_second_maximum_definition_stays_refused():
    """Thermo support grants no new ignition-definition semantics."""
    raw = rcm_excerpt().replace(b'type="d/dt max"', b'type="d/dt second max"')
    with pytest.raises(respecth.RespecthRefusal) as caught:
        respecth.parse_idt_record(raw, ARCHIVE, "x40011001.xml")
    assert caught.value.reason is respecth.RespecthRefusalReason.UNMAPPED_IGNITION_DEFINITION


def test_ic5_coefficients_match_the_pinned_burcat_excerpt():
    """Compare all fourteen polynomial coefficients with the source's fixed columns."""
    raw = (FIXTURES / "rcm/burcat-isopentane.dat").read_bytes()
    lines = raw.decode().splitlines()
    assert lines[0].startswith("C5H12,i-pentane   P10/85")
    coefficients = [float(line[i : i + 15]) for line in lines[1:] for i in range(0, 75, 15)][:14]
    assert NASA7["IC5H12"].high == tuple(coefficients[:7])
    assert NASA7["IC5H12"].low == tuple(coefficients[7:])
    assert NASA7["IC5H12"].temperature_ranges == (200, 1000, 6000)
    assert hashlib.sha256(raw).hexdigest() == "c7235a7bbcb3dde2a2cfb49b0f547fe30f7196bfa47cb1f3d9447fa898d209b3"


@pytest.mark.parametrize(
    "case",
    json.loads((FIXTURES / "rcm/ic5-cantera-3.2.0-isentropic.json").read_bytes())["cases"],
)
def test_ic5_isentrope_matches_independent_cantera_reference(case):
    """Frozen entropy-volume solutions cover the allowance and both polynomial regions."""
    temperature, pressure = isentropic_eoc(
        case["initial_temperature_k"], case["initial_pressure_pa"], case["volume_ratio"], case["composition"]
    )
    assert temperature == pytest.approx(case["temperature_k"], abs=1e-6)
    assert pressure == pytest.approx(case["pressure_pa"], rel=1e-9)


@pytest.mark.parametrize("temperature", [Decimal("292"), Decimal("294.999999999999999999")])
def test_ic5_mixture_retains_exact_295_k_floor(temperature):
    """A tiny sub-boundary source difference must not vanish during float conversion."""
    mixture = {"IC5H12": Decimal("0.01"), "O2": Decimal("0.21"), "N2": Decimal("0.78")}
    with pytest.raises(ValueError, match="polynomial range"):
        check_initial_temperature(temperature, mixture)
    assert check_initial_temperature(Decimal("295"), mixture)


@pytest.mark.parametrize("temperature", [199.999, 6000.001])
def test_ic5_own_temperature_domain_has_no_extra_allowance(temperature):
    """Only nitrogen and argon receive the existing five-kelvin allowance."""
    with pytest.raises(ValueError, match="polynomial range"):
        isentropic_eoc(temperature, 101325, 1, {"IC5H12": 1})


@pytest.mark.parametrize("temperature,expected", [(599.999, 0), (600, 1), (900, 1), (900.001, 0)])
def test_case3_classification_uses_compressed_window_edges(monkeypatch, temperature, expected):
    """Only case 3 classifies by the derived result, with inclusive window edges."""
    record = respecth.parse_idt_record(rcm_excerpt(), ARCHIVE, "x40011001.xml")
    before = record.model_dump_json()
    monkeypatch.setattr(iso_pentane, "isentropic_eoc", lambda *args: (temperature, 1e6))
    assert iso_pentane.series_counts(record) == (1, 1, expected, expected)
    assert n_pentane.series_counts(record) == (1, 1, 0, 0)
    assert record.model_dump_json() == before


def test_classification_counts_observations_and_honors_constant_temperature():
    """Row and scalar counts remain separate when a temperature vector is supplied."""
    record = SimpleNamespace(
        envelope=SimpleNamespace(
            series=(
                SimpleNamespace(
                    constants=(
                        SimpleNamespace(axis_id="temperature", value=SimpleNamespace(canonical_decimal_value="700")),
                    ),
                    points=(
                        SimpleNamespace(coordinates=(), observations=(1, 2)),
                        SimpleNamespace(coordinates=(), observations=(3,)),
                    ),
                ),
            )
        )
    )
    assert iso_pentane.series_counts(record) == n_pentane.series_counts(record) == (2, 3, 2, 3)
    assert n_pentane.series_counts(record, classification_temperatures=[599.999, 900]) == (2, 3, 1, 1)


@pytest.mark.parametrize("temperatures", [[], [600, 700], [float("nan")], [float("inf")], [0], [-1]])
def test_classification_vector_refuses_missing_extra_and_invalid_values(temperatures):
    """Invalid explicit classification must not fall back to source temperature."""
    record = respecth.parse_idt_record(rcm_excerpt(), ARCHIVE, "x40011001.xml")
    with pytest.raises(ValueError, match="finite, positive and aligned"):
        n_pentane.series_counts(record, classification_temperatures=temperatures)


@pytest.mark.parametrize("defect", ["states", "composition", "history"])
def test_case3_classification_refuses_missing_rcm_evidence(monkeypatch, defect):
    """Even an unvalidated in-memory edit cannot manufacture a classification."""
    record = respecth.parse_idt_record(rcm_excerpt(), ARCHIVE, "x40011001.xml")
    if defect == "states":
        record = record.model_copy(update={"rcm_states": ()})
    elif defect == "composition":
        envelope = record.envelope.model_copy(update={"composition": Absent(reason=AbsenceReason.NOT_APPLICABLE)})
        record = record.model_copy(update={"envelope": envelope})
    else:
        monkeypatch.setattr(iso_pentane, "point_history", lambda *args: (None, None))
    with pytest.raises(ValueError, match="RCM classification requires"):
        iso_pentane.series_counts(record)


def test_case3_history_links_survive_reordered_history_declarations(monkeypatch):
    """Each point's initial state and volume ratio follow its link, not history order."""
    root = ElementTree.fromstring(rcm_excerpt())
    group, history = root.findall("dataGroup")
    second_point = copy.deepcopy(group.find("dataPoint"))
    second_point.find("x2").text = "393.00"
    group.append(second_point)
    second_history = copy.deepcopy(history)
    second_history.set("dataPointLink", "2")
    second_history.findall("dataPoint")[9].find("x5").text = "0.07"
    root.insert(list(root).index(history), second_history)
    record = respecth.parse_idt_record(ElementTree.tostring(root), ARCHIVE, "x40011001.xml")
    seen = []

    def solve(temperature, pressure, ratio, mixture):
        """Capture solver inputs and make the two classifications distinguishable."""
        seen.append((temperature, ratio))
        return (600 if temperature == 378 else 901), pressure

    monkeypatch.setattr(iso_pentane, "isentropic_eoc", solve)
    assert iso_pentane.series_counts(record) == (2, 2, 1, 1)
    assert seen[0] == pytest.approx((378, 1 / 0.088703215))
    assert seen[1] == pytest.approx((393, 1 / 0.07))


def test_stated_rcm_temperature_keeps_source_classification():
    """An end-of-compression source condition needs no history-based derivation."""
    pin = next(a for a in load_manifest().archives if a.name.startswith("syngas"))
    raw = (FIXTURES / "respecth/x40001058_19.xml").read_bytes()
    record = respecth.parse_idt_record(raw, pin, "x40001058_19.xml")
    assert iso_pentane.series_counts(record) == n_pentane.series_counts(record)


def test_a2_leaf_reversal_preserves_holdout_and_entire_revision2_document():
    """The explicit ledger recovers both old hashes and leaves every other field intact."""
    document = json.loads(CONTRACT_PATH.read_bytes())
    previous = reverse_a2(document)
    assert document["revision"] == previous["revision"]
    assert document["scoring"] == previous["scoring"]
    for new, old in zip(document["split"]["studies"], previous["split"]["studies"], strict=True):
        if new["set"] == "holdout":
            assert json.dumps(new, ensure_ascii=False) == json.dumps(old, ensure_ascii=False)
        else:
            for nf, of in zip(new["files"], old["files"], strict=True):
                changed = {key for key in nf if nf[key] != of[key]}
                if nf["path"] in A2_LEAVES:
                    expected = {"acceptance", "refusal_reason"}
                    if A2_LEAVES[nf["path"]][1]:
                        expected |= {"low_temperature_rows", "low_temperature_observations"}
                    assert changed == expected
                else:
                    assert not changed
    assert sum(row[1] for row in A2_LEAVES.values()) == 50


@pytest.mark.parametrize("defect", ["low_count", "undeclared_leaf", "missing_transition"])
def test_a2_contract_changes_require_matching_digest(tmp_path, defect):
    """Count tampering, extra edits and missing acceptance cannot hide in the reversal."""
    document = json.loads(CONTRACT_PATH.read_bytes())
    item = document["split"]["studies"][0]["files"][3]
    assert item["path"] == "x40011001.xml"
    if defect == "low_count":
        item["low_temperature_rows"] = 0
    elif defect == "undeclared_leaf":
        document["identity"]["historical_identity_caveat"] += " edited"
    else:
        item.update(acceptance="refused", refusal_reason="rcm_thermo_unavailable")
    with pytest.raises(AssertionError):
        reverse_a2(document)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        iso_pentane.load_contract(path)
    with pytest.raises(ValueError):
        validate_contract_json(path.read_bytes())


@pytest.mark.skipif(not os.environ.get("CARMEL_LOCAL_RUNS"), reason="local pinned archive unavailable")
def test_real_a2_split_and_reproducer_agree_on_every_file():
    """Recompute all counts from verified source bytes without mechanism conversion."""
    archive = Path(os.environ["CARMEL_LOCAL_RUNS"], "i116-alkane-census/evidence/pentane_indirect_v2.5.zip")
    contract = iso_pentane.load_contract(CONTRACT_PATH)
    assert iso_pentane.load_split(contract, archive) == contract.expected_totals
    reproducer = runpy.run_path(str(ROOT / "benchmarks/iso-pentane-low-t/reproduce.py"))
    split = reproducer["frozen_split"](archive.read_bytes())
    for actual, frozen in zip(split["studies"], contract.split.studies, strict=True):
        assert actual["set"] == frozen.set
        assert actual["files"] == [item.model_dump(mode="json") for item in frozen.files]
    with zipfile.ZipFile(archive) as source:
        for study in contract.split.studies:
            for item in study.files:
                if study.set == "development" and item.acceptance == "supported":
                    record = respecth.parse_idt_record(source.read(item.path), ARCHIVE, item.path)
                    before = record.model_dump_json()
                    iso_pentane.series_counts(record)
                    assert record.model_dump_json() == before
