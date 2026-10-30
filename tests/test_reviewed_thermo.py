"""Warnings require reviewed fixed-width values, never a label alone."""

import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carmel.benchmarks import cantera_worker as worker
from carmel.benchmarks.cantera_worker import (
    REVIEWED_CH_NASA_SHA256,
    REVIEWED_OH_NASA_SHA256,
    cantera_validation_refusals,
    nasa7_record_digests,
)


def _source() -> bytes:
    return (Path(__file__).parent / "data/reviewed_excited_thermo.dat").read_bytes()


def _warning(label: str, quantity: str = "h/RT") -> str:
    return f"NasaPoly2::validate: \nFor species {label}, discontinuity in {quantity} detected at Tmid = 1000"


@pytest.mark.parametrize(
    ("original", "alias", "digest"), [("OH*", "OHV", REVIEWED_OH_NASA_SHA256), ("CH*", "CHV", REVIEWED_CH_NASA_SHA256)]
)
def test_alias_requires_exact_reviewed_fields(original: str, alias: str, digest: str) -> None:
    source = _source().replace(original.encode(), alias.encode())
    assert nasa7_record_digests((source,))[alias] == digest
    assert cantera_validation_refusals([_warning(alias)], (source,)) == []
    # One digit in a coefficient, including case-2 original names, revokes approval.
    needle = b"0.02882730" if original == "OH*" else b"0.02196223"
    changed = source.replace(needle, needle[:-1] + b"9")
    assert cantera_validation_refusals([_warning(alias)], (changed,)) == [_warning(alias)]
    changed = changed.replace(alias.encode(), original.encode())
    assert cantera_validation_refusals([_warning(original)], (changed,)) == [_warning(original)]


def test_other_species_warning_types_and_temperatures_are_refused() -> None:
    source = _source().replace(b"OH*", b"XYZ")
    messages = [
        _warning("XYZ"),
        _warning("OH*", "cp/R"),
        _warning("OH*").replace("1000", "999"),
        _warning("OH*").replace("NasaPoly2::validate", "Other::validate"),
        _warning("OH*") + "\nOther warning",
    ]
    assert cantera_validation_refusals(messages, (source, _source())) == messages


def test_first_duplicate_record_wins_and_malformed_records_fail_closed() -> None:
    source = _source()
    changed = source.replace(b"0.02882730", b"0.02882739")
    warning = _warning("OH*")
    assert cantera_validation_refusals([warning], (changed, source)) == [warning]
    assert cantera_validation_refusals([warning], (source, changed)) == []
    malformed = source.replace(b"0.02882730E+02", b"NOT_A_NUMBER ")
    assert nasa7_record_digests((malformed, source))["OH*"] is None
    assert cantera_validation_refusals([warning], (malformed, source)) == [warning]
    assert cantera_validation_refusals([warning], (source[: source.index(b"CH*")].replace(b"    3", b"    9"),)) == [
        warning
    ]


def test_full_cantera_message_is_accepted_but_extra_text_is_not() -> None:
    message = (
        _warning("OH*")
        + "\n\tValue computed using low-temperature polynomial:  53.62056162666667\n"
        + "\tValue computed using high-temperature polynomial: 53.5841554314\n"
    )
    assert cantera_validation_refusals([message], (_source(),)) == []
    assert cantera_validation_refusals([message + "other warning"], (_source(),)) == [message + "other warning"]


@pytest.mark.parametrize("field", [45, 55, 65])
def test_invalid_temperature_fields_cannot_receive_reviewed_digest(field: int) -> None:
    lines = _source().splitlines(keepends=True)
    header = next(i for i, line in enumerate(lines) if line.startswith(b"OH*"))
    lines[header] = lines[header][:field] + b" NOT_FLOAT" + lines[header][field + 10 :]
    source = b"".join(lines)
    assert nasa7_record_digests((source,))["OH*"] is None
    assert cantera_validation_refusals([_warning("OH*")], (source,)) == [_warning("OH*")]


@pytest.mark.parametrize("source_kind", ["reviewed", "altered", "embedded_override"])
def test_worker_validation_reads_actual_numeric_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    mechanism, thermo = tmp_path / "mechanism.inp", tmp_path / "thermo.dat"
    source = _source().replace(b"OH*", b"OHV")
    changed = source.replace(b"0.02882730", b"0.02882739")
    mechanism.write_bytes(changed if source_kind == "embedded_override" else b"SPECIES\nOHV\nEND\n")
    thermo.write_bytes(changed if source_kind == "altered" else source)
    converted = tmp_path / "converted.yaml"
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "mechanism_path": str(mechanism),
                "thermo_path": str(thermo),
                "converted_yaml_path": str(converted),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "rows": [],
            }
        )
    )

    def convert(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        converted.write_text("phases: []\nspecies: []\nreactions: []\n")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def solution(_: Path) -> object:
        warnings.warn(_warning("OHV"), stacklevel=2)
        return object()

    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(__version__="test", Solution=solution))
    monkeypatch.setattr(worker.subprocess, "run", convert)
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request), str(response)])
    assert worker.main() == 0
    result = json.loads(response.read_text())
    assert result["validation_warnings"] == [_warning("OHV")]
    assert result["status"] == ("ok" if source_kind == "reviewed" else "refused")
    if source_kind != "reviewed":
        assert result["refusal"] == "unallowlisted_cantera_validation_warning"
