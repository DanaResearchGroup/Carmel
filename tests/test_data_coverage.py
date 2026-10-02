"""Tests for the deterministic curated-data acceptance report."""

import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from Carmel import main
from carmel.services.data_coverage import CoverageRow, coverage_payload, render_table


def _rows() -> list[CoverageRow]:
    return [
        CoverageRow(
            "ReSpecTh",
            "speciation",
            files_mapped=2,
            points_mapped=5,
            refusals=Counter({"schema_rejected": 1}),
            replay_passed=2,
            identity={"commit": "never", "manifest_sha256": "manifest"},
        ),
        CoverageRow(
            "ChemKED",
            "ignition_delay",
            files_mapped=3,
            points_mapped=7,
            identity={"commit": "pin", "manifest_sha256": "manifest"},
        ),
    ]


def test_coverage_table_has_stable_columns_and_order() -> None:
    table = render_table(_rows())
    assert table.splitlines()[0].split() == ["source", "observable", "files", "points", "refused", "replay", "identity"]
    assert table.splitlines()[2].startswith("ReSpecTh")
    assert table.splitlines()[3].startswith("ChemKED")


def test_coverage_json_schema_is_machine_readable() -> None:
    payload = coverage_payload(_rows())
    decoded = json.loads(json.dumps(payload))
    assert set(decoded) == {"generated", "rows"}
    assert set(decoded["rows"][0]) == {
        "source",
        "observable",
        "files_mapped",
        "points_mapped",
        "refusals",
        "replay_passed",
        "replay_failed",
        "identity",
        "license",
    }


def test_coverage_command_returns_nonzero_for_replay_failure(monkeypatch, capsys) -> None:
    import carmel.services.data_coverage as coverage

    monkeypatch.setattr(coverage, "build_coverage", lambda **_: [CoverageRow("x", "y", replay_failed=1)])
    assert main(["data", "coverage", "--json", "--offline"]) == 1
    assert json.loads(capsys.readouterr().out)["rows"][0]["replay_failed"] == 1


def test_build_coverage_replays_local_fixture_lanes(monkeypatch, tmp_path: Path) -> None:
    import carmel.services.data_coverage as coverage
    from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest
    from carmel.services.respecth_archive import load_manifest as load_respecth_manifest

    fixture_root = Path(__file__).parent / "fixtures"
    respecth = load_respecth_manifest()
    chemked_raw = (fixture_root / "chemked" / "Bec_2014_2-b_20atm.yaml").read_bytes()
    chemked = ChemkedManifest(
        repository="fixture/repository",
        commit="0" * 40,
        files=(ChemkedFile("fixture.yaml", hashlib.sha256(chemked_raw).hexdigest()),),
    )
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        for path in (fixture_root / "respecth").glob("*.xml"):
            archive.writestr(path.name, path.read_bytes())
    monkeypatch.setattr(coverage, "load_respecth_manifest", lambda: respecth)
    monkeypatch.setattr(coverage, "load_chemked_manifest", lambda: chemked)
    monkeypatch.setattr(coverage, "fetch_file", lambda *args, **kwargs: chemked_raw)
    monkeypatch.setattr(coverage, "fetch_archive", lambda *args, **kwargs: bundle.getvalue())
    rows = coverage.build_coverage(cache_root=tmp_path, download=False)
    by_key = {(row.source, row.observable): row for row in rows}
    expected = {
        ("ChemKED", "ignition_delay"),
        ("ReSpecTh", "ignition_delay"),
        ("ReSpecTh", "laminar_flame_speed"),
        ("ReSpecTh", "speciation"),
    }
    for key in expected:
        row = by_key[key]
        assert row.files_mapped > 0
        assert row.points_mapped > 0
        assert row.replay_failed == 0
        assert row.replay_passed == row.files_mapped


def test_build_coverage_records_typed_refusals_and_replay_failures(monkeypatch, tmp_path: Path) -> None:
    import carmel.services.data_coverage as coverage
    from carmel.services.chemked import ChemkedRefusalReason
    from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest
    from carmel.services.respecth import RespecthRefusal, RespecthRefusalReason
    from carmel.services.respecth_archive import load_manifest as load_respecth_manifest

    respecth = load_respecth_manifest()
    chemked = ChemkedManifest(
        "fixture/repository", "0" * 40, (ChemkedFile("good", "0" * 64), ChemkedFile("bad", "1" * 64))
    )
    fake_record = SimpleNamespace(envelope=SimpleNamespace(series=[SimpleNamespace(points=[1])]))
    replay_failure = SimpleNamespace(verified=False)
    monkeypatch.setattr(coverage, "load_respecth_manifest", lambda: respecth)
    monkeypatch.setattr(coverage, "load_chemked_manifest", lambda: chemked)
    monkeypatch.setattr(coverage, "fetch_file", lambda item, *args, **kwargs: item.path.encode())
    monkeypatch.setattr(
        coverage,
        "parse_idt_record",
        lambda raw, *args: (
            fake_record
            if raw == b"good"
            else (_ for _ in ()).throw(coverage.ChemkedRefusal(ChemkedRefusalReason.SCHEMA_REJECTED, "bad"))
        ),
    )
    monkeypatch.setattr(
        coverage,
        "replay_idt_record",
        lambda record, raw: (_ for _ in ()).throw(
            coverage.ChemkedRefusal(ChemkedRefusalReason.UNRESOLVABLE_PATH, "drift")
        ),
    )
    members = [(name, name.encode()) for name in ("badxml", "idtbad", "idtok", "unknown", "seriesbad", "seriesok")]
    monkeypatch.setattr(coverage, "fetch_archive", lambda *args, **kwargs: b"archive")
    monkeypatch.setattr(coverage, "iter_xml_members", lambda raw: iter(members))

    def read_type(raw: bytes) -> str:
        if raw == b"badxml":
            raise RespecthRefusal(RespecthRefusalReason.MALFORMED_XML, "bad")
        return (
            "ignition delay measurement"
            if raw.startswith(b"idt")
            else "laminar burning velocity measurement"
            if raw.startswith(b"series")
            else "unknown"
        )

    monkeypatch.setattr(coverage, "read_experiment_type", read_type)
    monkeypatch.setattr(
        coverage,
        "parse_respecth_idt",
        lambda raw, *args: (
            (_ for _ in ()).throw(RespecthRefusal(RespecthRefusalReason.SCHEMA_REJECTED, "bad"))
            if raw == b"idtbad"
            else fake_record
        ),
    )
    monkeypatch.setattr(coverage, "replay_respecth_idt", lambda *args: replay_failure)
    monkeypatch.setattr(
        coverage,
        "parse_series_record",
        lambda raw, *args: (
            (_ for _ in ()).throw(RespecthRefusal(RespecthRefusalReason.UNMAPPED_UNIT, "bad"))
            if raw == b"seriesbad"
            else fake_record
        ),
    )
    monkeypatch.setattr(coverage, "replay_series_record", lambda *args: replay_failure)
    rows = coverage.build_coverage(cache_root=tmp_path, download=False)
    by_key = {(row.source, row.observable): row for row in rows}
    assert by_key[("ChemKED", "ignition_delay")].replay_failed == 1
    assert by_key[("ReSpecTh", "unclassified")].refusals["malformed_xml"] > 0
    assert by_key[("ReSpecTh", "ignition_delay")].replay_failed >= 1
    assert by_key[("ReSpecTh", "laminar_flame_speed")].replay_failed >= 1
