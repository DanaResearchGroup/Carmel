"""ReSpecTh apparatus whitespace must not split main and RCM census counts."""

import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from importlib import resources
from io import BytesIO
from pathlib import Path

import pytest

from carmel.schemas.campaign import ReactorType
from carmel.services import data_coverage
from carmel.services.respecth import parse_idt_record, replay_idt_record
from carmel.services.respecth_archive import cached_archive_path, load_manifest


@pytest.mark.parametrize(
    "kind", ["rapid compression machine", " rapid compression machine ", "\n\t rapid compression machine \t\n"]
)
def test_padded_rcm_kind_keeps_main_and_subset_counts_equal(monkeypatch, tmp_path: Path, kind: str) -> None:
    fixture_root = Path(__file__).parent / "fixtures"
    root = ET.fromstring((fixture_root / "respecth/x40001039.xml").read_bytes())
    root.find("apparatus/kind").text = kind
    raw = ET.tostring(root)
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        bundle.writestr("padded.xml", raw)
    archive_bytes = stream.getvalue()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    chemked_raw = (fixture_root / "chemked/Bec_2014_2-b_20atm.yaml").read_bytes()
    chemked_sha = hashlib.sha256(chemked_raw).hexdigest()
    cache = tmp_path / "cache"
    (cache / "sha256").mkdir(parents=True)
    cached_archive_path(cache, archive_sha).write_bytes(archive_bytes)
    cached_archive_path(cache, chemked_sha).write_bytes(chemked_raw)

    manifest_root = tmp_path / "manifests"
    manifest_root.mkdir()
    respecth = json.loads(Path("carmel/data/respecth_manifest.json").read_text())
    pin = next(item for item in respecth["archives"] if item["name"].startswith("syngas"))
    pin.update(sha256=archive_sha, size=len(archive_bytes))
    respecth["archives"] = [pin]
    (manifest_root / "respecth_manifest.json").write_text(json.dumps(respecth))
    chemked = json.loads(Path("carmel/data/chemked_manifest.json").read_text())
    chemked["files"] = [{"path": "fixture.yaml", "sha256": chemked_sha}]
    (manifest_root / "chemked_manifest.json").write_text(json.dumps(chemked))
    packaged_files = resources.files
    # Substitute only the bundled-manifest filesystem boundary; real bounded
    # reads, hashes, unpacking, parsing, replay and census counting all run.
    monkeypatch.setattr(
        resources, "files", lambda package: manifest_root if package == "carmel.data" else packaged_files(package)
    )

    rows = {(row.source, row.observable): row for row in data_coverage.build_coverage(cache_root=cache, download=False)}
    main = rows[("ReSpecTh", "ignition_delay")]
    subset = rows[("ReSpecTh", "ignition_delay_rcm")]
    assert main.files_mapped == subset.files_mapped == 1
    assert main.points_mapped == subset.points_mapped == 18
    assert main.replay_passed == subset.replay_passed == 1
    assert main.replay_failed == subset.replay_failed == 0
    assert main.refusals == subset.refusals == {}

    record = parse_idt_record(raw, load_manifest().archives[0], "padded.xml")
    assert record.apparatus.device_class is ReactorType.RCM
    assert record.apparatus.kind_raw.raw == kind
    assert replay_idt_record(record, raw).verified


def test_export_refuses_eoc_rounding_onto_a_sample() -> None:
    from carmel.services.chemked import parse_idt_record as parse_chemked
    from carmel.services.t3_export import export_idt
    from tests.test_rcm_review_round7 import history_source

    pytest.importorskip("rdkit")
    records = [
        parse_chemked(history_source([(0, 100), (1, 1), (2, "1E18")], eoc), f"eoc-{i}.yaml")
        for i, eoc in enumerate(("1.00000000000000001", "1", "0.5"))
    ]
    assert records[0].rcm_histories[0].volume_ratio == pytest.approx(100 / 11)
    payload, report = export_idt(records)
    assert len(payload["points"]) == 2
    assert report["refused"] == {"t3_constraint": 1}
    assert [item["file"] for item in report["refusals"]] == ["eoc-0.yaml"]


def test_isentropic_composition_tolerance_has_inclusive_exact_boundaries() -> None:
    from carmel.services.rcm_thermo import isentropic_eoc

    for fraction in (0.995, 1.005):
        assert isentropic_eoc(300, 101325, 8, {"AR": fraction}) == pytest.approx((1200, 3242400))
    for fraction in (0.9949, 1.0051):
        with pytest.raises(ValueError, match="composition must be normalized"):
            isentropic_eoc(300, 101325, 8, {"AR": fraction})


def test_chemked_refuses_rewritten_integer_lexemes() -> None:
    from carmel.services import chemked
    from tests.test_rcm_review_round7 import history_source

    raw = history_source([(0, 1000), (1, "0.01"), (2, "0.02")], 1)
    chemked.replay_idt_record(chemked.parse_idt_record(raw, "integer.yaml"), raw)
    for lexeme in (b"0x10", b"1_000", b"+1", b"-0", b"9" * 5000):
        with pytest.raises(chemked.ChemkedRefusal) as caught:
            chemked.parse_idt_record(raw.replace(b"1000", lexeme), "integer.yaml")
        assert caught.value.reason is chemked.ChemkedRefusalReason.MALFORMED_YAML


def test_export_refuses_float_histories_without_compression() -> None:
    from carmel.services import chemked, t3_export
    from tests.test_rcm_review_round7 import history_source

    pytest.importorskip("rdkit")
    records = [
        chemked.parse_idt_record(history_source([(0, initial), (1, 1), (2, 1)], eoc), f"{initial}-{eoc}.yaml")
        for initial in ("1.00000000000000001", 2)
        for eoc in (1, 1.5)
    ]
    _, report = t3_export.export_idt(records)
    assert report["exported"] == 2
    assert report["refused"] == {"t3_constraint": 2}
