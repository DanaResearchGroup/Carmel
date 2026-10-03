"""Replay failures abort intake; parser refusals remain countable."""

import hashlib
from dataclasses import replace

import pytest

from Carmel import main
from carmel.services import chemked, t3_export
from carmel.services.chemked_archive import ChemkedFile, ChemkedManifest
from carmel.services.respecth_archive import load_manifest
from tests.test_rcm_export import chemked_history, respecth_record


def source_lane(monkeypatch, source, *, include_bad=False, mismatch=False):
    if source == "chemked":
        raw = chemked_history()
        members = [("first.yaml", raw), ("second.yaml", raw)]
        if include_bad:
            members.append(("bad.yaml", b"bad"))
        manifest = ChemkedManifest(
            "fixture/repository",
            "0" * 40,
            tuple(ChemkedFile(name, hashlib.sha256(data).hexdigest()) for name, data in members),
        )
        monkeypatch.setattr(t3_export, "load_manifest", lambda: manifest)
        monkeypatch.setattr(t3_export, "fetch_file", lambda item, *args, **kwargs: dict(members)[item.path])
        if mismatch:

            def parse(raw, path, sha):
                record = chemked.parse_idt_record(raw, path, sha)
                return replace(record, fuels=("forged",)) if path == "second.yaml" else record

            monkeypatch.setattr(t3_export, "parse_idt_record", parse)
    else:
        record, raw = respecth_record()
        manifest = load_manifest()
        pin = next(a for a in manifest.archives if a.name == record.archive.archive_name)
        monkeypatch.setattr(t3_export, "respecth_manifest", lambda: replace(manifest, archives=(pin,)))
        monkeypatch.setattr(t3_export, "fetch_archive", lambda *args, **kwargs: b"archive")
        members = [("first.xml", raw), ("second.xml", raw)]
        if include_bad:
            members.append(("bad.xml", b"bad"))
        monkeypatch.setattr(t3_export, "iter_xml_members", lambda _: iter(members))
        if mismatch:
            from carmel.services import respecth

            def parse(raw, archive, member):
                record = respecth.parse_idt_record(raw, archive, member)
                return record.model_copy(update={"species_identifiers": ()}) if member == "second.xml" else record

            monkeypatch.setattr(t3_export, "parse_respecth", parse)


@pytest.mark.parametrize("source", ["chemked", "respecth"])
def test_load_records_aborts_replay_mismatch_after_a_valid_record(monkeypatch, tmp_path, source):
    source_lane(monkeypatch, source, mismatch=True)
    with pytest.raises(ValueError) as caught:
        t3_export.load_records(source=source, cache_root=tmp_path, download=False)
    if source == "chemked":
        assert isinstance(caught.value, chemked.ChemkedRefusal)
        assert caught.value.reason is chemked.ChemkedRefusalReason.UNRESOLVABLE_PATH
    else:
        assert "source replay failed for second.xml" in str(caught.value)


@pytest.mark.parametrize("source", ["chemked", "respecth"])
def test_load_records_counts_parser_refusals_without_losing_valid_records(monkeypatch, tmp_path, source):
    source_lane(monkeypatch, source, include_bad=True)
    records, refusals = t3_export.load_records(source=source, cache_root=tmp_path, download=False)
    assert len(records) == 2
    assert sum(refusals.values()) == 1
    assert all(reason.startswith(source + ":") for reason in refusals)


@pytest.mark.parametrize("source", ["chemked", "respecth"])
def test_cli_replay_mismatch_refuses_without_writing_partial_export(monkeypatch, tmp_path, capsys, source):
    source_lane(monkeypatch, source, mismatch=True)
    output = tmp_path / "partial.yaml"
    assert main(["data", "export-t3", "--source", source, "--output", str(output), "--offline"]) == 1
    assert "Refusing T3 export" in capsys.readouterr().err
    assert not output.exists()
    assert not output.with_suffix(".report.json").exists()
