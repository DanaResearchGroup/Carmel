"""Published T3 artifacts retain their old bytes if atomic replacement fails."""

import pytest

from carmel.services import artifacts
from carmel.services.t3_export import write_export


@pytest.mark.parametrize("destination", ["yaml", "report"])
def test_export_atomic_replace_failure_preserves_existing_artifact(monkeypatch, tmp_path, destination):
    output = tmp_path / "points.yaml"
    report = output.with_suffix(".report.json")
    old = b"previous complete artifact\n"
    output.write_bytes(old)
    report.write_bytes(old)
    failing_path = output if destination == "yaml" else report
    replace = artifacts.os.replace

    def fail_replace(source, target):
        if target == failing_path:
            raise OSError("injected replacement failure")
        return replace(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        write_export({"version": 1, "points": []}, {"exported": 0}, output)
    assert failing_path.read_bytes() == old
    assert not list(tmp_path.glob(".*.tmp"))


def test_export_and_report_serialization_bytes_stay_identical(tmp_path):
    output = tmp_path / "nested/points.yaml"
    report = write_export({"version": 1, "points": []}, {"z": [1], "a": "μ"}, output)
    assert output.read_bytes() == b"version: 1\npoints: []\n"
    assert report.read_bytes() == b'{\n  "a": "\\u03bc",\n  "z": [\n    1\n  ]\n}\n'
