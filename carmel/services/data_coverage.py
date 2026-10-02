"""Acceptance coverage for the pinned curated-data lanes."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

from carmel.services.chemked import ChemkedRefusal, parse_idt_record, replay_idt_record
from carmel.services.chemked_archive import ChemkedManifest, fetch_file
from carmel.services.chemked_archive import load_manifest as load_chemked_manifest
from carmel.services.respecth import (
    RespecthRefusal,
)
from carmel.services.respecth import (
    parse_idt_record as parse_respecth_idt,
)
from carmel.services.respecth import (
    replay_idt_record as replay_respecth_idt,
)
from carmel.services.respecth_archive import (
    RespecthManifest,
    fetch_archive,
    iter_xml_members,
)
from carmel.services.respecth_archive import (
    load_manifest as load_respecth_manifest,
)
from carmel.services.respecth_series import (
    EXPERIMENT_KINDS,
    RespecthExperimentKind,
    parse_series_record,
    read_experiment_type,
    replay_series_record,
)


class _DatasetRecord(Protocol):
    @property
    def envelope(self) -> Any: ...


@dataclass
class CoverageRow:
    source: str
    observable: str
    files_mapped: int = 0
    points_mapped: int = 0
    refusals: Counter[str] = field(default_factory=Counter)
    replay_passed: int = 0
    replay_failed: int = 0
    identity: dict[str, Any] = field(default_factory=dict)
    license: str = "CC-BY-4.0"

    def jsonable(self) -> dict[str, Any]:
        result = asdict(self)
        result["refusals"] = dict(sorted(self.refusals.items()))
        return result


def _manifest_sha256(package: str, filename: str) -> str:
    return hashlib.sha256(resources.files(package).joinpath(filename).read_bytes()).hexdigest()


def _respecth_identity(manifest: RespecthManifest) -> dict[str, Any]:
    return {
        "archive": manifest.source,
        "versions": [
            {"name": item.name, "version": item.osf_version, "sha256": item.sha256} for item in manifest.archives
        ],
        "manifest_sha256": _manifest_sha256("carmel.data", "respecth_manifest.json"),
        "citation": manifest.doi,
    }


def _chemked_identity(manifest: ChemkedManifest) -> dict[str, Any]:
    return {
        "repository": manifest.repository,
        "commit": manifest.commit,
        "manifest_sha256": _manifest_sha256("carmel.data", "chemked_manifest.json"),
    }


def _points(record: _DatasetRecord) -> int:
    return len(record.envelope.series[0].points)


def _new_rows(respecth: RespecthManifest, chemked: ChemkedManifest) -> dict[str, CoverageRow]:
    respecth_identity = _respecth_identity(respecth)
    return {
        "ChemKED|ignition_delay": CoverageRow("ChemKED", "ignition_delay", identity=_chemked_identity(chemked)),
        "ReSpecTh|ignition_delay": CoverageRow(
            "ReSpecTh", "ignition_delay", identity=respecth_identity, license=respecth.license
        ),
        "ReSpecTh|laminar_flame_speed": CoverageRow(
            "ReSpecTh", "laminar_flame_speed", identity=respecth_identity, license=respecth.license
        ),
        "ReSpecTh|speciation": CoverageRow(
            "ReSpecTh", "speciation", identity=respecth_identity, license=respecth.license
        ),
        "ReSpecTh|unclassified": CoverageRow(
            "ReSpecTh", "unclassified", identity=respecth_identity, license=respecth.license
        ),
    }


def build_coverage(*, cache_root: Path, download: bool = True) -> list[CoverageRow]:
    """Load every pinned file/member, map it, and replay every mapped record."""
    respecth = load_respecth_manifest()
    chemked = load_chemked_manifest()
    rows = _new_rows(respecth, chemked)
    chemked_row = rows["ChemKED|ignition_delay"]
    for item in chemked.files:
        try:
            raw = fetch_file(item, chemked, cache_root, download=download)
            chemked_record = parse_idt_record(raw, item.path, item.sha256)
        except ChemkedRefusal as exc:
            chemked_row.refusals[exc.reason.value] += 1
            continue
        chemked_row.files_mapped += 1
        chemked_row.points_mapped += _points(chemked_record)
        try:
            replay_idt_record(chemked_record, raw)
        except ChemkedRefusal:
            chemked_row.replay_failed += 1
        else:
            chemked_row.replay_passed += 1

    for archive in respecth.archives:
        archive_bytes = fetch_archive(archive, manifest=respecth, cache_root=cache_root, download=download)
        for member_path, raw in iter_xml_members(archive_bytes):
            try:
                experiment_type = read_experiment_type(raw)
            except RespecthRefusal as exc:
                rows["ReSpecTh|unclassified"].refusals[exc.reason.value] += 1
                continue
            if experiment_type == "ignition delay measurement":
                try:
                    respecth_record = parse_respecth_idt(raw, archive, member_path)
                except RespecthRefusal as exc:
                    rows["ReSpecTh|ignition_delay"].refusals[exc.reason.value] += 1
                    continue
                row = rows["ReSpecTh|ignition_delay"]
                row.files_mapped += 1
                row.points_mapped += _points(respecth_record)
                replay = replay_respecth_idt(respecth_record, raw)
                if replay.verified:
                    row.replay_passed += 1
                else:
                    row.replay_failed += 1
                continue
            kind = EXPERIMENT_KINDS.get(experiment_type)
            if kind is None:
                rows["ReSpecTh|speciation"].refusals["unmapped_experiment_type"] += 1
                continue
            observable = (
                "laminar_flame_speed" if kind is RespecthExperimentKind.LAMINAR_BURNING_VELOCITY else "speciation"
            )
            row = rows[f"ReSpecTh|{observable}"]
            try:
                series = parse_series_record(raw, archive, member_path)
            except RespecthRefusal as exc:
                row.refusals[exc.reason.value] += 1
            else:
                row.files_mapped += 1
                row.points_mapped += _points(series)
                replay = replay_series_record(series, raw)
                if replay.verified:
                    row.replay_passed += 1
                else:
                    row.replay_failed += 1
    return [rows[key] for key in sorted(rows)]


def coverage_payload(rows: list[CoverageRow]) -> dict[str, Any]:
    return {"generated": date.today().isoformat(), "rows": [row.jsonable() for row in rows]}


def render_table(rows: list[CoverageRow]) -> str:
    headers = ["source", "observable", "files", "points", "refused", "replay", "identity"]
    values = []
    for row in rows:
        refused = ", ".join(f"{key}={value}" for key, value in sorted(row.refusals.items())) or "none"
        identity = row.identity.get("commit") or ";".join(
            f"{item['name']}@v{item['version']}" for item in row.identity.get("versions", [])
        )
        identity = f"{identity}; manifest={row.identity['manifest_sha256']}"
        values.append(
            [
                row.source,
                row.observable,
                str(row.files_mapped),
                str(row.points_mapped),
                refused,
                f"{row.replay_passed}/{row.replay_failed}",
                str(identity),
            ]
        )
    widths = [max(len(headers[i]), *(len(value[i]) for value in values)) for i in range(len(headers))]
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend("  ".join(value[i].ljust(widths[i]) for i in range(len(headers))) for value in values)
    return "\n".join(lines)
