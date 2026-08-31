# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""The project's FIRST real condition-set target, wired for durable production.

This module is the single definition of one condition set: Table 1, page 4 of
the paper whose sha256 is :data:`TARGET_DOCUMENT_SHA256` -- the "Measurement
conditions" table the i024 condition-set work grounded cell-by-cell. It carries
the measurement conditions bound to the flame-speed experiments the paper
reports: the fuel and oxidiser identities (categorical), and the equivalence
ratio and pressure ranges the extractor deliberately refuses to flatten
(unextracted). Two deliberate omissions are load-bearing and preserved verbatim
from the acceptance test's rationale -- see below.

WHY IT EXISTS. The condition-set production path
(:func:`carmel.services.condition_set_producer.produce_condition_set_from_artifact`)
works and is proven by :mod:`tests.test_condition_set_target_acceptance`, but
that acceptance test stages the document into a throw-away ``tmp_path`` and
stores the envelope there: the artifact is built and destroyed inside the test,
so the project had a working conditions pipeline and zero durable output -- the
exact gap :mod:`carmel.services.tabular_dataset_target` closed for the dataset
lane. This module holds the ONE definition of that condition set -- footprint,
table key, categoricals, ranges, subject and attribution -- so both the
acceptance test and the durable production entry point
(:func:`produce_and_store_target`, exposed as ``carmel store-condition-set``)
construct it from the same source, and :func:`render_condition_set_text` renders
the stored conditions so an operator can read them rather than a sha256.

TWO DELIBERATE OMISSIONS, preserved. (1) The 25 deg-C TEMPERATURE row is NOT
stored: the unit is absent from both lanes (zero U+00B0 in the extracted text,
and no cell equals "degC"), so it cannot ground a scalar's required unit quote,
and no honest ``UnextractedReason`` classifies a clean single value -- so it is
neither laundered into a scalar nor into a range. Its two "25" cells stay in the
inventory grid, uncited. (2) The attribution span is SUPPORT-ONLY: nothing
records what ``OWN_EXPERIMENT`` MEANS, so replay files it as the one remaining
unchecked semantic claim. Both facts are why the honest replay outcome is
``overall_outcome=UNVERIFIABLE`` with ``evidence_outcome=VERIFIED`` -- and this
module does nothing to drive either green.

It does NOT reimplement the producer, the grounding types, or the store: it
wires what already exists. It is deliberately scoped to this ONE document and
table -- generalising to other papers is a separate ticket, not this one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carmel.paths import default_workspaces_root
from carmel.schemas.datasets import (
    CaptionLabelKey,
    ConditionAttribution,
    ConditionSetEnvelope,
    DeviceClassDeclaration,
    EmbeddedTableInventory,
    TableCellLocator,
    UnextractedReason,
    UnresolvedSubject,
    iter_source_refs,
)
from carmel.services import units
from carmel.services.condition_set_bridge import store_condition_set_envelope
from carmel.services.condition_set_producer import (
    CategoricalConditionSpec,
    DeviceClassSpec,
    TableCellGrounding,
    UnextractedConditionSpec,
    produce_condition_set_from_artifact,
)
from carmel.services.dataset_store import StoredDataset, canonical_json_bytes
from carmel.services.pdf_fragments import extract_fragments
from carmel.services.pdf_table_record import inventory_record_payload
from carmel.services.pdf_tables import ClaimedFootprint, build_inventory

__all__ = [
    "ATTRIBUTION_QUOTE",
    "SUBJECT_OCCURRENCE",
    "SUBJECT_QUOTE",
    "TARGET_ATTRIBUTION",
    "TARGET_CAMPAIGN",
    "TARGET_CATEGORICALS",
    "TARGET_DOCUMENT_SHA256",
    "TARGET_RANGES",
    "TARGET_TABLE_FOOTPRINT",
    "TARGET_TABLE_KEY",
    "TARGET_WORKSPACES_ROOTS",
    "ConditionSetTargetError",
    "StoredTargetConditionSet",
    "build_categoricals",
    "build_embedded_inventory",
    "build_envelope",
    "build_unextracted",
    "locate_target_workspace",
    "produce_and_store_target",
    "read_target_raw",
    "render_condition_set_text",
    "write_condition_set_export",
]

#: The raw artifact sha256 of the source document (10.1016-j.ijhydene.2013.10.164).
TARGET_DOCUMENT_SHA256 = "9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b"

#: The campaign whose literature store holds the document, relative to a
#: workspaces root. The document was admitted into this campaign, so its evidence
#: store -- and the condition-set store the produced envelope is written into --
#: live under ``<workspaces-root>/<TARGET_CAMPAIGN>``.
TARGET_CAMPAIGN = "live-syngas"

#: Both roots the acceptance test searches, in order: the packaged default, then
#: the operator's ``~/runs/carmel/workspaces``. Discovery mirrors the test exactly
#: so the durable entry point finds the same document the test proves against.
TARGET_WORKSPACES_ROOTS = (default_workspaces_root(), Path.home() / "runs/carmel/workspaces")

#: The registered whole-table footprint -- this project's own box claim, identical
#: to the one tests.test_target_table_acceptance measured the 9x3 grid under.
TARGET_TABLE_FOOTPRINT = ClaimedFootprint(
    page=4,
    x_start=50.0,
    x_end=290.0,
    y_top=145.0,
    y_bottom=45.0,
    caption_text="Table1–Measurementconditions.",
    caption_x_start=53.0,
    caption_baseline_y=148.8,
)

TARGET_TABLE_KEY = CaptionLabelKey(label="Table 1")

#: Whose conditions these are asserted to be -- an extractor ASSERTION, recorded
#: unverified. The attribution span grounds only WHERE it was read, never that it
#: is correct, which is why replay leaves it as the one unchecked semantic claim.
TARGET_ATTRIBUTION = ConditionAttribution.OWN_EXPERIMENT

#: The apparatus is named unambiguously and repeatedly ("heat flux method (HFM)");
#: occurrence 3 is "...using the heat flux method at elevated pressure", the
#: authors' own measurements. The 'fl' is the ligature U+FB02, verbatim as the
#: extraction yields it.
SUBJECT_QUOTE = "heat ﬂux method"
SUBJECT_OCCURRENCE = 3

#: "were conducted" appears exactly once -- "Experiments were conducted at
#: elevated pressure".
ATTRIBUTION_QUOTE = "were conducted"

#: (claim_id, label, token, label_row, label_col, token_row, token_col) for the 9
#: genuine categorical cells: Fuel across both columns, then every Oxidizer cell.
TARGET_CATEGORICALS: tuple[tuple[str, str, str, int, int, int, int], ...] = (
    ("cat0_fuel_c1", "Fuel", "H2/CO(50:50%)", 0, 0, 0, 1),
    ("cat1_fuel_c2", "Fuel", "H2/CO(85:15%)", 0, 0, 0, 2),
    ("cat2_oxidizer_r1c1", "Oxidizer", "Air", 1, 0, 1, 1),
    ("cat3_oxidizer_r2c1", "Oxidizer", "O2/N2(15:85%)", 1, 0, 2, 1),
    ("cat4_oxidizer_r2c2", "Oxidizer", "O2/N2(15:85%)", 1, 0, 2, 2),
    ("cat5_oxidizer_r3c1", "Oxidizer", "O2/N2(10:90%)", 1, 0, 3, 1),
    ("cat6_oxidizer_r3c2", "Oxidizer", "O2/He(12:88%)", 1, 0, 3, 2),
    ("cat7_oxidizer_r4c1", "Oxidizer", "O2/He(10:90%)", 1, 0, 4, 1),
    ("cat8_oxidizer_r5c1", "Oxidizer", "O2/He(12.5:87.5%)", 1, 0, 5, 1),
)

#: (statement_id, label, statement, quantity_kind, label_row, label_col,
#: stmt_row, stmt_col) for the four ranges the extractor REFUSES to reduce to a
#: single value. The dashes are real en-dashes (U+2013), verbatim.
TARGET_RANGES: tuple[tuple[str, str, str, units.QuantityKind, int, int, int, int], ...] = (
    ("unx0_phi_c1", "φ", "0.6–1.0", units.QuantityKind.EQUIVALENCE_RATIO, 6, 0, 6, 1),
    ("unx1_phi_c2", "φ", "0.5–0.7", units.QuantityKind.EQUIVALENCE_RATIO, 6, 0, 6, 2),
    ("unx2_pressure_c1", "P(atm)", "1–9", units.QuantityKind.PRESSURE, 8, 0, 8, 1),
    ("unx3_pressure_c2", "P(atm)", "1–8", units.QuantityKind.PRESSURE, 8, 0, 8, 2),
)


class ConditionSetTargetError(Exception):
    """The target condition set could not be produced or stored honestly.

    Raised for a precondition the build depends on and cannot proceed without --
    the store not holding the document, the stored bytes not being the measured
    document, or the table grid refusing. Never a way to route around a refusal:
    each case fails closed with the reason named.
    """


def _raw_path(workspace_root: Path) -> Path:
    return workspace_root / "evidence" / "literature" / TARGET_DOCUMENT_SHA256 / "raw.bin"


def locate_target_workspace(roots: tuple[Path, ...] = TARGET_WORKSPACES_ROOTS) -> Path | None:
    """Return the campaign workspace holding the target document, or ``None``.

    Mirrors the acceptance test's discovery: search each root for
    ``<root>/<TARGET_CAMPAIGN>/evidence/literature/<sha>/raw.bin`` and return that
    campaign workspace (the directory the condition-set store is written into).
    """
    for root in roots:
        if _raw_path(root / TARGET_CAMPAIGN).exists():
            return root / TARGET_CAMPAIGN
    return None


def read_target_raw(workspace_root: Path) -> bytes:
    """Read and authenticate the target document's ``raw.bin`` from a workspace.

    Raises:
        ConditionSetTargetError: The document is not stored under
            ``workspace_root``, its stored bytes cannot be read, or they are not
            the measured document.
    """
    raw_path = _raw_path(workspace_root)
    if not raw_path.exists():
        raise ConditionSetTargetError(f"target document is not stored under {workspace_root}: no {raw_path}")
    try:
        raw = raw_path.read_bytes()
    except OSError as exc:
        raise ConditionSetTargetError(f"cannot read the stored raw.bin at {raw_path}: {exc}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != TARGET_DOCUMENT_SHA256:
        raise ConditionSetTargetError(
            f"stored raw.bin is {actual}, not the measured {TARGET_DOCUMENT_SHA256}; refusing to build"
        )
    return raw


def build_embedded_inventory(raw: bytes) -> tuple[EmbeddedTableInventory, dict[str, Any]]:
    """Derive the registered table's cell grid from ``raw`` and embed it.

    Fails closed: the grid must refuse nothing -- the precondition the acceptance
    test asserts. A refused grid is a defect to surface, never something to
    ground a partial condition set on. Returns the embedded inventory and the raw
    record payload (the latter for callers that re-verify the grid against the
    bytes, as the acceptance test does).

    Raises:
        ConditionSetTargetError: The grid refused.
    """
    inventory = build_inventory(extract_fragments(raw), TARGET_TABLE_FOOTPRINT)
    if inventory.refusals != ():
        raise ConditionSetTargetError(f"the target grid refused: {inventory.refusals}")
    payload = inventory_record_payload(inventory, raw_sha256=TARGET_DOCUMENT_SHA256)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    embedded = EmbeddedTableInventory(
        inventory_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        raw_sha256=TARGET_DOCUMENT_SHA256,
        canonical_json=canonical,
    )
    return embedded, payload


def _cell_of(embedded: EmbeddedTableInventory, row: int, col: int) -> TableCellGrounding:
    return TableCellGrounding(table_key=TARGET_TABLE_KEY, row=row, col=col, inventory=embedded)


def build_categoricals(embedded: EmbeddedTableInventory) -> tuple[CategoricalConditionSpec, ...]:
    """The 9 categorical conditions, each grounded to two cells of the grid."""
    return tuple(
        CategoricalConditionSpec(
            claim_id=cid,
            label_quote=label,
            token_quote=token,
            label_cell=_cell_of(embedded, lr, lc),
            token_cell=_cell_of(embedded, tr, tc),
        )
        for cid, label, token, lr, lc, tr, tc in TARGET_CATEGORICALS
    )


def build_unextracted(embedded: EmbeddedTableInventory) -> tuple[UnextractedConditionSpec, ...]:
    """The 4 refused ranges, each recorded with reason VALUE_RANGE and its span."""
    return tuple(
        UnextractedConditionSpec(
            statement_id=sid,
            label_quote=label,
            statement_quote=statement,
            reason=UnextractedReason.VALUE_RANGE,
            quantity_kind=qk,
            label_cell=_cell_of(embedded, lr, lc),
            statement_cell=_cell_of(embedded, sr, sc),
        )
        for sid, label, statement, qk, lr, lc, sr, sc in TARGET_RANGES
    )


def build_envelope(workspace_root: Path, embedded: EmbeddedTableInventory) -> ConditionSetEnvelope:
    """Assemble the fully validated envelope through the production producer.

    The temperature row is deliberately absent (no groundable unit) and the
    attribution span is support-only -- see the module docstring; neither is a
    gap to close here, both are what the honest record looks like.
    """
    return produce_condition_set_from_artifact(
        workspace_root,
        sha256=TARGET_DOCUMENT_SHA256,
        attribution=TARGET_ATTRIBUTION,
        attribution_quote=ATTRIBUTION_QUOTE,
        subject=DeviceClassSpec(label_quote=SUBJECT_QUOTE, label_occurrence=SUBJECT_OCCURRENCE),
        categoricals=build_categoricals(embedded),
        unextracted=build_unextracted(embedded),
    )


@dataclass(frozen=True, slots=True)
class StoredTargetConditionSet:
    """The durable outcome of :func:`produce_and_store_target`.

    ``sha256`` and ``path`` locate the stored condition-set envelope; ``envelope``
    is the in-memory object that was stored, kept so a caller can render its
    export without a second load.
    """

    sha256: str
    path: Path
    envelope: ConditionSetEnvelope


def produce_and_store_target(workspace_root: Path) -> StoredTargetConditionSet:
    """Produce the target condition set from ``workspace_root`` and store it durably.

    The whole vertical slice, run against a real workspace: authenticate the
    stored document, derive and embed its table grid, assemble the condition-set
    envelope through the production producer, and write the envelope into the
    workspace's own condition-set store. Reuses the producer and the store
    wrapper -- nothing here reimplements them.

    Raises:
        ConditionSetTargetError: A precondition failed (see
            :func:`read_target_raw`, :func:`build_embedded_inventory`).
    """
    raw = read_target_raw(workspace_root)
    embedded, _ = build_embedded_inventory(raw)
    envelope = build_envelope(workspace_root, embedded)
    stored: StoredDataset = store_condition_set_envelope(workspace_root, envelope)
    return StoredTargetConditionSet(sha256=stored.sha256, path=stored.path, envelope=envelope)


def _stored_table_reference(envelope: ConditionSetEnvelope) -> tuple[str, str]:
    """Read the table label and source page the envelope itself cites.

    Walks every :class:`~carmel.schemas.datasets.SourceRef` reachable in the
    envelope (the same choke point V1 validates every ref through) for the
    first ``TableCellLocator`` naming a ``CaptionLabelKey``, then reads the
    page from the embedded inventory its ``pdf_table_inventory_sha256`` names.
    Both facts are read from the STORED envelope, never a module constant, so
    a previously stored envelope renders with ITS OWN label and page even if
    ``TARGET_TABLE_KEY``/``TARGET_TABLE_FOOTPRINT`` are edited later -- the
    drift the docstring above already promised was impossible.

    Falls back to ``"unknown"`` for both, exactly like the ``raw_sha256``
    fallback below, for an envelope with no PDF table-cell citation to read
    either fact from.
    """
    for _, ref in iter_source_refs(envelope):
        locator = ref.locator
        if not (isinstance(locator, TableCellLocator) and isinstance(locator.table_key, CaptionLabelKey)):
            continue
        label = locator.table_key.label
        page: object = "unknown"
        if isinstance(locator.pdf_table_inventory_sha256, str):
            for inventory in envelope.table_inventories:
                if inventory.inventory_sha256 == locator.pdf_table_inventory_sha256:
                    page = json.loads(inventory.canonical_json)["footprint"]["page"]
                    break
        return label, str(page)
    return "unknown", "unknown"


def render_condition_set_text(envelope: ConditionSetEnvelope) -> str:
    """Render the stored condition set as a human-readable text report.

    Derived entirely from ``envelope`` so it cannot drift from what was grounded:
    a header identifying the source document, the table, the attribution and the
    subject, then the categorical conditions, then the unextracted (refused)
    statements with their reason. Scalar conditions render too, for completeness,
    though this document has none.
    """
    raw_sha256 = envelope.table_inventories[0].raw_sha256 if envelope.table_inventories else "unknown"
    table_label, table_page = _stored_table_reference(envelope)

    if isinstance(envelope.subject, DeviceClassDeclaration):
        subject_text = f'device class "{envelope.subject.label_raw}"'
    else:
        assert isinstance(envelope.subject, UnresolvedSubject)  # noqa: S101 - the sum has exactly two arms
        subject_text = f"unresolved ({envelope.subject.reason.value})"

    lines: list[str] = [
        "Condition set",
        f"Source document: sha256 {raw_sha256}",
        f"Source table   : {table_label}, page {table_page}",
        f"Attribution    : {envelope.attribution.value}",
        f"Subject        : {subject_text}",
        "",
    ]

    lines.append(f"Scalar conditions ({len(envelope.scalar_claims)}):")
    for claim in envelope.scalar_claims:
        value = claim.value
        lines.append(f"  {claim.label_raw} = {value.raw_text} {value.unit_normalized}")
    if not envelope.scalar_claims:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"Categorical conditions ({len(envelope.categorical_claims)}):")
    for categorical in envelope.categorical_claims:
        lines.append(f"  {categorical.label_raw} = {categorical.token_raw}")
    if not envelope.categorical_claims:
        lines.append("  (none)")
    lines.append("")

    lines.append(f"Unextracted statements ({len(envelope.unextracted)}):")
    for statement in envelope.unextracted:
        quantity = (
            statement.quantity_kind.value if isinstance(statement.quantity_kind, units.QuantityKind) else "unknown"
        )
        lines.append(
            f"  {statement.label_raw}: {statement.statement_raw}  "
            f"[reason={statement.reason.value}, quantity={quantity}]"
        )
    if not envelope.unextracted:
        lines.append("  (none)")

    return "\n".join(lines) + "\n"


def write_condition_set_export(workspace_root: Path, condition_set: StoredTargetConditionSet) -> Path:
    """Write the rendered condition set into the workspace's ``reports`` directory.

    Returns the path of the written file: ``reports/condition-set-<sha256>.txt``.
    """
    reports_dir = workspace_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"condition-set-{condition_set.sha256}.txt"
    path.write_text(render_condition_set_text(condition_set.envelope), encoding="utf-8")
    return path
