# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``carmel.services.dataset_replay`` -- the standalone replayer
service that independently re-verifies a stored :class:`DatasetEnvelope`
against the content-addressed evidence store it was produced from.

This module promotes what used to be test-local helpers in
``tests/test_dataset_producer.py`` (``_assert_every_char_span_grounds`` and
``_independently_verified_text``) into the real service under test here.
``tests/test_dataset_producer.py`` no longer contains its own copy of this
logic -- it calls this service too, so there is exactly one replayer
implementation, not two that could quietly drift apart.

SYNTHETIC evidence text only -- see ``tests/test_dataset_producer.py``'s
module docstring for why (the project's real corpus is closed-access,
non-redistributable).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from carmel.agents.tools.extract import ExtractedText, PageExtractionFailure
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisRole,
    CharSpanLocator,
    DatasetEnvelope,
    ExtractionBinding,
    LocatorKind,
    MeasuredValue,
    SemanticDependencyUse,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    TextSpace,
    ValueOrigin,
    XPathLocator,
)
from carmel.schemas.literature import StoredArtifact
from carmel.services.dataset_bridge import load_dataset_envelope, store_dataset_envelope
from carmel.services.dataset_producer import MeasurementSpec, produce_envelope_from_artifact
from carmel.services.dataset_replay import (
    ReplayFinding,
    ReplayOutcome,
    _independently_verify_node_text,
    check_char_spans,
    replay_envelope,
    verify_measured_value_unit,
    verify_measured_value_unit_boundary,
    verify_measured_value_value_boundary,
)
from carmel.services.evidence import artifact_dir, store_artifact
from carmel.services.extraction_record import (
    compute_extraction_sha,
    extraction_record_dir,
    load_extraction_record,
    store_extraction_record,
)
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import TABLE_V1, QuantityKind

MAX_BYTES = 10_000_000

_TEXT = (
    "The reactor was held at a temperature of 1023 K while the measured mole "
    "fraction (-) of the fuel species was 0.0123 at steady state."
)
"""Synthetic grounded quotes: "temperature", "1023", "K", "mole fraction",
"-", "0.0123" -- each appears exactly once."""

_MUTATED_TEXT = _TEXT.replace("1023 K", "1024 K")
"""Differs from ``_TEXT`` by exactly ONE character ('3' -> '4'), INSIDE the
grounded value quote "1023"."""

_SPECS = (
    MeasurementSpec(
        axis_id="temperature",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.TEMPERATURE,
        label_quote="temperature",
        value_quote="1023",
        unit_quote="K",
    ),
    MeasurementSpec(
        axis_id="mole_fraction",
        role=AxisRole.OBSERVATION,
        quantity_kind=QuantityKind.MOLE_FRACTION,
        label_quote="mole fraction",
        value_quote="0.0123",
        unit_quote="-",
    ),
)


def _store_synthetic_artifact(
    workspace_root: Path,
    text: str,
    *,
    content_type: str = "application/pdf",
    extractor: str = "pdf:pypdf",
) -> StoredArtifact:
    data = text.encode("utf-8")
    artifact = FetchedArtifact(
        url="https://example.invalid/paper.pdf",
        final_url="https://example.invalid/paper.pdf",
        sha256=hashlib.sha256(data).hexdigest(),
        content_type=content_type,
        n_bytes=len(data),
        fetched_at=datetime.now(UTC),
    )
    extracted = ExtractedText(text=text, normalized=text.casefold(), sections=[], extractor=extractor, lossy=False)
    return store_artifact(workspace_root, data=data, artifact=artifact, extracted=extracted, max_bytes=MAX_BYTES)


def _produce_and_load(tmp_path: Path, text: str = _TEXT):
    stored_artifact = _store_synthetic_artifact(tmp_path, text)
    datasets_root = tmp_path / "datasets"
    envelope = produce_envelope_from_artifact(
        tmp_path,
        sha256=stored_artifact.sha256,
        series_id="s1",
        value_origin=ValueOrigin.EXPERIMENTAL,
        measurements=_SPECS,
    )
    stored_dataset = store_dataset_envelope(datasets_root, envelope)
    loaded = load_dataset_envelope(datasets_root, stored_dataset.sha256)
    return stored_artifact, loaded


class TestReplayEnvelopeCleanRoundTrip:
    def test_replay_verifies_clean_round_trip(self, tmp_path: Path) -> None:
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        report = replay_envelope(tmp_path, loaded)
        assert report.outcome is ReplayOutcome.VERIFIED
        assert report.checked_char_spans == 6
        assert report.failures == ()
        assert report.unverifiable == ()


class TestReplayEnvelopeCatchesMutation:
    def test_replay_fails_against_single_character_mutation(self, tmp_path: Path) -> None:
        original = _store_synthetic_artifact(tmp_path, _TEXT)
        mutated = _store_synthetic_artifact(tmp_path, _MUTATED_TEXT)
        assert original.sha256 != mutated.sha256
        assert len(_TEXT) == len(_MUTATED_TEXT)

        datasets_root = tmp_path / "datasets"
        envelope = produce_envelope_from_artifact(
            tmp_path,
            sha256=original.sha256,
            series_id="s1",
            value_origin=ValueOrigin.EXPERIMENTAL,
            measurements=_SPECS,
        )
        stored_dataset = store_dataset_envelope(datasets_root, envelope)
        loaded = load_dataset_envelope(datasets_root, stored_dataset.sha256)

        clean_report = replay_envelope(tmp_path, loaded)
        assert clean_report.outcome is ReplayOutcome.VERIFIED

        # Corrupt the stored evidence *underneath the same sha256 node the
        # envelope already points at* is not representable (the store is
        # content-addressed), so instead we prove the mutation is caught the
        # way a real re-extraction divergence would surface: by replaying
        # against evidence whose extracted text differs from what the
        # envelope's ExtractionBinding recorded. We simulate that by
        # physically overwriting extracted.json on disk, in place, at the
        # extraction-RECORD store's own address -- NOT the older
        # ``artifact_dir(raw_sha256)`` layout. ``_independently_verify_node_text``
        # (carmel/services/dataset_replay.py) resolves a node's extracted.json
        # exclusively via that node's own ExtractionBinding
        # (parent_raw_sha256, extraction_sha256); mutating the old
        # ``artifact_dir`` path instead would tamper with evidence the
        # replayer never re-reads, and this test would go green vacuously
        # (report would stay VERIFIED even though the assertion below claims
        # otherwise). The ONLY guard capable of turning this mutation into a
        # non-VERIFIED outcome is the
        # ``actual_extracted_sha256 != extraction.extracted_sha256`` digest
        # comparison in ``_independently_verify_node_text``.
        node = loaded.source_graph.node("paper")
        binding = node.extraction
        assert isinstance(binding, ExtractionBinding), "extraction must be present, not Absent"
        extracted_path = (
            extraction_record_dir(tmp_path, binding.parent_raw_sha256, binding.extraction_sha256)
            / "extracted.json"
        )
        mutated_extracted = ExtractedText(
            text=_MUTATED_TEXT, normalized=_MUTATED_TEXT.casefold(), sections=[], extractor="pdf:pypdf", lossy=False
        )
        extracted_path.write_text(json.dumps(mutated_extracted.model_dump(mode="json")), encoding="utf-8")

        mutated_report = replay_envelope(tmp_path, loaded)
        assert mutated_report.outcome is not ReplayOutcome.VERIFIED
        assert any("extracted.json" in f.reason for f in (mutated_report.failures + mutated_report.unverifiable))


class TestReplayEnvelopeMissingEvidence:
    def test_missing_evidence_is_unverifiable_not_failed_not_verified(self, tmp_path: Path) -> None:
        stored_artifact, loaded = _produce_and_load(tmp_path)
        # Delete the entire evidence artifact directory out from under the
        # envelope's own recorded sha256, simulating evidence that has been
        # lost/garbage-collected since the envelope was produced.
        import shutil

        shutil.rmtree(artifact_dir(tmp_path, stored_artifact.sha256))

        report = replay_envelope(tmp_path, loaded)
        assert report.outcome is ReplayOutcome.UNVERIFIABLE
        assert report.failures == ()
        assert report.unverifiable != ()

    def test_absent_extraction_record_is_unverifiable_naming_the_missing_record(self, tmp_path: Path) -> None:
        """The missing-record branch must report WHY, not merely UNVERIFIABLE.

        Deleting the extraction record also breaks a LATER check -- the read of
        ``extracted.json`` -- and that one yields UNVERIFIABLE too. So asserting the
        outcome alone tests nothing here: delete the missing-record branch entirely
        and the outcome is unchanged. The two are distinguishable only by the reason
        they give, and that distinction is the point. "No extraction record is stored
        at this address" says the envelope references evidence that was never present
        or has been garbage-collected; "could not read extracted.json" says the record
        is there but damaged. Those imply different operator actions.

        Hence: assert the reason names the missing RECORD, and delete only the record
        directory, leaving ``raw.bin`` and the artifact intact so every earlier guard
        passes and this branch is the first thing able to fire.
        """
        import shutil

        _stored, loaded = _produce_and_load(tmp_path)
        node = next(n for n in loaded.source_graph.nodes if not isinstance(n.extraction, Absent))
        shutil.rmtree(
            extraction_record_dir(
                tmp_path,
                node.extraction.parent_raw_sha256,
                node.extraction.extraction_sha256,
            )
        )

        report = replay_envelope(tmp_path, loaded)

        assert report.outcome is ReplayOutcome.UNVERIFIABLE
        assert report.failures == ()
        reasons = " ".join(finding.reason for finding in report.unverifiable)
        assert "no extraction record" in reasons.lower()
        assert node.extraction.extraction_sha256 in reasons


class TestReplayEnvelopeExtractedTextShaMismatch:
    def test_mismatched_extracted_text_sha256_fails_with_named_reason(self, tmp_path: Path) -> None:
        # The extraction-record store is self-authenticating: its address
        # (extraction_sha256) is recomputed from meta.json's OWN fields on
        # every load (see carmel/services/extraction_record.py's
        # load_extraction_record / _load_meta). Editing meta.json's
        # extracted_sha256 in place -- the pattern the OLD, single-record
        # artifact_dir store used -- therefore either breaks self-
        # authentication (UNVERIFIABLE, never reaching the guard this test
        # targets) or, if left untouched, trips an EARLIER "sidecar
        # bookkeeping inconsistent" FAILED check in
        # _independently_verify_node_text before the extracted_text_sha256
        # comparison this test isolates is ever reached. So this test does
        # not tamper any file on disk at all: it leaves the on-disk
        # extraction record (extracted.json/text.txt/meta.json) exactly as
        # produced, and patches ONLY the loaded envelope's own
        # ExtractionBinding.extracted_text_sha256 to a value that disagrees
        # with the untouched evidence. Every OTHER field the replayer checks
        # (extracted_sha256, the sidecar's own extracted_sha256/
        # extracted_text_sha256, parse-ability, lossy flag) still agrees, so
        # the ONLY guard capable of rejecting this input is the
        # ``actual_text_sha256 != extraction.extracted_text_sha256``
        # comparison in _independently_verify_node_text (guard 8 in the
        # module's ordered check sequence).
        _, loaded = _produce_and_load(tmp_path)

        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        wrong_text_sha256 = hashlib.sha256(b"not the real extracted text").hexdigest()
        assert wrong_text_sha256 != node.extraction.extracted_text_sha256
        patched_extraction = node.extraction.model_copy(update={"extracted_text_sha256": wrong_text_sha256})
        patched_node = node.model_copy(update={"extraction": patched_extraction})
        patched_nodes = tuple(patched_node if n is node else n for n in loaded.source_graph.nodes)
        patched_graph = loaded.source_graph.model_copy(update={"nodes": patched_nodes})
        tampered = loaded.model_copy(update={"source_graph": patched_graph})

        report = replay_envelope(tmp_path, tampered)
        assert report.outcome is ReplayOutcome.FAILED
        assert any("extracted_text_sha256" in f.reason for f in report.failures)


class TestVerifyMeasuredValueUnit:
    """Focused tests for the round-44 requirement: a ``MeasuredValue`` is
    re-verified against the conversion table its own ``conversion_table_sha256``
    recorded -- never against whatever table is current -- via
    ``verify_measured_value_unit``. Uses ``MeasuredValue.model_construct`` to
    build inputs the normal validated constructor would refuse, since these
    two failure shapes (unknown table sha; unit that no longer normalizes
    under its own recorded table) can only arise from a corrupted/forged
    record, never from a schema-valid one.
    """

    @staticmethod
    def _dummy_ref() -> SourceRef:
        return SourceRef(
            node_id="paper",
            locator=CharSpanLocator(kind=LocatorKind.CHAR_SPAN, text_space=TextSpace.EXTRACTED_TEXT, start=0, end=4),
        )

    @staticmethod
    def _repair_dependency() -> SemanticDependencyUse:
        return SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )

    def test_unknown_recorded_table_sha_is_unverifiable(self) -> None:
        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256="0" * 64,
            value_ref=self._dummy_ref(),
            unit_ref=self._dummy_ref(),
        )
        finding = verify_measured_value_unit("dummy.path", value)
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "0" * 64 in finding.reason

    def test_unit_that_no_longer_normalizes_under_recorded_table_fails(self) -> None:
        from carmel.services import units

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="not-a-real-unit",
            unit_normalized="K",
            conversion_table_sha256=units.TABLE_V1.sha256,
            value_ref=self._dummy_ref(),
            unit_ref=self._dummy_ref(),
        )
        finding = verify_measured_value_unit("dummy.path", value)
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert "not-a-real-unit" in finding.reason

    def test_clean_measured_value_verifies(self) -> None:
        from carmel.services import units

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=units.TABLE_V1.sha256,
            value_ref=self._dummy_ref(),
            unit_ref=self._dummy_ref(),
        )
        assert verify_measured_value_unit("dummy.path", value) is None


class TestReplayEnvelopeCatchesSpanResliceMismatchIndependentOfEvidence:
    """The span-reslice comparison in ``check_char_spans`` and the
    evidence-identity re-read in ``_independently_verify_node_text`` are two
    DIFFERENT checks; this test exercises the reslice comparison ALONE, with
    the evidence store left completely untouched, so a mutation that
    disables only the reslice comparison (e.g. neutering
    ``if actual != expected`` to a no-op) cannot hide behind the
    evidence-identity check catching the same tampering by a different
    route -- the other tests in this module all tamper with
    ``extracted.json`` on disk, which the evidence-identity check alone is
    sufficient to catch."""

    def test_shifted_locator_offsets_fail_replay_even_with_untouched_evidence(self, tmp_path: Path) -> None:
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        clean_report = replay_envelope(tmp_path, loaded)
        assert clean_report.outcome is ReplayOutcome.VERIFIED

        series = loaded.series[0]
        point = series.points[0]
        coordinate = point.coordinates[0]
        value = coordinate.value
        locator = value.value_ref.locator
        assert isinstance(locator, CharSpanLocator)

        # Shift the span by one character in the SAME (untouched) evidence
        # text, so it now slices to something other than the recorded
        # raw_text -- a structurally valid envelope (frozen models copied
        # without re-validation) whose claim about the evidence is simply
        # wrong, exactly the kind of defect this check exists to catch.
        shifted_locator = locator.model_copy(update={"start": locator.start + 1, "end": locator.end + 1})
        shifted_ref = value.value_ref.model_copy(update={"locator": shifted_locator})
        shifted_value = value.model_copy(update={"value_ref": shifted_ref})
        shifted_coordinate = coordinate.model_copy(update={"value": shifted_value})
        shifted_coordinates = tuple(
            shifted_coordinate if c is coordinate else c for c in point.coordinates
        )
        shifted_point = point.model_copy(update={"coordinates": shifted_coordinates})
        shifted_points = tuple(shifted_point if p is point else p for p in series.points)
        shifted_series_obj = series.model_copy(update={"points": shifted_points})
        shifted_series = tuple(shifted_series_obj if s is series else s for s in loaded.series)
        tampered = loaded.model_copy(update={"series": shifted_series})

        report = replay_envelope(tmp_path, tampered)
        assert report.outcome is ReplayOutcome.FAILED
        assert any("char-span re-slice mismatch" in f.reason for f in report.failures)


class TestReplayEnvelopeGenericWalkExhaustiveness:
    def test_checked_count_matches_iter_source_refs_char_span_count(self, tmp_path: Path) -> None:
        from carmel.schemas.datasets import iter_source_refs

        _stored_artifact, loaded = _produce_and_load(tmp_path)
        report = replay_envelope(tmp_path, loaded)
        total_char_span_refs = sum(
            1 for _, ref in iter_source_refs(loaded) if isinstance(ref.locator, CharSpanLocator)
        )
        assert report.checked_char_spans == total_char_span_refs


class TestReplayEnvelopeRejectsMutableSidecarAsAnchor:
    """P0 (round 45): the replayer must authenticate ``extracted.json``
    against the ENVELOPE's own :class:`ExtractionBinding.extracted_sha256`
    -- never against the evidence store's ``meta.json`` sidecar, which lives
    right next to the very file it would otherwise authenticate and is
    trivially rewritable.

    The exploit: rewrite ``extracted.json`` changing a field OTHER than
    ``.text`` (so the recorded ``extracted_text_sha256`` still matches),
    then patch ``meta.json``'s ``extracted_sha256`` to the new bytes'
    digest so a meta.json-anchored replayer sees a consistent-looking
    sidecar. The envelope's own ``ExtractionBinding.extracted_sha256`` --
    written once at production time and never touched -- still names the
    ORIGINAL bytes. A replayer that trusts meta.json returns VERIFIED here;
    that is vacuous, because the stored evidence has visibly changed.
    """

    def test_meta_json_sidecar_rewrite_does_not_launder_a_verified_result(self, tmp_path: Path) -> None:
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        clean_report = replay_envelope(tmp_path, loaded)
        assert clean_report.outcome is ReplayOutcome.VERIFIED

        # Tamper with the files replay actually reads: the ADDRESSED
        # extraction record's extracted.json and the meta.json sidecar
        # right next to it (the root evidence store's copies are no longer
        # an input to replay at all -- see
        # TestReplayVerifiesAgainstTheRecordNotTheRootSidecar).
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        record_dir = extraction_record_dir(
            tmp_path, node.extraction.parent_raw_sha256, node.extraction.extraction_sha256
        )
        extracted_path = record_dir / "extracted.json"
        raw_bytes = extracted_path.read_bytes()
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))

        # Change a field OTHER than `.text` -- `.text` stays byte-identical
        # so extracted_text_sha256 (which only ever digests `.text`) still
        # matches what the envelope recorded. `lossy` flips from False to
        # True: a forger silently marking previously-clean evidence as lossy
        # without changing a single character of the text the char-spans
        # actually re-slice.
        forged = ExtractedText(
            text=extracted.text,
            normalized=extracted.normalized,
            sections=extracted.sections,
            page_count=extracted.page_count,
            extractor=extracted.extractor,
            lossy=True,
        )
        forged_bytes = json.dumps(forged.model_dump(mode="json")).encode("utf-8")
        assert forged_bytes != raw_bytes
        extracted_path.write_bytes(forged_bytes)

        # First, with the record's meta.json sidecar UNTOUCHED (it still
        # authenticates to its address): the envelope's own
        # ExtractionBinding.extracted_sha256 anchor must catch the rewrite
        # as a definite FAILED, naming the anchor.
        report = replay_envelope(tmp_path, loaded)
        assert report.outcome is ReplayOutcome.FAILED
        assert any("ExtractionBinding.extracted_sha256" in f.reason for f in report.failures)

        # Then patch the record's meta.json extracted_sha256 to match the
        # forged bytes, so a replayer that (wrongly) anchored on the
        # mutable sidecar would see a self-consistent record. The record's
        # digest fields are folded into its own content ADDRESS, so the
        # patched sidecar no longer authenticates to the address the
        # envelope names -- the record becomes unresolvable (UNVERIFIABLE),
        # and no combination of sidecar edits can ever launder VERIFIED.
        meta_path = record_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["extracted_sha256"] = hashlib.sha256(forged_bytes).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        report = replay_envelope(tmp_path, loaded)
        assert report.outcome is not ReplayOutcome.VERIFIED, (
            "replay must not launder a rewritten extracted.json just because meta.json was "
            "patched to match it -- the envelope's own ExtractionBinding.extracted_sha256 is "
            "the only acceptable anchor"
        )
        assert any(
            "does not authenticate" in f.reason or "extracted_sha256" in f.reason or "extracted.json" in f.reason
            for f in (report.failures + report.unverifiable)
        )


class TestReplayEnvelopeZeroCheckedSpansCannotVerify:
    """Hole #1 (round 45): a replay that independently re-slices ZERO
    character spans must never report VERIFIED -- that would launder the
    "verified" label onto an envelope where nothing was actually checked.
    """

    def test_envelope_with_no_char_span_refs_is_unverifiable_not_verified(self, tmp_path: Path) -> None:
        _stored_artifact, loaded = _produce_and_load(tmp_path)

        # Strip every series so no CharSpanLocator remains reachable via
        # iter_measured_values/axes -- the source graph and its evidence
        # stay completely intact and verifiable; only the char-span-bearing
        # content is gone.
        empty_envelope = loaded.model_copy(update={"series": ()})

        report = replay_envelope(tmp_path, empty_envelope)
        assert report.checked_char_spans == 0
        assert report.outcome is not ReplayOutcome.VERIFIED, (
            "a replay that independently checked zero character spans must never report "
            "VERIFIED -- there is nothing behind that label"
        )
        assert report.outcome is ReplayOutcome.UNVERIFIABLE
        assert any("0" in f.reason or "zero" in f.reason.lower() for f in report.unverifiable)


class TestVerifyMeasuredValueUnitBoundary:
    """Focused tests for ``verify_measured_value_unit_boundary`` -- the
    closed gap this module's docstring used to describe as deliberate and
    open. Replay used to re-verify evidence identity, every character span,
    and unit NORMALIZATION against the recorded table, but never re-ran
    boundary/admission: an envelope that RECORDED ``unit_raw="bar"`` with a
    unit locator pointing inside evidence text ``"1 bar(a)"`` replayed
    ``VERIFIED`` even though ``ground_quote`` would refuse that exact quote
    at write time. These tests build the adversarial ``MeasuredValue`` via
    ``model_construct``/direct construction, deliberately WITHOUT going
    through ``ground_quote``/the producer's gate, simulating a buggy or
    malicious producer whose output the replayer must still catch on its
    own.
    """

    @staticmethod
    def _ref(node_id: str, start: int, end: int) -> SourceRef:
        return SourceRef(
            node_id=node_id,
            locator=CharSpanLocator(
                kind=LocatorKind.CHAR_SPAN, text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end
            ),
        )

    @staticmethod
    def _repair_dependency() -> SemanticDependencyUse:
        return SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )

    def test_bar_glued_to_parenthetical_fails_not_verified(self) -> None:
        # Empirically confirmed (see numeric.unit_boundary_violation) that
        # "bar" inside "1 bar(a)" is refused by Layers 1-2 ALONE, table-free,
        # via the "unit_trailing_unclassified_char" discriminant -- exactly
        # the quote ground_quote would refuse at write time.
        text = "The pressure was 1 bar(a) at steady state."
        start = text.index("bar")
        end = start + len("bar")
        assert text[start:end] == "bar"

        value = MeasuredValue.model_construct(
            raw_text="1",
            canonical_decimal_value="1",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.PRESSURE,
            unit_raw="bar",
            unit_normalized="Pa",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("other_node", 0, 1),
            unit_ref=self._ref("paper", start, end),
        )

        finding = verify_measured_value_unit_boundary("dummy.path", value, {"paper": text})
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert "unit_trailing_unclassified_char" in finding.reason

    def test_unknown_recorded_table_sha_stays_unverifiable(self) -> None:
        # The unit itself ("K" in "... 1023 K while ...") passes Layers 1-2
        # cleanly on its own -- only Layer 3 (table admission) is blocked,
        # by a conversion_table_sha256 that names no known table. This must
        # stay UNVERIFIABLE, never FAILED (no disagreement was demonstrated)
        # and never VERIFIED (Layer 3 never actually ran).
        start = _TEXT.index("1023 K") + len("1023 ")
        end = start + 1
        assert _TEXT[start:end] == "K"

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256="0" * 64,
            value_ref=self._ref("paper", _TEXT.index("1023"), _TEXT.index("1023") + 4),
            unit_ref=self._ref("paper", start, end),
        )

        finding = verify_measured_value_unit_boundary("dummy.path", value, {"paper": _TEXT})
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "0" * 64 in finding.reason

    def test_clean_measured_value_boundary_verifies(self) -> None:
        start = _TEXT.index("1023 K") + len("1023 ")
        end = start + 1
        assert _TEXT[start:end] == "K"

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("paper", _TEXT.index("1023"), _TEXT.index("1023") + 4),
            unit_ref=self._ref("paper", start, end),
        )

        assert verify_measured_value_unit_boundary("dummy.path", value, {"paper": _TEXT}) is None

    def test_unrecoverable_value_span_still_fails_closed_not_verified(self) -> None:
        # Digit-glue shape: unit token immediately abutting a digit run with
        # no delimiter between them ("1023K", no space). Layer 2's narrow
        # digit-glue exception can only forgive this when it is handed a
        # well-formed value_span naming the adjacent numeral. Empirically
        # confirmed (carmel.services.numeric.unit_boundary_violation): with
        # no value_span this text refuses with
        # "unit_digit_glue_no_value_span"; supplying the correct value_span
        # makes it clean.
        #
        # Here value_ref names a DIFFERENT node than unit_ref, so
        # _recover_value_span deliberately returns None (it will not borrow
        # a value_span across nodes) -- the value_span the exception would
        # need is therefore NOT recoverable from what the envelope recorded.
        # This must fail closed (FAILED, the check ran and refused) rather
        # than silently verify-by-omission (never a bare None/VERIFIED).
        text = "The value was 1023K exactly."
        start = text.index("K")
        end = start + 1
        assert text[start:end] == "K"

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("some_other_node", 0, 4),
            unit_ref=self._ref("paper", start, end),
        )

        finding = verify_measured_value_unit_boundary("dummy.path", value, {"paper": text})
        assert finding is not None, (
            "an unrecoverable value_span must never silently verify -- the digit-glue-shaped "
            "text still needs an explanation, and none was recoverable from the recorded refs"
        )
        assert finding.category is ReplayOutcome.FAILED
        assert "unit_digit_glue_no_value_span" in finding.reason

    def test_non_char_span_unit_locator_is_unverifiable_not_silently_clean(self) -> None:
        # A unit_ref whose locator is NOT a CharSpanLocator (e.g. XPathLocator)
        # used to make verify_measured_value_unit_boundary return a bare
        # `None` -- indistinguishable from "checked and clean". Nothing in
        # check_char_spans catches this either (it also only handles
        # CharSpanLocator refs), so such a value was invisible to every
        # boundary/admission check the replayer runs, yet still replayed
        # VERIFIED. This must surface as a named UNVERIFIABLE finding.
        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("paper", _TEXT.index("1023"), _TEXT.index("1023") + 4),
            unit_ref=SourceRef(node_id="paper", locator=XPathLocator(kind=LocatorKind.XPATH, xpath="//p[1]")),
        )

        finding = verify_measured_value_unit_boundary("dummy.path", value, {"paper": _TEXT})
        assert finding is not None, (
            "a non-CharSpanLocator unit_ref must never silently verify -- the boundary/admission "
            "gate cannot re-run against a locator that carries no character span, so this must be "
            "reported, not skipped"
        )
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "unit_ref" in finding.ref_path

    def test_quantity_kind_other_is_unverifiable_not_failed(self) -> None:
        # QuantityKind.OTHER has no admission vocabulary at all --
        # binding.spellings_by_quantity has no entry for OTHER (by design,
        # see _ActiveTableBinding.derive), so
        # _unit_table_boundary_violation's `.get(quantity, frozenset())`
        # always falls back to an empty set and unconditionally returns
        # "unit_not_in_vocabulary" for OTHER, reported as FAILED -- as if the
        # recorded unit were demonstrably wrong. It is not wrong; it is
        # simply unmodelled, and ground_quote itself refuses to ever produce
        # an OTHER unit quote at write time (Layer 0). Replay must report
        # this honestly as UNVERIFIABLE, not FAILED.
        text = "The reading was 5 widgets exactly."
        start = text.index("widgets")
        end = start + len("widgets")
        assert text[start:end] == "widgets"

        value = MeasuredValue.model_construct(
            raw_text="5",
            canonical_decimal_value="5",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.OTHER,
            unit_raw="widgets",
            unit_normalized="widgets",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("paper", text.index("5"), text.index("5") + 1),
            unit_ref=self._ref("paper", start, end),
        )

        finding = verify_measured_value_unit_boundary("dummy.path", value, {"paper": text})
        assert finding is not None, (
            "QuantityKind.OTHER has no admission vocabulary to check against -- this must be "
            "reported as unverifiable, not silently passed and not reported as a demonstrated "
            "disagreement"
        )
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "OTHER" in finding.reason


class TestVerifyMeasuredValueValueBoundary:
    """Focused tests for ``verify_measured_value_value_boundary`` -- closes
    the symmetric gap ``verify_measured_value_unit_boundary`` closed for
    UNIT, but for VALUE: ``check_char_spans`` only re-slices ``value_ref``
    and string-compares to the recorded ``raw_text``, it never re-runs
    VALUE-role maximality/boundary. A ``value_ref`` moved to land on "1023"
    INSIDE "11023" replayed VERIFIED even though ``ground_quote`` would
    refuse that exact quote at write time (an interior fragment of a larger
    numeral). These tests build the adversarial ``MeasuredValue`` directly,
    simulating a buggy/hostile producer.
    """

    @staticmethod
    def _ref(node_id: str, start: int, end: int) -> SourceRef:
        return SourceRef(
            node_id=node_id,
            locator=CharSpanLocator(
                kind=LocatorKind.CHAR_SPAN, text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end
            ),
        )

    @staticmethod
    def _repair_dependency() -> SemanticDependencyUse:
        return SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        )

    def test_value_ref_moved_to_interior_numeral_fragment_fails_not_verified(self) -> None:
        # "1023" glued inside "11023" -- a genuine interior fragment of the
        # larger numeral "11023". check_char_spans alone would happily
        # accept this (text[start:end] == "1023" == raw_text), but
        # ground_quote's VALUE-role maximality check would refuse it at
        # write time.
        text = "The reading was 11023 units at closure."
        start = text.index("11023") + 1
        end = start + 4
        assert text[start:end] == "1023"

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("paper", start, end),
            unit_ref=self._ref("paper", 0, 1),
        )

        finding = verify_measured_value_value_boundary("dummy.path", value, {"paper": text})
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert "value_" in finding.reason

    def test_non_char_span_value_locator_is_unverifiable_not_silently_clean(self) -> None:
        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=SourceRef(node_id="paper", locator=XPathLocator(kind=LocatorKind.XPATH, xpath="//p[1]")),
            unit_ref=self._ref("paper", 0, 1),
        )

        finding = verify_measured_value_value_boundary("dummy.path", value, {"paper": _TEXT})
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "value_ref" in finding.ref_path

    def test_clean_measured_value_verifies(self) -> None:
        start = _TEXT.index("1023")
        end = start + 4
        assert _TEXT[start:end] == "1023"

        value = MeasuredValue.model_construct(
            raw_text="1023",
            canonical_decimal_value="1023",
            repairs=(),
            repair_dependency=self._repair_dependency(),
            quantity_kind=QuantityKind.TEMPERATURE,
            unit_raw="K",
            unit_normalized="K",
            conversion_table_sha256=TABLE_V1.sha256,
            value_ref=self._ref("paper", start, end),
            unit_ref=self._ref("paper", 0, 1),
        )

        assert verify_measured_value_value_boundary("dummy.path", value, {"paper": _TEXT}) is None


class TestReplayEnvelopeCatchesUnitBoundaryViolation:
    """End-to-end (round D-U2 gap closure): a full ``replay_envelope`` call
    against a tampered envelope whose recorded unit locator points at a
    quote ``ground_quote`` would refuse today must report ``FAILED``, not
    ``VERIFIED`` -- this is the headline scenario the reviewer brief names
    explicitly (``unit_raw="bar"`` grounded inside evidence text
    ``"1 bar(a)"``).
    """

    def test_replay_fails_when_recorded_unit_locator_lands_on_glued_bar(self, tmp_path: Path) -> None:
        text_with_glued_bar = _TEXT + " The pressure was 1 bar(a) at closure."
        _stored_artifact, loaded = _produce_and_load(tmp_path, text=text_with_glued_bar)
        clean_report = replay_envelope(tmp_path, loaded)
        assert clean_report.outcome is ReplayOutcome.VERIFIED

        series = loaded.series[0]
        point = series.points[0]
        coordinate = point.coordinates[0]
        value = coordinate.value

        start = text_with_glued_bar.index("bar", text_with_glued_bar.index("1 bar(a)"))
        end = start + len("bar")
        assert text_with_glued_bar[start:end] == "bar"
        glued_locator = CharSpanLocator(
            kind=LocatorKind.CHAR_SPAN, text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end
        )
        # A buggy/malicious producer recording a unit locator that ground_quote
        # would have refused, WITHOUT going through ground_quote's own gate --
        # `model_copy` bypasses validation the same way the rest of this
        # module's tamper tests do.
        glued_unit_ref = value.unit_ref.model_copy(update={"locator": glued_locator})
        tampered_value = value.model_copy(update={"unit_raw": "bar", "unit_ref": glued_unit_ref})
        tampered_coordinate = coordinate.model_copy(update={"value": tampered_value})
        tampered_coordinates = tuple(
            tampered_coordinate if c is coordinate else c for c in point.coordinates
        )
        tampered_point = point.model_copy(update={"coordinates": tampered_coordinates})
        tampered_points = tuple(tampered_point if p is point else p for p in series.points)
        tampered_series_obj = series.model_copy(update={"points": tampered_points})
        tampered_series = tuple(tampered_series_obj if s is series else s for s in loaded.series)
        tampered = loaded.model_copy(update={"series": tampered_series})

        report = replay_envelope(tmp_path, tampered)
        assert report.outcome is ReplayOutcome.FAILED
        assert any("unit_trailing_unclassified_char" in f.reason for f in report.failures)


class TestReplayEnvelopeContainsParserExceptions:
    """Hole #3 (round 45): a hash-consistent ``extracted.json`` that fails
    to *parse* as :class:`ExtractedText` (e.g. pathologically deep nesting
    that blows Python's recursion limit) must become a named UNVERIFIABLE
    finding, never an escaping crash that takes down the whole replay run.
    """

    def test_recursion_error_while_parsing_extracted_json_is_unverifiable_not_a_crash(
        self, tmp_path: Path
    ) -> None:
        # ``store_extraction_record`` (the real front door) itself calls
        # ``ExtractedText.model_validate(json.loads(extracted_json_bytes))``
        # BEFORE hashing/writing anything, so it cannot be used to create
        # this fixture: the RecursionError this test needs to happen INSIDE
        # the replayer would instead happen inside the store call itself,
        # crashing test setup. And the OLD tamper-in-place pattern (editing
        # an existing record's meta.json to match forged bytes at its
        # UNCHANGED address) fails self-authentication -- the store
        # recomputes extraction_sha256 from meta.json's own fields on every
        # load, so a meta.json edited to claim a different extracted_sha256
        # no longer authenticates to the address it lives at, and
        # load_extraction_record returns None (UNVERIFIABLE for a different
        # reason, never reaching the parse guard this test targets).
        #
        # So this test hand-builds a brand-new, self-authenticating record
        # at a FRESH address computed from the forged bytes' own digest --
        # exactly what store_extraction_record would compute, just without
        # running the parse it would perform along the way -- reusing the
        # real record's own extractor/extractor_code_sha256/pypdf_version/
        # identity_payload_version so only extracted_sha256 (and therefore
        # the address) differs. The envelope's ExtractionBinding is then
        # repointed at this new address. Every guard before the parse step
        # (raw.bin integrity, record self-authentication, extracted_sha256
        # anchor, sidecar extracted_sha256 cross-check) passes cleanly; the
        # ONLY thing that can reject this input is the
        # ``except (ValueError, RecursionError):`` parse guard around
        # ``ExtractedText.model_validate(json.loads(raw_bytes))`` in
        # ``_independently_verify_node_text``.
        _, loaded = _produce_and_load(tmp_path)

        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        binding = node.extraction
        existing_meta = load_extraction_record(tmp_path, binding.parent_raw_sha256, binding.extraction_sha256)
        assert existing_meta is not None

        # Build a deeply, deeply nested JSON array as the value of a field
        # ExtractedText doesn't even declare -- pydantic's `extra="forbid"`
        # validation still has to walk/reject the payload, and json.loads
        # itself recurses per nesting level, so a large enough depth blows
        # the recursion limit during parse/validate, independent of the top
        # -level dict's own shape.
        deep_depth = 1_000_000
        nested = "[" * deep_depth + "]" * deep_depth
        payload = (
            f'{{"text": {json.dumps("x")}, "normalized": {json.dumps("x")}, '
            f'"extractor": "pdf:pypdf", "lossy": false, "bogus_extra_field": {nested}}}'
        )
        forged_bytes = payload.encode("utf-8")
        forged_extracted_sha256 = hashlib.sha256(forged_bytes).hexdigest()
        # Reuse the real record's own extracted_text_sha256: this fixture
        # targets the parse guard, not the text-sha guard, so every OTHER
        # identity field is left exactly as the real record recorded it.
        identity_payload = {
            "identity_payload_version": existing_meta.identity_payload_version,
            "parent_raw_sha256": existing_meta.parent_raw_sha256,
            "extractor": existing_meta.extractor,
            "extractor_code_sha256": existing_meta.extractor_code_sha256,
            "extracted_sha256": forged_extracted_sha256,
            "extracted_text_sha256": existing_meta.extracted_text_sha256,
            "pypdf_version": existing_meta.pypdf_version,
        }
        forged_extraction_sha256 = compute_extraction_sha(identity_payload)
        forged_dir = extraction_record_dir(tmp_path, existing_meta.parent_raw_sha256, forged_extraction_sha256)
        forged_dir.mkdir(parents=True)
        (forged_dir / "extracted.json").write_bytes(forged_bytes)
        (forged_dir / "text.txt").write_text("x", encoding="utf-8")
        (forged_dir / "meta.json").write_text(
            json.dumps(
                {
                    "extraction_sha256": forged_extraction_sha256,
                    "parent_raw_sha256": existing_meta.parent_raw_sha256,
                    "extractor": existing_meta.extractor,
                    "extractor_code_sha256": existing_meta.extractor_code_sha256,
                    "pypdf_version": existing_meta.pypdf_version,
                    "extracted_sha256": forged_extracted_sha256,
                    "extracted_text_sha256": existing_meta.extracted_text_sha256,
                    "identity_payload_version": existing_meta.identity_payload_version,
                    "stored_at": existing_meta.stored_at,
                }
            ),
            encoding="utf-8",
        )
        # Confirm the fixture really does self-authenticate before using it
        # -- if it didn't, the test would vacuously pass via the WRONG
        # guard (UNVERIFIABLE from a failed record load, not from parsing).
        assert load_extraction_record(tmp_path, existing_meta.parent_raw_sha256, forged_extraction_sha256) is not None

        forged_extraction = binding.model_copy(
            update={"extraction_sha256": forged_extraction_sha256, "extracted_sha256": forged_extracted_sha256}
        )
        forged_node = node.model_copy(update={"extraction": forged_extraction})
        forged_nodes = tuple(forged_node if n is node else n for n in loaded.source_graph.nodes)
        forged_graph = loaded.source_graph.model_copy(update={"nodes": forged_nodes})
        tampered = loaded.model_copy(update={"source_graph": forged_graph})

        report = replay_envelope(tmp_path, tampered)  # must not raise
        assert report.outcome is not ReplayOutcome.VERIFIED
        assert any(
            "parse" in f.reason.lower() or "RecursionError" in f.reason for f in report.unverifiable
        )


class TestReplayEnvelopeReportsPartialCounts:
    def test_total_and_unchecked_char_spans_are_reported(self, tmp_path: Path) -> None:
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        report = replay_envelope(tmp_path, loaded)
        assert report.total_char_spans == report.checked_char_spans
        assert report.unchecked_char_spans == 0
        assert report.total_char_spans == report.checked_char_spans + report.unchecked_char_spans


class TestReplayEnvelopeSurfacesUnreferencedNodeProblems:
    """Hole #2 (round 45): a problem on a node must be reported even when
    no char-span ref happens to reach that node -- an unreferenced node's
    tampered/missing evidence must never be silently invisible."""

    def test_unreferenced_node_with_missing_evidence_is_reported(self, tmp_path: Path) -> None:
        stored_artifact, loaded = _produce_and_load(tmp_path)

        extra_node = loaded.source_graph.node("paper").model_copy(
            update={"node_id": "orphan", "parent_node_id": None, "sha256": "1" * 64}
        )
        extra_nodes = loaded.source_graph.nodes + (extra_node,)
        extra_graph = loaded.source_graph.model_copy(update={"nodes": extra_nodes})
        tampered = loaded.model_copy(update={"source_graph": extra_graph})

        report = replay_envelope(tmp_path, tampered)
        assert report.outcome is ReplayOutcome.UNVERIFIABLE
        assert any("orphan" in f.reason for f in report.unverifiable)


class TestReplayEnvelopeVerifiesRawBinIntegrity:
    """Replay never touched ``raw.bin`` at all: it hashed and cross-checked
    ``extracted.json`` and its parsed text, but a node's ``sha256`` is a
    content-address of ``raw.bin``, and nothing ever confirmed that file
    exists or hashes to it. A missing or corrupted ``raw.bin`` behind a node
    was therefore invisible to replay as long as ``extracted.json`` still
    checked out -- a vacuous pass on the very artifact the node claims to be
    the content-address of.
    """

    def test_replay_is_unverifiable_when_raw_bin_is_missing(self, tmp_path: Path) -> None:
        stored_artifact, loaded = _produce_and_load(tmp_path)
        raw_path = artifact_dir(tmp_path, stored_artifact.sha256) / "raw.bin"
        raw_path.unlink()

        report = replay_envelope(tmp_path, loaded)

        assert report.outcome is ReplayOutcome.UNVERIFIABLE
        assert any("raw.bin" in f.reason for f in report.unverifiable)

    def test_replay_fails_when_raw_bin_does_not_hash_to_node_sha256(self, tmp_path: Path) -> None:
        stored_artifact, loaded = _produce_and_load(tmp_path)
        raw_path = artifact_dir(tmp_path, stored_artifact.sha256) / "raw.bin"
        raw_path.write_bytes(b"forged raw bytes that do not hash to node.sha256")

        report = replay_envelope(tmp_path, loaded)

        assert report.outcome is ReplayOutcome.FAILED
        assert any("raw.bin" in f.reason for f in report.failures)


class TestReplayEnvelopeChecksLossyExtraction:
    """Replay ignored ``ExtractedText.lossy``/``page_failures`` entirely.
    ``dataset_producer.produce_envelope_from_artifact`` itself refuses to
    produce an envelope from a knowingly-partial extraction, but replay had
    no equivalent check: an existing envelope whose stored ``extracted.json``
    was later mutated to ``lossy=True`` (e.g. by re-running extraction after
    a pypdf upgrade that now surfaces a previously-silent page failure) would
    still be reported VERIFIED by replay, because every digest it checks
    still lines up -- lossiness just isn't one of the things checked.
    """

    def test_replay_is_unverifiable_against_a_lossy_extraction(self, tmp_path: Path) -> None:
        # As with the RecursionError fixture above, in-place tampering under
        # the record's UNCHANGED address cannot isolate this guard: editing
        # meta.json to match mutated extracted.json bytes breaks self-
        # authentication (a stale meta.json's recomputed extraction_sha256
        # would no longer equal the address it lives at), so
        # load_extraction_record would return None -- UNVERIFIABLE for
        # "no such record", never reaching the lossy check this test
        # targets. Unlike the RecursionError fixture, though, `lossy=True`
        # parses perfectly cleanly as ExtractedText, so this one can go
        # through the REAL store_extraction_record front door: mint a
        # legitimate new record at a fresh, genuinely self-authenticating
        # address, reusing the existing record's own extractor/
        # extractor_code_sha256/pypdf_version so only lossy/page_failures
        # (and, as a consequence, extracted_sha256/extraction_sha256) differ
        # from the real record. extracted_text_sha256 is untouched: the text
        # itself doesn't change, only the lossy metadata wrapped around it,
        # so store_extraction_record recomputes the SAME extracted_text_sha256
        # from the SAME text -- leaving the envelope's own
        # extracted_text_sha256 field valid without needing to patch it.
        # After repointing the envelope at the new record, every guard
        # before the lossy check (raw.bin integrity, record self-
        # authentication, extracted_sha256 anchor + sidecar cross-check,
        # parse, extracted_text_sha256 + its sidecar cross-check) passes
        # cleanly; the ONLY thing that can reject this input is the
        # ``if extracted.lossy:`` guard in
        # ``_independently_verify_node_text``.
        _, loaded = _produce_and_load(tmp_path)

        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        binding = node.extraction
        existing_meta = load_extraction_record(tmp_path, binding.parent_raw_sha256, binding.extraction_sha256)
        assert existing_meta is not None
        existing_record_dir = extraction_record_dir(tmp_path, binding.parent_raw_sha256, binding.extraction_sha256)
        extracted = ExtractedText.model_validate(json.loads((existing_record_dir / "extracted.json").read_bytes()))

        lossy_extracted = extracted.model_copy(
            update={
                "lossy": True,
                "page_failures": (PageExtractionFailure(page=3, error="ValueError: bad xref"),),
            }
        )
        new_bytes = json.dumps(lossy_extracted.model_dump(mode="json")).encode("utf-8")
        new_extraction_sha256 = store_extraction_record(
            tmp_path,
            raw_sha256=existing_meta.parent_raw_sha256,
            extractor=existing_meta.extractor,
            extractor_code_sha256=existing_meta.extractor_code_sha256,
            pypdf_version=existing_meta.pypdf_version,
            extracted_json_bytes=new_bytes,
        )
        new_extracted_sha256 = hashlib.sha256(new_bytes).hexdigest()

        patched_extraction = binding.model_copy(
            update={"extraction_sha256": new_extraction_sha256, "extracted_sha256": new_extracted_sha256}
        )
        patched_node = node.model_copy(update={"extraction": patched_extraction})
        patched_nodes = tuple(patched_node if n.node_id == node.node_id else n for n in loaded.source_graph.nodes)
        patched_graph = loaded.source_graph.model_copy(update={"nodes": patched_nodes})
        tampered = loaded.model_copy(update={"source_graph": patched_graph})

        report = replay_envelope(tmp_path, tampered)

        assert report.outcome is ReplayOutcome.UNVERIFIABLE
        assert any("lossy" in f.reason and "1 page" in f.reason for f in report.unverifiable)


class TestReplayVerifiesAgainstTheRecordNotTheRootSidecar:
    """The addressed extraction record is self-authenticating: its own
    ``meta.json`` must recompute to the address it is stored under, and the
    envelope's ``ExtractionBinding`` carries every identity field needed to
    recompute that same address. The ROOT ``evidence/literature/<raw_sha>/
    meta.json`` sidecar is therefore not consulted by replay at all -- it
    proves only that a mutable sidecar agrees with itself, never anything
    about the addressed record. These tests pin both directions of that
    contract: a perfect record must VERIFY with the root sidecar gone
    (contract a), and no mutation of the root sidecar may move the outcome
    (contract d).
    """

    def test_replay_verifies_when_root_meta_json_is_missing(self, tmp_path: Path) -> None:
        """Contract (a): the nested record is present, self-authenticating,
        and agrees with the envelope on every digest -- deleting the ROOT
        sidecar removes nothing replay legitimately needs, so the outcome
        must be VERIFIED with no findings at all (not UNVERIFIABLE: nothing
        replay checks is missing). raw.bin stays in place, so the only thing
        that could turn this input non-VERIFIED is a replay-side dependence
        on the root sidecar itself -- the exact dependence this test exists
        to keep out.
        """
        stored_artifact, loaded = _produce_and_load(tmp_path)
        root_meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
        root_meta_path.unlink()

        report = replay_envelope(tmp_path, loaded)

        assert report.outcome is ReplayOutcome.VERIFIED
        assert report.findings == ()

    def test_replay_outcome_is_unmoved_by_a_tampered_root_meta_json(self, tmp_path: Path) -> None:
        """Contract (d), tamper flavour: rewrite the root sidecar's own
        identity claims (extractor_version, derivation_binding) to garbage
        that would have FAILED the old root-derived cross-check. The
        addressed record and the envelope are untouched, so the outcome must
        remain VERIFIED with no findings -- the root sidecar is not part of
        what an envelope asserts about itself.
        """
        stored_artifact, loaded = _produce_and_load(tmp_path)
        assert replay_envelope(tmp_path, loaded).outcome is ReplayOutcome.VERIFIED

        root_meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
        raw_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
        raw_meta["extractor_version"] = "forged-extractor-identity"
        raw_meta["derivation_binding"] = "0" * 64
        root_meta_path.write_text(json.dumps(raw_meta), encoding="utf-8")

        report = replay_envelope(tmp_path, loaded)

        assert report.outcome is ReplayOutcome.VERIFIED
        assert report.findings == ()

    def test_replay_outcome_is_unmoved_by_an_unparseable_root_meta_json(self, tmp_path: Path) -> None:
        """Contract (d), corruption flavour: the root sidecar is not even
        JSON any more. A replayer that still consulted it would crash or
        report UNVERIFIABLE; one that honestly ignores it must still return
        VERIFIED with no findings.
        """
        stored_artifact, loaded = _produce_and_load(tmp_path)
        assert replay_envelope(tmp_path, loaded).outcome is ReplayOutcome.VERIFIED

        root_meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
        root_meta_path.write_bytes(b"\x00not json at all")

        report = replay_envelope(tmp_path, loaded)

        assert report.outcome is ReplayOutcome.VERIFIED
        assert report.findings == ()


class TestReplayCrossChecksRecordIdentityAgainstTheBinding:
    """Contract (b): the envelope's ``ExtractionBinding`` and the stored
    record's ``meta.json`` each claim the extractor identity that produced
    the extraction; replay must cross-check the two and report a definite
    disagreement as FAILED -- the record is present and readable, so this is
    never inability-to-check (UNVERIFIABLE).
    """

    def test_replay_fails_when_binding_extractor_code_sha256_disagrees_with_the_record(
        self, tmp_path: Path
    ) -> None:
        """The binding's ``extractor_code_sha256`` is forged (via
        ``model_copy``, which bypasses schema validation exactly like a
        producer bug or a hand-edited envelope would) while
        ``extraction_sha256`` still addresses the real stored record. Every
        guard ahead of the identity cross-check passes on this input:
        raw.bin verifies, the record self-authenticates at its own address,
        extracted.json matches ``extracted_sha256``, and the text matches
        ``extracted_text_sha256`` -- only the identity cross-check can
        reject it, and it must say FAILED (the record exists and disagrees;
        the operator remedy is an integrity investigation, not a re-fetch).
        """
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)

        forged_code_sha = "0" * 64
        assert forged_code_sha != getattr(node.extraction, "extractor_code_sha256", None)
        patched_extraction = node.extraction.model_copy(update={"extractor_code_sha256": forged_code_sha})
        patched_node = node.model_copy(update={"extraction": patched_extraction})
        patched_nodes = tuple(patched_node if n.node_id == node.node_id else n for n in loaded.source_graph.nodes)
        patched_graph = loaded.source_graph.model_copy(update={"nodes": patched_nodes})
        tampered = loaded.model_copy(update={"source_graph": patched_graph})

        report = replay_envelope(tmp_path, tampered)

        assert report.outcome is ReplayOutcome.FAILED
        assert any("extractor_code_sha256" in f.reason for f in report.failures)
        # The record was present and readable throughout -- nothing about
        # this input is inability-to-check.
        assert not any("extractor_code_sha256" in f.reason for f in report.unverifiable)

    def test_replay_fails_when_binding_pypdf_version_disagrees_with_the_record(self, tmp_path: Path) -> None:
        """Same contract, for the pypdf-dependent identity half: the
        synthetic artifact is stored with extractor="pdf:pypdf", so the
        record's identity includes the pypdf version that ran, and a binding
        claiming a different one is a definite disagreement -- FAILED.
        """
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)

        patched_extraction = node.extraction.model_copy(update={"pypdf_version": "0.0.0-forged"})
        patched_node = node.model_copy(update={"extraction": patched_extraction})
        patched_nodes = tuple(patched_node if n.node_id == node.node_id else n for n in loaded.source_graph.nodes)
        patched_graph = loaded.source_graph.model_copy(update={"nodes": patched_nodes})
        tampered = loaded.model_copy(update={"source_graph": patched_graph})

        report = replay_envelope(tmp_path, tampered)

        assert report.outcome is ReplayOutcome.FAILED
        assert any("pypdf_version" in f.reason for f in report.failures)


def _char_span_ref(node_id: str, start: int, end: int) -> SourceRef:
    return SourceRef(
        node_id=node_id,
        locator=CharSpanLocator(kind=LocatorKind.CHAR_SPAN, text_space=TextSpace.EXTRACTED_TEXT, start=start, end=end),
    )


def _constructed_measured_value(
    *,
    quantity_kind: QuantityKind = QuantityKind.TEMPERATURE,
    unit_raw: str = "K",
    unit_normalized: str = "K",
    conversion_table_sha256: str = TABLE_V1.sha256,
    value_ref: SourceRef,
    unit_ref: SourceRef,
) -> MeasuredValue:
    """A ``MeasuredValue`` built via ``model_construct`` -- the established
    idiom in this module (see ``TestVerifyMeasuredValueUnit``) for shapes a
    validated constructor would refuse, which can only arise from a
    corrupted or forged record."""
    return MeasuredValue.model_construct(
        raw_text="1023",
        canonical_decimal_value="1023",
        repairs=(),
        repair_dependency=SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        ),
        quantity_kind=quantity_kind,
        unit_raw=unit_raw,
        unit_normalized=unit_normalized,
        conversion_table_sha256=conversion_table_sha256,
        value_ref=value_ref,
        unit_ref=unit_ref,
    )


_SYNTHETIC_NODE_PROBLEM = ReplayFinding(
    category=ReplayOutcome.FAILED,
    ref_path="source_graph.node('paper')",
    reason="synthetic node-level FAILED problem: raw bytes on disk do not hash to node.sha256",
)
"""A node-level problem whose category is FAILED -- the discriminating case
for the propagation guards: a mutation that disables propagation falls
through to the unknown-node branch, which reports UNVERIFIABLE, so only a
FAILED problem can tell propagation apart from that fallback."""


class TestIndependentNodeVerificationOutcomeCategories:
    """Pins the outcome CATEGORY of ``_independently_verify_node_text``'s
    inability-to-check guards, each isolated so that the guard under test is
    the only thing between the input and a clean ``(text, None)`` return.

    The three-outcome model is load-bearing: UNVERIFIABLE ("cannot check";
    operator re-fetches evidence or shrugs at a GC'd store) must never be
    conflated with FAILED ("checked and caught tampering"; operator starts an
    integrity investigation)."""

    def test_node_without_extraction_binding_is_unverifiable_even_with_intact_raw_bin(self, tmp_path: Path) -> None:
        """Guards satisfied by construction: the artifact is genuinely
        stored, so ``raw.bin`` exists AND hashes to ``node.sha256`` --
        neither the missing-raw.bin guard (UNVERIFIABLE) nor the
        tampered-raw.bin guard (FAILED) can produce this finding, and the
        record-address checks further down would crash on an ``Absent``
        binding rather than report this category. Only the absent-binding
        guard stands between this node and the rest of the function."""
        stored = _store_synthetic_artifact(tmp_path, _TEXT)
        node = SourceNode(
            node_id="raw-only",
            kind=SourceNodeKind.PAPER_PDF,
            sha256=stored.sha256,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
            glyph_health=Absent(reason=AbsenceReason.NOT_EXTRACTED_YET),
        )
        text, finding = _independently_verify_node_text(tmp_path, node)
        assert text is None
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        # The operationally meaningful distinction: NOTHING was ever
        # recorded for this node (the binding itself is absent) -- not "a
        # recorded artifact is missing or damaged on disk".
        assert "ExtractionBinding" in finding.reason

    def test_corrupt_record_meta_json_is_unverifiable_not_failed(self, tmp_path: Path) -> None:
        """Guards satisfied by construction: raw.bin is untouched (identity
        checks pass), and the record DIRECTORY exists at the addressed
        location -- only its ``meta.json`` is rewritten to non-JSON, which
        ``load_extraction_record`` surfaces as ``ExtractionRecordError``
        (corruption-of-the-sidecar, distinct from the record-absent path
        that returns ``None``). An unreadable sidecar blocks the check from
        running; it is not itself proof the EVIDENCE was tampered with, so
        the category must be UNVERIFIABLE, never FAILED."""
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        record_dir = extraction_record_dir(
            tmp_path, node.extraction.parent_raw_sha256, node.extraction.extraction_sha256
        )
        (record_dir / "meta.json").write_text("{this is not json", encoding="utf-8")

        text, finding = _independently_verify_node_text(tmp_path, node)
        assert text is None
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        # Distinguish "the sidecar is unreadable" from "no record stored at
        # this address" (the record-meta-None guard) and from every
        # raw.bin-related guard.
        assert "meta.json" in finding.reason
        assert "unreadable" in finding.reason

    def test_missing_extracted_json_with_intact_meta_is_unverifiable_not_failed(self, tmp_path: Path) -> None:
        """Guards satisfied by construction: raw.bin verifies, ``meta.json``
        is left fully intact and self-authenticating (so the unreadable-meta
        and record-absent guards pass), and the binding's identity fields
        still match the record's (so the identity cross-check passes). Only
        ``extracted.json`` is deleted -- absence of the evidence file is
        inability to check (re-fetch and retry), never positive evidence of
        tampering, so the category must be UNVERIFIABLE."""
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        record_dir = extraction_record_dir(
            tmp_path, node.extraction.parent_raw_sha256, node.extraction.extraction_sha256
        )
        (record_dir / "extracted.json").unlink()

        text, finding = _independently_verify_node_text(tmp_path, node)
        assert text is None
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "extracted.json" in finding.reason


class TestReplaySidecarBookkeepingInconsistencyIsFailed:
    """The two late sidecar cross-checks in ``_independently_verify_node_text``:
    a record that IS present, readable, and self-authenticated to its own
    address, whose ``meta.json`` nevertheless records an ``extracted_sha256``
    (or ``extracted_text_sha256``) that disagrees with the bytes actually on
    disk. The record existed and was readable throughout, so this is FAILED
    (inconsistent bookkeeping is positive evidence of alteration outside
    validated construction), never inability-to-check.

    Reaching these guards requires a record forged to self-authenticate at a
    FORGED address (its identity payload hashes to the directory name it
    lives under) plus a binding altered outside validated construction to
    point there -- exactly the "forged envelope field, hand-edited record"
    scenario the module docstring names. ``model_copy`` bypasses
    ``ExtractionBinding``'s self-authentication validator the same way the
    established forged-binding tests in this module do
    (``TestReplayCrossChecksRecordIdentityAgainstTheBinding``); through
    fully-validated construction these two guards are unreachable, which is
    exactly why they exist as defense-in-depth -- replay never trusts the
    envelope's validators to have run.
    """

    @staticmethod
    def _forge_record_with(tmp_path: Path, loaded: DatasetEnvelope, forged_field: str) -> tuple[SourceNode, str, str]:
        """Clone the real extraction record to a new address whose identity
        payload carries a decoy sha in ``forged_field``, so the forged
        ``meta.json`` STILL self-authenticates (its own fields hash to the
        directory name it lives under) and ``load_extraction_record``
        returns it rather than ``None``. Returns ``(forged_node,
        decoy_sha, real_sha_for_that_field)``.

        Guards satisfied by construction, in order: raw.bin verifies (the
        store is untouched); the forged record authenticates at its own
        address (record-absent and unreadable-meta guards pass); the
        identity cross-check passes (extractor, extractor_code_sha256,
        identity_payload_version, and pypdf_version are all copied from the
        real record on BOTH the forged meta and the forged binding); and
        the envelope-anchored digest checks pass (the binding keeps the
        REAL ``extracted_sha256``/``extracted_text_sha256``, and the copied
        ``extracted.json`` bytes are the real ones). Only the sidecar
        cross-check under test can reject this input."""
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        binding = node.extraction
        real_dir = extraction_record_dir(tmp_path, binding.parent_raw_sha256, binding.extraction_sha256)
        meta = json.loads((real_dir / "meta.json").read_text(encoding="utf-8"))
        decoy_sha = hashlib.sha256(b"synthetic decoy digest").hexdigest()
        real_sha = meta[forged_field]
        assert decoy_sha != real_sha

        forged_payload = {
            "identity_payload_version": meta["identity_payload_version"],
            "parent_raw_sha256": meta["parent_raw_sha256"],
            "extractor": meta["extractor"],
            "extractor_code_sha256": meta["extractor_code_sha256"],
            "pypdf_version": meta["pypdf_version"],
            "extracted_sha256": meta["extracted_sha256"],
            "extracted_text_sha256": meta["extracted_text_sha256"],
        }
        forged_payload[forged_field] = decoy_sha
        forged_address = compute_extraction_sha(forged_payload)

        forged_dir = extraction_record_dir(tmp_path, binding.parent_raw_sha256, forged_address)
        forged_dir.mkdir(parents=True)
        for item in real_dir.iterdir():
            (forged_dir / item.name).write_bytes(item.read_bytes())
        forged_meta = dict(meta)
        forged_meta[forged_field] = decoy_sha
        forged_meta["extraction_sha256"] = forged_address
        (forged_dir / "meta.json").write_text(json.dumps(forged_meta), encoding="utf-8")

        # Isolation probe: the forged record must genuinely authenticate at
        # its forged address -- otherwise this test would only be
        # re-exercising the record-absent guard, not the one under test.
        assert load_extraction_record(tmp_path, binding.parent_raw_sha256, forged_address) is not None

        forged_binding = binding.model_copy(update={"extraction_sha256": forged_address})
        return node.model_copy(update={"extraction": forged_binding}), decoy_sha, real_sha

    def test_record_meta_extracted_sha256_disagreeing_with_bytes_on_disk_is_failed(self, tmp_path: Path) -> None:
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        forged_node, decoy_sha, real_sha = self._forge_record_with(tmp_path, loaded, "extracted_sha256")

        text, finding = _independently_verify_node_text(tmp_path, forged_node)
        assert text is None
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        # The finding must name the disagreement itself: the sidecar's
        # inconsistent claim vs the digest of the bytes actually on disk.
        assert finding.actual == decoy_sha
        assert finding.expected == real_sha

    def test_record_meta_extracted_text_sha256_disagreeing_with_reread_text_is_failed(self, tmp_path: Path) -> None:
        """Additionally passes the sibling ``extracted_sha256`` sidecar
        cross-check (that field is left real), so the LAST guard in the
        function is the only one that can reject this input."""
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        forged_node, decoy_sha, real_sha = self._forge_record_with(tmp_path, loaded, "extracted_text_sha256")

        text, finding = _independently_verify_node_text(tmp_path, forged_node)
        assert text is None
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert finding.actual == decoy_sha
        assert finding.expected == real_sha


class TestCheckCharSpansOutcomeCategories:
    """Pins the outcome CATEGORY of ``check_char_spans``'s per-ref guards by
    calling it directly, so no earlier stage of ``replay_envelope`` (node
    verification, node-level findings) can mask which branch produced the
    finding."""

    def test_node_problem_category_and_reason_are_propagated_to_span_findings(self, tmp_path: Path) -> None:
        """A ref pointing at a node with a recorded node-level problem must
        inherit that problem's CATEGORY and REASON -- the propagation guard
        chooses no literal category of its own. Using a FAILED problem is
        the discriminating construction: with propagation disabled, the
        lookup falls through to the unknown-node branch, which reports
        UNVERIFIABLE, so a FAILED problem is the only way to tell the two
        apart. Other guards satisfied by construction: the envelope is
        produced and loaded for real, so every ref is a well-formed
        CharSpanLocator (the isinstance early-return does not fire)."""
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        checked, total, findings = check_char_spans(loaded, {}, {"paper": _SYNTHETIC_NODE_PROBLEM})
        assert checked == 0
        assert total > 0
        assert len(findings) == total
        assert all(f.category is ReplayOutcome.FAILED for f in findings)
        assert all(f.reason == _SYNTHETIC_NODE_PROBLEM.reason for f in findings)

    def test_ref_to_unknown_node_is_unverifiable_never_failed(self, tmp_path: Path) -> None:
        """A ref whose node has neither verified text nor a recorded node
        problem was never independently checked at all -- that is inability
        to check, never demonstrated disagreement, so every finding must be
        UNVERIFIABLE. Other guards satisfied by construction: node_problems
        is empty (the propagation guard cannot fire), and the refs are real
        CharSpanLocators from a produced envelope (the isinstance
        early-return does not fire); with this guard disabled the code
        would crash slicing ``None``."""
        _stored_artifact, loaded = _produce_and_load(tmp_path)
        checked, total, findings = check_char_spans(loaded, {}, {})
        assert checked == 0
        assert total > 0
        assert len(findings) == total
        assert all(f.category is ReplayOutcome.UNVERIFIABLE for f in findings)
        assert all("never independently checked" in f.reason for f in findings)
        assert not any(f.category is ReplayOutcome.FAILED for f in findings)

    def test_reachable_ref_unpaired_by_the_field_walk_trips_the_accounting_guard(self, tmp_path: Path) -> None:
        """The count cross-check against ``iter_source_refs`` exists to catch
        a FUTURE ref-bearing field added without updating
        ``check_char_spans``'s field-by-field pairing. This test simulates
        exactly that future: a subclass declares one extra ``SourceRef``
        field that ``iter_source_refs`` (generic over model fields) sees but
        the pairing never checks. Other guards satisfied by construction:
        every PAIRED ref checks cleanly (the real text is supplied for the
        real node), so the accounting guard's finding is the only finding.
        The category must be FAILED: a reachable-but-unchecked ref means
        the replayer's own coverage claim is wrong, which must never read
        as a mere inability to check."""

        class _EnvelopeWithUncheckedRef(DatasetEnvelope):
            decoy_ref: SourceRef

        _stored_artifact, loaded = _produce_and_load(tmp_path)
        envelope = _EnvelopeWithUncheckedRef.model_construct(**dict(loaded), decoy_ref=_char_span_ref("paper", 0, 3))
        checked, total, findings = check_char_spans(envelope, {"paper": _TEXT}, {})
        assert checked == total - 1
        assert len(findings) == 1
        finding = findings[0]
        assert finding.category is ReplayOutcome.FAILED
        assert "iter_source_refs" in finding.reason


class TestVerifyMeasuredValueUnitNormalizationDisagreement:
    def test_recorded_normalization_disagreeing_with_recorded_table_is_failed(self) -> None:
        """``unit_raw`` IS admitted by the recorded table (so the
        unknown-unit guard, which the existing not-a-real-unit test
        exercises, passes), and the table sha names a real table (so the
        unknown-table guard passes) -- the only thing left to reject this
        input is the final comparison of the recorded ``unit_normalized``
        against the table's own re-derived normalization. A resolvable
        table that disagrees is a completed check that did not pass:
        FAILED, never UNVERIFIABLE."""
        value = _constructed_measured_value(
            unit_raw="K",
            unit_normalized="degR",
            value_ref=_char_span_ref("paper", 0, 4),
            unit_ref=_char_span_ref("paper", 5, 6),
        )
        finding = verify_measured_value_unit("p", value)
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        # The finding targets the normalization field specifically, not the
        # raw unit (which the table admits) and not the table sha (which
        # resolves).
        assert finding.ref_path == "p.unit_normalized"
        assert finding.actual == "degR"
        assert finding.expected == "K"


class TestUnitBoundaryOutcomeCategories:
    """Pins the outcome CATEGORY of each ``verify_measured_value_unit_boundary``
    guard via direct calls, constructed so the guard under test is the only
    one standing between the input and a clean ``None`` return."""

    def test_node_problem_category_and_reason_are_propagated(self) -> None:
        """Same discriminating construction as the ``check_char_spans``
        propagation test: a FAILED node problem, because with propagation
        disabled the lookup falls through to the unknown-node branch and
        reports UNVERIFIABLE. Guards satisfied by construction: the
        unit_ref locator IS a CharSpanLocator, so the non-char-span guard
        ahead of propagation passes."""
        value = _constructed_measured_value(
            value_ref=_char_span_ref("other", 0, 4),
            unit_ref=_char_span_ref("paper", 5, 6),
        )
        finding = verify_measured_value_unit_boundary("p", value, {}, {"paper": _SYNTHETIC_NODE_PROBLEM})
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert finding.reason == _SYNTHETIC_NODE_PROBLEM.reason
        assert finding.ref_path == "p.unit_ref"

    def test_unknown_node_is_unverifiable_never_failed(self) -> None:
        """Guards satisfied by construction: CharSpanLocator (non-char-span
        guard passes) and empty node_problems (propagation guard passes).
        With this guard disabled the code would crash on ``len(None)``. An
        unchecked node is inability to check: UNVERIFIABLE, never FAILED."""
        value = _constructed_measured_value(
            value_ref=_char_span_ref("ghost", 0, 4),
            unit_ref=_char_span_ref("ghost", 5, 6),
        )
        finding = verify_measured_value_unit_boundary("p", value, {}, None)
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "never independently checked" in finding.reason

    def test_out_of_range_locator_is_failed_naming_the_range(self) -> None:
        """An out-of-range locator against text whose identity ALREADY
        verified is a definite disagreement between the envelope's claim
        and the evidence -- FAILED. Guards satisfied by construction:
        CharSpanLocator, no node problem, text present. The reason must
        name the range problem: with the guard disabled, Python's
        clamping slice semantics would hand the boundary layers a
        DIFFERENT quote and produce some other finding (or none), so
        asserting the range-specific reason is what actually pins this
        guard rather than its neighbours."""
        text = "held at 1023 K in the reactor"
        value = _constructed_measured_value(
            value_ref=_char_span_ref("other", 8, 12),
            unit_ref=_char_span_ref("paper", 13, 999),
        )
        finding = verify_measured_value_unit_boundary("p", value, {"paper": text}, None)
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert "out of range" in finding.reason

    def test_layer3_admission_violation_is_failed(self) -> None:
        """Layers 1-2 pass by construction: the span slices exactly the
        whitespace-delimited token "Zz" (clean boundaries on both sides, no
        digit run glued to it, and value_ref points at a DIFFERENT node so
        no value_span is recovered or needed). The table sha resolves (so
        the unknown-table UNVERIFIABLE guard passes) and quantity_kind is
        not OTHER (so the unmodelled-quantity guard passes). "Zz" is not in
        the recorded table's TEMPERATURE vocabulary, so only the Layer 3
        admission re-check can reject this input -- a check that ran and
        refused: FAILED, never UNVERIFIABLE."""
        text = "held at 1023 Zz in the reactor"
        value = _constructed_measured_value(
            unit_raw="Zz",
            unit_normalized="Zz",
            value_ref=_char_span_ref("other", 8, 12),
            unit_ref=_char_span_ref("paper", 13, 15),
        )
        assert text[13:15] == "Zz"
        finding = verify_measured_value_unit_boundary("p", value, {"paper": text}, None)
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        # Layer 3 (table admission), not the table-free Layers 1-2, must be
        # what rejected it.
        assert "admission" in finding.reason


class TestValueBoundaryOutcomeCategories:
    """Symmetric with ``TestUnitBoundaryOutcomeCategories``, for
    ``verify_measured_value_value_boundary``'s guards."""

    def test_node_problem_category_and_reason_are_propagated(self) -> None:
        """FAILED node problem as the discriminating construction (see the
        unit-boundary twin). CharSpanLocator by construction, so the
        non-char-span guard passes."""
        value = _constructed_measured_value(
            value_ref=_char_span_ref("paper", 8, 12),
            unit_ref=_char_span_ref("other", 13, 14),
        )
        finding = verify_measured_value_value_boundary("p", value, {}, {"paper": _SYNTHETIC_NODE_PROBLEM})
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert finding.reason == _SYNTHETIC_NODE_PROBLEM.reason
        assert finding.ref_path == "p.value_ref"

    def test_unknown_node_is_unverifiable_never_failed(self) -> None:
        """CharSpanLocator and empty node_problems by construction; with
        this guard disabled the code would crash on ``len(None)``."""
        value = _constructed_measured_value(
            value_ref=_char_span_ref("ghost", 8, 12),
            unit_ref=_char_span_ref("ghost", 13, 14),
        )
        finding = verify_measured_value_value_boundary("p", value, {}, None)
        assert finding is not None
        assert finding.category is ReplayOutcome.UNVERIFIABLE
        assert "never independently checked" in finding.reason

    def test_out_of_range_locator_is_failed_naming_the_range(self) -> None:
        """Same construction and rationale as the unit-boundary twin: the
        range-specific reason is asserted because a disabled guard would
        clamp the slice and produce a different finding (or none)."""
        text = "held at 1023 K in the reactor"
        value = _constructed_measured_value(
            value_ref=_char_span_ref("paper", 8, 999),
            unit_ref=_char_span_ref("other", 13, 14),
        )
        finding = verify_measured_value_value_boundary("p", value, {"paper": text}, None)
        assert finding is not None
        assert finding.category is ReplayOutcome.FAILED
        assert "out of range" in finding.reason
