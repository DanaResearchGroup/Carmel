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

from carmel.agents.tools.extract import ExtractedText
from carmel.agents.tools.fetch import FetchedArtifact
from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisRole,
    CharSpanLocator,
    LocatorKind,
    MeasuredValue,
    SemanticDependencyUse,
    SourceRef,
    TextSpace,
    ValueOrigin,
)
from carmel.schemas.literature import StoredArtifact
from carmel.services.dataset_bridge import load_dataset_envelope, store_dataset_envelope
from carmel.services.dataset_producer import MeasurementSpec, produce_envelope_from_artifact
from carmel.services.dataset_replay import (
    ReplayOutcome,
    replay_envelope,
    verify_measured_value_unit,
    verify_measured_value_unit_boundary,
)
from carmel.services.evidence import artifact_dir, store_artifact
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
        # physically overwriting extracted.json on disk for the ORIGINAL
        # sha256 with the mutated text's ExtractedText payload, while
        # leaving StoredArtifact.meta.extracted_sha256 stale (pointing at
        # the original bytes) -- exactly the "extracted.json bytes on disk
        # were tampered with after store time" scenario, which must FAIL,
        # not silently re-verify.
        extracted_path = artifact_dir(tmp_path, original.sha256) / "extracted.json"
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


class TestReplayEnvelopeExtractedTextShaMismatch:
    def test_mismatched_extracted_text_sha256_fails_with_named_reason(self, tmp_path: Path) -> None:
        stored_artifact, loaded = _produce_and_load(tmp_path)
        extracted_path = artifact_dir(tmp_path, stored_artifact.sha256) / "extracted.json"
        raw_bytes = extracted_path.read_bytes()
        extracted = ExtractedText.model_validate(json.loads(raw_bytes))
        # Rewrite extracted.json with DIFFERENT text but keep the file's own
        # bytes digest self-consistent with what the ENVELOPE anchors on --
        # the replayer authenticates extracted.json against the envelope's
        # own ExtractionBinding.extracted_sha256 (never meta.json; see
        # dataset_replay's module docstring for why), so to isolate the
        # extracted_text_sha256 comparison specifically, both the envelope's
        # own anchor AND meta.json's copy are patched to the new bytes'
        # digest. That leaves the ONLY thing that disagrees as the recorded
        # ExtractionBinding.extracted_text_sha256 on the envelope itself.
        new_text = extracted.text + " EXTRA UNRECORDED SENTENCE."
        new_extracted = ExtractedText(
            text=new_text, normalized=new_text.casefold(), sections=[], extractor=extracted.extractor, lossy=False
        )
        new_bytes = json.dumps(new_extracted.model_dump(mode="json")).encode("utf-8")
        extracted_path.write_bytes(new_bytes)
        new_extracted_sha256 = hashlib.sha256(new_bytes).hexdigest()
        meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["extracted_sha256"] = new_extracted_sha256
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        patched_extraction = node.extraction.model_copy(update={"extracted_sha256": new_extracted_sha256})
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
        stored_artifact, loaded = _produce_and_load(tmp_path)
        clean_report = replay_envelope(tmp_path, loaded)
        assert clean_report.outcome is ReplayOutcome.VERIFIED

        extracted_path = artifact_dir(tmp_path, stored_artifact.sha256) / "extracted.json"
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

        # Patch meta.json's extracted_sha256 to match the forged bytes, so
        # a replayer that (wrongly) anchors on the mutable sidecar sees a
        # self-consistent record.
        meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
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
            "extracted_sha256" in f.reason or "extracted.json" in f.reason
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
        stored_artifact, loaded = _produce_and_load(tmp_path)

        extracted_path = artifact_dir(tmp_path, stored_artifact.sha256) / "extracted.json"

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
        extracted_path.write_bytes(forged_bytes)

        meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["extracted_sha256"] = hashlib.sha256(forged_bytes).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        # The envelope's own ExtractionBinding.extracted_sha256 must also
        # name the forged bytes so the run gets PAST the P0 anchor check
        # and actually reaches the parse step this test targets.
        node = loaded.source_graph.node("paper")
        assert not isinstance(node.extraction, Absent)
        forged_extraction = node.extraction.model_copy(
            update={"extracted_sha256": hashlib.sha256(forged_bytes).hexdigest()}
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
