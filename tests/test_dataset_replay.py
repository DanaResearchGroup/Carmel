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
)
from carmel.services.evidence import artifact_dir, store_artifact
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import QuantityKind

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
        # bytes digest self-consistent by NOT touching meta.json -- so the
        # extracted.json-bytes-vs-StoredArtifact.extracted_sha256 check
        # would itself fail first for a naive re-read of raw bytes; to
        # isolate the extracted_text_sha256 comparison specifically we craft
        # extracted.json bytes whose sha256 we then also patch into
        # meta.json, so the ONLY thing that disagrees is the recorded
        # ExtractionBinding.extracted_text_sha256 on the envelope itself.
        new_text = extracted.text + " EXTRA UNRECORDED SENTENCE."
        new_extracted = ExtractedText(
            text=new_text, normalized=new_text.casefold(), sections=[], extractor=extracted.extractor, lossy=False
        )
        new_bytes = json.dumps(new_extracted.model_dump(mode="json")).encode("utf-8")
        extracted_path.write_bytes(new_bytes)
        meta_path = artifact_dir(tmp_path, stored_artifact.sha256) / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["extracted_sha256"] = hashlib.sha256(new_bytes).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        report = replay_envelope(tmp_path, loaded)
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
