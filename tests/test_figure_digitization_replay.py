# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Replay-time re-derivation of the figure lane's citation joins.

The five figure validators (V9, FD1-FD3, and ``EmbeddedFigureDigitization``'s own T1) fire only
when an envelope is constructed or loaded THROUGH the model. An envelope that reaches replay by a
route that skips model validation -- ``model_construct``, an in-memory object handed straight to
the replayer -- carries figure citations that nothing independent re-checked. These tests pin that
``carmel.services.dataset_replay._verify_figure_digitizations`` closes that hole: the failing
fixtures are built the way the schema CANNOT intercept, so a test that passed would still pass if
the validators were the only guard, is exactly the test this module refuses to be.

The counting rule is the load-bearing one (see the module under test): a figure check increments
NO "how much was checked" counter, so a replay whose only evidence is a figure still reads
UNVERIFIABLE. ``test_figure_checks_pass_yet_the_replay_is_unverifiable`` pins that directly.

Synthetic evidence only, exactly as ``tests/test_dataset_replay.py`` (the real corpus is
closed-access, non-redistributable).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from carmel.schemas.datasets import (
    AbsenceReason,
    Absent,
    AxisDeclaration,
    AxisRole,
    BBox,
    Coordinate,
    CoordinateFrame,
    DataPoint,
    DatasetEnvelope,
    EmbeddedFigureDigitization,
    MeasuredValue,
    SemanticDependencyUse,
    Series,
    SourceForm,
    SourceGraph,
    SourceNode,
    SourceNodeKind,
    SourceRef,
    ValueOrigin,
)
from carmel.services import units
from carmel.services.dataset_producer import _ACTIVE, _ROOT_NODE_ID, ground_quote
from carmel.services.dataset_replay import ReplayOutcome, SemanticGap, replay_envelope
from carmel.services.figure_digitization_record import digitization_record_bytes
from carmel.services.numeric import QuoteRole
from carmel.services.semantic_deps import CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID, current_sha_for
from carmel.services.units import TABLE_V1, QuantityKind
from tests.figure_digitization_fixtures import cite_digitization
from tests.test_dataset_replay import (
    _TEXT,
    _prepare_grounding,
    _store_synthetic_artifact,
)
from tests.test_series_digitization_citation import (
    _digitized_series,
    _envelope,
    _point,
    _valid_envelope,
)


def _replace_node(graph: SourceGraph, node_id: str, **changes: object) -> SourceGraph:
    """A copy of ``graph`` with node ``node_id`` rebuilt through ``model_construct`` -- so a
    corrupted node (wrong kind, wrong sha, a parent that forms a cycle) reaches replay exactly as
    it would on an envelope no validator ever saw."""
    nodes = tuple(_reconstruct(node, **changes) if node.node_id == node_id else node for node in graph.nodes)
    return SourceGraph.model_construct(nodes=nodes)


def _reconstruct[M: BaseModel](model: M, **changes: object) -> M:
    """Rebuild ``model`` with ``changes`` through ``model_construct`` -- so NO validator runs.

    This is the whole point: an envelope (or the record it embeds) that never passed the figure
    validators is precisely the case ``_verify_figure_digitizations`` exists for. Building the
    fixture through the ordinary constructor would prove only that the validators still work.
    """
    fields: dict[str, Any] = {name: getattr(model, name) for name in type(model).model_fields}
    fields.update(changes)
    return type(model).model_construct(**fields)


def _figure_findings(report: object) -> list[Any]:
    """The findings this ticket added: the record-level ones at ``figure_digitizations[...]`` and
    the per-series join ones whose path is EXACTLY ``series[i]`` (never the longer dotted paths the
    value-boundary gates emit)."""
    out = []
    for finding in report.findings:  # type: ignore[attr-defined]
        path = finding.ref_path
        record_level = path.startswith("figure_digitizations[")
        per_series = path.startswith("series[") and path.endswith("]") and "." not in path
        if record_level or per_series:
            out.append(finding)
    return out


class TestJoinsAreRederivedOnAModelConstructEnvelope:
    """Verifier 1 and 2: every failure is observed on an envelope built through
    ``model_construct``, and each asserts the SPECIFIC finding -- its category and its path --
    never merely that the report is not VERIFIED."""

    def test_a_record_whose_stored_address_disagrees_with_its_payload_is_caught(self) -> None:
        """Verifier 1. The embedded record's ``digitization_sha256`` is tampered to
        ``0``\\ *64 while its ``canonical_json`` still hashes to the true address, and the CITING
        series is pointed at the tampered address too -- so the ONLY thing wrong is that the record
        does not live where it says. A FAILED finding fires at the record's path, and it is the
        only figure finding.
        """
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        bad = "0" * 64
        tampered = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=bad, raw_sha256=good.raw_sha256, canonical_json=good.canonical_json
        )
        series = _reconstruct(env.series[0], digitization_sha256=bad)
        broken = _reconstruct(env, series=(series,), figure_digitizations=(tampered,))

        report = replay_envelope(Path("/nonexistent-store"), broken)

        figure = _figure_findings(report)
        assert len(figure) == 1, [f.reason for f in figure]
        finding = figure[0]
        assert finding.category is ReplayOutcome.FAILED
        assert finding.ref_path == f"figure_digitizations[{bad!r}]"
        assert "does not live at the address" in finding.reason

    def test_a_citation_resolving_to_no_embedded_record_is_caught(self) -> None:
        """Verifier 2a. The series still cites its real digitization, but the envelope embeds
        none -- a dangling citation. FAILED at the series path."""
        env = _valid_envelope()
        broken = _reconstruct(env, figure_digitizations=())

        report = replay_envelope(Path("/nonexistent-store"), broken)

        figure = _figure_findings(report)
        assert len(figure) == 1, [f.reason for f in figure]
        finding = figure[0]
        assert finding.category is ReplayOutcome.FAILED
        assert finding.ref_path == "series[0]"
        assert "does not embed" in finding.reason

    def test_an_embedded_record_no_series_cites_is_caught(self) -> None:
        """Verifier 2b. A second, well-formed record is embedded that no DIGITIZED series names --
        an orphan the exact-cover rule (FD1) forbids. FAILED at the orphan record's path."""
        env = _valid_envelope()
        cited = env.figure_digitizations[0]
        # A valid record for a DIFFERENT series id, embedded but never cited.
        orphan_series = _digitized_series(digitization_sha256="0" * 64, series_id="s2")
        _s2, orphan = cite_digitization(orphan_series, env.source_graph, "fig")
        assert orphan.digitization_sha256 != cited.digitization_sha256
        broken = _reconstruct(env, figure_digitizations=(cited, orphan))

        report = replay_envelope(Path("/nonexistent-store"), broken)

        # In memory, with no store, the orphan ALSO earns the honest "bytes not in hand"
        # UNVERIFIABLE at the same path; the cover violation is the FAILED one, and it is what
        # this test pins.
        orphan_findings = [
            f
            for f in _figure_findings(report)
            if f.ref_path == f"figure_digitizations[{orphan.digitization_sha256!r}]"
            and "no DIGITIZED series cites it" in f.reason
        ]
        assert len(orphan_findings) == 1, [f.reason for f in _figure_findings(report)]
        assert orphan_findings[0].category is ReplayOutcome.FAILED
        # The cited record is NOT flagged an orphan: exactly one series names it.
        assert not any(
            f.ref_path == f"figure_digitizations[{cited.digitization_sha256!r}]"
            and "no DIGITIZED series cites it" in f.reason
            for f in _figure_findings(report)
        )

    def test_a_series_id_mismatch_join_is_caught(self) -> None:
        """A V9 join re-derived: the record must be ABOUT the series that cites it. The series is
        renamed via ``model_construct`` while its citation stays, so the record is now about a
        different series id. FAILED at the series path."""
        env = _valid_envelope()
        series = _reconstruct(env.series[0], series_id="renamed")
        broken = _reconstruct(env, series=(series,))

        report = replay_envelope(Path("/nonexistent-store"), broken)

        mismatch = [f for f in _figure_findings(report) if f.ref_path == "series[0]" and "is about series" in f.reason]
        assert len(mismatch) == 1, [f.reason for f in _figure_findings(report)]
        assert mismatch[0].category is ReplayOutcome.FAILED

    def test_a_valid_in_memory_envelope_yields_no_failed_figure_finding(self) -> None:
        """The valid envelope, replayed WITHOUT its evidence store: the only figure finding is the
        honest UNVERIFIABLE that the document's hash-verified bytes are not in hand -- never a
        FAILED, because nothing about the citation disagrees."""
        env = _valid_envelope()

        report = replay_envelope(Path("/nonexistent-store"), env)

        figure = _figure_findings(report)
        assert all(f.category is ReplayOutcome.UNVERIFIABLE for f in figure), [f.reason for f in figure]
        assert not any(f.category is ReplayOutcome.FAILED for f in figure)
        assert any("no hash-verified raw bytes" in f.reason for f in figure)


class TestTheFigureLimitIsStatedInTheReport:
    """Half 2: the ceiling a coherent citation cannot pass, recorded as an
    ``UncheckedSemanticClaim`` per DIGITIZED series."""

    def test_each_digitized_series_gets_one_location_unresolved_semantic_claim(self) -> None:
        from carmel.services.dataset_replay import SemanticGap

        env = _valid_envelope()

        report = replay_envelope(Path("/nonexistent-store"), env)

        figure_claims = [c for c in report.unchecked_semantic_claims if c.claim_path == "series[0]"]
        assert len(figure_claims) == 1
        claim = figure_claims[0]
        assert claim.gap is SemanticGap.LOCATION_UNRESOLVED
        # The claim names the DERIVED value whose support went unchecked -- the series' DIGITIZED
        # source_form -- never the digitization_sha256 content address (which is not a derived value
        # and does not identify the recovered coordinates); see
        # ``test_the_claim_names_a_derived_value_not_the_content_address``.
        assert claim.claim == "digitized"
        assert claim.claim != env.figure_digitizations[0].digitization_sha256
        assert "operator attestation replay structurally cannot test" in claim.reason
        # A semantic claim downgrades the overall verdict, exactly as the condition-set
        # attribution claim does -- so a digitized envelope is never VERIFIED overall.
        assert report.overall_outcome is not ReplayOutcome.VERIFIED


class TestFigureChecksNeverLaunderVerified:
    """Verifier 3, on disk: a replay whose figure checks all PASS, with zero char spans re-sliced
    and zero cells matched, still reads UNVERIFIABLE -- because a coherent figure citation is not a
    reproduced measurement and increments no ``checked`` counter."""

    @staticmethod
    def _crop_node(node_id: str, sha256: str, parent_id: str) -> SourceNode:
        return SourceNode(
            node_id=node_id,
            kind=SourceNodeKind.FIGURE_CROP,
            sha256=sha256,
            parent_node_id=parent_id,
            origin=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            extraction=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            glyph_health=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            verification=Absent(reason=AbsenceReason.NOT_APPLICABLE),
            crop_region=BBox(
                frame=CoordinateFrame(
                    render_fingerprint="fp-fig",
                    cropbox=("0", "0", "612", "792"),
                    mediabox=("0", "0", "612", "792"),
                    rotation=0,
                    units="pt",
                    dpi="300",
                    render_settings="antialias=on",
                ),
                x0="72",
                y0="500",
                x1="300",
                y1="700",
            ),
        )

    def _on_disk_digitized_envelope(self, tmp_path: Path) -> DatasetEnvelope:
        # The ROOT document has real, hash-verified bytes in the store -- the join the figure
        # check confirms. The crop's own bytes are stored too, so no NODE-level complaint
        # confounds the one thing this test is about.
        root = _store_synthetic_artifact(tmp_path, "a synthetic paper the figure was cut from")
        crop = _store_synthetic_artifact(tmp_path, "synthetic crop bytes, not the parent's")
        grounding = _prepare_grounding(
            tmp_path, root.sha256, envelope_noun="dataset envelope", envelope_subject="A dataset"
        )
        root_node_id = grounding.graph.nodes[0].node_id
        crop_node = self._crop_node("fig", crop.sha256, root_node_id)
        graph = grounding.graph.model_copy(update={"nodes": (*grounding.graph.nodes, crop_node)})
        series = _digitized_series(digitization_sha256="0" * 64)
        series_cited, embedded = cite_digitization(series, graph, "fig")
        return _envelope(series=(series_cited,), figure_digitizations=(embedded,), graph=graph)

    def test_figure_checks_pass_yet_the_replay_is_unverifiable(self, tmp_path: Path) -> None:
        envelope = self._on_disk_digitized_envelope(tmp_path)

        report = replay_envelope(tmp_path, envelope)

        # The figure lane produced NO finding at all -- every join re-derived and held.
        assert _figure_findings(report) == [], [f.reason for f in _figure_findings(report)]
        # Nothing incremented either checked counter: a figure is not a re-sliced span or a cell.
        assert report.checked_char_spans == 0
        assert report.checked_table_cells == 0
        # So the widened zero-check fires, unchanged, and the report is UNVERIFIABLE not VERIFIED.
        assert report.evidence_outcome is ReplayOutcome.UNVERIFIABLE
        assert report.overall_outcome is ReplayOutcome.UNVERIFIABLE
        assert any(
            "matched ZERO table cells" in f.reason and "ZERO character spans" in f.reason for f in report.findings
        ), [f.reason for f in report.findings]

    def test_the_valid_figure_envelope_has_no_figure_lane_failure(self, tmp_path: Path) -> None:
        """A separate, blunt guard: the on-disk valid envelope has NO figure FAILED finding, so
        the UNVERIFIABLE above is the zero-check and the attested-coordinates limit, never a
        broken join."""
        envelope = self._on_disk_digitized_envelope(tmp_path)

        report = replay_envelope(tmp_path, envelope)

        assert not any(f.category is ReplayOutcome.FAILED for f in report.findings), [
            f.reason for f in report.findings if f.category is ReplayOutcome.FAILED
        ]


def _assert_one(report: object, *, path: str, category: ReplayOutcome, marker: str) -> Any:
    """Exactly one figure finding at ``path`` with ``category`` and ``marker`` in its reason --
    the specific branch, never merely 'some finding fired'."""
    hits = [f for f in _figure_findings(report) if f.ref_path == path and f.category is category and marker in f.reason]
    assert len(hits) == 1, (
        path,
        category,
        marker,
        [(f.category.value, f.ref_path, f.reason) for f in _figure_findings(report)],
    )
    return hits[0]


class TestEveryFailClosedBranchFires:
    """One test per fail-closed branch of ``_verify_figure_digitizations`` /
    ``_verify_series_digitization_joins``, each on a ``model_construct`` fixture and each asserting
    the branch's own category AND path -- a branch that decides what happens when a figure citation
    IS wrong must not be a guard nobody has watched fire."""

    _STORE = Path("/nonexistent-store")

    # --- Pass 1: the embedded record re-derived from its own bytes ------------------------------

    def test_canonical_json_that_will_not_parse_is_unverifiable(self) -> None:
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        sha = "1" * 64
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json="{ not valid json"
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report, path=f"figure_digitizations[{sha!r}]", category=ReplayOutcome.UNVERIFIABLE, marker="does not parse"
        )

    def test_canonical_json_that_will_not_recanonicalize_is_unverifiable(self) -> None:
        # Parses as JSON, but is a list -- ``compute_digitization_sha`` does ``dict(payload)`` and
        # raises, so the address could not even be recomputed.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        sha = "2" * 64
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json="[1, 2, 3]"
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report,
            path=f"figure_digitizations[{sha!r}]",
            category=ReplayOutcome.UNVERIFIABLE,
            marker="could not be re-canonicalized",
        )

    def test_a_record_that_will_not_reconstruct_is_failed(self) -> None:
        # Address MATCHES (recomputed over the mutated payload), so the parse and address checks
        # pass; ``from_payload`` then rejects a record whose ``coverage`` is not a FigureCoverage
        # member. That isolates the reconstruction branch on a payload that is genuinely malformed
        # at the KNOWN version (``payload_version`` untouched) -- so this keeps testing "the record
        # is WRONG -> FAILED", the fact it means to pin, rather than the reader's inability to read
        # an unknown version, which is now a different verdict (see ``test_an_unknown_payload_
        # version_is_unverifiable_naming_the_version``). Substituting the trigger is honest because
        # the old ``payload_version = 999`` was only ever a convenient way to reach this branch, not
        # an assertion about versions; a corrupt ``coverage`` reaches the SAME branch for the reason
        # the test's name states.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        payload = json.loads(good.canonical_json)
        payload["coverage"] = "not-a-real-coverage"
        canonical = digitization_record_bytes(payload)
        sha = hashlib.sha256(canonical).hexdigest()
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json=canonical.decode("utf-8")
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report, path=f"figure_digitizations[{sha!r}]", category=ReplayOutcome.FAILED, marker="does not reconstruct"
        )

    def test_an_unknown_payload_version_is_unverifiable_naming_the_version(self) -> None:
        # The pair to the FAILED test above, and the ticket's headline: a record whose ONLY defect
        # is a payload_version this reader does not know is NOT wrong, merely unreadable here. The
        # address matches (recomputed over the mutated payload) and the shape is otherwise a legal
        # version-1 record, so the sole thing ``from_payload`` refuses is the version -- which now
        # raises ``UnknownPayloadVersion`` and lands UNVERIFIABLE, not FAILED. The reason must NAME
        # the version, so the report says which shape it could not read.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        payload = json.loads(good.canonical_json)
        payload["payload_version"] = 999
        canonical = digitization_record_bytes(payload)
        sha = hashlib.sha256(canonical).hexdigest()
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json=canonical.decode("utf-8")
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        finding = _assert_one(
            report,
            path=f"figure_digitizations[{sha!r}]",
            category=ReplayOutcome.UNVERIFIABLE,
            marker="declares payload_version",
        )
        # Naming the version is the whole point: a report that could not name the shape it could not
        # read would be no better than the plain FAILED it replaces.
        assert "999" in finding.reason
        # And it is filed as the reader's admission, never as a charge against the record.
        assert "newer rather than wrong" in finding.reason
        # No FAILED "does not reconstruct" leaked out alongside it -- the version arm caught it first.
        assert not any(
            f.category is ReplayOutcome.FAILED and "does not reconstruct" in f.reason for f in report.findings
        )

    def test_a_payload_with_no_version_key_is_failed_and_declares_nothing(self) -> None:
        # A record whose payload has NO payload_version key at all. The first-cut narrowing read
        # ``payload.get("payload_version")`` as None, found None != 1, and raised
        # UnknownPayloadVersion -- reporting a KEYLESS payload as UNVERIFIABLE and, worse, as
        # "declares payload_version None", a version a payload with no version key never declared.
        # A payload that declares no version is a MALFORMED record, not a newer one: it falls to the
        # ordinary shape check, which names the missing key, so the verdict is FAILED and the reason
        # never claims the record declared a version. This is the pair that proves the boundary was
        # NARROWED (a keyless payload stays FAILED) rather than moved.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        payload = json.loads(good.canonical_json)
        del payload["payload_version"]
        canonical = digitization_record_bytes(payload)
        sha = hashlib.sha256(canonical).hexdigest()
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json=canonical.decode("utf-8")
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        finding = _assert_one(
            report, path=f"figure_digitizations[{sha!r}]", category=ReplayOutcome.FAILED, marker="does not reconstruct"
        )
        # The shape check names the missing key -- the honest reason -- and never the old overclaim
        # that the record "declares" a version it has none of.
        assert "missing keys" in finding.reason and "payload_version" in finding.reason
        assert "declares payload_version" not in finding.reason
        # And no UNVERIFIABLE version finding leaked out: a keyless payload is never an unknown VERSION.
        assert not any(
            f.category is ReplayOutcome.UNVERIFIABLE and "declares payload_version" in f.reason for f in report.findings
        )

    def test_a_payload_whose_version_is_the_wrong_type_is_failed(self) -> None:
        # payload_version present but a STRING "1" -- the same integer spelled as text. This is not a
        # newer version (a version bump writes a larger INTEGER, it never respells the discriminator);
        # it is a malformed record, so the verdict is FAILED. The narrowing matters precisely here:
        # the shape check below PASSES this payload (its key set is exactly right), so without an
        # explicit type refusal a record whose only defect is a mistyped version would reconstruct
        # clean and be reported as verified. A version is an ``int`` because ``DIGITIZATION_PAYLOAD_
        # VERSION`` is one and a version bump only ever raises that integer; a discriminator of a
        # different type is bytes this reader cannot trust to tell it how to read the rest.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        payload = json.loads(good.canonical_json)
        payload["payload_version"] = "1"
        canonical = digitization_record_bytes(payload)
        sha = hashlib.sha256(canonical).hexdigest()
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json=canonical.decode("utf-8")
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        finding = _assert_one(
            report, path=f"figure_digitizations[{sha!r}]", category=ReplayOutcome.FAILED, marker="does not reconstruct"
        )
        # A mistyped version is refused as a malformed discriminator, naming the offending type --
        # never mistaken for an unknown (newer) version.
        assert "not the integer a version is" in finding.reason
        assert not any(
            f.category is ReplayOutcome.UNVERIFIABLE and "declares payload_version" in f.reason for f in report.findings
        )

    def test_a_boolean_version_does_not_sail_through_as_version_one(self) -> None:
        # payload_version present as the JSON literal ``true``. ``isinstance(True, int)`` is True and
        # ``True == DIGITIZATION_PAYLOAD_VERSION`` (both 1), so an int-only gate would let this
        # OTHERWISE-VALID record reconstruct AS version 1 -- reading a mistyped record as a valid
        # one. This is why the gate also excludes ``bool``. The defect is invisible to a minimal
        # probe: ``{"payload_version": True}`` on its own is refused by the key-set check anyway, so
        # the payload here is a full, valid version-1 record whose ONLY defect is the boolean
        # version. Removing ``not isinstance(version, bool)`` from ``from_payload`` turns this FAILED
        # into a silent clean reconstruction, so this test is what pins the exclusion.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        payload = json.loads(good.canonical_json)
        assert payload["payload_version"] == 1  # the value True would otherwise compare equal to
        payload["payload_version"] = True
        canonical = digitization_record_bytes(payload)
        sha = hashlib.sha256(canonical).hexdigest()
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=sha, raw_sha256=good.raw_sha256, canonical_json=canonical.decode("utf-8")
        )
        series = _reconstruct(env.series[0], digitization_sha256=sha)
        env2 = _reconstruct(env, series=(series,), figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        finding = _assert_one(
            report, path=f"figure_digitizations[{sha!r}]", category=ReplayOutcome.FAILED, marker="does not reconstruct"
        )
        # Refused as a mistyped discriminator, named as a bool -- never accepted as version 1, and
        # never mistaken for an unknown (newer) version.
        assert "not the integer a version is" in finding.reason and "bool" in finding.reason
        assert not any(
            f.category is ReplayOutcome.UNVERIFIABLE and "declares payload_version" in f.reason for f in report.findings
        )

    def test_a_declared_raw_sha256_the_payload_disagrees_with_is_failed(self) -> None:
        # canonical_json unchanged (its payload names raw_sha256=SHA_A), but the embedded object's
        # OWN raw_sha256 field is a different digest an envelope-level join would have trusted.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=good.digitization_sha256, raw_sha256="d" * 64, canonical_json=good.canonical_json
        )
        env2 = _reconstruct(env, figure_digitizations=(broken,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report,
            path=f"figure_digitizations[{good.digitization_sha256!r}]",
            category=ReplayOutcome.FAILED,
            marker="not the declared raw_sha256",
        )

    # --- Pass 2: the series-record-crop joins re-walked ---------------------------------------

    def test_a_digitized_series_that_cites_nothing_is_unverifiable(self) -> None:
        env = _valid_envelope()
        series = _reconstruct(env.series[0], digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE))
        env2 = _reconstruct(env, series=(series,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report, path="series[0]", category=ReplayOutcome.UNVERIFIABLE, marker="cites no figure digitization"
        )

    def test_a_recovered_count_that_disagrees_with_the_series_is_failed(self) -> None:
        env = _valid_envelope()
        # The record recovered one marker; strip the series to zero points via model_construct.
        series = _reconstruct(env.series[0], points=())
        env2 = _reconstruct(env, series=(series,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(report, path="series[0]", category=ReplayOutcome.FAILED, marker="the series has 0 point(s)")

    def test_a_crop_id_absent_from_the_graph_is_failed(self) -> None:
        env = _valid_envelope()
        paper_only = SourceGraph.model_construct(nodes=(env.source_graph.nodes[0],))
        env2 = _reconstruct(env, source_graph=paper_only)

        report = replay_envelope(self._STORE, env2)

        _assert_one(report, path="series[0]", category=ReplayOutcome.FAILED, marker="not the id of any node")

    def test_a_crop_id_resolving_to_a_non_crop_is_failed(self) -> None:
        env = _valid_envelope()
        graph = _replace_node(env.source_graph, "fig", kind=SourceNodeKind.SI_MEMBER)
        env2 = _reconstruct(env, source_graph=graph)

        report = replay_envelope(self._STORE, env2)

        _assert_one(report, path="series[0]", category=ReplayOutcome.FAILED, marker="not a FIGURE_CROP")

    def test_a_crop_sha256_that_disagrees_with_the_node_is_failed(self) -> None:
        env = _valid_envelope()
        graph = _replace_node(env.source_graph, "fig", sha256="e" * 64)
        env2 = _reconstruct(env, source_graph=graph)

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report,
            path="series[0]",
            category=ReplayOutcome.FAILED,
            marker="two halves of the crop's identity must agree",
        )

    def test_a_point_grounding_in_a_different_crop_is_failed(self) -> None:
        # One point whose observation grounds in "paper", not the crop "fig" the record names --
        # the same-crop join. Built through model_construct because V9 refuses it at construction.
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        mixed = _digitized_series(digitization_sha256=good.digitization_sha256, points=(_point(obs_node="paper"),))
        env2 = _reconstruct(env, series=(mixed,))

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report, path="series[0]", category=ReplayOutcome.FAILED, marker="assembled from two different crops"
        )

    def test_a_crop_ancestry_that_cannot_be_walked_is_unverifiable(self) -> None:
        # A two-node cycle (paper<->fig) via model_construct: the root-bytes join walks ancestors,
        # which raises ValueError on the cycle rather than looping.
        env = _valid_envelope()
        graph = _replace_node(env.source_graph, "paper", parent_node_id="fig")
        env2 = _reconstruct(env, source_graph=graph)

        report = replay_envelope(self._STORE, env2)

        _assert_one(
            report, path="series[0]", category=ReplayOutcome.UNVERIFIABLE, marker="ancestry could not be walked"
        )

    def test_a_record_document_digest_that_misses_the_root_is_failed(self) -> None:
        # The record names raw_sha256=SHA_A (the paper), but the paper node's own sha is changed,
        # so the crop's ROOT no longer carries the digest the record chains to.
        env = _valid_envelope()
        graph = _replace_node(env.source_graph, "paper", sha256="f" * 64)
        env2 = _reconstruct(env, source_graph=graph)

        report = replay_envelope(self._STORE, env2)

        _assert_one(report, path="series[0]", category=ReplayOutcome.FAILED, marker="must reach the crop's root")


def _char_span_grounded_series_envelope(tmp_path: Path, source_form: object) -> DatasetEnvelope:
    """An envelope whose one series grounds its VALUE, unit and label in re-sliceable char spans
    over a stored document -- so replay re-slices them and the evidence lane reads VERIFIED.

    Built through ``model_construct`` for TWO reasons the fixture depends on: a char span cannot
    ground a series value through validation (P0-c), and ``source_form`` is handed in as a raw value
    so the caller can pass the plain string ``"digitized"`` the schema would coerce. The ONLY thing
    that varies between the honest and the spoofed envelope is ``source_form``; everything the
    evidence lane checks is identical, which is what makes the verdict difference attributable to the
    figure selector alone.
    """
    stored = _store_synthetic_artifact(tmp_path, _TEXT)
    grounding = _prepare_grounding(
        tmp_path, stored.sha256, envelope_noun="dataset envelope", envelope_subject="A dataset"
    )

    def char_span(quote: str, role: QuoteRole, quantity: QuantityKind | None = None) -> SourceRef:
        locator = (
            ground_quote(grounding.text, quote, role=role, quantity=quantity)
            if quantity is not None
            else ground_quote(grounding.text, quote, role=role)
        )
        return SourceRef(node_id=_ROOT_NODE_ID, locator=locator)

    value = MeasuredValue(
        raw_text="1023",
        canonical_decimal_value="1023",
        repairs=(),
        repair_dependency=SemanticDependencyUse(
            dependency_id=CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID,
            content_sha256=current_sha_for(CONTEXT_FREE_SPAN_REPAIR_DEPENDENCY_ID),
            input_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        ),
        quantity_kind=QuantityKind.TEMPERATURE,
        unit_raw="K",
        unit_normalized=units.normalize_unit(QuantityKind.TEMPERATURE, "K", table=_ACTIVE.table),
        conversion_table_sha256=TABLE_V1.sha256,
        value_ref=char_span("1023", QuoteRole.VALUE),
        unit_ref=char_span("K", QuoteRole.UNIT, QuantityKind.TEMPERATURE),
    )
    axis = AxisDeclaration(
        axis_id="temperature",
        role=AxisRole.COORDINATE,
        quantity_kind=QuantityKind.TEMPERATURE,
        label_raw="temperature",
        label_ref=char_span("temperature", QuoteRole.LABEL),
    )
    coordinate = Coordinate(
        axis_id="temperature", value=value, uncertainty=Absent(reason=AbsenceReason.NOT_REPORTED_HERE)
    )
    point = DataPoint(
        point_id="pt1",
        coordinates=(coordinate,),
        observations=(),
        composition=Absent(reason=AbsenceReason.SAME_AS_DATASET),
    )
    series = Series.model_construct(
        series_id="s1",
        source_form=source_form,
        value_origin=ValueOrigin.EXPERIMENTAL,
        axes=(axis,),
        constants=(),
        points=(point,),
        digitization_sha256=Absent(reason=AbsenceReason.NOT_APPLICABLE),
    )
    return DatasetEnvelope.model_construct(
        source_graph=grounding.graph,
        composition=Absent(reason=AbsenceReason.NOT_APPLICABLE),
        series=(series,),
        conversion_tables=(_ACTIVE.embedded,),
        table_inventories=(),
        figure_digitizations=(),
    )


class TestASpoofedSourceFormCannotLaunderVerified:
    """Verifier 1 and 2: the ticket's headline. A series built through ``model_construct`` carrying
    the plain string ``"digitized"`` -- which validation's ``==`` treats as digitized -- must not be
    waved past the figure lane by an ``is`` identity miss, laundering a VERIFIED verdict onto a lie
    about where its numbers came from.
    """

    def test_the_string_spoof_cannot_reach_verified_while_its_honest_control_does(self, tmp_path: Path) -> None:
        """The SAME char-span-grounded series, differing ONLY in ``source_form``: honestly TEXTUAL it
        verifies (its numbers re-slice, nothing is left unchecked); spoofed to the string
        ``"digitized"`` it must NOT verify -- the figure lane now recognizes it and files the
        attestation the honest text series never owed. Asserting the split, not merely 'not
        VERIFIED', is what pins the fix: a test that only saw one side would stay green on the bug.
        """
        honest = _char_span_grounded_series_envelope(tmp_path, SourceForm.TEXTUAL)
        honest_report = replay_envelope(tmp_path, honest)
        # The control proves the envelope's numbers genuinely verify -- so the ONLY thing that can
        # sink the spoof below is its source_form, not some unrelated blemish.
        assert honest_report.evidence_outcome is ReplayOutcome.VERIFIED
        assert honest_report.overall_outcome is ReplayOutcome.VERIFIED
        assert honest_report.checked_char_spans >= 1

        spoof = _char_span_grounded_series_envelope(tmp_path, "digitized")
        spoof_report = replay_envelope(tmp_path, spoof)

        # A genuinely successful check still ran (the same 3 char spans the control re-sliced) ...
        assert spoof_report.checked_char_spans == honest_report.checked_char_spans
        # ... yet the verdict is no longer VERIFIED: failing closed. (Verifier 2: a finding/claim,
        # never a traceback -- reaching this assertion at all proves no exception escaped.)
        assert spoof_report.overall_outcome is not ReplayOutcome.VERIFIED
        # And the reason is the figure lane: the attestation claim the honest text series never owed
        # is now filed for the spoofed 'digitized' series. This assertion is ABSENT-on-the-bug: the
        # unfixed selector skips the series and files no such claim.
        measurement = [c for c in spoof_report.unchecked_semantic_claims if c.claim_path == "series[0]"]
        assert len(measurement) == 1, [c.claim_path for c in spoof_report.unchecked_semantic_claims]
        assert "operator attestation replay structurally cannot test" in measurement[0].reason


class TestADanglingCitationOnANonDigitizedSeriesCannotLaunderVerified:
    """I-036: the lie that points the OTHER way. The sibling fix
    (:class:`TestASpoofedSourceFormCannotLaunderVerified`) made replay catch a series lying TOWARD
    ``DIGITIZED``. A series lying AWAY from it -- keeping a non-digitized ``source_form`` while still
    carrying a figure citation -- was examined by no figure check, so it laundered a VERIFIED verdict
    for numbers that name the plotted curve they came from. Replay now re-derives BOTH parts of V9's
    non-DIGITIZED branch (``_validate_series_digitization_citation``).
    """

    def _non_digitized_series_with_citation(self, tmp_path: Path, citation: object) -> DatasetEnvelope:
        """The honest char-span-grounded TEXTUAL envelope, its one series reconstructed to carry
        ``citation`` in ``digitization_sha256`` -- the ONLY change, so any verdict shift is that
        citation alone. Rebuilt through ``model_construct`` because V9 refuses this at construction.
        """
        env = _char_span_grounded_series_envelope(tmp_path, SourceForm.TEXTUAL)
        series = _reconstruct(env.series[0], digitization_sha256=citation)
        return _reconstruct(env, series=(series,))

    def test_a_dangling_figure_citation_cannot_reach_verified(self, tmp_path: Path) -> None:
        """Verifier 1. A TEXTUAL series whose numbers genuinely re-slice, given a citation that
        resolves to no record whatsoever, must NO LONGER read VERIFIED, and a finding must name the
        offending series. The honest control (Verifier 3) proves the numbers themselves verify, so
        the ONLY thing sinking this envelope is the forbidden citation."""
        honest = self._non_digitized_series_with_citation(tmp_path, Absent(reason=AbsenceReason.NOT_APPLICABLE))
        honest_report = replay_envelope(tmp_path, honest)
        assert honest_report.overall_outcome is ReplayOutcome.VERIFIED
        assert honest_report.checked_char_spans >= 1

        dangling = self._non_digitized_series_with_citation(tmp_path, "a" * 64)
        report = replay_envelope(tmp_path, dangling)

        # The same numbers still re-sliced -- the sink is not a broken value.
        assert report.checked_char_spans == honest_report.checked_char_spans
        # ... yet the verdict falls: failing closed on the forbidden citation.
        assert report.overall_outcome is not ReplayOutcome.VERIFIED
        finding = _assert_one(
            report, path="series[0]", category=ReplayOutcome.FAILED, marker="only a DIGITIZED series may cite one"
        )
        # The finding NAMES the offending series (Verifier 1), by id, not merely by index.
        assert "'s1'" in finding.reason

    def test_an_absence_reason_other_than_not_applicable_is_caught(self, tmp_path: Path) -> None:
        """Verifier 2, asserted separately: a non-digitized series whose ``digitization_sha256`` is
        Absent for a reason other than NOT_APPLICABLE (here NOT_REPORTED_HERE -- a gap the form does
        not have) is caught with its own FAILED finding, distinct from the dangling-citation half."""
        env = self._non_digitized_series_with_citation(tmp_path, Absent(reason=AbsenceReason.NOT_REPORTED_HERE))

        report = replay_envelope(tmp_path, env)

        assert report.overall_outcome is not ReplayOutcome.VERIFIED
        finding = _assert_one(report, path="series[0]", category=ReplayOutcome.FAILED, marker="the only true absence")
        assert "'not_reported_here'" in finding.reason
        assert "'s1'" in finding.reason

    def test_the_honest_non_digitized_control_verifies_untouched(self, tmp_path: Path) -> None:
        """Verifier 3, asserted positively: the honest non-digitized series -- properly Absent
        (NOT_APPLICABLE) citation -- replays exactly as before, VERIFIED with NO figure finding. This
        is what proves the new check refuses the forbidden case, not everything."""
        env = _char_span_grounded_series_envelope(tmp_path, SourceForm.TEXTUAL)

        report = replay_envelope(tmp_path, env)

        assert _figure_findings(report) == [], [f.reason for f in _figure_findings(report)]
        assert report.overall_outcome is ReplayOutcome.VERIFIED


class TestFigureClaimAddressesAndGaps:
    """Verifier 3, 4, 5: the support paths resolve to a coordinate, a None canonical_json degrades
    to a finding, and the two gaps are separable by machine."""

    def test_support_paths_resolve_to_a_specific_coordinate_not_a_crop_node(self) -> None:
        """Verifier 3. The measurement claim's ``support_paths`` must be positional refs a consumer
        follows back to the exact coordinate/observation -- never the crop NODE id (which cannot say
        WHICH value it supported) and never the ``where`` prose. Asserted on content."""
        env = _valid_envelope()

        report = replay_envelope(Path("/nonexistent-store"), env)

        (measurement,) = [c for c in report.unchecked_semantic_claims if c.claim_path == "series[0]"]
        assert measurement.gap is SemanticGap.LOCATION_UNRESOLVED
        # The one point of _valid_envelope grounds a coordinate and an observation; both address
        # back to their exact value_ref, positionally.
        assert measurement.support_paths == (
            "series[0].points[0].coordinates[0].value.value_ref",
            "series[0].points[0].observations[0].value.value_ref",
        )
        # Never the old crop-node shape, and never a human sentence.
        assert not any(path.startswith("node ") for path in measurement.support_paths)
        assert all(
            path.startswith("series[0].points[") and path.endswith(".value_ref") for path in measurement.support_paths
        )

    def test_canonical_json_of_none_is_a_finding_not_a_traceback(self) -> None:
        """Verifier 4. A ``canonical_json`` of None raises TypeError from json.loads; the guard must
        degrade it to an UNVERIFIABLE finding rather than let the traceback escape. Built through
        ``model_construct`` so the None reaches replay exactly as an unvalidated envelope carries it.
        """
        env = _valid_envelope()
        good = env.figure_digitizations[0]
        broken = EmbeddedFigureDigitization.model_construct(
            digitization_sha256=good.digitization_sha256, raw_sha256=good.raw_sha256, canonical_json=None
        )
        env2 = _reconstruct(env, figure_digitizations=(broken,))

        # Reaching a report at all (no exception) is half the assertion.
        report = replay_envelope(Path("/nonexistent-store"), env2)

        _assert_one(
            report,
            path=f"figure_digitizations[{good.digitization_sha256!r}]",
            category=ReplayOutcome.UNVERIFIABLE,
            marker="does not parse",
        )

    def test_the_provenance_and_measurement_gaps_are_separately_machine_visible(self) -> None:
        """Verifier 5. The crop-provenance gap and the coordinate-measurement gap are two DISTINCT
        unverifiable facts, filed as two claims a consumer partitions by ``claim_path`` -- a
        structured coordinate read programmatically, never a phrase grepped out of ``reason``."""
        env = _valid_envelope()

        report = replay_envelope(Path("/nonexistent-store"), env)

        by_path = {c.claim_path: c for c in report.unchecked_semantic_claims}
        # Measurement: the recovered coordinates, addressed at the series.
        assert "series[0]" in by_path
        # Provenance: the crop's region within its parent, addressed at the citation field.
        assert "series[0].digitization_sha256" in by_path
        measurement = by_path["series[0]"]
        provenance = by_path["series[0].digitization_sha256"]
        # They are genuinely different claims about different things, told apart with zero prose.
        assert measurement.claim_path != provenance.claim_path
        assert "markers from the crop's pixels" in measurement.reason
        assert "crop's OWN region within its parent" in provenance.reason
        # The provenance claim's support names the embedded record carrying the crop identity.
        cited = env.series[0].digitization_sha256
        assert provenance.support_paths == (f"figure_digitizations[{cited!r}]",)

    def test_the_claim_names_a_derived_value_not_the_content_address(self) -> None:
        """Verifier 4. ``UncheckedSemanticClaim.claim`` must hold the DERIVED value whose support
        went unchecked -- here the series' recorded DIGITIZED ``source_form`` -- never the
        ``digitization_sha256`` content address, which is not a derived value and (per the record
        module) does not even identify the recovered coordinates. Asserted on content, and on the
        field's hard constraint: it passes through no redaction gate, so it must carry no slice of
        source text."""
        env = _valid_envelope()

        report = replay_envelope(Path("/nonexistent-store"), env)

        by_path = {c.claim_path: c for c in report.unchecked_semantic_claims}
        measurement = by_path["series[0]"]
        provenance = by_path["series[0].digitization_sha256"]
        # Both claims name the recovered value's DERIVED classification: the series is DIGITIZED.
        assert measurement.claim == "digitized"
        assert provenance.claim == "digitized"
        # And NOT the content address a first cut used -- the digest names something else entirely.
        cited = env.series[0].digitization_sha256
        assert isinstance(cited, str)
        assert measurement.claim != cited
        assert cited not in measurement.claim
        # The redaction-gate constraint, made concrete: no slice of the stored paper text (the one
        # recorded string this envelope carries, _TEXT) may appear in the ungated claim field.
        assert measurement.claim not in _TEXT
        assert _TEXT[:16] not in measurement.claim


class TestAnUnrecognisedSourceFormProducesAFinding:
    """Verifier 5. A ``model_construct`` series whose ``source_form`` is neither a
    :class:`SourceForm` member nor a recognised member string is one replay cannot classify: the
    figure joins need it DIGITIZED, the non-digitized citation rule needs it some real non-figure
    form, and it is neither. Before this fix it took the non-digitized path and, citing nothing,
    produced NO finding on any axis -- an unclassifiable series waved through in silence. It must
    now produce a finding saying replay cannot tell what it claims to be."""

    def test_an_unclassifiable_source_form_yields_an_unverifiable_finding(self, tmp_path: Path) -> None:
        """The honest control (a real non-figure form, properly Absent citation) verifies with no
        figure finding; the unrecognised form -- the ONLY change -- yields an UNVERIFIABLE finding
        naming the series. Asserting the split is what pins the fix: on the bug the unrecognised
        form produced nothing, so a one-sided test would have stayed green."""
        honest = _char_span_grounded_series_envelope(tmp_path, SourceForm.TEXTUAL)
        honest_report = replay_envelope(tmp_path, honest)
        assert honest_report.overall_outcome is ReplayOutcome.VERIFIED
        assert _figure_findings(honest_report) == [], [f.reason for f in _figure_findings(honest_report)]

        unclassifiable = _char_span_grounded_series_envelope(tmp_path, "marginalia")
        report = replay_envelope(tmp_path, unclassifiable)

        # ABSENT-on-the-bug: the unfixed lane files no figure finding for this series at all.
        finding = _assert_one(
            report, path="series[0]", category=ReplayOutcome.UNVERIFIABLE, marker="no recognised source form"
        )
        # It NAMES the series and the offending form, and it is the reader's inability, never a
        # charge -- so the report is no longer VERIFIED, but nothing is DEMONSTRATED wrong.
        assert "'s1'" in finding.reason
        assert "'marginalia'" in finding.reason
        assert report.overall_outcome is not ReplayOutcome.VERIFIED
        assert not any(f.category is ReplayOutcome.FAILED for f in _figure_findings(report))
