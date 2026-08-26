"""The cited text is not the repaired text (ticket I-019).

The table lane runs every PDF glyph through a repair registry
(``carmel.services.pdf_fragments._GLYPH_VERDICTS``); the running-text lane
(``carmel.agents.tools.extract`` -> pypdf ``extract_text``) never traverses it, so a
char-span citation grounded into the stored text can resolve against a character the
codebase has already concluded is mis-decoded. This module pins the closure of that
divergence:

* the SEAM -- ``_prepare_grounding`` now carries the document's in-force glyph repairs
  (``registry_glyph_repairs_for_document``), and ``ground_quote`` refuses a citation into a
  character the text lane stores mis-decoded, naming the mis-decode
  (``TextLaneMisdecodeError``);
* the DECISION -- refuse, not carry-repairs-into-text: the text lane has discarded the
  per-glyph font-program identity the repairs are keyed on (a genuine U+2212 minus and a
  genuine ``φ`` occur in the same document as the impostor glyphs), so the refusal fires ONLY
  on the not-found path and never rejects a character that is genuinely present;
* the REGRESSION guard -- a structural check that fails if a future change reintroduces a
  text lane that bypasses the repair path (the ticket's grep-count-zero fact, inverted).

The end-to-end proof on the real document -- that the stored condition-set artifact still
replays, and that ``"°C"`` in running text now refuses with the degree mis-decode named --
lives in :mod:`tests.test_condition_set_target_acceptance`, which owns the corpus gate.
"""

from __future__ import annotations

import inspect

import pytest

from carmel.services import dataset_producer
from carmel.services.dataset_producer import (
    QuoteGroundingError,
    TextLaneMisdecodeError,
    _prepare_grounding,
    _text_lane_misdecode,
    ground_quote,
)
from carmel.services.numeric import QuoteRole
from carmel.services.pdf_fragments import (
    GlyphRefusal,
    GlyphRepair,
    GlyphVerdictScope,
    registry_glyph_repairs_for_document,
)

_TARGET_SHA = "9c59f1c6924f73d3c8f190b3e14b93cb889d1f6c6fb867e51d900a0f4b2cf84b"

#: A synthetic DOCUMENT-scoped degree repair, shaped exactly like the target document's
#: ``/C14`` -> ``°`` entry, so these unit tests exercise the seam without the corpus.
_DEGREE_REPAIR = GlyphRepair(
    scope=GlyphVerdictScope.DOCUMENT,
    document_sha256="deadbeef",
    font_program_sha256="f" * 64,
    font_base_name="TESTFN+SymbolSubset",
    glyph_name="/C14",
    replacement="°",
    evidence="synthetic fixture for the seam test; not a corpus claim",
)


class TestTheRegistryQueryIsTheSeamSource:
    """``registry_glyph_repairs_for_document`` is the fragment lane exposing what it repairs."""

    def test_it_returns_the_document_scoped_repairs_for_the_target(self) -> None:
        repairs = registry_glyph_repairs_for_document(_TARGET_SHA)
        # The three DOCUMENT-scoped repairs the target's table lane applies: degree, en-dash,
        # exponent minus. Named by their replacement so a drift in the registry is visible here.
        assert {r.replacement for r in repairs} == {"°", "–", "−"}
        assert all(isinstance(r, GlyphRepair) for r in repairs)
        assert all(r.scope is GlyphVerdictScope.DOCUMENT for r in repairs)
        assert all(r.document_sha256 == _TARGET_SHA for r in repairs)

    def test_it_is_empty_for_an_unregistered_document(self) -> None:
        assert registry_glyph_repairs_for_document("0" * 64) == ()

    def test_it_excludes_refusals_and_font_program_scope(self) -> None:
        """The two deliberate exclusions (see the function's docstring): a GlyphRefusal leaves
        the impostor's wrong Latin decode in place -- which collides with the same character
        legitimately elsewhere -- and a FONT_PROGRAM verdict's in-force-ness needs the embedded
        program present, which a sha-only query cannot confirm. Neither may reach this seam."""
        for repair in registry_glyph_repairs_for_document(_TARGET_SHA):
            assert not isinstance(repair, GlyphRefusal)
            assert repair.scope is GlyphVerdictScope.DOCUMENT


class TestGroundQuoteRefusesAMisdecodeCitation:
    """The refusal is precise: it fires ONLY on the not-found path, and names the mis-decode."""

    def test_it_refuses_a_not_found_quote_that_cites_a_repaired_character(self) -> None:
        with pytest.raises(TextLaneMisdecodeError) as excinfo:
            ground_quote(
                "the header reads T (C) with no ring",
                "T (°C)",
                role=QuoteRole.LABEL,
                repairs=(_DEGREE_REPAIR,),
            )
        message = str(excinfo.value)
        # Verifier 2: the reason names the SPECIFIC mis-decode -- the glyph, the repaired
        # character, and the table lane -- not merely that something was not found. A test that
        # would still pass on a different refusal reason has pinned nothing.
        assert "°" in message
        assert "/C14" in message
        assert "table" in message.lower()
        assert excinfo.value.repair is _DEGREE_REPAIR

    def test_the_refusal_is_a_quote_grounding_error_subclass(self) -> None:
        """So every existing ``except QuoteGroundingError`` / ``pytest.raises`` still catches it."""
        assert issubclass(TextLaneMisdecodeError, QuoteGroundingError)

    def test_a_present_character_is_never_refused_even_when_it_is_a_replacement(self) -> None:
        """The decision's crux: a genuine U+2212 minus (which really occurs in the target
        document, 10 times, correctly decoded) must ground normally. The refusal only ever
        UPGRADES a not-found miss; it never rejects a character that is present."""
        minus_repair = GlyphRepair(
            scope=GlyphVerdictScope.DOCUMENT,
            document_sha256="deadbeef",
            font_program_sha256="a" * 64,
            font_base_name="TESTFN+SymbolSubset",
            glyph_name="L",
            replacement="−",
            evidence="synthetic fixture; not a corpus claim",
        )
        located = ground_quote(
            "the exponent is n = −1 in this row",
            "−1",
            role=QuoteRole.VALUE,
            repairs=(minus_repair,),
        )
        assert located.start >= 0  # grounded, not refused

    def test_without_repairs_the_miss_stays_a_plain_not_found(self) -> None:
        with pytest.raises(QuoteGroundingError) as excinfo:
            ground_quote("no ring here", "T (°C)", role=QuoteRole.LABEL)
        assert not isinstance(excinfo.value, TextLaneMisdecodeError)
        assert "was not found" in str(excinfo.value)

    def test_the_helper_returns_none_when_the_quote_cites_no_repaired_character(self) -> None:
        assert _text_lane_misdecode("heat flux method", (_DEGREE_REPAIR,)) is None


class TestTheLigatureIsOutOfScope:
    """Defect #2 (ligatures) is deliberately NOT a refusal.

    A ligature (``ﬂ`` = U+FB02) is a valid Unicode character the document legitimately emitted,
    not a registry mis-decode -- so it is absent from the repair registry, and refusing it would
    require widening that registry (a non-goal). The stored artifact cites the ligature form
    verbatim; a refusal would break it. Handling the ligature belongs to
    ``carmel.agents.tools.normalize_for_match``, not this seam.
    """

    def test_no_registry_repair_targets_a_ligature(self) -> None:
        for repair in registry_glyph_repairs_for_document(_TARGET_SHA):
            assert "ﬂ" not in repair.replacement
            assert repair.glyph_name != "ﬂ"

    def test_a_ligature_quote_is_not_refused_by_the_seam(self) -> None:
        # The seam neither locates nor refuses on the ligature: it grounds exactly as before.
        located = ground_quote(
            "measured with the heat ﬂux method",
            "heat ﬂux method",
            role=QuoteRole.LABEL,
            repairs=registry_glyph_repairs_for_document(_TARGET_SHA),
        )
        assert located.start >= 0


class TestTheLaneSeparationIsClosedAtANamedSeam:
    """Verifier 3: guard the grep-count-zero shape so a future change cannot re-open a text
    lane that bypasses the repair path. These are STRUCTURAL checks on the seam's source, so
    they run with no corpus and fail loudly if the wiring is removed."""

    def test_prepare_grounding_consults_the_repair_registry(self) -> None:
        source = inspect.getsource(_prepare_grounding)
        assert "registry_glyph_repairs_for_document" in source, (
            "the grounding preamble no longer consults the fragment lane's repair registry -- "
            "the text lane has been re-separated from the repair path"
        )

    def test_the_grounding_context_carries_the_repairs(self) -> None:
        assert "glyph_repairs" in dataset_producer._GroundingContext.__dataclass_fields__

    def test_ground_quote_consumes_the_repairs(self) -> None:
        assert "repairs" in inspect.signature(ground_quote).parameters
        assert "_text_lane_misdecode" in inspect.getsource(ground_quote)

    def test_the_module_imports_the_seam_function(self) -> None:
        source = inspect.getsource(dataset_producer)
        assert "from carmel.services.pdf_fragments import" in source
        assert "registry_glyph_repairs_for_document" in source
