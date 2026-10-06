# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Numeric policy for whole structured-record fields."""

from __future__ import annotations

import re

from carmel.services.numeric import (
    GlyphHealth,
    NormalizedNumeral,
    SourceContext,
    Unresolvable,
    normalize_numeric_span,
)

__all__ = ["is_structured_trailing_dot_numeral", "normalize_structured_numeric_span"]


_TRAILING_DOT_RE = re.compile(r"(?P<lead_sign>[-+]?)(?P<digits>\d+)\.(?P<exponent>[eE][-+]?\d+)?")


def _structured_trailing_dot_match(span: str) -> re.Match[str] | None:
    return _TRAILING_DOT_RE.fullmatch(span.strip())


def is_structured_trailing_dot_numeral(span: str) -> bool:
    """Return whether ``span`` selects this module's structured-only extension."""
    return _structured_trailing_dot_match(span) is not None


def normalize_structured_numeric_span(
    span: str,
    *,
    source_context: SourceContext,
    glyph_health: GlyphHealth,
) -> NormalizedNumeral | Unresolvable:
    """Normalize a whole YAML/XML value, admitting an integer mantissa ending in ``.``.

    Structured fields delimit the value, so the final point cannot be sentence
    punctuation. This opt-in policy deliberately leaves the shared PDF/prose
    grammar unchanged. Every other shape is delegated to that grammar verbatim.

    Args:
        span: The complete scalar text from a structured source field.
        source_context: The source context passed through to the shared grammar.
        glyph_health: The glyph-health assessment passed through to the shared grammar.

    Returns:
        The normalized numeral, preserving ``span`` as its raw text, or a typed
        refusal from the shared grammar.
    """
    match = _structured_trailing_dot_match(span)
    if match is None:
        return normalize_numeric_span(span, source_context=source_context, glyph_health=glyph_health)
    normalized = normalize_numeric_span(
        f"{match.group('lead_sign')}{match.group('digits')}{match.group('exponent') or ''}",
        source_context=source_context,
        glyph_health=glyph_health,
    )
    if isinstance(normalized, Unresolvable):
        return Unresolvable(raw=span, reason=normalized.reason)
    return NormalizedNumeral(raw=span, text=normalized.text, repairs=normalized.repairs)
