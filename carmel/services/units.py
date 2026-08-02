# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Unit normalization and conversion for literature-extracted kinetics datasets.

Carmel's dataset schema (:mod:`carmel.schemas.datasets`) records a measured
value's ``unit_raw`` (whatever spelling the source paper used) alongside a
``unit_canonical``, ``conversion_factor``, and ``conversion_table_version`` --
but, as of this module, no conversion table actually exists to populate those
fields. This module builds that table: a small, closed-world, versioned set of
unit-conversion rules covering exactly the syngas/laminar-burning-velocity
corpus this milestone targets, plus the machinery to look a rule up, apply it
exactly, and round the result to a *documented*, defensible precision.

Three design choices carry the same cardinal rule the rest of Carmel's dataset
machinery serves -- every load-bearing number must be auditable, never
fabricated -- into this module specifically:

1. **A conversion rule is a closed three-member union
   (:class:`IdentityRule` / :class:`ScaleRule` / :class:`AffineRule`), and
   ``IdentityRule`` carries no numeric parameters at all.** An identity
   conversion is structurally incapable of emitting a fake ``scale="1"`` for a
   conversion that never actually happened; "this unit needed no conversion"
   and "this unit was scaled by exactly 1" are different facts, and only the
   union's shape -- not a convention about which fields happen to be set --
   keeps them distinguishable. See :data:`ConversionRule`.

2. **A conversion table is versioned and content-addressed
   (:attr:`ConversionTable.sha256`), and a shipped table is NEVER mutated.**
   :data:`TABLE_V1` is the first (and, as of this module, only) table. If its
   rules ever need to change -- a new alias spelling discovered in a later
   paper, a corrected conversion factor -- that is not an edit to this file;
   it is a new ``TABLE_V2`` added alongside it, so that any dataset already
   recorded against ``TABLE_V1``'s sha256 stays reproducible forever, byte for
   byte, against the exact rules that produced it. This is the whole reason a
   sha256 is threaded through here at all rather than a bare version string
   like ``"v1"``: a string can be reused by mistake or by a rebase; a sha256
   of the table's own canonical content cannot silently drift out from under a
   dataset that already cites it.

3. **Exact arithmetic and rounding are two separate, explicit steps, and the
   rounding policy differs by rule kind for a physical reason, not a stylistic
   one.** See :func:`convert` and :class:`Converted` for the full rationale;
   in short, ``Decimal`` arithmetic preserves *value*, not the *measurement
   precision* the source paper actually reported, and a scale conversion
   (which preserves relative precision) and an affine conversion (which
   preserves absolute precision) are rounded differently for that reason.

Deliberately NOT built here: this module does not import from
``carmel.schemas`` (schemas import services; the reverse would be circular),
and it does not rewire the dataset schema's ``unit_raw``/``unit_canonical``
fields to actually call this module -- that wiring is a later milestone.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, Inexact, InvalidOperation, Rounded, localcontext
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

from carmel.services.dataset_store import CanonicalDecimalError, canonical_decimal, canonical_json_bytes

__all__ = [
    "TABLES_BY_SHA",
    "TABLE_V1",
    "AffineRule",
    "ConversionRule",
    "ConversionTable",
    "ConversionTableInvariantError",
    "Converted",
    "IdentityRule",
    "QuantityKind",
    "ScaleRule",
    "UnitAlias",
    "UnitError",
    "UnknownConversionTableError",
    "UnknownQuantityUnitPairError",
    "UnknownUnitError",
    "convert",
    "normalize_unit",
    "table_for_sha",
]


class QuantityKind(StrEnum):
    """The physical (or dimensionless-bookkeeping) quantities this table models.

    ``OTHER`` is an honest absence state, not a wildcard: it means "a quantity
    this table deliberately does not model" (e.g. a field this milestone's
    8-paper corpus never actually reports in a non-base unit). It permits
    *only* identity conversion -- see :func:`normalize_unit` and
    :func:`convert` -- so that a value tagged ``OTHER`` can still flow through
    the same code paths as a modeled quantity without this table ever having
    to fabricate a conversion rule for something it does not understand.
    """

    LENGTH = "length"
    VELOCITY = "velocity"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    TIME = "time"
    VOLUME = "volume"
    STRAIN_RATE = "strain_rate"
    MOLE_FRACTION = "mole_fraction"
    MASS_FRACTION = "mass_fraction"
    EQUIVALENCE_RATIO = "equivalence_ratio"
    RELATIVE_UNCERTAINTY = "relative_uncertainty"
    OTHER = "other"


class UnitError(ValueError):
    """Base class for every error this module raises.

    A subclass of ``ValueError`` (matching :mod:`carmel.services.dataset_store`'s
    own convention) so callers can catch unit errors specifically while still
    satisfying code that catches ``ValueError`` broadly.
    """


class UnknownUnitError(UnitError):
    """Raised when a raw unit spelling is neither a known unit nor a known alias."""


class UnknownQuantityUnitPairError(UnitError):
    """Raised when no conversion rule exists for a requested (quantity, unit) pair."""


class UnknownConversionTableError(UnitError):
    """Raised when a sha256 does not name any table this module ships."""


class ConversionTableInvariantError(UnitError):
    """Raised when a :class:`ConversionTable` violates one of its own invariants.

    Every invariant this exception can report is checked in
    :meth:`ConversionTable.__post_init__`, at construction time -- including
    for :data:`TABLE_V1`, which is constructed at import time. A table that
    violates its own invariants is never allowed to exist as a live object at
    all, let alone be looked up or used for a conversion.
    """


@dataclass(frozen=True, slots=True)
class IdentityRule:
    """A conversion rule stating that ``unit`` needs no conversion at all.

    Carries no numeric parameters -- deliberately. Adding a ``scale`` field
    here "for uniformity" with :class:`ScaleRule` would let this table
    represent "no conversion happened" and "conversion by exactly 1" as the
    same shape, which is exactly the distinction this rule type exists to
    keep structurally impossible to blur: a dataset entry that never needed
    converting must never be able to silently grow a fabricated ``"1"``
    conversion factor.
    """

    kind: Literal["identity"]
    quantity: QuantityKind
    unit: str


@dataclass(frozen=True, slots=True)
class ScaleRule:
    """A pure multiplicative conversion rule: ``to = from * scale``.

    ``scale`` is a canonical decimal string (see
    :func:`~carmel.services.dataset_store.canonical_decimal`), never a float,
    for the same reason every numeric fact in this project is a decimal
    string: float ``repr`` is unstable across platforms/interpreter versions,
    which would silently change this table's content address.
    """

    kind: Literal["scale"]
    quantity: QuantityKind
    from_unit: str
    to_unit: str
    scale: str


@dataclass(frozen=True, slots=True)
class AffineRule:
    """An affine conversion rule: ``to = from * scale + offset``, in exactly that form.

    The order of operations (multiply first, then add) is fixed and
    documented here because it is not commutative with rounding: this
    module's rounding policy (see :func:`convert`) depends on which operation
    ran, and "scale-then-offset" is the only order that makes the
    "affine rules preserve absolute precision" rounding rule below correct
    for a source coefficient reported to N significant figures.
    """

    kind: Literal["affine"]
    quantity: QuantityKind
    from_unit: str
    to_unit: str
    scale: str
    offset: str


ConversionRule = IdentityRule | ScaleRule | AffineRule
"""A three-member discriminated union of every conversion rule shape this table supports.

Discriminated on the ``kind`` field (``"identity"`` / ``"scale"`` /
``"affine"``) rather than on ``isinstance`` alone, so a rule's canonical JSON
projection (see :meth:`ConversionTable.identity_payload`) can be driven by the
same literal string a caller would use to branch on ``rule_kind`` in
:class:`Converted`.
"""


@dataclass(frozen=True, slots=True)
class UnitAlias:
    """A pure SPELLING normalization: the same physical unit, a different raw spelling.

    An alias is never itself a conversion -- ``normalized`` must already be a
    known unit of ``quantity`` in the same table (see invariant 6 in
    :meth:`ConversionTable.__post_init__`), so resolving an alias never
    changes a value's magnitude, only which unit string names it. Case is
    NEVER folded as part of alias resolution (see :func:`normalize_unit`):
    ``K`` (kelvin) and ``k`` (a kilo- prefix fragment, not a unit this table
    models on its own) are genuinely different tokens in real papers, and
    silently folding one into the other would be exactly the kind of silent
    reinterpretation of raw evidence this project exists to prevent.
    """

    quantity: QuantityKind
    raw: str
    normalized: str


def _rule_key(rule: ConversionRule) -> tuple[QuantityKind, str]:
    """Return the ``(quantity, from-unit-like-field)`` key identifying ``rule``'s source unit.

    An :class:`IdentityRule` is keyed on its own ``unit`` (there is no
    separate ``from_unit``/``to_unit`` pair to choose between); a
    :class:`ScaleRule`/:class:`AffineRule` is keyed on ``from_unit``. This is
    the single place that mapping lives, so :meth:`ConversionTable.__post_init__`'s
    duplicate-key check and :func:`convert`'s rule lookup can never disagree
    about what "the same rule key" means.
    """
    if isinstance(rule, IdentityRule):
        return (rule.quantity, rule.unit)
    return (rule.quantity, rule.from_unit)


def _rule_identity_payload(rule: ConversionRule) -> dict[str, Any]:
    """Project ``rule`` to its canonical-JSON dict form.

    Each rule kind projects ONLY the parameters it actually has -- an
    :class:`IdentityRule`'s projection carries no ``scale``/``offset`` key at
    all, matching that class's "no numeric parameters" contract all the way
    through to the bytes this table's sha256 is computed from.
    """
    if isinstance(rule, IdentityRule):
        return {"kind": "identity", "quantity": rule.quantity.value, "unit": rule.unit}
    if isinstance(rule, ScaleRule):
        return {
            "kind": "scale",
            "quantity": rule.quantity.value,
            "from_unit": rule.from_unit,
            "to_unit": rule.to_unit,
            "scale": rule.scale,
        }
    return {
        "kind": "affine",
        "quantity": rule.quantity.value,
        "from_unit": rule.from_unit,
        "to_unit": rule.to_unit,
        "scale": rule.scale,
        "offset": rule.offset,
    }


@dataclass(frozen=True, slots=True)
class ConversionTable:
    """A closed, versioned, content-addressed set of unit-conversion rules.

    Every invariant below is checked in ``__post_init__`` -- at construction
    time, which for :data:`TABLE_V1` means at import time -- so a
    ``ConversionTable`` instance that violates one of them can never exist to
    be looked up or converted against in the first place:

    1. Every :class:`QuantityKind` except ``OTHER`` has exactly one entry in
       ``base_units``; ``OTHER`` has none (it is the deliberate absence state
       this table does not model numerically at all).
    2. Every :class:`ScaleRule`/:class:`AffineRule`'s ``to_unit`` equals its
       quantity's base unit -- there is one canonical conversion target per
       quantity, and this table never needs to traverse a graph of rules to
       find it.
    3. ``from_unit != to_unit`` on every scale/affine rule; an
       :class:`IdentityRule` exists only for a quantity's own base unit.
    4. No duplicate ``(quantity, from_unit)`` rule keys (see :func:`_rule_key`);
       no duplicate ``(quantity, raw)`` alias keys.
    5. Every ``scale``/``offset`` is a canonical decimal string (round-trips
       through :func:`~carmel.services.dataset_store.canonical_decimal`
       unchanged); ``scale`` is strictly positive and finite.
    6. An alias's ``normalized`` spelling must already be a known unit of its
       quantity; its ``raw`` spelling must NOT already be a known unit of that
       quantity (an alias whose raw form is itself a real unit would be
       ambiguous -- is it being renamed, or is the renaming a mistake?); and
       ``raw != normalized`` (an alias that does nothing is not an alias).
    7. No rules and no aliases are registered for ``OTHER`` -- the whole
       point of ``OTHER`` is that this table has nothing to say about it.
    """

    table_id: str
    version: int
    base_units: tuple[tuple[QuantityKind, str], ...]
    aliases: tuple[UnitAlias, ...]
    rules: tuple[ConversionRule, ...]

    def __post_init__(self) -> None:
        base_quantities = [quantity for quantity, _ in self.base_units]
        if len(base_quantities) != len(set(base_quantities)):
            raise ConversionTableInvariantError(
                f"table {self.table_id!r} v{self.version}: base_units contains a duplicate quantity entry "
                f"among {base_quantities!r}"
            )
        expected_quantities = set(QuantityKind) - {QuantityKind.OTHER}
        if set(base_quantities) != expected_quantities:
            missing = expected_quantities - set(base_quantities)
            unexpected = set(base_quantities) - expected_quantities
            raise ConversionTableInvariantError(
                f"table {self.table_id!r} v{self.version}: base_units must cover every QuantityKind except "
                f"OTHER exactly once; missing={sorted(q.value for q in missing)!r}, "
                f"unexpected={sorted(q.value for q in unexpected)!r}"
            )
        base_unit_by_quantity = dict(self.base_units)

        for rule in self.rules:
            if rule.quantity is QuantityKind.OTHER:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: rule {rule!r} is registered for "
                    "QuantityKind.OTHER, which this table must never carry rules for"
                )

        seen_rule_keys: set[tuple[QuantityKind, str]] = set()
        for rule in self.rules:
            key = _rule_key(rule)
            if key in seen_rule_keys:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: duplicate rule key {key!r} "
                    f"(quantity, from-unit-like field)"
                )
            seen_rule_keys.add(key)

            base_unit = base_unit_by_quantity[rule.quantity]
            if isinstance(rule, IdentityRule):
                if rule.unit != base_unit:
                    raise ConversionTableInvariantError(
                        f"table {self.table_id!r} v{self.version}: identity rule for "
                        f"{rule.quantity.value!r} names unit {rule.unit!r}, but that quantity's base "
                        f"unit is {base_unit!r}; identity rules exist only for a quantity's base unit"
                    )
                continue

            if rule.to_unit != base_unit:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: {rule.kind} rule "
                    f"{rule.from_unit!r} -> {rule.to_unit!r} for {rule.quantity.value!r} targets "
                    f"{rule.to_unit!r}, but that quantity's base unit is {base_unit!r}"
                )
            if rule.from_unit == rule.to_unit:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: {rule.kind} rule for "
                    f"{rule.quantity.value!r} has from_unit == to_unit == {rule.from_unit!r}; "
                    "use an IdentityRule instead"
                )
            _require_canonical_positive_decimal(
                rule.scale,
                what=f"{rule.kind} rule {rule.from_unit!r} -> {rule.to_unit!r} scale",
                table_id=self.table_id,
                version=self.version,
            )
            if isinstance(rule, AffineRule):
                _require_canonical_decimal(
                    rule.offset,
                    what=f"affine rule {rule.from_unit!r} -> {rule.to_unit!r} offset",
                    table_id=self.table_id,
                    version=self.version,
                )

        for alias in self.aliases:
            if alias.quantity is QuantityKind.OTHER:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: alias {alias!r} is registered for "
                    "QuantityKind.OTHER, which this table must never carry aliases for"
                )

        seen_alias_keys: set[tuple[QuantityKind, str]] = set()
        for alias in self.aliases:
            alias_key = (alias.quantity, alias.raw)
            if alias_key in seen_alias_keys:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: duplicate alias key {alias_key!r} (quantity, raw)"
                )
            seen_alias_keys.add(alias_key)

            if alias.raw == alias.normalized:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: alias {alias!r} has raw == normalized; "
                    "an alias that does not change spelling is not an alias"
                )
            known = self.known_units(alias.quantity)
            if alias.normalized not in known:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: alias {alias!r} normalizes to "
                    f"{alias.normalized!r}, which is not a known unit of {alias.quantity.value!r} "
                    f"({sorted(known)!r})"
                )
            if alias.raw in known:
                raise ConversionTableInvariantError(
                    f"table {self.table_id!r} v{self.version}: alias {alias!r} has raw spelling "
                    f"{alias.raw!r}, which is already a known unit of {alias.quantity.value!r} "
                    "-- an alias's raw form must not already be an unambiguous known unit"
                )

    def identity_payload(self) -> dict[str, Any]:
        """Return this table's canonical-JSON identity projection.

        Plain ``str``/``int``/``list``/``dict`` only -- passed straight to
        :func:`~carmel.services.dataset_store.canonical_json_bytes` by
        :attr:`sha256`, which rejects anything else (in particular, any stray
        ``float``).
        """
        return {
            "table_id": self.table_id,
            "version": self.version,
            "base_units": [[quantity.value, unit] for quantity, unit in self.base_units],
            "aliases": [
                {"quantity": alias.quantity.value, "raw": alias.raw, "normalized": alias.normalized}
                for alias in self.aliases
            ],
            "rules": [_rule_identity_payload(rule) for rule in self.rules],
        }

    @property
    def sha256(self) -> str:
        """The content address of this table: sha256 of its canonical identity payload.

        Two ``ConversionTable`` instances built independently but with equal
        field values always produce this same digest, because
        :func:`~carmel.services.dataset_store.canonical_json_bytes` is a
        deterministic function of content alone (sorted keys, no float, no
        platform-dependent rendering) -- never of construction order or
        object identity.
        """
        return hashlib.sha256(canonical_json_bytes(self.identity_payload())).hexdigest()

    def base_unit(self, quantity: QuantityKind) -> str:
        """Return ``quantity``'s single canonical base unit.

        Raises:
            UnitError: If ``quantity`` is ``QuantityKind.OTHER`` (which has no
                base unit by design) or, in principle, any quantity this table
                does not cover -- though invariant 1 above guarantees every
                non-``OTHER`` quantity has exactly one.
        """
        for candidate_quantity, unit in self.base_units:
            if candidate_quantity == quantity:
                return unit
        raise UnitError(
            f"table {self.table_id!r} v{self.version} has no base unit for {quantity.value!r} "
            "(QuantityKind.OTHER never has one, by design)"
        )

    def known_units(self, quantity: QuantityKind) -> frozenset[str]:
        """Return every unit spelling this table recognizes (pre-alias) for ``quantity``.

        ``QuantityKind.OTHER`` always returns the empty set: this table
        models nothing about ``OTHER`` at all, including having no notion of
        which spellings are "known" for it (see :func:`normalize_unit` for how
        ``OTHER`` is handled instead -- text is passed through unchanged
        rather than looked up here).
        """
        if quantity is QuantityKind.OTHER:
            return frozenset()
        units = {unit for candidate_quantity, unit in self.base_units if candidate_quantity == quantity}
        for rule in self.rules:
            if rule.quantity != quantity:
                continue
            if isinstance(rule, IdentityRule):
                units.add(rule.unit)
            else:
                units.add(rule.from_unit)
                units.add(rule.to_unit)
        return frozenset(units)


def _require_canonical_decimal(text: str, *, what: str, table_id: str, version: int) -> None:
    try:
        canonical = canonical_decimal(text)
    except CanonicalDecimalError as exc:
        raise ConversionTableInvariantError(
            f"table {table_id!r} v{version}: {what} {text!r} is not a valid canonical decimal string: {exc}"
        ) from exc
    if canonical != text:
        raise ConversionTableInvariantError(
            f"table {table_id!r} v{version}: {what} {text!r} is not already in canonical decimal form "
            f"(canonical form is {canonical!r})"
        )


def _require_canonical_positive_decimal(text: str, *, what: str, table_id: str, version: int) -> None:
    _require_canonical_decimal(text, what=what, table_id=table_id, version=version)
    if Decimal(text) <= 0:
        raise ConversionTableInvariantError(f"table {table_id!r} v{version}: {what} {text!r} must be strictly positive")


def _base_units_v1() -> tuple[tuple[QuantityKind, str], ...]:
    return (
        (QuantityKind.LENGTH, "m"),
        (QuantityKind.VELOCITY, "m/s"),
        (QuantityKind.TEMPERATURE, "K"),
        (QuantityKind.PRESSURE, "Pa"),
        (QuantityKind.TIME, "s"),
        (QuantityKind.VOLUME, "m3"),
        (QuantityKind.STRAIN_RATE, "1/s"),
        (QuantityKind.MOLE_FRACTION, "1"),
        (QuantityKind.MASS_FRACTION, "1"),
        (QuantityKind.EQUIVALENCE_RATIO, "1"),
        (QuantityKind.RELATIVE_UNCERTAINTY, "1"),
    )


def _identity_rules_v1() -> tuple[ConversionRule, ...]:
    # One identity rule per non-OTHER quantity's own base unit. This is a
    # design judgment call the spec for this table did not fully settle: it
    # is what lets convert() resolve a value already reported in a
    # quantity's base unit (e.g. quantity=PRESSURE, from_unit="Pa",
    # to_unit="Pa") without a special no-op path outside the rule table, and
    # it is the ONLY way a quantity with no scale/affine rules at all
    # (STRAIN_RATE, EQUIVALENCE_RATIO) can be converted through this
    # machinery at all.
    return tuple(IdentityRule(kind="identity", quantity=quantity, unit=unit) for quantity, unit in _base_units_v1())


def _scale_and_affine_rules_v1() -> tuple[ConversionRule, ...]:
    return (
        ScaleRule(kind="scale", quantity=QuantityKind.LENGTH, from_unit="cm", to_unit="m", scale="0.01"),
        ScaleRule(kind="scale", quantity=QuantityKind.LENGTH, from_unit="mm", to_unit="m", scale="0.001"),
        ScaleRule(kind="scale", quantity=QuantityKind.VELOCITY, from_unit="cm/s", to_unit="m/s", scale="0.01"),
        AffineRule(
            kind="affine", quantity=QuantityKind.TEMPERATURE, from_unit="C", to_unit="K", scale="1", offset="273.15"
        ),
        ScaleRule(kind="scale", quantity=QuantityKind.PRESSURE, from_unit="atm", to_unit="Pa", scale="101325"),
        ScaleRule(kind="scale", quantity=QuantityKind.PRESSURE, from_unit="bar", to_unit="Pa", scale="100000"),
        ScaleRule(kind="scale", quantity=QuantityKind.PRESSURE, from_unit="kPa", to_unit="Pa", scale="1000"),
        ScaleRule(kind="scale", quantity=QuantityKind.PRESSURE, from_unit="MPa", to_unit="Pa", scale="1000000"),
        ScaleRule(kind="scale", quantity=QuantityKind.TIME, from_unit="ms", to_unit="s", scale="0.001"),
        ScaleRule(kind="scale", quantity=QuantityKind.TIME, from_unit="us", to_unit="s", scale="0.000001"),
        ScaleRule(kind="scale", quantity=QuantityKind.VOLUME, from_unit="cm3", to_unit="m3", scale="0.000001"),
        ScaleRule(kind="scale", quantity=QuantityKind.VOLUME, from_unit="L", to_unit="m3", scale="0.001"),
        ScaleRule(kind="scale", quantity=QuantityKind.MOLE_FRACTION, from_unit="%", to_unit="1", scale="0.01"),
        ScaleRule(kind="scale", quantity=QuantityKind.MASS_FRACTION, from_unit="%", to_unit="1", scale="0.01"),
        ScaleRule(kind="scale", quantity=QuantityKind.RELATIVE_UNCERTAINTY, from_unit="%", to_unit="1", scale="0.01"),
    )


def _aliases_v1() -> tuple[UnitAlias, ...]:
    return (
        UnitAlias(quantity=QuantityKind.TEMPERATURE, raw="°C", normalized="C"),
        UnitAlias(quantity=QuantityKind.TEMPERATURE, raw="degC", normalized="C"),
        UnitAlias(quantity=QuantityKind.TEMPERATURE, raw="deg C", normalized="C"),
        UnitAlias(quantity=QuantityKind.VELOCITY, raw="cm s^-1", normalized="cm/s"),
        UnitAlias(quantity=QuantityKind.VELOCITY, raw="cm s-1", normalized="cm/s"),
        UnitAlias(quantity=QuantityKind.VELOCITY, raw="cm/sec", normalized="cm/s"),
        UnitAlias(quantity=QuantityKind.VELOCITY, raw="m s^-1", normalized="m/s"),
        UnitAlias(quantity=QuantityKind.VOLUME, raw="cm^3", normalized="cm3"),
        UnitAlias(quantity=QuantityKind.VOLUME, raw="cc", normalized="cm3"),
        UnitAlias(quantity=QuantityKind.VOLUME, raw="l", normalized="L"),
        UnitAlias(quantity=QuantityKind.VOLUME, raw="liter", normalized="L"),
        UnitAlias(quantity=QuantityKind.VOLUME, raw="litre", normalized="L"),
        # U+00B5 MICRO SIGN and U+03BC GREEK SMALL LETTER MU: a real glyph
        # hazard -- PDF text extraction and OCR both regularly emit either
        # one for what a source paper intends as the SI micro- prefix, and
        # the two are visually near-identical but distinct codepoints.
        UnitAlias(quantity=QuantityKind.TIME, raw="µs", normalized="us"),
        UnitAlias(quantity=QuantityKind.TIME, raw="μs", normalized="us"),
        UnitAlias(quantity=QuantityKind.MOLE_FRACTION, raw="percent", normalized="%"),
        UnitAlias(quantity=QuantityKind.MOLE_FRACTION, raw="-", normalized="1"),
        UnitAlias(quantity=QuantityKind.MOLE_FRACTION, raw="dimensionless", normalized="1"),
        UnitAlias(quantity=QuantityKind.MASS_FRACTION, raw="percent", normalized="%"),
        UnitAlias(quantity=QuantityKind.MASS_FRACTION, raw="-", normalized="1"),
        UnitAlias(quantity=QuantityKind.MASS_FRACTION, raw="dimensionless", normalized="1"),
        UnitAlias(quantity=QuantityKind.EQUIVALENCE_RATIO, raw="-", normalized="1"),
        UnitAlias(quantity=QuantityKind.EQUIVALENCE_RATIO, raw="dimensionless", normalized="1"),
        UnitAlias(quantity=QuantityKind.RELATIVE_UNCERTAINTY, raw="percent", normalized="%"),
        UnitAlias(quantity=QuantityKind.RELATIVE_UNCERTAINTY, raw="-", normalized="1"),
        UnitAlias(quantity=QuantityKind.RELATIVE_UNCERTAINTY, raw="dimensionless", normalized="1"),
    )


TABLE_V1 = ConversionTable(
    table_id="carmel-unit-conversions",
    version=1,
    base_units=_base_units_v1(),
    aliases=_aliases_v1(),
    rules=_identity_rules_v1() + _scale_and_affine_rules_v1(),
)
"""The first (and, as of this module, only) conversion table.

Deliberately tiny: it covers only the 8-paper syngas/laminar-burning-velocity
corpus this milestone targets, not every unit that could ever appear in a
combustion-kinetics paper. A future paper that needs a unit this table does
not cover gets a new ``TABLE_V2`` -- see the module docstring for why
``TABLE_V1`` itself must never be edited in place once datasets exist that
cite its sha256.
"""

TABLES_BY_SHA: Mapping[str, ConversionTable] = MappingProxyType({TABLE_V1.sha256: TABLE_V1})
"""Every conversion table this module ships, keyed by content address.

A dataset record only ever needs to remember a table's sha256 (via
``conversion_table_version`` in :mod:`carmel.schemas.datasets`, once that
field is wired up); this mapping is how a later reader turns that sha256 back
into the actual rules that produced a stored conversion factor.
"""


def table_for_sha(sha256: str) -> ConversionTable:
    """Return the :class:`ConversionTable` whose content address is ``sha256``.

    Args:
        sha256: The table's sha256 content address (see
            :attr:`ConversionTable.sha256`).

    Returns:
        The matching table.

    Raises:
        UnknownConversionTableError: If ``sha256`` does not name any table
            this module ships. There is deliberately no fallback to "the
            current table" -- a shipped table is never mutated (see the
            module docstring), so an unrecognized sha256 can only mean a
            genuinely unknown or future table version, never "an old version
            of this one."
    """
    try:
        return TABLES_BY_SHA[sha256]
    except KeyError:
        raise UnknownConversionTableError(
            f"no conversion table known for sha256 {sha256!r}; known tables: {sorted(TABLES_BY_SHA)!r}"
        ) from None


def normalize_unit(quantity: QuantityKind, unit_raw: str, *, table: ConversionTable = TABLE_V1) -> str:
    """Normalize ``unit_raw``'s SPELLING to this table's canonical spelling for ``quantity``.

    Strips surrounding whitespace ONLY -- no case folding, no NFC/NFKC
    normalization. Raw evidence text is never silently reinterpreted in this
    project; ``K`` (kelvin) and ``k`` (a kilo- prefix fragment) are genuinely
    different tokens, and a future reader must not "helpfully" case-fold this
    function into treating them as the same unit.

    Args:
        quantity: Which quantity ``unit_raw`` is claimed to measure.
        unit_raw: The raw, as-extracted unit spelling.
        table: The conversion table to normalize against. Defaults to
            :data:`TABLE_V1`.

    Returns:
        - If ``quantity`` is ``QuantityKind.OTHER``: the stripped text,
          unchanged -- this table models nothing about ``OTHER``, so there is
          nothing to normalize against.
        - Else if the stripped text is already a known unit of ``quantity``:
          the stripped text, unchanged.
        - Else if the stripped text is a known alias of ``quantity``: that
          alias's ``normalized`` spelling.

    Raises:
        UnknownUnitError: If ``unit_raw`` is empty or whitespace-only, or if
            the stripped text is neither a known unit nor a known alias of
            ``quantity``.
    """
    stripped = unit_raw.strip()
    if not stripped:
        raise UnknownUnitError(
            f"unit_raw is empty or whitespace-only for quantity {quantity.value!r} (raw input: {unit_raw!r})"
        )
    if quantity is QuantityKind.OTHER:
        return stripped
    known = table.known_units(quantity)
    if stripped in known:
        return stripped
    for alias in table.aliases:
        if alias.quantity == quantity and alias.raw == stripped:
            return alias.normalized
    raise UnknownUnitError(f"unknown unit {stripped!r} for quantity {quantity.value!r}; known units: {sorted(known)!r}")


@dataclass(frozen=True, slots=True)
class Converted:
    """The result of a single :func:`convert` call.

    ``exact`` and ``rounded`` are both canonical decimal strings (see
    :func:`~carmel.services.dataset_store.canonical_decimal`). They
    deliberately are not always equal -- see :func:`convert`'s docstring for
    the full rounding-policy rationale.
    """

    exact: str
    """The exact result of the conversion arithmetic, with no rounding applied."""

    rounded: str
    """``exact`` rounded per :attr:`rounding_policy`, to a defensible precision."""

    rule_kind: str
    """Which :class:`ConversionRule` kind produced this result: ``"identity"``/``"scale"``/``"affine"``."""

    rounding_policy: str
    """Which rounding branch ran: ``"identity"`` / ``"significant_digits"`` / ``"decimal_exponent"``."""


# Decimal context precision used for both the exact-arithmetic step and the
# subsequent explicit rounding (quantize) step in convert(). Generous relative
# to dataset_store.canonical_decimal's own _MAX_COEFFICIENT_DIGITS (1000): a
# scale/affine rule's own scale/offset coefficients are always small (at most
# a handful of digits, per TABLE_V1), so multiplying/adding them to a
# 1000-digit input coefficient can add at most a few digits beyond 1000, and
# quantize()'s own working precision must be at least as large as the
# operand's digit count or it raises InvalidOperation rather than rounding.
_CONVERT_PRECISION = 4096


def _round_to_exponent(value: Decimal, exponent: int) -> Decimal:
    """Round ``value`` to ``exponent`` (``ROUND_HALF_EVEN``), inside a high-precision context.

    A plain, undecorated ``value.quantize(...)`` under the ambient default
    context (28 significant digits) would raise ``InvalidOperation`` for any
    operand needing more digits than that to represent at the requested
    exponent -- a real possibility here, since canonical_decimal permits up
    to 1000-significant-digit input. Running the quantize step inside a
    :data:`_CONVERT_PRECISION`-digit context avoids that spurious failure.

    Raises:
        UnitError: If ``value`` still cannot be quantized to ``exponent`` even
            at :data:`_CONVERT_PRECISION` (e.g. the result would need more
            digits than that to represent) -- named with the operands, not
            surfaced as a bare ``decimal.InvalidOperation``.
    """
    quantum = Decimal(1).scaleb(exponent)
    with localcontext() as ctx:
        ctx.prec = _CONVERT_PRECISION
        try:
            return value.quantize(quantum, rounding=ROUND_HALF_EVEN)
        except InvalidOperation as exc:
            raise UnitError(
                f"cannot round {value} to exponent {exponent}: result is too large to quantize ({exc})"
            ) from exc


def _require_canonical_result(
    result: Decimal,
    *,
    value: str,
    from_unit: str,
    to_unit: str,
    part: str,
) -> str:
    """Return ``result`` as a canonical decimal string, or refuse the conversion.

    ``str(Decimal)`` is already the canonical rendering, so for any in-range
    result this is a pure bounds check that returns the same characters. What
    it catches is the edge where exact arithmetic carries a legal input OUT of
    the range :func:`~carmel.services.dataset_store.canonical_decimal` accepts:
    that function bounds both the coefficient digit count and the adjusted
    exponent, and multiplying by a scale factor grows both. Measured: ``1E+999``
    atm is a perfectly acceptable canonical decimal, and converting it to Pa
    yields ``1.01325E+1004`` -- past the adjusted-exponent bound, so a value
    this module handed back as "a canonical decimal string" would be REFUSED by
    the store it is meant to be written to.

    Returning it anyway would push the failure to a later, more confusing
    boundary (or to a consumer that never re-validates), so the conversion is
    refused here instead, naming the operands. This is a bound on the
    CONVERSION, not an opinion about whether the measurement is physical --
    ``convert`` deliberately has no finiteness opinion, exactly as
    ``canonical_decimal`` does not (that judgement belongs to
    ``MeasuredValue``, which is a measured quantity and can hold one).
    """
    try:
        return canonical_decimal(str(result))
    except CanonicalDecimalError as exc:
        raise UnitError(
            f"converting value={value!r} from {from_unit!r} to {to_unit!r} produced a {part} result "
            f"that is not a representable canonical decimal: {exc}"
        ) from exc


def convert(
    value: str,
    *,
    quantity: QuantityKind,
    from_unit: str,
    to_unit: str,
    table: ConversionTable = TABLE_V1,
) -> Converted:
    """Convert ``value`` from ``from_unit`` to ``to_unit`` for ``quantity``.

    ``value`` must already be a canonical decimal string -- this function
    calls :func:`~carmel.services.dataset_store.canonical_decimal` on it and
    REJECTS it if that changes anything, rather than silently canonicalizing.
    ``from_unit``/``to_unit`` must already be NORMALIZED unit spellings --
    callers run :func:`normalize_unit` on both before calling this function;
    ``convert`` itself never resolves an alias.

    ROUNDING POLICY (why each branch differs, pinned here because it is easy
    to get "obviously" wrong): exact ``Decimal`` arithmetic preserves VALUE,
    not the measurement PRECISION the source paper actually reported.
    Converting ``"1.23"`` atm to Pa via ``1.23 * 101325`` gives
    ``124629.75`` -- 8 coefficient digits manufactured out of a 3-significant-
    figure measurement. ``Converted.exact`` always carries that literal exact
    result; ``Converted.rounded`` applies one of three branches depending on
    ``rule_kind``:

    - **identity** -> unchanged. ``rounded == exact == value``, byte-identical
      to the input string, because nothing was computed at all.
    - **scale** -> rounded to the number of SIGNIFICANT DIGITS in the source
      coefficient (``len(Decimal(value).as_tuple().digits)``), mode
      ``ROUND_HALF_EVEN``. Multiplying by a constant scale factor preserves
      RELATIVE precision: a value reported as ``"1.230"`` (4 significant
      figures) and one reported as ``"1.23"`` (3 significant figures) carry
      genuinely different claimed precision and must round differently after
      conversion, even though they are numerically equal before it.
    - **affine** -> rounded to the source value's own DECIMAL EXPONENT
      (absolute precision), mode ``ROUND_HALF_EVEN``. Adding an exact constant
      preserves ABSOLUTE precision, not relative precision: ``25`` degrees C
      to kelvin is ``298.15`` exactly, but rounding that to 2 significant
      figures would wrongly claim ``"300"`` K; rounding to the source value's
      own exponent (``25`` has decimal exponent 0, i.e. no fractional digits)
      correctly gives ``"298"`` K instead.

    Args:
        value: A canonical decimal string (validated via
            :func:`~carmel.services.dataset_store.canonical_decimal`).
        quantity: Which quantity is being converted.
        from_unit: The already-normalized source unit.
        to_unit: The already-normalized target unit.
        table: The conversion table to use. Defaults to :data:`TABLE_V1`.

    Returns:
        A :class:`Converted` recording both the exact and the rounded result.

    Raises:
        UnitError: If ``value`` is not already in canonical decimal form, or
            if the rounding step cannot quantize its result.
        CanonicalDecimalError: If ``value`` is not a valid canonical decimal
            string at all (propagated from
            :func:`~carmel.services.dataset_store.canonical_decimal`).
        UnknownQuantityUnitPairError: If no rule matches ``(quantity,
            from_unit)`` in ``table``, if a matching rule's target unit is not
            ``to_unit``, or if ``quantity`` is ``QuantityKind.OTHER`` and
            ``from_unit != to_unit`` (``OTHER`` permits only the identity
            pair).
    """
    canonical_value = canonical_decimal(value)
    if canonical_value != value:
        raise UnitError(
            f"value {value!r} is not a canonical decimal string (canonical form is {canonical_value!r}); "
            "convert() requires already-canonical input rather than silently canonicalizing it -- run it "
            "through canonical_decimal() at the call site instead"
        )

    if quantity is QuantityKind.OTHER:
        if from_unit != to_unit:
            raise UnknownQuantityUnitPairError(
                f"QuantityKind.OTHER permits only an identity conversion, but from_unit={from_unit!r} != "
                f"to_unit={to_unit!r}"
            )
        return Converted(
            exact=canonical_value, rounded=canonical_value, rule_kind="identity", rounding_policy="identity"
        )

    rule: ConversionRule | None = None
    for candidate in table.rules:
        if candidate.quantity == quantity and _rule_key(candidate)[1] == from_unit:
            rule = candidate
            break
    if rule is None:
        raise UnknownQuantityUnitPairError(
            f"no conversion rule for quantity={quantity.value!r}, from_unit={from_unit!r} in table "
            f"{table.table_id!r} v{table.version}"
        )

    if isinstance(rule, IdentityRule):
        if to_unit != from_unit:
            raise UnknownQuantityUnitPairError(
                f"identity rule for quantity={quantity.value!r}, unit={rule.unit!r} does not match "
                f"requested to_unit={to_unit!r}"
            )
        return Converted(
            exact=canonical_value, rounded=canonical_value, rule_kind="identity", rounding_policy="identity"
        )

    if to_unit != rule.to_unit:
        raise UnknownQuantityUnitPairError(
            f"{rule.kind} rule for quantity={quantity.value!r}, from_unit={from_unit!r} targets "
            f"{rule.to_unit!r}, not the requested to_unit={to_unit!r}"
        )

    source = Decimal(canonical_value)
    with localcontext() as ctx:
        ctx.prec = _CONVERT_PRECISION
        ctx.traps[Inexact] = True
        ctx.traps[Rounded] = True
        try:
            if isinstance(rule, ScaleRule):
                exact_dec = source * Decimal(rule.scale)
            else:
                exact_dec = source * Decimal(rule.scale) + Decimal(rule.offset)
        except (Inexact, Rounded) as exc:
            raise UnitError(
                f"conversion arithmetic for {value!r} {from_unit!r} -> {to_unit!r} was not exact at "
                f"{_CONVERT_PRECISION}-digit precision: {exc}"
            ) from exc

    if isinstance(rule, ScaleRule):
        significant_digits = len(source.as_tuple().digits)
        exponent = exact_dec.adjusted() - (significant_digits - 1)
        rounded_dec = _round_to_exponent(exact_dec, exponent)
        rounding_policy = "significant_digits"
    else:
        source_exponent = source.as_tuple().exponent
        assert isinstance(source_exponent, int)  # canonical_decimal never yields 'n'/'N'/'F' special exponents
        rounded_dec = _round_to_exponent(exact_dec, source_exponent)
        rounding_policy = "decimal_exponent"

    return Converted(
        exact=_require_canonical_result(exact_dec, value=value, from_unit=from_unit, to_unit=to_unit, part="exact"),
        rounded=_require_canonical_result(
            rounded_dec, value=value, from_unit=from_unit, to_unit=to_unit, part="rounded"
        ),
        rule_kind=rule.kind,
        rounding_policy=rounding_policy,
    )
