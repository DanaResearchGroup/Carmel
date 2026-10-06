"""Cheminformatics helpers backed by the optional ``rdkit`` dependency.

Every conversion helper here fails SOFT: a missing ``rdkit`` install or an unparseable
input returns ``None``, never an exception. The one exception is
``load_inchikey_table``, which fails CLOSED with ``ValueError`` because a pinned identity
table that cannot be verified must never be used. ``rdkit`` is imported lazily inside each
function so importing this module never fails when the optional dependency is absent.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib import resources
from pathlib import Path
from typing import cast

INCHIKEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")
_INCHIKEY_TABLE_LIMIT = 64 * 1024


def _disable_rdkit_logging() -> None:
    """Silence RDKit's C++-level stderr chatter (e.g. "SMILES Parse Error") for
    invalid input. Best-effort: if the logger API is unavailable, proceed anyway."""
    try:
        # Imported from `rdkit.rdBase` (the compiled extension `RDLogger.py` re-exports
        # this from) rather than from `rdkit.RDLogger` itself: the `rdkit-stubs` package
        # (bundled inside every `rdkit` install, verified via its dist-info RECORD) types
        # `rdBase.DisableLog` fully, but `RDLogger.pyi`'s `__all__` does not re-export it,
        # so mypy sees it as missing there even though it exists at runtime.
        from rdkit.rdBase import DisableLog

        DisableLog("rdApp.*")
    except Exception:  # noqa: BLE001 - logging suppression must never break callers
        pass


def rdkit_available() -> bool:
    """Report whether the optional ``rdkit`` dependency is importable.

    Returns:
        True if ``rdkit`` can be imported, False otherwise.
    """
    try:
        # No type-ignore here: mypy only honours that directive when it is the FIRST
        # comment on the line, so the noqa directive that has to precede it would silently
        # neutralise it anyway. None is needed -- `make typecheck` requires the `agents`
        # extra (see docs/development.md), and rdkit resolves in that environment.
        import rdkit  # noqa: F401
    except ImportError:
        return False
    return True


def canonical_smiles(raw: str) -> str | None:
    """Canonicalize a SMILES string via RDKit.

    Args:
        raw: A SMILES string as printed in a source document.

    Returns:
        The RDKit canonical SMILES, or None if RDKit is not installed or ``raw``
        cannot be parsed. Never raises.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None
    _disable_rdkit_logging()
    try:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return None
        return str(Chem.MolToSmiles(mol))
    except Exception:  # noqa: BLE001 - fail soft on any parse/canonicalization error
        return None


def inchikey(raw_smiles: str) -> str | None:
    """Compute the InChIKey for a SMILES string via RDKit.

    Args:
        raw_smiles: A SMILES string as printed in a source document.

    Returns:
        The InChIKey, or None if RDKit is not installed or ``raw_smiles`` cannot be
        parsed. Never raises.
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None
    _disable_rdkit_logging()
    try:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            return None
        # `rdkit-stubs`' `inchi.pyi` leaves `MolToInchiKey` entirely unannotated (no
        # parameter or return types at all), so this is a genuine gap in the third-party
        # stub, not something a typed rewrite on our side can close. Narrowly ignored by
        # error code, not blanket-ignored.
        return str(Chem.MolToInchiKey(mol))  # type: ignore[no-untyped-call]
    except Exception:  # noqa: BLE001 - fail soft on any parse/conversion error
        return None


def smiles_from_inchi(raw: str) -> str | None:
    """Resolve a source-stated InChI, refusing absent RDKit or invalid identity."""
    try:
        from rdkit import Chem
    except ImportError:
        return None
    _disable_rdkit_logging()
    try:
        mol = Chem.MolFromInchi(raw if raw.startswith("InChI=") else "InChI=" + raw)  # type: ignore[no-untyped-call]
        return str(Chem.MolToSmiles(mol)) if mol is not None else None
    except Exception:  # noqa: BLE001 - third-party identity conversion fails closed
        return None


def load_inchikey_table(path: Path | None = None) -> dict[str, str]:
    """Load and re-derive the pinned InChIKey identities offline.

    Raises:
        ValueError: The table is oversized, not UTF-8 JSON, not a string-to-string map,
            has a malformed entry, RDKit is unavailable, or RDKit does not recompute an
            entry's key from its InChI.
    """
    source = path if path is not None else resources.files("carmel.data").joinpath("inchikey_to_inchi.json")
    with source.open("rb") as handle:
        raw = handle.read(_INCHIKEY_TABLE_LIMIT + 1)
    if len(raw) > _INCHIKEY_TABLE_LIMIT:
        raise ValueError("pinned InChIKey table exceeds 64 KiB")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"pinned InChIKey table is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in payload.items()
    ):
        raise ValueError("pinned InChIKey table must map strings to strings")
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ValueError("RDKit is required to verify the pinned InChIKey table") from exc
    _disable_rdkit_logging()
    table = cast("dict[str, str]", payload)
    verified: dict[str, str] = {}
    for key, inchi in sorted(table.items()):
        if INCHIKEY_PATTERN.fullmatch(key) is None or not inchi.startswith("InChI="):
            raise ValueError(f"invalid pinned InChIKey table entry {key!r}")
        try:
            molecule = Chem.MolFromInchi(inchi)  # type: ignore[no-untyped-call]
            actual = str(Chem.MolToInchiKey(molecule)) if molecule is not None else ""  # type: ignore[no-untyped-call]
        except Exception as exc:  # noqa: BLE001 - a broken identity must reject the whole table
            raise ValueError(f"RDKit could not verify pinned InChIKey {key}") from exc
        if actual != key:
            raise ValueError(f"RDKit recomputed {actual!r}, not pinned InChIKey {key!r}")
        verified[key] = inchi
    return verified


@cache
def _default_inchikey_table() -> dict[str, str]:
    return load_inchikey_table()


def inchi_from_inchikey(raw: str) -> str | None:
    """Resolve a key-shaped identifier only through the verified pinned table."""
    if INCHIKEY_PATTERN.fullmatch(raw) is None or not rdkit_available():
        return None
    return _default_inchikey_table().get(raw)
