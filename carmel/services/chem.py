"""Cheminformatics helpers backed by the optional ``rdkit`` dependency.

Every function here fails SOFT: a missing ``rdkit`` install or an unparseable input
returns ``None``, never an exception. ``rdkit`` is imported lazily inside each
function so importing this module never fails when the optional dependency is absent.
"""

from __future__ import annotations


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
        # No `# type: ignore` here: mypy only honours that comment when it is the FIRST
        # comment on the line, so the `# noqa` that has to precede it would silently
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
