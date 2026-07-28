"""Tests for carmel.services.chem.

RDKit is an optional dependency. These tests use monkeypatch to inject fake
`rdkit`/`rdkit.Chem` modules into sys.modules so both the "installed" and
"not installed" code paths are exercised regardless of what's actually available in
the test environment (setting `sys.modules["rdkit"] = None` makes `import rdkit`
raise ImportError, which is exactly how Python's import system behaves for a module
explicitly marked absent in the cache).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from carmel.services.chem import canonical_smiles, inchikey, rdkit_available


class _FakeMol:
    """Stand-in for an rdkit Mol object."""


def _install_fake_rdkit(monkeypatch: pytest.MonkeyPatch, *, garbage_value: str = "garbage") -> None:
    """Install a fake rdkit module that parses everything except `garbage_value`."""

    def mol_from_smiles(raw: str) -> _FakeMol | None:
        if raw == garbage_value:
            return None
        return _FakeMol()

    def mol_to_smiles(mol: _FakeMol) -> str:
        return "CANONICAL_SMILES"

    def mol_to_inchikey(mol: _FakeMol) -> str:
        return "FAKEINCHIKEY-N"

    fake_chem = types.SimpleNamespace(
        MolFromSmiles=mol_from_smiles,
        MolToSmiles=mol_to_smiles,
        MolToInchiKey=mol_to_inchikey,
    )
    fake_rdlogger = types.SimpleNamespace(DisableLog=lambda *a, **k: None)
    fake_rdkit = types.ModuleType("rdkit")
    fake_rdkit.Chem = fake_chem  # type: ignore[attr-defined]
    fake_rdkit.RDLogger = fake_rdlogger  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rdkit", fake_rdkit)


def _install_missing_rdkit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate rdkit not being installed at all."""
    monkeypatch.setitem(sys.modules, "rdkit", None)


class TestRdkitAvailable:
    def test_available_when_importable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_rdkit(monkeypatch)
        assert rdkit_available() is True

    def test_unavailable_when_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_missing_rdkit(monkeypatch)
        assert rdkit_available() is False

    def test_consistent_with_real_environment(self) -> None:
        # Whatever the real environment has, rdkit_available() must agree with a
        # direct import attempt and never raise.
        try:
            import rdkit  # noqa: F401

            expected = True
        except ImportError:
            expected = False
        assert rdkit_available() is expected


class TestCanonicalSmiles:
    def test_returns_none_when_rdkit_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_missing_rdkit(monkeypatch)
        assert canonical_smiles("CCO") is None

    def test_returns_canonical_form_when_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_rdkit(monkeypatch)
        assert canonical_smiles("CCO") == "CANONICAL_SMILES"

    def test_returns_none_for_garbage_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_rdkit(monkeypatch)
        assert canonical_smiles("garbage") is None

    def test_never_raises_on_internal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(raw: str) -> Any:
            raise RuntimeError("boom")

        fake_chem = types.SimpleNamespace(MolFromSmiles=boom, MolToSmiles=lambda m: "x", MolToInchiKey=lambda m: "y")
        fake_rdkit = types.ModuleType("rdkit")
        fake_rdkit.Chem = fake_chem  # type: ignore[attr-defined]
        fake_rdkit.RDLogger = types.SimpleNamespace(DisableLog=lambda *a, **k: None)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rdkit", fake_rdkit)
        assert canonical_smiles("anything") is None


class TestInchikey:
    def test_returns_none_when_rdkit_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_missing_rdkit(monkeypatch)
        assert inchikey("CCO") is None

    def test_returns_inchikey_when_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_rdkit(monkeypatch)
        assert inchikey("CCO") == "FAKEINCHIKEY-N"

    def test_returns_none_for_garbage_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_rdkit(monkeypatch)
        assert inchikey("garbage") is None

    def test_never_raises_on_internal_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(raw: str) -> Any:
            raise RuntimeError("boom")

        fake_chem = types.SimpleNamespace(MolFromSmiles=boom, MolToSmiles=lambda m: "x", MolToInchiKey=lambda m: "y")
        fake_rdkit = types.ModuleType("rdkit")
        fake_rdkit.Chem = fake_chem  # type: ignore[attr-defined]
        fake_rdkit.RDLogger = types.SimpleNamespace(DisableLog=lambda *a, **k: None)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "rdkit", fake_rdkit)
        assert inchikey("anything") is None


def test_real_rdkit_end_to_end_if_installed() -> None:
    """When rdkit is genuinely installed, sanity-check against real chemistry."""
    pytest.importorskip("rdkit")
    assert canonical_smiles("CCO") is not None
    assert canonical_smiles("not a smiles string $$$") is None
    assert inchikey("CCO") is not None
