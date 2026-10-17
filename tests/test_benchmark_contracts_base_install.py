"""Contract validation that must work without the optional RDKit dependency."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PENTANE_CONTRACT = ROOT / "benchmarks/n-pentane-low-t/contract.json"
HEPTANE_CONTRACT = ROOT / "benchmarks/n-heptane-low-t/contract.json"


def test_both_contracts_load_and_validate_when_rdkit_is_unimportable() -> None:
    code = f"""
import sys
from pathlib import Path

sys.modules["rdkit"] = None
from carmel.benchmarks.n_pentane import load_contract
from carmel.benchmarks.schema import validate_contract_json

pentane = Path({str(PENTANE_CONTRACT)!r})
heptane = Path({str(HEPTANE_CONTRACT)!r})
assert load_contract(pentane).case == "n-pentane-low-t"
assert validate_contract_json(pentane.read_bytes()).case == "n-pentane-low-t"
assert validate_contract_json(heptane.read_bytes()).case == "n-heptane-low-t"
"""

    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
