"""Explicit leaf ledger for reconstructing the historical case-3 revision chain."""

import copy
import hashlib
import json

REVISION2_CANONICAL = "de584a9e6991e01a824dadf48c522fdd0251007863580d0654d61c5282548b1b"
REVISION2_FILE = "cdd9ef2606e331d4fd7097415afa3c12a849567d1fbc47938b77aff6d3abc347"
A2_LEAVES = {
    "x40011001.xml": (3, 1),
    "x40011002.xml": (7, 5),
    "x40011003.xml": (9, 7),
    "x40011004.xml": (9, 7),
    "x40011005.xml": (1, 0),
    "x40011006.xml": (10, 10),
    "x40011007.xml": (2, 0),
    "x40011009.xml": (5, 3),
    "x40011010.xml": (7, 7),
    "x40011011.xml": (4, 3),
    "x40011012.xml": (7, 7),
}


def reverse_a2(document):
    """Undo exactly the declared development leaves and verify both predecessor hashes.

    Current values and membership must match the explicit ledger before reversal.
    The whole-document hashes detect any additional edit, including holdout changes.
    """
    previous = copy.deepcopy(document)
    seen = set()
    for study in previous["split"]["studies"]:
        for item in study["files"]:
            if item["path"] not in A2_LEAVES:
                continue
            assert study["set"] == "development"
            assert item["path"] not in seen
            seen.add(item["path"])
            rows, low = A2_LEAVES[item["path"]]
            assert item["rows"] == item["observations"] == rows
            assert item["low_temperature_rows"] == item["low_temperature_observations"] == low
            assert item["acceptance"] == "supported" and item["refusal_reason"] is None
            item.update(
                acceptance="refused",
                refusal_reason="rcm_thermo_unavailable",
                low_temperature_rows=0,
                low_temperature_observations=0,
            )
    assert seen == set(A2_LEAVES)
    canonical = json.dumps(previous, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == REVISION2_CANONICAL
    file_bytes = (json.dumps(previous, ensure_ascii=False, indent=2) + "\n").encode()
    assert hashlib.sha256(file_bytes).hexdigest() == REVISION2_FILE
    return previous
