"""Tests for carmel.services.dataset_store: canonical JSON, decimal canonicalization,
and content-addressed storage for literature dataset payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from carmel.services.dataset_store import (
    _DECIMAL_REPR_VERSION_KEY,
    _MAX_JSON_DEPTH,
    DATASET_STORE_DIR,
    CanonicalDecimalError,
    CanonicalJsonError,
    _raw_bytes_nest_too_deeply,
    canonical_decimal,
    canonical_json_bytes,
    compute_dataset_sha,
    dataset_decimal_repr_version,
    dataset_path,
    list_datasets,
    load_dataset,
    store_dataset,
    verify_dataset,
)


class TestCanonicalJsonBytes:
    def test_key_order_does_not_affect_output(self) -> None:
        a = {"b": "2", "a": "1"}
        b = {"a": "1", "b": "2"}
        assert canonical_json_bytes(a) == canonical_json_bytes(b)

    def test_output_is_sorted_compact_utf8_with_trailing_newline(self) -> None:
        payload = {"b": "2", "a": "1"}
        out = canonical_json_bytes(payload)
        assert out == b'{"a":"1","b":"2"}\n'

    def test_non_ascii_is_kept_literal_not_escaped(self) -> None:
        payload = {"note": "þ¼"}
        out = canonical_json_bytes(payload)
        assert "þ¼".encode() in out

    def test_rejects_top_level_float(self) -> None:
        with pytest.raises(ValueError, match="float"):
            canonical_json_bytes(1.5)

    def test_rejects_float_nested_in_list(self) -> None:
        with pytest.raises(ValueError, match="float"):
            canonical_json_bytes({"values": [1, 2.0, 3]})

    def test_rejects_float_nested_in_dict_value(self) -> None:
        with pytest.raises(ValueError, match="float"):
            canonical_json_bytes({"outer": {"inner": 3.14}})

    def test_bool_is_allowed(self) -> None:
        # bool is a subclass of int in Python, but is a legitimate JSON type and must
        # not be confused with the float rejection above.
        out = canonical_json_bytes({"flag": True})
        assert out == b'{"flag":true}\n'

    def test_rejects_non_string_dict_key(self) -> None:
        with pytest.raises(ValueError, match="key"):
            canonical_json_bytes({1: "a"})

    def test_rejects_arbitrary_object(self) -> None:
        class Thing:
            pass

        with pytest.raises(ValueError):
            canonical_json_bytes({"x": Thing()})

    def test_deterministic_across_repeated_calls(self) -> None:
        payload = {"z": "1", "a": ["x", "y"], "m": {"k": "v"}}
        assert canonical_json_bytes(payload) == canonical_json_bytes(payload)


class TestCanonicalDecimal:
    def test_preserves_trailing_zeros_as_distinct_significance(self) -> None:
        # Trailing zeros carry measurement precision (5 sig figs vs 3) and are a
        # genuinely different fact, so they must NOT collapse to the same output.
        # `.normalize()` is deliberately never called here for exactly this reason --
        # see the module docstring / canonical_decimal docstring.
        assert canonical_decimal("1.2300") != canonical_decimal("1.23")

    def test_differing_significance_changes_dataset_address(self) -> None:
        payload_a = {"k": canonical_decimal("1.23")}
        payload_b = {"k": canonical_decimal("1.2300")}
        assert compute_dataset_sha(payload_a) != compute_dataset_sha(payload_b)

    def test_idempotent(self) -> None:
        for text in ("1.2300", "-0.005", "1000", "0"):
            once = canonical_decimal(text)
            twice = canonical_decimal(once)
            assert once == twice

    def test_equal_value_and_significance_normalizes_sign_and_exponent_form(self) -> None:
        # "+1.50" and "1.50" carry equal value and equal significance (both 3 sig
        # figs), so they must canonicalize identically; a redundant leading "+" is
        # not itself significant precision information.
        assert canonical_decimal("+1.50") == canonical_decimal("1.50")

    def test_pins_str_decimal_form_for_small_magnitude(self) -> None:
        # Chosen exponent-form rule: canonicalize via Python's own `str(Decimal)`
        # rendering -- never plain fixed-point expansion (`format(d, "f")`, which is
        # unsound for exponent-form input, see canonical_decimal's docstring). This
        # also happens to match how `Decimal.__str__` itself switches to scientific
        # notation for small magnitudes.
        assert canonical_decimal("0.0000001") == "1E-7"

    def test_pins_str_decimal_form_for_large_magnitude(self) -> None:
        assert canonical_decimal("123000000") == "123000000"

    def test_exponent_form_input_is_not_collapsed_onto_equal_value_integer(self) -> None:
        # Regression pin: an earlier (wrong) design rendered via plain fixed-point
        # (`format(d, "f")`), which silently collapsed "1E+3" (1 significant figure)
        # onto the same canonical form as "1000" (up to 4 significant figures as
        # literally written) -- both became "1000". That destroyed exactly the kind
        # of significance distinction this function exists to preserve. The three
        # inputs below share the same numeric value but different significance and
        # must all canonicalize distinctly.
        one_sig_fig = canonical_decimal("1E+3")
        two_sig_figs = canonical_decimal("1.0E+3")
        four_digits_as_written = canonical_decimal("1000")
        assert len({one_sig_fig, two_sig_figs, four_digits_as_written}) == 3
        assert one_sig_fig == "1E+3"
        assert two_sig_figs == "1.0E+3"
        assert four_digits_as_written == "1000"

    def test_does_not_expand_absurd_exponent_into_huge_string(self) -> None:
        # Regression pin: `format(Decimal("1E+1000000"), "f")` used to materialize a
        # million-character string -- an unbounded memory blowup driven purely by the
        # exponent, exactly the shape of corruption PDF-extracted numeric text can
        # produce (e.g. OCR mangling "1.0e3" into "1.0e3000"). Magnitudes this extreme
        # are now rejected outright rather than canonicalized. Note `str(Decimal)`
        # itself never had this failure mode (str(Decimal("1E+1000000")) is just
        # "1E+1000000", ~10 characters) -- the bound is kept anyway as a loud,
        # independent guard against corrupted/garbage exponents on their own merits.
        with pytest.raises(ValueError, match="magnitude"):
            canonical_decimal("1E+1000000")

    def test_rejects_out_of_range_magnitude(self) -> None:
        with pytest.raises(ValueError, match="magnitude"):
            canonical_decimal("1E+3000")

    def test_preserves_significant_figures_for_large_scientific_notation(self) -> None:
        # Regression pin: a combustion A-factor like "7.000E+17" (4 significant
        # figures) must never expand to the 18-digit integer "700000000000000000",
        # which would misrepresent it as having far more precision than reported.
        # str(Decimal) additionally keeps it rendered exactly as printed in the
        # source paper, rather than re-encoded into an unfamiliar form.
        canonical = canonical_decimal("7.000E+17")
        assert canonical == "7.000E+17"
        # 4 significant figures preserved as 4 coefficient digits, not 18.
        digits_part = canonical.split("E")[0].replace(".", "").lstrip("-")
        assert len(digits_part) == 4

    def test_avogadro_and_small_scientific_values_round_trip(self) -> None:
        assert canonical_decimal("6.022e23") == canonical_decimal("6.022e23")
        assert canonical_decimal("1.23e-20") == canonical_decimal("1.23e-20")
        # Different significance at the same order of magnitude must still differ.
        assert canonical_decimal("6.022e23") != canonical_decimal("6.0220e23")

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            canonical_decimal("nan")

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError):
            canonical_decimal("inf")

    def test_rejects_negative_inf(self) -> None:
        with pytest.raises(ValueError):
            canonical_decimal("-inf")

    def test_rejects_empty_string(self) -> None:
        with pytest.raises(ValueError):
            canonical_decimal("")

    def test_rejects_whitespace_only(self) -> None:
        with pytest.raises(ValueError):
            canonical_decimal("   ")

    def test_rejects_garbage_text(self) -> None:
        with pytest.raises(ValueError):
            canonical_decimal("not-a-number")

    def test_rejects_digit_separator(self) -> None:
        # Old loose Decimal("1_000") silently returned "1000" -- Python's Decimal (like
        # float) accepts '_' digit separators. That let a value that could never
        # survive paper extraction (the strict core refuses '_' outright) sneak into a
        # canonical dataset via any caller that types a numeric string by hand.
        with pytest.raises(CanonicalDecimalError):
            canonical_decimal("1_000")

    def test_rejects_leading_dot_form(self) -> None:
        # Old loose Decimal(".5") silently returned "0.5". The strict core's mantissa
        # grammar requires at least one digit before an optional decimal point, so a
        # leading-dot numeral -- never produced by the paper-extraction path -- must
        # not be accepted here either.
        with pytest.raises(CanonicalDecimalError):
            canonical_decimal(".5")

    def test_rejects_trailing_dot_form(self) -> None:
        # Old loose Decimal("1.") silently returned "1". The strict core's mantissa
        # grammar requires at least one digit after a decimal point, so a
        # trailing-dot numeral must not be accepted here either.
        with pytest.raises(CanonicalDecimalError):
            canonical_decimal("1.")

    def test_rejects_unicode_minus_sign_because_it_requires_an_unrecorded_repair(self) -> None:
        # The strict core CAN repair U+2212 MINUS SIGN to ASCII '-', but
        # canonical_decimal must not accept a value that needed a repair: a canonical
        # decimal string is the end of the pipeline, and any repair must already have
        # happened upstream where it can be recorded alongside its provenance. Message
        # must point at the repair path so a caller understands why this was rejected.
        with pytest.raises(CanonicalDecimalError, match="repair"):
            canonical_decimal("−1.5")

    def test_surrounding_whitespace_is_still_accepted_not_a_divergence(self) -> None:
        # Both the strict core (span.strip()) and bare Decimal() strip surrounding
        # whitespace -- this is NOT one of the closed divergences. Pinned so nobody
        # later "fixes" this as if it were a gap.
        assert canonical_decimal(" 1.5 ") == canonical_decimal("1.5")


class TestDatasetPath:
    def test_rejects_short_sha(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            dataset_path(tmp_path, "abc123")

    def test_rejects_uppercase_hex(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            dataset_path(tmp_path, "A" * 64)

    def test_rejects_path_traversal_attempt(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            dataset_path(tmp_path, "../../etc/passwd")

    def test_valid_sha_resolves_under_datasets_dir(self, tmp_path: Path) -> None:
        sha = "0" * 64
        path = dataset_path(tmp_path, sha)
        assert path == (tmp_path / DATASET_STORE_DIR / f"{sha}.json").resolve()

    def test_rejects_trailing_newline(self, tmp_path: Path) -> None:
        """``$`` matches just BEFORE a trailing newline under ``match``; ``fullmatch`` closes it."""
        digest_with_newline = "a" * 64 + "\n"
        with pytest.raises(ValueError, match="invalid dataset sha256"):
            dataset_path(tmp_path, digest_with_newline)

    def test_accepts_well_formed_digest(self, tmp_path: Path) -> None:
        well_formed_digest = "a" * 64
        path = dataset_path(tmp_path, well_formed_digest)
        assert path == (tmp_path / DATASET_STORE_DIR / f"{well_formed_digest}.json").resolve()


class TestComputeDatasetSha:
    def test_returns_64_char_lowercase_hex(self) -> None:
        sha = compute_dataset_sha({"a": "1"})
        assert len(sha) == 64
        assert sha == sha.lower()
        int(sha, 16)  # raises if not valid hex


class TestStoreAndLoadDataset:
    def test_round_trip_returns_equal_content(self, tmp_path: Path) -> None:
        payload = {"compound": "toluene", "k": canonical_decimal("1.2300")}

        stored = store_dataset(tmp_path, payload)
        loaded = load_dataset(tmp_path, stored.sha256)

        assert loaded == payload
        assert verify_dataset(tmp_path, stored.sha256) is True

    def test_stored_path_matches_dataset_path(self, tmp_path: Path) -> None:
        payload = {"a": "1"}
        stored = store_dataset(tmp_path, payload)
        assert stored.path == dataset_path(tmp_path, stored.sha256)

    def test_storing_identical_payload_twice_is_idempotent(self, tmp_path: Path) -> None:
        payload = {"compound": "benzene", "k": canonical_decimal("2.500")}

        first = store_dataset(tmp_path, payload)
        mtime_before = first.path.stat().st_mtime_ns

        second = store_dataset(tmp_path, payload)

        assert second.sha256 == first.sha256
        assert second.path == first.path
        assert first.path.stat().st_mtime_ns == mtime_before

    def test_key_order_produces_same_stored_file(self, tmp_path: Path) -> None:
        a = store_dataset(tmp_path, {"b": "2", "a": "1"})
        b = store_dataset(tmp_path, {"a": "1", "b": "2"})
        assert a.sha256 == b.sha256
        assert a.path == b.path

    def test_corrupted_on_disk_bytes_raise_on_restore(self, tmp_path: Path) -> None:
        payload = {"compound": "toluene", "k": "1.00"}
        stored = store_dataset(tmp_path, payload)

        with open(stored.path, "r+b") as f:
            f.seek(0)
            byte = f.read(1)
            f.seek(0)
            f.write(bytes([byte[0] ^ 0xFF]))

        with pytest.raises(ValueError, match="(?i)collision|corrupt"):
            store_dataset(tmp_path, payload)

    def test_verify_dataset_returns_false_on_tampered_file(self, tmp_path: Path) -> None:
        payload = {"compound": "toluene", "k": "1.00"}
        stored = store_dataset(tmp_path, payload)

        with open(stored.path, "r+b") as f:
            f.seek(0)
            byte = f.read(1)
            f.seek(0)
            f.write(bytes([byte[0] ^ 0xFF]))

        assert verify_dataset(tmp_path, stored.sha256) is False

    def test_load_dataset_raises_file_not_found_when_absent(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path, "0" * 64)

    def test_non_ascii_content_round_trips_byte_exact_without_nfc_normalization(self, tmp_path: Path) -> None:
        # A string containing a combining-character sequence whose NFC-normalized form
        # differs byte-for-byte from the original: "é" built from "e" + combining
        # acute accent (U+0301) normalizes under NFC to the single precomposed
        # codepoint U+00E9. Storing bytes through NFC normalization would silently
        # rewrite the evidence the grounding gate depends on, so this is asserted
        # directly against the raw on-disk bytes (never through json.loads, which
        # would not by itself reveal a prior normalization pass having happened).
        raw_string = "Cor sat at 298.15 K, k = 1.0 s⁻¹, mangled: /C0þ¼, élan"
        payload = {"note": raw_string}

        stored = store_dataset(tmp_path, payload)
        on_disk = stored.path.read_bytes()

        assert raw_string.encode("utf-8") in on_disk
        import unicodedata

        nfc_string = unicodedata.normalize("NFC", raw_string)
        assert nfc_string != raw_string  # sanity: this case exercises real NFC drift
        assert nfc_string.encode("utf-8") not in on_disk


class TestListDatasets:
    def test_lists_stored_shas_in_sorted_order(self, tmp_path: Path) -> None:
        s1 = store_dataset(tmp_path, {"a": "1"})
        s2 = store_dataset(tmp_path, {"a": "2"})

        result = list_datasets(tmp_path)

        assert result == sorted([s1.sha256, s2.sha256])

    def test_empty_store_returns_empty_list(self, tmp_path: Path) -> None:
        assert list_datasets(tmp_path) == []

    def test_skips_files_not_matching_sha_shape(self, tmp_path: Path) -> None:
        store_dataset(tmp_path, {"a": "1"})
        datasets_dir = tmp_path / DATASET_STORE_DIR
        (datasets_dir / "not-a-sha.json").write_text("{}", encoding="utf-8")
        (datasets_dir / ("a" * 64 + ".txt")).write_text("nope", encoding="utf-8")

        result = list_datasets(tmp_path)

        assert all(len(sha) == 64 for sha in result)
        assert "not-a-sha" not in result

    def test_rejects_trailing_newline_before_json_suffix(self, tmp_path: Path) -> None:
        """A POSIX-legal filename ``"<64 hex>\\n.json"`` must not be listed as a genuine digest.

        ``$`` matches just BEFORE a trailing newline under ``match``, and
        ``Path.stem`` on such a filename yields the hex digits plus the embedded
        newline; ``fullmatch`` closes the hole.
        """
        datasets_dir = tmp_path / DATASET_STORE_DIR
        datasets_dir.mkdir(parents=True, exist_ok=True)
        digest_with_newline = "a" * 64 + "\n"
        (datasets_dir / f"{digest_with_newline}.json").write_text("{}", encoding="utf-8")

        result = list_datasets(tmp_path)

        assert digest_with_newline not in result
        assert result == []

    def test_accepts_well_formed_digest_filename(self, tmp_path: Path) -> None:
        datasets_dir = tmp_path / DATASET_STORE_DIR
        datasets_dir.mkdir(parents=True, exist_ok=True)
        well_formed_digest = "a" * 64
        (datasets_dir / f"{well_formed_digest}.json").write_text("{}", encoding="utf-8")

        result = list_datasets(tmp_path)

        assert result == [well_formed_digest]


class TestLoadDatasetVerifiesContentAddress:
    """Finding #1: ``load_dataset`` must recompute the digest of the bytes it is
    about to return and refuse to return anything that does not hash to the
    requested address -- otherwise a hand-placed or corrupted file that merely
    *parses* as JSON is returned as if it were the genuine, address-verified
    dataset."""

    def test_raises_when_on_disk_bytes_do_not_hash_to_the_requested_sha256(self, tmp_path: Path) -> None:
        stored = store_dataset(tmp_path, {"compound": "toluene", "k": "1.00"})

        # Simulate a hand-placed (or corrupted-in-place) file: perfectly valid,
        # perfectly parseable canonical bytes for SOME dataset, sitting under a
        # filename (sha256) that they do not actually hash to.
        wrong_sha = "0" * 64
        wrong_path = dataset_path(tmp_path, wrong_sha)
        wrong_path.write_bytes(stored.path.read_bytes())

        with pytest.raises(ValueError, match="(?i)corrupt|tamper"):
            load_dataset(tmp_path, wrong_sha)

    def test_has_no_verify_false_escape_hatch(self, tmp_path: Path) -> None:
        # A content-addressed load must never offer an opt-out of the address
        # check -- that would just be a footgun with a different spelling. This
        # asserts the parameter does not exist at all, so a load call can never
        # accidentally silence the verification added for finding #1.
        stored = store_dataset(tmp_path, {"a": "1"})
        with pytest.raises(TypeError):
            load_dataset(tmp_path, stored.sha256, verify=False)  # type: ignore[call-arg]


class TestVerifyDatasetRequiresCanonicalForm:
    """Finding #2: a file whose bytes hash to their own filename can still be
    non-canonical (different key order, different whitespace, a stale/missing
    version marker, ...). ``verify_dataset`` must reject that, not just check
    the hash."""

    def test_rejects_hash_matching_bytes_that_are_not_canonical_json(self, tmp_path: Path) -> None:
        # Valid JSON, and its filename is genuinely the sha256 of these exact
        # bytes -- but the bytes are not the canonical encoding this store
        # guarantees (extra whitespace after the colon, no trailing newline).
        non_canonical_bytes = b'{"a": "1"}'
        sha = hashlib.sha256(non_canonical_bytes).hexdigest()
        path = dataset_path(tmp_path, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(non_canonical_bytes)

        assert verify_dataset(tmp_path, sha) is False

    def test_rejects_non_canonical_encoding_of_an_otherwise_complete_payload(self, tmp_path: Path) -> None:
        # The test above passes for the WRONG REASON and cannot be relied on
        # alone: its bytes also lack the decimal-representation version marker,
        # so `verify_dataset` returns False via the version check even when the
        # canonicality check is disabled entirely (measured by mutation).
        #
        # This fixture differs from a genuinely canonical file in ENCODING ONLY:
        # same parsed content, version marker present, keys in a non-sorted
        # order with JSON's default whitespace and no trailing newline. So the
        # canonicality check is the only thing that can reject it, which is what
        # makes this test load-bearing for finding #2.
        payload = {"a": "1", _DECIMAL_REPR_VERSION_KEY: 1}
        non_canonical_bytes = json.dumps(payload, sort_keys=False).encode("utf-8")
        assert non_canonical_bytes != canonical_json_bytes(payload)
        assert json.loads(non_canonical_bytes) == json.loads(canonical_json_bytes(payload))

        sha = hashlib.sha256(non_canonical_bytes).hexdigest()
        path = dataset_path(tmp_path, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(non_canonical_bytes)

        assert verify_dataset(tmp_path, sha) is False

    def test_accepts_a_genuinely_canonical_stored_dataset(self, tmp_path: Path) -> None:
        stored = store_dataset(tmp_path, {"a": "1"})
        assert verify_dataset(tmp_path, stored.sha256) is True


class TestIntIsBoundedNotBanned:
    """Finding #3: unlike ``float`` (banned outright because its repr is
    unstable across platforms/interpreter versions), ``int`` repr is exact and
    stable, so it is not banned -- schema fields like ``rotation`` and
    ``TableCellLocator.row``/``col`` legitimately serialize to int. What is real
    is unbounded magnitude (arbitrary-precision Python ints), so int is bounded
    by digit count instead."""

    def test_ordinary_magnitude_int_is_allowed(self) -> None:
        canonical_json_bytes({"rotation": 270, "row": 12})

    def test_int_at_the_digit_bound_is_allowed(self) -> None:
        value = 10**999  # exactly 1000 digits
        canonical_json_bytes({"n": value})

    def test_int_past_the_digit_bound_raises_canonical_json_error(self) -> None:
        value = 10**1000  # 1001 digits
        with pytest.raises(CanonicalJsonError, match="magnitude"):
            canonical_json_bytes({"n": value})

    def test_bool_remains_exempt_from_the_int_magnitude_bound(self) -> None:
        # bool is an int subclass in Python; confirms the magnitude check does
        # not misfire on it.
        canonical_json_bytes({"flag": True})


class TestStoreDatasetClosesTheCollisionCheckToWriteRace:
    """Finding #4: the old ``path.exists()`` check followed by a separate write
    left a TOCTOU window -- another writer (or a hand-placed file) landing
    between the check and the write was silently overwritten. The fix publishes
    via an atomic exclusive create (``os.link``) so a file that is already
    there when the write is attempted is detected by the write itself, not by
    an earlier, racy check."""

    def test_raises_instead_of_overwriting_a_pre_existing_file_at_the_target_path(self, tmp_path: Path) -> None:
        payload = {"compound": "toluene", "k": "1.00"}
        sha = compute_dataset_sha(payload)
        path = dataset_path(tmp_path, sha)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Simulate a concurrent writer (or corruption) landing at the target
        # path before this call's own write is attempted.
        other_bytes = b"not the canonical bytes for this payload"
        path.write_bytes(other_bytes)

        with pytest.raises(ValueError, match="(?i)collision|corrupt"):
            store_dataset(tmp_path, payload)

        # The pre-existing file must be left exactly as it was -- never
        # silently overwritten by the losing side of the race.
        assert path.read_bytes() == other_bytes


class TestCanonicalJsonBytesBoundsRecursionDepth:
    """Finding #5: both this module's own recursive validator and ``json.dumps``
    itself recurse on nested structures with no bound, so a sufficiently deep
    payload raised a bare ``RecursionError`` instead of the documented
    ``CanonicalJsonError`` contract -- and was a cheap way to make any caller
    crash. ``_MAX_JSON_DEPTH`` bounds the validator, chosen low enough that
    ``json.dumps`` cannot then blow the interpreter's recursion limit on a
    payload that passed validation (verified below, not just assumed)."""

    @staticmethod
    def _nested(depth: int) -> dict:
        value: dict = {"v": 1}
        for _ in range(depth):
            value = {"n": value}
        return value

    def test_payload_nested_past_the_limit_raises_canonical_json_error(self) -> None:
        with pytest.raises(CanonicalJsonError, match="500"):
            canonical_json_bytes(self._nested(600))

    def test_payload_at_the_accepted_depth_does_not_hit_the_recursion_limit(self) -> None:
        payload = self._nested(499)

        # canonical_json_bytes runs the validator AND calls json.dumps
        # internally; both must survive without RecursionError at this depth.
        result = canonical_json_bytes(payload)
        assert isinstance(result, bytes)

        # Also exercise json.dumps directly on the same structure, since the
        # requirement being verified is specifically that json.dumps (not just
        # our own validator) does not blow the interpreter's recursion limit on
        # a payload that passed validation at the accepted depth.
        assert isinstance(json.dumps(payload), str)


class TestDecimalReprVersionIsRecordedInTheAddressedPayload:
    """Finding #6: if ``canonical_decimal``'s rendering ever changes, every
    already-stored dataset would silently re-address with no way to tell old
    from new bytes apart. The store now injects a reserved, versioned marker
    key into the payload before hashing/writing it, so the on-disk bytes always
    say which decimal representation they were written under."""

    def test_stored_payload_records_the_decimal_repr_version_key_on_disk(self, tmp_path: Path) -> None:
        stored = store_dataset(tmp_path, {"compound": "toluene", "k": "1.00"})

        on_disk = json.loads(stored.path.read_bytes())
        assert on_disk["_carmel_decimal_repr_version"] == 1

    def test_load_dataset_strips_the_version_key_so_callers_see_their_own_payload(self, tmp_path: Path) -> None:
        payload = {"compound": "toluene", "k": "1.00"}
        stored = store_dataset(tmp_path, payload)

        loaded = load_dataset(tmp_path, stored.sha256)

        assert loaded == payload
        assert "_carmel_decimal_repr_version" not in loaded

    def test_caller_supplied_reserved_namespace_key_is_rejected(self, tmp_path: Path) -> None:
        # The `_carmel_` prefix is a namespace this store owns for its own
        # bookkeeping -- a caller setting a key in it (even accidentally) must
        # never be silently accepted or silently overwritten.
        payload = {"a": "1", "_carmel_decimal_repr_version": 999}

        with pytest.raises(ValueError, match="reserved"):
            store_dataset(tmp_path, payload)

        with pytest.raises(ValueError, match="reserved"):
            compute_dataset_sha(payload)


class TestRawBytesNestingPrescan:
    """Pins the prescan's OWN contract, directly.

    The deep-nesting case in the class below passes even with this prescan
    disabled entirely (measured by mutation): the broadened ``except`` catches
    the resulting ``RecursionError`` and still returns ``False``. That is the
    intended backstop, but it means the end-to-end test cannot tell whether the
    prescan works -- it would go green on a prescan that never fired. The
    prescan is the PRIMARY defence precisely because recovering from a
    ``RecursionError`` is fragile, so its behaviour is pinned here on its own.
    """

    def test_rejects_genuinely_deep_nesting(self) -> None:
        raw = b"[" * (_MAX_JSON_DEPTH + 1) + b"]" * (_MAX_JSON_DEPTH + 1)
        assert _raw_bytes_nest_too_deeply(raw) is True

    def test_accepts_nesting_at_the_limit(self) -> None:
        raw = b"[" * _MAX_JSON_DEPTH + b"]" * _MAX_JSON_DEPTH
        assert _raw_bytes_nest_too_deeply(raw) is False

    def test_brackets_inside_a_string_are_not_structural(self) -> None:
        # The whole reason this is a scanner and not a byte count: 50 000 '['
        # characters inside a JSON string value nest nothing. A naive counter
        # would reject this perfectly ordinary payload.
        raw = b'{"s":"' + b"[" * 50_000 + b'"}'
        assert _raw_bytes_nest_too_deeply(raw) is False

    def test_escaped_quote_does_not_end_the_string(self) -> None:
        # A backslash-escaped quote must not be read as closing the string --
        # otherwise every bracket after it would be miscounted as structural.
        raw = b'{"s":"a\\"' + b"[" * (_MAX_JSON_DEPTH + 1) + b'"}'
        assert _raw_bytes_nest_too_deeply(raw) is False


class TestVerifyDatasetFailsClosedOnHostileButHashCorrectBytes:
    """A file's bytes can hash to their own filename and still be adversarial
    on-disk content chosen to blow up whatever parses them next. Before this
    fix, three such hash-correct payloads made ``verify_dataset`` *raise*
    (``UnicodeEncodeError``, ``ValueError`` from CPython's int/str conversion
    digit limit, or ``RecursionError``) instead of returning ``False`` --
    turning a boolean health check into a crash any hostile file on disk could
    trigger. ``verify_dataset``'s job is exactly to say "no" to bytes like
    this, not to propagate whatever exception parsing them happens to throw."""

    @staticmethod
    def _write_raw(tmp_path: Path, raw_bytes: bytes) -> str:
        sha = hashlib.sha256(raw_bytes).hexdigest()
        path = dataset_path(tmp_path, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
        return sha

    def test_lone_utf16_surrogate_in_a_string_value_returns_false(self, tmp_path: Path) -> None:
        # A lone (unpaired) UTF-16 surrogate is representable in a Python str
        # (json.loads happily decodes a `\ud800`-style escape into one) but
        # cannot be re-encoded to UTF-8 -- `str.encode("utf-8")` raises
        # `UnicodeEncodeError` on it. Hand-craft raw bytes containing the JSON
        # *escape* for a lone surrogate so the bytes themselves stay valid
        # UTF-8 all the way through `path.read_bytes()` and `json.loads`; the
        # failure only surfaces when this store tries to re-canonicalize the
        # parsed string back to bytes.
        raw_bytes = b'{"a": "\\ud800"}\n'
        sha = self._write_raw(tmp_path, raw_bytes)

        assert verify_dataset(tmp_path, sha) is False

    def test_five_thousand_digit_int_value_returns_false(self, tmp_path: Path) -> None:
        # CPython bounds int<->str conversion by digit count (see
        # `sys.set_int_max_str_digits`); parsing a 5000-digit integer literal
        # out of JSON text trips that limit and raises `ValueError`, not
        # anything specific to this module.
        huge_digits = "9" * 5000
        raw_bytes = f'{{"a": {huge_digits}}}\n'.encode()
        sha = self._write_raw(tmp_path, raw_bytes)

        assert verify_dataset(tmp_path, sha) is False

    def test_hundred_thousand_deep_nested_array_returns_false(self, tmp_path: Path) -> None:
        # Bracket-only nesting with no dict wrapper: `_raw_bytes_nest_too_deeply`
        # must catch this via the raw-byte prescan run *before* `json.loads`,
        # since letting `json.loads` itself hit the interpreter's recursion
        # limit on 100 000 levels of `[` is exactly the crash this prescan
        # exists to avoid.
        depth = 100_000
        raw_bytes = (b"[" * depth) + b"0" + (b"]" * depth) + b"\n"
        sha = self._write_raw(tmp_path, raw_bytes)

        assert verify_dataset(tmp_path, sha) is False


class TestLoadDatasetRejectsNonDictPayloadAsValueError:
    """``verify_dataset`` has always required the parsed payload to be a
    ``dict``, but ``load_dataset`` lacked that guard: a canonical, hash-correct,
    non-dict payload (e.g. a bare canonical string) sailed through the digest
    and canonicality checks and only failed later, on ``.pop(...)``, with a
    bare ``AttributeError`` -- not the ``ValueError`` this function documents
    for every other rejection. Now both functions share one dict-checking
    read path, so ``load_dataset`` fails the same documented way."""

    def test_canonical_non_dict_payload_raises_value_error_not_attribute_error(self, tmp_path: Path) -> None:
        raw_bytes = canonical_json_bytes("hello")
        sha = hashlib.sha256(raw_bytes).hexdigest()
        path = dataset_path(tmp_path, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)

        with pytest.raises(ValueError, match="(?i)dict"):
            load_dataset(tmp_path, sha)


class TestCanonicalDecimalBoundsCoefficientLength:
    """``_MAX_ADJUSTED_EXPONENT_MAGNITUDE`` bounds the *exponent* (order of
    magnitude) of a decimal value, but nothing bounded the *coefficient*
    (significant-digit count) -- an all-nines fractional tail like
    ``"0." + "9" * 500000`` has an adjusted exponent of only -1, so it sails
    straight past the exponent bound and was canonicalized into a
    500 002-character string. (The similar-looking integer shape
    ``"9" * 500000`` is already rejected -- but only incidentally, via the
    *exponent* bound, since an all-nines integer's adjusted exponent equals its
    digit count; that shape is not evidence a coefficient bound exists, which
    is why the fractional-tail shape is the one that must be tested here.)"""

    def test_long_fractional_tail_with_small_exponent_is_rejected(self) -> None:
        text = "0." + "9" * 500_000

        with pytest.raises(CanonicalDecimalError, match="(?i)coefficient|too long"):
            canonical_decimal(text)


class TestDatasetDecimalReprVersionAccessor:
    """Finding #4: callers need to know which decimal-repr version a stored
    dataset was written under without changing ``load_dataset``'s return type
    (which must keep returning exactly the caller's own payload, version key
    stripped). ``dataset_decimal_repr_version`` exposes that marker directly."""

    def test_returns_the_version_recorded_at_store_time(self, tmp_path: Path) -> None:
        stored = store_dataset(tmp_path, {"compound": "toluene", "k": "1.00"})

        assert dataset_decimal_repr_version(tmp_path, stored.sha256) == 1

    def test_raises_file_not_found_for_a_sha_that_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            dataset_decimal_repr_version(tmp_path, "0" * 64)
