"""Tests for carmel.services.dataset_store: canonical JSON, decimal canonicalization,
and content-addressed storage for literature dataset payloads."""

from __future__ import annotations

from pathlib import Path

import pytest

from carmel.services.dataset_store import (
    DATASET_STORE_DIR,
    CanonicalDecimalError,
    canonical_decimal,
    canonical_json_bytes,
    compute_dataset_sha,
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
