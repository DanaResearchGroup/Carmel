# Curated-data acceptance coverage

Generated 2026-10-04 by `carmel data coverage` against the pinned source bytes.
Both lanes are CC-BY-4.0.

| Source | Observable | Files mapped | Points mapped | Refusals | Replay |
|---|---:|---:|---:|---|---:|
| ChemKED | ignition delay | 314 | 1,507 | `incomplete_record=15`, `schema_rejected=12`, `unmapped_unit=10` | 314/0 |
| ChemKED | ignition delay, RCM subset | 172 | 177 | `incomplete_record=15`, `unmapped_unit=10` | 172/0 |
| ReSpecTh | ignition delay | 216 | 1,927 | `unmapped_apparatus=3`, `unmapped_ignition_definition=4`, `unmapped_property=17` | 216/0 |
| ReSpecTh | ignition delay, RCM subset | 122 | 629 | `unmapped_property=2` | 122/0 |
| ReSpecTh | laminar flame speed | 286 | 2,719 | `unmapped_unit=1` | 286/0 |
| ReSpecTh | speciation | 84 | 1,248 | `unidentified_species=2`, `unmapped_experiment_type=3`, `unmapped_property=2` | 84/0 |
| ReSpecTh | unclassified | 0 | 0 | none | 0/0 |

Every mapped record replayed successfully against the downloaded, pinned bytes.

## Frozen source identities

- ChemKED: repository `pr-omethe-us/ChemKED-database`, commit
  `606005bfc8f5214b3f0b5ca7300a96a82815c2ae`; manifest SHA-256
  `826fb3dd4ec5798bd1be7c8f1938f5972f67a07d57383fdb013e215ede5e9444`.
- ReSpecTh: OSF source `10.17605/OSF.IO/NBMZV`, versions pinned at v1:
  `H2_indirect_v2_3.zip` SHA-256
  `7f7247f0c95dfe65cf784807b6fde5539d34f3fce43b47a779d4ad288b96ddda` and
  `syngas_indirect_v2_3.zip` SHA-256
  `a14628e5300a7922f50bb354b95f46f1ae853eb6b31e2a33555ffab17812321b`;
  manifest SHA-256
  `d268155744c0a19503249c9fd4f1e5c64dad974ccd484bb7a7f0daaabd3882a5`.

## Refusal reasons

- `incomplete_record`: required source fields are absent; lift it by adding or recovering those fields in the source record.
- `history_nonmonotone_time`, `history_nonpositive_volume`, `history_no_compression`, `history_missing_initial_state`, `history_invalid`: an unusable compression history or initial state; each carries its source-specific detail.
- `rcm_thermo_unavailable`: derived labels require the supported seven-species NASA7 mixture. Initial, trial and solved temperatures must lie in the material species' admissible intersection: H2/O2/CO/CO2/H2O require 200–3500 K; N2/Ar's published range is 300–5000 K, with an operator-approved 5 K low-temperature allowance down to 295 K. No other species or upper bound receives a tolerance. Exact-zero components do not constrain that intersection. Exact source Kelvin admission precedes float conversion.
- `implausible_ignition_temperature`: stated or derived ignition conditions below the 500 K backstop; initial pre-compression temperatures are not ignition labels.
- `schema_rejected`: the mapped value does not satisfy the lane schema; lift it by adding an explicit schema mapping that preserves the source evidence.
- `unmapped_unit`: the unit is outside the pinned conversion table, or a native value cannot be bound or converted under the schema and canonical numeric bounds. Unknown units need a reviewed table entry; unrepresentable values remain refused.
- `unmapped_apparatus`: the apparatus/mode combination is not in the explicit device map; lift it by adding a reviewed mapping for that combination.
- `unmapped_ignition_definition`: the ignition target or criterion is not in the supported vocabulary; lift it by specifying and implementing a replayable definition.
- `unmapped_property`: the record contains a property the lane does not model; lift it by adding a replayable field and its source locator.
- `unidentified_species`: a species column cannot be resolved to a unique species identity; lift it by supplying an unambiguous species identifier.
- `unmapped_experiment_type`: the experiment type has no supported parser; lift it by implementing a parser and replay contract for that type.

The machine-readable source for this document is `carmel data coverage --json`; the command exits nonzero if any mapped record fails replay.

## RCM change from the 2026-10-02 baseline

ChemKED compressed-temperature, compressed-pressure and compression-time require
an accompanying volume-history. The native record pairs each history with its
initial state and optional compressed labels; it does not map those labels
independently. Orphan fields and optional quantities without a string value/unit
pair refuse as `history_invalid`. YAML floating scalars retain their source
decimal lexemes. History ordering, EOC selection/interpolation, compression
admission and source slope checks use exact base-unit Decimals; float views are
only numerical inputs/outputs and downstream representability checks. ReSpecTh
also validates time values, ordering and units on expanding post-compression
traces, and refuses a declared volume history on a non-RCM apparatus.

ChemKED now maps all 76 formerly refused compression-history files/points,
carrying their stated compressed labels. ReSpecTh maps all 109 compression-history
records (616 points) and retains thirteen postcompression records. The 24 Torr
history records convert correctly through V5 and their stated initial
temperatures of 296.8–298.3 K lie within the operator-approved N2/Ar band down to
295 K. Their derived state basis records `thermo=extrapolated-below-300K`, which
replay verifies and export carries in `source.record`, including when derived
labels are omitted. This restores 24 files and 24 points in both ReSpecTh
ignition-delay rows from the previous census (192/1,903 to 216/1,927; RCM 98/605
to 122/629). Every other census count is unchanged and no mapped record failed
replay. The one named tolerance, `N2_AR_LOW_T_EXTRAPOLATION_K`, is justified by
the nearly flat N2/Ar low-T polynomials over a few kelvin; coefficients and
published ranges stay pinned. V5 also admits two non-RCM Torr files (33 points).

See [RCM export](rcm-t3-export.md) for selection, provenance and typed export
refusals. Mapping coverage counts differ from export acceptance: the T3 schema
requires a stricter mole-fraction sum than some source records provide.
