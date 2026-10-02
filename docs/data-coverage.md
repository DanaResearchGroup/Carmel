# Curated-data acceptance coverage

Generated 2026-10-02 by `carmel data coverage` against the pinned source bytes.
Both lanes are CC-BY-4.0.

| Source | Observable | Files mapped | Points mapped | Refusals | Replay |
|---|---:|---:|---:|---|---:|
| ChemKED | ignition delay | 238 | 1,431 | `incomplete_record=15`, `rcm_pre_compression_conditions=76`, `schema_rejected=12`, `unmapped_unit=10` | 238/0 |
| ReSpecTh | ignition delay | 105 | 1,278 | `rcm_pre_compression_conditions=85`, `unmapped_apparatus=3`, `unmapped_ignition_definition=4`, `unmapped_property=17`, `unmapped_unit=26` | 105/0 |
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
- `rcm_pre_compression_conditions`: the RCM values are before compression; lift it with RCM volume-history simulation in T3, which is in progress.
- `schema_rejected`: the mapped value does not satisfy the lane schema; lift it by adding an explicit schema mapping that preserves the source evidence.
- `unmapped_unit`: the unit is outside the pinned conversion table; lift it by adding a reviewed conversion-table entry or leave it refused.
- `unmapped_apparatus`: the apparatus/mode combination is not in the explicit device map; lift it by adding a reviewed mapping for that combination.
- `unmapped_ignition_definition`: the ignition target or criterion is not in the supported vocabulary; lift it by specifying and implementing a replayable definition.
- `unmapped_property`: the record contains a property the lane does not model; lift it by adding a replayable field and its source locator.
- `unidentified_species`: a species column cannot be resolved to a unique species identity; lift it by supplying an unambiguous species identifier.
- `unmapped_experiment_type`: the experiment type has no supported parser; lift it by implementing a parser and replay contract for that type.

The machine-readable source for this document is `carmel data coverage --json`; the command exits nonzero if any mapped record fails replay.
