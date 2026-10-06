# Curated IDT export for T3

Install the `agents` extra to resolve source-stated SMILES/InChI with RDKit.
Export uses pinned source bytes and verifies every mapped record by replay.
Parser refusals are counted in the report. A replay mismatch aborts the entire
load and the CLI exits nonzero without writing an export or report. ReSpecTh
parsing and replay share the mixture, linked-history and derived-temperature
admission checks.

```sh
carmel data export-t3 --source chemked --fuel n-heptane --output n-heptane.yaml
carmel data export-t3 --source all --history-only --output rcm.yaml --offline
carmel data export-t3 --source all --history-only --derived-labels --output rcm-labelled.yaml --offline
carmel data export-t3 --source respecth --study x40000001.xml --study x40000002_x.xml --output split.yaml
```

`--study` is repeatable and matches an exact file/member path, basename or DOI.
An unknown requested study fails the command. `--fuel` matches a source fuel name
or a ChemKED fuel directory. A sibling `.report.json` gives exported points,
stated/derived/unlabelled counts, per-point typed refusals and source mapping
refusals. Source fractions and ignition definitions are preserved; a missing or
conflicting chemical identity, incompatible definition or invalid mole-fraction
sum is refused. Exact decimal totals in the newly admitted rounding band
`1e-6 < |total - 1| <= 1e-5` are normalized for both export and thermodynamic
derivation. Each affected point records `source.composition.source_total` and
`source.composition.renormalized`; totals farther from unity remain refused.
The pinned-corpus survey found 30 ChemKED point compositions and 12 ReSpecTh
record compositions at the `1e-5` band, while the next band begins at `1e-4`.
Previously admitted totals within `1e-6` retain their source representation for
byte compatibility. The independent T3 float-sum check remains in force.
Nonzero source fractions that underflow float representation are typed
composition refusals; exact zeros carry no material and are omitted.

An identifier matching the InChIKey shape is resolved only through the pinned
offline `carmel/data/inchikey_to_inchi.json` table. RDKit re-derives every key
from its InChI when the table loads; a mismatched table refuses to load and an
unknown key remains `no_confident_smiles`. A successful lookup is recorded in
the point's `source.identity_lookups` provenance.

Compression histories retain all source samples, their source time axis and the
initial temperature/pressure. End of compression uses the source's explicit
compression time when present, otherwise the first minimum-volume sample, with
that derivation recorded in `source.record`. Source file SHA-256 and archive pin
are also included. Histories must have increasing finite times, positive volumes
and a compression phase. Existing ReSpecTh postcompression expansion traces stay
on their stated, history-free ignition-condition path.

ChemKED compressed temperature/pressure labels are carried as stated. ReSpecTh
compression records carry initial states and histories, and omit derived
end-of-compression temperature/pressure labels by default. `--derived-labels`
(or `include_derived_labels=True` in the export API) opts in to labels marked
`;eoc=derived-isentropic`. Stated labels remain present in either mode.
The pure-Python ideal-gas entropy solver evaluates
NASA7 temperature-dependent heat capacities for H2, O2, N2, AR, CO, CO2 and H2O.
H2, O2, CO, CO2 and H2O support 200–3500 K; N2 and Ar support 300–5000 K.
The operator-approved `N2_AR_LOW_T_EXTRAPOLATION_K = 5` admits initial N2/Ar
temperatures down to 295 K: their low-T polynomials are nearly flat over a few
kelvin. This is the only extrapolation allowance; no other species or upper
bound receives a tolerance. Every initial, trial and solved temperature must
lie in the resulting intersection for species with nonzero material, otherwise
the record refuses as `rcm_thermo_unavailable`. Source initial temperatures are
checked as exact base-unit Decimals before float conversion.

A derived state that uses the 295–300 K band carries
`derived-isentropic;thermo=extrapolated-below-300K` in its `eoc_basis`. Replay
re-derives that marker from source bytes, mixture and exact initial temperature.
T3's `source.record` carries `;eoc=derived-isentropic;thermo=extrapolated-below-300K`
even when derived labels are omitted. ChemKED derived-label exports also mark
this band when their solve uses it. Thermodynamics
normalizes independently rounded source ratios within 0.005, matching Cantera's
TPX treatment; export still enforces T3's 1e-6 sum tolerance. Derived values are
excluded from queries requiring stated ignition conditions.

Derived ignition labels must meet the same 500 K plausibility backstop as
ReSpecTh's stated ignition conditions. Weak compression below that limit is
refused with `implausible_ignition_temperature`. ChemKED histories may still
export without derived labels. ChemKED identities come from each point's own
source row; ReSpecTh identities remain scoped to the file's common composition.

Coefficients come from GRI-Mech 3.0 in [Cantera 3.2.0's gri30.yaml](https://raw.githubusercontent.com/Cantera/cantera/v3.2.0/data/gri30.yaml),
SHA-256 `06650b1e0ee0012f6903d5328b1bb218cb6007d07f8ebe375d18f24811039345`.
Carmel requires no Cantera dependency. Both export modes validate with T3's
`ExperimentalIDTFile` at commit `4043bb9` (merged T3 #222), which accepts
history points without end-of-compression labels.

Database measurements bind conversion table V5. Torr uses the exact rational
factor `101325/760`, with Decimal working precision 4096 and half-even rounding,
preserving source significant digits in the rounded value. Tables V1–V4 retain
their published serialized identities.
