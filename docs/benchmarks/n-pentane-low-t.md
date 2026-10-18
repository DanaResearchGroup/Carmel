# Frozen n-pentane low-temperature benchmark

This benchmark asks whether a mechanism-revision process can recover four structural
reaction types added between the frozen `nc5_49` and July-2015 C5 files. Structural
recovery is primary. Holdout jet-stirred-reactor (JSR) speciation error is secondary
and is declared `requires: jsr-adapter`; Carmel can validate and load the contract
without that adapter.

The contract uses the same frozen-model, pin-verification, species, invariant-validator
and structural-rule implementation as the n-heptane case. Both cases validate through
the discriminated shared schema in `carmel.benchmarks.schema`; the C5 model adds the
ReSpecTh whole-study split and its acceptance ledger. The n-heptane contract and
generated case schema are not changed.

## Mechanism pins and identity

| Endpoint | Source | Retrieved | SHA-256 | Unique species / reaction entries |
|---|---|---|---|---:|
| v_old | [NUIG `nc5_49`](https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nc5_49_mech.dat) | 2026-10-14 | `aae8ebbe144b081a52aae08efb14740d8b90cbda59d6e300f5cb34acf41614bd` | 293 / 1,593 |
| v_new | [NUIG July-2015 C5](https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nuigmech_c5_july2015_1215.chem) | 2026-10-14 | `b54aa3569d8f3796a242402b08c4395381fced58f6edcf1fc060b0ac3446be48` | 697 / 3,214 |

`Pin.verify()` checks byte count and SHA-256 and raises `PIN CHANGED` before parsing
changed bytes. Mechanism bytes are not committed.

The v_new bytes are a post-2015 maintained file, not a publication-era snapshot. The
maintenance markers and adjacent reaction edits are:

| v_new marker line | Bytes show | Classification |
|---|---|---|
| 1568, `!DONG 15/12/2018` | The two `C2H3OO=CO+CH3O` PLOG/DUP pairs and the two `C2H3OO=CO2+CH3` pairs are retained as comments, while each pair is folded into one active PLOG/DUP representation. | Rate representation change; same reactions, not an added or removed reaction type. |
| 1646, `!DONG 15/12/2018` | A second marker for the same 2018 maintenance block; no separate reaction edit is attached directly to this marker. | Maintenance marker only. |
| 5812, `!CZ 27/01/2014` | `C3H5-T+CH3(+M)<=>IC4H8(+M)` and `IC4H7+H(+M)<=>IC4H8(+M)` are commented out, with their LOW/TROE data; no active copy of either equation follows. | Removed reaction type, not a rate re-expression. |

The marker inventory does not touch any G1–G4 member. The contract reproducer resolves
all 30/30, 30/30, 12/12 and 12/12 members and verifies their structural absence from
v_old; the marked reactions are C2/C3/C4 reactions, whereas G1–G4 are the listed C5
OOQOOH/P(OOH)2 channels.

v_old is also **UNVERIFIED**, but has no dated or initialled maintenance markers and no
commented-out reaction lines adjacent to active reactions. Its `!REV/` lines are reverse
rate data, not provenance markers. Bugler et al. 2015 section 2 names Healy's model as
the starting mechanism, but neither paper nor either endpoint authenticates these exact
SHA-256 values as publication-era bytes.

Snapshot search (2026-10-18): the [current NUIG mechanism-download page](https://nuigalway.ie/combustionchemistrycentre/mechanismdownloads/)
links the current July-2015 C5 file and the [Bugler et al. 2015 paper](https://doi.org/10.1021/acs.jpca.5b00837),
but supplies no dated byte snapshot or immutable release identifier. The [Bugler et al.
2016 page's supplementary material](https://www.universityofgalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/c5_15_supplementary_material.pdf)
contains figures and experimental/model data, not an earlier mechanism copy. The [Bugler
et al. 2016 mechanism link](https://www.universityofgalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nuigmech_c5_july2015_1215.chem)
resolves to the same current endpoint. The Bugler et al. 2017 JSR paper page was checked
for supplementary mechanism material; none was located. Wayback CDX checks for both
mechanism URLs for 2015–2017 returned an archive-service internal error, so no candidate
2015 re-pin was found. A genuine snapshot would require a byte-preserved 2015–2017 copy
from the paper supplement, NUIG release archive, or a successful Wayback capture, followed
by operator review and a new SHA-256 pin; none is available here.

## Mapped ground truth

Bugler et al. 2015, DOI `10.1021/acs.jpca.5b00837`, sections 2.2.3–2.2.6 and
Figs. 13–15 define the four scored additions. Korcek chemistry is excluded because
section 2.2.6 says it was not included. Thermochemistry and rate-rule changes are also
excluded.

| Type | Members | Resolved | Resolved species | Example |
|---|---:|---:|---:|---|
| G1: alternative OOQOOH H-shift to P(OOH)2 | 30 | 30 | 26 / 26 | `C5H10OOH1-2O2 ⇌ C5H93-1,2OOH` |
| G2: P(OOH)2 to hydroperoxy cyclic ether + OH | 30 | 30 | 33 / 33 | `C5H91-2,3OOH ⇌ C5H9O1-2OOH-3 + OH` |
| G3: P(OOH)2 to olefinic hydroperoxide + HO2 | 12 | 12 | 18 / 18 | `C5H91-2,3OOH ⇌ C5H91-3OOH + HO2` |
| G4: direct OOQOOH concerted HO2 elimination | 12 | 12 | 16 / 16 | `C5H10OOH1-2O2 ⇌ C5H92-1OOH + HO2` |

The tested label grammar builds straight-chain molecular graphs for `C5H10OOHi-jO2`,
`C5H9r-i,jOOH`, `C5H9d-iOOH`, and `C5H9Oi-jOOH-k`. Positions name the hydroperoxide,
peroxy radical, carbon radical, double bond or ether-ring closure encoded by the label.
`N?CnKETij` is reused for conventional ketohydroperoxide negatives. Unknown, impossible
or conflicting labels are unresolved rather than guessed.

Every mapped member is a v_new entry accepted by the shared graph classifier with
`carbon_count=5`. Absence is established against v_old by canonical, direction-free
reaction signatures plus an exhaustive elemental-formula profile scan in both reaction
orientations. Because `nc5_49` has no THERMO block, its condensed NUIG labels supply an
independent formula inventory; the reproduction accepts it only after all 1,593 baseline
reactions balance. The old conventional OOQOOH-to-ketohydroperoxide channels are mapped and
rejected as negatives. All 293 old species are inventoried: 31 structures resolve under
the tested grammar and the other 262 are explicitly `UNRESOLVED`, never omitted or
guessed, but all 293 retain their elemental formulas. Every old reaction whose formula
profile could match a scored member must be fully
structurally resolved before the reproduction can pass. Exact-label absence alone is
never the test.

## Frozen whole-study split

The ReSpecTh OSF mirror archive `pentane_indirect_v2.5.zip` is pinned at version 1,
242,617 bytes, SHA-256
`b34d6e78a0600bffba9bf7f04f59d9479d6d1314df9964e4071902c929a65e56`.
Each selected member has its own SHA-256 in the contract. `load_split()` reverifies the
archive and opens the ZIP from those exact verified bytes; it never reopens the path.
The reproducer derives the split directly from each XML bibliography DOI and its
`NC5H12` composition, and counts response cells from the pinned archive, so no external
census file is an input.

| Set | Study | DOI | Files | Rows | Observations | 600–900 K |
|---|---|---|---:|---:|---:|---:|
| Development | Bugler et al. 2016 n-pentane ignition delay | `10.1016/j.combustflame.2015.09.014` | 26 | 226 | 226 IDT | 5 IDT |
| Holdout | Bugler et al. 2017 n-pentane JSR | `10.1016/j.proci.2016.05.048` | 6 | 144 | 2,080 scalar | 72 rows / 1,040 scalar |

The legacy v2.5 records omit `fileDOI`. Carmel now applies the documented ReSpecTh
member-filename convention (`x00011003.xml` → `10.24388/x00011003`) with an explicit
derived basis. The fallback accepts only `[xg]` followed by exactly eight digits and
`.xml`; suffix variants are refused as ambiguous. Replay re-derives this DOI from the
record's pinned member path. This is the only reader generalization needed for the JSR study.

## Data acceptance

No refused point is patched into the contract. All six holdout files are supported.
Development admits 18 files / 177 points. Four RCM files / 31 points remain
`rcm_thermo_unavailable` because at least one point starts below the unchanged 295 K
mixture-domain boundary, and `x40011033.xml` remains refused because point 4 repeats a
time with a different volume. Exact duplicate rows in `x40011041.xml` and
`x40011042.xml` are collapsed with provenance. The three `d/dt second max` files remain
unsupported because T3's existing `ExperimentalIDTFile` vocabulary cannot express that
criterion.

| File | Set | Rows | Observations | Reader result | Reason |
|---|---|---:|---:|---|---|
| `x10011004.xml` | development | 8 | 8 | supported | — |
| `x10011005.xml` | development | 17 | 17 | supported | — |
| `x10011006.xml` | development | 6 | 6 | supported | — |
| `x10011007.xml` | development | 9 | 9 | supported | — |
| `x10011008.xml` | development | 11 | 11 | supported | — |
| `x10011009.xml` | development | 11 | 11 | supported | — |
| `x10011010.xml` | development | 8 | 8 | supported | — |
| `x10011011.xml` | development | 17 | 17 | supported | — |
| `x10011012.xml` | development | 25 | 25 | supported | — |
| `x40011029.xml` | development | 12 | 12 | supported | — |
| `x40011030.xml` | development | 11 | 11 | refused | `rcm_thermo_unavailable` |
| `x40011031.xml` | development | 5 | 5 | refused | `rcm_thermo_unavailable` |
| `x40011032.xml` | development | 5 | 5 | refused | `rcm_thermo_unavailable` |
| `x40011033.xml` | development | 6 | 6 | refused | `history_nonmonotone_time` |
| `x40011034.xml` | development | 2 | 2 | supported | — |
| `x40011035.xml` | development | 10 | 10 | supported | — |
| `x40011036.xml` | development | 3 | 3 | supported | — |
| `x40011037.xml` | development | 6 | 6 | supported | — |
| `x40011038.xml` | development | 10 | 10 | refused | `rcm_thermo_unavailable` |
| `x40011039.xml` | development | 9 | 9 | supported | — |
| `x40011040.xml` | development | 8 | 8 | supported | — |
| `x40011041.xml` | development | 9 | 9 | supported | — |
| `x40011042.xml` | development | 6 | 6 | supported | — |
| `x40011043.xml` | development | 3 | 3 | refused | `unmapped_ignition_definition` |
| `x40011044.xml` | development | 4 | 4 | refused | `unmapped_ignition_definition` |
| `x40011045.xml` | development | 5 | 5 | refused | `unmapped_ignition_definition` |
| `x00011003.xml` | holdout | 20 | 280 | supported | — |
| `x00011004.xml` | holdout | 20 | 280 | supported | — |
| `x00011005.xml` | holdout | 20 | 260 | supported | — |
| `x00011006.xml` | holdout | 29 | 435 | supported | — |
| `x00011007.xml` | holdout | 28 | 420 | supported | — |
| `x00011008.xml` | holdout | 27 | 405 | supported | — |

## Scoring and leakage limits

The primary score follows n-heptane: classify the generated baseline first; a type it
already contains gets baseline coverage and zero revision credit. Each newly recovered
type earns one unit once, for a denominator of four.

For secondary JSR error, define `L_s` per species as its stated detection limit, or half
its smallest strictly positive frozen observation when no limit is stated. Clamp measured
and simulated values below `L_s` to `L_s`. The per-species score is the mean over rows of
`abs(log10(max(x_sim,L_s) / max(x_exp,L_s)))`; the overall score is an unweighted mean
over species. Report every species and row count, and report the 600–900 K subset
separately. A species with neither a positive observation nor stated limit is refused.
Missing/nonfinite/negative values and solver failures are also explicit refusals.

The ignition study validated v_new and therefore stays in development. The JSR study
shares mechanism-community authors, so this is search-time separation, not historical
blindness. Keep v_new bytes, mapped truth and labels away from the revision process until
candidate locking. ChemKED's NOx-doped points are excluded.

## Reproduction

From the repository root, using the required environment:

```bash
/home/alon/anaconda3/envs/crml_env/bin/python \
  benchmarks/n-pentane-low-t/reproduce.py \
  --v-old /home/alon/runs/i116-alkane-census/evidence/nc5-49.txt \
  --v-new /home/alon/runs/i116-alkane-census/evidence/c5-2015.txt \
  --archive /home/alon/runs/i116-alkane-census/evidence/pentane_indirect_v2.5.zip
```

The script reparses both untouched mechanisms, rebuilds all member graphs and structural
signatures, compares the result with the committed contract, verifies archive/member
hashes, reruns all 32 selected XML files through Carmel, and reproduces the frozen totals.
It runs no chemical simulation.
