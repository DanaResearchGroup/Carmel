# Frozen iso-pentane low-temperature benchmark

This third case transfers the four Bugler et al. 2015 reaction classes from n-pentane
to iso-pentane's 2-methylbutane skeleton. Structural recovery is primary; the frozen
whole-study JSR error is secondary. It shares the validator, class rules, archive
loader and scoring definition with the first two cases. No kinetics simulation is
part of this contract reproduction.

## Mechanism and supporting pins

Every input is pinned by source URL, retrieval date, exact byte count and SHA-256 in
`benchmarks/iso-pentane-low-t/contract.json`. Reproduction refuses changed bytes.
Full third-party mechanism, thermo, glossary and observation files are not committed.

| Input | Retrieved | Bytes | Unique species / source reaction entries | Raw SHA-256 |
|---|---|---:|---:|---|
| v_old [ic8_ver3_mech.txt](https://combustion.llnl.gov/sites/combustion/files/ic8_ver3_mech.txt) | 2026-10-02 | 357350 | 874 / 3796 | `2fc57d65d8dd7b4e9793f09639f41aca114aa878624f4b93b726ba2f39a68d5f` |
| v_old [prf_v3_therm_dat.txt](https://combustion.llnl.gov/sites/combustion/files/prf_v3_therm_dat.txt) | 2026-11-03 | 488769 | 1406 / 0 | `acb98b7574d9568714b928444ebea8aae86e794168d1d597297dc2c00ff5ae4e` |
| v_new [nuigmech_c5_july2015_1215.chem](https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nuigmech_c5_july2015_1215.chem) | 2026-10-14 | 488335 | 697 / 3214 | `b54aa3569d8f3796a242402b08c4395381fced58f6edcf1fc060b0ac3446be48` |
| v_new [nuigmech_c5_july2015.therm](https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/nuigmech_c5_july2015.therm) | 2026-10-17 | 243515 | 733 / 0 | `c8e465ea61c061f466352b2e6a84c2d6854f255d33e7e1363aa4cd0a8d3ea9fa` |

The [LLNL iso-octane v3 page](https://combustion.llnl.gov/mechanisms/alkanes/iso-octane-version-3)
links `prf_v3_therm_dat.txt` as its thermodynamic companion, alongside `ic8_ver3_mech.txt`.
Its archived evidence is `llnl-iso-page-2026-10-02.html`, lines 760–761, SHA-256
`15c3fe15836b7a1f55460bc6f1034be54317ee4f47310054b224be26899aeb16`. The companion covers **874/874** old mechanism species;
NUIG thermo covers **697/697** new species. Extra thermo species are inventoried and
reported by the converter, not inserted into the mechanism.

**Count convention:** pin CHEMKIN equation/Arrhenius entries, counting each source
equation once. v_old has 3,796 entries and 3,726 explicit REV records: 3,068 nonzero
reverse rates become separate reverse reactions, while 658 zero reverse rates do not.
Thus Cantera 3.2.0 reports **3,796 + 3,068 = 6,864** reactions. This is conversion of
explicit reverse kinetics, not 3,068 chemistry additions. v_new has no REV records
and retains 3,214 entries after conversion.

## Normalization and reviewed warnings

The shared [conversion rules](holdout-simulation.md) apply unchanged: scoped E-less
Fortran exponents, NASA fourth-card indices, `--permissive --verbose`, the warning
allowlist, and domain-exact PLOG trimming. Only five missing fourth-card indices in
PRF thermo change (lines 5905, 5909, 5933, 5957, 5961). The other three inputs are
byte-identical after normalization. All species names, temperature fields and fourteen
coefficient strings are checked independently for preservation.

| Input | Normalized SHA-256 | Rewrites |
|---|---|---:|
| v_old mechanism | `2fc57d65d8dd7b4e9793f09639f41aca114aa878624f4b93b726ba2f39a68d5f` | 0 |
| v_old thermo | `8983fd46446f8d2457ccd19092b7e7624931b648e42dbb6be64a06d1fa9fc0c6` | 5 |
| v_new mechanism | `b54aa3569d8f3796a242402b08c4395381fced58f6edcf1fc060b0ac3446be48` | 0 |
| v_new thermo | `c8e465ea61c061f466352b2e6a84c2d6854f255d33e7e1363aa4cd0a8d3ea9fa` | 0 |

v_new trims the unsound 100-atm PLOG level of `C2H3OO ⇌ CO2 + CH3` using the
shared rule; all four holdout files specify 1 atm, outside the affected high-pressure
interpolation interval. v_old has no PLOG rates. No reaction or coefficient is repaired.

The reviewed NASA registry stores labels, quantity, Tmid, numeric-field hash, source
file/line and review reason. The original OH*/OHV and CH*/CHV h/RT exceptions at
1000 K retain the exact case-2 acceptance behavior and simulation-lock fingerprint.
They are existing reviewed exceptions, independent of the below-Tmid justification.

The PM reviewed **EC6H13O2 s/R at 1388 K**, PRF thermo line 2987: a **0.64% jump
(107.806 versus 107.116)** above the **1100 K** holdout maximum. The high-temperature
polynomial is never used at these isothermal scored conditions. Its numeric-field
SHA-256 (three 10-byte temperature fields, then fourteen 15-byte coefficients) is
`d1d4928748f6538395afde16ef286c6a78bc4cf4994c1f979f2290b1e6efb0b6`.
The loader refuses a maximum at or above every Tmid on which a temperature-dependent
review relies; the archive loader also recomputes the actual per-file maxima from
verified XML bytes. Unknown labels, quantities, midpoints, changed numeric fields,
and other warnings are refused. The generic worker requires an explicit JSR bound
for this exception; it is unavailable to evolving-temperature IDT simulations.

## Historical identity: UNVERIFIED

The old header dates the website release to 16 December 2009, names Mehl, Curran,
Pitz and Westbrook, and explicitly states that it includes Curran’s latest IC5 version.
IC5H12 and its low-temperature submechanism are present. The LLNL page cites 2009
mechanism work. This makes it a documented pre-2015 iso-pentane baseline, but does
not authenticate immutable 2009 bytes or establish direct parentage of NUIG C5.
The same maintained v_new artifact and historical limitations as case 2 apply.

## Mapped ground truth

[Bugler et al. 2015](https://doi.org/10.1021/acs.jpca.5b00837), §§2.2.3–2.2.6,
defines G1–G4. Korcek chemistry, fragmentation, concurrent thermo changes and rate-rule
changes are excluded. The matcher selects the exact 2-methylbutane carbon tree
(five carbons with degrees 1, 1, 1, 2, 3), preserving the original linear-chain default.
It checks elemental balance and the same radical/peroxide/ether/alkene features as
case 2, in either reaction orientation. Conventional KHP production and RO2/QOOH
chemistry, n-pentane, neo-pentane and carbon-free reactions are negatives.

| Type | Members | Resolved entries | Resolved species | Example |
|---|---:|---:|---:|---|
| G1: alternative OOQOOH H-shift to P(OOH)2 | 33 | 33 | 31/31 | `AC5H10OOH-AO2<=>C5H9B-A,AOOH` |
| G2: P(OOH)2 to hydroperoxy cyclic ether + OH | 33 | 33 | 37/37 | `C5H9A-A,BOOH<=>C5H9OA-AOOH-B+OH` |
| G3: P(OOH)2 to olefinic hydroperoxide + HO2 | 14 | 14 | 20/20 | `C5H9A-A,BOOH<=>C5H9A-AOOH+HO2` |
| G4: direct OOQOOH concerted HO2 elimination | 14 | 14 | 19/19 | `AC5H10OOH-AO2<=>C5H9A-AOOH+HO2` |

All member structures come from the pinned author glossary already used by case 1
(SHA-256 `f910c37571ed2242e18909cec72616c9919a151ac4e39cb4ae5ba4babb89e12f`).
Its labels and SMILES are checked against both endpoints’ independent thermo formulas.
The old inventory contains all 874 species: 319 resolved, 555 explicitly UNRESOLVED.
No unresolved structure is guessed. A later glossary supplies structure evidence,
not evidence that a reaction existed in either earlier endpoint.

Absence uses canonical direction-free reaction signatures plus an exhaustive formula
profile screen. In a partially mapped old reaction, a resolved participant can exclude
identity in both orientations, retaining multiplicity (for example a neo-C5 source
cannot match an iC5 source or product). Any remaining formula-compatible and
structure-compatible unresolved candidate stops reproduction. New C5H11O4 sources
are inventoried exhaustively before classifying the necessary G1–G4 formula profiles.

The **33 alternative isomerizations** correct the census estimate of about 36; the
**14 direct eliminations** agree with its estimate. The scored total is **94 entries**.
The rough 66-added/107-dropped label census is not this structural ground truth:
it is neither a target nor a denominator, and dropped reactions earn no credit.

## Frozen split and acceptance

The archive is case 2’s unchanged `pentane_indirect_v2.5.zip` (242,617 bytes), SHA-256
`b34d6e78a0600bffba9bf7f04f59d9479d6d1314df9964e4071902c929a65e56`. Bibliographies and membership are derived
from each XML record, keeping whole studies together.

| Set | XML bibliography | Files | Rows | Scalar observations | 600–900 K rows / observations |
|---|---|---:|---:|---:|---:|
| Development | Bugler et al., Combustion and Flame 163 (2016) 138–156; DOI 10.1016/j.combustflame.2015.09.014 | 20 | 121 | 121 | Reader-refused |
| Holdout | Zhong, Qiu, Li, Huang and Han, Combustion and Flame 235 (2022) 111678; DOI 10.1016/j.combustflame.2021.111678 | 4 | 58 | 812 | 31 / 434 |

The holdout title is *Experimental and kinetic study on the pyrolysis and oxidation
of isopentane in a jet-stirred reactor*. Its XML publication year is 2022; the DOI
contains 2021. It is not Bugler 2015. The whole study includes oxidation files
`x00011009`–`011` and pyrolysis `012`. The observations are digitized in the archive.
All four files pass the existing reader. All 121 development points remain refused:
28 unmapped-property, 75 RCM-thermo-unavailable, and 18 ignition-definition points.
No refused point is repaired or scored. This is a material limitation for a future
development/search run, even though the structural benchmark and holdout are loadable.

| File | Set | Rows | Observations | Reader result | Reason |
|---|---|---:|---:|---|---|
| `x10011001.xml` | development | 9 | 9 | refused | unmapped_property |
| `x10011002.xml` | development | 12 | 12 | refused | unmapped_property |
| `x10011003.xml` | development | 7 | 7 | refused | unmapped_property |
| `x40011001.xml` | development | 3 | 3 | refused | rcm_thermo_unavailable |
| `x40011002.xml` | development | 7 | 7 | refused | rcm_thermo_unavailable |
| `x40011003.xml` | development | 9 | 9 | refused | rcm_thermo_unavailable |
| `x40011004.xml` | development | 9 | 9 | refused | rcm_thermo_unavailable |
| `x40011005.xml` | development | 1 | 1 | refused | rcm_thermo_unavailable |
| `x40011006.xml` | development | 10 | 10 | refused | rcm_thermo_unavailable |
| `x40011007.xml` | development | 2 | 2 | refused | rcm_thermo_unavailable |
| `x40011008.xml` | development | 11 | 11 | refused | rcm_thermo_unavailable |
| `x40011009.xml` | development | 5 | 5 | refused | rcm_thermo_unavailable |
| `x40011010.xml` | development | 7 | 7 | refused | rcm_thermo_unavailable |
| `x40011011.xml` | development | 4 | 4 | refused | rcm_thermo_unavailable |
| `x40011012.xml` | development | 7 | 7 | refused | rcm_thermo_unavailable |
| `x40011013.xml` | development | 5 | 5 | refused | unmapped_ignition_definition |
| `x40011014.xml` | development | 5 | 5 | refused | unmapped_ignition_definition |
| `x40011015.xml` | development | 3 | 3 | refused | unmapped_ignition_definition |
| `x40011016.xml` | development | 3 | 3 | refused | unmapped_ignition_definition |
| `x40011017.xml` | development | 2 | 2 | refused | unmapped_ignition_definition |
| `x00011009.xml` | holdout | 16 | 224 | supported | — |
| `x00011010.xml` | holdout | 12 | 168 | supported | — |
| `x00011011.xml` | holdout | 17 | 238 | supported | — |
| `x00011012.xml` | holdout | 13 | 182 | supported | — |

## Scores and leakage limits

Classify the generated baseline first. Each of G1–G4 present in the baseline earns
baseline coverage and zero revision credit. Recovery is counted once per type, with
a denominator of four; duplicate entries and fitted rates do not increase credit.

For secondary JSR error, define `L_s` per species as its stated detection limit, or half
its smallest strictly positive frozen observation when no limit is stated. Clamp measured
and simulated values below `L_s` to `L_s`. The per-species score is the mean over rows of
`abs(log10(max(x_sim,L_s) / max(x_exp,L_s)))`; the overall score is an unweighted mean
over species. Report every species and row count, and report the 600–900 K subset
separately. A species with neither a positive observation nor stated limit is refused.
Missing/nonfinite/negative values and solver failures are also explicit refusals.

v_new and the four Bugler2015 chemistry classes are shared with case 2; this tests transfer to a branched fuel, not an independent historical chemistry discovery.
The development study shares authors, archive and mechanism validation with case 2. The fuel-specific files are distinct; study identity is shared across cases.
The XML holdout bibliography is Zhong et al., Combustion and Flame 235 (2022) 111678, DOI 10.1016/j.combustflame.2021.111678. These authors differ from Bugler et al. and the study postdates the named 2015 revision, strengthening historical blindness; the maintained v_new bytes are not an authenticated 2015 snapshot, so absolute blindness is not established.
The shared ReSpecTh archive is a common curation channel, not independent evidence of chemistry.
Keep v_new, author glossary, mapped truth and labels out of revision prompts and retrieval until locking; score type recovery once with baseline credit, not rate or thermo tuning.
Korcek chemistry is excluded. Preserve all reader refusals, including all 121 development IDT points; this contract currently provides no reader-supported development observations.
Holdout includes pyrolysis x00011012 and oxidation x00011009-011 as one whole study. Its digitized numeric XML observations retain their provenance; no point is redigitized.

Lock candidate, baseline, contract hash, matcher and simulator before one holdout
evaluation, with no holdout-driven tuning, retries or exclusions. Case 3 tests transfer
to a branched skeleton; it is not statistically independent evidence of a new class
discovery from case 2. The post-v_new holdout and different authors improve the
historical separation, subject to v_new’s unauthenticated publication-era identity.

## Reproduction

```bash
~/anaconda3/envs/crml_env/bin/python benchmarks/iso-pentane-low-t/reproduce.py
```

Run under the dispatch’s systemd resource limits and capture stdout/stderr in
`~/runs/i137-ic5-contract/`. The script verifies raw pins before parsing,
normalizes and converts both endpoints, audits every source-family candidate,
reproduces the explicit members and structural absence, and replays all 24 XML files.
It requires local pinned evidence and Poppler; it downloads nothing and simulates
nothing. `--write-contract` is for deliberate construction; normal invocation compares
the reproduced result to the committed frozen contract. Committed tests need no
network or full third-party files; the EC6 fixture contains only one NASA record.
