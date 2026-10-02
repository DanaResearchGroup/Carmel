# Frozen n-heptane low-temperature benchmark

This benchmark tests whether a mechanism revision process recovers five documented
low-temperature reaction types for **n-heptane**. Structural recovery is the primary
score. Improvement in held-out ignition delay is secondary. Korcek chemistry is carried
separately for speciation, and never earns ignition or primary credit.

The frozen data are [contract.json](https://github.com/DanaResearchGroup/Carmel/blob/main/benchmarks/n-heptane-low-t/contract.json);
[schema.json](https://github.com/DanaResearchGroup/Carmel/blob/main/benchmarks/n-heptane-low-t/schema.json) is generated from
`carmel.benchmarks.n_heptane.Contract`. Python validation rejects changed totals,
missing species, guessed structures, crossed study groups, unsafe file paths and
incorrect credit scopes. All molecular matching requires Carmel's `agents` extra.

## Mechanism and structure pins

| Endpoint | Source | Retrieved | SHA-256 | Unique species / reaction entries |
|---|---|---|---|---|
| v_old | [LLNL n-heptane v3.1](https://combustion.llnl.gov/sites/combustion/files/nc7_ver3.1_mech.txt) | 2026-10-02 | `ea0980fcb6415b0508dd31fbab2e3f765bc01b10783fb9455baf906e61c68089` | 654 / 2827 |
| v_new | [NUIG 2016 n-heptane](https://nuigalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/n-heptane/nc7_16mech.dat.txt) | 2026-10-02 | `9f6bc7101d34d7cea07e1cf90d05648e5f54f8cf8ae5de0176b0d7987d7e734e` | 1268 / 5336 |

A reaction entry is one CHEMKIN equation/Arrhenius record; reversible equations count
once, and duplicate/pressure-dependent entries retain their own source line. v_old has
658 species tokens but only 654 unique labels. No full mechanism, thermochemistry,
glossary PDF or ChemKED observation bytes are distributed in this change.

`Pin.verify(downloaded_bytes)` fails with **PIN CHANGED** if the exact byte count or
SHA-256 changes. Structure support files have their own retrieval date, URL, size and
hash in `support_sources`, so updating a live download cannot silently change the map.
The reproduction verifies those support pins before accepting its result.

### Historical identity: UNVERIFIED

The retrieved releases are **not proven to be the publication-era files**. LLNL's file
header dates v3.1 to 2012-03-30 and explains a March 2012 correction to v3.0, dated
2009-12-11. The [LLNL release page](https://combustion.llnl.gov/mechanisms/alkanes/n-heptane-detailed-mechanism-version-3)
associates v3.1 with Mehl2011 but describes bug fixes; this does not authenticate a
2011 file.

The [Zhang2016 manuscript](https://researchrepository.universityofgalway.ie/bitstreams/42402629-3058-4da5-b679-e9c36af2dd15/download),
Chemical Kinetic Mechanism, manuscript pp.6–7, describes an updated C0–C4 base and
NUIG pentane/n-hexane submechanisms, refined rate rules and updated thermochemistry.
It does not establish the retrieved LLNL v3.1 as its parent. The census archives and
mechanism headers were inspected for a separately identified publication-era
v3.0/Mehl2011 model; none was established. No authenticated publication-era old model
was found to fingerprint. The author glossary's PDF creation date, 2016-02-19, supplies
contemporaneous structure evidence, not mechanism parentage.

This contract defines a reproducible **cross-group file contrast**. It does not assert
a direct Mehl2011-to-Zhang2016 historical patch or first discovery of these classes.

## Mapped ground truth

Every member reaction and each participating species' SMILES are explicit in the
contract. The following totals were reproduced from all 5,336 v_new reaction entries,
then compared structurally against the 2,827 v_old entries. These are **n-heptane
seven-carbon source reactions**; lower-carbon homologues in the same large mechanism
are outside this case.

| Type | Member entries | Fully resolved entries | Resolved unique member species | Example |
|---|---:|---:|---:|---|
| G1: alternative OOQOOH H shift to P(OOH)₂ | 77 | 77 | 63 / 63 | `C7H14OOH1-3O2 ⇌ C7H13Q13-5` |
| G2: P(OOH)₂ → hydroperoxy cyclic ether + OH | 77 | 77 | 91 / 91 | `C7H13Q13-2 ⇌ C7H13O12-3OOH + OH` |
| G3: P(OOH)₂ → olefinic hydroperoxide + HO₂ | 26 | 26 | 39 / 39 | `C7H13Q13-2 ⇌ C7H13-1D3OOH + HO2` |
| G4: direct OOQOOH concerted HO₂ elimination | 26 | 26 | 32 / 32 | `C7H14OOH1-3O2 ⇌ C7H13-2D1OOH + HO2` |
| G5: P(OOH)₂ fragmentation | 34 | 34 | 75 / 75 | `C7H13Q13-5 → C3KET13 + C4H8-1 + OH` |
| G6: Korcek, speciation only | 4 | 4 | 10 / 10 | `C7KET13 → CH3COOH + PC4H9CHO` |

The primary paper locations are Zhang2016 Chemical Kinetic Mechanism / Low temperature
mechanism / Fig.1 for G1–G5, and Results and discussion → Jet-stirred reactor data,
manuscript p.14 lines12–15 for G6. G1–G5 are related channels of one extension family,
not five independent historical discoveries. The raw mechanism class tags corroborate
the mapping; the matcher never consumes those tags or NUIG labels.

### Structure derivation and uncertainty

The [author species glossary](https://c3.universityofgalway.ie/media/researchcentres/combustionchemistrycentre/files/mechanismdownloads/n-heptane/glossary_nheptane.pdf)
contains a row for each of v_new's 1,268 labels. Extraction normalizes the PDF's Unicode
hyphen to ASCII and reads the SMILES column next to its formula and InChI. All member
species have valid, directly sourced structures. Invalid non-member glossary strings
are refused; they are not repaired by guessing. Canonical comparison omits unassigned
stereochemistry and preserves radicals, connectivity and stoichiometric multiplicity.

All 654 old species are inventoried against the independently pinned
[LLNL v3.1 thermochemistry](https://combustion.llnl.gov/sites/combustion/files/n_heptane_v3.1_therm.dat.txt).
432 have established structures; the remaining 222 are explicitly **UNRESOLVED**,
with formula evidence retained. Shared labels are accepted only where their sourced
SMILES agrees with the old thermochemistry formula.

The tested old-label grammar `N?CnKETij` denotes an unbranched n-carbon chain with a
carbonyl at position i and hydroperoxide at j, numbered from the named chain end.
`N` denotes the normal chain; i and j must be distinct and within the chain. For example,
`NC7KET13` generates `C(=O)CC(OO)CCCC`, identical to the glossary's `C7KET13` graph.
The test checks all 18 old n-C7 KET isomers against the explicit map, and refuses unknown
or impossible labels. No other unresolved label receives a guessed SMILES.

### Structural class rules and absence proof

The executable rules are in `carmel.benchmarks.structure.classify`. They accept either
net-reaction orientation, require elemental balance including hydrogen, and inspect
molecular graphs:

- **OOQOOH:** a linear seven-carbon chain with one C–O–OH, one C–O–O•,
  one oxygen radical and no carbon radical, carbonyl, alkene or ether ring.
- **P(OOH)₂:** a linear seven-carbon chain with two C–O–OH groups and one
  carbon radical on a carbon bearing neither OOH group, with no carbonyl, alkene or ether ring.
- **G1:** OOQOOH becomes a single P(OOH)₂ molecule.
- **G2:** P(OOH)₂ becomes one closed-shell C7 cyclic ether retaining one OOH,
  plus exactly OH.
- **G3/G4:** P(OOH)₂ / OOQOOH respectively becomes one closed-shell C7
  alkene retaining one OOH, plus exactly HO₂.
- **G5:** P(OOH)₂ produces at least two carbon-containing fragments; all elements
  and product multiplicities must be retained.
- **G6:** a closed-shell C7 ketohydroperoxide produces exactly two closed-shell
  carbon-containing molecules, one acid and an additional carbonyl.

These are net structural classes, not claims that an experimental observable uniquely
identifies a transition state. Tests reject ordinary QOOH cyclic-ether formation,
RO₂ ⇌ QOOH, conventional ketohydroperoxide formation (including its two-step α-hydroperoxy intermediate), branched/shorter-carbon
analogues, invalid structures and unbalanced reactions.

Absence is not inferred from NUIG labels. First, old species' **thermochemistry formulas**
exhaustively select every possible C7H15O4 or C7H14O3 source; all such species have
established graphs. The C7H15O4 candidates are 18 OOQOOH species, with no P(OOH)₂;
the C7H14O3 candidates are the 18 mapped ketohydroperoxides. Next, canonical reaction
signatures compare both sides, ignoring labels, rate parameters and direction while
retaining multiplicities. There are 1,689 distinct fully mapped old signatures.
Finally, every partially mapped old equation is compared by side cardinalities and
formula multisets; no unresolved equation can be a chemically identical member. An
unresolved formula-compatible comparison stops reproduction with NEEDS-INPUT.

## Frozen whole-study split

ChemKED is pinned to commit `606005bfc8f5214b3f0b5ca7300a96a82815c2ae` of
`pr-omethe-us/ChemKED-database`. The adopted split preserves the census freeze timestamp
2026-10-02T04:33:18.068272+00:00. Each study ID, DOI where supplied, source path,
Git blob hash, SHA-256, point count and reactor flag is in the contract. Burcat1981 has
no DOI in the supplied census; its named study group and blob pins remain explicit.

| Set | Studies | Files | IDT points | Below 750 K | 750–900 K inclusive |
|---|---:|---:|---:|---:|---:|
| Development | 10 | 45 | 521 | 20 | 39 |
| Holdout | 6 | 24 | 189 | 26 | 63 |

Development: Burcat1981, Ciezki1993, Colket2001, Fieweger1997, Gauthier2004,
Heufer2010, Horning2002, Smith2005, Vermeer1972 and Zhang2016.
Holdout: Di Sante2012, Hartmann2011, Herzler2005, Karwat2013, Shen2009 and
Vandersickel2012. The Duisburg studies and Zürich collaboration stay together.
These are whole-study assignments; temperature bins are coverage proxies, not a claim
that each point lies on a measured NTC branch.

The **15 RCM points** (12 Di Sante and 3 Karwat) have no supplied volume history.
Every one has a zero-based point index in `rcm_ideal_points`. Simulate them with the
supplied T/p as ideal constant-volume reactors and report them on their own line;
do not invent compression histories or a heat-loss correction. The other 174 holdout
points are shock-tube observations.

`load_split(contract, root=Path('path/to/chemked'))` rechecks exact Git blobs and
SHA-256 hashes, actual YAML point counts and temperature bins, and all RCM flags.
Without a root it uses Carmel's pinned ChemKED manifest and cache/fetch machinery;
a caller-supplied fresh fetch is accepted only after the same hash checks. Changed
bytes fail before use. Committed tests use synthetic observations and six tiny
reaction-line excerpts, without network access.

## Credit, metrics and locking

Classify the **RMG-generated baseline** structurally first. A type already present
receives baseline coverage, with zero revision credit for that type. The revision score
is `|candidate_types − baseline_types| / 5`; baseline coverage and total candidate
coverage are reported separately over the same five types. A proposed balanced net
reaction accepted by a class rule recovers its type once. Duplicate kinetic entries,
more proposed members and rate fitting cannot increase that type's credit.

G5 is structurally distinguishable in the supplied matcher and remains a separate
unit. If another declared matcher cannot distinguish it from G1, group G1/G5 into one
union type **before candidate locking**, apply the same grouping to baseline and
candidate, and use four units and denominator 4. G6 is excluded in both modes; its
structural coverage belongs only to a separate speciation report.

For IDT, use `mean(abs(log10(tau_sim / tau_exp)))`, in identical units and with the
stored ignition definition. Both delays must be finite and positive. Equal point
weights apply. Report separate shock-tube and RCM-ideal means, their scored/189
coverage broken down by group, per-point errors, and every refusal with its point ID.
Improvement is **baseline error minus candidate error on identical eligible point IDs**,
reported per reactor group; do not compare means from different successful subsets.
Do not treat a reduced successful subset as improved benchmark coverage.

No numeric holdout n-heptane speciation data are frozen here: its metric is **UNAVAILABLE**,
not a zero error. Future acquisition requires a separately versioned source and
study-level assignment/metric frozen before search. Zhang2016 JSR observations remain
development. Uncalibrated signals and figure-only observations are unscored.

A missing ignition definition, unsupported units, invalid observation, solver failure,
nonpositive/nonfinite simulated delay or failure to ignite within the predeclared
horizon is an explicit refusal. Missing/conflicting structures make the affected
reaction unscored. Hash drift, unreproduced split counts or unresolved eligible old
species fail the benchmark; they cannot be silently dropped. Report resolved entry
and species counts, baseline/candidate type coverage and all observable refusals.

Before **one** holdout evaluation, record the candidate, baseline and contract SHA-256,
matcher/version and G5 grouping, simulator/version, ignition definitions and time
horizon. Lock all of these after development. The single evaluation retains failures
and refuses holdout-driven retries, tuning, searches or point exclusions.

## Leakage limits

The contract concretizes the census risks:

- Keep all Zhang2016 observations in development, including later acquired JSR data.
  Older holdout studies were already seen by v_new's authors. This is search-time
  separation, not historically blind validation; independent labs do not erase exposure.
- Keep v_new mechanism bytes, mapped truth and its labels out of the revision agent's
  prompts, pathway proposals, retrieval indexes and search tools. Only the evaluator
  consumes truth after locking. Label any truth-exposed experiment separately.
- Distinguish chemistry additions from rate/thermochemistry tuning; weak ignition
  sensitivity and bulk speciation cannot uniquely establish structural recovery.
- Record RMG generator settings and actual baseline structural coverage. Existing
  families earn baseline credit. Declare any multistep-to-net Korcek matcher in advance.
- Preserve source DOI/study grouping, byte pins and response meanings. Histories are
  not IDT points and speciation cells are not independent experiments. Figure-only
  and uncalibrated observations require a named numeric source before use.
- Retain UNVERIFIED historical identity and the RCM idealization with all reports.
  A changing live file or a different publication-era release needs a new contract.

The current delivery does not generate an RMG baseline, run RMG/T3, simulate mechanisms,
choose another case or score a revision candidate.

## Reproduction

The committed `benchmarks/n-heptane-low-t/reproduce.py` script takes the census checkout,
support-file directory and whole-study split report as arguments. Existing mechanism and
support files are verified against the contract pins before use. If a pinned mechanism or
support file is absent, `--download-dir` downloads it from the URL in `contract.json` and
verifies its byte count and SHA-256 before use. The ChemKED checkout must contain the
pinned commit because its split files are verified by Git blob and SHA-256.

The glossary audit also requires Poppler's `pdftotext` command to be installed and available
on `PATH`.

From the repository root, run:

```bash
python benchmarks/n-heptane-low-t/reproduce.py \
  --census-root path/to/i116-alkane-census \
  --support-root path/to/i117-contract \
  --split-report path/to/R-032a_n-heptane-study-split_2026-10-02.json \
  --download-dir path/to/n-heptane-downloads
python -m pytest
ruff check
ruff format --check
mypy carmel Carmel.py
```

The reproduction reparses the untouched census mechanism texts and author glossary,
rebuilds the old structural/formula audit and all member lists, checks the committed
map against those results, verifies all mechanism/support hashes and reproduces the
frozen split from actual ChemKED bytes. It runs no chemical kinetics calculations.
The schema and committed offline tests remain usable independently of that research
archive; re-acquiring mechanism/support files must reproduce their pins.
