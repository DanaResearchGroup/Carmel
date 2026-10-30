# Frozen-holdout simulation

Carmel's holdout harness turns a frozen benchmark contract and a historical CHEMKIN
mechanism into a provenance-rich secondary-score report. The harness is observable-oriented:
this first adapter evaluates JSR species profiles, while a later adapter can add ignition delay
without changing the contract, conversion, mapping, or scoring boundary.

## Process and provenance boundary

Carmel does not depend on or import Cantera. `carmel.benchmarks.holdout` writes a versioned,
Pydantic-validated JSON request and invokes the self-contained `cantera_worker.py` with a
configurable interpreter, which must have Cantera installed (in practice T3's environment).
The interpreter is the `--cantera-python` path if given, else `$T3_PYTHON`. If neither is set, or the
chosen path is not an existing executable file, the run is refused before anything is launched or
written. There is no fallback to Carmel's own interpreter, which has no Cantera.
The response is validated before Carmel uses it, and the child's stdout and stderr are always
captured as `stdout.log` and `stderr.log` in the run directory.

The worker converts kinetics and thermo together with the invoked Cantera version's `ck2yaml`.
Every raw input SHA-256, normalized-input SHA-256, converter version, converted warning, and
trimmed-YAML SHA-256 is recorded. CHEMKIN's Fortran E-less numeric tokens receive an explicit
`E` only within numeric fields: NASA-7 header columns 46–75 (three temperature fields),
coefficient-card columns 1–75 (five 15-column fields), the last three numeric rate parameters
after a reaction equation, and numeric auxiliary lines (`LOW`, `HIGH`, `TROE`, `SRI`, `PLOG`,
`REV`, `CHEB`, `TCHEB`, `PCHEB`, `LT`, `RLT`). Fixed-width fields retain their width by consuming
padding; a field without space for `E` remains unchanged for conversion to refuse. No rewriting
occurs in `SPECIES`, reaction equations, other header text, or `!` comments. In particular,
identifiers such as `C8H141-5,3-4` and `C4H6OHOOH1-4-3` retain their exact spelling.
Every rewrite records its input file, line, before text, and after text.

Inside a `THERMO` section, one additional format-only rule restores a missing NASA-7
fourth-card index. After three consecutive cards with valid indices `1`, `2`, `3`, valid
temperature fields, and fixed-width coefficients, a fourth card containing four valid
15-column coefficient fields and a blank trailer is padded through column 79 and receives
`4` in column 80. The same rule also accepts an exactly 81-column card whose only nonblank
trailer byte is `4` in column 81: it moves that index to column 80 and drops the extra byte.
Coefficient and temperature bytes, species names, comments, line endings, and record order are
preserved. Any other nonblank trailer, malformed coefficient, or interrupted card sequence is
left unchanged for conversion to refuse; two populated index columns are never repaired.
Each repair records its source line and before/after text; the rule is named in the simulation
lock. Validation of a new thermo pin must independently compare all fourteen coefficient
strings, temperature fields and species names in every record before and after normalization.
Both normalization rules are versioned in the simulation lock. The n-pentane regression also
requires byte-identical normalized inputs and unchanged saved JSR results.

Conversion always uses `--permissive --verbose`. Only redundant species declarations, redundant
thermo entries, and unparsable thermo blocks containing whitespace and `!` characters are allowed
permissive warnings. Verbose notices for thermo entries whose species are absent from the kinetics
input are recorded as information; they are also emitted by strict conversion and are not
permissive warnings. Any other warning or skipped source block refuses the run. Duplicate thermo
records are reported with both headers, whether their coefficients differ, and the fact that the
first entry was used.

Cantera validation accepts only the reviewed `NasaPoly2::validate` **h/RT discontinuity at
1000 K** for OH*/OHV or CH*/CHV, and only when that species' parsed NASA numeric fields
match the corresponding pinned SHA-256. The reviewed sources are
`/home/alon/runs/i127-holdout-sim/v_new/thermo.normalized.dat`, OH* header line 25 and CH*
header line 145. The digest concatenates the three 10-byte temperature fields (columns
46–75), then fourteen 15-byte coefficient fields in card order, preserving every padding
byte. Species labels and card indices are excluded; values are never converted and rounded.

| Reviewed source | Allowed labels | Numeric-field SHA-256 |
| --- | --- | --- |
| OH*, line 25 | OH*, OHV | `85ab13a7ce62508df7daf3411313e27f5e650008f665ec790e04c194aa7fb2d7` |
| CH*, line 145 | CH*, CHV | `4a3c493e51e59ada6dfba5d198e487ab5e78b907d4bd0ba32f217df460ae880f` |

The first record wins, including across kinetics and companion thermo; a malformed or
changed first record cannot acquire approval through a later duplicate. Any other species,
warning type, midpoint, or changed numeric byte remains refused. Original case-2 labels use
the same digest check. The simulation lock records `reviewed_nasa_numeric_field_sha256_v2`
and all four label-to-digest entries. This warning rule changes no input, coefficient or JSR
result and does not map the ignition targets OH*/CH* to OHV/CHV.

Before validation, every PLOG pressure level is checked at Cantera's validation temperatures.
A rate expression's activation energy is converted from the YAML document's inherited
`units.activation-energy` value or its own explicit unit; an energy unit the harness cannot
convert exactly refuses the run before any trimming.
A non-positive sum is removable only when no scored pressure lies in an interpolation interval
adjacent to that level. The report records the reaction, CHEMKIN line, expressions, removed
pressure, failing temperatures, and kept bracket. A condition in an affected interval is refused
as `mechanism_rate_unsound_at_condition`; a whole reaction is never removed.

CHEMKIN source lines are matched to converted-YAML reactions by one canonical key
(`canonical_reaction_key`: whitespace, species order, `2OH` versus `2 OH`, and falloff or
third-body markers are normalised; `=` and `<=>` agree, `=>` is distinct). A reaction with a `REV`
auxiliary line is registered under both of the irreversible forms ck2yaml emits, forward and
reverse, with the same line. Lines are paired per occurrence in source order; if a key's source
and YAML counts differ, the reaction is recorded as `ambiguous` with its candidate lines rather
than guessed.

## JSR model

Each frozen row is simulated independently as an isothermal, isobaric, perfectly stirred reactor:

- the initial reactor and feed state are the row's inlet composition, temperature, and pressure;
- the reactor is an `IdealGasReactor` with energy disabled and the row's reported volume;
- a mass-flow controller uses `mdot = initial reactor mass / residence time`;
- a pressure controller follows that mass flow with `K = 1e-5`;
- a `ReactorNet` advances to steady state with relative tolerance `1e-9`, absolute tolerance
  `1e-15`, and at most 100,000 steps.

There is no temperature continuation: every row starts from its own inlet state. Every row runs
in a separate, killable process. Its wall-clock limit is configurable with `--row-timeout-s` and
defaults to 600 seconds; expiry is the explicit row refusal `steady_state_timeout`. The worker
count is configurable with `--workers` and defaults to `max(1, os.cpu_count()-2)`. Both settings,
each row's wall time, and each row's integrator step count are recorded in the report. Changing
these process controls does not change the fixed solver tolerances or initial state.

The Cantera worker is launched with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` before NumPy or Cantera is imported. Row
processes use Python's fresh-interpreter `spawn` start method, so no initialized Cantera/native
state is inherited through `fork`. Resume fingerprints include the complete simulation-lock hash,
so they pin the Cantera version, timeout, solver settings, thread regime, start method, and conversion rules.
They also include a canonical hash of the pre-trim PLOG validation report, so a cached success or
prevalidation refusal cannot survive a changed removed level, affected interval, or kept bracket.
The child-observed values and the 1-, 5-, and 15-minute load averages at run start are recorded
with the configured and active worker counts.

Each row child enables Python's fatal-signal handler into its own fault log and writes its latest
construction/integration phase, reactor time, temperature, and step count to a sidecar. A negative
signal after integration begins is the row refusal `solver_crash`, including the signal number and
sidecar paths. A signal during construction remains a run-level construction failure. The report
also records a canonical SHA-256 lock over the tolerances, timeout, inlet start, lack of temperature
continuation, thread pins, Cantera version, and all three approved CHEMKIN
conversion/preprocessing rulings.

As each row finishes, its result is appended and flushed to `rows.jsonl` in the run directory.
Restarting the same run resumes exact row requests whose fingerprint matches the trimmed mechanism
and leaves completed work untouched. The resume identity is a semantic hash of the trimmed YAML;
only ck2yaml's volatile top-level generation date is excluded. The exact trimmed YAML byte hash is
still pinned separately. A flow-network construction error refuses the entire run
because it indicates a broken adapter; it can never be counted as row non-convergence. A solver
failure after successful construction refuses only that row and preserves its exception text.

## Species identity and scoring

Record identities are resolved to canonical structures from reported InChI/SMILES where present
and the documented mechanism label grammar otherwise. Mechanism labels are resolved through the
same structure grammar, and a match is accepted only when exactly one mechanism species has the
same canonical structure. Bare label equality is never a matching rule.

For species `s`, the clamp `L_s` is its stated detection limit or half its smallest strictly
positive frozen observation. Measured and simulated values below `L_s` are clamped to it. The
species score is the row mean of `abs(log10(max(x_sim, L_s) / max(x_exp, L_s)))`; the overall
score is the unweighted mean of species scores. The report gives per-species row counts and scores,
the overall score, and the inclusive 600–900 K subset separately.
ReSpecTh records carry no detection limits, so the half-smallest-positive fallback applies to every
species; the report's `detection_limit_sources` states which branch each species used (`stated` or
`half_min_positive`). A holdout record that declares a detection limit is refused at load, because
the parser has no field to carry it.
Separation scoring computes each `L_s` once from the complete frozen observation set and reuses it
for both mechanisms' shared valid `(row_id, species)` intersection; excluding a row or observable
cannot raise the clamp. The separation report counts observables excluded by each mechanism and
species. Each mechanism report also pins the contract bytes, split/archive identity, and a canonical
hash of every scored observation and reactor condition; separation refuses unequal holdout fingerprints.

Explicit refusal classes include:

- no stated detection limit and no positive observation;
- unmapped or ambiguous species structure;
- missing, nonfinite, or negative measured/simulated values;
- unsupported reactor conditions;
- steady-state wall-clock timeout;
- solver failure;
- fatal solver signal (`solver_crash`), reported separately from timeouts;
- a condition adjacent to a removed unsound PLOG level;
- conversion, validation, subprocess, or reactor-construction failure at run level.

## CLI

Run `carmel score-holdout --help` for the complete interface. A run supplies the frozen contract,
its pinned split archive, one contract mechanism key and CHEMKIN file, the companion thermo file
and its URL/date/SHA-256 pin, a run directory, and an output JSON path. `--cantera-python` selects
the isolated Cantera interpreter (default: `$T3_PYTHON`; see above). `--row-timeout-s` and `--workers` control only per-row process
execution; the defaults are 600 seconds and `max(1, os.cpu_count()-2)`, respectively.
The versioned worker request carries `observable: "jsr"`; unsupported observable values are an
explicit run-level refusal, leaving a typed discriminator for future IDT adapters.
