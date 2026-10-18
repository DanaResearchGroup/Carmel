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
trimmed-YAML SHA-256 is recorded. CHEMKIN's Fortran E-less numeric tokens are normalized by a
general preprocessing rule: outside `!` comments, a whole mantissa-plus-signed-digits token gets
an explicit `E`. Every rewrite records its input file, line, before text, and after text.

Conversion always uses `--permissive --verbose`. Only redundant species declarations, redundant
thermo entries, and unparsable thermo blocks containing whitespace and `!` characters are allowed
permissive warnings. Verbose notices for thermo entries whose species are absent from the kinetics
input are recorded as information; they are also emitted by strict conversion and are not
permissive warnings. Any other warning or skipped source block refuses the run. Duplicate thermo
records are reported with both headers, whether their coefficients differ, and the fact that the
first entry was used.

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
