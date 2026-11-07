# Frozen-holdout simulation

Carmel's holdout harness turns a frozen benchmark contract and a historical CHEMKIN
mechanism into a provenance-rich secondary-score report. The harness is observable-oriented:
its adapters evaluate JSR species profiles and ignition delay through a shared contract,
conversion, mapping, and subprocess boundary.

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
  `1e-15`, and at most 100,000 steady-state residual-check iterations. Cantera 3.2.0 takes
  ten accepted integrator steps per iteration, giving a 1,000,000-step budget.

There is no temperature continuation: every row starts from its own inlet state. Every row runs
in a separate, killable process. The run safety ceiling for any active row process is configurable
with the retained `--row-timeout-s` option and defaults to **3600 seconds**. Expiry raises
`RunLevelSimulationError` with `row_wall_ceiling_exceeded`, aborting the run without checkpointing
a row refusal or publishing a score. The worker
count is configurable with `--workers` and defaults to `max(1, os.cpu_count()-2)`. Both settings,
each row's wall time, and each row's integrator step count are recorded in the report. Changing
these process controls does not change the fixed solver tolerances or initial state.

Reaching the JSR steady-state cap gives `max_steps_exceeded`. Classification is confined to
`advance_to_steady_state(max_steps=100000)`: the exact Cantera max-steps exception or Cantera
3.2.0's `NameError("name 'CanteraError' is not defined")` must be corroborated by exactly
1,000,000 accepted steps. A matching message below the cap, an unrelated `NameError`, or an
exception outside that call stays `solver_failure`. Cantera's final-iteration boundary behavior
is preserved, including rejection on the last iteration even if its residual converges.

The cap rule preserves the existing JSR iteration budget and uses the same accepted-step budget
for IDT; it is not tuned against holdout scores. The two-row Cantera 3.2.0 probe reproduced exact
statuses, step counts and values under normal and contended CPU scheduling. Its slowest row was
648.638 s; `max(3600, ceil(4 * slowest_wall_s))` fixes the default run safety ceiling at 3600 s.

The Cantera worker is launched with `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` before NumPy or Cantera is imported. Row
processes use Python's fresh-interpreter `spawn` start method, so no initialized Cantera/native
state is inherited through `fork`. Resume fingerprints include the complete simulation-lock hash,
so they pin the Cantera version, wall ceiling, deterministic caps, solver settings, thread regime,
start method, and conversion rules. Simulation lock **version 2** invalidates old checkpoint
fingerprints; historical `steady_state_timeout` and `ignition_timeout` row reasons are accepted
only with their version-1 simulation lock.
They also include a canonical hash of the pre-trim PLOG validation report, so a cached success or
prevalidation refusal cannot survive a changed removed level, affected interval, or kept bracket.
The child-observed values and the 1-, 5-, and 15-minute load averages at run start are recorded
with the configured and active worker counts.

Each row child enables Python's fatal-signal handler into its own fault log and writes its latest
construction/integration phase, reactor time, temperature, and step count to a sidecar. A negative
signal after integration begins is the row refusal `solver_crash`, including the signal number and
sidecar paths. A signal during construction remains a run-level construction failure. The report
also records a canonical SHA-256 lock over the tolerances, caps, wall ceiling, inlet start, lack of temperature
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
- deterministic solver budget exhaustion (`max_steps_exceeded`);
- solver failure;
- fatal solver signal (`solver_crash`), reported separately from budget exhaustion;
- a condition adjacent to a removed unsound PLOG level;
- conversion, validation, subprocess, or reactor-construction failure at run level.

## CLI

Run `carmel score-holdout --help` for the complete interface. A run supplies the frozen contract,
its pinned split archive, one contract mechanism key and CHEMKIN file, the companion thermo file
and its URL/date/SHA-256 pin, a run directory, and an output JSON path. `--cantera-python` selects
the isolated Cantera interpreter (default: `$T3_PYTHON`; see above). `--row-timeout-s` and `--workers` control only per-row process
execution; expiry aborts the run. The defaults are 3600 seconds and `max(1, os.cpu_count()-2)`, respectively.
The versioned worker request carries `observable: "jsr"` or `"idt"`, with matching typed row
models; unsupported values and mismatched row types are refused.

## Ignition delay

For revision-2 case 1, the typed IDT request pins the small FFCM-1 excited-state data
file described in the [case contract](n-heptane-low-t.md). After conversion and before
validation/row execution, the worker replaces native OH*/CH*/OHV/CHV species and
associated reactions with the pinned chemistry. Every ground-state record is retained;
missing dependencies or a required change to a ground-state collider are refused.
The appended rates carry their own units, independent of the parent mechanism's units.
This applies identically to historical mechanisms, baseline and candidates. Raw legacy
worker JSON without the pin remains usable for historical reproduction; all newly
constructed typed IDT requests carry the pin automatically.

The single target resolver maps OH* ↔ OHV and CH* ↔ CHV during both preflight and
trace integration. Stored target text remains unchanged in the row and report.
Unknown labels are still `target_species_absent`. The chemistry pin and alias map
are included in the lock; checkpoints from the former refusal policy cannot resume.

Case 1 now reports `target_chemistry` alongside `metric`, restricted to 600–1000 K
inclusive and split by reactor group. Case 2 reports its five named C5 ethers in
`target_chemistry`, preserving `metric.all` and `metric.low_temperature_600_900_k`.

`carmel score-holdout-idt` uses the same worker, per-row subprocesses, thread pins,
conversion rules, source-line reaction keys, lock fingerprint and `rows.jsonl` checkpoint.
Supply `--split-root` pointing at the pinned ChemKED tree instead of the JSR `--archive`.
The n-heptane loader verifies every split SHA-256/git blob and its point/temperature/RCM
counts before exporting 189 holdout points: 174 shock-tube and 15 RCM-ideal. Inlet labels
are resolved with the existing structural mapper; ignition target labels remain exact.

Every point starts an adiabatic constant-volume `IdealGasReactor` at its stored temperature,
pressure and mole fractions, including the RCM-ideal points. There is no compression history
or heat loss. The locked horizon is **1.0 s**, independently of the measured delay; relative
and absolute tolerances are **1e-8** and **1e-16**. The initial t=0 state is included, and the
integrator steps until it reaches the horizon, retaining the final step as T3 does. A resolved
event must have a finite positive delay at or before 1.0 s and a temperature rise of at least
**50 K**. Refusals report the first failed condition: `no_ignition_within_horizon` when
the source definition finds no delay, `source_delay_nonfinite`,
`source_delay_outside_horizon` (including a nonpositive delay),
`temperature_rise_below_minimum`, or `trace_unresolved`.

IDT execution carries every removed PLOG level's kept pressure bracket into the row
process. It checks the initial pressure and every integration step, including a step
that crosses an entire bracket, and refuses the trace as `plog_trimmed_interval_entered`
before scoring. Brackets are open at their kept levels and may be unbounded at either
end. This applies to every ignition target, since pressure can change even when the
observed signal is a species or temperature. The simulation lock versions this guard;
the execution row's brackets also participate in its checkpoint fingerprint.

Event definitions follow the T3 experimental numerical path: `max` selects the first maximum;
`d/dt max` selects the first maximum positive `numpy.gradient(values, times)` slope;
`1/2 max` interpolates the first rising half-absolute-peak crossing; `d/dt max extrapolated`
intersects the maximum-slope tangent with the initial target baseline. Species traces use mole
fractions, pressure uses Pa, and temperature uses K. T3's history-free trace-span and settled-tail
checks establish that the event finishes within the trace. For the excited targets in the
shared alias map (`OH*`/`OHV`, `CH*`/`CHV`), the span check is relative:
`max(abs(y)) > 0` and `span > 1e-6 * max(abs(y))`. Ground-state species retain the
absolute `1e-12` floor; pressure and temperature retain their existing relative floor.
The peak endpoint and derivative-tail checks keep their relative thresholds; their
machine-epsilon floor is absolute for ground-state targets and scaled to the signal
(`eps * max(abs(y))`, `eps * max(abs(dy/dt))`) for excited targets, so a faint trace still
rising at the horizon stays unresolved. The simulation lock records
`t3_source_defined_idt_is_resolved_excited_relative_span_v3`, preventing reuse of checkpoints
from earlier excited-target rules. The original event semantics were checked against
T3 commit `4043bb98fcbbad5c252d8e1d57bbfabc6aa76087`. The dispatch deliberately fixes constant
volume and 1.0 s for every point; T3's broader experimental wrapper has different defaults for
RCM and its observation-dependent horizon.

The recorded model check uses five h2o2 states and pressure d/dt max, OH max and OH d/dt max.
The local T3 package import was blocked by unavailable ARC/broken OpenBabel dependencies.
The check therefore executes the unchanged `CanteraIDT` class and event-helper AST nodes
from that detached reference, using real Cantera/NumPy and replacing only the unused adapter
base with `object`. It records the source SHA-256 and this limitation; it does not claim a
full T3 application integration test.

An undeclared target produces `target_species_absent`, naming the target. The shared alias
map resolves OHV/CHV and OH*/CH* to the pinned excited-state submechanism.
Missing targets are checkpointed before simulation. IDT integrates to the existing one-second
reactor-time horizon with at most **1,000,000 accepted integrator steps**. Exhausting the step
budget before the horizon gives `max_steps_exceeded`; a completed trace without a qualifying
ignition gives the first failed condition listed above. Solver failure and fatal signals remain row refusals;
construction defects and wall-ceiling expiry refuse the run. The wall ceiling and worker count
match the JSR CLI controls.

IDT scoring is the equal-point mean `abs(log10(tau_sim_seconds / tau_exp_seconds))` over
finite, strictly positive delays. Reports retain all point IDs, per-point errors/refusals,
coverage, reactor groups and contract temperature bands (<750, 750–900 inclusive, >900 K).
The revised dispatch gate permits at most 58 non-solver refusals among the 174 shock points
and requires at least 12 of the 15 RCM-ideal points to score for each mechanism. A failed gate is reported after
saving its report; settings must not be tuned using that result.

## Paired uncertainty

`carmel.benchmarks.holdout_intervals` exposes `idt_separation` and `jsr_intervals`. Both require
matching simulation locks and frozen holdout fingerprints and compare the intersection of valid
point/observable IDs. `rescore_saved_jsr(report_path)` rebuilds simulated values from the sibling
`rows.jsonl`, verifies agreement with the saved report, pins both input hashes, and never writes
to either input or launches a simulation.

Rescoring also requires the sibling `request.json`. Checkpoint entries are selected by
the saved request row, mechanism semantic hash, simulation-lock hash, and PLOG-validation
hash, using the recorded lock rather than the installed worker's current settings. Stale
entries are ignored before checking for conflicts within that identity; identical repeated
entries are allowed. Every dispatched row must have a matching entry, with the same status
as the report. A refused entry must carry a nonempty reason matching the reported row and
its mapped metric observations. A known `unmapped_inlet_species` refusal omitted from the
request needs no checkpoint entry, but its status, unmapped-species evidence, and metric
refusals must agree. Missing provenance or mismatched fingerprints are never rescued by
matching row IDs or simulated values alone.
If no rows were dispatched, the worker may never have created `rows.jsonl`; rescoring then
records a null checkpoint hash and validates every row as a pre-dispatch refusal.

Intervals use 10,000 paired row bootstrap replicates, seed **20261018**, NumPy PCG64, and the
2.5th/97.5th percentiles with linear interpolation. Each replicate samples N whole row IDs
with replacement (implemented as multinomial row counts), using identical weights for both
mechanisms. The reported gap is **v_old error minus v_new error**; a positive value favors
v_new. JSR species stay together within a row; each sampled species mean receives equal weight,
using the fixed detection limits from the complete frozen observations. Species with no sampled
observation are omitted exactly as in the existing JSR metric. Empty diagnostics have null
scores and intervals, never zero-error claims. Re-running with the same seed and input reports
produces byte-identical serialized interval reports in the same recorded NumPy environment.

Diagnostic intervals accompany the contract aggregate: IDT by reactor group and contract
band; JSR overall, inclusive 600–900 K, C0–C2 (CH4, C2H4, C2H6, CH2O, CH3CHO, CO, CO2),
fuel/alkenes (NC5H12, C5H10-1, C5H10-2, C3H6), C5 cyclic ethers (C5H10O1-2, C5H10O1-3,
C5H10O1-4, C5H10O2-3, C5H10O2-4), and O2. The decompositions do not replace the contract metric.
