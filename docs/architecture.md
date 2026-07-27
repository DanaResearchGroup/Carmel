# Architecture

## Design Principles

- **Deterministic code first, LLM second** — prefer algorithmic solutions over language model calls.
- **Bounded autonomy** — actions are drawn from a curated catalog, not free-form reasoning.
- **Typed contracts** — all data flows use validated pydantic schemas.
- **Full provenance** — every action is recorded with inputs, outputs, versions, and costs.
- **Human-in-the-loop governance** — expensive compute and wet-lab actions require approval.
- **File-first state** — canonical campaign state lives on disk; no database.

## Package Layout

```
Carmel.py                       # CLI entrypoint (repo root, ARC/T3 convention)
carmel/
├── __init__.py
├── version.py                  # Single source of truth for version
├── config.py                   # Pydantic config models, YAML loading
├── paths.py                    # Path utilities, workspace dir initialization
├── logger.py                   # Centralized logger with archival
├── schemas/                    # Phase 1 domain schemas
│   ├── campaign.py             # CampaignInput, Campaign, mixture/observable/reactor types
│   ├── approval.py             # ApprovalPolicy, ApprovalDecision, ActionKind
│   ├── state.py                # CampaignStateValue, CampaignState
│   ├── plan.py                 # Plan, PlannedAction
│   ├── run.py                  # RunRecord, RunStatus, FailureCode, SubmissionMode
│   └── diagnostics.py          # DiagnosticsV1, ObservableSummary, *Selection
├── services/                   # Deterministic service layer
│   ├── artifacts.py            # Atomic JSON/YAML/text I/O
│   ├── decision_log.py         # Append-only JSONL writer
│   ├── campaigns.py            # create_campaign, load_campaign, list_campaigns
│   ├── state_machine.py        # Transition table + persistence
│   ├── approvals.py            # Policy evaluation + decision recording
│   ├── planner.py              # Deterministic Phase 1 plan generator
│   ├── provenance.py           # Per-action provenance records
│   ├── execution.py            # T3 orchestration + diagnostics persistence
│   ├── drawing.py              # Pure-Python SVG renderer for selections
│   └── intake.py               # Free-text intake protocol + stub backend
├── adapters/
│   └── t3.py                   # Real T3 subprocess adapter
└── ui/
    ├── app.py                  # Flask app factory + routes
    ├── templates/              # Jinja templates (HTMX via CDN)
    └── static/style.css        # Minimal custom CSS
```

## Campaign Workspace Layout

A workspace is the source of truth for a single campaign. Canonical files
are machine-readable; markdown is a rendered view.

```
my-campaign/
├── campaign.yaml               # Canonical structured input
├── approval_policy.yaml        # Active approval policy
├── campaign_state.json         # Current lifecycle state
├── plan.json                   # Current plan (canonical)
├── plan.md                     # Rendered plan summary
├── decision_log.jsonl          # Append-only decision stream
├── diagnostics.json            # Normalized T3 output (DiagnosticsV1)
├── intake_review.md            # Optional advisory free-text review
├── benchmarks/                 # Curated benchmarks
├── evidence/                   # Literature memos and source links
├── models/                     # Mechanism versions, SVG selection artifacts
├── provenance/                 # Per-action provenance records
├── reports/                    # Reports
└── runs/                       # T3 (and future) run records
```

## Lifecycle State Machine

```
draft → validated → ready_for_planning → plan_pending_approval
                                     ↓
                       approved_for_execution → running_t3
                                     ↓
                          diagnostics_ready → completed_phase1

(any state can also transition to → blocked or → failed)
```

`carmel/services/state_machine.py` contains the explicit transition
table. Invalid transitions raise `InvalidTransitionError`.

## Approval Policy

Phase 1 enforces compute-side approval for T3 actions only. The
`ApprovalPolicy` model is designed so future ARC, experiment, and
literature actions fit the same framework:

| Field                                      | Default | Effect |
|--------------------------------------------|---------|--------|
| `auto_approve_t3_under_cpu_hours`          | 10.0    | T3 runs ≤ this estimate auto-approve |
| `auto_approve_arc_under_cpu_hours`         | 5.0     | (reserved for Phase 2+) |
| `require_approval_for_experiments`         | True    | (reserved) |
| `require_approval_for_literature`          | False   | (reserved) |

The deterministic planner always asks the approval engine before marking
an action as auto-approved.

## T3 Adapter

`carmel/adapters/t3.py` is the only place that knows how to invoke T3.
Every assumption Carmel makes about T3's input/output contract is
centralized in the `T3Layout` constants block at the top of that file.

- **Real T3 contract** (validated against the upstream T3 repo at
  `ReactionMechanismGenerator/T3` on 2026-04-07):
  - **Input** is a YAML file with top-level keys
    `{project, t3, rmg, qm}`. Species use `{label, smiles, concentration,
    SA_observable}`; reactors use `{type: 'gas batch constant T P', T, P,
    termination_time}`; level of theory lives in `qm.level_of_theory`.
  - **Invocation:** `<t3-interpreter> <T3_PATH>/T3.py <input.yml>` (no
    `--output` flag — T3 writes results next to the input file). The
    interpreter is **not** assumed to be Carmel's own — see "Three-env
    deployment model" below.
  - **Output layout:** `<project_dir>/iteration_*/ARC/<project>_info.yml` (real
    file: `{species: [{label, success}], reactions: [...]}`),
    `<project_dir>/iteration_*/RMG/pdep/network*.py`, and
    `<project_dir>/t3.log`. Level of theory is **never** written back —
    it must be read from the input dict.
- **Submission modes:** `SUBPROCESS` (Phase 1, real `<t3-interpreter>
  T3.py …`), `SERVER` and `LOCAL` (reserved).
- **Input building:** `build_t3_input(campaign)` produces a typed dict
  matching T3's real schema and `write_t3_input_file()` writes it
  atomically under `runs/<run_id>/input.yml`. Carmel never invents
  chemistry; it forwards user-provided structure.
- **Output normalization:** `normalize_t3_outputs(project_dir,
  input_dict, …)` walks `iteration_*/` subdirs, parses each
  `<project>_info.yml` (see `arc_info_filename()`), aggregates species/reactions across iterations, counts
  PDep networks, and pulls LOT from the input dict. The result is a
  typed `DiagnosticsV1`.
- **Failure handling:** every error produces a typed `RunRecord` with a
  specific `FailureCode` (`TOOL_NOT_FOUND`, `INPUT_BUILD_ERROR`,
  `SUBPROCESS_ERROR`, `INVALID_OUTPUT`, `TIMEOUT`).
- **Discovery:** `is_t3_importable()` probes whether `t3` imports **in
  T3's own resolved interpreter** (see below), not Carmel's own process
  — this avoids both the false-positive case where T3 is on
  Carmel's `sys.path` but fails at import time, and the false-negative
  case where T3 lives in a wholly separate environment that Carmel was
  never meant to import into directly.

### Three-env deployment model and T3 invocation

Carmel, T3, and RMG have mutually exclusive Python requirements and
normally live in **three separate conda environments**:

| Env         | Contains        | Python pin |
|-------------|-----------------|------------|
| `rmg_env`   | RMG-Py, Arkane  | `>=3.9,<3.12` |
| `t3_env`    | T3 **and** ARC  | `=3.14` (T3 imports ARC in-process, so they must share an env; T3 in turn launches RMG as a subprocess under `rmg_env`) |
| `crml_env`  | Carmel itself   | `>=3.14` |

Because of this, `_find_t3_executable()` and `is_t3_importable()` /
`_t3_version()` must never assume T3 runs under Carmel's own
`sys.executable`. They all route through `_t3_python_command()`, which
returns the argv prefix that runs a Python interpreter inside T3's
environment. Resolution order:

- If `$T3_CONDA_ENV` is set and `conda` is on `PATH`, T3 is launched and
  probed with `conda run -n <env> --no-capture-output python …`. This is
  the supported way to run a command in a conda environment, and the same
  mechanism T3 itself uses to launch RMG under `rmg_env`.
- Otherwise, if `$T3_PYTHON` is set and points to an existing, executable
  file, that interpreter is used directly. This is right for deployments
  that do not involve conda.
- Otherwise Carmel falls back to `sys.executable` — the correct choice
  for a single-env developer setup where Carmel and T3 share an
  environment (e.g. a from-source checkout with `t3`/`arc` installed
  directly into `crml_env`).
- If either variable is set but unusable (`conda` absent from `PATH`;
  `$T3_PYTHON` missing or non-executable), Carmel logs a warning and
  degrades to the next rule rather than raising, so a bad env var cannot
  crash the adapter's typed success/failure contract; the misconfiguration
  still surfaces honestly downstream as an ordinary
  not-importable/not-found failure.

**Naming the interpreter is not equivalent to activating the environment.**
Conda packages may ship activation hooks under
`$CONDA_PREFIX/etc/conda/activate.d/`, which running `<env>/bin/python`
directly never executes. ARC depends on Open Babel, whose conda package
exports `BABEL_LIBDIR` and `BABEL_DATADIR` from such a hook; without them
Open Babel registers no plugins and `from openbabel import pybel` raises
`ValueError: not enough values to unpack` while building its format table.
The visible symptom is that `import arc` — and therefore `import t3` —
fails, `is_t3_importable()` returns `False`, and every real-subprocess test
skips. That is why `$T3_CONDA_ENV` takes precedence over `$T3_PYTHON`.

### Process trees, timeouts, and what TIMEOUT means

Carmel never launches a leaf process. It launches `conda run`, which
launches T3, which launches RMG — so killing only the direct child kills
the *wrapper* and leaves the actual computation running.

The adapter therefore starts every tool invocation with
`start_new_session=True`, giving it a process group of its own, and on
timeout signals the whole group: `SIGTERM`, then `SIGKILL` after a grace
period. The direct child is reaped last, so its pid — which is also the
group id — cannot be recycled before the final signal is sent.

Nothing polls the child during that grace period, deliberately.
`Popen.poll` *reaps* an exited child, releasing the very pid the group id
depends on; buying an early exit with it is what would make the recycling
race real. So the grace is waited out in full and the `SIGKILL` is sent
unconditionally — signalling an already-empty group is a harmless `ESRCH`.
This is the timeout path, which has already waited hours.

`start_new_session` is not an optimisation, and on its own it fixes
nothing: without it the child would share Carmel's *own* process group,
and signalling that group would kill Carmel and its web server too.

This is what makes a recorded `TIMEOUT` true. Previously a timed-out run
left T3 and RMG alive, still holding the inherited log file descriptors
and still writing into a run directory Carmel had already declared
finished.

Two limits are deliberate and worth stating:

- **Anything that leaves the process group survives.** A T3 run that
  submits jobs to a cluster scheduler puts that work beyond any group
  Carmel can signal; killing the tree does not cancel a queued job.
- **The timeout is `estimated_cpu_hours * 3600 + 600` seconds.** It bounds
  a hung run, not a slow one — a run is killed once it exceeds its own
  estimated budget by ten minutes.

### Runs execute in the background

`POST /run` performs the `RUNNING_T3` transition in the request thread and
then hands the work to a daemon thread, returning immediately. Executed
inline, a run holds the request open for hours: the browser times out, the
redirect to the auto-refreshing dashboard never arrives, and the whole
`RUNNING_T3` UX is unreachable from the tab that started the run.

The *transition* stays synchronous on purpose. It is what rejects a
double-submitted run — with a `409`, in the request that submitted it —
rather than accepting it and racing a run already in flight.

Because the worker is a daemon thread, interpreter exit does not join it.
An `atexit` sweep therefore kills every still-running tool process tree on
shutdown, so stopping the server does not leave T3 and RMG behind — the
same orphaning described above, one layer up. It cannot help if Carmel
itself is `SIGKILL`ed; nothing running inside Carmel can.

## Diagnostics Schema (DiagnosticsV1)

The single Carmel-internal contract for T3 output. Includes:

- per-observable sensitivity summaries (rates and thermo)
- `species_to_compute`, `reactions_to_compute`, `pdep_networks_to_compute`
- `level_of_theory` and `model_version` when reported
- `pdep_sensitivity_flag` and free-form `warnings`
- arbitrary `tool_metadata`

The dashboard reads only this schema; it never reads raw T3 output.

## Graphical Compute Selection

`carmel/services/drawing.py` is a pure-Python SVG generator (no RDKit
required) that produces three persisted artifacts under
`workspace/models/`:

- `species_selection.svg` — labeled rounded rectangles per species
- `reactions_selection.svg` — reactant→product arrow notation
- `pdep_networks_selection.svg` — small radial node graph per network

The Flask UI serves these as `<object>` embeds via the
`/campaigns/<id>/svg/<artifact>` route. The artifacts are deterministic
and backed by persisted `DiagnosticsV1`, so the UI never depends on
in-memory state.

## Free-text Intake (Advisory Only)

`carmel/services/intake.py` defines an `IntakeParser` protocol and a
`StubIntakeParser` no-op backend. The UI exposes a free-text box that
writes the parsed result to `intake_review.md` — an advisory file that
**never becomes canonical state** without an explicit user-driven
structured form submission. Real LLM-backed parsing is deferred to a
later phase.

## CLI

| Command                       | Purpose                              |
|-------------------------------|--------------------------------------|
| `carmel version`              | Print version                        |
| `carmel validate-config FILE` | Validate a config file               |
| `carmel init-workspace DIR`   | Initialize a workspace scaffold      |
| `carmel serve`                | Launch the local Flask UI            |

`carmel serve` accepts `--workspaces`, `--host`, `--port`, `--debug`.

## External Tools

| Tool    | Trust Level             | Phase 1 status |
|---------|-------------------------|----------------|
| T3      | Trusted                 | Real subprocess adapter |
| RMG-Py  | Trusted with caution    | Required by T3 |
| ARC     | Trusted                 | Reserved (Phase 2+) |
| Cantera | Trusted                 | Used by T3 |
| TCKDB   | Trusted                 | Reserved |

## Hard Constraints

- One top-level deterministic planner only
- No dynamic agent spawning
- No free-form tool invocation
- Typed schemas for every tool call
- Literature never writes directly into the model
- All expensive actions gated by budget checks
- All high-stakes actions gated by HITL policy
- Append-only decision log
- Full provenance for every action
