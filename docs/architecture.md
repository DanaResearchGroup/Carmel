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
│   ├── authorization.py        # Per-adapter execution envelopes + combined approval gate
│   ├── spend.py                # Cumulative CPU-hour spend (consumed + reserved)
│   ├── planner.py              # Deterministic plan generators (T3 + standalone ARC)
│   ├── provenance.py           # Per-action provenance records
│   ├── execution.py            # T3/ARC orchestration + diagnostics persistence
│   ├── recovery.py             # Run-liveness probing + abandonment
│   ├── processes.py            # /proc-based process-group identity checks
│   ├── drawing.py              # Pure-Python SVG renderer for selections
│   └── intake.py               # Free-text intake protocol + stub backend
├── adapters/
│   ├── _launcher.py            # Shared subprocess launcher (process groups, tree kill, conda-env resolution)
│   ├── t3.py                   # Real T3 subprocess adapter
│   └── arc.py                  # Real standalone-ARC subprocess adapter
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
├── active_run.json             # Present only while a run is in flight
├── active_run.lock             # Held by the process supervising that run
├── plan.json                   # Current plan (canonical)
├── plan.md                     # Rendered plan summary
├── decision_log.jsonl          # Append-only decision stream
├── diagnostics.json            # Normalized T3/ARC output (DiagnosticsV1)
├── intake_review.md            # Optional advisory free-text review
├── benchmarks/                 # Curated benchmarks
├── evidence/                   # Literature memos and source links
├── models/                     # Mechanism versions, SVG selection artifacts
├── provenance/                 # Per-action provenance records
├── reports/                    # Reports
└── runs/                       # T3/ARC run records
```

## Lifecycle State Machine

```
draft → validated → ready_for_planning → plan_pending_approval
                                     ↓
                       approved_for_execution
                          ↓                  ↓
                     running_t3         running_arc
                          ↓                  ↓
                  diagnostics_ready    results_ready
                          ↘                 ↙
                         completed_phase1

(a T3 plan runs the left branch, a standalone-ARC plan the right one;
 every state except completed_phase1 can also transition to → failed;
 only plan_pending_approval can transition to → blocked, by rejection)
```

`carmel/services/state_machine.py` contains the explicit transition
table. Invalid transitions raise `InvalidTransitionError`.

### Recovery: getting out of `failed` and `blocked`

A campaign that cannot leave `failed` is unusable, and every button on
its dashboard raises. But an unconditional edge back to
`approved_for_execution` would be worse: `failed` is reachable from
`draft` and `validated` too, so that edge would let a campaign reach
execution having never passed the approval gate.

Recovery therefore obeys one rule — **an exit from `failed` may only
return a campaign to a state it demonstrably already reached, never to a
later one.** Three edges follow from it:

| Exit | Available when | Meaning |
|------|----------------|---------|
| → `ready_for_planning` | always | Discard the plan and re-plan. Bypasses nothing: planning and the approval policy both run again from the start. |
| → `approved_for_execution` | `failed_from == running_t3`, `running_arc`, or `approved_for_execution` | Retry a tool run of a plan that was already approved — whether it failed during the run, or between approval and launch. |
| → `diagnostics_ready` | `failed_from == diagnostics_ready` | Adopt diagnostics already durable on disk, for a run that succeeded and then failed while being recorded as complete. |
| → `results_ready` | `failed_from == results_ready` | The same adoption edge for a standalone ARC run's results. |

`CampaignState.failed_from` records the origin, and the direct resumes
are gated on it. A state file Carmel did not write has no origin
recorded, so only re-planning is offered — which is exactly the edge
that needs no history to be safe.

`blocked` (a rejected plan) exits the same way, to `ready_for_planning`.
The plan is never un-rejected; a new one is generated and judged afresh.

### Recovering a run whose supervisor died

`running_t3` and `running_arc` are the states whose truth cannot be read
off disk. If the Carmel process supervising a run is killed, no `except`
runs, nobody writes the run's ending, and the campaign sits in the
running state forever while the dashboard promises progress. (The rest
of this section says `running_t3`/T3 for concreteness; the identical
machinery supervises `running_arc`/ARC.)

Guessing is harmful in both directions. Guessing "still running" wedges
the campaign, which is the bug. Guessing "finished" is worse: it records
the run as failed while T3 and RMG carry on writing into the workspace —
the same defect the process-tree kill prevents, moved one layer up.

So each run leaves two things behind, both at the workspace root:

* `active_run.lock`, held under an exclusive `flock` for the run's
  duration. The kernel releases it when the holder dies, however it
  dies. A lock that can be taken therefore *proves* no supervisor
  survives — unlike a heartbeat, it cannot go stale, and unlike a
  recorded pid, it cannot be fooled by pid reuse.
* `active_run.json`, holding the tool's process group id **together with
  the group leader's start time and its kernel-observed command line**,
  read back from `/proc` at launch.

The lock is taken *before* the campaign ever reads as `running_t3`, and
released by whichever thread finishes the run. Were it taken by the
background thread instead, there would be an instant in which the
campaign is running with no lock held and no record written — and a probe
landing there would conclude nothing had ever started and offer to
abandon a run that was about to launch.

`probe_run_liveness()` combines them into one of five findings —
`supervised`, `orphaned`, `unsupervised`, `no_record`, or `unknown` —
which the dashboard reports verbatim rather than paraphrasing. It also
decides whether the page auto-refreshes: a run nobody is supervising
stops promising progress and offers a way out instead.

Abandoning a run makes "this run is over" *true* rather than merely
recorded. A supervised run is refused outright. An orphaned tool tree is
stopped first, and only then is the campaign moved to `failed`.

The leader's identity is recorded next to the pgid because a pgid alone
is not proof of identity. Once every process in a group has exited, the
leader's pid is free for the kernel to reuse, and an unrelated process
that later becomes a group leader inherits that number. Signalling on the
pgid alone would kill a stranger. `carmel/services/processes.py`
therefore re-confirms, from `/proc`, that the group's leader is the very
process this run launched before it signals anything.

Identity is the leader's **start time** (`/proc/<pid>/stat` field 22),
not its command line. Start time is the only reuse-proof pin: the kernel
does not carry it over when it recycles a pid, so a group leader that
started at a different instant is not this run's, whatever it is running.
The command line is recorded too — as a human-readable label and a
fallback for records written before the start-time pin — but it is *not*
the identity, for two reasons. A reused pid can rerun the same argv, so
the command line is too weak on its own. And it is also wrong for the
deployment that actually ships: a `conda run` launch execs a `#!`
wrapper, and the kernel rewrites its argv to prepend the interpreter, so
the argv Carmel passed to `Popen` never matches what `/proc` reports.
Recovery records the kernel's own view, read back at launch, so a later
recovery compares like with like.

That check has three distinct ways to come back negative, and they are
**not** the same finding:

| What `/proc` shows | Meaning | Carmel's response |
|--------------------|---------|-------------------|
| Leader started at a *different* time | The id was reused, which is only possible once the group emptied — so this run's processes have ended | `unsupervised`; the run is over |
| Leader is gone, group still alive | Most likely T3 and RMG outliving the `conda run` that launched them | `unknown`; neither signalled nor called over |
| No `/proc` at all, or no recorded identity | Nothing can be established | `unknown`; same |

Collapsing the middle row into the first is a data-corruption bug, not a
cosmetic one: it reports a live RMG as finished, and abandoning then
writes a terminal run record while the tool keeps writing into the same
workspace. A campaign in that state is reported honestly, with its
process group id, and the operator is asked to stop the tree by hand —
after which the group is empty and the ordinary path applies. Refusing to
guess is the correct direction to fail in; the alternatives are killing a
stranger's processes or corrupting a live run.

Once a signal *has* been sent, the question changes from "is this group
still recognizable?" to "is anything in it still running?" A SIGTERM that
the leader honours and a descendant ignores leaves the group alive with
its leader gone — no longer identifiable, but emphatically not stopped.
Escalation to SIGKILL therefore waits on the group falling quiet, counting
only processes that are actually executing: a zombie answers `killpg` but
holds no file descriptors and can write nothing.

## Approval Policy

Compute-side approval is enforced for T3 and ARC actions. The
`ApprovalPolicy` model is designed so future experiment and literature
actions fit the same framework:

| Field                                      | Default | Effect |
|--------------------------------------------|---------|--------|
| `auto_approve_t3_under_cpu_hours`          | 10.0    | T3 runs ≤ this estimate auto-approve |
| `auto_approve_arc_under_cpu_hours`         | 5.0     | ARC runs ≤ this estimate auto-approve |
| `require_approval_for_experiments`         | True    | (reserved) |
| `require_approval_for_literature`          | False   | (reserved) |

The deterministic planner always asks the approval engine before marking
an action as auto-approved.

### The cumulative-budget gate

The policy thresholds above are per-action; on their own they would let a
sequence of small auto-approved runs spend past the campaign's declared
`Budgets.cpu_hours`. `carmel/services/authorization.py` therefore runs
ONE combined gate (`decide_requirement`) for every planned action — T3
and ARC symmetrically: the per-kind policy threshold, a per-adapter
`ExecutionEnvelope` cap on what a single action may cost, and the
campaign's *remaining* budget (declared budget minus
`carmel/services/spend.py`'s consumed + reserved CPU-hours). An action
auto-approves only if all of them clear; the action whose estimate
crosses the remaining budget escalates to the user. The launch paths
re-run the same gate against the then-current remaining budget and raise
`BudgetExceededError` — before any state transition — if approval is
missing, so the gate cannot be raced by planning ahead of spending.
Every envelope decision is appended to the decision log.

## Adapters: T3, ARC, and the shared launcher

`carmel/adapters/` holds one adapter per external tool plus
`_launcher.py`, the shared subprocess machinery both adapters delegate
to: process-group launch (`start_new_session`), whole-tree
SIGTERM→SIGKILL termination on timeout, the live-tree registry with its
`atexit` sweep, executable discovery, and conda-environment resolution
(`$T3_CONDA_ENV`/`$ARC_CONDA_ENV` etc.). The per-tool adapters supply
their own layout constants, grace periods, and loggers.

### ARC Adapter

`carmel/adapters/arc.py` runs **one standalone ARC job** (thermochemistry
/ rates for explicitly requested species and reactions), where T3 drives
a whole generation/refinement loop. Same protocol shape as the T3
adapter: build a typed `input.yml` from the campaign/action, invoke
`ARC.py` as a subprocess in ARC's own environment, then normalize ARC's
real project tree (`<project>_info.yml`, `output/output.yml`) into
`DiagnosticsV1`, cross-checking the requested labels against ARC's
per-entry `success`/`converged` flags so an exit code of 0 cannot
masquerade as converged chemistry. All ARC layout assumptions live in
the `ARCLayout` constants block. A plan targets ARC via
`generate_arc_plan` (`run_arc` actions), and execution flows
`approved_for_execution → running_arc → results_ready`.

### T3 Adapter

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

The Flask UI (`carmel serve`, `carmel/ui/app.py`) is the production
caller for both tools: `POST /plan` generates a T3 or standalone-ARC plan
depending on the selected tool, and `POST /run` dispatches on the planned
action's kind. It performs the `RUNNING_T3`/`RUNNING_ARC` transition in
the request thread and then hands the work to a daemon thread, returning
immediately. Executed inline, a run holds the request open for hours: the
browser times out, the redirect to the auto-refreshing dashboard never
arrives, and the whole running-state UX is unreachable from the tab that
started the run.

The *transition* stays synchronous on purpose. It is what rejects a
double-submitted run — with a `409`, in the request that submitted it —
rather than accepting it and racing a run already in flight.

Because the worker is a daemon thread, interpreter exit does not join it.
An `atexit` sweep therefore kills every still-running tool process tree on
shutdown, so stopping the server does not leave T3 and RMG behind — the
same orphaning described above, one layer up. It cannot help if Carmel
itself is `SIGKILL`ed; nothing running inside Carmel can.

## Diagnostics Schema (DiagnosticsV1)

The single Carmel-internal contract for tool output (T3 and standalone
ARC alike). Includes:

- per-observable sensitivity summaries (rates and thermo)
- `species_to_compute`, `reactions_to_compute`, `pdep_networks_to_compute`
- `level_of_theory` and `model_version` when reported
- `pdep_sensitivity_flag` and free-form `warnings`
- arbitrary `tool_metadata`

The dashboard reads only this schema; it never reads raw tool output.

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
| ARC     | Trusted                 | Real subprocess adapter (standalone jobs) |
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
