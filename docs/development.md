# Development Guide

## Environment Setup

### Prerequisites

- [Conda](https://conda-forge.org/download/), git, and a C toolchain
- For real T3 execution: T3 and ARC in their own environment (see
  "Three-env deployment model" below) — do **not** try to install them
  into `crml_env`, their Python pins are mutually exclusive with Carmel's.
  `make install` sets all three environments up for you.

### Initial Setup

Working on Carmel's own code, with no real T3 execution:

```bash
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel
conda env create -f environment.yml
conda activate crml_env
make install-dev
```

To also run real campaigns, use `make install` instead — it builds all three
environments and the external chemistry stack. See
[installation.md](installation.md) for the full guide, the individual targets,
and how to point it at checkouts you already have.

### Three-env deployment model

Carmel, T3, and RMG have mutually exclusive Python requirements and
normally live in three separate conda environments:

| Env        | Contains          | Python pin       |
|------------|-------------------|------------------|
| `rmg_env`  | RMG-Py, Arkane    | `>=3.9,<3.12`    |
| `t3_env`   | T3 **and** ARC    | `=3.14`          |
| `crml_env` | Carmel itself     | `>=3.14`         |

T3 imports ARC in-process, so they must share an environment (`t3_env`);
T3 in turn launches RMG as a subprocess under `rmg_env`. Name T3's
environment with the `T3_CONDA_ENV` environment variable:

```bash
export T3_CONDA_ENV=t3_env
```

Carmel launches T3 as `conda run -n t3_env --no-capture-output python ...`,
the same way T3 itself launches RMG under `rmg_env`
(`t3/runners/rmg_runner.py`).

**Why `conda run` and not just the interpreter path.** A conda environment
is not reducible to its interpreter. Packages may ship activation hooks in
`$CONDA_PREFIX/etc/conda/activate.d/`, and running `<env>/bin/python`
directly never executes them. ARC depends on Open Babel, whose conda
package sets `BABEL_LIBDIR` and `BABEL_DATADIR` from exactly such a hook.
Without them Open Babel loads zero plugins and `from openbabel import
pybel` raises `ValueError: not enough values to unpack` while building its
format table — so `import arc` fails, and with it `import t3`.

`T3_PYTHON` still names T3's interpreter directly and is the right choice
when conda is not involved at all:

```bash
export T3_PYTHON=/path/to/conda/envs/t3_env/bin/python
```

Resolution order is `T3_CONDA_ENV` → `T3_PYTHON` → `sys.executable`. The
last is only correct for a single-env developer setup where T3 and ARC are
installed directly into `crml_env`. See
[architecture.md](architecture.md#three-env-deployment-model-and-t3-invocation)
for full details, including how an invalid value for either variable is
handled.

## Running the UI

```bash
carmel serve --workspaces ./workspaces
```

Then open http://127.0.0.1:5000 in a browser.

The `--workspaces` argument is the parent directory under which campaign
workspaces are created. Defaults to `$CARMEL_WORKSPACES` or `./workspaces`.

## Creating a Campaign

1. Open the Carmel UI in your browser.
2. Click **New Campaign**.
3. Fill in the structured form:
   - **Workspace name** — short identifier (becomes the directory name)
   - **Initial mixture** — one component per line: `species,mole_fraction[,smiles]`
   - **Target observables** — one per line: `name[,species]`
   - **Reactor systems** — one per line: `type,Tmin,Tmax,Pmin,Pmax[,residence_s]`
   - **CPU hours** and **experiment budget**
4. Submit. You'll be redirected to the dashboard for the new campaign.

## Approval and Execution

From the dashboard:

1. **Generate plan** — produces a deterministic Phase 1 plan with cost estimate.
2. If the estimate exceeds the policy threshold, the plan moves to
   **plan_pending_approval**. Use **Approve** or **Reject** to decide.
3. Once approved, click **Run T3** to invoke the T3 subprocess.
4. The dashboard updates with diagnostics and graphical compute-selection
   SVGs once the run completes.

## Free-text Intake

The dashboard has a free-text box. Anything you paste is processed by
the stub intake parser into `intake_review.md` — this is **advisory
only** and never becomes canonical campaign state without an explicit
structured form submission.

## Testing

Carmel follows test-driven development. Write tests before or alongside
implementation.

```bash
make test                                       # Run all tests with coverage

# Or target specific tests directly:
pytest tests/test_schemas.py
pytest tests/test_services.py
pytest tests/test_t3_adapter.py
pytest tests/test_ui.py
```

### Test Organization

| File                       | What it covers                                  |
|----------------------------|-------------------------------------------------|
| `test_version.py`          | Phase 0 version surface                         |
| `test_paths.py`            | Path utilities and workspace init               |
| `test_config.py`           | Config loading and validation                   |
| `test_logger.py`           | Logger setup, archival, header/footer           |
| `test_Carmel.py`           | CLI commands                                    |
| `test_schemas.py`          | Phase 1 pydantic schemas                        |
| `test_services.py`         | Artifacts, state machine, approvals, planner, drawing, intake, provenance |
| `test_t3_adapter.py`       | T3 input building, output parsing, failure handling, optional subprocess |
| `test_ui.py`               | Flask routes via the test client                |

### Test Expectations

Every public function must have tests covering:

- Trivial / empty input
- Standard / normal case
- Realistic / complex case
- Edge cases
- Invalid input / failure paths

CLI commands must test exit codes, stdout, and stderr.

**Coverage target:** 90%+

### T3-dependent Tests

There are two layers of T3 testing:

1. **Golden fixture parser tests** (`TestGoldenFixture` in
   `tests/test_t3_adapter.py`) — these run unconditionally against a
   small set of **real captured T3 artifacts** under
   `tests/fixtures/t3/sample_project/`. The fixture contains real
   `<project>_info.yml`, `input.yml`, and `RMG/pdep/network*.py` files copied
   from the upstream T3 repo's own test data; see
   `tests/fixtures/t3/README.md` for provenance. These tests are how we
   guarantee the parser/normalization layer keeps matching T3's real
   output schema even when no live T3 is available.

2. **Real subprocess tests** (`TestT3AdapterRealSubprocess`) — only run
   when T3 can actually be imported (`is_t3_importable()` returns
   True), which means a working `t3_env` reachable through
   `$T3_CONDA_ENV`. `make install` sets that up; without it these tests
   skip.

   A skip is silent by design, which makes it dangerous: the suite goes
   green having never launched T3. The `tools` lane therefore *fails*
   when T3 is unreachable, rather than reporting the skip and passing.

Locally, all parser/normalizer/execution-path tests still run; only the
real-subprocess test is skipped.

## Linting and Type Checking

```bash
make lint        # Check for lint and format errors
make format      # Auto-fix formatting
make typecheck   # Run mypy in strict mode
```

Carmel uses mypy in strict mode. All public functions require complete
type annotations and Google-style docstrings.

## Full Verification

```bash
make check       # lint + typecheck + test
```

## CI

GitHub Actions has **two lanes** that run on every push to `main` and
on pull requests.

### Required lane (must pass for merge)

- **`lint`** — `make lint` (ruff check + ruff format check) and
  `make typecheck` (mypy strict).
- **`test`** — pytest with branch coverage, plus a **packaging smoke
  step** that exercises:
  - `carmel version`
  - `carmel --help`
  - `carmel serve --help`
  - `from carmel.ui import create_app; create_app().test_client().get('/')`

  This catches console-script regressions (e.g. broken entrypoints,
  missing template folder) without spinning up the full server.

### Best-effort lane (`tools`)

The `tools` job installs the full external chemistry stack
(RMG-Py / RMG-database / ARC / T3 from GitHub) by running `make install` —
the same command the README gives users — and then runs the T3-dependent
tests. The job display name is `Real T3/ARC/RMG integration (best-effort)`.

`continue-on-error: true` is set at the **job** level, not per step. So
nothing in this lane can block a merge, including a genuine test failure.
That is deliberate for now — the lane builds a large third-party stack and a
flaky upstream must not redden a PR — but it has a sharp edge worth knowing:
`gh run list` reports the *run* as successful while the *job* is red. Check
the job, or `gh pr checks`, not the run.

The lane is expected to be green with the real integration test executing.
Once it has been for a while, three things should change together: drop
`continue-on-error`, drop "(best-effort)" from the job name, and add the job
to the Protect Main ruleset. Required checks pin by job name, so renaming
without updating the ruleset orphans the requirement and blocks every PR
while every check shows green.

Caches are keyed on the upstream commit SHAs rather than on Carmel's own
`environment.yml`, since those revisions are what actually gets installed.
No install step is gated on a cache hit: the installers decide from what is
on disk, so the caches supply state and never a verdict.

## Adding New Functionality

1. Write tests first
2. Implement the feature in the appropriate module:
   - Schemas → `carmel/schemas/`
   - Business logic → `carmel/services/`
   - External tool I/O → `carmel/adapters/`
   - HTTP → `carmel/ui/app.py` (keep route handlers thin)
3. Add type hints and Google-style docstrings
4. Run `make check`
5. Update documentation if the feature is user-facing
