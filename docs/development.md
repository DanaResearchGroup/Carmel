# Development Guide

## Environment Setup

### Prerequisites

- Python 3.12+
- [Conda](https://docs.conda.io/en/latest/)
- For real T3 execution: T3, ARC, RMG-Py, RMG-database installed in the same env

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel

# Create conda environment
conda env create -f environment.yml
conda activate crml_env

# Install in editable mode with dev dependencies
make install
```

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
   `T3_info.yml`, `input.yml`, and `RMG/pdep/network*.py` files copied
   from the upstream T3 repo's own test data; see
   `tests/fixtures/t3/README.md` for provenance. These tests are how we
   guarantee the parser/normalization layer keeps matching T3's real
   output schema even when no live T3 is available.

2. **Real subprocess tests** (`TestT3AdapterRealSubprocess`) — only run
   when T3 can actually be imported (`is_t3_importable()` returns
   True). This is stricter than just having T3 on `sys.path`: T3 imports
   ARC, which currently uses `from distutils.spawn import find_executable`
   in `arc/main.py`. Since `distutils` was removed from the Python
   standard library in 3.12, T3 cannot be imported on 3.12 until that
   migrates to `shutil.which` upstream. While that blocker is in
   effect, these tests are skipped both locally and in CI.

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
(RMG-Py / RMG-database / ARC / T3 from GitHub) and runs the T3-dependent
tests. It is currently **best-effort, blocked on the upstream ARC
distutils issue** described in the T3-dependent Tests section above.

The `continue-on-error: true` flags are scoped narrowly to the heavy
**install steps** (where the conda environment can legitimately fail
on flaky upstream); the **test step itself does not skip on error**.
When the ARC fix lands upstream, `is_t3_importable()` will return True,
the real subprocess test will execute, and the lane should be promoted
to required. At that point the `continue-on-error` lines should also
be reviewed and likely removed.

The job display name is `Real T3/ARC/RMG (best-effort, blocked on ARC
distutils)` so reviewers see the status at a glance.

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
