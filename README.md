# Carmel

[![CI](https://github.com/DanaResearchGroup/Carmel/actions/workflows/ci.yml/badge.svg)](https://github.com/DanaResearchGroup/Carmel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DanaResearchGroup/Carmel/branch/main/graph/badge.svg)](https://codecov.io/gh/DanaResearchGroup/Carmel)
[![version](https://img.shields.io/badge/version-0.1.0-informational.svg)](https://github.com/DanaResearchGroup/Carmel)
[![Python 3.14+](https://img.shields.io/badge/python-3.14%2B-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

```
        ██████╗ █████╗ ██████╗ ███╗   ███╗███████╗██╗
        ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║
        ██║     ███████║██████╔╝██╔████╔██║█████╗  ██║
        ██║     ██╔══██║██╔══██╗██║╚██╔╝██║██╔══╝  ██║
        ╚██████╗██║  ██║██║  ██║██║ ╚═╝ ██║███████╗███████╗
         ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝
             Agentic Predictive Chemical Kinetics Engine
```

**Closed-loop campaign manager for predictive chemical kinetics.**

Carmel automates the iterative cycle of building, validating, refining, validating, and revising predictive chemical kinetic models. It orchestrates simulation tools, literature evidence, experiment design, and model revision through a bounded ensemble of specialized agents with full provenance tracking and human-in-the-loop governance.

## Installation

You need [conda](https://conda-forge.org/download/) on your `PATH`, git, and a
C toolchain. Then:

```bash
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel
make install
conda activate crml_env
carmel --help
```

`make install` clones the external chemistry stack (RMG-Py, RMG-database, ARC,
T3), builds the three conda environments Carmel needs, installs Carmel, and
records where everything ended up. Budget around 40 minutes the first time;
re-running takes seconds, because every step checks what is already on disk.

Working on Carmel's own code and not running real campaigns? `make install-dev`
installs Carmel and its dev dependencies into the current environment and
stops there.

Run `make help` for the individual targets, and see
[docs/installation.md](docs/installation.md) for pointing the installer at
checkouts you already have, and for troubleshooting.

### Why three environments

T3, ARC and RMG-Py have mutually exclusive Python requirements, so they cannot
share one environment. Carmel runs them as **separate processes**: `rmg_env`
holds RMG-Py and Arkane, `t3_env` holds T3 and ARC together, and `crml_env`
holds Carmel.

Carmel launches T3 with `conda run -n $T3_CONDA_ENV`, which runs that
environment's activation hooks. That is not the same as naming the
environment's interpreter, and the difference is not cosmetic: ARC depends on
Open Babel, whose conda package sets `BABEL_LIBDIR`/`BABEL_DATADIR` from an
activation hook. Skip the hook and Open Babel loads no plugins, so `import arc`
— and therefore `import t3` — fails outright.

`make install` writes `T3_CONDA_ENV` (along with `T3_PATH`, `RMG_PATH` and
`RMG_DB_PATH`) into `crml_env`'s own activation hook, so `conda activate
crml_env` is all the setup there is. `T3_PYTHON` remains supported for
deployments with no conda in the picture, and names T3's interpreter directly:

```bash
export T3_PYTHON=/path/to/conda/envs/t3_env/bin/python
```

If neither is set, Carmel falls back to its own interpreter — only correct if
T3/ARC happen to be installed directly into `crml_env`. See
[docs/architecture.md](docs/architecture.md#three-env-deployment-model-and-t3-invocation)
for the full layout.

## Usage

```bash
# Show version
carmel version

# Validate a configuration file
carmel validate-config config.yaml

# Initialize a new workspace
carmel init-workspace my-campaign
```

### Configuration

Carmel workspaces are configured via YAML:

```yaml
workspace_name: ethanol-combustion
workspace_root: ./workspaces/ethanol
logging_level: INFO
budgets:
  cpu_hours: 500.0
  experiment_budget: 10000.0
metadata:
  author: researcher
  description: Ethanol oxidation mechanism development
```

### Workspace Structure

`carmel init-workspace` creates the standard directory scaffold:

```
my-campaign/
├── benchmarks/    # Curated benchmark bundles and credence records
├── evidence/      # Literature memos, extracted records, source links
├── models/        # Generated mechanism versions and diffs
├── provenance/    # Hashes, versions, tool settings, costs
├── reports/       # Final and intermediate reports
└── runs/          # Executed tool runs and statuses
```

## Development

```bash
make test        # Run tests with coverage
make lint        # Lint and format check
make typecheck   # Type check with mypy
make check       # All of the above
make format      # Auto-fix formatting
make install-dev # Editable install with dev deps, current environment only
```

To run a specific test:

```bash
pytest tests/test_config.py
pytest tests/test_config.py::TestCarmelConfig::test_minimal_config
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Carmel is the open-source **core**. Contributions are welcome under the terms in
[CONTRIBUTING.md](CONTRIBUTING.md); a signed [Contributor License Agreement](CLA.md)
(or DCO sign-off for small fixes) is required before a first contribution can be
merged.
