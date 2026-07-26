# Carmel

[![CI](https://github.com/DanaResearchGroup/Carmel/actions/workflows/ci.yml/badge.svg)](https://github.com/DanaResearchGroup/Carmel/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DanaResearchGroup/Carmel/branch/main/graph/badge.svg)](https://codecov.io/gh/DanaResearchGroup/Carmel)
[![version](https://img.shields.io/badge/version-0.1.0-informational.svg)](https://github.com/DanaResearchGroup/Carmel)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
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

### Prerequisites

- Python 3.12+
- [Conda](https://docs.conda.io/en/latest/)

### Setup

```bash
# Clone the repository
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel

# Create and activate the conda environment
conda env create -f environment.yml
conda activate crml_env

# Install Carmel in editable mode with dev dependencies
make install
```

### Running real T3

T3 and ARC have Python requirements incompatible with Carmel's own
environment, so real T3 execution expects T3 (with ARC) installed in a
**separate** conda environment. Name that environment with `T3_CONDA_ENV`:

```bash
export T3_CONDA_ENV=t3_env
```

Carmel then launches T3 with `conda run -n t3_env`, which runs the
environment's activation hooks. This is not the same as naming the
environment's interpreter, and the difference is not cosmetic: ARC depends
on Open Babel, whose conda package sets `BABEL_LIBDIR`/`BABEL_DATADIR` from
an activation hook. Skip the hook and Open Babel loads no plugins, so
`import arc` — and therefore `import t3` — fails outright.

`T3_PYTHON` remains supported for deployments with no conda in the picture,
and names T3's interpreter directly:

```bash
export T3_PYTHON=/path/to/conda/envs/t3_env/bin/python
```

If neither is set, Carmel falls back to its own interpreter — only
correct if T3/ARC happen to be installed directly into `crml_env`. See
[docs/development.md](docs/development.md#three-env-deployment-model)
for the full three-env layout (`rmg_env` / `t3_env` / `crml_env`).

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
make install     # Editable install with dev deps
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
