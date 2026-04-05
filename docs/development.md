# Development Guide

## Environment Setup

### Prerequisites

- Python 3.12+
- [Conda](https://docs.conda.io/en/latest/)

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

## Testing

Carmel follows test-driven development. Write tests before or alongside implementation.

```bash
make test        # Run all tests with coverage

# Or target specific tests directly:
pytest tests/test_config.py
pytest tests/test_config.py::TestCarmelConfig
pytest tests/test_config.py::TestCarmelConfig::test_minimal_config
```

### Test Expectations

Every public function must have tests covering:

- Trivial / empty input
- Standard / normal case
- Realistic / complex case
- Edge cases
- Invalid input / failure paths

CLI commands must test exit codes, stdout, and stderr.

**Coverage target: 90%+**

## Linting

```bash
make lint        # Check for lint and format errors
make format      # Auto-fix formatting
```

## Type Checking

```bash
make typecheck
```

Carmel uses mypy in strict mode. All functions require complete type annotations.

## Full Verification

Run all checks before committing:

```bash
make check
```

## CI

GitHub Actions runs on every push to `main` and on pull requests:

1. Lint (`ruff check`)
2. Format check (`ruff format --check`)
3. Type check (`mypy carmel`)
4. Tests with coverage (`pytest --cov --cov-report=term-missing`)

All checks must pass before merging.

## Adding New Functionality

1. Write tests first
2. Implement the feature
3. Add type hints and Google-style docstrings
4. Run `make check`
5. Update documentation if the feature is user-facing
