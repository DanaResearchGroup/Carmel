DEVTOOLS_DIR := devtools
# The dev extra is named so every target syncs the toolchain it runs (ruff,
# pytest, mypy, mkdocs) from the lock. `uv run` syncs inexactly: it adds what
# the lock says is missing and leaves an installed extra (such as `agents`) in place.
UV_RUN := uv run --locked --extra dev

.PHONY: help install install-dev install-agents-dev install-stack install-rmg install-t3 install-carmel \
        test lint typecheck format check docs

.DEFAULT_GOAL := help

help:
	@echo "Installation:"
	@echo "  install          Install everything: the chemistry stack, the two conda tool envs, Carmel"
	@echo "  install-dev      Sync Carmel and its dev dependencies into the uv project .venv"
	@echo "  install-agents-dev  install-dev, plus the [agents] extra (pydantic-ai/pypdf/rdkit)"
	@echo "  install-stack    Clone RMG-Py, RMG-database, ARC and T3 (reuses existing checkouts)"
	@echo "  install-rmg      Create rmg_env and build RMG-Py"
	@echo "  install-t3       Create t3_env and install ARC and T3 into it"
	@echo "  install-carmel   Sync Carmel into .venv with uv, record the tool paths"
	@echo ""
	@echo "Development:"
	@echo "  test             Run the tests with coverage"
	@echo "  lint             Lint and format check"
	@echo "  typecheck        Type check with mypy"
	@echo "  format           Auto-fix formatting and lint"
	@echo "  check            lint + typecheck + test"
	@echo "  docs             Serve the documentation locally"
	@echo ""
	@echo "Every install target is safe to re-run; see docs/installation.md."

# The full install. Around 40 minutes from cold, seconds when already installed:
# each step checks what is actually on disk rather than trusting a flag.
install:
	bash $(DEVTOOLS_DIR)/install_all.sh

# Carmel alone, into the uv project environment (.venv), exactly as uv.lock pins
# it. This is what the required CI lane runs: it needs Carmel importable and the
# dev tools present, and has no business building a chemistry stack in order to
# lint a docstring.
install-dev:
	uv sync --locked --extra dev

# Same as install-dev, plus the optional `agents` extra (pydantic-ai, pypdf,
# rdkit). This is what the "agents" CI lane runs -- the plain required `lint`/
# `test` lane deliberately does NOT install this extra, so it never exercised
# the LLM/PDF/RDKit code path at all, and `make typecheck` passed there while
# missing type errors that only surface once pydantic-ai/pypdf/rdkit's stubs
# are actually on the path.
install-agents-dev:
	uv sync --locked --extra dev --extra agents

install-stack:
	bash $(DEVTOOLS_DIR)/install_stack.sh

install-rmg:
	bash $(DEVTOOLS_DIR)/install_rmg.sh

install-t3:
	bash $(DEVTOOLS_DIR)/install_t3.sh

install-carmel:
	bash $(DEVTOOLS_DIR)/install_carmel.sh

test:
	$(UV_RUN) pytest --cov --cov-report=term-missing

lint:
	$(UV_RUN) ruff check .
	$(UV_RUN) ruff format --check .

typecheck:
	@$(UV_RUN) python -c "import pydantic_ai, pypdf, rdkit" 2>/dev/null || \
		{ echo "make typecheck requires the agents extra: run 'make install-agents-dev' first" >&2; exit 1; }
	$(UV_RUN) mypy carmel Carmel.py

format:
	$(UV_RUN) ruff format .
	$(UV_RUN) ruff check --fix .

check: lint typecheck test

docs:
	$(UV_RUN) mkdocs serve
