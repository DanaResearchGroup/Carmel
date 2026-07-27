#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Create crml_env, install Carmel into it, and record where the external tools
# live so an activated crml_env can find them.

set -euo pipefail

# shellcheck source=devtools/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

setup_conda

if conda_env_exists "$CARMEL_ENV"; then
    have "conda environment $CARMEL_ENV"
else
    info "Creating $CARMEL_ENV from $CARMEL_ROOT/environment.yml"
    conda env create -n "$CARMEL_ENV" -f "$CARMEL_ROOT/environment.yml"
fi

# Always re-run: Carmel's own code changes every commit, and an editable
# install that is already current costs a couple of seconds.
info "Installing Carmel (editable, with dev dependencies) into $CARMEL_ENV"
(cd "$CARMEL_ROOT" && conda_run "$CARMEL_ENV" python -m pip install -e ".[dev]")

# The tool locations are written as a conda activation hook rather than into a
# shell rc file: they belong to this environment, they follow it, and they go
# away with it. `conda activate crml_env` is then the whole of the setup a user
# has to remember.
#
# Skipped when the stack is not installed, so a Carmel-only environment is not
# handed paths to directories that do not exist — and any hook left by an
# earlier full install is removed rather than left exporting dead paths.
env_prefix="$(conda run -n "$CARMEL_ENV" python -c 'import sys; print(sys.prefix)' | tail -n 1 | tr -d '\r')"
activate_hook="$env_prefix/etc/conda/activate.d/carmel.sh"
deactivate_hook="$env_prefix/etc/conda/deactivate.d/carmel.sh"

if [[ -d "$T3_PATH" ]]; then
    # Not a warning. T3 reads the database path from ARC's settings and dies in
    # its constructor with "expected str, bytes or os.PathLike object, not
    # NoneType" before it creates a single iteration. Recording a path that is
    # not there produces an environment that looks installed and cannot run.
    [[ -d "$RMG_DB_PATH" ]] ||
        die "no RMG-database at $RMG_DB_PATH, which T3 needs. Run 'make install-stack', or set RMG_DB_PATH."

    mkdir -p "$(dirname "$activate_hook")"

    # Values go through %q rather than straight into the heredoc. This file is
    # sourced by every `conda activate`, so an unquoted path containing a space
    # would silently truncate a variable and one containing $(...) would be
    # executed.
    {
        cat <<'EOF'
# Written by Carmel's devtools/install_carmel.sh. Re-run 'make install-carmel'
# to refresh.
#
# T3_CONDA_ENV is how Carmel launches T3: it runs 'conda run -n <env>', so the
# environment's own activation hooks run too. That is not the same as naming
# the environment's interpreter — ARC needs Open Babel, whose conda package
# exports BABEL_LIBDIR and BABEL_DATADIR from an activation hook, and without
# them Open Babel loads zero plugins and importing ARC fails.
EOF
        printf 'export T3_CONDA_ENV=%q\n' "$T3_CONDA_ENV"
        printf 'export T3_PATH=%q\n' "$T3_PATH"
        echo "# Read by ARC's settings module, which is where T3 gets the database path from."
        printf 'export RMG_PATH=%q\n' "$RMG_PATH"
        printf 'export RMG_DB_PATH=%q\n' "$RMG_DB_PATH"
    } >"$activate_hook.tmp"
    # Renamed into place, never written in place: this file is sourced by every
    # `conda activate`, so a write interrupted half way through would break the
    # environment for good rather than merely fail this install.
    mv -f "$activate_hook.tmp" "$activate_hook"

    # There is deliberately NO deactivate.d counterpart, and any left by an
    # earlier install is removed.
    #
    # `conda run -n t3_env` — exactly how Carmel launches T3 — first
    # DEACTIVATES the active environment, running its deactivate.d hooks. A
    # hook that unset these variables therefore removed them from T3's
    # environment at the very moment T3 needed them: T3 reads the database path
    # from ARC's settings, got None, and died in its constructor with
    # "expected str, bytes or os.PathLike object, not NoneType" before creating
    # a single iteration. Leaving four inert path variables set after
    # `conda deactivate` is a much smaller cost than that.
    rm -f "$deactivate_hook"
    info "Tool paths recorded in $activate_hook"
else
    if [[ -f "$activate_hook" ]]; then
        rm -f "$activate_hook" "$deactivate_hook"
        info "Removed the stale tool-path hook: no T3 checkout at $T3_PATH any more."
    fi
    info "No T3 checkout at $T3_PATH — skipping the tool-path activation hook."
    info "Run 'make install' for the full stack, then Carmel can execute T3."
fi

info "Carmel ready in $CARMEL_ENV."
