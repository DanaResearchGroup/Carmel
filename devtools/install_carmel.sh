#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Sync Carmel into its uv project environment (.venv) from uv.lock, and record
# where the external tools live in carmel-tools.env, which both
# `uv run --env-file carmel-tools.env …` and `set -a; source carmel-tools.env`
# read.

set -euo pipefail

# shellcheck source=devtools/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

setup_uv

# Always re-run: Carmel's own code changes every commit, and a sync that is
# already current costs a couple of seconds.
info "Syncing Carmel (with dev dependencies) into $CARMEL_ROOT/.venv from uv.lock"
(cd "$CARMEL_ROOT" && uv sync --locked --extra dev)

# Refuse a value that cannot be single-quoted in a way both bash and a dotenv
# parser read back unchanged. Run on every value before the file is opened, so a
# refusal never leaves a partial file behind.
require_quotable() {
    local name="$1" value="$2"
    case "$value" in
        *"'"* | *$'\n'*)
            die "$name=$value holds a single quote or a newline, which carmel-tools.env cannot record. Move it or set a different path."
            ;;
    esac
}

# Print one `export NAME='value'` line.
write_export() {
    printf "export %s='%s'\n" "$1" "$2"
}

# Skipped when the stack is not installed, so a Carmel-only environment is not
# handed paths to directories that do not exist — and a file left by an
# earlier full install is removed rather than left exporting dead paths.
if [[ -d "$T3_PATH" ]]; then
    # Not a warning. T3 reads the database path from ARC's settings and dies in
    # its constructor with "expected str, bytes or os.PathLike object, not
    # NoneType" before it creates a single iteration. Recording a path that is
    # not there produces an environment that looks installed and cannot run.
    [[ -d "$RMG_DB_PATH" ]] ||
        die "no RMG-database at $RMG_DB_PATH, which T3 needs. Run 'make install-stack', or set RMG_DB_PATH."
    for name in T3_CONDA_ENV T3_PATH ARC_PATH RMG_PATH RMG_DB_PATH; do
        require_quotable "$name" "${!name}"
    done

    {
        cat <<'EOF'
# Written by Carmel's devtools/install_carmel.sh. Re-run 'make install-carmel'
# to refresh. Read it with `uv run --env-file carmel-tools.env …`, or with
# `set -a; source carmel-tools.env; set +a`.
#
# T3_CONDA_ENV is how Carmel launches T3: it runs 'conda run -n <env>', so the
# environment's own activation hooks run too. That is not the same as naming
# the environment's interpreter — ARC needs Open Babel, whose conda package
# exports BABEL_LIBDIR and BABEL_DATADIR from an activation hook, and without
# them Open Babel loads zero plugins and importing ARC fails.
EOF
        write_export T3_CONDA_ENV "$T3_CONDA_ENV"
        write_export T3_PATH "$T3_PATH"
        # Carmel launches ARC as 'python ARC.py' through 'conda run -n <t3_env>',
        # so it needs the checkout on ARC_PATH to find the script.
        if [[ -d "$ARC_PATH" ]]; then
            write_export ARC_PATH "$ARC_PATH"
        fi
        echo "# Read by ARC's settings module, which is where T3 gets the database path from."
        write_export RMG_PATH "$RMG_PATH"
        write_export RMG_DB_PATH "$RMG_DB_PATH"
    } >"$CARMEL_TOOLS_ENV_FILE.tmp"
    # Renamed into place, never written in place, so an interrupted write
    # never leaves a half-written file behind.
    mv -f "$CARMEL_TOOLS_ENV_FILE.tmp" "$CARMEL_TOOLS_ENV_FILE"
    info "Tool paths recorded in $CARMEL_TOOLS_ENV_FILE"
else
    if [[ -f "$CARMEL_TOOLS_ENV_FILE" ]]; then
        rm -f "$CARMEL_TOOLS_ENV_FILE"
        info "Removed the stale tool-path file: no T3 checkout at $T3_PATH any more."
    fi
    info "No T3 checkout at $T3_PATH — skipping the tool-path file."
    info "Run 'make install' for the full stack, then Carmel can execute T3."
fi

if [[ -f "$CARMEL_TOOLS_ENV_FILE" ]]; then
    info "Carmel ready in $CARMEL_ROOT/.venv. Run it with: uv run --env-file $CARMEL_TOOLS_ENV_FILE carmel --help"
else
    info "Carmel ready in $CARMEL_ROOT/.venv. Run it with: uv run carmel --help"
fi
