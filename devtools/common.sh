#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Shared settings and helpers for Carmel's installers. Source this file; do not
# run it.
#
# THE RULE EVERY INSTALLER FOLLOWS: branch on ACTUAL STATE — does this conda
# environment exist, is this extension compiled for this interpreter — never on
# a "the cache said so" flag passed in from outside. That is what makes
# `make install` safe to run unconditionally: on a fresh laptop it builds
# everything, on a warm CI runner it verifies and exits in seconds, and there
# is no third state where a step is skipped because something *else* was
# cached.

CARMEL_DEVTOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CARMEL_ROOT="$(cd -- "$CARMEL_DEVTOOLS_DIR/.." && pwd)"

# Where the external chemistry stack lives. Defaults to Carmel's sibling
# directory, which is the layout ARC and T3 assume. CI sets it to $HOME.
CARMEL_STACK_ROOT="${CARMEL_STACK_ROOT:-$(cd -- "$CARMEL_ROOT/.." && pwd)}"

# Each repository can be pointed somewhere else individually, so an existing
# checkout is reused rather than cloned a second time. RMG_PATH and RMG_DB_PATH
# keep the names ARC's settings module already reads.
RMG_PATH="${RMG_PATH:-$CARMEL_STACK_ROOT/RMG-Py}"
RMG_DB_PATH="${RMG_DB_PATH:-$CARMEL_STACK_ROOT/RMG-database}"
ARC_PATH="${ARC_PATH:-$CARMEL_STACK_ROOT/ARC}"
T3_PATH="${T3_PATH:-$CARMEL_STACK_ROOT/T3}"

# The three environments. Their python requirements are mutually exclusive, so
# no single environment can hold the stack; see docs/installation.md.
RMG_ENV="${RMG_ENV:-rmg_env}"
T3_CONDA_ENV="${T3_CONDA_ENV:-t3_env}"
CARMEL_ENV="${CARMEL_ENV:-crml_env}"

CARMEL_RMG_PY_URL="${CARMEL_RMG_PY_URL:-https://github.com/ReactionMechanismGenerator/RMG-Py.git}"
CARMEL_RMG_DB_URL="${CARMEL_RMG_DB_URL:-https://github.com/ReactionMechanismGenerator/RMG-database.git}"
CARMEL_ARC_URL="${CARMEL_ARC_URL:-https://github.com/ReactionMechanismGenerator/ARC.git}"
CARMEL_T3_URL="${CARMEL_T3_URL:-https://github.com/ReactionMechanismGenerator/T3.git}"

info() { printf '>>> %s\n' "$*"; }
have() { printf '    already there: %s\n' "$*"; }
die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

# Check for conda without sourcing its shell hook. Every command below runs
# through `conda run`, which activates the target environment itself — so no
# installer ever needs `conda activate`, and none of them has to survive the
# shell-state surgery that entails.
setup_conda() {
    command -v conda >/dev/null 2>&1 ||
        die "conda is not on PATH. Install Miniforge first: https://conda-forge.org/download/"
}

# -F, not a regex: an environment named `t3.env` must not match `t3xenv`.
conda_env_exists() {
    conda env list | awk '{print $1}' | grep -Fqx -- "$1"
}

# `conda run` buffers by default, which hides progress during builds that take
# tens of minutes.
conda_run() {
    conda run -n "$1" --no-capture-output "${@:2}"
}
