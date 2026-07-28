#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Install everything Carmel needs to run a real campaign: the external
# chemistry stack, the three conda environments, and Carmel itself.
#
# Safe to re-run. Every step checks what is actually on disk, so a second run
# verifies in seconds instead of rebuilding.

set -euo pipefail

# shellcheck source=devtools/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

STACK_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --update) STACK_ARGS+=("--update") ;;
        -h | --help)
            cat <<EOF
Usage: $0 [--update]

Installs, in order:
  1. RMG-Py, RMG-database, ARC and T3 into \$CARMEL_STACK_ROOT
  2. $RMG_ENV   — RMG-Py and Arkane
  3. $T3_CONDA_ENV    — T3 and ARC together
  4. $CARMEL_ENV  — Carmel itself, plus a hook recording the tool paths

  --update   Fast-forward existing upstream checkouts

See docs/installation.md for the environment variables that redirect any of
this at checkouts you already have.
EOF
            exit 0
            ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

run_devtool() { bash "$CARMEL_DEVTOOLS_DIR/$1" "${@:2}"; }

info "Installing the full Carmel stack. This takes around 40 minutes from cold."
run_devtool install_stack.sh "${STACK_ARGS[@]}"
run_devtool install_rmg.sh
run_devtool install_t3.sh
run_devtool install_carmel.sh

cat <<EOF

Done. To use Carmel:

    conda activate $CARMEL_ENV
    carmel --help

The tool paths (T3_CONDA_ENV, T3_PATH, ARC_PATH, RMG_PATH, RMG_DB_PATH) are
exported by that activation, so nothing else has to be set by hand.
EOF
