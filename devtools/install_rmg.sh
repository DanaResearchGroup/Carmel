#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Create rmg_env and compile RMG-Py into it. T3 launches RMG as a subprocess in
# this environment; Carmel never imports it.

set -euo pipefail

# shellcheck source=devtools/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

setup_conda

[[ -d "$RMG_PATH" ]] || die "RMG-Py not found at $RMG_PATH. Run 'make install-stack' first."

if conda_env_exists "$RMG_ENV"; then
    have "conda environment $RMG_ENV"
else
    info "Creating $RMG_ENV from $RMG_PATH/environment.yml"
    conda env create -n "$RMG_ENV" -f "$RMG_PATH/environment.yml"
fi

# `make` in RMG-Py does two separate things: it registers the package in the
# environment (pip install -e) and it compiles Cython extensions into the
# SOURCE tree. Those land in two different places, and either can be present
# without the other — a restored environment with a fresh source tree has the
# registration but no extensions; a restored source tree in a rebuilt
# environment has the extensions but no registration. Neither half is what
# callers need, so ask for the thing that actually matters: can this
# environment import a compiled RMG module?
#
# From a neutral directory, because inside $RMG_PATH `import rmgpy` finds the
# source tree whether or not it was ever installed.
#
# The probe also insists the module it found lives under $RMG_PATH. Importable
# alone is too weak a canary: a leftover editable install pointing at a
# different checkout, or an ambient PYTHONPATH, satisfies it while the tree
# this install is supposed to be building stays unbuilt.
rmg_is_built() {
    (cd / && conda run -n "$RMG_ENV" python -c '
import pathlib, sys
import rmgpy, rmgpy.molecule.graph  # noqa: F401
wanted = pathlib.Path(sys.argv[1]).resolve()
found = pathlib.Path(rmgpy.__file__).resolve()
sys.exit(0 if wanted in found.parents else 1)
' "$RMG_PATH" >/dev/null 2>&1)
}

if rmg_is_built; then
    have "RMG-Py from $RMG_PATH, importable in $RMG_ENV"
else
    info "Building RMG-Py in $RMG_PATH (this takes a while)"
    (cd "$RMG_PATH" && conda_run "$RMG_ENV" make)
fi

info "RMG-Py ready in $RMG_ENV."
