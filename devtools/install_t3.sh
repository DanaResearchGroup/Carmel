#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Create t3_env and install ARC and T3 into it. They share one environment
# because T3 imports ARC in-process (arc.molecule, arc.species, ARC(...) in
# T3/t3/main.py); Carmel reaches both by launching T3 as a subprocess.

set -euo pipefail

# shellcheck source=devtools/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

setup_conda

[[ -d "$ARC_PATH" ]] || die "ARC not found at $ARC_PATH. Run 'make install-stack' first."
[[ -d "$T3_PATH" ]] || die "T3 not found at $T3_PATH. Run 'make install-stack' first."

if conda_env_exists "$T3_CONDA_ENV"; then
    have "conda environment $T3_CONDA_ENV"
else
    info "Creating $T3_CONDA_ENV from T3's and ARC's environment files"
    conda env create -n "$T3_CONDA_ENV" -f "$T3_PATH/environment.yml"
    conda env update -n "$T3_CONDA_ENV" -f "$ARC_PATH/environment.yml"
fi

# Deliberately NOT skipped when the environment already exists. The editable
# install's registration lives in the environment, but the Cython extensions it
# compiles (arc.molecule.graph and friends) live in ARC's source tree. Skip this
# because the environment was there and you get a registered package whose C
# extensions do not exist, so `import arc` fails on arc.molecule.graph — which
# in turn makes `import t3` fail and every real-T3 test skip. Re-running is
# cheap: setuptools rebuilds only what is out of date.
info "Installing ARC (editable) into $T3_CONDA_ENV"
conda_run "$T3_CONDA_ENV" python -m pip install -e "$ARC_PATH"
info "Installing T3 (editable) into $T3_CONDA_ENV"
conda_run "$T3_CONDA_ENV" python -m pip install -e "$T3_PATH"

# Prove it before claiming it. "pip install succeeded" is not the same as "ARC
# and T3 import": the Cython extensions may be missing, and pip resolving
# dependencies into a conda-solved environment can move a package out from
# under one of them. Both failures are silent until Carmel launches T3 and gets
# nothing back, so ask the question here, where the traceback is still useful.
#
# As with RMG, the check is on identity and not merely importability —
# `import arc` succeeding says nothing about *which* ARC — and it imports
# arc.molecule.graph by name, because that Cython extension is the thing that
# goes missing when the compile is skipped, and a plain `import arc` need not
# touch it.
t3_smoke='
import pathlib, sys
import arc, t3
import arc.molecule.graph  # the Cython extension that goes missing
for module, wanted in ((arc, sys.argv[1]), (t3, sys.argv[2])):
    root = pathlib.Path(wanted).resolve()
    found = pathlib.Path(module.__file__).resolve()
    if root not in found.parents:
        raise SystemExit(f"{module.__name__} resolves to {found}, not under {root}")
'
info "Checking that $T3_CONDA_ENV imports ARC and T3 from the configured checkouts"
if ! (cd / && conda run -n "$T3_CONDA_ENV" python -c "$t3_smoke" "$ARC_PATH" "$T3_PATH" >/dev/null 2>&1); then
    (cd / && conda run -n "$T3_CONDA_ENV" --no-capture-output python -c "$t3_smoke" "$ARC_PATH" "$T3_PATH") || true
    die "$T3_CONDA_ENV cannot import ARC and T3 from $ARC_PATH and $T3_PATH (see above)."
fi

info "ARC and T3 ready in $T3_CONDA_ENV."
