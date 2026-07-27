#!/usr/bin/env bash
# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
#
# Fetch the external chemistry stack Carmel drives: RMG-Py, RMG-database, ARC
# and T3. An existing checkout is reused as-is, never overwritten and never
# silently updated — a `git pull` into a tree someone is working in is not this
# script's call to make. Pass --update to opt into fast-forward pulls.

set -euo pipefail

# shellcheck source=devtools/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

UPDATE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --update) UPDATE=true ;;
        -h | --help)
            cat <<EOF
Usage: $0 [--update]

  --update   Fast-forward existing checkouts (default: leave them untouched)

Environment:
  CARMEL_STACK_ROOT   Where to place the repositories (default: Carmel's parent
                      directory). CI sets it to \$HOME.
  RMG_PATH, RMG_DB_PATH, ARC_PATH, T3_PATH
                      Point an individual repository somewhere else, e.g. at a
                      checkout you already have.
EOF
            exit 0
            ;;
        *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
done

# Ask git, rather than looking for a `.git` directory: in a worktree or a
# submodule `.git` is a regular file, and those are perfectly good checkouts to
# build against.
is_git_checkout() {
    git -C "$1" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

clone_or_reuse() {
    local name="$1" url="$2" path="$3" remote matched=false
    if [[ -d "$path" ]] && is_git_checkout "$path"; then
        have "$name at $path"
        # An existing checkout is used as-is, so say out loud when it is not the
        # repository this expects — a stale fork or a repurposed directory
        # otherwise produces a stack nobody can reproduce. Every remote is
        # checked, not just `origin`: the canonical remote is often named
        # something else.
        while read -r remote; do
            if [[ "${remote%.git}" == "${url%.git}" ]]; then
                matched=true
            fi
        done < <(git -C "$path" remote -v 2>/dev/null | awk '{print $2}' | sort -u)
        if [[ "$matched" == false ]]; then
            printf '    NOTE: %s at %s has no remote pointing at %s. Using it anyway.\n' "$name" "$path" "$url"
        fi
        if [[ "$UPDATE" == true ]]; then
            info "Updating $name"
            git -C "$path" pull --ff-only
        fi
        return
    fi
    [[ -e "$path" ]] && die "$path exists but is not a git checkout. Move it aside or set a different path."
    info "Cloning $name into $path"
    git clone --depth 1 "$url" "$path"
}

info "Chemistry stack root: $CARMEL_STACK_ROOT"
mkdir -p "$CARMEL_STACK_ROOT"

clone_or_reuse RMG-Py "$CARMEL_RMG_PY_URL" "$RMG_PATH"
clone_or_reuse RMG-database "$CARMEL_RMG_DB_URL" "$RMG_DB_PATH"
clone_or_reuse ARC "$CARMEL_ARC_URL" "$ARC_PATH"
clone_or_reuse T3 "$CARMEL_T3_URL" "$T3_PATH"

info "Chemistry stack ready."
