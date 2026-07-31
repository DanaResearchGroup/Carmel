# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Path utilities and workspace initialization for Carmel."""

import os
from pathlib import Path

#: Where campaign workspaces live when nothing else says otherwise.
#:
#: Top-level ``~/carmel_workspaces``, per commit a986543 ("Move default workspaces dir to
#: ``~/carmel_workspaces``: repo-independent, user-level, no longer dependent on cwd").
#:
#: This value was previously changed to ``("runs", "carmel", "workspaces")`` with a comment
#: arguing that a tool should not claim a home-level name. That was a reversal of the
#: decision above, made without flagging it as one, and it split live state across two
#: roots on at least one machine -- campaigns created before the change sat in
#: ``~/carmel_workspaces`` while later ones went under ``~/runs/``, and a bare
#: ``carmel requests --campaign <id>`` could not see the other half. Restored here.
#:
#: If nesting under ``~/runs/`` is wanted per-machine, that is what the
#: ``$CARMEL_WORKSPACES`` override below is for -- it needs no code change.
#:
#: Callers should prefer :func:`default_workspaces_root`, which honours that override.
DEFAULT_WORKSPACES_SUBPATH: tuple[str, ...] = ("carmel_workspaces",)

#: Environment variable that overrides :func:`default_workspaces_root`.
WORKSPACES_ROOT_ENV_VAR = "CARMEL_WORKSPACES"

#: Default location of the file-backed daily cost ledger.
#:
#: Deliberately NOT inside a campaign workspace. The daily cap exists to bound what
#: this machine spends in aggregate across every campaign in a day, so a per-workspace
#: ledger would hand each new workspace a fresh full allowance and silently turn the
#: cap into a per-campaign one.
DEFAULT_DAILY_LEDGER_SUBPATH: tuple[str, ...] = (".carmel", "daily_ledger.json")

#: Environment variable that overrides :func:`default_daily_ledger_path`.
DAILY_LEDGER_ENV_VAR = "CARMEL_DAILY_LEDGER"

WORKSPACE_SUBDIRS: tuple[str, ...] = (
    "benchmarks",
    "evidence",
    "models",
    "provenance",
    "reports",
    "runs",
)


def normalize_path(path: Path | str) -> Path:
    """Normalize a path by expanding user home and resolving to absolute.

    Args:
        path: A file system path as string or Path object.

    Returns:
        The fully resolved absolute path.
    """
    return Path(path).expanduser().resolve()


def resolve_path(path: Path | str, base: Path | None = None) -> Path:
    """Resolve a path, optionally relative to a base directory.

    If *path* is relative and *base* is given, the path is joined to *base*
    before resolving. Tilde expansion is applied to both arguments.

    Args:
        path: The path to resolve.
        base: Optional base directory for relative paths.

    Returns:
        The resolved absolute path.
    """
    p = Path(path).expanduser()
    if not p.is_absolute() and base is not None:
        p = Path(base).expanduser() / p
    return p.resolve()


def ensure_directory(path: Path | str) -> Path:
    """Ensure a directory exists, creating it and parents if necessary.

    Args:
        path: Directory path to create or verify.

    Returns:
        The resolved path to the directory.

    Raises:
        NotADirectoryError: If the path exists but is not a directory.
    """
    resolved = normalize_path(path)
    if resolved.exists() and not resolved.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def is_valid_workspace_name(name: str) -> bool:
    """Check whether a string is a valid workspace name.

    Valid names contain only alphanumeric characters, hyphens, and underscores,
    and do not start with a hyphen or dot.

    Args:
        name: The candidate workspace name.

    Returns:
        True if the name is valid.
    """
    if not name:
        return False
    if name[0] in ("-", "."):
        return False
    return all(c.isalnum() or c in ("-", "_") for c in name)


def init_workspace(directory: Path | str) -> Path:
    """Initialize a workspace directory with standard Carmel subdirectories.

    Creates the workspace root and all standard subdirectories
    (benchmarks, evidence, models, provenance, reports, runs).
    Safe to call on an existing workspace — existing files are preserved.

    Args:
        directory: Path to the workspace root.

    Returns:
        The resolved workspace root path.
    """
    root = ensure_directory(directory)
    for subdir in WORKSPACE_SUBDIRS:
        (root / subdir).mkdir(exist_ok=True)
    return root


def default_workspaces_root() -> Path:
    """Resolve the workspaces root from the environment, or the packaged default.

    Preference order:

    1. ``$CARMEL_WORKSPACES`` when set and non-empty (an empty value is treated as
       unset, so ``CARMEL_WORKSPACES=`` in a sourced env file cannot silently redirect
       every campaign to the filesystem root).
    2. :data:`DEFAULT_WORKSPACES_SUBPATH` under the user's home.

    Returns:
        Absolute path to the workspaces root. The directory is NOT created here --
        callers that need it on disk create it, so a read-only query never has a side
        effect.
    """
    env = os.environ.get(WORKSPACES_ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home().joinpath(*DEFAULT_WORKSPACES_SUBPATH)


def resolve_workspaces_root(workspaces_root: Path | str | None) -> Path:
    """Resolve the workspaces root, preferring an explicit value over the default.

    Preference order:

    1. ``workspaces_root`` when given (an explicit ``--workspaces`` or constructor
       argument always wins).
    2. :func:`default_workspaces_root` — ``$CARMEL_WORKSPACES``, then the packaged
       default.

    Lives here rather than in the UI because the CLI and the web UI must resolve the
    SAME root: two copies of this rule let them disagree about where a campaign
    lives, which an operator experiences as a campaign that vanished. Two CLI
    subcommands were reaching into ``carmel.ui.app`` for the private helper that used
    to hold it, which made a UI-internal detail load-bearing for the CLI.

    Returns:
        Absolute path to the workspaces root. The directory is NOT created here.
    """
    if workspaces_root is not None:
        return Path(workspaces_root).expanduser()
    return default_workspaces_root()


def default_daily_ledger_path() -> Path:
    """Resolve the machine-wide daily cost ledger path.

    Preference order mirrors :func:`default_workspaces_root`:

    1. ``$CARMEL_DAILY_LEDGER`` when set and non-empty.
    2. :data:`DEFAULT_DAILY_LEDGER_SUBPATH` under the user's home.

    A path is ALWAYS returned. ``BudgetLedger`` treats ``daily_ledger_path=None`` as
    "no daily cap at all" and skips the check entirely, so a resolver that could
    return ``None`` would make the cap quietly optional -- which is how it came to be
    unenforced in production despite being configurable.

    Returns:
        The daily ledger path. The file and its parent are NOT created here; the
        ledger creates them on first write, so a read-only query has no side effect.
    """
    env = os.environ.get(DAILY_LEDGER_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home().joinpath(*DEFAULT_DAILY_LEDGER_SUBPATH)
