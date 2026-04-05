"""Path utilities and workspace initialization for Carmel."""

from pathlib import Path

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
