"""Command-line interface for Carmel."""

import argparse
import sys
from pathlib import Path

from carmel.config import validate_config_file
from carmel.paths import init_workspace
from carmel.version import __version__


def create_parser() -> argparse.ArgumentParser:
    """Create the Carmel CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="carmel",
        description="Carmel: closed-loop campaign manager for predictive chemical kinetics",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Show the Carmel version")

    validate = subparsers.add_parser("validate-config", help="Validate a configuration file")
    validate.add_argument("file", type=Path, help="Path to a YAML config file")

    init = subparsers.add_parser("init-workspace", help="Initialize a workspace directory")
    init.add_argument("directory", type=Path, help="Path to the workspace directory")

    return parser


def _cmd_version() -> int:
    """Print the Carmel version."""
    print(f"carmel {__version__}")
    return 0


def _cmd_validate_config(file: Path) -> int:
    """Validate a config file and report results."""
    errors = validate_config_file(file)
    if errors:
        print("Config validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Config is valid.")
    return 0


def _cmd_init_workspace(directory: Path) -> int:
    """Initialize a workspace directory."""
    try:
        root = init_workspace(directory)
    except OSError as e:
        print(f"Failed to initialize workspace: {e}", file=sys.stderr)
        return 1
    print(f"Workspace initialized at {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the appropriate command.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        return _cmd_version()
    if args.command == "validate-config":
        return _cmd_validate_config(args.file)
    if args.command == "init-workspace":
        return _cmd_init_workspace(args.directory)

    parser.print_help()
    return 1


def cli() -> None:
    """Console script entrypoint."""
    sys.exit(main())
