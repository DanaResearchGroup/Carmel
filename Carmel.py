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
        description="Carmel: Agentic Predictive Chemical Kinetics Engine",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Show the Carmel version")

    validate = subparsers.add_parser("validate-config", help="Validate a configuration file")
    validate.add_argument("file", type=Path, help="Path to a YAML config file")

    init = subparsers.add_parser("init-workspace", help="Initialize a workspace directory")
    init.add_argument("directory", type=Path, help="Path to the workspace directory")

    serve = subparsers.add_parser("serve", help="Launch the local Flask UI")
    serve.add_argument("--workspaces", type=Path, default=None, help="Parent workspaces directory")
    serve.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=5000, help="Bind port")
    serve.add_argument("--debug", action="store_true", help="Enable Flask debug mode")

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


def _cmd_serve(workspaces: Path | None, host: str, port: int, debug: bool) -> int:
    """Launch the local Flask UI."""
    from carmel.ui import create_app

    app = create_app(workspaces_root=workspaces)
    print(f"Carmel UI listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
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
    if args.command == "serve":
        return _cmd_serve(args.workspaces, args.host, args.port, args.debug)

    parser.print_help()
    return 1


def cli() -> None:
    """Console script entrypoint."""
    sys.exit(main())
