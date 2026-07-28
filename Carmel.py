# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

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
    serve.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Carmel config file whose 'agents' section enables the literature auto-run",
    )

    literature = subparsers.add_parser("literature", help="Run the literature-search step for an existing campaign")
    literature.add_argument("--campaign", type=str, required=True, help="Campaign ID")
    literature.add_argument("--workspaces", type=Path, default=None, help="Parent workspaces directory")
    literature.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Carmel config file with an 'agents' section (required for a real run)",
    )

    return parser


def _load_agent_config(config_file: Path | None) -> object | None:
    """Load the ``agents`` section of a Carmel config file, if any."""
    if config_file is None:
        return None
    from carmel.config import load_config

    return load_config(config_file).agents


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


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _cmd_serve(workspaces: Path | None, host: str, port: int, debug: bool, config: Path | None) -> int:
    """Launch the local Flask UI."""
    if debug and host not in LOOPBACK_HOSTS:
        print(
            f"Refusing to serve with --debug on non-loopback host {host!r}: "
            "the Werkzeug debugger allows remote code execution. "
            "Use --host 127.0.0.1 or drop --debug.",
            file=sys.stderr,
        )
        return 1
    from carmel.ui import create_app

    try:
        agent_config = _load_agent_config(config)
    except (OSError, ValueError) as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1
    app = create_app(workspaces_root=workspaces, agent_config=agent_config)  # type: ignore[arg-type]
    print(f"Carmel UI listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
    return 0


def _cmd_literature(campaign_id: str, workspaces: Path | None, config: Path | None) -> int:
    """Run the literature-search step for an existing campaign.

    Uses the same single-owner service hook as campaign creation
    (:func:`carmel.services.campaigns.start_literature_at_creation`);
    the CLI never owns its own copy of the auto-run logic.
    """
    from carmel.services.campaigns import (
        find_campaign_workspace,
        load_campaign,
        start_literature_at_creation,
    )
    from carmel.ui.app import _resolve_workspaces_root

    try:
        agent_config = _load_agent_config(config)
    except (OSError, ValueError) as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1
    if agent_config is None:
        print(
            "No agent config available: pass --config FILE whose 'agents' section "
            "sets provider, consent and API key env vars.",
            file=sys.stderr,
        )
        return 1

    workspaces_root = _resolve_workspaces_root(workspaces)
    ws = find_campaign_workspace(workspaces_root, campaign_id)
    if ws is None:
        print(f"Campaign {campaign_id!r} not found under {workspaces_root}", file=sys.stderr)
        return 1
    campaign = load_campaign(ws)

    from carmel.agents.bridge import AgentBridgeError

    try:
        outcome = start_literature_at_creation(ws, campaign, agent_config)  # type: ignore[arg-type]
    except AgentBridgeError as e:
        print(f"Literature run unavailable (consent/API key): {e}", file=sys.stderr)
        return 1
    if outcome.result is None:
        # Report the reason the service actually determined. This used to list three
        # candidate causes it had not checked, which meant a run that started and died
        # on a provider outage was reported as a configuration problem.
        print(f"Literature run did not start: {outcome.explain()}.", file=sys.stderr)
        return 1
    result = outcome.result
    print(f"Literature action {result.action_id} finished with outcome {result.outcome.value}")
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
        return _cmd_serve(args.workspaces, args.host, args.port, args.debug, args.config)
    if args.command == "literature":
        return _cmd_literature(args.campaign, args.workspaces, args.config)

    parser.print_help()
    return 1


def cli() -> None:
    """Console script entrypoint."""
    sys.exit(main())


if __name__ == "__main__":
    # Without this, `python Carmel.py literature ...` (and `python -m Carmel`) parse
    # nothing, run nothing, print nothing, and exit 0 -- indistinguishable from a
    # successful run that happened to produce no output. Only the installed `carmel`
    # console script worked, so the most obvious way to invoke a checkout was also the
    # most quietly misleading.
    cli()
