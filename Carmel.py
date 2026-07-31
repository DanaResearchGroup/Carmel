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

    new_campaign = subparsers.add_parser(
        "new-campaign",
        help="Create a campaign from the 'campaign' section of a config file",
    )
    new_campaign.add_argument("--config", type=Path, required=True, help="Carmel config file with a 'campaign' section")
    new_campaign.add_argument(
        "--workspaces",
        type=Path,
        default=None,
        help="Parent workspaces directory (default: the config's workspace_root)",
    )

    requests_cmd = subparsers.add_parser(
        "requests",
        help="List papers awaiting a human, or hand one to Carmel",
    )
    requests_cmd.add_argument("--campaign", type=str, required=True, help="Campaign ID")
    requests_cmd.add_argument("--workspaces", type=Path, default=None, help="Parent workspaces directory")
    requests_cmd.add_argument(
        "--add",
        type=Path,
        default=None,
        help=(
            "A downloaded paper to admit, or a directory of them. A file is identity-checked "
            "immediately and the verdict reported. A directory is swept non-recursively: every "
            "direct-child file is admitted by content (filenames are irrelevant), one failure "
            "does not abort the rest, and a per-file report plus a summary line is printed. "
            "Exit code is 0 only if at least one file was admitted and none were rejected."
        ),
    )
    requests_cmd.add_argument(
        "--collect",
        action="store_true",
        help=(
            "Sweep every file already dropped into the inbox directory and admit or reject "
            "each one, instead of admitting a single file with --add."
        ),
    )
    requests_cmd.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Which request --add satisfies. Inferred when unambiguous; required when it is not.",
    )
    requests_cmd.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Carmel config file, for the artifact size limit",
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


def _cmd_new_campaign(config_file: Path, workspaces: Path | None) -> int:
    """Create a campaign from a config file's ``campaign:`` section.

    Exists so an operator never has to hand-write a Python script that builds a
    ``CampaignInput`` -- a private mock of Carmel's own API that inevitably rots against
    the real one. The config file is the single description of a run.
    """
    from carmel.config import load_config
    from carmel.services.campaigns import (
        CampaignWorkspaceConflictError,
        MissingCampaignConfigError,
        create_campaign_from_config,
    )

    try:
        config = load_config(config_file)
    except (OSError, ValueError) as exc:
        print(f"Could not load {config_file}: {exc}")
        return 1

    # `--workspaces` means the same thing in EVERY command: the PARENT directory that
    # holds one subdirectory per campaign. `create_campaign_from_config` instead takes
    # the campaign's own workspace root, so the parent has to be joined with the
    # workspace name here. Passing the parent straight through created the campaign
    # directly in it, and `carmel requests --workspaces <same dir>` -- which scans
    # subdirectories via `find_campaign_workspace` -- then could never find the campaign
    # it had just made. The flag's help text already promised "parent"; only this
    # command disagreed.
    campaign_root: Path | None = None
    if workspaces is not None:
        parent = workspaces.expanduser().resolve()
        campaign_root = (parent / config.workspace_name).resolve()
        # `workspace_name` is only validated as non-blank, so it can still carry path
        # separators or `..`. Without this it would escape the parent the operator named.
        if not campaign_root.is_relative_to(parent):
            print(
                f"Refusing to create a campaign outside {parent}: workspace_name {config.workspace_name!r} escapes it.",
                file=sys.stderr,
            )
            return 1

    try:
        campaign = create_campaign_from_config(config, workspaces_root=campaign_root)
    except CampaignWorkspaceConflictError as exc:
        # Fail closed, matching the --workspaces containment check above: print the
        # actionable reason to stderr and create/modify nothing.
        print(str(exc), file=sys.stderr)
        return 1
    except MissingCampaignConfigError as exc:
        print(str(exc))
        return 1
    except (OSError, ValueError) as exc:
        print(f"Could not create the campaign: {exc}")
        return 1

    print(f"Campaign ID : {campaign.campaign_id}")
    print(f"Workspace   : {campaign.workspace_root}")
    print()
    print(f"Next: carmel literature --campaign {campaign.campaign_id} --config {config_file}")
    return 0


def _admit_directory(ws: Path, directory: Path, *, max_bytes: int) -> int:
    """Admit every direct-child file of ``directory`` by content-based matching.

    Loops :func:`carmel.services.acquisition.admit_file` with ``slug=None`` over each
    entry, so a folder of publisher-named downloads (spaces, parentheses and all) can
    be handed to Carmel in one command instead of one ``--add`` per paper -- filenames
    are irrelevant, matching is on document content via ``_infer_slug``. Non-recursive:
    only direct children are considered, and subdirectories are reported, not entered.
    Dotfiles, non-regular files, and obvious partial-download junk (``.part``,
    ``.crdownload``, ``.tmp``) are skipped and reported, not treated as papers. One
    file's failure never aborts the batch: every entry is attempted and its outcome
    printed before moving on.

    Exit code: 0 if at least one file was admitted and none were rejected; 1
    otherwise -- including an empty directory, which is reported rather than treated
    as a silent success.
    """
    from carmel.schemas.acquisition import AcquisitionStatus
    from carmel.services.acquisition import AlreadyAcquired, admit_file

    try:
        entries = sorted(directory.iterdir())
    except OSError as exc:
        print(f"Could not read directory {directory}: {exc}")
        return 1

    if not entries:
        print(f"{directory} is empty; nothing to admit.")
        return 1

    accepted = 0
    rejected = 0
    skipped = 0

    for entry in entries:
        if entry.is_dir():
            skipped += 1
            print(f"SKIPPED   {entry.name} (subdirectory; --add on a directory is non-recursive)")
            continue
        if entry.name.startswith("."):
            skipped += 1
            print(f"SKIPPED   {entry.name} (dotfile)")
            continue
        if entry.suffix.lower() in {".part", ".crdownload", ".tmp"}:
            skipped += 1
            print(f"SKIPPED   {entry.name} (partial download)")
            continue
        if not entry.is_file():
            skipped += 1
            print(f"SKIPPED   {entry.name} (not a regular file)")
            continue

        try:
            request = admit_file(ws, entry, slug=None, max_bytes=max_bytes)
        except AlreadyAcquired as exc:
            # Caught BEFORE ValueError, which it subclasses. Re-offering a paper the
            # store already holds is the ordinary shape of re-running an ingest over a
            # download folder, so it is a skip: it must not count as a rejection, and
            # must not by itself turn a successful run into a non-zero exit.
            skipped += 1
            print(f"SKIPPED   {entry.name} (already acquired as {exc.slug})")
            continue
        except (OSError, ValueError) as exc:
            rejected += 1
            print(f"REJECTED  {entry.name}: {exc}")
            continue

        if request.status == AcquisitionStatus.FULFILLED:
            accepted += 1
            print(f"ACCEPTED  {entry.name} -> {request.slug}")
        else:
            rejected += 1
            print(f"REJECTED  {entry.name} -> {request.slug}")
        print(f"          {request.identity_note}")
        print()

    print(f"{accepted} accepted, {rejected} rejected, {skipped} skipped")
    # A folder in which every paper was already acquired is a success, not a failure --
    # requiring `accepted` outright would fail the second run of an ingest that fully
    # succeeded the first time.
    if rejected:
        return 1
    return 0 if (accepted or skipped) else 1


def _cmd_requests(
    campaign_id: str,
    workspaces: Path | None,
    add: Path | None,
    slug: str | None,
    config_file: Path | None,
    collect: bool = False,
) -> int:
    """List papers awaiting a human, or admit one (or all) that have been obtained.

    Without ``--add``/``--collect`` this prints the queue. With ``--add`` pointed at a
    file, it copies that single file into the inbox under the right name AND runs the
    identity check immediately. With ``--add`` pointed at a directory, it instead
    admits every direct-child file by content (see :func:`_admit_directory`) -- so the
    operator can download several publisher-named papers into one arbitrary folder and
    hand Carmel the whole folder in a single command, without renaming anything. With
    ``--collect`` it instead sweeps every file already dropped into the workspace's own
    inbox directory in one shot. The long feedback loop and the per-paper typing were
    both what made this step painful.
    """
    if add is not None and collect:
        print("--collect and --add are mutually exclusive. Use one or the other.")
        return 1

    from carmel.config import AgentBudgetConfig, load_config
    from carmel.paths import default_workspaces_root
    from carmel.schemas.acquisition import AcquisitionStatus
    from carmel.services.acquisition import (
        AlreadyAcquired,
        admit_file,
        collect_inbox,
        drop_path_for,
        inbox_dir,
        pending_requests,
    )
    from carmel.services.campaigns import find_campaign_workspace

    root = workspaces.expanduser() if workspaces is not None else default_workspaces_root()
    ws = find_campaign_workspace(root, campaign_id)
    if ws is None:
        print(f"No campaign {campaign_id!r} under {root}")
        return 1

    max_bytes = AgentBudgetConfig().max_artifact_bytes
    if config_file is not None:
        try:
            config = load_config(config_file)
        except (OSError, ValueError) as exc:
            print(f"Could not load {config_file}: {exc}")
            return 1
        if config.agents is not None:
            max_bytes = config.agents.budget.max_artifact_bytes

    if collect:
        changed = collect_inbox(ws, max_bytes=max_bytes)
        if not changed:
            print("Nothing new in the inbox.")
            print(f"Drop papers into: {inbox_dir(ws)}")
            return 0
        accepted = 0
        rejected = 0
        for request in changed:
            if request.status == AcquisitionStatus.FULFILLED:
                accepted += 1
                print(f"ACCEPTED  {request.title}")
            else:
                rejected += 1
                print(f"REJECTED  {request.title}")
            print(f"          {request.identity_note}")
            print()
        print(f"{accepted} accepted, {rejected} rejected")
        return 1 if rejected else 0

    if add is not None:
        if add.is_dir():
            if slug is not None:
                print("--slug identifies a single request; it cannot be combined with a directory --add.")
                return 1
            return _admit_directory(ws, add, max_bytes=max_bytes)
        try:
            request = admit_file(ws, add, slug=slug, max_bytes=max_bytes)
        except AlreadyAcquired as exc:
            # Exit 0: the operator asked for this paper to be in the store, and it is.
            print(f"SKIPPED   {add.name}")
            print(f"          already acquired as {exc.slug}; the evidence store is unchanged.")
            return 0
        except (OSError, ValueError) as exc:
            print(f"Could not admit {add}: {exc}")
            return 1
        if request.status == AcquisitionStatus.FULFILLED:
            print(f"ACCEPTED  {request.title}")
            print(f"          {request.identity_note}")
            print()
            print("Re-run `carmel literature` to ground findings against it.")
            return 0
        print(f"REJECTED  {request.title}")
        print(f"          {request.identity_note}")
        print()
        print("Nothing was admitted to the evidence store. Drop the correct document.")
        return 1

    pending = pending_requests(ws)
    if not pending:
        print("No papers are awaiting acquisition.")
        return 0

    print(f"{len(pending)} paper(s) awaiting a human:")
    print()
    for request in pending:
        print(f"  {request.title}")
        if request.doi:
            print(f"    doi   : {request.doi}")
        print(f"    get   : {request.landing_url}")
        detail = f" -- {request.detail}" if request.detail else ""
        print(f"    why   : {request.reason.value}{detail}")
        if request.status == AcquisitionStatus.REJECTED and request.identity_note:
            print(f"    last  : REJECTED -- {request.identity_note}")
        print(f"    then  : carmel requests --campaign {campaign_id} --add <file> --slug {request.slug}")
        print(f"    or copy to: {drop_path_for(ws, request.slug)}")
        print()
    print(f"Or drop all the files into {inbox_dir(ws)} and run `carmel requests --campaign {campaign_id} --collect`.")
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

    if args.command == "new-campaign":
        return _cmd_new_campaign(args.config, args.workspaces)

    if args.command == "requests":
        return _cmd_requests(args.campaign, args.workspaces, args.add, args.slug, args.config, args.collect)

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
