# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Command-line interface for Carmel."""

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from carmel.config import validate_config_file
from carmel.paths import init_workspace
from carmel.version import __version__

if TYPE_CHECKING:
    # Type-only: keeps this module importable without eagerly pulling the schema
    # package at CLI start-up, matching how every other symbol here is imported
    # inside the function that uses it.
    from carmel.schemas import Campaign


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
            "Exit code is 0 only if at least one file was admitted or was already held, "
            "and none were rejected."
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

    corpus = subparsers.add_parser(
        "corpus-pass",
        help="Append a second literature pass over the papers this campaign already holds",
    )
    corpus.add_argument("--campaign", type=str, required=True, help="Campaign ID")
    corpus.add_argument(
        "--budget-tokens",
        type=int,
        default=None,
        help=(
            "How many model tokens this pass may consume. Required (except with "
            "--dispatch-queued, which runs a pass whose budget was named when it was "
            "appended) and explicit: an operator-authorised action names its own "
            "budget rather than drawing on the campaign's autonomous-spend ceiling. "
            "Tokens rather than dollars because tokens are what the run consumes and "
            "what the provider meters, so this number is the one that binds; the "
            "equivalent dollar cost is estimated and printed for you."
        ),
    )
    corpus.add_argument(
        "--dispatch-queued",
        action="store_true",
        help=(
            "Run the corpus pass that is ALREADY queued instead of appending a new "
            "one. Needed because a pass that requires approval is appended by one "
            "command and can only be dispatched after a human approves it -- and "
            "re-running the plain command then refuses, correctly, rather than "
            "queueing a second identical pass. Its budget is the one named when it "
            "was appended, so --budget-tokens is not accepted here: silently running "
            "under a different cap than the approver saw would defeat the approval."
        ),
    )
    corpus.add_argument(
        "--reread-all",
        action="store_true",
        help=(
            "Re-read documents an earlier pass already mined. By default a corpus "
            "pass reads only what is new, because the prompt for a given document is "
            "identical between passes and re-reading it buys the same answer twice. "
            "Use this when the question has actually changed -- a different model, or "
            "a revised prompt."
        ),
    )
    corpus.add_argument(
        "--allow-unauthenticated-legacy-roots",
        action="store_true",
        help=(
            "Read held artifacts whose stored text cannot be authenticated against "
            "anything, because they were stored before the digest that would bind it "
            "existed. By default a corpus pass refuses these -- the honest "
            "alternative is to re-extract the document into an authenticated record "
            "instead of setting this flag. This does NOT admit an artifact whose "
            "bytes are actually damaged; a failed integrity check is refused "
            "regardless of this flag."
        ),
    )
    corpus.add_argument("--workspaces", type=Path, default=None, help="Parent workspaces directory")
    corpus.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Carmel config file with an 'agents' section (required for a real run)",
    )
    corpus.add_argument(
        "--dry-run",
        action="store_true",
        help="Append the action and stop, without running it",
    )

    reextract = subparsers.add_parser(
        "reextract",
        help="Re-parse a stored artifact's raw.bin and append a new extraction record",
        description=(
            "Re-parse a stored artifact's raw.bin with today's extractor and append a new, "
            "separately-addressed extraction record under evidence/literature/<raw_sha256>/"
            "extractions/. The corpus pass PREFERS such a record over the root sidecar, so "
            "re-extracting is how a legacy artifact becomes readable without the "
            "unauthenticated-legacy-root opt-in, and dataset production now requires one. "
            "Dry run (the default) does the real read/parse/cleanliness check and reports "
            "what it would do without writing anything; pass --apply to actually write."
        ),
    )
    reextract.add_argument("--campaign", help="Campaign ID")
    reextract.add_argument(
        "--sha",
        default=None,
        help="raw_sha256 of a single stored artifact to re-extract (mutually exclusive with --all)",
    )
    reextract.add_argument(
        "--all",
        action="store_true",
        help="Re-extract every artifact in the campaign's evidence store (mutually exclusive with --sha)",
    )
    reextract.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Actually append the new extraction record. Without this flag the command is a "
            "dry run: it still does the real read/parse/cleanliness check, but writes nothing."
        ),
    )
    reextract.add_argument("--workspaces", type=Path, default=None, help="Parent workspaces directory")
    reextract.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Carmel config file with an 'agents' section (required; supplies "
            "budget.max_artifact_bytes -- no default is invented here)"
        ),
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
    from carmel.paths import resolve_workspaces_root
    from carmel.services.campaigns import (
        find_campaign_workspace,
        load_campaign,
        start_literature_at_creation,
    )

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

    workspaces_root = resolve_workspaces_root(workspaces)
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


_CONSUMER_NOTICE = (
    "NOTE: extraction records ARE read. The corpus pass prefers an authenticated current "
    "record over the root sidecar, and dataset production requires one, so appending a "
    "record here changes what later passes read."
)


def _looks_like_sha256(name: str) -> bool:
    """True if ``name`` has the shape of a lowercase hex sha256 digest.

    Used to scan ``evidence/literature/`` directly by directory name, deliberately
    bypassing ``list_artifacts()``/``list_artifacts_with_unreadable()``: those helpers
    read each artifact's meta.json to build their listing, so an artifact whose
    meta.json is unreadable would silently drop out of a ``--all`` run instead of
    being attempted and having its refusal reported.
    """
    return len(name) == 64 and all(c in "0123456789abcdef" for c in name)


def _reextract_one(ws: Path, *, raw_sha256: str, max_bytes: int, apply: bool) -> int:
    """Re-extract a single artifact and print its outcome. Never raises.

    Always previews first (real read/parse/cleanliness check, no write) so both the
    dry-run and --apply paths report "already present" from the exact same check --
    they can never disagree about whether a write is needed. Only when --apply is set
    AND the extraction is not already present does it actually write, via
    reextract_artifact.

    Returns 0 on success (including "already present" / dry-run preview), 1 on a
    refusal (ReextractionError), so callers can count failures for --all.
    """
    from carmel.services.reextraction import ReextractionError, preview_reextraction, reextract_artifact

    try:
        extraction_sha256, already_present = preview_reextraction(ws, raw_sha256=raw_sha256, max_bytes=max_bytes)
    except ReextractionError as exc:
        print(f"REFUSED         {raw_sha256}: {exc}")
        return 1

    if not apply:
        if already_present:
            print(f"DRY-RUN         {raw_sha256}: extraction {extraction_sha256} already present; nothing to write")
        else:
            print(f"DRY-RUN         {raw_sha256}: would write extraction {extraction_sha256}; nothing written")
        return 0

    if already_present:
        print(f"ALREADY-PRESENT {raw_sha256}: extraction {extraction_sha256}")
        return 0

    written_sha256 = reextract_artifact(ws, raw_sha256=raw_sha256, max_bytes=max_bytes)
    print(f"WRITTEN         {raw_sha256}: extraction {written_sha256}")
    return 0


def _cmd_reextract(
    campaign_id: str,
    workspaces: Path | None,
    config: Path | None,
    *,
    sha: str | None,
    all_artifacts: bool,
    apply: bool,
) -> int:
    """Re-parse a stored artifact's raw.bin and append a new extraction record.

    Dry run is the default (opposite polarity from ``corpus-pass --dry-run``, which
    opts IN to dry mode): here ``--apply`` opts IN to mutation, so an operator who
    forgets the flag gets a safe preview rather than an accidental write.

    No consumer reads extraction records yet -- see :mod:`carmel.services.reextraction`.
    This command only appends evidence for a future one.
    """
    if sha is not None and all_artifacts:
        print("--sha and --all are mutually exclusive. Use one or the other.", file=sys.stderr)
        return 1
    if sha is None and not all_artifacts:
        print("Either --sha <raw_sha256> or --all is required.", file=sys.stderr)
        return 1

    from carmel.paths import resolve_workspaces_root
    from carmel.services.campaigns import find_campaign_workspace
    from carmel.services.evidence import EVIDENCE_LITERATURE_DIR

    try:
        agent_config = _load_agent_config(config)
    except (OSError, ValueError) as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1
    if agent_config is None:
        print(
            "No agent config available: pass --config FILE whose 'agents' section "
            "sets budget.max_artifact_bytes.",
            file=sys.stderr,
        )
        return 1
    max_bytes = agent_config.budget.max_artifact_bytes  # type: ignore[attr-defined]

    workspaces_root = resolve_workspaces_root(workspaces)
    ws = find_campaign_workspace(workspaces_root, campaign_id)
    if ws is None:
        print(f"Campaign {campaign_id!r} not found under {workspaces_root}", file=sys.stderr)
        return 1

    print(_CONSUMER_NOTICE)

    if sha is not None:
        return _reextract_one(ws, raw_sha256=sha, max_bytes=max_bytes, apply=apply)

    evidence_root = ws / EVIDENCE_LITERATURE_DIR
    try:
        candidates = sorted(p.name for p in evidence_root.iterdir() if p.is_dir() and _looks_like_sha256(p.name))
    except OSError as exc:
        print(f"Could not scan evidence store at {evidence_root}: {exc}", file=sys.stderr)
        return 1

    if not candidates:
        print(f"No artifacts found under {evidence_root}.")
        return 0

    failures = 0
    for raw_sha256 in candidates:
        if _reextract_one(ws, raw_sha256=raw_sha256, max_bytes=max_bytes, apply=apply) != 0:
            failures += 1

    print(f"{len(candidates)} artifact(s) processed, {failures} refused.")
    return 1 if failures else 0


def _cmd_corpus_pass(
    campaign_id: str,
    budget_tokens: int | None,
    workspaces: Path | None,
    config: Path | None,
    *,
    dry_run: bool,
    reread_all: bool = False,
    dispatch_queued: bool = False,
    allow_unauthenticated_legacy_roots: bool = False,
) -> int:
    """Append a corpus pass to a campaign's plan and run it.

    Two steps, deliberately separable by ``--dry-run``: appending the action is the
    operator's authorisation and is recorded in the plan whether or not the run then
    succeeds, so a failed run leaves an approved action to retry rather than
    vanishing without trace.
    """
    from carmel.paths import resolve_workspaces_root
    from carmel.schemas.action_state import ActionOutcome
    from carmel.services.campaigns import find_campaign_workspace, load_campaign
    from carmel.services.planner import append_corpus_pass_action

    # The two modes are mutually exclusive by construction: --dispatch-queued runs an
    # action whose budget is already fixed in the plan. Accepting a budget alongside it
    # would either be ignored (a lie) or silently override what the approver agreed to.
    if dispatch_queued:
        if budget_tokens is not None:
            print(
                "--budget-tokens cannot be combined with --dispatch-queued: the queued "
                "pass already carries the budget it was approved under.",
                file=sys.stderr,
            )
            return 1
        if reread_all:
            print(
                "--reread-all cannot be combined with --dispatch-queued: re-read scope is "
                "fixed when the pass is appended, not when it is dispatched.",
                file=sys.stderr,
            )
            return 1
        if allow_unauthenticated_legacy_roots:
            print(
                "--allow-unauthenticated-legacy-roots cannot be combined with --dispatch-queued: "
                "the queued pass already carries the parameters it was approved under, and "
                "bolting a fresh permission on at dispatch time would defeat the approval.",
                file=sys.stderr,
            )
            return 1
    elif budget_tokens is None:
        print("--budget-tokens is required (or use --dispatch-queued to run an already-queued pass).", file=sys.stderr)
        return 1

    try:
        agent_config = _load_agent_config(config)
    except (OSError, ValueError) as e:
        print(f"Failed to load config: {e}", file=sys.stderr)
        return 1
    if agent_config is None and not dry_run:
        print(
            "No agent config available: pass --config FILE whose 'agents' section "
            "sets provider, consent and API key env vars.",
            file=sys.stderr,
        )
        return 1

    workspaces_root = resolve_workspaces_root(workspaces)
    ws = find_campaign_workspace(workspaces_root, campaign_id)
    if ws is None:
        print(f"Campaign {campaign_id!r} not found under {workspaces_root}", file=sys.stderr)
        return 1
    campaign = load_campaign(ws)

    # The model name only shapes the REPORTED dollar estimate; the token cap binds
    # regardless. A dry run without a config therefore still appends a valid action,
    # it just cannot say what the tokens would have cost.
    # `_load_agent_config` returns `object | None` so this module stays importable
    # without the agents extra, which is why every use site here carries a targeted
    # ignore rather than a real annotation.
    model_name = (
        agent_config.resolved_model_name()  # type: ignore[attr-defined]
        if agent_config is not None
        else None
    )

    if dispatch_queued:
        # Locate the queued pass rather than appending one. Deliberately reported as a
        # typed refusal when there is nothing to dispatch: "I ran nothing" must never be
        # indistinguishable from "I ran your pass".
        from carmel.schemas.action_state import ActionExecutionStatus
        from carmel.schemas.approval import ActionKind as _ActionKind
        from carmel.services.plan_progress import load_progress as _load_progress

        try:
            queued = [
                a
                for a in _load_progress(ws).actions
                if a.kind == _ActionKind.LITERATURE_CORPUS_PASS
                and a.execution_status == ActionExecutionStatus.PENDING
                and a.outcome != ActionOutcome.REJECTED
            ]
        except OSError as e:
            print(f"Could not read plan progress: {e}", file=sys.stderr)
            return 1
        if not queued:
            print(
                "No corpus pass is queued for this campaign. Run without --dispatch-queued to append one.",
                file=sys.stderr,
            )
            return 1
        queued_action_id = queued[0].action_id
        print(f"Dispatching the already-queued corpus-pass action {queued_action_id}.")
        return _dispatch_corpus_action(ws, campaign, queued_action_id, agent_config=agent_config, dry_run=dry_run)

    # Narrowed by the mutual-exclusion block above: the append path returns early
    # when no budget was named, so reaching here means one was.
    assert budget_tokens is not None
    try:
        action = append_corpus_pass_action(
            ws,
            budget_tokens=budget_tokens,
            model_name=model_name,
            reread_all=reread_all,
            allow_unauthenticated_legacy_roots=allow_unauthenticated_legacy_roots,
        )
    except FileNotFoundError:
        # Distinguished from the generic OSError below because the remedy is
        # specific and the raw message ("JSON file not found: .../plan.json") tells
        # an operator nothing about what to do.
        print(
            f"Campaign {campaign_id!r} has no plan yet, so there is nothing to append to. "
            "Run the literature step (or create a plan) first.",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError) as e:
        print(f"Could not append the corpus pass: {e}", file=sys.stderr)
        return 1
    # Report the cap in the unit that binds, and the dollars as what they are. Naming
    # the model the estimate came from matters: the same token cap costs different
    # amounts on different tiers, so a bare figure would invite reading it as fixed.
    if action.estimated_spend_usd > 0:
        # "at most", not "about": the figure comes from
        # `estimate_worst_case_model_cost_usd`, which prices an unknown model at a
        # deliberately punitive fallback rate so no call can escape the ledger at zero
        # cost. That is right for a reservation and wrong to print as a point
        # estimate -- an operator told "~$2.50" who then spends $0.40 has been
        # misled in exactly the way this whole change exists to stop.
        print(
            f"Appended corpus-pass action {action.action_id} with a budget of "
            f"{budget_tokens:,} tokens (at most ~${action.estimated_spend_usd:.2f} "
            f"on {model_name}; the token cap is what binds)"
        )
    else:
        print(
            f"Appended corpus-pass action {action.action_id} with a budget of "
            f"{budget_tokens:,} tokens (no model configured, so the dollar cost "
            f"cannot be estimated)"
        )
    if dry_run:
        print("--dry-run: the action was appended but not run.")
        return 0

    return _dispatch_corpus_action(ws, campaign, action.action_id, agent_config=agent_config, dry_run=dry_run)


def _dispatch_corpus_action(
    ws: Path,
    campaign: Campaign,
    action_id: str,
    *,
    agent_config: object,
    dry_run: bool,
) -> int:
    """Dispatch one already-appended corpus-pass action through the dispatcher.

    Shared by both entry paths -- appending-then-running, and ``--dispatch-queued``
    against a pass a human has since approved. It is one function precisely because
    the guards below are the valuable part: duplicating them for the second path is
    how one of the two ends up missing the cursor check.
    """
    from carmel.agents.bridge import AgentBridgeError
    from carmel.schemas.action_state import ActionOutcome
    from carmel.services.dispatcher import default_handlers, execute_next_action

    if dry_run:
        print("--dry-run: the queued action was not run.")
        return 0

    try:
        # Go through the DISPATCHER, not execute_action (spar round 7 P1).
        # execute_action is only the by-kind router: it runs the handler and nothing
        # else. Calling it directly skipped reconcile, the approval gate, the exclusive
        # dispatch lease, the campaign pre/post state transitions, attempt recording,
        # and -- the damaging one -- the cursor advance. A corpus pass would complete
        # and write its report while plan_progress.json still showed the action
        # pending, so the next dispatcher run would execute it a second time. Since
        # findings are deliberately never deduped across passes, that silently doubles
        # them in the accumulated report.
        #
        # The agent config MUST be threaded through to the handler registry. Without
        # it the literature handler is built with nothing to run on and returns a
        # typed "no agent config available" failure -- which looks like a failed run
        # rather than a command that never started one.
        # Check WHAT the cursor points at before dispatching, not after. The
        # dispatcher runs the plan's next runnable action, which is not necessarily the
        # corpus pass just appended -- an earlier LITERATURE_SEARCH still pending sits
        # ahead of it. Reporting the mismatch afterwards is too late: by then the
        # search has already run, reached the network and spent the operator's money on
        # a pass they did not ask for, under a budget they named for something else.
        from carmel.services.plan_progress import load_progress

        progress = load_progress(ws)
        next_state = progress.actions[progress.cursor] if progress.cursor < len(progress.actions) else None
        if next_state is not None and next_state.action_id != action_id:
            print(
                f"Refusing to dispatch: the plan's next action is {next_state.kind.value} "
                f"({next_state.action_id}), not the corpus pass "
                f"({action_id}). The corpus pass stays queued. Run or retire the "
                f"earlier action first, then dispatch again.",
                file=sys.stderr,
            )
            return 1

        ticket = execute_next_action(
            ws,
            campaign,
            handlers=default_handlers(agent_config=agent_config),  # type: ignore[arg-type]
        )
    except AgentBridgeError as e:
        print(f"Corpus pass unavailable (consent/API key): {e}", file=sys.stderr)
        return 1

    if ticket is None:
        print(
            f"The corpus-pass action {action_id} is in the plan but the dispatcher did "
            "not start it. The plan has no runnable action right now -- most often the "
            "action is still awaiting approval. It stays in the plan; approve it, then "
            "run `carmel corpus-pass --campaign <id> --dispatch-queued` to run it "
            "(re-running the plain command would refuse rather than queue a second "
            "identical pass).",
            file=sys.stderr,
        )
        return 1
    if ticket.action_id != action_id:
        # The dispatcher runs the plan's next executable action, which is the correct
        # behaviour and is deliberately not overridden here: jumping the queue would
        # run the corpus pass out of the order the plan records. Say plainly that
        # something else ran, rather than reporting another action's outcome as though
        # it were the corpus pass.
        print(
            f"The corpus-pass action {action_id} is queued, but the next action due in "
            f"the plan is {ticket.action_id} ({ticket.kind.value}), which ran instead. "
            "The corpus pass remains queued; dispatch again to reach it.",
            file=sys.stderr,
        )
        return 1

    result = ticket.wait()
    if result is None:
        print(
            f"Corpus pass {action_id} failed before producing a result"
            + (f": {ticket.error}" if ticket.error is not None else "")
            + ". The failure is recorded in the workspace.",
            file=sys.stderr,
        )
        return 1
    print(f"Corpus pass {action_id} finished with outcome {result.outcome.value}")
    return 0 if result.outcome == ActionOutcome.SUCCEEDED else 1


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

    Exit code: 0 if at least one file was admitted (or was already held, which means the
    operator's intent is already satisfied) and none were rejected; 1 otherwise --
    including an empty directory, and including one holding nothing but dotfiles,
    partial downloads or subdirectories. Those are reported rather than treated as a
    silent success: a run that admitted nothing must not look like one that worked.
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
    already = 0

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
            # download folder, so it is a skip rather than a rejection. Counted
            # SEPARATELY from `skipped`, which covers junk (dotfiles, partial downloads,
            # subdirectories): only an already-held PAPER means the operator's intent was
            # satisfied, so only it may stand in for an acceptance in the exit code.
            already += 1
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

    print(f"{accepted} accepted, {rejected} rejected, {already} already acquired, {skipped} skipped")
    # A folder in which every paper was already acquired is a success, not a failure --
    # requiring `accepted` outright would fail the second run of an ingest that fully
    # succeeded the first time. `skipped` deliberately does NOT count: a directory holding
    # nothing but dotfiles, partial downloads or subdirectories did no useful work, and
    # exiting 0 on it would report success for a run that admitted nothing at all.
    if rejected:
        return 1
    return 0 if (accepted or already) else 1


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
        ManifestUnreadable,
        admit_file,
        collect_inbox,
        drop_path_for,
        inbox_dir,
        pending_requests,
        reason_phrase,
    )
    from carmel.services.campaigns import find_campaign_workspace

    # Load the config BEFORE resolving the root: it is one of the inputs to that
    # resolution. This command used to resolve the root first and read the config only
    # for `max_artifact_bytes`, so `carmel requests --config <file>` silently scanned
    # the default workspaces directory rather than the one the config named -- while
    # `new-campaign --workspaces` advertises "default: the config's workspace_root".
    # The two commands disagreed, and the failure is quiet: you get "no campaign
    # found" while looking in a directory you never asked about.
    max_bytes = AgentBudgetConfig().max_artifact_bytes
    config_root: Path | None = None
    if config_file is not None:
        try:
            config = load_config(config_file)
        except (OSError, ValueError) as exc:
            print(f"Could not load {config_file}: {exc}")
            return 1
        if config.agents is not None:
            max_bytes = config.agents.budget.max_artifact_bytes
        # `--workspaces` means the PARENT directory in every command, whereas
        # `config.workspace_root` is the campaign's OWN directory -- so its parent is
        # what `find_campaign_workspace` needs to scan.
        config_root = config.workspace_root.expanduser().parent

    if workspaces is not None:
        root = workspaces.expanduser()
    elif config_root is not None:
        root = config_root
    else:
        root = default_workspaces_root()

    ws = find_campaign_workspace(root, campaign_id)
    if ws is None:
        print(f"No campaign {campaign_id!r} under {root}")
        return 1

    if collect:
        try:
            changed = collect_inbox(ws, max_bytes=max_bytes)
        except ManifestUnreadable as exc:
            print(f"Could not read the acquisition manifest: {exc}")
            print("Fix or restore the file above before retrying -- it was left untouched.")
            return 1
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

    try:
        pending = pending_requests(ws)
    except ManifestUnreadable as exc:
        print(f"Could not read the acquisition manifest: {exc}")
        print("Fix or restore the file above before retrying -- it was left untouched.")
        return 1
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
        print(f"    why   : {reason_phrase(request.reason)}{detail}")
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

    if args.command == "corpus-pass":
        return _cmd_corpus_pass(
            args.campaign,
            args.budget_tokens,
            args.workspaces,
            args.config,
            dry_run=args.dry_run,
            reread_all=args.reread_all,
            dispatch_queued=args.dispatch_queued,
            allow_unauthenticated_legacy_roots=args.allow_unauthenticated_legacy_roots,
        )

    if args.command == "new-campaign":
        return _cmd_new_campaign(args.config, args.workspaces)

    if args.command == "requests":
        return _cmd_requests(args.campaign, args.workspaces, args.add, args.slug, args.config, args.collect)

    if args.command == "reextract":
        return _cmd_reextract(
            args.campaign,
            args.workspaces,
            args.config,
            sha=args.sha,
            all_artifacts=args.all,
            apply=args.apply,
        )

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
