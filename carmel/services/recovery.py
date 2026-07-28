# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a campaign left in ``RUNNING_T3`` is still running.

A T3 run is supervised by a background thread in the Carmel process that
started it. If that process dies — SIGKILL, host reboot, a dev-server
reload — the thread dies with it and nobody ever writes the run's ending.
The campaign is left in ``RUNNING_T3`` for good, and the dashboard
promises a run that no longer exists.

Getting the campaign unstuck needs an answer to one question first: is
anything still running? Guessing either way is harmful. Guessing "yes"
leaves the campaign wedged forever, which is the bug. Guessing "no" is
worse — it marks the run failed while T3 and RMG carry on writing into
the workspace, which is the same defect the process-tree kill exists to
prevent, moved one layer up.

So Carmel records two things when a run starts and reads them back later:

* An exclusive ``flock`` on ``active_run.lock``, held for the run's
  duration. The kernel releases it when the holding process dies, however
  it dies, so a lock that can be taken means no supervisor survives. This
  is why liveness is not a heartbeat or a pid check: it cannot go stale
  and it cannot be fooled by pid reuse.
* ``active_run.json``, holding the tool's process group id together with
  the group leader's kernel-observed argv and its start time, so an
  orphaned tool tree can be positively identified — reuse-proof, by start
  time — and stopped.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO

from carmel.logger import get_logger
from carmel.schemas.run import ActiveRun
from carmel.services.artifacts import read_json, write_json
from carmel.services.processes import (
    ProcessGroupStatus,
    inspect_process_group,
    process_group_command,
    process_starttime,
)

ACTIVE_RUN_FILE_NAME = "active_run.json"
ACTIVE_RUN_LOCK_FILE_NAME = "active_run.lock"
"""Both live at the workspace root, alongside ``campaign_state.json``,
rather than under ``runs/``. They describe the campaign's lifecycle, not a
finished run — and ``runs/`` is globbed for run records, where a file that
is not one would be mistaken for the latest run."""

_log = get_logger("services.recovery")

_LOCK_HELD_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK})
"""The errnos ``flock`` uses to say the lock is held by somebody else.

``EAGAIN`` and ``EWOULDBLOCK`` are the same number on Linux; both are
listed because that is not guaranteed everywhere."""


class RunAlreadySupervisedError(RuntimeError):
    """Raised when a run is started while another is provably still in flight."""


class ProcessGroupNotRecordedError(RuntimeError):
    """Raised when a launched tool's process group could not be recorded.

    The tool is running at this point. Nothing that reads the workspace
    afterwards would be able to tell, which is why this is an error and
    not a warning.
    """


class LockStateUnknownError(RuntimeError):
    """Raised when the run lock can be neither taken nor shown to be held.

    Distinct from contention, which is an answer. This is the absence of
    one: no ``flock`` support on the filesystem (an NFS mount with no lock
    daemon), no file descriptors left, an unwritable workspace. Carmel
    cannot establish liveness by its usual means and must say so rather
    than pick whichever guess is convenient.
    """


class RunLiveness(StrEnum):
    """What Carmel can establish about a campaign sitting in ``RUNNING_T3``."""

    SUPERVISED = "supervised"
    """A living Carmel process holds the run's lock. The run is genuinely
    in progress and will record its own ending."""

    ORPHANED = "orphaned"
    """No supervisor survives, but the tool's process group is still alive
    and confirmed to be this run's. T3 is running with nothing watching
    it: it must be stopped before the run can be called over."""

    UNSUPERVISED = "unsupervised"
    """No supervisor survives and no process of this run's remains. The
    campaign state is merely stale and can be corrected."""

    NO_RECORD = "no_record"
    """No supervisor survives and no run was ever recorded. Nothing is
    known to be running, and nothing is known to have been started."""

    UNKNOWN = "unknown"
    """Carmel cannot establish whether anything is still running.

    Something is alive in the run's process group but cannot be shown to
    be this run's, or the lock itself could not be interrogated. Recovery
    is refused rather than guessed: the operator is told the process group
    id and can stop it by hand, after which the group is empty and the
    ordinary path applies."""


@dataclass(frozen=True)
class RunLivenessReport:
    """The outcome of :func:`probe_run_liveness`."""

    liveness: RunLiveness
    active_run: ActiveRun | None
    detail: str

    @property
    def is_finished(self) -> bool:
        """Whether nothing of this run is still executing."""
        return self.liveness in {RunLiveness.UNSUPERVISED, RunLiveness.NO_RECORD}


def active_run_path(workspace_root: Path) -> Path:
    """Return the path of the in-flight run record."""
    return workspace_root / ACTIVE_RUN_FILE_NAME


def active_run_lock_path(workspace_root: Path) -> Path:
    """Return the path of the supervisor lock file."""
    return workspace_root / ACTIVE_RUN_LOCK_FILE_NAME


def load_active_run(workspace_root: Path) -> ActiveRun | None:
    """Load the in-flight run record, if one is present.

    Args:
        workspace_root: The campaign workspace root.

    Returns:
        The recorded :class:`ActiveRun`, or None when no run is recorded
        or the record cannot be parsed. An unreadable record is treated as
        absent rather than fatal: it must not be what stops an operator
        from recovering a campaign.
    """
    path = active_run_path(workspace_root)
    if not path.exists():
        return None
    try:
        return ActiveRun.model_validate(read_json(path))
    except (OSError, ValueError) as e:
        _log.warning("Ignoring unreadable active-run record at %s: %s", path, e)
        return None


def clear_active_run(workspace_root: Path) -> None:
    """Remove the in-flight run record, if present."""
    active_run_path(workspace_root).unlink(missing_ok=True)


def _open_lock_file(workspace_root: Path) -> IO[str]:
    """Open the supervisor lock file for locking, creating it if needed.

    Opened in append mode rather than write mode so that merely probing
    the lock never truncates a file another process is holding.

    Args:
        workspace_root: The campaign workspace root.

    Returns:
        The opened lock file.
    """
    path = active_run_lock_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def supervisor_is_alive(workspace_root: Path) -> bool:
    """Report whether a living process is supervising this campaign's run.

    Implemented by trying to take the run lock *shared*: ``flock`` is
    released by the kernel when its holder dies, so acquiring it proves no
    supervisor remains. The probe releases the lock again immediately — it
    establishes a fact, it does not claim supervision.

    The shared mode matters. The supervisor holds the lock ``LOCK_EX``, so
    a shared probe is still refused while a supervisor lives — the correct
    "held" answer. But two concurrent probes take compatible shared locks
    instead of blocking each other, which an exclusive probe would do:
    Werkzeug serves requests threaded, and ``flock`` conflicts across
    independent descriptors even within one process, so an exclusive probe
    landing during another's microsecond hold would falsely report a live
    supervisor and re-wedge a dead run in the dashboard.

    Only the errnos that mean "somebody else holds this" count as a live
    supervisor. Every other ``OSError`` means the question could not be
    asked at all — an unwritable workspace, a filesystem without working
    ``flock`` — and answering "supervised" there would silently reinstate
    the permanent wedge this module exists to remove.

    Args:
        workspace_root: The campaign workspace root.

    Returns:
        True if some process still holds the run lock exclusively.

    Raises:
        LockStateUnknownError: If the lock could be neither acquired nor
            shown to be held.
    """
    try:
        lock_file = _open_lock_file(workspace_root)
    except OSError as e:
        raise LockStateUnknownError(f"Could not open the run lock for {workspace_root}: {e}") from e
    with lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno in _LOCK_HELD_ERRNOS:
                return True
            raise LockStateUnknownError(
                f"Could not determine whether {workspace_root} has a live supervisor: {e}"
            ) from e
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        return False


def record_process_group(workspace_root: Path, process_group_id: int, command: list[str]) -> None:
    """Attach a launched tool's process group to the in-flight run record.

    This is not best-effort, though it was once. A tool whose process
    group went unrecorded is a tool no later Carmel can identify or stop,
    and the run it belongs to reads afterwards exactly like a run that
    never launched — so recovery would offer to abandon a campaign whose
    T3 is still writing into it. An untrackable run is worse than no run:
    the caller is expected to stop the tree it just launched and fail.

    What is recorded is the leader's identity *as ``/proc`` reports it*,
    not the argv Carmel passed to ``Popen``: read back the kernel-observed
    command line and the start time now, so a later recovery compares like
    with like. This is what makes a ``conda`` launch identifiable at all —
    the kernel prepends the interpreter to a ``#!`` wrapper's argv, so the
    launched argv never matches ``/proc`` — and the start time is the
    reuse-proof pin. If ``/proc`` cannot be read here, the passed argv is
    kept as a last-resort label and the start time is left unset; recovery
    then falls back to the (weaker) command-line comparison.

    Args:
        workspace_root: The campaign workspace root.
        process_group_id: The launched tool's process group id.
        command: The argv launched into that group, used only as a fallback
            label if the kernel's command line cannot be read.

    Raises:
        ProcessGroupNotRecordedError: If the record is missing or could
            not be written.
    """
    active = load_active_run(workspace_root)
    if active is None:
        raise ProcessGroupNotRecordedError(f"No in-flight run record to attach process group {process_group_id} to")
    observed_command = process_group_command(process_group_id) or command
    leader_starttime = process_starttime(process_group_id)
    try:
        write_json(
            active_run_path(workspace_root),
            active.model_copy(
                update={
                    "process_group_id": process_group_id,
                    "command": observed_command,
                    "leader_starttime": leader_starttime,
                }
            ),
        )
    except OSError as e:
        raise ProcessGroupNotRecordedError(f"Could not record process group {process_group_id}: {e}") from e


class RunSupervision:
    """A held run lock and the in-flight record that goes with it.

    Taken in the thread that decides a run may start and released by
    whichever thread finishes it. ``flock`` belongs to the open file
    description, not to a thread, so the handover costs nothing — and it
    closes the window in which a campaign is already ``RUNNING_T3`` while
    no lock has been taken yet. A probe landing in that window would find
    no supervisor and no record, conclude nothing had ever started, and
    offer to abandon a run that was about to launch.
    """

    def __init__(self, workspace_root: Path, lock_file: IO[str]) -> None:
        """Store the supervised workspace and the locked file holding it.

        Args:
            workspace_root: The campaign workspace root.
            lock_file: The opened, already-locked run lock file.
        """
        self._workspace_root = workspace_root
        self._lock_file = lock_file

    def record_process_group(self, process_group_id: int, command: list[str]) -> None:
        """Attach a launched tool's process group to this run's record.

        Args:
            process_group_id: The launched tool's process group id.
            command: The argv launched into that group.

        Raises:
            ProcessGroupNotRecordedError: If the record could not be written.
        """
        record_process_group(self._workspace_root, process_group_id, command)

    def close(self) -> None:
        """Clear the in-flight record and release the run lock.

        Clearing before releasing is what makes a surviving record mean
        something: a record still present once the lock is free is exactly
        the evidence of a supervisor that died without finishing.
        """
        try:
            clear_active_run(self._workspace_root)
        finally:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()


def start_supervision(workspace_root: Path, action_id: str) -> RunSupervision:
    """Take the run lock and write the in-flight record for one T3 run.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The planned action being run.

    Returns:
        The held :class:`RunSupervision`. The caller owns it and must
        :meth:`~RunSupervision.close` it, from any thread.

    Raises:
        RunAlreadySupervisedError: If another live process already holds
            this campaign's run lock.
        LockStateUnknownError: If the lock could not be interrogated.
    """
    try:
        lock_file = _open_lock_file(workspace_root)
    except OSError as e:
        raise LockStateUnknownError(f"Could not open the run lock for {workspace_root}: {e}") from e
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        lock_file.close()
        if e.errno in _LOCK_HELD_ERRNOS:
            raise RunAlreadySupervisedError(
                f"Another process is already running a T3 action for {workspace_root}"
            ) from e
        raise LockStateUnknownError(f"Could not take the run lock for {workspace_root}: {e}") from e
    try:
        write_json(
            active_run_path(workspace_root),
            ActiveRun(action_id=action_id, started_at=datetime.now(UTC), supervisor_pid=os.getpid()),
        )
    except OSError:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
        raise
    return RunSupervision(workspace_root, lock_file)


@contextlib.contextmanager
def supervise_run(workspace_root: Path, action_id: str) -> Iterator[RunSupervision]:
    """Hold the run lock and the in-flight record for the duration of a block.

    The scoped form of :func:`start_supervision`, for callers that both
    start and finish a run in one place.

    Args:
        workspace_root: The campaign workspace root.
        action_id: The planned action being run.

    Yields:
        The held :class:`RunSupervision`.

    Raises:
        RunAlreadySupervisedError: If another live process already holds
            this campaign's run lock.
        LockStateUnknownError: If the lock could not be interrogated.
    """
    supervision = start_supervision(workspace_root, action_id)
    try:
        yield supervision
    finally:
        supervision.close()


def probe_run_liveness(workspace_root: Path) -> RunLivenessReport:
    """Establish whether this campaign's recorded run is still executing.

    Args:
        workspace_root: The campaign workspace root.

    Returns:
        A :class:`RunLivenessReport` describing what was found, including
        prose suitable for showing an operator deciding what to do.
    """
    active = load_active_run(workspace_root)
    try:
        supervised = supervisor_is_alive(workspace_root)
    except LockStateUnknownError as e:
        return RunLivenessReport(
            liveness=RunLiveness.UNKNOWN,
            active_run=active,
            detail=(
                f"Carmel could not determine whether a process is supervising this run: {e} "
                f"Until that is resolved, no recovery can be offered without guessing."
            ),
        )
    if supervised:
        return RunLivenessReport(
            liveness=RunLiveness.SUPERVISED,
            active_run=active,
            detail="A Carmel process is supervising this run; it will record its own outcome.",
        )
    if active is None:
        return RunLivenessReport(
            liveness=RunLiveness.NO_RECORD,
            active_run=None,
            detail=(
                "No process is supervising this campaign and no run is recorded as in flight. "
                "Nothing is known to be running."
            ),
        )
    if active.process_group_id is None:
        return RunLivenessReport(
            liveness=RunLiveness.UNSUPERVISED,
            active_run=active,
            detail="The run was recorded but its tool was never launched, and no process supervises it.",
        )

    status = inspect_process_group(active.process_group_id, active.command, active.leader_starttime)
    if status == ProcessGroupStatus.RUNNING:
        return RunLivenessReport(
            liveness=RunLiveness.ORPHANED,
            active_run=active,
            detail=(
                f"No Carmel process is supervising this run, but its tool is still running as "
                f"process group {active.process_group_id}. It must be stopped before the run can "
                f"be recorded as over."
            ),
        )
    if status == ProcessGroupStatus.UNKNOWN_LIVE:
        return RunLivenessReport(
            liveness=RunLiveness.UNKNOWN,
            active_run=active,
            detail=(
                f"No Carmel process is supervising this run, and something is still alive in "
                f"process group {active.process_group_id} that Carmel cannot confirm either way. "
                f"Most likely this run's tool outlived the process that launched it. Stop process "
                f"group {active.process_group_id} by hand, then this campaign can be recovered."
            ),
        )
    if status == ProcessGroupStatus.UNRECOGNIZED:
        return RunLivenessReport(
            liveness=RunLiveness.UNSUPERVISED,
            active_run=active,
            detail=(
                f"No Carmel process is supervising this run. Process group "
                f"{active.process_group_id} is now led by a different process than the one this "
                f"run launched, which the id could only have been reused for once this run's own "
                f"processes had ended."
            ),
        )
    return RunLivenessReport(
        liveness=RunLiveness.UNSUPERVISED,
        active_run=active,
        detail=(
            f"No Carmel process is supervising this run and its process group {active.process_group_id} has ended."
        ),
    )
