# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Inspect and terminate a process group Carmel no longer owns.

Distinct from the process-tree kill in :mod:`carmel.adapters.t3`, which
supervises a live ``Popen`` it spawned and can reap. Everything here works
from a bare process group id read back from disk, for the case that
motivates it: the Carmel process that launched a T3 run was killed, and a
later Carmel process — a fresh server, a different request — must decide
whether that run is still going before declaring it over.

A recorded pgid is not by itself proof of identity. Once every process in
the group has exited, the leader's pid is free for the kernel to reuse,
and an unrelated process that later becomes a group leader can inherit
exactly that number. Signalling on the pgid alone would then kill a
stranger. Identity is therefore pinned to the leader's *start time* (field
22 of ``/proc/<pid>/stat``, ticks since boot), which the kernel does not
carry over when it reuses a pid: a group whose leader started at a
different instant than the one Carmel recorded is not Carmel's, whatever
its command line happens to read. The recorded command line is kept too,
as a human-readable label and as a fallback for records that predate the
start-time pin — but it is not the identity, and on its own it is both too
weak (a reused pid can rerun the same argv) and, for a tool launched
through a ``#!`` wrapper such as ``conda``, wrong: the kernel rewrites the
argv to prepend the interpreter, so the launched argv never matches what
``/proc`` later reports.

Identity is re-established from ``/proc`` immediately before the *first*
signal of a kill, not before every signal. Once a SIGTERM has gone out the
group is watched for whether it is still *running*, not for whether it is
still recognizable — so a SIGTERM the leader honours while a descendant
ignores it still escalates to SIGKILL, instead of the vanished leader
being read as "no longer Carmel's" and the survivors left alone. See
:func:`kill_process_group`, whose own docstring names the residual race.

The three ways a live group can fail the identity check are not the same
finding, and collapsing them is how a recovery ends up lying. A group led
by a process that started at a *different time* is evidence the run ended —
the id could only have been reused once the group emptied. A group whose
leader is simply gone is the opposite: its descendants are most likely T3
and RMG still running, and they are exactly what must not be declared
finished. That second case is :attr:`ProcessGroupStatus.UNKNOWN_LIVE`,
along with a missing ``/proc`` and an unrecorded identity: neither
signalled nor called over, because Carmel reports what it found and
refuses to guess.

The identity check needs ``/proc``, so it is Linux-only.
"""

from __future__ import annotations

import os
import signal
import time
from enum import StrEnum
from pathlib import Path

from carmel.logger import get_logger

_log = get_logger("services.processes")

PROC_ROOT = Path("/proc")

KILL_GRACE_PERIOD_S: float = 10.0
"""Seconds between SIGTERM and SIGKILL when stopping an orphaned group."""

_REAP_TIMEOUT_S: float = 10.0
"""Bound on waiting for the kernel to clear the group after SIGKILL."""

_POLL_INTERVAL_S: float = 0.1


class ProcessGroupStatus(StrEnum):
    """What Carmel can currently say about a recorded process group."""

    GONE = "gone"
    """No process is in the group. Nothing to stop."""

    RUNNING = "running"
    """The group is alive and its leader is the command Carmel launched."""

    UNRECOGNIZED = "unrecognized"
    """The group is alive, and its leader is provably a different process.

    A process group id is only free for reuse once the group has emptied,
    so a live group whose leader started at a different instant than the
    one Carmel recorded — or, absent a recorded start time, runs a command
    Carmel did not launch — is positive evidence that Carmel's own
    processes have ended and the number has since been recycled. Nothing is
    signalled, and the run is over.
    """

    UNKNOWN_LIVE = "unknown_live"
    """The group is alive but its identity cannot be established.

    Either the leader has exited while descendants survive — its ``/proc``
    entry is gone, so nothing remains to identify it by — or ``/proc`` is
    unavailable, or no identity was ever recorded. Carmel neither signals
    such a group nor treats the run behind it as finished. Both would be
    guesses, and each guesses wrong in a way that costs something
    irreversible: one kills a stranger, the other declares a run over
    while it is still writing.
    """


def process_group_exists(pgid: int) -> bool:
    """Report whether any process is still in the process group *pgid*.

    ``EPERM`` counts as existing: a process Carmel may not signal is still
    a process that is running. Treating it as gone would let a caller
    conclude a group had been cleaned up when it had merely become
    unsignallable.

    Args:
        pgid: The process group id.

    Returns:
        True if at least one process remains in the group.
    """
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stat_fields(pid: int) -> list[str] | None:
    """Return the fields of ``/proc/<pid>/stat`` after the ``comm`` column.

    ``comm`` (field 2) can itself contain spaces and parentheses, so the
    split is anchored on the final ``)``. The returned list therefore
    begins at field 3: ``fields[0]`` is state, ``fields[2]`` is pgrp, and
    ``fields[19]`` is start time (field 22).

    Args:
        pid: The process id.

    Returns:
        The post-``comm`` fields, or None if ``/proc/<pid>/stat`` could not
        be read — the process exited, or the platform has no ``/proc``.
    """
    try:
        stat = (PROC_ROOT / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    return stat.rsplit(")", 1)[-1].split()


def _process_group_members(pgid: int) -> list[int] | None:
    """Return the pids in *pgid* that are actually executing.

    Zombies are excluded deliberately. An exited-but-unreaped process is
    still a member of its group and still answers ``killpg``, but it holds
    no file descriptors and cannot write anything — treating it as a live
    member would make a fully stopped tree look like one that ignored
    SIGKILL.

    Args:
        pgid: The process group id.

    Returns:
        The executing pids in the group, or None if ``/proc`` could not be
        enumerated at all.
    """
    try:
        entries = [entry for entry in PROC_ROOT.iterdir() if entry.name.isdigit()]
    except OSError:
        return None
    members = []
    for entry in entries:
        fields = _stat_fields(int(entry.name))
        if fields is None:
            # Exited between listing the directory and reading it.
            continue
        if len(fields) < 3 or fields[0] == "Z":
            continue
        try:
            if int(fields[2]) == pgid:
                members.append(int(entry.name))
        except ValueError:  # pragma: no cover -- malformed /proc entry
            continue
    return members


def process_group_is_running(pgid: int) -> bool:
    """Report whether any process in *pgid* is still executing.

    Stricter than :func:`process_group_exists`, which counts zombies
    because they can still be signalled. This answers the question that
    matters after a kill: is anything left that could still be writing?

    Args:
        pgid: The process group id.

    Returns:
        True if at least one non-zombie process remains in the group. If
        ``/proc`` cannot be read the group is reported as running, since
        nothing has been shown to have stopped.
    """
    if not process_group_exists(pgid):
        return False
    members = _process_group_members(pgid)
    if members is None:
        return True
    return bool(members)


def process_group_command(pgid: int) -> list[str] | None:
    """Return the argv of the group leader of *pgid*, if it can be read.

    The leader is the process whose pid equals the pgid. Reads
    ``/proc/<pgid>/cmdline``, which holds the argv NUL-separated with a
    trailing NUL.

    Args:
        pgid: The process group id.

    Returns:
        The leader's argv, or None if there is no such process, its
        command cannot be read, or the platform has no ``/proc``.
    """
    try:
        raw = (PROC_ROOT / str(pgid) / "cmdline").read_bytes()
    except OSError:
        return None
    # Strip only the trailing NUL terminator, not the internal separators:
    # an argv can legitimately contain an empty element, and dropping it
    # would make an otherwise identical command line miscompare. A wholly
    # empty cmdline (a kernel thread, a zombie) reads back as no command.
    stripped = raw.rstrip(b"\0")
    if not stripped:
        return None
    return [part.decode("utf-8", errors="replace") for part in stripped.split(b"\0")]


def process_starttime(pgid: int) -> int | None:
    """Return the group leader's start time — field 22 of ``/proc/<pgid>/stat``.

    Start time is measured in clock ticks since boot and is not carried
    over when the kernel recycles a pid, which makes it the only
    reuse-proof identity for a group leader read back from disk.

    Args:
        pgid: The process group id, whose leader is the process with that pid.

    Returns:
        The leader's start time in clock ticks, or None if it could not be
        read — no such process, or the platform has no ``/proc``.
    """
    fields = _stat_fields(pgid)
    if fields is None or len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:  # pragma: no cover -- malformed /proc entry
        return None


def inspect_process_group(
    pgid: int,
    expected_command: list[str] | None,
    expected_starttime: int | None,
) -> ProcessGroupStatus:
    """Classify a recorded process group against the process that created it.

    Identity is the group leader's start time, which a recycled pid does
    not carry over (see the module docstring). The recorded command line is
    a fallback only, for records written before start times were pinned;
    on its own it cannot tell a launched tool apart from a reused pid
    rerunning the same argv, and it never matches for a ``#!`` wrapper such
    as ``conda`` because the kernel rewrites the argv it reports.

    A live group whose leader is gone is deliberately *not* reported as
    finished. The leader is the only member Carmel can identify — it is the
    one whose pid it recorded — so once its ``/proc`` entry goes, the
    surviving descendants cannot be told apart from a stranger's, even
    though they are far more likely to be T3 and RMG still writing.

    Args:
        pgid: The process group id Carmel recorded when it launched the run.
        expected_command: The kernel-observed argv of the leader, recorded
            alongside the pgid. Used only when *expected_starttime* is None.
        expected_starttime: The leader's start time recorded at launch, in
            clock ticks. The reuse-proof identity; None only for records
            that predate it, when the command line is used instead.

    Returns:
        The group's :class:`ProcessGroupStatus`.
    """
    if not process_group_is_running(pgid):
        # Nothing in the group is executing. An exited-but-unreaped leader
        # still answers ``killpg`` and still has a readable start time, so
        # the identity check below would match a zombie and call a stopped
        # run "running". Excluding zombies here keeps this consistent with
        # what the kill loop treats as running.
        return ProcessGroupStatus.GONE
    if expected_starttime is not None:
        leader_starttime = process_starttime(pgid)
        if leader_starttime is None:
            # The leader is gone though the group lives on, or there is no
            # ``/proc`` to read: unidentifiable, not finished.
            return ProcessGroupStatus.UNKNOWN_LIVE
        if leader_starttime == expected_starttime:
            return ProcessGroupStatus.RUNNING
        return ProcessGroupStatus.UNRECOGNIZED
    if expected_command is None:
        return ProcessGroupStatus.UNKNOWN_LIVE
    leader_command = process_group_command(pgid)
    if leader_command is None:
        return ProcessGroupStatus.UNKNOWN_LIVE
    if leader_command != expected_command:
        return ProcessGroupStatus.UNRECOGNIZED
    return ProcessGroupStatus.RUNNING


def _wait_until_group_stops(pgid: int, timeout_s: float) -> bool:
    """Poll *pgid* until nothing in it is executing, or *timeout_s* elapses.

    Whether anything is still running, not whether the group is still
    recognizable, is the right question once a signal has been sent.
    Re-identifying instead would report a group whose *leader* honoured
    SIGTERM as no longer Carmel's, and stop escalating against the
    descendants that ignored it.

    Polling is safe here in a way it is not for a supervised child: nothing
    in this module reaps anything, so a poll cannot itself release the pid
    that doubles as the group id.

    Args:
        pgid: The process group id.
        timeout_s: How long to keep polling.

    Returns:
        True if the group stopped within the timeout.
    """
    deadline = time.monotonic() + timeout_s
    while process_group_is_running(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_INTERVAL_S)
    return True


def kill_process_group(
    pgid: int,
    expected_command: list[str] | None,
    expected_starttime: int | None,
    grace_period_s: float | None = None,
) -> bool:
    """Stop the process group *pgid*, if it is still recognizably Carmel's.

    SIGTERM, then SIGKILL after a grace period. Identity is established
    once, before the first signal; afterwards the group is watched for
    whether it is still *running*, not for whether it is still
    recognizable. That distinction is the whole point: a SIGTERM the
    leader honours and a descendant ignores leaves the group running with
    its leader gone — no longer identifiable, but emphatically not
    stopped. Re-identifying there would report success over a live T3.

    The residual race is that the group stops and its id is recycled
    between the last poll and the SIGKILL. It is not closed, because
    nothing this side of the kernel can close it; it is made negligible.
    Escalation only happens if the group never stopped throughout the
    grace period, and reaching the same id again means cycling the whole
    pid space (``/proc/sys/kernel/pid_max``, millions of entries) inside
    one poll interval.

    Args:
        pgid: The process group id.
        expected_command: The leader's recorded argv, used to confirm the
            group is Carmel's only when *expected_starttime* is None.
        expected_starttime: The leader's recorded start time — the
            reuse-proof identity checked before anything is signalled.
        grace_period_s: Seconds to wait between SIGTERM and SIGKILL.
            Defaults to :data:`KILL_GRACE_PERIOD_S`.

    Returns:
        True if nothing from Carmel's group is still running. False if the
        group survived SIGKILL, or could not be identified as Carmel's in
        the first place — in both cases nothing may claim the run is over.
    """
    if grace_period_s is None:
        grace_period_s = KILL_GRACE_PERIOD_S

    status = inspect_process_group(pgid, expected_command, expected_starttime)
    if status == ProcessGroupStatus.GONE:
        return True
    if status != ProcessGroupStatus.RUNNING:
        _log.warning("Refusing to signal process group %s: it is %s, not recognizably Carmel's", pgid, status)
        return False

    _log.info("Stopping orphaned process group %s with SIGTERM", pgid)
    if not _signal_process_group(pgid, signal.SIGTERM):
        return False
    if _wait_until_group_stops(pgid, grace_period_s):
        return True

    _log.warning("Process group %s ignored SIGTERM; escalating to SIGKILL", pgid)
    if not _signal_process_group(pgid, signal.SIGKILL):
        return False
    if not _wait_until_group_stops(pgid, _REAP_TIMEOUT_S):
        _log.error("Process group %s survived SIGKILL; T3 may still be running", pgid)
        return False
    return True


def _signal_process_group(pgid: int, sig: signal.Signals) -> bool:
    """Send *sig* to *pgid*, returning whether it was delivered.

    Args:
        pgid: The process group id.
        sig: The signal to send.

    Returns:
        True if the signal was delivered or the group had already gone.
        False if the group exists but Carmel may not signal it.
    """
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return True
    except PermissionError:
        _log.error("Not permitted to send %s to process group %s", sig.name, pgid)
        return False
    return True
