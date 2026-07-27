"""Helpers shared by the tests that drive real process groups and run records.

Kept out of any single test module because both the service tests and the
UI tests need to reproduce the same situations: a live ``conda run``-shaped
process tree, and the in-flight record a killed supervisor leaves behind.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from carmel.schemas.run import ActiveRun
from carmel.services.artifacts import write_json
from carmel.services.recovery import active_run_path


class _Tree(NamedTuple):
    """A running test process group and the pids inside it."""

    pgid: int
    leader_pid: int
    grandchild_pid: int
    command: list[str]
    proc: subprocess.Popen[str]
    """The leader, so a test that kills it can also reap it.

    Nothing reaps it otherwise: it is this process's own child, and an
    unreaped zombie keeps both its ``/proc`` entry and its group
    membership, which is precisely what several of these tests are
    distinguishing between. In production the leader is orphaned onto init
    and reaped there."""


@contextlib.contextmanager
def _tool_tree(ignore_sigterm: bool = False, only_grandchild_ignores: bool = False) -> Iterator[_Tree]:
    """Start a leader-plus-grandchild tree in its own process group.

    Mirrors the shape Carmel actually launches — ``conda run`` (the group
    leader) with T3 and RMG beneath it — because a kill that reaches only
    the leader is exactly the defect these primitives exist to prevent.

    Args:
        ignore_sigterm: Whether both processes should ignore SIGTERM,
            forcing escalation to SIGKILL.
        only_grandchild_ignores: Whether the leader should honour SIGTERM
            while the grandchild ignores it. This is the asymmetric case:
            the group survives with its leader gone, so it is alive and
            unidentifiable at the same time — which is what distinguishes
            "stopped" from "no longer recognizable".

    Yields:
        The running :class:`_Tree`.
    """
    ignore_stmt = "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    leader_ignore = ignore_stmt if ignore_sigterm and not only_grandchild_ignores else ""
    child_ignore = ignore_stmt if ignore_sigterm or only_grandchild_ignores else ""
    child = f"import signal, time; {child_ignore}print('up', flush=True); time.sleep(300)"
    leader = (
        f"import signal, subprocess, sys, time; {leader_ignore}"
        f"grandchild = subprocess.Popen([sys.executable, '-c', {child!r}], stdout=subprocess.PIPE);"
        "print(grandchild.stdout.readline().decode().strip(), flush=True);"
        "print(grandchild.pid, flush=True); time.sleep(300)"
    )
    command = [sys.executable, "-c", leader]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "up"
    grandchild_pid = int(proc.stdout.readline().strip())
    try:
        yield _Tree(
            pgid=proc.pid,
            leader_pid=proc.pid,
            grandchild_pid=grandchild_pid,
            command=command,
            proc=proc,
        )
        # Reap the leader before asserting it is gone. It is this process's
        # own child, so it lingers as a zombie until waited on, and a zombie
        # still has a /proc entry. Nothing reaps it in production -- an
        # orphaned leader is reparented to init, which does -- so this is an
        # artifact of the test holding the handle, not of the code.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=15)
        assert _died_within(grandchild_pid), "the grandchild outlived the kill"
        assert _died_within(proc.pid), "the group leader outlived the kill"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)


@contextlib.contextmanager
def _shebang_leader_tree() -> Iterator[_Tree]:
    """Start a tree whose leader is launched through a ``#!`` wrapper.

    This is the shape a real ``conda run`` launch has, and it is the case a
    direct-exec :func:`_tool_tree` hides. When the kernel execs a ``#!``
    script it rewrites the argv to prepend the interpreter, so the argv
    Carmel passed to ``Popen`` — ``[<wrapper>, run, ...]`` — never matches
    what ``/proc`` reports — ``[<python>, <wrapper>, run, ...]``. Recording
    the launched argv and comparing it later would misfire on exactly this
    shape; recording the kernel-observed identity is what fixes it.

    ``_Tree.command`` is the launched argv, deliberately *not* the kernel's,
    so a test can prove the two differ and that identity still holds.
    """
    tmpdir = tempfile.mkdtemp()
    try:
        child = "import time; print('up', flush=True); time.sleep(300)"
        leader_code = (
            "import subprocess, sys, time\n"
            f"grandchild = subprocess.Popen([sys.executable, '-c', {child!r}], stdout=subprocess.PIPE)\n"
            "print(grandchild.stdout.readline().decode().strip(), flush=True)\n"
            "print(grandchild.pid, flush=True)\n"
            "time.sleep(300)\n"
        )
        wrapper = Path(tmpdir) / "conda"
        wrapper.write_text(f"#!{sys.executable}\n{leader_code}", encoding="utf-8")
        wrapper.chmod(0o755)
        command = [str(wrapper), "run", "-n", "t3_env", "--no-capture-output", "python", "-m", "T3"]
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True, start_new_session=True)
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "up"
        grandchild_pid = int(proc.stdout.readline().strip())
        try:
            yield _Tree(
                pgid=proc.pid,
                leader_pid=proc.pid,
                grandchild_pid=grandchild_pid,
                command=command,
                proc=proc,
            )
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=30)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _is_running(pid: int) -> bool:
    """Report whether *pid* is a live process, excluding zombies.

    ``kill(pid, 0)`` and a bare ``/proc/<pid>`` check both succeed for a
    zombie, so neither can tell "still executing" from "already killed and
    not yet reaped". Any assertion that a process was *left alone* has to
    exclude the zombie state, or it passes against code that killed it.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


def _died_within(pid: int, timeout_s: float = 15.0) -> bool:
    """Wait for *pid* to disappear entirely, zombie state included.

    A killed process is reparented and lingers as a zombie until it is
    reaped, and ``kill(pid, 0)`` succeeds for a zombie. Death has to be
    awaited rather than sampled.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.05)
    return False


def _strand_active_run(
    ws: Path,
    process_group_id: int | None,
    command: list[str] | None,
    leader_starttime: int | None = None,
) -> None:
    """Write the record a killed supervisor would have left behind.

    Written directly rather than through :func:`supervise_run` because the
    situation being reproduced is precisely the one that context manager
    never gets to finish: the process died holding it.

    ``leader_starttime`` defaults to None, which drives recovery down the
    command-line fallback path — the right default for a direct-exec
    :func:`_tool_tree`, whose launched argv does match ``/proc``. Pass the
    leader's real start time to exercise the reuse-proof identity, or a
    stale one to reproduce a recycled pid.
    """
    write_json(
        active_run_path(ws),
        ActiveRun(
            action_id="act-1",
            started_at=datetime.now(UTC),
            supervisor_pid=os.getpid(),
            process_group_id=process_group_id,
            command=command,
            leader_starttime=leader_starttime,
        ),
    )
