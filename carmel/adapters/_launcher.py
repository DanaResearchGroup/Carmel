# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0
"""Shared, tool-agnostic subprocess launcher core for adapter modules.

Both the T3 adapter (``carmel/adapters/t3.py``) and the ARC adapter
(``carmel/adapters/arc.py``) launch and probe a *different* external tool
that lives in its own environment under the three-env deployment model
(``rmg_env`` / ``t3_env`` / ``crml_env``). The launch/probe machinery —
process-group lifecycle management, conda-env resolution, subprocess-based
importability/version probing, executable discovery — is identical between
the two; only the specific env vars, executable names, and loggers differ.

This module holds that shared machinery. Every function that logs or
performs discovery takes its logger/``which``/``find_spec``/runner/terminate
callable as an INJECTED seam rather than importing or hardcoding one, so
that each adapter module can supply its own module-level objects (its own
``_log``, its own ``shutil``, its own wrapper around its own
``_run_in_process_group``). This is what lets each adapter's test suite go
on monkeypatching names on *its own* module (e.g. ``t3.shutil``,
``t3._log``) after the shared logic moved here: the adapter module's thin
wrapper functions pass those same (possibly monkeypatched) objects into this
module's functions on every call.
"""

from __future__ import annotations

import atexit
import contextlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from logging import Logger
from pathlib import Path
from typing import Any, Protocol

from carmel.logger import get_logger

_log = get_logger("adapters._launcher")


class _Runner(Protocol):
    """Callable shape of an adapter's own ``_run_in_process_group`` wrapper."""

    def __call__(
        self,
        command: list[str],
        *,
        timeout: float,
        cwd: Any = None,
        stdout: Any = None,
        stderr: Any = None,
        text: bool = False,
    ) -> subprocess.CompletedProcess[Any]: ...


class _Terminate(Protocol):
    """Callable shape of an adapter's own ``_terminate_process_tree`` wrapper."""

    def __call__(self, proc: subprocess.Popen[Any], grace_period_s: float | None = None) -> None: ...


# ---------------------------------------------------------------------------
# Process-tree lifecycle — a single shared registry for every adapter.
#
# A run executes on a daemon thread, which is *not* joined at exit, so
# without this a stopped server leaves T3/ARC (and RMG) running with no
# supervisor. One registry and one ``atexit`` hook cover every adapter that
# launches through this module, rather than each adapter module needing its
# own copy.
# ---------------------------------------------------------------------------

_LIVE_TREES: set[subprocess.Popen[Any]] = set()
_LIVE_TREES_LOCK = threading.Lock()


def _register_live_tree(proc: subprocess.Popen[Any]) -> None:
    """Track *proc* so it can be killed if the interpreter shuts down.

    Args:
        proc: A process started with ``start_new_session=True``.
    """
    with _LIVE_TREES_LOCK:
        _LIVE_TREES.add(proc)


def _forget_live_tree(proc: subprocess.Popen[Any]) -> None:
    """Stop tracking *proc*; it has finished or has already been killed.

    Args:
        proc: The process to forget.
    """
    with _LIVE_TREES_LOCK:
        _LIVE_TREES.discard(proc)


def _terminate_live_trees() -> None:
    """Kill every still-running tool process tree. Registered with ``atexit``."""
    with _LIVE_TREES_LOCK:
        survivors = [proc for proc in _LIVE_TREES if proc.poll() is None]
    for proc in survivors:
        _log.warning("Interpreter is shutting down; killing a tool process tree at pid %s", proc.pid)
        with contextlib.suppress(Exception):
            terminate_process_tree(proc, grace_period_s=10.0, logger=_log, tool_label="tool")


atexit.register(_terminate_live_trees)


def terminate_process_tree(
    proc: subprocess.Popen[Any], *, grace_period_s: float, logger: Logger, tool_label: str
) -> None:
    """Kill *proc* and every descendant sharing its process group, then reap it.

    *proc* must have been started with ``start_new_session=True``, which
    makes it the leader of a brand-new process group whose id equals its
    pid — so its pid doubles as the group id, and no ``getpgid`` lookup
    that could race with the child exiting is needed.

    The escalation is SIGTERM, then SIGKILL after *grace_period_s*, and
    the direct child is reaped **last**. That ordering is the whole point:
    an unreaped child — even a zombie — keeps its pid, and therefore the
    group id, reserved, so the SIGKILL cannot land on an unrelated process
    that recycled the pid in the meantime.

    Nothing here polls the child during the grace period, deliberately.
    ``Popen.poll`` *reaps* an exited child, which would release exactly the
    pid this function still needs reserved. Buying an early exit with that
    reservation is what makes the recycling race real, so the full grace is
    waited out instead and the SIGKILL is sent unconditionally — signalling
    an already-empty group is a harmless ESRCH. This is the timeout path;
    it has already waited hours, and a few more seconds cost nothing.

    On a platform without ``os.killpg`` (i.e. not POSIX) this degrades to
    killing the direct child only, which is what the standard library
    would have done anyway.

    Args:
        proc: The running child process.
        grace_period_s: Seconds to wait between SIGTERM and SIGKILL, and
            the bound on each subsequent attempt to reap the child.
        logger: Logger to use if the process cannot be reaped after SIGKILL.
        tool_label: Human-facing name of the tool whose process tree this
            is (e.g. ``"T3"``/``"ARC"``), used only in the log message text
            below.
    """
    if not hasattr(os, "killpg"):  # pragma: no cover -- POSIX-only branch
        proc.kill()
        proc.wait()
        return

    pgid = proc.pid
    # A group that is already gone is a success, not an error: the tree can
    # exit between the timeout firing and the signal being delivered.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGTERM)
    time.sleep(grace_period_s)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, signal.SIGKILL)

    # Bounded, because an unbounded wait here would hang the calling thread
    # forever in the one case that matters: a SIGKILL that never landed.
    try:
        proc.wait(timeout=grace_period_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            proc.kill()
        try:
            proc.wait(timeout=grace_period_s)
        except subprocess.TimeoutExpired:
            logger.error(
                f"Could not reap process %s after SIGKILL; a {tool_label} process tree may still be running "
                "as group %s",
                proc.pid,
                pgid,
            )


def run_in_process_group(
    command: list[str],
    *,
    timeout: float,
    cwd: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    text: bool = False,
    terminate: _Terminate,
    register: Any = _register_live_tree,
    forget: Any = _forget_live_tree,
    drain_timeout: float = 10.0,
) -> subprocess.CompletedProcess[Any]:
    """Run *command* in its own process group, killing the whole tree on timeout.

    A drop-in replacement for ``subprocess.run(...)`` for every call that
    launches a tool (directly or through ``conda run``). The difference is
    the only one that matters here: when the timeout expires, or the
    calling thread is interrupted, every descendant is killed rather than
    just the process the caller happens to hold a handle to.

    Args:
        command: The argv to execute.
        timeout: Seconds to wait before killing the tree.
        cwd: Working directory for the child.
        stdout: Passed through to ``Popen`` (a file object, ``PIPE``, or None).
        stderr: Passed through to ``Popen``.
        text: Whether to decode captured output as text.
        terminate: Callable used to kill the tree on timeout/interruption —
            the caller's own ``_terminate_process_tree`` wrapper, so it
            applies the caller's own grace period and logger.
        register: Callable used to track the started process for the
            shutdown sweep. Defaults to this module's own registry, but a
            caller wrapper should pass its *own* module-level name (e.g.
            T3's ``_register_live_tree``) looked up dynamically, so a test
            that monkeypatches that name on the caller's module still takes
            effect here.
        forget: Callable used to stop tracking the process once it is done.
            Same rationale as *register*.
        drain_timeout: Seconds to wait while draining and closing the
            pipes after *terminate* has already killed the tree. Defaults
            to 10.0; a caller wrapper should pass its own kill-grace-period
            constant (e.g. T3's ``_KILL_GRACE_PERIOD_S``), read at call
            time so it can be shortened for tests.

    Returns:
        A ``CompletedProcess`` exactly as ``subprocess.run`` would return.

    Raises:
        subprocess.TimeoutExpired: If *timeout* expires. The process tree
            has already been killed and reaped when this propagates.
        OSError: If the child cannot be spawned at all.
    """
    proc: subprocess.Popen[Any] = subprocess.Popen(  # noqa: S603 -- resolved, locally-configured commands only
        command,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
    )
    register(proc)
    try:
        captured_stdout, captured_stderr = proc.communicate(timeout=timeout)
    except BaseException:
        # Covers TimeoutExpired and anything that interrupts the wait
        # (KeyboardInterrupt, a thread being torn down): in every case the
        # tree must not outlive the call that started it.
        terminate(proc)
        # Drain and close the pipes. communicate() abandoned them when it
        # raised, and the tree is dead now, so this returns immediately —
        # without it the read ends stay open until the Popen is collected.
        with contextlib.suppress(Exception):
            proc.communicate(timeout=drain_timeout)
        raise
    finally:
        forget(proc)
    return subprocess.CompletedProcess(command, proc.returncode, captured_stdout, captured_stderr)


# ---------------------------------------------------------------------------
# Discovery / availability
# ---------------------------------------------------------------------------


def resolve_python(python_env_vars: list[str], *, logger: Logger) -> str:
    """Resolve the interpreter to launch/probe a tool with, absent a conda env.

    Args:
        python_env_vars: Env var names to check, in precedence order. The
            first one that is set and points to an existing, executable
            file wins.
        logger: Logger used to warn if a var is set but unusable.

    Returns:
        The first usable ``$VAR`` value, or ``sys.executable`` if none of
        *python_env_vars* is usable.
    """
    for var in python_env_vars:
        env_python = os.environ.get(var)
        if not env_python:
            continue
        if os.path.isfile(env_python) and os.access(env_python, os.X_OK):
            return env_python
        logger.warning(
            "%s is set to %r but is not an existing executable file; falling back to %s",
            var,
            env_python,
            sys.executable,
        )
    return sys.executable


def launch_command(
    sources: list[tuple[str, str]],
    *,
    logger: Logger,
    tool_label: str,
    which: Any = shutil.which,
) -> list[str]:
    """Resolve the argv prefix used to run a python interpreter for a tool's environment.

    Walks *sources* — an ordered list of ``(kind, env_var)`` pairs, ``kind``
    one of ``"conda"``/``"python"`` — in precedence order. An empty-string
    value counts as unset. The first source that is both set and usable
    wins:
        - A ``"conda"`` source that is set and a ``conda`` executable is
          discoverable via *which*: returns ``[conda, "run", "-n", <env>,
          "--no-capture-output", "python"]``.
        - A ``"conda"`` source that is set but ``conda`` is not found: logs
          a warning and continues to the next source (lenient — a
          misconfigured env var must never crash the caller). When there is
          a next source in *sources*, the warning names it directly (e.g.
          ``"falling back to T3_PYTHON"``); otherwise it names *tool_label*
          generically.
        - A ``"python"`` source that is set: returns ``[<that value>]``
          only if it is an existing, executable file (same check as
          :func:`resolve_python`). Otherwise logs a warning naming
          ``sys.executable`` as the fallback and continues to the next
          source — a set-but-invalid python path must never be selected
          and handed to the caller, which would otherwise die later with a
          raw OSError trying to exec it.

    If nothing in *sources* is set/usable, falls through to
    ``[sys.executable]`` — any set-and-valid python source above already
    returned, so by this point every python source encountered was either
    unset or already warned about.

    Args:
        sources: Ordered ``(kind, env_var)`` pairs.
        logger: Logger used to warn on a set-but-broken source.
        tool_label: Human-facing name of the tool (e.g. ``"T3"``/``"ARC"``),
            used only in message text when no more specific hint (the next
            source's env var name) is available.
        which: Injected in place of ``shutil.which``, so tests can control
            whether ``conda`` is "found".

    Returns:
        The argv prefix to prepend to a python invocation to run it inside
        the tool's environment.
    """
    for idx, (kind, env_var) in enumerate(sources):
        value = os.environ.get(env_var)
        if not value:
            continue
        if kind == "conda":
            conda = which("conda")
            if conda is not None:
                return [conda, "run", "-n", value, "--no-capture-output", "python"]
            next_var = sources[idx + 1][1] if idx + 1 < len(sources) else None
            if next_var is not None:
                logger.warning(
                    "%s is set to %r but no 'conda' executable was found on PATH; falling back to %s",
                    env_var,
                    value,
                    next_var,
                )
            else:
                logger.warning(
                    "%s is set to %r but no 'conda' executable was found on PATH; trying the next %s source",
                    env_var,
                    value,
                    tool_label,
                )
            continue
        if os.path.isfile(value) and os.access(value, os.X_OK):
            return [value]
        logger.warning(
            "%s is set to %r but is not an existing executable file; falling back to %s",
            env_var,
            value,
            sys.executable,
        )
    return [sys.executable]


def conda_env_error(
    sources: list[tuple[str, str]],
    *,
    probe_timeout: float,
    runner: _Runner,
    tool_label: str,
    which: Any = shutil.which,
) -> str | None:
    """Return why the tool's resolved environment cannot be used, or None if fine.

    :func:`launch_command` is deliberately non-raising: it always returns
    *some* usable argv prefix. That is correct as internal plumbing
    (discovery must never throw), but it must never be used to silently
    launch a tool under the wrong interpreter when an operator explicitly
    asked for a named conda environment. This function is the explicit,
    callable check an adapter's ``run()`` uses to turn a broken conda
    source into a clean, typed failure instead of a silent downgrade.

    Walks *sources* in precedence order:
        - A SET ``"conda"`` source: if ``conda`` is missing, returns a
          typed message naming that var. Otherwise probes ``conda run -n
          <env> --no-capture-output python -c pass`` via *runner*; a
          nonzero return code or a raised (OSError, SubprocessError)
          returns a typed message naming that var. On success, returns
          None. Either way, resolution STOPS here — a set-but-broken
          conda source never falls through to a later source.
        - A SET ``"python"`` source: if it exists and is executable,
          short-circuits to None (no further sources checked). Otherwise
          continues to the next source (validity of the *chosen* fallback
          is what matters, not every python source encountered).
        - An unset source is skipped.

    If no source is set, or resolution falls off the end, returns None.

    Args:
        sources: Ordered ``(kind, env_var)`` pairs, same shape as
            :func:`launch_command`.
        probe_timeout: Timeout, in seconds, for the trivial conda probe.
        runner: The caller's own process-group-aware runner (e.g. its
            ``_run_in_process_group`` wrapper).
        tool_label: Human-facing name of the tool (e.g. ``"T3"``/``"ARC"``),
            used only in the "refusing to silently launch" message text.
        which: Injected in place of ``shutil.which``.

    Returns:
        None if every set source is usable (or nothing is set). Otherwise
        a human-readable message identifying the first broken source,
        suitable for direct use as a ``RunRecord.error_message``.
    """
    for kind, env_var in sources:
        value = os.environ.get(env_var)
        if not value:
            continue
        if kind == "conda":
            conda = which("conda")
            if conda is None:
                return (
                    f"{env_var} is set to {value!r} but no 'conda' executable "
                    f"was found on PATH; refusing to silently launch {tool_label} under a different interpreter"
                )
            try:
                completed = runner(
                    [conda, "run", "-n", value, "--no-capture-output", "python", "-c", "pass"],
                    timeout=probe_timeout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            except (OSError, subprocess.SubprocessError) as e:
                return f"{env_var} is set to {value!r} but conda could not run it: {e}"
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                return (
                    f"{env_var} is set to {value!r} but conda could not run "
                    f"a trivial command in it (exit {completed.returncode}); the environment may not "
                    f"exist or may be corrupt: {detail}"
                )
            return None
        if os.path.isfile(value) and os.access(value, os.X_OK):
            return None
    return None


def probe_importable(python_command: list[str], module_name: str, *, timeout: float, runner: _Runner) -> bool:
    """Return True if *module_name* can actually be imported by *python_command*.

    Args:
        python_command: The resolved argv prefix (see :func:`launch_command`).
        module_name: The module to try importing.
        timeout: Seconds to wait before killing the probe.
        runner: The caller's own process-group-aware runner.
    """
    try:
        completed = runner(
            [*python_command, "-c", f"import {module_name}"],
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError, subprocess.SubprocessError:
        return False
    return completed.returncode == 0


def probe_version(python_command: list[str], module_name: str, *, timeout: float, runner: _Runner) -> str | None:
    """Return *module_name*'s version string if importable by *python_command*, else None.

    Args:
        python_command: The resolved argv prefix (see :func:`launch_command`).
        module_name: The module to import and read ``__version__`` from.
        timeout: Seconds to wait before killing the probe.
        runner: The caller's own process-group-aware runner.
    """
    try:
        completed = runner(
            [*python_command, "-c", f"import {module_name}; print(getattr({module_name}, '__version__', ''))"],
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if completed.returncode != 0:
        return None
    version = completed.stdout.strip()
    return version or None


def find_executable(
    *,
    path_env_var: str,
    executable_script: str,
    executable_module: str,
    package_name: str,
    python_command: list[str],
    host_discovery_allowed: bool,
    is_importable: Any,
    which: Any = shutil.which,
    find_spec: Any = importlib.util.find_spec,
) -> list[str] | None:
    """Locate a tool's executable, mirroring T3's original discovery order.

    Preference order:
        1. ``$<path_env_var>/<executable_script>`` if that env var is set
           and the file exists (explicit operator wins, always checked).
        2. If *host_discovery_allowed*: try ``find_spec(package_name)`` to
           find the script alongside the installed package, then
           ``which(executable_script)`` on ``$PATH``.
        3. Fallback: ``[*python_command, "-m", executable_module]``, but
           ONLY if ``is_importable()`` returns True. Otherwise None.

    *host_discovery_allowed* encodes the conda-authoritative rule: when a
    conda env has been explicitly selected for the tool, Carmel's own
    ``find_spec``/``which`` discovery (which runs in Carmel's own
    interpreter/environment) risks finding the wrong installation, so that
    step is skipped in favor of going straight to the ``-m`` fallback
    launched through the resolved *python_command*.

    The final ``-m`` fallback is gated on *is_importable* — called only if
    reached — rather than returned unconditionally: a tool that cannot
    actually be imported by *python_command* would otherwise be "found" as
    a command that is certain to fail the moment it runs.

    Args:
        path_env_var: Env var giving an explicit directory containing
            *executable_script*.
        executable_script: The script filename to look for.
        executable_module: The module name to use with the ``-m`` fallback.
        package_name: The importable package name to resolve via *find_spec*.
        python_command: The resolved argv prefix (see :func:`launch_command`).
        host_discovery_allowed: Whether step 2's find_spec/which checks
            should run.
        is_importable: Zero-arg callable (e.g. the caller's own
            ``is_t3_importable``) invoked only if steps 1-2 found nothing,
            to gate the ``-m`` fallback.
        which: Injected in place of ``shutil.which``.
        find_spec: Injected in place of ``importlib.util.find_spec``.

    Returns:
        The argv to invoke the tool, or None if nothing was found and the
        tool is not importable either.
    """
    env_path = os.environ.get(path_env_var)
    if env_path:
        candidate = Path(env_path) / executable_script
        if candidate.exists():
            return [*python_command, str(candidate)]
    if host_discovery_allowed:
        spec = find_spec(package_name)
        if spec is not None and spec.origin is not None:
            # spec.origin is .../<package>/__init__.py; the script lives at
            # the repo root, one directory up from the package.
            candidate = Path(spec.origin).parent.parent / executable_script
            if candidate.exists():
                return [*python_command, str(candidate)]
        found = which(executable_script)
        if found is not None:
            return [*python_command, found]
    if is_importable():
        return [*python_command, "-m", executable_module]
    return None
