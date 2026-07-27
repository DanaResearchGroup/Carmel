# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Run record schemas."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RunStatus(StrEnum):
    """Status of a tool run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailureCode(StrEnum):
    """Typed failure codes for tool runs."""

    NONE = "none"
    SUBPROCESS_ERROR = "subprocess_error"
    INVALID_OUTPUT = "invalid_output"
    TIMEOUT = "timeout"
    TOOL_NOT_FOUND = "tool_not_found"
    INPUT_BUILD_ERROR = "input_build_error"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


class SubmissionMode(StrEnum):
    """How the tool run was submitted."""

    SUBPROCESS = "subprocess"
    SERVER = "server"
    LOCAL = "local"


class ActiveRun(BaseModel):
    """A tool run believed to be in flight, and what to reap if it is not.

    Written when a run starts and removed when it finishes, so its mere
    presence means "a run was started and never recorded an ending". That
    is not the same as "a run is still going": the Carmel process
    supervising it may have been killed. Liveness is established
    separately, from the supervisor lock and the recorded process group —
    see :mod:`carmel.services.recovery`.
    """

    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1)
    started_at: datetime
    supervisor_pid: int = Field(gt=0)
    """The Carmel process that started the run. Recorded for operators to
    read; liveness is never inferred from it, because a pid outlives its
    process only until the kernel reuses the number."""

    process_group_id: int | None = Field(default=None, gt=0)
    """The tool's process group, once it has been launched. None while the
    run is being prepared, and on any run whose tool never started."""

    command: list[str] | None = None
    """The group leader's kernel-observed argv, read back from ``/proc``
    when the tool was launched. A human-readable label and the fallback
    identity for records written before ``leader_starttime`` existed; not
    the primary identity, because a reused pid can rerun the same argv."""

    leader_starttime: int | None = Field(default=None, ge=0)
    """The group leader's start time (``/proc/<pid>/stat`` field 22), in
    clock ticks since boot. The reuse-proof identity confirming a surviving
    group really is this run's before anything signals it: the kernel does
    not carry a start time over when it recycles a pid. None on records
    written before this field, and when ``/proc`` could not be read at
    launch."""


class RunRecord(BaseModel):
    """A record of a tool execution attempt."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tool_version: str | None = None
    status: RunStatus
    failure_code: FailureCode = FailureCode.NONE
    started_at: datetime
    ended_at: datetime | None = None
    estimated_cpu_hours: float = Field(default=0.0, ge=0)
    actual_cpu_hours: float | None = None
    submission_mode: SubmissionMode
    command: list[str] | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    level_of_theory: str | None = None
    error_message: str | None = None
