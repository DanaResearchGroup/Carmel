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
    UNKNOWN = "unknown"


class SubmissionMode(StrEnum):
    """How the tool run was submitted."""

    SUBPROCESS = "subprocess"
    SERVER = "server"
    LOCAL = "local"


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
    log_path: Path | None = None
    level_of_theory: str | None = None
    error_message: str | None = None
