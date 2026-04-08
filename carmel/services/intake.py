"""Assistive free-text intake parsing protocol and stub backend.

The intake parser is intentionally a typed protocol with a no-op stub
backend. Real LLM-backed parsing will be added in a later phase. The
parser MUST never produce canonical campaign artifacts directly — it
emits a typed review note that the user must accept before it becomes
canonical state.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from carmel.services.artifacts import write_text

INTAKE_REVIEW_FILE_NAME = "intake_review.md"


class IntakeParser(Protocol):
    """Protocol for free-text intake parsers."""

    def parse(self, free_text: str) -> str:
        """Parse free text into a review note.

        Args:
            free_text: The user-provided free text.

        Returns:
            A markdown review note describing what the parser inferred.
            This is *advisory only* and must not become canonical state
            without explicit user confirmation.
        """
        ...


class StubIntakeParser:
    """No-op stub parser used when no real backend is configured."""

    def parse(self, free_text: str) -> str:
        """Echo the free text back inside a review note."""
        timestamp = datetime.now(UTC).isoformat()
        return (
            f"# Intake Review\n\n"
            f"_Generated at {timestamp} by the stub parser._\n\n"
            f"No structured fields were inferred from the free-text input. "
            f"This is an assistive review only — please populate the structured "
            f"campaign form to create the canonical campaign.\n\n"
            f"## Original Input\n\n"
            f"```\n{free_text}\n```\n"
        )


def write_intake_review(workspace_root: Path, review: str) -> Path:
    """Persist an intake review note to the workspace.

    Args:
        workspace_root: The campaign workspace root.
        review: The review markdown content.

    Returns:
        The path of the written file.
    """
    path = workspace_root / INTAKE_REVIEW_FILE_NAME
    write_text(path, review)
    return path
