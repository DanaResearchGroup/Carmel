# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for P1-11: ``AcquisitionRequest.landing_url`` scheme validation.

Kept in its own file (rather than added to ``tests/test_acquisition.py``) because that
file has other in-flight edits from a concurrent session on this branch; this file only
exercises the schema-level validator added to
:class:`carmel.schemas.acquisition.AcquisitionRequest` and does not touch anything else.

``landing_url`` reaches an ``<a href=...>`` in the operator dashboard unsanitized. Before
this fix, ``AcquisitionRequest`` (and, in ``carmel/agents/literature_agent.py``,
``RequestedPaper.landing_url`` -- see this module's docstring note for why that one is
not fixed here) accepted any non-empty string, so an LLM-authored or otherwise
attacker-influenced value of ``javascript:...`` could reach that ``href`` and execute
when a human operator clicked it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from carmel.agents.literature_agent import RequestedPaper
from carmel.schemas.acquisition import AcquisitionReason, AcquisitionRequest


def _make_request(landing_url: str) -> AcquisitionRequest:
    return AcquisitionRequest(
        slug="smith-2020",
        title="A Paper",
        doi="10.1/x",
        landing_url=landing_url,
        reason=AcquisitionReason.PAYWALLED,
        requested_at=datetime.now(UTC),
    )


class TestLandingUrlSchemeValidation:
    @pytest.mark.parametrize(
        "unsafe_url",
        [
            "javascript:alert(document.cookie)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
        ],
    )
    def test_rejects_unsafe_schemes(self, unsafe_url: str) -> None:
        with pytest.raises(ValidationError):
            _make_request(unsafe_url)

    @pytest.mark.parametrize(
        "safe_url",
        [
            "https://doi.org/10.1/x",
            "http://example.com/paper.pdf",
        ],
    )
    def test_accepts_http_and_https(self, safe_url: str) -> None:
        request = _make_request(safe_url)
        assert request.landing_url == safe_url


class TestRequestedPaperLandingUrl:
    """The LLM-authored twin of the field above.

    ``RequestedPaper.landing_url`` is what the Literature Agent proposes when it cannot
    obtain a paper, and it flows into ``AcquisitionRequest`` and on to the dashboard's
    "obtain" link. Validating only the downstream model would leave the agent free to
    hold an unsafe value in memory and would rely on the hand-off never being widened.
    """

    @pytest.mark.parametrize(
        "unsafe_url",
        [
            "javascript:alert(document.cookie)",
            "data:text/html,<script>alert(1)</script>",
            "file:///etc/passwd",
        ],
    )
    def test_rejects_unsafe_schemes(self, unsafe_url: str) -> None:
        with pytest.raises(ValidationError):
            RequestedPaper(title="A paper", landing_url=unsafe_url)

    def test_accepts_http_and_https_and_none(self) -> None:
        assert RequestedPaper(title="A paper", landing_url="https://doi.org/10.1/x").landing_url
        assert RequestedPaper(title="A paper", landing_url="http://example.com/p.pdf").landing_url
        assert RequestedPaper(title="A paper").landing_url is None
