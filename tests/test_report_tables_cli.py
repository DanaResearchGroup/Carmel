# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""CLI wiring for ``report-tables --extract-series`` (I-103).

The library function ``report_document_tables`` already takes ``series_agent`` and is tested in
``test_general_table_report``; these tests pin the CLI boundary only: the flag is OPT-IN (off makes
zero model calls), on with an injected mock stores a replayable series, per-candidate isolation
survives the CLI boundary, and every missing-configuration path fails CLOSED with a clean non-zero
exit that names what is missing. No test reaches the network, writes into the operator's live
corpus, or depends on ``~/.carmel`` being writable: the injected-agent seam uses a MockModel with a
``daily_ledger_path=None`` ledger, and the missing-config cases never run a model at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import Carmel
from carmel.services.dataset_replay import ReplayOutcome, replay_stored_dataset
from carmel.services.dataset_store import verify_dataset
from tests.test_general_table_report import (
    _extraction,
    _inject,
    _measured_grid,
    _place_measured_document,
    _series_agent,
    _series_proposal,
)


def _base_config(tmp_path: Path, agents: dict[str, object] | None) -> dict[str, object]:
    """A minimal otherwise-valid CarmelConfig mapping, optionally carrying an [agents] section."""
    config: dict[str, object] = {"workspace_name": "cli-test", "workspace_root": str(tmp_path / "ws")}
    if agents is not None:
        config["agents"] = agents
    return config


def _series_sha_from_stdout(out: str) -> str:
    for line in out.splitlines():
        if "stored series sha256" in line:
            return line.split(":", 1)[1].strip().split()[0]
    raise AssertionError(f"no stored-series line in output:\n{out}")


class TestFlagOff:
    def test_off_makes_no_model_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default path constructs no agent and calls no model, even when one is injected."""
        sha = _place_measured_document(tmp_path, monkeypatch)
        agent = _series_agent([_series_proposal(sha256=sha)])

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=False, series_agent=agent)

        assert code == 0
        assert agent.model.calls == []  # the injected model was never touched

    def test_off_output_reports_the_deferred_series(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sha = _place_measured_document(tmp_path, monkeypatch)

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=False)

        assert code == 0
        out = capsys.readouterr().out
        assert "[STORED]" in out
        assert "series deferred" in out
        assert "Model spend" not in out  # no agent, so no spend line


class TestFlagOn:
    def test_on_stores_a_replayable_series(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sha = _place_measured_document(tmp_path, monkeypatch)
        agent = _series_agent([_series_proposal(sha256=sha)])

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, series_agent=agent)

        assert code == 0
        out = capsys.readouterr().out
        assert "[SERIES_STORED]" in out
        assert "Model spend" in out
        assert len(agent.model.calls) == 1

        series_sha = _series_sha_from_stdout(out)
        # The dataset is genuinely on disk, canonical, and content-addressed (verify_dataset), and an
        # independent off-disk replay finds no FAILED cell. (The synthetic mock grid carries a
        # dimensionless coordinate the char-span replay marks UNVERIFIABLE rather than VERIFIED; a
        # FAILED finding would mean a stored value that does not reproduce, which SERIES_STORED forbids.)
        assert verify_dataset(tmp_path, series_sha) is True
        report = replay_stored_dataset(tmp_path, series_sha)
        assert report.overall_outcome is not ReplayOutcome.FAILED

    def test_on_per_candidate_isolation_holds_at_exit_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Two MEASURED grids: one series succeeds, one fails. Both reported, exit 0, no abort."""
        from tests.test_dataset_producer import _store_synthetic_artifact
        from tests.test_general_table_report import _SERIES_DOCUMENT_TEXT

        _inject(monkeypatch, _extraction(*_measured_grid(page=1), *_measured_grid(page=2)))
        sha = _store_synthetic_artifact(tmp_path, _SERIES_DOCUMENT_TEXT).sha256
        agent = _series_agent([_series_proposal(sha256=sha), {"not": "a valid tabular series proposal"}])

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, series_agent=agent)

        assert code == 0
        out = capsys.readouterr().out
        assert "[SERIES_STORED]" in out
        assert "[SERIES_REFUSED]" in out


class TestFailsClosed:
    def test_extract_series_without_config_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sha = _place_measured_document(tmp_path, monkeypatch)

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, config=None)

        assert code == 2
        err = capsys.readouterr().err
        assert "--config" in err
        assert "no series" not in err.lower()  # never a silent "produced nothing" success

    def test_config_without_agents_section_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sha = _place_measured_document(tmp_path, monkeypatch)
        config = tmp_path / "no_agents.yaml"
        config.write_text(yaml.dump(_base_config(tmp_path, None)), encoding="utf-8")

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, config=config)

        assert code == 2
        err = capsys.readouterr().err
        assert "[agents]" in err

    def test_missing_config_file_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sha = _place_measured_document(tmp_path, monkeypatch)

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, config=tmp_path / "absent.yaml")

        assert code == 2
        assert "absent.yaml" in capsys.readouterr().err

    def test_malformed_yaml_config_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Malformed YAML raises ``yaml.YAMLError`` from ``load_config``, not a ``ValueError`` --
        this must still be caught and turned into a clean fail-closed exit, never a traceback."""
        sha = _place_measured_document(tmp_path, monkeypatch)
        config = tmp_path / "malformed.yaml"
        config.write_text("agents: [unterminated\n", encoding="utf-8")

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, config=config)

        assert code == 2
        err = capsys.readouterr().err
        assert "cannot load --config" in err

    def test_provider_without_credentials_exits_nonzero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A real provider with no data-egress consent cannot build a model -- fail closed, never
        a silent downgrade to the mock, and never a traceback."""
        sha = _place_measured_document(tmp_path, monkeypatch)
        config = tmp_path / "needs_creds.yaml"
        config.write_text(
            yaml.dump(
                _base_config(
                    tmp_path,
                    {
                        "tier": "prod",
                        "provider": "openai",
                        "api_key_env": "CARMEL_TEST_UNSET_KEY",
                        "external_provider_consent": False,
                    },
                )
            ),
            encoding="utf-8",
        )

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, config=config)

        assert code == 2
        assert "cannot construct" in capsys.readouterr().err.lower()

    def test_a_budget_guard_trip_is_a_clean_nonzero_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A ``BudgetExceededError`` escaping the report (the guard refusing further spend mid-run)
        is converted to a clean, named non-zero exit rather than a traceback."""
        from carmel.agents.budget import BudgetDimension, BudgetExceededError

        sha = _place_measured_document(tmp_path, monkeypatch)
        agent = _series_agent([_series_proposal(sha256=sha)])

        def _boom(*_args: object, **_kwargs: object) -> object:
            raise BudgetExceededError(BudgetDimension.COST_USD, 1.0, 0.0)

        monkeypatch.setattr("carmel.services.general_table_report.report_document_tables", _boom)

        code = Carmel._cmd_report_tables(tmp_path, sha, extract_series=True, series_agent=agent)

        assert code == 1
        assert "budget" in capsys.readouterr().err.lower()
