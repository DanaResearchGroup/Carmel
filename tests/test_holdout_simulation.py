"""Tests for the frozen-holdout simulation harness."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import carmel.benchmarks.cantera_worker as cantera_worker
import carmel.benchmarks.holdout as holdout
from Carmel import main
from carmel.benchmarks.cantera_worker import (
    THREAD_ENV,
    RunLevelSimulationError,
    _row_fingerprint,
    _write_progress,
    cantera_validation_refusals,
    execute_rows,
    mechanism_semantic_sha256,
    parse_ck2yaml_log,
    simulation_lock,
    simulation_lock_sha256,
    trim_unsound_plog_levels,
)
from carmel.benchmarks.chemkin import normalize_for_ck2yaml
from carmel.benchmarks.holdout import (
    WorkerRequest,
    WorkerRequestRow,
    WorkerResponse,
    build_separation_report,
    invoke_cantera_worker,
    run_harness,
    score_metric,
)

_CANTERA_PYTHON = Path(os.environ.get("T3_PYTHON") or "T3_PYTHON-unset")
_OLD_MECHANISM = Path("/home/alon/runs/i116-alkane-census/evidence/nc5-49.txt")
_OLD_THERMO = Path("/home/alon/runs/i127-holdout-sim/input/nc5_49_therm.dat")
_TEST_CANTERA_VERSION = "3.2.0"


class _Quantity:
    def __init__(self, exact: float) -> None:
        self.exact = exact

    def converted_to_base(self) -> _Quantity:
        return self


def _fake_slow_row(_: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    time.sleep(1.0)
    return "row", {"row_id": row["row_id"], "status": "ok", "mole_fractions": {}}


def _fake_fast_row(_: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return "row", {
        "row_id": row["row_id"],
        "status": "ok",
        "mole_fractions": {"A": 0.25},
        "wall_time_s": 0.001,
        "integrator_steps": 7,
    }


def _must_not_run(_: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    raise AssertionError(f"resumed row was unexpectedly rerun: {row['row_id']}")


def _fake_solver_crash(_: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _write_progress(
        "integration",
        last_reactor_time_s=0.125,
        temperature_k=float(row["temperature_k"]),
        integrator_steps=1,
    )
    os.kill(os.getpid(), signal.SIGSEGV)
    raise AssertionError("SIGSEGV did not terminate the child")


def _fake_construction_crash(_: str, row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _write_progress("construction", temperature_k=float(row["temperature_k"]), integrator_steps=0)
    os.kill(os.getpid(), signal.SIGSEGV)
    raise AssertionError("SIGSEGV did not terminate the child")


def _worker_row() -> dict[str, Any]:
    return {
        "row_id": "fake-row",
        "temperature_k": 700.0,
        "pressure_pa": 101325.0,
        "residence_time_s": 2.0,
        "volume_m3": 8.5e-5,
        "composition": {"A": 1.0},
        "output_species": ["A"],
    }


def test_ck2yaml_normalization_rewrites_only_fortran_eless_exponents() -> None:
    source = (
        b"LOW / 1.5270+104 -2.4826E+01 9.4800E+04 /\n"
        b"RATE 2.5-03 ! comment 7.25+99 and 1.0E+04\n"
        b"SPECIES C5H10-2 TOKEN1.0+04\n"
    )

    result = normalize_for_ck2yaml(source, file="mechanism.inp")

    expected = (
        b"LOW / 1.5270E+104 -2.4826E+01 9.4800E+04 /\n"
        b"RATE 2.5E-03 ! comment 7.25+99 and 1.0E+04\n"
        b"SPECIES C5H10-2 TOKEN1.0+04\n"
    )
    assert result.data == expected
    assert result.raw_sha256 == hashlib.sha256(source).hexdigest()
    assert result.normalized_sha256 == hashlib.sha256(expected).hexdigest()
    assert [rewrite.model_dump() for rewrite in result.rewrites] == [
        {"file": "mechanism.inp", "line": 1, "before": "1.5270+104", "after": "1.5270E+104"},
        {"file": "mechanism.inp", "line": 2, "before": "2.5-03", "after": "2.5E-03"},
    ]


def _unsound_plog() -> dict[str, object]:
    return {
        "reactions": [
            {
                "equation": "A <=> B",
                "type": "pressure-dependent-Arrhenius",
                "rate-constants": [
                    {"P": "31.6 atm", "A": 1.0, "b": 0.0, "Ea": 0.0},
                    {"P": "100 atm", "A": -1.0, "b": 0.0, "Ea": 0.0},
                    {"P": "100 atm", "A": 0.5, "b": 0.0, "Ea": 0.0},
                ],
                "source_line": 17,
            }
        ]
    }


def test_plog_trim_removes_unsound_level_outside_scored_interval() -> None:
    trimmed, trims, refusals = trim_unsound_plog_levels(_unsound_plog(), [10 * 101325.0])

    assert refusals == []
    assert len(trimmed["reactions"][0]["rate-constants"]) == 1  # type: ignore[index]
    assert trims[0]["reaction"] == "A <=> B"
    assert trims[0]["source_line"] == 17
    assert trims[0]["removed_pressure_pa"] == 100 * 101325.0
    assert trims[0]["kept_bracket_pa"] == [31.6 * 101325.0, None]
    assert trims[0]["failing_temperatures_k"] == [300.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]


def test_plog_trim_refuses_row_inside_affected_interval() -> None:
    _, _, refusals = trim_unsound_plog_levels(_unsound_plog(), [50 * 101325.0])

    assert refusals == [
        {
            "pressure_pa": 50 * 101325.0,
            "reaction": "A <=> B",
            "reason": "mechanism_rate_unsound_at_condition",
            "removed_pressure_pa": 100 * 101325.0,
        }
    ]


def test_plog_validation_accepts_scientific_notation_pressures() -> None:
    mechanism = {
        "reactions": [
            {
                "equation": "A <=> B",
                "type": "pressure-dependent-Arrhenius",
                "rate-constants": [
                    {"P": "1e-6 atm", "A": 1.0, "b": 0.0, "Ea": 0.0},
                    {"P": "1.0E+01 atm", "A": 1.0, "b": 0.0, "Ea": 0.0},
                    {"P": "1e5 Pa", "A": 1.0, "b": 0.0, "Ea": 0.0},
                ],
            }
        ]
    }

    unchanged, trims, refusals = trim_unsound_plog_levels(mechanism, [])

    assert unchanged == mechanism
    assert trims == []
    assert refusals == []


@pytest.mark.parametrize(
    "mechanism",
    [
        {
            "units": {"activation-energy": "kcal/mol"},
            "reactions": [
                {
                    "equation": "A <=> B",
                    "type": "pressure-dependent-Arrhenius",
                    "rate-constants": [
                        {"P": "1 atm", "A": 1.0, "b": 0.0, "Ea": 0.0},
                        {"P": "1 atm", "A": -2.0, "b": 0.0, "Ea": 20.0},
                    ],
                }
            ],
        },
        {
            "units": {"activation-energy": "cal/mol"},
            "reactions": [
                {
                    "equation": "A <=> B",
                    "type": "pressure-dependent-Arrhenius",
                    "rate-constants": [
                        {"P": "1 atm", "A": 1.0, "b": 0.0, "Ea": 0.0},
                        {"P": "1 atm", "A": -2.0, "b": 0.0, "Ea": "2.0e+1 kcal/mol"},
                    ],
                }
            ],
        },
    ],
)
def test_plog_validation_honors_inherited_and_explicit_activation_energy_units(
    mechanism: dict[str, object],
) -> None:
    unchanged, trims, refusals = trim_unsound_plog_levels(mechanism, [101325.0])

    assert unchanged == mechanism
    assert trims == []
    assert refusals == []


def test_ck2yaml_warning_allowlist_refuses_unknown_class() -> None:
    result = parse_ck2yaml_log("Unexpected converter warning\n")

    assert result["refusals"] == ["unallowlisted ck2yaml output: Unexpected converter warning"]


def test_ck2yaml_warning_allowlist_accepts_only_authorized_classes() -> None:
    log = (
        "Ignoring redundant declaration for species A\n"
        "Ignoring redundant thermo data for species B\n"
        "Unparsable lines while reading thermo data in thermo.dat starting on line 8:\n"
        '"""\n'
        "!!!!\n"
        "  !  \n"
        '"""\n'
        "Lines could not be parsed as a NASA7 entry.\n"
        "Wrote YAML mechanism file to mechanism.yaml.\n"
    )

    result = parse_ck2yaml_log(log)

    assert result["refusals"] == []
    assert [event["kind"] for event in result["events"]] == [
        "redundant_species",
        "redundant_thermo",
        "comment_only_unparsable",
        "information",
    ]


def test_ck2yaml_warning_allowlist_refuses_noncomment_skipped_block() -> None:
    log = (
        "Unparsable lines while reading thermo data in thermo.dat starting on line 8:\n"
        '"""\n'
        " SPECIES H2 O2\n"
        '"""\n'
        "Lines could not be parsed as a NASA7 entry.\n"
    )

    result = parse_ck2yaml_log(log)

    assert result["refusals"] == ["unallowlisted unparsable thermo content:  SPECIES H2 O2"]


def test_cantera_validation_warning_allowlist_is_species_specific() -> None:
    allowed = [
        "NasaPoly2::validate: \nFor species OH*, discontinuity in h/RT detected at Tmid = 1000",
        "NasaPoly2::validate: \nFor species CH*, discontinuity in h/RT detected at Tmid = 1000",
    ]

    assert cantera_validation_refusals(allowed) == []
    assert cantera_validation_refusals([*allowed, "NasaPoly2::validate: For species OTHER, discontinuity"])[-1] == (
        "NasaPoly2::validate: For species OTHER, discontinuity"
    )


def test_row_timeout_is_an_explicit_checkpointed_refusal(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rows.jsonl"

    results, resumed = execute_rows(
        [_worker_row()],
        trimmed_path=str(tmp_path / "trimmed.yaml"),
        mechanism_sha256="a" * 64,
        cantera_version=_TEST_CANTERA_VERSION,
        checkpoint_path=checkpoint,
        workers=1,
        row_timeout_s=0.05,
        simulation=_fake_slow_row,
        poll_interval_s=0.005,
    )

    assert resumed == 0
    result = results["fake-row"]
    assert result["status"] == "refused"
    assert result["reason"] == "steady_state_timeout"
    assert result["wall_time_s"] >= 0.05
    assert result["integrator_steps"] is None
    persisted = json.loads(checkpoint.read_text().strip())
    assert persisted["result"]["reason"] == "steady_state_timeout"


def test_row_timeout_kills_a_process_that_ignores_terminate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        exitcode = -signal.SIGKILL

        def __init__(self, **_: Any) -> None:
            self.alive = True
            self.events: list[str] = []

        def start(self) -> None:
            self.events.append("start")

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.events.append("terminate")

        def join(self, timeout: float | None = None) -> None:
            self.events.append(f"join:{timeout}")

        def kill(self) -> None:
            self.events.append("kill")
            self.alive = False

    processes: list[Process] = []

    class Context:
        @staticmethod
        def Process(**kwargs: Any) -> Process:
            process = Process(**kwargs)
            processes.append(process)
            return process

    monkeypatch.setattr(cantera_worker, "get_context", lambda _: Context())

    results, _ = execute_rows(
        [_worker_row()],
        trimmed_path="unused.yaml",
        mechanism_sha256="a" * 64,
        cantera_version=_TEST_CANTERA_VERSION,
        checkpoint_path=tmp_path / "rows.jsonl",
        workers=1,
        row_timeout_s=1.0e-9,
        poll_interval_s=0.0,
    )

    assert results["fake-row"]["reason"] == "steady_state_timeout"
    assert processes[0].events == ["start", "terminate", "join:1.0", "kill", "join:None"]


def test_execute_rows_collects_an_in_process_child_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        exitcode = 0

        def __init__(self, *, args: tuple[Any, ...], **_: Any) -> None:
            self.args = args

        def start(self) -> None:
            simulation, trimmed_path, row, result_path = self.args
            scope, result = simulation(trimmed_path, row)
            Path(result_path).write_text(json.dumps({"scope": scope, "result": result}))

        @staticmethod
        def is_alive() -> bool:
            return False

        @staticmethod
        def join(timeout: float | None = None) -> None:
            assert timeout is None

    class Context:
        @staticmethod
        def Process(**kwargs: Any) -> Process:
            return Process(**kwargs)

    monkeypatch.setattr(cantera_worker, "get_context", lambda _: Context())

    results, resumed = execute_rows(
        [_worker_row()],
        trimmed_path="trimmed.yaml",
        mechanism_sha256="b" * 64,
        cantera_version=_TEST_CANTERA_VERSION,
        checkpoint_path=tmp_path / "rows.jsonl",
        workers=1,
        row_timeout_s=1.0,
        simulation=_fake_fast_row,
        poll_interval_s=0.0,
    )

    assert resumed == 0
    assert results["fake-row"]["mole_fractions"] == {"A": 0.25}
    assert not any((tmp_path / ".row-results").iterdir())


def test_completed_row_resumes_from_jsonl_without_rerunning(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rows.jsonl"
    arguments = {
        "trimmed_path": str(tmp_path / "trimmed.yaml"),
        "mechanism_sha256": "b" * 64,
        "cantera_version": _TEST_CANTERA_VERSION,
        "checkpoint_path": checkpoint,
        "workers": 1,
        "row_timeout_s": 1.0,
        "poll_interval_s": 0.005,
    }
    first, first_resumed = execute_rows([_worker_row()], simulation=_fake_fast_row, **arguments)
    second, second_resumed = execute_rows([_worker_row()], simulation=_must_not_run, **arguments)

    assert first_resumed == 0
    assert second_resumed == 1
    assert second == first
    assert len(checkpoint.read_text().splitlines()) == 1


def test_resume_recovers_only_an_incomplete_final_checkpoint_record(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rows.jsonl"
    completed_row = {**_worker_row(), "row_id": "completed"}
    interrupted_row = {**_worker_row(), "row_id": "interrupted"}
    completed_result = _fake_fast_row("unused", completed_row)[1]
    cantera_worker._append_checkpoint(
        checkpoint,
        "b" * 64,
        completed_row,
        completed_result,
        5.0,
        _TEST_CANTERA_VERSION,
    )
    partial = b'{"fingerprint":"interrupted'
    with checkpoint.open("ab") as stream:
        stream.write(partial)
    recoveries: list[dict[str, Any]] = []

    results, resumed = execute_rows(
        [completed_row, interrupted_row],
        trimmed_path=str(tmp_path / "trimmed.yaml"),
        mechanism_sha256="b" * 64,
        checkpoint_path=checkpoint,
        workers=1,
        row_timeout_s=5.0,
        cantera_version=_TEST_CANTERA_VERSION,
        simulation=_fake_fast_row,
        poll_interval_s=0.005,
        checkpoint_recoveries=recoveries,
    )

    assert resumed == 1
    assert results["completed"] == completed_result
    assert results["interrupted"]["status"] == "ok"
    assert recoveries == [{"kind": "recovered_truncated_checkpoint_tail", "byte_length": len(partial)}]
    assert len(checkpoint.read_text().splitlines()) == 2
    assert all(json.loads(line) for line in checkpoint.read_text().splitlines())


def test_resume_invalidates_cached_success_when_plog_validation_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "rows.jsonl"
    row = {**_worker_row(), "pressure_pa": 50 * 101325.0}
    baseline_validation = cantera_worker.plog_validation_sha256([], [])
    _, trims, refusals = trim_unsound_plog_levels(_unsound_plog(), [row["pressure_pa"]])
    changed_validation = cantera_worker.plog_validation_sha256(trims, refusals)
    common = {
        "trimmed_path": str(tmp_path / "trimmed.yaml"),
        "mechanism_sha256": "b" * 64,
        "cantera_version": _TEST_CANTERA_VERSION,
        "checkpoint_path": checkpoint,
        "workers": 1,
        "row_timeout_s": 5.0,
        "poll_interval_s": 0.005,
    }

    first, first_resumed = execute_rows(
        [row],
        simulation=_fake_fast_row,
        plog_validation_sha256=baseline_validation,
        **common,
    )
    refusal = {
        "row_id": row["row_id"],
        "status": "refused",
        "reason": "mechanism_rate_unsound_at_condition",
    }
    second, second_resumed = execute_rows(
        [row],
        simulation=_must_not_run,
        initial_results={row["row_id"]: refusal},
        plog_validation_sha256=changed_validation,
        **common,
    )

    assert first_resumed == 0
    assert first[row["row_id"]]["status"] == "ok"
    assert second_resumed == 0
    assert second[row["row_id"]]["status"] == "refused"


def test_negative_signal_during_integration_is_a_solver_crash_refusal(tmp_path: Path) -> None:
    results, resumed = execute_rows(
        [_worker_row()],
        trimmed_path=str(tmp_path / "trimmed.yaml"),
        mechanism_sha256="e" * 64,
        cantera_version=_TEST_CANTERA_VERSION,
        checkpoint_path=tmp_path / "rows.jsonl",
        workers=1,
        row_timeout_s=5.0,
        simulation=_fake_solver_crash,
        poll_interval_s=0.005,
    )

    assert resumed == 0
    result = results["fake-row"]
    assert result["status"] == "refused"
    assert result["reason"] == "solver_crash"
    assert result["signal"] == signal.SIGSEGV
    assert result["progress"]["last_reactor_time_s"] == 0.125
    assert "Fatal Python error: Segmentation fault" in Path(result["fault_log"]).read_text()


def test_negative_signal_during_construction_is_a_run_level_failure(tmp_path: Path) -> None:
    with pytest.raises(RunLevelSimulationError, match="construction"):
        execute_rows(
            [_worker_row()],
            trimmed_path=str(tmp_path / "trimmed.yaml"),
            mechanism_sha256="f" * 64,
            cantera_version=_TEST_CANTERA_VERSION,
            checkpoint_path=tmp_path / "rows.jsonl",
            workers=1,
            row_timeout_s=5.0,
            simulation=_fake_construction_crash,
            poll_interval_s=0.005,
        )


def test_resume_fingerprint_includes_complete_simulation_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = _row_fingerprint("c" * 64, _worker_row(), 600.0, _TEST_CANTERA_VERSION)

    monkeypatch.setitem(THREAD_ENV, "OMP_NUM_THREADS", "2")

    assert _row_fingerprint("c" * 64, _worker_row(), 600.0, _TEST_CANTERA_VERSION) != baseline
    assert _row_fingerprint("c" * 64, _worker_row(), 601.0, _TEST_CANTERA_VERSION) != baseline
    assert _row_fingerprint("c" * 64, _worker_row(), 600.0, "3.3.0") != baseline


def test_thread_pins_are_visible_inside_a_child_process() -> None:
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import json, os; print(json.dumps({key: os.environ[key] for key in " + repr(tuple(THREAD_ENV)) + "}))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **THREAD_ENV},
    )

    assert json.loads(completed.stdout) == THREAD_ENV


def test_simulation_lock_hash_covers_the_frozen_timeout() -> None:
    assert simulation_lock(600.0, "3.2.0")["jsr"]["rtol"] == 1.0e-9
    assert simulation_lock(600.0, "3.2.0")["simulator"] == {"cantera_version": "3.2.0"}
    assert simulation_lock_sha256(600.0, "3.2.0") == simulation_lock_sha256(600.0, "3.2.0")
    assert simulation_lock_sha256(601.0, "3.2.0") != simulation_lock_sha256(600.0, "3.2.0")
    assert simulation_lock_sha256(600.0, "3.3.0") != simulation_lock_sha256(600.0, "3.2.0")


def test_mechanism_resume_hash_ignores_only_generation_date() -> None:
    first = {"date": "first", "species": [{"name": "A"}], "reactions": []}
    second = {"date": "second", "species": [{"name": "A"}], "reactions": []}

    assert mechanism_semantic_sha256(first) == mechanism_semantic_sha256(second)
    second["species"] = [{"name": "B"}]
    assert mechanism_semantic_sha256(first) != mechanism_semantic_sha256(second)


def test_worker_progress_and_numeric_parsers_cover_refusal_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "row.progress.json"
    monkeypatch.setattr(cantera_worker, "_ROW_PROGRESS_PATH", progress)
    _write_progress("integration", last_reactor_time_s=0.25, temperature_k=700.0, integrator_steps=2)
    assert json.loads(progress.read_text()) == {
        "integrator_steps": 2,
        "last_reactor_time_s": 0.25,
        "phase": "integration",
        "temperature_k": 700.0,
    }

    monkeypatch.setattr(cantera_worker, "_ROW_PROGRESS_PATH", None)
    _write_progress("ignored")
    assert cantera_worker._pressure_pa(2) == 2.0
    assert cantera_worker._pressure_pa("+.5 atm") == 0.5 * 101325.0
    assert cantera_worker._number(3) == 3.0
    assert cantera_worker._number(" 4.5 units") == 4.5
    assert cantera_worker._number("-.5") == -0.5
    with pytest.raises(ValueError, match="unsupported PLOG pressure"):
        cantera_worker._pressure_pa("1 psi")
    with pytest.raises(ValueError, match="unsupported numeric value"):
        cantera_worker._number("not-a-number")
    assert math.isinf(cantera_worker._arrhenius({"A": 1.0, "b": 1000.0, "Ea": 0.0}, 10_000.0))
    assert cantera_worker._affected(1.0, None, 2.0)
    assert cantera_worker._affected(3.0, 2.0, None)
    assert cantera_worker._affected(2.0, 1.0, 3.0)


def test_ck2yaml_parser_refuses_malformed_and_accepts_blank_lines() -> None:
    malformed = "\nUnparsable lines while reading thermo data in thermo.dat starting on line 1:\nnot-a-fence\n"
    result = parse_ck2yaml_log(malformed)
    assert result["refusals"] == [
        "malformed ck2yaml unparsable block: "
        "Unparsable lines while reading thermo data in thermo.dat starting on line 1:",
        "unallowlisted ck2yaml output: not-a-fence",
    ]


def test_plog_rules_ignore_sound_reactions_and_refuse_removing_a_whole_reaction() -> None:
    sound = {
        "reactions": [
            {"equation": "A <=> B", "type": "elementary", "rate-constant": {"A": 1.0}},
            {
                "equation": "B <=> C",
                "type": "pressure-dependent-Arrhenius",
                "rate-constants": [{"P": 1.0, "A": 1.0, "b": 0.0, "Ea": 0.0}],
            },
        ]
    }
    unchanged, trims, refusals = trim_unsound_plog_levels(sound, [101325.0])
    assert unchanged == sound
    assert trims == refusals == []

    wholly_unsound = {
        "reactions": [
            {
                "equation": "A <=> B",
                "type": "pressure-dependent-Arrhenius",
                "rate-constants": [{"P": "1 atm", "A": -1.0, "b": 0.0, "Ea": 0.0}],
            }
        ]
    }
    with pytest.raises(ValueError, match="remove whole PLOG reaction"):
        trim_unsound_plog_levels(wholly_unsound, [])


def test_checkpoint_validation_and_initial_refusal_path(tmp_path: Path) -> None:
    row = _worker_row()
    checkpoint = tmp_path / "rows.jsonl"
    checkpoint.write_text("\nnot-json\n")
    with pytest.raises(ValueError, match="malformed row checkpoint at line 2"):
        cantera_worker._load_checkpoint(checkpoint, [row], "a" * 64, 1.0, _TEST_CANTERA_VERSION)

    with pytest.raises(ValueError, match="workers must be at least 1"):
        execute_rows(
            [row],
            trimmed_path="unused",
            mechanism_sha256="a" * 64,
            cantera_version=_TEST_CANTERA_VERSION,
            checkpoint_path=tmp_path / "unused.jsonl",
            workers=0,
            row_timeout_s=1.0,
        )
    with pytest.raises(ValueError, match="row timeout must be positive"):
        execute_rows(
            [row],
            trimmed_path="unused",
            mechanism_sha256="a" * 64,
            cantera_version=_TEST_CANTERA_VERSION,
            checkpoint_path=tmp_path / "unused.jsonl",
            workers=1,
            row_timeout_s=0.0,
        )

    checkpoint.unlink()
    result = {"row_id": row["row_id"], "status": "refused", "reason": "prevalidated"}
    results, resumed = execute_rows(
        [row],
        trimmed_path="unused",
        mechanism_sha256="a" * 64,
        cantera_version=_TEST_CANTERA_VERSION,
        checkpoint_path=checkpoint,
        workers=1,
        row_timeout_s=1.0,
        simulation=_must_not_run,
        initial_results={row["row_id"]: result},
    )
    assert resumed == 0
    assert results[row["row_id"]]["wall_time_s"] == 0.0
    assert results[row["row_id"]]["integrator_steps"] == 0

    fingerprint = cantera_worker._row_fingerprint("a" * 64, row, 1.0, _TEST_CANTERA_VERSION)
    checkpoint.write_text(
        "\n".join(
            (
                json.dumps({"fingerprint": fingerprint, "result": {"row_id": row["row_id"], "status": "ok"}}),
                json.dumps({"fingerprint": fingerprint, "result": {"row_id": row["row_id"], "status": "refused"}}),
            )
        )
    )
    with pytest.raises(ValueError, match="conflicting row checkpoint entries"):
        cantera_worker._load_checkpoint(checkpoint, [row], "a" * 64, 1.0, _TEST_CANTERA_VERSION)


def test_row_child_turns_python_exception_into_run_level_payload(tmp_path: Path) -> None:
    def explode(_: str, __: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        raise RuntimeError("boom")

    result_path = tmp_path / "result.json"
    cantera_worker._row_child(explode, "unused", _worker_row(), str(result_path))
    payload = json.loads(result_path.read_text())
    assert payload["scope"] == "run"
    assert payload["result"]["refusal"] == "simulation_worker_failed"

    success_path = tmp_path / "success.json"
    cantera_worker._row_child(_fake_fast_row, "unused", _worker_row(), str(success_path))
    assert json.loads(success_path.read_text())["scope"] == "row"


def test_stop_processes_terminates_then_kills_survivors() -> None:
    class Process:
        def __init__(self, alive: list[bool]) -> None:
            self.alive = alive
            self.events: list[str] = []

        def is_alive(self) -> bool:
            return self.alive.pop(0) if len(self.alive) > 1 else self.alive[0]

        def terminate(self) -> None:
            self.events.append("terminate")

        def join(self, timeout: float | None = None) -> None:
            self.events.append(f"join:{timeout}")

        def kill(self) -> None:
            self.events.append("kill")

    survivor = Process([True, True, False])
    stopped = Process([False, False])
    cantera_worker._stop_processes({survivor: {}, stopped: {}})
    assert survivor.events == ["terminate", "join:1.0", "kill", "join:None"]
    assert stopped.events == ["join:1.0"]


def test_worker_main_refusal_paths_are_serialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py"])
    assert cantera_worker.main() == 2

    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(__version__=_TEST_CANTERA_VERSION))
    request = {
        "mechanism_path": "mechanism.inp",
        "thermo_path": "thermo.dat",
        "converted_yaml_path": str(tmp_path / "converted.yaml"),
        "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
        "rows": [],
        "observable": "idt",
    }
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request_path), str(response_path)])
    assert cantera_worker.main() == 0
    unsupported = WorkerResponse.model_validate_json(response_path.read_text())
    assert unsupported.refusal == "unsupported_observable: idt"
    assert unsupported.converter == {}

    request.pop("observable")
    request_path.write_text(json.dumps(request))
    monkeypatch.setattr(
        cantera_worker.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="converter failed\n", stderr=""),
    )
    assert cantera_worker.main() == 0
    assert json.loads(response_path.read_text())["refusal"] == "ck2yaml_conversion_refused"


def test_cantera_worker_script_is_self_contained_without_pythonpath(tmp_path: Path) -> None:
    worker = Path(__file__).parents[1] / "carmel" / "benchmarks" / "cantera_worker.py"
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(json.dumps({"observable": "idt"}))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(worker), str(request_path), str(response_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    response = WorkerResponse.model_validate_json(response_path.read_text())
    assert response.refusal == "unsupported_observable: idt"
    tree = ast.parse(worker.read_text())
    imported_modules = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(module == "carmel" or module.startswith("carmel.") for module in imported_modules)


def test_worker_main_refuses_unsupported_plog_activation_energy_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    converted_path = tmp_path / "converted.yaml"
    request_path.write_text(
        json.dumps(
            {
                "mechanism_path": "mechanism.inp",
                "thermo_path": "thermo.dat",
                "converted_yaml_path": str(converted_path),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "rows": [],
            }
        )
    )

    def converted_ok(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        converted_path.write_text(
            "units:\n"
            "  activation-energy: eV\n"
            "reactions:\n"
            "- equation: A <=> B\n"
            "  type: pressure-dependent-Arrhenius\n"
            "  rate-constants:\n"
            "  - {P: 1 atm, A: 1.0, b: 0.0, Ea: 1.0}\n"
        )
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(__version__=_TEST_CANTERA_VERSION))
    monkeypatch.setattr(cantera_worker.subprocess, "run", converted_ok)
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request_path), str(response_path)])

    assert cantera_worker.main() == 0
    response = WorkerResponse.model_validate_json(response_path.read_text())
    assert response.refusal == "unsupported_plog_activation_energy_unit"
    assert response.validation_error == "unsupported PLOG activation-energy unit 'eV'"


@pytest.mark.parametrize("mode", ["validation", "warning", "run_level"])
def test_worker_main_rejects_after_conversion(mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    converted = tmp_path / "converted.yaml"
    request_path.write_text(
        json.dumps(
            {
                "mechanism_path": "mechanism.inp",
                "thermo_path": "thermo.dat",
                "converted_yaml_path": str(converted),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "rows": [],
            }
        )
    )

    def converted_ok(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        converted.write_text("phases: []\nspecies: []\nreactions: []\n")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    class FakeCantera:
        __version__ = "test"

        @staticmethod
        def Solution(_: Path) -> object:
            if mode == "validation":
                raise ValueError("invalid mechanism")
            if mode == "warning":
                import warnings

                warnings.warn("unexpected validation warning", stacklevel=2)
            return object()

    monkeypatch.setitem(sys.modules, "cantera", FakeCantera)
    monkeypatch.setattr(cantera_worker.subprocess, "run", converted_ok)
    if mode == "run_level":
        monkeypatch.setattr(
            cantera_worker,
            "execute_rows",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RunLevelSimulationError({"refusal": "reactor_construction_failed", "construction_error": "boom"})
            ),
        )
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request_path), str(response_path)])

    assert cantera_worker.main() == 0
    response = json.loads(response_path.read_text())
    expected = {
        "validation": "trimmed_mechanism_validation_failed",
        "warning": "unallowlisted_cantera_validation_warning",
        "run_level": "reactor_construction_failed",
    }
    assert response["refusal"] == expected[mode]


def test_worker_main_preserves_duplicate_plog_source_line_occurrences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    converted = tmp_path / "converted.yaml"
    trimmed = tmp_path / "trimmed.yaml"
    affected_pressure = 50.0 * 101325.0
    rows = [
        {**_worker_row(), "row_id": "affected", "pressure_pa": affected_pressure},
        {**_worker_row(), "row_id": "sound", "pressure_pa": 101325.0},
    ]
    request_path.write_text(
        json.dumps(
            {
                "mechanism_path": "mechanism.inp",
                "thermo_path": "thermo.dat",
                "converted_yaml_path": str(converted),
                "trimmed_yaml_path": str(trimmed),
                "checkpoint_path": str(tmp_path / "rows.jsonl"),
                "reaction_source_lines": {"A <=> B": [1628, 1700]},
                "rows": rows,
                "workers": 2,
                "row_timeout_s": 3.0,
            }
        )
    )

    def converted_ok(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        converted.write_text(
            "reactions:\n"
            "- equation: A <=> B\n"
            "  type: pressure-dependent-Arrhenius\n"
            "  rate-constants:\n"
            "  - {P: 31.6 atm, A: 1.0, b: 0.0, Ea: 0.0}\n"
            "  - {P: 100 atm, A: -1.0, b: 0.0, Ea: 0.0}\n"
            "  - {P: 100 atm, A: 0.5, b: 0.0, Ea: 0.0}\n"
            "- equation: A <=> B\n"
            "  type: pressure-dependent-Arrhenius\n"
            "  rate-constants:\n"
            "  - {P: 1 atm, A: 1.0, b: 0.0, Ea: 0.0}\n"
        )
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    fake_cantera = SimpleNamespace(__version__="test", Solution=lambda _: object())
    monkeypatch.setitem(sys.modules, "cantera", fake_cantera)
    monkeypatch.setattr(cantera_worker.subprocess, "run", converted_ok)

    def fake_execute(input_rows: list[dict[str, Any]], **kwargs: Any) -> tuple[dict[str, dict[str, Any]], int]:
        assert kwargs["initial_results"] == {
            "affected": {
                "row_id": "affected",
                "status": "refused",
                "reason": "mechanism_rate_unsound_at_condition",
            }
        }
        assert "source_line" not in trimmed.read_text()
        return {
            "affected": kwargs["initial_results"]["affected"],
            "sound": {
                "row_id": "sound",
                "status": "ok",
                "mole_fractions": {"A": 0.25},
                "wall_time_s": 0.1,
                "integrator_steps": 4,
            },
        }, 0

    monkeypatch.setattr(cantera_worker, "execute_rows", fake_execute)
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request_path), str(response_path)])

    assert cantera_worker.main() == 0
    response = json.loads(response_path.read_text())
    assert response["status"] == "ok"
    assert response["rows"][0]["reason"] == "mechanism_rate_unsound_at_condition"
    assert response["rows"][1]["mole_fractions"] == {"A": 0.25}
    assert response["plog_trims"][0]["source_line"] == 1628


def test_reaction_source_lines_are_ambiguous_when_occurrence_counts_differ() -> None:
    mechanism = {
        "reactions": [
            {"equation": "A <=> B"},
            {"equation": "A <=> B"},
        ]
    }

    cantera_worker._annotate_reaction_source_lines(mechanism, {"A <=> B": [11]})

    assert [reaction["source_line"] for reaction in mechanism["reactions"]] == [
        {"status": "ambiguous", "candidate_lines": [11]},
        {"status": "ambiguous", "candidate_lines": [11]},
    ]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("A+B=C", "A + B <=> C"),
        ("A+B<=>C", "B + A = C"),
        ("2OH=H2O2", "2 OH <=> H2O2"),
        ("2OH=H2O2", "OH+OH=H2O2"),
        ("A+B(+M)=C(+M)", "A + B (+M) <=> C (+M)"),
        ("A+M=C+M", "A + M <=> C + M"),
        ("A+B(+AR)=C(+AR)", "A + B (+AR) <=> C (+AR)"),
    ],
)
def test_canonical_reaction_key_agrees(left: str, right: str) -> None:
    assert cantera_worker.canonical_reaction_key(left) == cantera_worker.canonical_reaction_key(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("A=B", "A=>B"),
        ("A+B(+M)=C(+M)", "A+B=C"),
        ("A+B(+M)=C(+M)", "A+B(+AR)=C(+AR)"),
        ("A+M=C+M", "A=C"),
        ("a=b", "A=B"),
        ("2OH=H2O2", "OH=H2O2"),
    ],
)
def test_canonical_reaction_key_distinguishes(left: str, right: str) -> None:
    assert cantera_worker.canonical_reaction_key(left) != cantera_worker.canonical_reaction_key(right)


def test_source_line_annotation_matches_converted_yaml_forms() -> None:
    source = b"""REACTIONS
A+B=C+D 1.0 0.0 0.0
REV / 2.0 0.0 0.0 /
E+F(+M)=G(+M) 1.0 0.0 0.0
2OH=H2O2 1.0 0.0 0.0 ! REV / 1 1 1 /
X=Y 1.0 0.0 0.0
DUPLICATE
X=Y 2.0 0.0 0.0
DUPLICATE
Z=W 1.0 0.0 0.0
Z=W 1.0 0.0 0.0
U=V 1.0 0.0 0.0
END
"""
    reactions = holdout.mechanism(source)[1]
    assert holdout._rev_reaction_lines(source, {r["line"] for r in reactions}) == {2}
    keys = holdout._source_line_keys(source, reactions)
    mechanism = {
        "reactions": [
            {"equation": "A + B => C + D"},
            {"equation": "C + D => A + B"},
            {"equation": "E + F (+M) <=> G (+M)"},
            {"equation": "2 OH <=> H2O2"},
            {"equation": "X <=> Y"},
            {"equation": "X <=> Y"},
            {"equation": "U <=> V"},
            {"equation": "U <=> V"},
            {"equation": "Z <=> W"},
        ]
    }
    cantera_worker._annotate_reaction_source_lines(mechanism, keys)
    got = [reaction["source_line"] for reaction in mechanism["reactions"]]
    assert got[:4] == [2, 2, 4, 5]
    assert got[4:6] == [6, 8]
    assert got[6] == got[7] == {"status": "ambiguous", "candidate_lines": [12]}
    assert got[8] == {"status": "ambiguous", "candidate_lines": [10, 11]}


def test_simulate_row_classifies_fake_cantera_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse_solution(_: str) -> None:
        raise ValueError("bad mechanism")

    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(Solution=refuse_solution))

    scope, result = cantera_worker._simulate_row("unused.yaml", _worker_row())

    assert scope == "run"
    assert result["refusal"] == "reactor_construction_failed"
    assert result["construction_error"] == "bad mechanism"


def test_simulate_row_returns_fake_cantera_steady_state(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Gas:
        TPX: object

    class Reactor:
        mass = 2.0
        T = 700.0
        phase = SimpleNamespace(X=[0.25], species_index=lambda _: 0)

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

    class Network:
        time = 0.0

        def __init__(self, _: list[Any]) -> None:
            self.rtol = self.atol = self.max_steps = None
            self.solver_stats = {"steps": 0}

        def advance_to_steady_state(self, *, max_steps: int) -> None:
            captured["max_steps"] = max_steps
            self.time = 0.5
            self.solver_stats["steps"] = 7

    def pressure_controller(*_: Any, **kwargs: Any) -> object:
        captured["pressure_K"] = kwargs["K"]
        return object()

    fake_cantera = SimpleNamespace(
        Solution=lambda _: Gas(),
        Reservoir=lambda *_args, **_kwargs: object(),
        IdealGasReactor=Reactor,
        MassFlowController=lambda *_args, **_kwargs: object(),
        PressureController=pressure_controller,
        ReactorNet=Network,
    )
    monkeypatch.setitem(sys.modules, "cantera", fake_cantera)

    scope, result = cantera_worker._simulate_row("unused.yaml", _worker_row())

    assert scope == "row"
    assert result["status"] == "ok"
    assert result["mole_fractions"] == {"A": 0.25}
    assert result["integrator_steps"] == 7
    assert captured == {"max_steps": 100000, "pressure_K": 1.0e-5}


def test_simulate_row_classifies_python_integrator_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gas:
        TPX: object

    class Reactor:
        mass = 1.0
        T = 700.0
        phase = SimpleNamespace(X=[0.5], species_index=lambda _: 0)

        def __init__(self, *_: Any, **__: Any) -> None:
            pass

    class Network:
        time = 0.1
        solver_stats = {"steps": 4}

        def __init__(self, _: list[Any]) -> None:
            self.rtol = self.atol = self.max_steps = None

        def advance_to_steady_state(self, **_: Any) -> None:
            raise RuntimeError("integrator refused")

    fake_cantera = SimpleNamespace(
        Solution=lambda _: Gas(),
        Reservoir=lambda *_args, **_kwargs: object(),
        IdealGasReactor=Reactor,
        MassFlowController=lambda *_args, **_kwargs: object(),
        PressureController=lambda *_args, **_kwargs: object(),
        ReactorNet=Network,
    )
    monkeypatch.setitem(sys.modules, "cantera", fake_cantera)

    scope, result = cantera_worker._simulate_row("unused.yaml", _worker_row())

    assert scope == "row"
    assert result["status"] == "refused"
    assert result["reason"] == "solver_failure: integrator refused"
    assert result["integrator_steps"] == 4


def test_structure_mapping_helpers_and_thermo_duplicate_audit() -> None:
    assert holdout.observable_smiles("O2") == holdout.canonical("O=O")
    assert holdout.observable_smiles("unknown") is None
    assert holdout._cyclic_ether("C5H10O1-4") is not None
    assert holdout._cyclic_ether("C5H10O1-1") is None
    from_inchi = SimpleNamespace(
        smiles=holdout.Absent(reason="not_reported_here"),
        inchi=SimpleNamespace(raw="InChI=1S/CO2/c2-1-3"),
    )
    assert holdout._record_smiles(from_inchi) == "O=C=O"
    absent = SimpleNamespace(
        smiles=holdout.Absent(reason="not_reported_here"),
        inchi=holdout.Absent(reason="not_reported_here"),
    )
    assert holdout._record_smiles(absent) is None

    mapped, refused = holdout._structure_mapping({"O2", "unknown"}, ["O2"])
    assert mapped == {"O2": "O2"}
    assert refused == ["unknown"]
    with pytest.raises(ValueError, match="conflicting structures"):
        holdout._structure_mapping(
            {"O2"},
            ["O2"],
            observed_structures={"O2": holdout.canonical("C")},
        )

    thermo = (
        b"A                 header first                             1\n"
        b"first coefficients                                             2\n"
        b"first coefficients                                             3\n"
        b"first coefficients                                             4\n"
        b"A                 header later                             1\n"
        b"later coefficients                                             2\n"
        b"later coefficients                                             3\n"
        b"later coefficients                                             4\n"
    )
    duplicates = holdout._thermo_duplicates(thermo)
    assert duplicates[0]["species"] == "A"
    assert duplicates[0]["first_used"] is True
    assert duplicates[0]["coefficients_differ"] is True


def test_load_rows_uses_verified_holdout_and_existing_respecth_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    member = b"record"
    archive_path = tmp_path / "split.zip"
    import zipfile

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("holdout.xml", member)

    class ArchivePin:
        def verify(self, _: bytes) -> None:
            return None

        def to_pinned_archive(self) -> object:
            return object()

    identity = SimpleNamespace(display="CO2", smiles=SimpleNamespace(raw="O=C=O"), inchi=None)
    point = SimpleNamespace(
        point_id="p1",
        coordinates=(SimpleNamespace(axis_id="temperature", value=_Quantity(700.0)),),
        observations=(SimpleNamespace(axis_id="product", value=_Quantity(0.01)),),
    )
    series = SimpleNamespace(
        constants=(
            SimpleNamespace(axis_id="pressure", value=_Quantity(101325.0)),
            SimpleNamespace(axis_id="residence_time", value=_Quantity(2.0)),
            SimpleNamespace(axis_id="volume", value=_Quantity(8.5e-5)),
        ),
        points=(point,),
    )
    record = SimpleNamespace(
        envelope=SimpleNamespace(
            series=(series,),
            composition=SimpleNamespace(components=(SimpleNamespace(species_raw_name="O2", amount=_Quantity(1.0)),)),
        ),
        species_columns=(SimpleNamespace(axis_id="product", species=identity),),
    )
    monkeypatch.setattr(holdout, "parse_series_record", lambda *_: record)
    item = SimpleNamespace(path="holdout.xml", sha256=hashlib.sha256(member).hexdigest())
    contract = SimpleNamespace(
        split=SimpleNamespace(
            archive=ArchivePin(),
            studies=(
                SimpleNamespace(set="train", files=(item,)),
                SimpleNamespace(set="holdout", files=(item,)),
            ),
        )
    )

    rows = holdout._load_rows(contract, archive_path)

    assert rows == [
        {
            "row_id": "holdout.xml:p1",
            "record": "holdout.xml",
            "temperature_k": 700.0,
            "pressure_pa": 101325.0,
            "residence_time_s": 2.0,
            "volume_m3": 8.5e-5,
            "inlet": {"O2": 1.0},
            "observations": {"CO2": 0.01},
            "record_structures": {"CO2": holdout.canonical("O=C=O")},
        }
    ]

    changed_item = SimpleNamespace(path="holdout.xml", sha256="0" * 64)
    changed_contract = SimpleNamespace(
        split=SimpleNamespace(
            archive=ArchivePin(),
            studies=(SimpleNamespace(set="holdout", files=(changed_item,)),),
        )
    )
    with pytest.raises(ValueError, match="SPLIT BLOB CHANGED"):
        holdout._load_rows(changed_contract, archive_path)

    missing_composition = SimpleNamespace(
        envelope=SimpleNamespace(series=(series,), composition=holdout.Absent(reason="not_reported_here")),
        species_columns=(SimpleNamespace(axis_id="product", species=identity),),
    )
    monkeypatch.setattr(holdout, "parse_series_record", lambda *_: missing_composition)
    missing_composition_contract = SimpleNamespace(
        split=SimpleNamespace(
            archive=ArchivePin(),
            studies=(SimpleNamespace(set="holdout", files=(item,)),),
        )
    )
    with pytest.raises(ValueError, match="has no inlet composition"):
        holdout._load_rows(missing_composition_contract, archive_path)


def test_separation_refuses_different_locks_and_names_all_temperature_bands() -> None:
    with pytest.raises(ValueError, match="different simulation locks"):
        build_separation_report(
            {"simulation_lock_sha256": simulation_lock_sha256(600.0, "3.2.0")},
            {"simulation_lock_sha256": simulation_lock_sha256(600.0, "3.3.0")},
        )
    assert holdout._temperature_band(500.0) == "below_600_k"
    assert holdout._temperature_band(700.0) == "600_900_k"
    assert holdout._temperature_band(1000.0) == "above_900_k"
    assert holdout._refusal_class(None) == "unspecified"


def test_cantera_boundary_and_cli_refuse_transport_and_gate_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = WorkerRequest(
        mechanism_path=tmp_path / "mechanism.inp",
        thermo_path=tmp_path / "thermo.dat",
        converted_yaml_path=run_dir / "converted.yaml",
        trimmed_yaml_path=run_dir / "trimmed.yaml",
        checkpoint_path=run_dir / "rows.jsonl",
        reaction_source_lines={},
        rows=(),
    )
    monkeypatch.setattr(
        "carmel.benchmarks.holdout.subprocess.run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 3)
    )
    with pytest.raises(ValueError, match="exit code 3"):
        invoke_cantera_worker(request, run_dir=run_dir, cantera_python=Path("fake"))

    output = tmp_path / "report.json"
    monkeypatch.setattr(
        "carmel.benchmarks.holdout.run_harness",
        lambda **_: {"convergence_gate": {"passed": False}},
    )
    status = main(
        [
            "score-holdout",
            "--contract",
            "contract.json",
            "--mechanism",
            "mechanism.inp",
            "--mechanism-key",
            "v_old",
            "--thermo",
            "thermo.dat",
            "--thermo-url",
            "https://example.test/thermo.dat",
            "--thermo-retrieval-date",
            "2026-10-17",
            "--thermo-sha256",
            "0" * 64,
            "--archive",
            "split.zip",
            "--run-dir",
            str(run_dir),
            "--output",
            str(output),
        ]
    )
    assert status == 1


def test_secondary_metric_clamps_and_balances_species() -> None:
    rows = [
        {
            "row_id": "a-low",
            "species": "A",
            "temperature_k": 700.0,
            "experimental": 1.0e-4,
            "simulated": 1.0e-5,
            "detection_limit": None,
        },
        {
            "row_id": "a-high",
            "species": "A",
            "temperature_k": 1000.0,
            "experimental": 1.0e-3,
            "simulated": 1.0e-2,
            "detection_limit": None,
        },
        {
            "row_id": "b-low",
            "species": "B",
            "temperature_k": 800.0,
            "experimental": 0.0,
            "simulated": 1.0e-9,
            "detection_limit": 1.0e-6,
        },
    ]

    result = score_metric(rows)

    assert result["all"]["per_species"]["A"] == {
        "score": pytest.approx((0.3010299956639812 + 1.0) / 2),
        "scored_rows": 2,
        "detection_limit": 5.0e-5,
    }
    assert result["all"]["per_species"]["B"]["score"] == 0.0
    assert result["all"]["overall"] == pytest.approx((0.6505149978319906 + 0.0) / 2)
    assert result["low_temperature_600_900_k"]["per_species"]["A"]["scored_rows"] == 1
    assert result["low_temperature_600_900_k"]["overall"] == pytest.approx(0.3010299956639812 / 2)


def test_secondary_metric_reports_every_refusal_class() -> None:
    rows = [
        {
            "row_id": "no-positive",
            "species": "NO_LIMIT",
            "temperature_k": 700.0,
            "experimental": 0.0,
            "simulated": 0.0,
            "detection_limit": None,
        },
        {
            "row_id": "negative-exp",
            "species": "REFUSALS",
            "temperature_k": 700.0,
            "experimental": -1.0,
            "simulated": 1.0,
            "detection_limit": 1.0e-6,
        },
        {
            "row_id": "missing-sim",
            "species": "REFUSALS",
            "temperature_k": 700.0,
            "experimental": 1.0,
            "simulated": None,
            "detection_limit": 1.0e-6,
        },
        {
            "row_id": "negative-sim",
            "species": "REFUSALS",
            "temperature_k": 700.0,
            "experimental": 1.0,
            "simulated": -1.0,
            "detection_limit": 1.0e-6,
        },
        {
            "row_id": "solver",
            "species": "REFUSALS",
            "temperature_k": 700.0,
            "experimental": 1.0,
            "refusal": "solver_failure: failed",
            "detection_limit": 1.0e-6,
        },
        {
            "row_id": "nonfinite",
            "species": "REFUSALS",
            "temperature_k": 700.0,
            "experimental": 1.0,
            "simulated": float("nan"),
            "detection_limit": 1.0e-6,
        },
        {
            "row_id": "nonfinite-exp",
            "species": "REFUSALS",
            "temperature_k": 700.0,
            "experimental": float("nan"),
            "simulated": 1.0,
            "detection_limit": 1.0e-6,
        },
    ]

    result = score_metric(rows)

    assert {item["reason"] for item in result["all"]["refusals"]} == {
        "no_detection_limit_or_positive_observation",
        "negative_experimental",
        "missing_or_nonfinite_simulated",
        "negative_simulated",
        "missing_or_nonfinite_experimental",
        "solver_failure: failed",
    }


def test_secondary_metric_refuses_a_nonpositive_stated_detection_limit() -> None:
    result = score_metric(
        [
            {
                "row_id": "bad-limit",
                "species": "A",
                "temperature_k": 700.0,
                "experimental": 1.0,
                "simulated": 1.0,
                "detection_limit": 0.0,
            }
        ]
    )

    assert result["all"]["overall"] is None
    assert result["all"]["refusals"] == [{"species": "A", "reason": "invalid_detection_limit"}]


def test_separation_uses_converged_intersection_and_bands_refusals() -> None:
    def metric_row(
        row_id: str, temperature: float, simulated: float | None, refusal: str | None = None
    ) -> dict[str, Any]:
        row = {
            "row_id": row_id,
            "species": "A",
            "temperature_k": temperature,
            "experimental": 0.1,
            "simulated": simulated,
            "detection_limit": 1.0e-6,
        }
        if refusal is not None:
            row["refusal"] = refusal
        return row

    old_metric_rows = [
        metric_row("a", 700.0, 0.2),
        metric_row("b", 800.0, 0.4),
        metric_row("c", 500.0, None, "steady_state_timeout"),
    ]
    new_metric_rows = [
        metric_row("a", 700.0, None, "solver_crash"),
        metric_row("b", 800.0, 0.05),
        metric_row("c", 500.0, 0.1),
    ]
    old = {
        "simulation_lock_sha256": "a" * 64,
        "holdout_fingerprint": {"sha256": "b" * 64},
        "row_results": [
            {"row_id": "a", "temperature_k": 700.0, "status": "ok", "reason": None},
            {"row_id": "b", "temperature_k": 800.0, "status": "ok", "reason": None},
            {"row_id": "c", "temperature_k": 500.0, "status": "refused", "reason": "steady_state_timeout"},
        ],
        "metric_rows": old_metric_rows,
        "metric": score_metric(old_metric_rows),
    }
    new = {
        "simulation_lock_sha256": "a" * 64,
        "holdout_fingerprint": {"sha256": "b" * 64},
        "row_results": [
            {"row_id": "a", "temperature_k": 700.0, "status": "refused", "reason": "solver_crash"},
            {"row_id": "b", "temperature_k": 800.0, "status": "ok", "reason": None},
            {"row_id": "c", "temperature_k": 500.0, "status": "ok", "reason": None},
        ],
        "metric_rows": new_metric_rows,
        "metric": score_metric(new_metric_rows),
    }

    result = build_separation_report(old, new)

    assert result["primary"]["row_ids"] == ["b"]
    assert result["primary"]["v_old"]["all"]["overall"] == pytest.approx(math.log10(4.0))
    assert result["primary"]["v_new"]["all"]["overall"] == pytest.approx(math.log10(2.0))
    assert result["refusals"]["v_old"]["by_temperature_band"] == {"below_600_k": {"steady_state_timeout": 1}}
    assert result["refusals"]["v_new"]["by_temperature_band"] == {"600_900_k": {"solver_crash": 1}}


def test_separation_keeps_frozen_limit_when_smallest_observation_is_excluded() -> None:
    def metric_row(row_id: str, experimental: float, simulated: float) -> dict[str, Any]:
        return {
            "row_id": row_id,
            "species": "A",
            "temperature_k": 700.0,
            "experimental": experimental,
            "simulated": simulated,
            "detection_limit": None,
        }

    metric_rows = [
        metric_row("smallest", 0.001, 0.001),
        metric_row("shared", 0.1, 0.001),
    ]
    common = {
        "simulation_lock_sha256": "a" * 64,
        "holdout_fingerprint": {"sha256": "b" * 64},
        "metric_rows": metric_rows,
        "metric": score_metric(metric_rows),
    }
    old = {
        **common,
        "row_results": [
            {"row_id": "smallest", "temperature_k": 700.0, "status": "ok", "reason": None},
            {"row_id": "shared", "temperature_k": 700.0, "status": "ok", "reason": None},
        ],
    }
    new = {
        **common,
        "row_results": [
            {"row_id": "smallest", "temperature_k": 700.0, "status": "refused", "reason": "solver_crash"},
            {"row_id": "shared", "temperature_k": 700.0, "status": "ok", "reason": None},
        ],
    }

    result = build_separation_report(old, new)

    for mechanism in ("v_old", "v_new"):
        species = result["primary"][mechanism]["all"]["per_species"]["A"]
        assert species["detection_limit"] == 0.0005
        assert species["score"] == pytest.approx(2.0)


def test_separation_scores_only_the_shared_valid_observable_intersection() -> None:
    def metric_row(species: str, simulated: float | None, refusal: str | None = None) -> dict[str, Any]:
        row = {
            "row_id": "shared",
            "species": species,
            "temperature_k": 700.0,
            "experimental": 0.1,
            "simulated": simulated,
            "detection_limit": 1.0e-6,
        }
        if refusal is not None:
            row["refusal"] = refusal
        return row

    old_rows = [metric_row("A", 0.1), metric_row("B", 1.0)]
    new_rows = [metric_row("A", 0.2), metric_row("B", None, "unmapped_species")]
    common = {
        "simulation_lock_sha256": "a" * 64,
        "holdout_fingerprint": {"sha256": "b" * 64},
        "row_results": [{"row_id": "shared", "temperature_k": 700.0, "status": "ok", "reason": None}],
    }

    result = build_separation_report(
        {**common, "metric_rows": old_rows, "metric": score_metric(old_rows)},
        {**common, "metric_rows": new_rows, "metric": score_metric(new_rows)},
    )

    assert result["primary"]["v_old"]["all"]["overall"] == 0.0
    assert result["primary"]["v_new"]["all"]["overall"] == pytest.approx(math.log10(2.0))
    assert set(result["primary"]["v_old"]["all"]["per_species"]) == {"A"}
    assert set(result["primary"]["v_new"]["all"]["per_species"]) == {"A"}
    assert result["primary"]["excluded_observables"] == {
        "v_old": {"total": 0, "by_species": {}},
        "v_new": {"total": 1, "by_species": {"B": 1}},
    }


def test_separation_refuses_changed_holdout_with_the_same_rows_and_lock() -> None:
    metric_row = {
        "row_id": "shared",
        "species": "A",
        "temperature_k": 700.0,
        "experimental": 0.1,
        "simulated": 0.1,
        "detection_limit": 1.0e-6,
    }
    frozen_row = {
        "row_id": "shared",
        "temperature_k": 700.0,
        "pressure_pa": 101325.0,
        "residence_time_s": 2.0,
        "volume_m3": 8.5e-5,
        "inlet": {"FUEL": 1.0},
        "observations": {"A": 0.1},
    }
    contract = SimpleNamespace(
        split=SimpleNamespace(
            archive=SimpleNamespace(sha256="f" * 64),
            model_dump=lambda **_: {"archive": {"sha256": "f" * 64}, "studies": []},
        )
    )
    old_fingerprint = holdout._frozen_holdout_fingerprint(b"contract", contract, [frozen_row])
    new_fingerprint = holdout._frozen_holdout_fingerprint(
        b"contract",
        contract,
        [{**frozen_row, "observations": {"A": 0.2}}],
    )
    common = {
        "simulation_lock_sha256": "a" * 64,
        "row_results": [{"row_id": "shared", "temperature_k": 700.0, "status": "ok", "reason": None}],
        "metric_rows": [metric_row],
        "metric": score_metric([metric_row]),
    }

    with pytest.raises(ValueError, match="different frozen holdout inputs"):
        build_separation_report(
            {**common, "holdout_fingerprint": old_fingerprint},
            {**common, "holdout_fingerprint": new_fingerprint},
        )


def test_cantera_subprocess_boundary_accepts_a_recorded_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = WorkerRequest(
        mechanism_path=tmp_path / "mechanism.inp",
        thermo_path=tmp_path / "thermo.dat",
        converted_yaml_path=run_dir / "converted.yaml",
        trimmed_yaml_path=run_dir / "trimmed.yaml",
        checkpoint_path=run_dir / "rows.jsonl",
        reaction_source_lines={},
        rows=(
            WorkerRequestRow(
                row_id="recorded",
                temperature_k=700.0,
                pressure_pa=101325.0,
                residence_time_s=2.0,
                volume_m3=8.5e-5,
                composition={"O2": 1.0},
                output_species=("O2",),
            ),
        ),
    )

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert {key: kwargs["env"][key] for key in THREAD_ENV} == THREAD_ENV
        recorded_request = json.loads(Path(command[-2]).read_text())
        assert recorded_request["rows"][0]["row_id"] == "recorded"
        Path(command[-1]).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ok",
                    "rows": [
                        {
                            "row_id": "recorded",
                            "status": "ok",
                            "mole_fractions": {"O2": 0.75},
                            "wall_time_s": 0.1,
                            "integrator_steps": 5,
                        }
                    ],
                    "converter": {"returncode": 0, "events": [], "refusals": [], "log": "recorded"},
                    "trimmed_yaml_sha256": "0" * 64,
                    "mechanism_semantic_sha256": "1" * 64,
                    "cantera_version": "recorded",
                    "thread_env": THREAD_ENV,
                    "row_process_start_method": "spawn",
                    "load_average_at_start": [1.0, 2.0, 3.0],
                    "simulation_lock": simulation_lock(600.0, _TEST_CANTERA_VERSION),
                    "simulation_lock_sha256": simulation_lock_sha256(600.0, _TEST_CANTERA_VERSION),
                }
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("carmel.benchmarks.holdout.subprocess.run", fake_run)

    response = invoke_cantera_worker(request, run_dir=run_dir, cantera_python=Path("fake-python"))

    assert response.rows[0].mole_fractions == {"O2": 0.75}
    assert response.thread_env == THREAD_ENV
    assert (run_dir / "stdout.log").is_file()
    assert (run_dir / "stderr.log").is_file()


@pytest.mark.parametrize(
    ("diagnostics"),
    [
        {"wall_time_s": 0.25, "signal": None, "fault_log": None},
        {"wall_time_s": None, "signal": 11, "fault_log": "/tmp/construction.fault.log"},
    ],
    ids=("in-row-construction-failure", "construction-crash"),
)
def test_cantera_subprocess_boundary_accepts_construction_refusal_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, diagnostics: dict[str, Any]
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = WorkerRequest(
        mechanism_path=tmp_path / "mechanism.inp",
        thermo_path=tmp_path / "thermo.dat",
        converted_yaml_path=run_dir / "converted.yaml",
        trimmed_yaml_path=run_dir / "trimmed.yaml",
        checkpoint_path=run_dir / "rows.jsonl",
        reaction_source_lines={},
        rows=(),
    )

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        response = {
            "schema_version": 1,
            "status": "refused",
            "rows": [],
            "converter": {},
            "refusal": "reactor_construction_failed",
            "construction_error": "reactor construction failed",
            "thread_env": THREAD_ENV,
            "row_process_start_method": "spawn",
            "load_average_at_start": [1.0, 2.0, 3.0],
            "simulation_lock": simulation_lock(600.0, _TEST_CANTERA_VERSION),
            "simulation_lock_sha256": simulation_lock_sha256(600.0, _TEST_CANTERA_VERSION),
            **diagnostics,
        }
        Path(command[-1]).write_text(json.dumps(response))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("carmel.benchmarks.holdout.subprocess.run", fake_run)

    response = invoke_cantera_worker(request, run_dir=run_dir, cantera_python=Path("fake-python"))

    assert response.wall_time_s == diagnostics["wall_time_s"]
    assert response.signal == diagnostics["signal"]
    assert response.fault_log == diagnostics["fault_log"]


def _fake_interpreter(directory: Path, name: str = "python") -> Path:
    path = directory / name
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_resolve_cantera_python_flag_wins_over_t3_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    flag = _fake_interpreter(tmp_path, "flag-python")
    monkeypatch.setenv("T3_PYTHON", str(_fake_interpreter(tmp_path, "env-python")))

    assert holdout.resolve_cantera_python(flag) == flag


def test_resolve_cantera_python_uses_t3_python_without_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_python = _fake_interpreter(tmp_path)
    monkeypatch.setenv("T3_PYTHON", str(env_python))

    assert holdout.resolve_cantera_python(None) == env_python


@pytest.mark.parametrize("value", [None, ""])
def test_resolve_cantera_python_refuses_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv("T3_PYTHON", raising=False)
    else:
        monkeypatch.setenv("T3_PYTHON", value)

    with pytest.raises(ValueError, match=r"--cantera-python.*T3_PYTHON.*Cantera"):
        holdout.resolve_cantera_python(None)


def test_resolve_cantera_python_names_a_bad_path_and_its_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing-python"
    not_executable = tmp_path / "plain-file"
    not_executable.write_text("")
    monkeypatch.setenv("T3_PYTHON", str(missing))

    with pytest.raises(ValueError, match=r"\$T3_PYTHON.*missing-python"):
        holdout.resolve_cantera_python(None)
    with pytest.raises(ValueError, match=r"--cantera-python.*plain-file"):
        holdout.resolve_cantera_python(not_executable)


def test_score_holdout_cli_refuses_without_an_interpreter_and_launches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("T3_PYTHON", raising=False)

    def must_not_run(**_: Any) -> dict[str, Any]:
        raise AssertionError("run_harness must not be invoked before the interpreter resolves")

    monkeypatch.setattr("carmel.benchmarks.holdout.run_harness", must_not_run)
    run_dir = tmp_path / "run"

    status = main(
        [
            "score-holdout",
            "--contract",
            "contract.json",
            "--mechanism",
            "mechanism.inp",
            "--mechanism-key",
            "v_old",
            "--thermo",
            "thermo.dat",
            "--thermo-url",
            "https://example.test/thermo.dat",
            "--thermo-retrieval-date",
            "2026-10-17",
            "--thermo-sha256",
            "0" * 64,
            "--archive",
            "split.zip",
            "--run-dir",
            str(run_dir),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )

    err = capsys.readouterr().err
    assert status == 1
    assert "REFUSED:" in err
    assert "--cantera-python" in err
    assert "T3_PYTHON" in err
    assert not run_dir.exists()
    assert not (tmp_path / "report.json").exists()


def test_score_holdout_cli_writes_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "result" / "report.json"
    atomic_writes: list[Path] = []

    def atomic_write(path: Path, content: str) -> None:
        atomic_writes.append(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    monkeypatch.setattr("carmel.services.artifacts.write_text", atomic_write)
    monkeypatch.setattr("carmel.benchmarks.holdout.run_harness", lambda **_: {"schema_version": 1, "metric": {}})
    monkeypatch.setenv("T3_PYTHON", str(_fake_interpreter(tmp_path)))

    status = main(
        [
            "score-holdout",
            "--contract",
            "contract.json",
            "--mechanism",
            "mechanism.inp",
            "--mechanism-key",
            "v_old",
            "--thermo",
            "thermo.dat",
            "--thermo-url",
            "https://example.test/thermo.dat",
            "--thermo-retrieval-date",
            "2026-10-17",
            "--thermo-sha256",
            "0" * 64,
            "--archive",
            "split.zip",
            "--run-dir",
            str(tmp_path / "run"),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert atomic_writes == [output]
    assert json.loads(output.read_text()) == {"schema_version": 1, "metric": {}}


def test_run_harness_builds_a_telemetry_rich_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract_path = tmp_path / "contract.json"
    mechanism_path = tmp_path / "mechanism.inp"
    thermo_path = tmp_path / "thermo.dat"
    archive_path = tmp_path / "split.zip"
    contract_path.write_bytes(b'{"frozen":true}\n')
    mechanism_path.write_bytes(b"mechanism\n")
    thermo_path.write_bytes(b"thermo\n")
    archive_path.write_bytes(b"archive\n")

    class Pin:
        def verify(self, _: bytes) -> None:
            return None

    contract = SimpleNamespace(
        case="synthetic-jsr",
        mechanisms={"v_old": Pin()},
        split=SimpleNamespace(
            archive=SimpleNamespace(sha256="f" * 64),
            model_dump=lambda **_: {"archive": {"sha256": "f" * 64}, "studies": []},
        ),
        old_species=(
            SimpleNamespace(label="FUEL", smiles="C"),
            SimpleNamespace(label="IGNORED", smiles=None),
        ),
        new_species=(SimpleNamespace(label="PRODUCT", smiles="O"),),
    )
    base_row = {
        "record": "synthetic.xml",
        "temperature_k": 700.0,
        "pressure_pa": 101325.0,
        "residence_time_s": 2.0,
        "volume_m3": 8.5e-5,
        "inlet": {"FUEL": 1.0},
        "observations": {"PRODUCT": 0.1},
        "record_structures": {},
    }
    rows = [
        {**base_row, "row_id": "synthetic:p1"},
        {
            **base_row,
            "row_id": "synthetic:p2",
            "inlet": {"UNMAPPABLE": 1.0},
            "observations": {"PRODUCT": 0.1, "UNMAPPED_PRODUCT": 0.01},
        },
        {**base_row, "row_id": "synthetic:p3"},
        {**base_row, "row_id": "synthetic:p4"},
    ]
    monkeypatch.setattr("carmel.benchmarks.holdout.load_contract", lambda _: contract)
    monkeypatch.setattr("carmel.benchmarks.holdout._load_rows", lambda *_: rows)
    monkeypatch.setattr(
        "carmel.benchmarks.holdout.mechanism",
        lambda _: (
            ["FUEL", "PRODUCT"],
            [
                {"equation": "FUEL=PRODUCT", "line": 17},
                {"equation": "FUEL=>WASTE", "line": 18},
            ],
        ),
    )

    def fake_worker(request: WorkerRequest, **_: Any) -> WorkerResponse:
        assert request.row_timeout_s == 12.0
        assert request.workers == 3
        assert request.observable == "jsr"
        assert request.rows[0].composition == {"FUEL": 1.0}
        assert [row.row_id for row in request.rows] == ["synthetic:p1", "synthetic:p3", "synthetic:p4"]
        assert request.reaction_source_lines == {
            "FUEL <=> PRODUCT": (17,),
            "FUEL => WASTE": (18,),
        }
        return WorkerResponse.model_validate(
            {
                "schema_version": 1,
                "status": "ok",
                "rows": [
                    {
                        "row_id": "synthetic:p1",
                        "status": "ok",
                        "mole_fractions": {"PRODUCT": 0.2},
                        "wall_time_s": 0.25,
                        "integrator_steps": 42,
                    },
                    {
                        "row_id": "synthetic:p4",
                        "status": "refused",
                        "wall_time_s": 0.25,
                        "integrator_steps": 42,
                    },
                ],
                "converter": {"events": [], "refusals": [], "returncode": 0, "log": ""},
                "trimmed_yaml_sha256": "d" * 64,
                "mechanism_semantic_sha256": "e" * 64,
                "cantera_version": "3.2.0",
                "row_workers": 1,
                "configured_workers": 3,
                "row_timeout_s": 12.0,
                "checkpoint_path": str(tmp_path / "run" / "rows.jsonl"),
                "checkpoint_recoveries": [{"kind": "recovered_truncated_checkpoint_tail", "byte_length": 23}],
                "thread_env": THREAD_ENV,
                "row_process_start_method": "spawn",
                "load_average_at_start": [1.0, 2.0, 3.0],
                "simulation_lock": simulation_lock(12.0, _TEST_CANTERA_VERSION),
                "simulation_lock_sha256": simulation_lock_sha256(12.0, _TEST_CANTERA_VERSION),
            }
        )

    monkeypatch.setattr("carmel.benchmarks.holdout.invoke_cantera_worker", fake_worker)

    report = run_harness(
        contract_path=contract_path,
        mechanism_path=mechanism_path,
        mechanism_key="v_old",
        thermo_path=thermo_path,
        thermo_url="https://example.test/thermo.dat",
        thermo_retrieval_date="2026-10-18",
        thermo_sha256=hashlib.sha256(b"thermo\n").hexdigest(),
        archive_path=archive_path,
        run_dir=tmp_path / "run",
        cantera_python=Path("fake-python"),
        row_timeout_s=12.0,
        workers=3,
    )

    assert report["convergence_gate"]["passed"] is False
    assert report["row_results"][0]["status"] == "ok"
    assert report["row_results"][1] == {
        "row_id": "synthetic:p2",
        "temperature_k": 700.0,
        "status": "refused",
        "reason": "unmapped_inlet_species",
        "wall_time_s": None,
        "integrator_steps": None,
        "signal": None,
        "fault_log": None,
        "progress": None,
    }
    assert report["row_results"][2]["reason"] == "unsupported_reactor_condition_or_solver_failure"
    assert report["row_results"][3]["reason"] == "unsupported_reactor_condition_or_solver_failure"
    assert report["thread_env"] == THREAD_ENV
    assert report["checkpoint_recoveries"] == ({"kind": "recovered_truncated_checkpoint_tail", "byte_length": 23},)
    assert report["holdout_fingerprint"]["contract_sha256"] == hashlib.sha256(b'{"frozen":true}\n').hexdigest()
    assert len(report["holdout_fingerprint"]["sha256"]) == 64
    assert report["metric"]["all"]["overall"] == pytest.approx(abs(math.log10(2.0)))
    assert any(
        item.get("species") == "UNMAPPED_PRODUCT" and item.get("refusal") == "unmapped_species"
        for item in report["metric_rows"]
    )

    with pytest.raises(ValueError, match="thermo bytes differ"):
        run_harness(
            contract_path=contract_path,
            mechanism_path=mechanism_path,
            mechanism_key="v_old",
            thermo_path=thermo_path,
            thermo_url="https://example.test/thermo.dat",
            thermo_retrieval_date="2026-10-18",
            thermo_sha256="0" * 64,
            archive_path=archive_path,
            run_dir=tmp_path / "bad-pin-run",
            cantera_python=Path("fake-python"),
        )

    def refused_worker(request: WorkerRequest, **kwargs: Any) -> WorkerResponse:
        refused = fake_worker(request, **kwargs).model_dump(mode="json")
        refused.update(
            status="refused",
            rows=[],
            refusal="trimmed_mechanism_validation_failed",
            validation_error="bad mechanism",
        )
        return WorkerResponse.model_validate(refused)

    monkeypatch.setattr("carmel.benchmarks.holdout.invoke_cantera_worker", refused_worker)
    with pytest.raises(ValueError, match="bad mechanism"):
        run_harness(
            contract_path=contract_path,
            mechanism_path=mechanism_path,
            mechanism_key="v_old",
            thermo_path=thermo_path,
            thermo_url="https://example.test/thermo.dat",
            thermo_retrieval_date="2026-10-18",
            thermo_sha256=hashlib.sha256(b"thermo\n").hexdigest(),
            archive_path=archive_path,
            run_dir=tmp_path / "refused-run",
            cantera_python=Path("fake-python"),
            row_timeout_s=12.0,
            workers=3,
        )


@pytest.mark.skipif(
    not all(path.is_file() for path in (_CANTERA_PYTHON, _OLD_MECHANISM, _OLD_THERMO)),
    reason="configured Cantera interpreter and pinned CHEMKIN inputs are unavailable",
)
def test_cantera_worker_round_trip_builds_a_real_reactor(tmp_path: Path) -> None:
    response = _run_real_cantera_worker(tmp_path, volume_m3=8.5e-5)

    assert response["status"] == "ok", response
    assert response["rows"][0]["row_id"] == "real-reactor"
    assert response["rows"][0]["status"] == "ok", response
    assert 0.0 <= response["rows"][0]["mole_fractions"]["CO2"] <= 1.0
    assert response["rows"][0]["wall_time_s"] > 0.0
    assert response["rows"][0]["integrator_steps"] > 0
    assert response["row_timeout_s"] == 600.0
    assert response["configured_workers"] == max(1, (os.cpu_count() or 1) - 2)
    assert response["thread_env"] == THREAD_ENV
    assert response["row_process_start_method"] == "spawn"
    assert len(response["load_average_at_start"]) == 3


@pytest.mark.skipif(
    not all(path.is_file() for path in (_CANTERA_PYTHON, _OLD_MECHANISM, _OLD_THERMO)),
    reason="configured Cantera interpreter and pinned CHEMKIN inputs are unavailable",
)
def test_cantera_worker_reports_construction_errors_at_run_level(tmp_path: Path) -> None:
    response = _run_real_cantera_worker(tmp_path, volume_m3="not-a-volume")

    assert response["status"] == "refused"
    assert response["refusal"] == "reactor_construction_failed"
    assert response["rows"] == []


def _run_real_cantera_worker(tmp_path: Path, *, volume_m3: float | str) -> dict[str, Any]:
    mechanism = normalize_for_ck2yaml(_OLD_MECHANISM.read_bytes(), file=_OLD_MECHANISM.name)
    thermo = normalize_for_ck2yaml(_OLD_THERMO.read_bytes(), file=_OLD_THERMO.name)
    mechanism_path = tmp_path / "mechanism.inp"
    thermo_path = tmp_path / "thermo.dat"
    mechanism_path.write_bytes(mechanism.data)
    thermo_path.write_bytes(thermo.data)
    request_path = tmp_path / "request.json"
    response_path = tmp_path / "response.json"
    request_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mechanism_path": str(mechanism_path),
                "thermo_path": str(thermo_path),
                "converted_yaml_path": str(tmp_path / "converted.yaml"),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "reaction_source_lines": {},
                "rows": [
                    {
                        "row_id": "real-reactor",
                        "temperature_k": 700.0,
                        "pressure_pa": 101325.0,
                        "residence_time_s": 2.0,
                        "volume_m3": volume_m3,
                        "composition": {"N2": 0.97012, "NC5H12": 0.00108, "O2": 0.0288},
                        "output_species": ["CO2"],
                    }
                ],
            }
        )
    )
    worker = Path(__file__).parents[1] / "carmel" / "benchmarks" / "cantera_worker.py"

    completed = subprocess.run(  # noqa: S603
        [str(_CANTERA_PYTHON), str(worker), str(request_path), str(response_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    return json.loads(response_path.read_text())


def _single_holdout_contract(tmp_path: Path, member: bytes) -> tuple[Any, Path]:
    import zipfile

    archive_path = tmp_path / "split.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("x_holdout.xml", member)

    class ArchivePin:
        def verify(self, _: bytes) -> None:
            return None

        def to_pinned_archive(self) -> object:
            return object()

    item = SimpleNamespace(path="x_holdout.xml", sha256=hashlib.sha256(member).hexdigest())
    contract = SimpleNamespace(
        split=SimpleNamespace(archive=ArchivePin(), studies=(SimpleNamespace(set="holdout", files=(item,)),))
    )
    return contract, archive_path


@pytest.mark.parametrize(
    "fragment",
    [
        b'<property name="Detection limit" units="ppm"/>',
        b'<property label="limit of detection"><value>1</value></property>',
        b'<property name="LOD"/>',
        b"<detectionLimit>1e-6</detectionLimit>",
        b'<experiment detection_limit="1e-6"/>',
    ],
)
def test_load_rows_refuses_a_record_declaring_a_detection_limit(tmp_path: Path, fragment: bytes) -> None:
    contract, archive_path = _single_holdout_contract(tmp_path, b"<experiment>" + fragment + b"</experiment>")

    with pytest.raises(ValueError, match=r"x_holdout\.xml.*detection-limit field"):
        holdout._load_rows(contract, archive_path)


def test_load_rows_does_not_refuse_prose_mentioning_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract, archive_path = _single_holdout_contract(
        tmp_path, b"<experiment><description>GC detection of products; flame detection</description></experiment>"
    )
    monkeypatch.setattr(holdout, "parse_series_record", lambda *_: (_ for _ in ()).throw(RuntimeError("parsed")))

    with pytest.raises(RuntimeError, match="parsed"):
        holdout._load_rows(contract, archive_path)


def test_detection_limit_sources_name_the_branch_used() -> None:
    fallback = [
        {"species": "A", "experimental": 0.0, "detection_limit": None},
        {"species": "A", "experimental": 4e-6, "detection_limit": None},
        {"species": "A", "experimental": 8e-6},
    ]
    limits, _, sources = holdout._metric_limits(fallback)
    assert limits == {"A": 2e-6}
    assert sources == {"A": "half_min_positive"}

    stated = [{**row, "detection_limit": 1e-7} for row in fallback]
    limits, _, sources = holdout._metric_limits(stated)
    assert limits == {"A": 1e-7}
    assert sources == {"A": "stated"}


_REAL_ARCHIVE = Path("/home/alon/runs/i116-alkane-census/evidence/pentane_indirect_v2.5.zip")


@pytest.mark.skipif(not _REAL_ARCHIVE.is_file(), reason="pinned ReSpecTh archive not available")
def test_load_rows_accepts_all_real_holdout_files() -> None:
    from carmel.benchmarks.n_pentane import load_contract

    contract_path = Path(__file__).resolve().parents[1] / "benchmarks" / "n-pentane-low-t" / "contract.json"
    rows = holdout._load_rows(load_contract(contract_path), _REAL_ARCHIVE)

    assert len({row["record"] for row in rows}) == 6
    assert len(rows) == 144
