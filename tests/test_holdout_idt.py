"""IDT source semantics and the Cantera-free harness boundary."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from carmel.benchmarks import cantera_worker as worker
from carmel.benchmarks import holdout_idt as idt
from carmel.benchmarks.holdout import IDTWorkerRequestRow, WorkerRequest, WorkerResponse

TIMES = [i / 10 for i in range(10)]
VALUES = [0.0, 0.0, 0.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]


@pytest.mark.parametrize(
    ("criterion", "expected"), [("max", 0.4), ("d/dt max", 0.3), ("1/2 max", 0.275), ("d/dt max extrapolated", 1 / 6)]
)
def test_hand_known_source_events(criterion: str, expected: float) -> None:
    assert worker.source_defined_idt(TIMES, VALUES, criterion) == pytest.approx(expected)
    assert worker.idt_trace_is_resolved(TIMES, VALUES, criterion, "OH")


@pytest.mark.parametrize(
    ("times", "values"),
    [
        ([], []),
        ([0.0], [1.0]),
        ([0.0, 0.0], [0.0, 1.0]),
        ([0.0, 1.0], [math.nan, 1.0]),
        ([0.0, 1.0], [1.0]),
        ([0.0, math.inf], [0.0, 1.0]),
    ],
)
def test_invalid_traces_are_unresolved(times: list[float], values: list[float]) -> None:
    assert worker.source_defined_idt(times, values, "max") is None
    assert not worker.idt_trace_is_resolved(times, values, "max", "OH")


def test_degenerate_and_incomplete_events_are_unresolved() -> None:
    assert worker.source_defined_idt([0.0, 1.0, 2.0], [2.0, 1.0, 0.0], "d/dt max") is None
    assert not worker.idt_trace_is_resolved(TIMES, [1.0] * 10, "max", "pressure")
    assert not worker.idt_trace_is_resolved(TIMES, list(TIMES), "max", "OH")
    assert not worker.idt_trace_is_resolved(TIMES, list(TIMES), "d/dt max", "OH")
    assert worker.source_defined_idt([0.0, 1.0, 2.0], [2.0, 3.0, 4.0], "1/2 max") == 0
    assert worker.source_defined_idt([0.0, 1.0, 2.0], [-4.0, -3.0, -2.0], "1/2 max") is None
    with pytest.raises(ValueError, match="criterion"):
        worker.source_defined_idt(TIMES, VALUES, "unknown")


def _row() -> dict[str, Any]:
    return {
        "row_id": "point",
        "temperature_k": 800.0,
        "pressure_pa": 101325.0,
        "composition": {"H2": 1.0},
        "ignition_target": "OH",
        "ignition_type": "d/dt max",
    }


@pytest.mark.parametrize(
    ("mode", "scope", "reason"),
    [
        ("ok", "row", None),
        ("cold", "row", "ignition_not_reached"),
        ("late", "row", "ignition_not_reached"),
        ("absent", "row", "target_species_absent"),
        ("solver", "row", "solver_failure: failure"),
        ("cap", "row", "max_steps_exceeded"),
        ("construction", "run", "reactor_construction_failed"),
    ],
)
def test_simulation_refusals_and_positive_delay(
    mode: str, scope: str, reason: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    gas = SimpleNamespace(species_names=["H2", "OH"], TPX=None)
    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(Solution=lambda _: gas))

    def trace(*_: Any) -> Any:
        if mode == "cap":
            raise worker.MaxStepsExceeded(1000000)
        if mode == "solver":
            raise RuntimeError("failure")
        if mode == "construction":
            raise worker.RunLevelSimulationError(
                {"refusal": "reactor_construction_failed", "construction_error": "failure"}
            )
        return (
            ([v * 10 for v in TIMES] if mode == "late" else TIMES),
            VALUES,
            [800.0, 849.0 if mode == "cold" else 850.0],
            100,
        )

    monkeypatch.setattr(worker, "integrate_idt_trace", trace)
    row = _row()
    if mode == "absent":
        row["ignition_target"] = "OH*"
    actual_scope, result = worker._simulate_idt_row("fake.yaml", row)
    assert actual_scope == scope
    assert result.get("reason", result.get("refusal")) == reason
    if mode == "cap":
        assert result["integrator_steps"] == 1000000
    if mode == "ok":
        assert result["ignition_delay_s"] == pytest.approx(0.3)
        assert result["status"] == "ok"
    if mode == "absent":
        assert result["ignition_target"] == "OH*"
        assert result["integrator_steps"] == 0
    if mode == "solver":
        assert result["integrator_steps"] is None


@pytest.mark.parametrize("phase", ["solution", "state"])
def test_idt_gas_construction_failures_are_run_level(phase: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class Gas:
        species_names = ["OH"]

        def __setattr__(self, name: str, value: Any) -> None:
            assert name == "TPX"
            raise ValueError("invalid initial state")

    def solution(_: str) -> Gas:
        if phase == "solution":
            raise ValueError("invalid mechanism")
        return Gas()

    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(Solution=solution))
    monkeypatch.setattr(worker, "integrate_idt_trace", lambda *a: pytest.fail("integration must not start"))
    scope, result = worker._simulate_idt_row("fake.yaml", _row())
    assert scope == "run"
    assert result == {
        "refusal": "reactor_construction_failed",
        "construction_error": "invalid mechanism" if phase == "solution" else "invalid initial state",
    }


def test_worker_refuses_exact_absent_targets_before_row_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {**_row(), "row_id": target, "ignition_target": target}
        for target in ("CH*", "OH*", "UNKNOWN*", "OH", "pressure", "temperature")
    ]
    converted = tmp_path / "converted.yaml"
    request, response = tmp_path / "request.json", tmp_path / "response.json"
    request.write_text(
        json.dumps(
            {
                "observable": "idt",
                "mechanism_path": "mechanism.inp",
                "thermo_path": "thermo.dat",
                "converted_yaml_path": str(converted),
                "trimmed_yaml_path": str(tmp_path / "trimmed.yaml"),
                "rows": rows,
            }
        )
    )

    def convert(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        converted.write_text("phases: []\nspecies: []\nreactions: []\n")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def execute(input_rows: list[dict[str, Any]], **kwargs: Any) -> tuple[dict[str, Any], int]:
        assert input_rows == rows
        assert kwargs["simulation"] is worker._simulate_idt_row
        expected = {
            target: {
                "row_id": target,
                "status": "refused",
                "reason": "target_species_absent",
                "ignition_target": target,
            }
            for target in ("UNKNOWN*",)
        }
        assert kwargs["initial_results"] == expected
        return {
            **expected,
            **{
                t: {"row_id": t, "status": "ok", "ignition_delay_s": 0.3}
                for t in ("CH*", "OH*", "OH", "pressure", "temperature")
            },
        }, 0

    gas = SimpleNamespace(species_names=["OH", "OHV", "CHV"])
    monkeypatch.setitem(sys.modules, "cantera", SimpleNamespace(__version__="test", Solution=lambda _: gas))
    monkeypatch.setattr(worker.subprocess, "run", convert)
    monkeypatch.setattr(worker, "execute_rows", execute)
    monkeypatch.setattr(sys, "argv", ["cantera_worker.py", str(request), str(response)])
    assert worker.main() == 0
    result = json.loads(response.read_text())
    assert result["status"] == "ok"
    assert [r.get("reason") for r in result["rows"]] == [
        None,
        None,
        "target_species_absent",
        None,
        None,
        None,
    ]


def test_request_discriminator_rejects_mismatched_rows(tmp_path: Path) -> None:
    kwargs = {
        key: tmp_path / key
        for key in ("mechanism_path", "thermo_path", "converted_yaml_path", "trimmed_yaml_path", "checkpoint_path")
    }
    row = IDTWorkerRequestRow(**_row())
    with pytest.raises(ValueError, match="row type"):
        WorkerRequest(**kwargs, rows=(row,), reaction_source_lines={})
    assert WorkerRequest(**kwargs, observable="idt", rows=(row,), reaction_source_lines={}).observable == "idt"


@pytest.mark.parametrize("target", ["pressure", "temperature", "OH"])
@pytest.mark.parametrize(
    ("pressures", "bracket", "refused"),
    [
        ([1, 3, 5], [2, 4], True),
        ([1, 5, 5], [2, 4], True),
        ([5, 1, 1], [2, 4], True),
        ([3, 3, 3], [2, 4], True),
        ([1, 3, 3], [2, None], True),
        ([5, 3, 3], [None, 4], True),
        ([1, 1.5, 2], [2, 4], False),
        ([4, 5, 5], [2, 4], False),
    ],
)
def test_idt_pressure_guard_refuses_entry_or_crossing_of_trimmed_interval(
    monkeypatch: pytest.MonkeyPatch, target: str, pressures: list[float], bracket: list[float | None], refused: bool
) -> None:
    gas = SimpleNamespace(species_names=["OH"], species_index=lambda _: 0, P=pressures[0], X=[0.0], TPX=None)
    reactor = SimpleNamespace(phase=gas, T=800.0)

    class Network:
        time = 0.0
        solver_stats = {"steps": 0}

        def __init__(self, _: Any) -> None:
            self.index = 0

        def step(self) -> None:
            self.index += 1
            self.time = self.index / 10
            self.solver_stats = {"steps": self.index}
            gas.P = pressures[min(self.index, len(pressures) - 1)]
            gas.X = [VALUES[min(self.index, len(VALUES) - 1)]]
            reactor.T = 800 + 10 * gas.X[0]

    monkeypatch.setitem(
        sys.modules,
        "cantera",
        SimpleNamespace(
            Solution=lambda _: gas,
            IdealGasReactor=lambda *a, **k: reactor,
            ReactorNet=Network,
        ),
    )
    row = {**_row(), "ignition_target": target, "plog_trimmed_brackets_pa": [bracket]}
    scope, result = worker._simulate_idt_row("fake.yaml", row)
    assert scope == "row"
    if refused:
        assert result["status"] == "refused"
        assert result["reason"] == "plog_trimmed_interval_entered"
        assert result["integrator_steps"] in (0, 1)
        assert "ignition_delay_s" not in result
    else:
        assert result.get("reason") in (None, "ignition_not_reached")
        if target != "pressure":
            assert result["status"] == "ok"


def _export() -> list[dict[str, Any]]:
    return [
        {
            **_row(),
            "row_id": str(i),
            "group": "RCM-ideal" if i < 15 else "shock-tube",
            "temperature_band": "750_to_900",
            "experimental_s": 0.1,
            "ignition_target": "pressure" if i < 15 else "CH*",
            "inlet": {"nC7H16": 1.0},
        }
        for i in range(189)
    ]


def test_metric_and_coverage_do_not_hide_invalid_delays() -> None:
    rows = [
        {
            **r,
            "status": "ok" if i < 15 else "refused",
            "reason": None if i < 15 else "target_species_absent",
            "ignition_delay_s": 0.2 if i < 15 else None,
        }
        for i, r in enumerate(_export())
    ]
    metric = idt.idt_metric(rows)
    assert metric["overall"] == pytest.approx(math.log10(2))
    assert metric["scored_count"] == 15 and len(metric["refusals"]) == 174
    assert not idt.idt_coverage_gate(rows)["passed"]
    for row in rows[15:]:
        row.update(status="ok", reason=None, ignition_delay_s=0.2)
    assert idt.idt_coverage_gate(rows)["passed"]
    for row in rows[:4]:
        row["ignition_delay_s"] = math.nan
    assert not idt.idt_coverage_gate(rows)["passed"]
    rows = rows[4:]
    assert not idt.idt_coverage_gate(rows)["passed"]
    assert [idt.temperature_band(t) for t in (749.0, 750.0, 900.0, 901.0)] == [
        "below_750",
        "750_to_900",
        "750_to_900",
        "above_900",
    ]


@pytest.fixture
def fake_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for name, data in (
        ("contract", b"contract"),
        ("mechanism", b"SPECIES\nNC7H16\nEND\nREACTIONS\nEND\n"),
        ("thermo", b"thermo"),
    ):
        (tmp_path / name).write_bytes(data)
    pin = SimpleNamespace(verify=lambda _: None)
    contract = SimpleNamespace(
        mechanisms={"v_old": pin, "v_new": pin},
        thermo_source=pin,
        case="n-heptane-low-t",
        scoring=SimpleNamespace(secondary_groups=("shock-tube", "RCM-ideal")),
    )
    monkeypatch.setattr(idt, "load_contract", lambda _: contract)
    rows = _export()
    monkeypatch.setattr(idt, "load_idt_rows", lambda *_: copy.deepcopy(rows))

    def fake(request: WorkerRequest, **_: Any) -> WorkerResponse:
        assert request.observable == "idt"
        assert len(request.rows) == 189
        assert request.rows[15].ignition_target == "CH*"
        result = [
            {
                "row_id": row.row_id,
                "status": "ok",
                "reason": None,
                "ignition_delay_s": 0.2,
                "ignition_target": row.ignition_target,
                "wall_time_s": 1.0,
                "integrator_steps": 5,
            }
            for row in request.rows
        ]
        return WorkerResponse(
            schema_version=1,
            observable="idt",
            status="ok",
            rows=result,
            converter={},
            excited_state_submechanism={"sha256": worker.FFCM1_EXCITED_SHA256},
            thread_env=worker.THREAD_ENV,
            row_process_start_method="spawn",
            load_average_at_start=(0.0, 0.0, 0.0),
            simulation_lock=worker.simulation_lock(600.0, "test"),
            simulation_lock_sha256=worker.simulation_lock_sha256(600.0, "test"),
        )

    monkeypatch.setattr(idt, "invoke_cantera_worker", fake)
    kwargs = dict(
        contract_path=tmp_path / "contract",
        mechanism_path=tmp_path / "mechanism",
        mechanism_key="v_old",
        thermo_path=tmp_path / "thermo",
        thermo_url="https://example.test/thermo",
        thermo_retrieval_date="2026-10-30",
        thermo_sha256=hashlib.sha256(b"thermo").hexdigest(),
        split_root=tmp_path,
        run_dir=tmp_path / "run",
        cantera_python=Path("fake"),
    )
    return kwargs, rows


@pytest.mark.parametrize("mechanism_key", ["v_old", "v_new"])
def test_fake_worker_harness_preserves_targets_and_reports_all_rows(
    fake_harness: tuple[dict[str, Any], list[dict[str, Any]]], mechanism_key: str
) -> None:
    kwargs, rows = fake_harness
    kwargs["mechanism_key"] = mechanism_key
    report = idt.run_idt_harness(**kwargs)
    assert report["coverage_gate"]["passed"]
    assert report["metric"]["scored_count"] == 189
    assert report["diagnostics"]["groups"]["shock-tube"]["overall"] == pytest.approx(math.log10(2))
    assert report["row_results"][15]["ignition_target"] == "CH*"
    rows.pop()
    with pytest.raises(ValueError, match="export gate"):
        idt.run_idt_harness(**kwargs)


@pytest.mark.parametrize(("defect", "message"), [("thermo", "PIN CHANGED"), ("inlet", "unmapped inlet")])
def test_harness_rejects_input_defects_before_launching_worker(
    fake_harness: tuple[dict[str, Any], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
) -> None:
    kwargs, rows = fake_harness
    if defect == "thermo":
        kwargs["thermo_path"].write_bytes(b"different thermo")
    else:
        rows[0]["inlet"] = {"UNKNOWN": 1.0}
    monkeypatch.setattr(idt, "invoke_cantera_worker", lambda *a, **k: pytest.fail("worker must not launch"))
    with pytest.raises(ValueError, match=message):
        idt.run_idt_harness(**kwargs)


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("validation", "validation failed"),
        ("construction", "construction failed"),
        ("observable", "jsr"),
        ("duplicate", "row IDs"),
        ("missing", "row IDs"),
        ("foreign", "row IDs"),
        ("target", "changed ignition target"),
    ],
)
def test_harness_rejects_inconsistent_worker_responses(
    fake_harness: tuple[dict[str, Any], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    defect: str,
    message: str,
) -> None:
    kwargs, _ = fake_harness
    original = idt.invoke_cantera_worker

    def corrupt(*args: Any, **kw: Any) -> WorkerResponse:
        data = original(*args, **kw).model_dump()
        if defect in ("validation", "construction"):
            data.update(status="refused", refusal="bad mechanism")
            data[f"{defect}_error"] = f"{defect} failed"
        elif defect == "observable":
            data["observable"] = "jsr"
        elif defect == "duplicate":
            data["rows"] = (*data["rows"], data["rows"][0])
        elif defect == "missing":
            data["rows"] = data["rows"][:-1]
        elif defect == "foreign":
            data["rows"][0]["row_id"] = "foreign"
        else:
            data["rows"][15]["ignition_target"] = "CH"
        return WorkerResponse.model_validate(data)

    monkeypatch.setattr(idt, "invoke_cantera_worker", corrupt)
    with pytest.raises(ValueError, match=message):
        idt.run_idt_harness(**kwargs)


def test_harness_retains_refusal_when_worker_omits_reason_or_target(
    fake_harness: tuple[dict[str, Any], list[dict[str, Any]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, _ = fake_harness
    original = idt.invoke_cantera_worker

    def omit(*args: Any, **kw: Any) -> WorkerResponse:
        data = original(*args, **kw).model_dump()
        data["rows"][15].update(status="refused", reason=None, ignition_target=None)
        return WorkerResponse.model_validate(data)

    monkeypatch.setattr(idt, "invoke_cantera_worker", omit)
    report = idt.run_idt_harness(**kwargs)
    assert report["row_results"][15]["reason"] == "unspecified_worker_refusal"
    assert report["row_results"][15]["ignition_target"] == "CH*"
    assert report["coverage_gate"]["unexpected_shock_rows"] == ["15"]
    assert report["coverage_gate"]["passed"]


@pytest.mark.parametrize("metadata", [None, {"sha256": "changed"}])
def test_harness_refuses_unconfirmed_excited_state_chemistry(
    fake_harness: tuple[dict[str, Any], list[dict[str, Any]]],
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, str] | None,
) -> None:
    kwargs, _ = fake_harness
    original = idt.invoke_cantera_worker

    def corrupt(*args: Any, **kw: Any) -> WorkerResponse:
        data = original(*args, **kw).model_dump()
        data["excited_state_submechanism"] = metadata
        return WorkerResponse.model_validate(data)

    monkeypatch.setattr(idt, "invoke_cantera_worker", corrupt)
    with pytest.raises(ValueError, match="pinned excited-state"):
        idt.run_idt_harness(**kwargs)


def test_trace_integrator_locks_physics_horizon_and_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    class Gas:
        P = 101325.0
        X = [0.01]

        def species_index(self, target: str) -> int:
            assert target == "OH"
            return 0

    gas = Gas()
    reactor = SimpleNamespace(phase=gas, T=800.0)
    networks = []

    def make_reactor(model: Any, *, energy: str) -> Any:
        assert model is gas and energy == "on"
        return reactor

    class Network:
        time = 0.0
        atol = rtol = 0.0
        solver_stats = {"steps": 0}

        def __init__(self, reactors: list[Any]) -> None:
            assert reactors == [reactor]
            networks.append(self)

        def step(self) -> None:
            self.time += 0.005
            self.solver_stats = {"steps": self.solver_stats["steps"] + 1}
            gas.P += 100.0
            gas.X[0] += 0.001
            reactor.T += 1.0

    ct = SimpleNamespace(IdealGasReactor=make_reactor, ReactorNet=Network)
    for target in ("pressure", "temperature", "OH"):
        row = {**_row(), "ignition_target": target}
        times, values, temperatures, steps = worker.integrate_idt_trace(ct, gas, row)
        assert times[0] == 0.0 and times[-2] < 1.0 <= times[-1]
        assert len(times) == len(values) == len(temperatures) == steps + 1
        assert networks[-1].atol == 1e-16 and networks[-1].rtol == 1e-8
    monkeypatch.setattr(worker, "IDT_MAX_STEPS", 3)
    with pytest.raises(worker.MaxStepsExceeded) as caught:
        worker.integrate_idt_trace(ct, gas, _row())
    assert caught.value.integrator_steps == 3
    assert networks[-1].solver_stats["steps"] == 3
    # Reaching the reactor horizon on the last permitted step completes normally.
    monkeypatch.setattr(worker, "IDT_HORIZON_S", 0.015)
    assert worker.integrate_idt_trace(ct, gas, _row())[-1] == 3
    ct.IdealGasReactor = lambda *a, **k: (_ for _ in ()).throw(ValueError("bad reactor"))
    with pytest.raises(worker.RunLevelSimulationError, match="bad reactor"):
        worker.integrate_idt_trace(ct, gas, _row())


@pytest.mark.parametrize("mode", ["valid", "fallback", "absent_composition", "mass_fraction", "absent_delay"])
def test_loader_verifies_pins_and_preserves_source_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from carmel.benchmarks.n_heptane import git_blob
    from carmel.schemas.datasets import Absent, Composition

    # Reuse a real envelope fixture shape via a minimal duck-typed record; unit
    # conversion and schema parsing are covered by the existing ChemKED suite.
    raw = b"pinned source"
    (tmp_path / "record.yaml").write_bytes(raw)
    pin = SimpleNamespace(
        path="record.yaml", sha256=hashlib.sha256(raw).hexdigest(), git_blob=git_blob(raw), rcm_ideal_points=(0,)
    )
    contract = SimpleNamespace(
        split=SimpleNamespace(
            studies=(SimpleNamespace(set="development", files=()), SimpleNamespace(set="holdout", files=(pin,)))
        )
    )
    called = []
    monkeypatch.setattr(idt, "load_split", lambda c, root: called.append((c, root)))
    # model_construct isolates the loader's traversal without duplicating numeric schemas.
    component = SimpleNamespace(species_raw_name="nC7H16", amount=1.0)
    composition = Composition.model_construct(basis="mole_fraction", components=(component,))
    point = SimpleNamespace(
        composition=composition,
        coordinates=[
            SimpleNamespace(axis_id="temperature", value=750.0),
            SimpleNamespace(axis_id="pressure", value=1e5),
        ],
        observations=[SimpleNamespace(axis_id="ignition_delay", value=0.004)],
    )
    record = SimpleNamespace(
        envelope=SimpleNamespace(composition=composition, series=[SimpleNamespace(points=[point])]),
        ignition=SimpleNamespace(
            target_raw=SimpleNamespace(raw="pressure"), criterion_raw=SimpleNamespace(raw="d/dt max")
        ),
    )
    monkeypatch.setattr(idt, "_base_unit_value", float)
    monkeypatch.setattr(idt, "parse_idt_record", lambda *_: record)
    if mode in ("fallback", "absent_composition"):
        point.composition = None
    if mode == "absent_composition":
        record.envelope.composition = None
    elif mode == "mass_fraction":
        point.composition = composition.model_copy(update={"basis": "mass_fraction"})
    elif mode == "absent_delay":
        point.observations[0].value = Absent.model_construct(reason="not_reported")
    if mode in ("absent_composition", "mass_fraction", "absent_delay"):
        message = "missing ignition delay" if mode == "absent_delay" else "unsupported inlet composition"
        with pytest.raises(ValueError, match=message):
            idt.load_idt_rows(contract, tmp_path)
        return
    rows = idt.load_idt_rows(contract, tmp_path)
    assert called == [(contract, tmp_path)]
    assert rows[0]["row_id"] == "record.yaml:0" and rows[0]["experimental_s"] == 0.004
    assert rows[0]["group"] == "RCM-ideal" and rows[0]["temperature_band"] == "750_to_900"
    (tmp_path / "record.yaml").write_bytes(b"changed")
    with pytest.raises(ValueError, match="SPLIT BLOB CHANGED"):
        idt.load_idt_rows(contract, tmp_path)


def test_idt_cli_routes_arguments_and_reports_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import Carmel
    from carmel.benchmarks import holdout

    monkeypatch.setattr(holdout, "resolve_cantera_python", lambda _: Path("fake"))
    expected = {"coverage_gate": {"passed": True}}

    def run(**kwargs: Any) -> Any:
        assert kwargs["split_root"] == tmp_path
        assert kwargs["cantera_python"] == Path("fake")
        return expected

    monkeypatch.setattr(idt, "run_idt_harness", run)
    args = [
        "score-holdout-idt",
        "--contract",
        "contract",
        "--mechanism",
        "mechanism",
        "--mechanism-key",
        "v_old",
        "--thermo",
        "thermo",
        "--thermo-url",
        "https://example.test/thermo",
        "--thermo-retrieval-date",
        "2026-10-30",
        "--thermo-sha256",
        "0" * 64,
        "--split-root",
        str(tmp_path),
        "--run-dir",
        str(tmp_path),
        "--output",
        str(tmp_path / "report.json"),
    ]
    assert Carmel.main(args) == 0
    expected["coverage_gate"]["passed"] = False
    assert Carmel.main(args) == 1
