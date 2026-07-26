# Copyright 2026 Dana Research Group
# SPDX-License-Identifier: Apache-2.0

"""T3 adapter — real subprocess execution and diagnostics normalization.

T3 is the *Tandem Tool* from the ReactionMechanismGenerator org. T3 is
trusted but it is *not* a single-file in/out tool: its real output lives
in a project directory tree of iterations, each containing an ``RMG/``
and an ``ARC/`` subdir. This adapter is the only place in Carmel that
knows about that layout.

The adapter is responsible for:

1. Building a typed T3 input dict from a Carmel campaign that matches
   T3's real input schema (``project``, ``t3``, ``rmg``, ``qm``).
2. Locating T3 (the executable script ``T3.py`` in the T3 repo, or the
   ``t3`` package as a module) and invoking it as a subprocess.
3. Walking T3's project directory after a run, parsing the per-iteration
   ``ARC/<project>_info.yml`` files (ARC's real output naming — see
   ``arc_info_filename``), and counting RMG PDep networks under
   ``RMG/pdep/network*.py``.
4. Normalizing the result into Carmel's ``DiagnosticsV1`` schema and
   producing a typed ``RunRecord``.

There is **no mock mode**. The pure-Python parsing helpers are tested
unconditionally against a captured real T3 fixture under
``tests/fixtures/t3/`` (see that directory's README). The
subprocess-execution path is exercised in the heavy CI lane and skipped
locally if T3 cannot actually be imported.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml

from carmel.logger import get_logger
from carmel.schemas.campaign import Campaign
from carmel.schemas.diagnostics import (
    DiagnosticsV1,
    PDepNetworkSelection,
    ReactionSelection,
    SpeciesSelection,
)
from carmel.schemas.plan import PlannedAction
from carmel.schemas.run import (
    FailureCode,
    RunRecord,
    RunStatus,
    SubmissionMode,
)

# ---------------------------------------------------------------------------
# T3 contract — every assumption Carmel makes about T3 lives here.
#
# Validated against real T3 source at /home/alon/Code/T3 on 2026-04-07:
#   - input schema: t3/schema.py (T3Sensitivity, RMGSpecies, RMGReactor)
#   - executable:   T3.py at the T3 repo root
#   - output:       Projects/<project>/iteration_N/{RMG,ARC}/...
#                   ARC/<project>_info.yml lists {species, reactions} with success flags
#                   RMG/pdep/network*.py are RMG-style network files
#                   t3.log is the top-level log
#   - LOT:          comes from input qm.level_of_theory (T3 never writes it back)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class T3Layout:
    """Centralized T3 file/key naming. The single source of truth.

    If T3 changes its output layout, this class is the only place to
    update — all walking, parsing, and assertions go through it.
    """

    # Subprocess invocation
    EXECUTABLE_SCRIPT: str = "T3.py"
    EXECUTABLE_MODULE: str = "T3"  # python -m T3 fallback
    # Env var giving the path to T3's own interpreter (t3_env under the
    # three-env deployment model: rmg_env / t3_env / crml_env — see
    # carmel/adapters/t3.py module docstring / CLAUDE.md). Carmel and T3
    # cannot generally share an interpreter, so T3 must never be launched
    # (or probed) with Carmel's own `sys.executable`.
    T3_PYTHON_ENV_VAR: str = "T3_PYTHON"

    # Output layout
    # LOG_FILENAME is T3's OWN log file: T3's logger (t3/logger.py) archives
    # any pre-existing file at this path into a ``log_archive/`` subdir and
    # then os.remove()s it before opening its own handle. Carmel must never
    # write to this path — a Carmel-owned file descriptor pointed at it would
    # be silently unlinked out from under the write the moment T3 starts
    # logging. Carmel's own stdout/stderr capture uses the CARMEL_* filenames
    # below instead.
    LOG_FILENAME: str = "t3.log"
    CARMEL_STDOUT_FILENAME: str = "carmel_stdout.log"
    CARMEL_STDERR_FILENAME: str = "carmel_stderr.log"
    ITERATION_GLOB: str = "iteration_*"
    ARC_SUBDIR: str = "ARC"
    RMG_SUBDIR: str = "RMG"
    PDEP_SUBDIR: str = "pdep"
    PDEP_NETWORK_GLOB: str = "network*.py"

    # Carmel-side input filename (we name our own copy)
    INPUT_FILENAME: str = "input.yml"

    # Top-level T3 input keys
    INPUT_PROJECT_KEY: str = "project"
    INPUT_T3_KEY: str = "t3"
    INPUT_RMG_KEY: str = "rmg"
    INPUT_QM_KEY: str = "qm"

    # Nested T3 input keys
    QM_LOT_KEY: str = "level_of_theory"
    QM_ADAPTER_KEY: str = "adapter"

    # ARC per-iteration info file fields (<project>_info.yml)
    INFO_SPECIES_KEY: str = "species"
    INFO_REACTIONS_KEY: str = "reactions"
    INFO_LABEL_KEY: str = "label"
    INFO_SUCCESS_KEY: str = "success"


T3_LAYOUT = T3Layout()
T3_TOOL_NAME: str = "t3"

_log = get_logger("adapters.t3")

# Keeps is_t3_importable()/_t3_version() subprocess probes cheap so they can
# never hang a CI run: importing t3 (which drags in ARC) is not free, but it
# should never take anywhere close to this long when it succeeds or fails.
_T3_IMPORT_PROBE_TIMEOUT_S: float = 30.0


# ---------------------------------------------------------------------------
# Discovery / availability
# ---------------------------------------------------------------------------


def _resolve_t3_python() -> str:
    """Resolve the interpreter used to launch and probe T3.

    Resolution order:
        1. ``$T3_PYTHON`` if set and it points to an existing, executable
           file — the interpreter of T3's own environment (``t3_env`` under
           the three-env deployment model: ``rmg_env`` / ``t3_env`` /
           ``crml_env``; T3 and ARC together pin ``python =3.14`` and must
           share an environment, which is generally *not* Carmel's own).
        2. ``sys.executable`` — Carmel's own interpreter, which stays
           correct for a single-env developer setup where Carmel and T3
           share an environment.

    If ``$T3_PYTHON`` is set but does not point to an existing executable
    file, this logs a warning and falls back to ``sys.executable`` rather
    than raising: a bad env var must not crash the adapter's typed
    success/failure contract (see :class:`T3Adapter`). The misconfiguration
    still surfaces honestly downstream, as an ordinary
    not-importable/not-found failure once the fallback interpreter is
    actually used to probe or launch T3.

    Returns:
        Absolute or relative path to the interpreter to use for T3.
    """
    env_python = os.environ.get(T3_LAYOUT.T3_PYTHON_ENV_VAR)
    if env_python:
        if os.path.isfile(env_python) and os.access(env_python, os.X_OK):
            return env_python
        _log.warning(
            "%s is set to %r but is not an existing executable file; falling back to %s",
            T3_LAYOUT.T3_PYTHON_ENV_VAR,
            env_python,
            sys.executable,
        )
    return sys.executable


def is_t3_importable() -> bool:
    """Return True if ``t3`` can actually be imported by T3's own interpreter.

    This probes the interpreter that will actually run T3 (see
    :func:`_resolve_t3_python`), never Carmel's own process: under the
    three-env deployment model those are different interpreters, so
    importing ``t3`` in Carmel's own process proves nothing about whether
    the resolved T3 interpreter can import it. In the single-env developer
    case (``$T3_PYTHON`` unset, same interpreter as Carmel) this reduces to
    the historical in-process check.

    This is stricter than :func:`is_t3_installed` because T3 may be
    discoverable on ``sys.path`` but fail to import (e.g. if a transitive
    dependency like ARC uses ``distutils`` on Python 3.12).
    """
    t3_python = _resolve_t3_python()
    try:
        completed = subprocess.run(  # noqa: S603 -- resolved, locally-configured interpreter only
            [t3_python, "-c", "import t3"],
            capture_output=True,
            timeout=_T3_IMPORT_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def is_t3_installed() -> bool:
    """Return True if a ``t3`` package is discoverable (may not be importable)."""
    return importlib.util.find_spec("t3") is not None


def _t3_version() -> str | None:
    """Return T3's version string if importable via T3's own interpreter, else None.

    Like :func:`is_t3_importable`, this probes the resolved T3 interpreter
    (see :func:`_resolve_t3_python`) in a subprocess rather than importing
    ``t3`` into Carmel's own process, for the same three-env reason.
    """
    t3_python = _resolve_t3_python()
    try:
        completed = subprocess.run(  # noqa: S603 -- resolved, locally-configured interpreter only
            [t3_python, "-c", "import t3; print(getattr(t3, '__version__', ''))"],
            capture_output=True,
            timeout=_T3_IMPORT_PROBE_TIMEOUT_S,
            check=False,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    version = completed.stdout.strip()
    return version or None


def _find_t3_executable() -> list[str] | None:
    """Locate the T3 executable.

    Preference order:
        1. ``$T3_PATH/T3.py`` if the env var is set
        2. ``T3.py`` next to the importable ``t3`` package
        3. ``T3.py`` discoverable via ``shutil.which``
        4. ``python -m T3`` if T3 is importable

    Every branch launches T3 with the resolved T3 interpreter (see
    :func:`_resolve_t3_python`), never unconditionally with Carmel's own
    ``sys.executable``.
    """
    t3_python = _resolve_t3_python()

    env_path = os.environ.get("T3_PATH")
    if env_path:
        candidate = Path(env_path) / T3_LAYOUT.EXECUTABLE_SCRIPT
        if candidate.exists():
            return [t3_python, str(candidate)]

    spec = importlib.util.find_spec("t3")
    if spec is not None and spec.origin is not None:
        # spec.origin is .../t3/__init__.py; T3.py lives at the repo root
        repo_root = Path(spec.origin).parent.parent
        candidate = repo_root / T3_LAYOUT.EXECUTABLE_SCRIPT
        if candidate.exists():
            return [t3_python, str(candidate)]

    which = shutil.which(T3_LAYOUT.EXECUTABLE_SCRIPT)
    if which is not None:
        return [t3_python, which]

    if is_t3_importable():
        return [t3_python, "-m", T3_LAYOUT.EXECUTABLE_MODULE]

    return None


# ---------------------------------------------------------------------------
# Input building — Carmel campaign → real T3 input dict
# ---------------------------------------------------------------------------


def _carmel_reactor_to_t3(reactor: Any) -> dict[str, Any]:
    """Translate a Carmel ``ReactorSystem`` into a T3 RMG reactor dict.

    Carmel's ``ReactorSystem`` exposes ranges; T3's RMG reactor takes
    either scalars or 2-element ``[min, max]`` lists. We forward the
    range as a list so T3 generates conditions across the range.
    """
    t_lo, t_hi = reactor.temperature_range_K
    p_lo, p_hi = reactor.pressure_range_bar
    t_value: float | list[float] = [t_lo, t_hi] if t_lo != t_hi else float(t_lo)
    p_value: float | list[float] = [p_lo, p_hi] if p_lo != p_hi else float(p_lo)
    out: dict[str, Any] = {
        "type": "gas batch constant T P",
        "T": t_value,
        "P": p_value,
    }
    if reactor.residence_time_s is not None:
        out["termination_time"] = [float(reactor.residence_time_s), "s"]
    return out


def _carmel_species_to_t3(component: Any, observable_labels: set[str]) -> dict[str, Any]:
    """Translate a Carmel mixture component into a T3 RMG species dict."""
    out: dict[str, Any] = {
        "label": component.species,
        "concentration": float(component.mole_fraction),
    }
    if component.smiles:
        out["smiles"] = component.smiles
    if component.species in observable_labels:
        out["SA_observable"] = True
    return out


def build_t3_input(campaign: Campaign) -> dict[str, Any]:
    """Build a T3 YAML input dictionary from a Carmel campaign.

    The output dictionary mirrors T3's real top-level input schema:
    ``{project, t3, rmg, qm}`` (see ``t3/schema.py`` upstream).

    Args:
        campaign: The campaign to translate.

    Returns:
        A dict suitable for serialization as ``input.yml`` for T3.
    """
    observable_labels = {o.species for o in campaign.input.target_observables if o.species}
    species = [_carmel_species_to_t3(c, observable_labels) for c in campaign.input.initial_mixture.components]
    reactors = [_carmel_reactor_to_t3(r) for r in campaign.input.target_reactor_systems]

    return {
        T3_LAYOUT.INPUT_PROJECT_KEY: campaign.input.workspace_name,
        T3_LAYOUT.INPUT_T3_KEY: {
            "options": {
                "max_T3_iterations": 1,
                "max_RMG_walltime": "00:00:30:00",
            },
            "sensitivity": {
                "adapter": "CanteraConstantTP",
                "top_SA_species": 10,
                "top_SA_reactions": 10,
            },
        },
        T3_LAYOUT.INPUT_RMG_KEY: {
            "database": {
                "thermo_libraries": ["primaryThermoLibrary"],
                "kinetics_libraries": [],
            },
            "species": species,
            "reactors": reactors,
            "model": {
                "core_tolerance": 0.01,
            },
        },
        T3_LAYOUT.INPUT_QM_KEY: {
            T3_LAYOUT.QM_ADAPTER_KEY: "ARC",
            T3_LAYOUT.QM_LOT_KEY: "b3lyp/6-31g(d,p)",
            "job_types": {
                "conf_opt": True,
                "opt": True,
                "freq": True,
                "sp": True,
                "rotors": False,
                "fine": False,
            },
        },
    }


def write_t3_input_file(target_dir: Path, payload: dict[str, Any]) -> Path:
    """Atomically write a T3 input YAML file under *target_dir*.

    Args:
        target_dir: Directory in which T3 will run (input + outputs).
        payload: The T3 input dict.

    Returns:
        The path of the written input file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / T3_LAYOUT.INPUT_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Output normalization — real T3 project tree → DiagnosticsV1
# ---------------------------------------------------------------------------


def arc_info_filename(input_dict: dict[str, Any]) -> str:
    """Resolve the real ARC per-iteration info filename for a T3 input.

    Mirrors upstream T3 exactly (``T3/t3/main.py``):
    ``f"{project}_info.yml"`` where ``project`` is ``qm.project`` if the
    ``qm`` block is a mapping with a non-empty ``project`` key, else the
    T3 input's top-level ``project`` key.

    Args:
        input_dict: The parsed T3 input dict.

    Returns:
        The expected ARC info filename, e.g. ``"my_project_info.yml"``.

    Raises:
        ValueError: If no project name can be resolved from either the
            ``qm`` block or the top-level input.
    """
    qm = input_dict.get(T3_LAYOUT.INPUT_QM_KEY)
    project: Any = None
    if isinstance(qm, dict):
        project = qm.get(T3_LAYOUT.INPUT_PROJECT_KEY)
    if not project:
        project = input_dict.get(T3_LAYOUT.INPUT_PROJECT_KEY)
    if not project:
        raise ValueError(
            "Cannot resolve ARC info filename: no non-empty 'project' key found in "
            "either the qm block or the top-level T3 input."
        )
    return f"{project}_info.yml"


def read_t3_info_file(path: Path) -> dict[str, Any]:
    """Read an ARC per-iteration ``<project>_info.yml`` file as a dict.

    Args:
        path: Path to the file.

    Returns:
        The parsed dict (always a mapping with at least ``species`` and
        ``reactions`` keys).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a YAML mapping or is not valid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"T3 info file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"T3 info file is not valid YAML: {path}: {err}") from err
    if not isinstance(data, dict):
        raise ValueError(f"T3 info file must be a YAML mapping: {path}")
    data.setdefault(T3_LAYOUT.INFO_SPECIES_KEY, [])
    data.setdefault(T3_LAYOUT.INFO_REACTIONS_KEY, [])
    return data


def _coerce_species_entry(entry: Any, iteration: int) -> SpeciesSelection | None:
    """Coerce a raw T3 species dict into a SpeciesSelection."""
    if not isinstance(entry, dict):
        return None
    label = entry.get(T3_LAYOUT.INFO_LABEL_KEY)
    if not label:
        return None
    success = entry.get(T3_LAYOUT.INFO_SUCCESS_KEY)
    smiles = entry.get("smiles")
    reason = f"iteration {iteration} · success={success!r}" if success is not None else f"iteration {iteration}"
    return SpeciesSelection(label=str(label), smiles=smiles, reason=reason)


def _coerce_reaction_entry(entry: Any, iteration: int) -> ReactionSelection | None:
    """Coerce a raw T3 reaction dict into a ReactionSelection."""
    if not isinstance(entry, dict):
        return None
    label = entry.get(T3_LAYOUT.INFO_LABEL_KEY) or entry.get("equation") or entry.get("reaction")
    if not label:
        return None
    reactants = list(entry.get("reactants", []) or [])
    products = list(entry.get("products", []) or [])
    success = entry.get(T3_LAYOUT.INFO_SUCCESS_KEY)
    reason = f"iteration {iteration} · success={success!r}" if success is not None else f"iteration {iteration}"
    return ReactionSelection(label=str(label), reactants=reactants, products=products, reason=reason)


def _walk_iterations(project_dir: Path) -> list[Path]:
    """Return iteration directories under a T3 project, sorted by index."""
    if not project_dir.exists():
        return []
    items = [p for p in project_dir.glob(T3_LAYOUT.ITERATION_GLOB) if p.is_dir()]

    def _idx(p: Path) -> int:
        suffix = p.name.split("_", 1)[-1]
        try:
            return int(suffix)
        except ValueError:
            return -1

    return sorted(items, key=_idx)


def _aggregate_species_and_reactions(
    iteration_dirs: list[Path],
    info_filename: str,
) -> tuple[list[SpeciesSelection], list[ReactionSelection]]:
    """Aggregate species and reactions across all iterations.

    Deduplicates by label, keeping the *latest* iteration's reason
    (so a species that converged later is reported as such).

    The discriminator for "did ARC actually run this iteration" is the
    presence of the iteration's ``ARC/`` subdirectory — T3 only creates
    ``iteration_N/ARC/`` inside ``run_arc()``, and only calls that when
    ARC calculations are actually required for that iteration;
    ``set_paths()`` never creates the folder up front. So:

    * No ``ARC/`` subdir at all: ARC legitimately did not run this
      iteration (e.g. a converged terminal iteration, or a trailing
      RMG-only iteration). This is skipped silently — it is normal.
    * ``ARC/`` subdir exists but the info file is missing, or exists but
      fails to parse into a mapping: ARC was launched but crashed (or
      wrote something invalid) before it could save its info file. ARC
      writes that file very late and unconditionally
      (``save_project_info_file()``), so its absence or corruption means
      something is genuinely wrong — this raises rather than being
      swallowed into a warning.

    Args:
        iteration_dirs: Iteration directories to walk, in order.
        info_filename: The ARC info filename to look for in each
            iteration's ``ARC/`` subdir (see :func:`arc_info_filename`).

    Returns:
        A tuple of (species, reactions).

    Raises:
        FileNotFoundError: If an iteration's ``ARC/`` subdir exists but
            its info file does not.
        ValueError: If an iteration's ARC info file exists but is not
            valid YAML or does not parse to a mapping.
    """
    species_by_label: dict[str, SpeciesSelection] = {}
    reactions_by_label: dict[str, ReactionSelection] = {}
    for it_dir in iteration_dirs:
        try:
            iteration = int(it_dir.name.split("_", 1)[-1])
        except ValueError:
            iteration = 0
        arc_dir = it_dir / T3_LAYOUT.ARC_SUBDIR
        if not arc_dir.exists():
            continue
        info_path = arc_dir / info_filename
        if not info_path.exists():
            raise FileNotFoundError(
                f"ARC ran for iteration {it_dir.name} (ARC/ subdir exists) but did not save "
                f"a project info file at {info_path}; something must be wrong with the ARC run."
            )
        info = read_t3_info_file(info_path)
        for raw in info.get(T3_LAYOUT.INFO_SPECIES_KEY, []) or []:
            sp_sel = _coerce_species_entry(raw, iteration)
            if sp_sel is not None:
                species_by_label[sp_sel.label] = sp_sel
        for raw in info.get(T3_LAYOUT.INFO_REACTIONS_KEY, []) or []:
            rxn_sel = _coerce_reaction_entry(raw, iteration)
            if rxn_sel is not None:
                reactions_by_label[rxn_sel.label] = rxn_sel
    return list(species_by_label.values()), list(reactions_by_label.values())


def _discover_pdep_networks(iteration_dirs: list[Path]) -> list[PDepNetworkSelection]:
    """Discover RMG-generated pressure-dependent networks across iterations."""
    networks: dict[str, PDepNetworkSelection] = {}
    for it_dir in iteration_dirs:
        pdep_dir = it_dir / T3_LAYOUT.RMG_SUBDIR / T3_LAYOUT.PDEP_SUBDIR
        if not pdep_dir.exists():
            continue
        for net_file in sorted(pdep_dir.glob(T3_LAYOUT.PDEP_NETWORK_GLOB)):
            network_id = net_file.stem
            networks.setdefault(
                network_id,
                PDepNetworkSelection(
                    network_id=network_id,
                    species=[],
                    reactions=[],
                    reason=f"discovered at {net_file.parent.parent.name}",
                ),
            )
    return list(networks.values())


def extract_level_of_theory(input_dict: dict[str, Any]) -> str | None:
    """Pull ``qm.level_of_theory`` from a parsed T3 input dict."""
    qm = input_dict.get(T3_LAYOUT.INPUT_QM_KEY)
    if not isinstance(qm, dict):
        return None
    lot = qm.get(T3_LAYOUT.QM_LOT_KEY)
    return str(lot) if lot is not None else None


def normalize_t3_outputs(
    project_dir: Path,
    input_dict: dict[str, Any],
    *,
    campaign_id: str,
    run_id: str,
) -> DiagnosticsV1:
    """Walk a real T3 project directory and produce a ``DiagnosticsV1``.

    Args:
        project_dir: The directory T3 wrote outputs into (the same dir as
            the input file). Should contain ``iteration_*/`` subdirs.
        input_dict: The parsed T3 input dict (used to extract LOT).
        campaign_id: The campaign these diagnostics belong to.
        run_id: The Carmel run that produced them.

    Returns:
        A validated ``DiagnosticsV1``.

    Raises:
        ValueError: If no iterations are found and there are no PDep
            networks either (i.e. T3 produced nothing parseable), or if
            an iteration's ``ARC/`` subdir exists but its info file is
            not valid YAML or does not parse to a mapping.
        FileNotFoundError: If an iteration's ``ARC/`` subdir exists but
            its info file is missing (ARC crashed before saving it).
    """
    iteration_dirs = _walk_iterations(project_dir)
    networks = _discover_pdep_networks(iteration_dirs)

    if not iteration_dirs and not networks:
        raise ValueError(
            f"No T3 iteration directories or PDep networks found under {project_dir}. "
            f"T3 may not have produced any output."
        )

    info_filename = arc_info_filename(input_dict)
    species, reactions = _aggregate_species_and_reactions(iteration_dirs, info_filename)

    return DiagnosticsV1(
        campaign_id=campaign_id,
        run_id=run_id,
        model_version=None,
        level_of_theory=extract_level_of_theory(input_dict),
        generated_at=datetime.now(UTC),
        observable_summaries=[],
        species_to_compute=species,
        reactions_to_compute=reactions,
        pdep_networks_to_compute=networks,
        pdep_sensitivity_flag=False,
        warnings=[],
        tool_metadata={
            "iteration_count": len(iteration_dirs),
            "pdep_network_count": len(networks),
        },
    )


# ---------------------------------------------------------------------------
# Adapter — orchestrates input → subprocess → normalization
# ---------------------------------------------------------------------------


class T3Adapter:
    """Deterministic adapter for invoking T3 and parsing its output.

    Phase 1 supports the ``subprocess`` submission mode only. The
    ``server`` and ``local`` modes are reserved for a future iteration.
    """

    def __init__(self, submission_mode: SubmissionMode = SubmissionMode.SUBPROCESS) -> None:
        self.submission_mode = submission_mode

    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
    ) -> tuple[RunRecord, DiagnosticsV1 | None]:
        """Execute T3 end-to-end and return a RunRecord plus diagnostics.

        On any failure, the returned RunRecord has ``status=FAILED`` and
        a typed ``failure_code``; the diagnostics tuple element is None.

        Args:
            workspace_root: The campaign workspace root.
            campaign: The campaign being run.
            action: The planned T3 action.

        Returns:
            Tuple of (RunRecord, DiagnosticsV1 or None on failure).
        """
        run_id = str(uuid4())
        started = datetime.now(UTC)
        run_dir = workspace_root / "runs" / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.SUBPROCESS_ERROR,
                error_message=f"Could not create run directory {run_dir}: {e}",
            ), None
        stdout_path = run_dir / T3_LAYOUT.CARMEL_STDOUT_FILENAME
        stderr_path = run_dir / T3_LAYOUT.CARMEL_STDERR_FILENAME

        # 1. Build and write input
        try:
            payload = build_t3_input(campaign)
            input_path = write_t3_input_file(run_dir, payload)
        except (KeyError, TypeError, ValueError, OSError) as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.INPUT_BUILD_ERROR,
                error_message=str(e),
            ), None

        # 2. Locate T3 executable
        t3_executable = _find_t3_executable()
        if t3_executable is None:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.TOOL_NOT_FOUND,
                error_message="T3 executable not found (no T3.py and t3 module not importable)",
                input_path=input_path,
            ), None

        # 3. Invoke T3 — note T3 has no --output flag, it writes next to the input
        command = [*t3_executable, str(input_path)]
        _log.info("Invoking T3: %s", " ".join(command))
        try:
            with (
                open(stdout_path, "w", encoding="utf-8") as stdout_file,
                open(stderr_path, "w", encoding="utf-8") as stderr_file,
            ):
                completed = subprocess.run(  # noqa: S603 -- T3 is a trusted tool
                    command,
                    cwd=run_dir,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=int(action.estimated_cpu_hours * 3600 + 600),
                    check=False,
                )
        except subprocess.TimeoutExpired as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.TIMEOUT,
                error_message=str(e),
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
            ), None
        except (OSError, subprocess.SubprocessError) as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.SUBPROCESS_ERROR,
                error_message=str(e),
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
            ), None

        ended = datetime.now(UTC)

        if completed.returncode != 0:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.SUBPROCESS_ERROR,
                error_message=f"T3 exited with code {completed.returncode}",
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
                ended=ended,
            ), None

        # 4. Parse the project directory
        try:
            diagnostics = normalize_t3_outputs(
                project_dir=run_dir,
                input_dict=payload,
                campaign_id=campaign.campaign_id,
                run_id=run_id,
            )
        except (FileNotFoundError, ValueError) as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                failure_code=FailureCode.INVALID_OUTPUT,
                error_message=str(e),
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
                ended=ended,
            ), None

        run_record = RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name=T3_TOOL_NAME,
            tool_version=_t3_version(),
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=started,
            ended_at=ended,
            estimated_cpu_hours=action.estimated_cpu_hours,
            actual_cpu_hours=(ended - started).total_seconds() / 3600.0,
            submission_mode=self.submission_mode,
            command=command,
            input_path=input_path,
            output_path=run_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            level_of_theory=diagnostics.level_of_theory,
        )
        return run_record, diagnostics

    def _failed_record(
        self,
        run_id: str,
        action: PlannedAction,
        started: datetime,
        failure_code: FailureCode,
        error_message: str,
        input_path: Path | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        command: list[str] | None = None,
        ended: datetime | None = None,
    ) -> RunRecord:
        """Build a typed failure RunRecord."""
        return RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name=T3_TOOL_NAME,
            tool_version=_t3_version(),
            status=RunStatus.FAILED,
            failure_code=failure_code,
            started_at=started,
            ended_at=ended or datetime.now(UTC),
            estimated_cpu_hours=action.estimated_cpu_hours,
            submission_mode=self.submission_mode,
            command=command,
            input_path=input_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error_message=error_message,
        )
