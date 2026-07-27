"""ARC adapter — real subprocess execution and diagnostics normalization.

ARC (*Automatic Rate Calculator*, DanaResearchGroup/ARC) computes thermochemistry
and rate coefficients for individual species and reactions via quantum-chemistry
jobs. Where the T3 adapter (:mod:`carmel.adapters.t3`) drives a whole
generation/refinement *loop*, this adapter runs **one standalone ARC job**: it
builds a typed ARC input, invokes ARC as a subprocess, walks ARC's real project
directory tree, and normalizes the result into Carmel's :class:`DiagnosticsV1`
contract — the same protocol shape as :class:`carmel.adapters.t3.T3Adapter`.

This adapter is the only place in Carmel that knows ARC's on-disk layout
(``<project>_info.yml`` at the project root, ``output/output.yml``,
``output/status.yml``). If ARC changes its layout, :class:`ARCLayout` is the
single place to update.

**No mock mode.** The pure-Python parsing helpers (input building, project-tree
walking, normalization, level-of-theory extraction) are tested unconditionally
against a captured real ARC/Mockter fixture under ``tests/fixtures/arc/`` (see the
directory's README). The subprocess-execution path is exercised in the heavy CI
lane and skipped locally when ARC cannot actually be imported.

Guardrails (the *spirit* of the babysit-arc operator skill, encoded as native
Carmel policy — see :class:`ARCGuardrails`): a standalone ARC job is a single
orchestrated submission (no cluster spamming), no automatic re-submission, and a
bounded wall-clock derived from the action's estimate. We do **not** import or
shell out to the babysit-arc Claude-Code skill — only its principles.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
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
# Real ARC contract (verified against DanaResearchGroup/ARC @ main, 2026-07-26):
#   - input:  <project_dir>/input.yml with top-level {project, level_of_theory,
#             species: [{label, smiles}], reactions: [{label}], job_types, ...}
#   - executable: ARC.py at the ARC repo root (python ARC.py input.yml)
#   - output: <project_dir>/<project>_info.yml  (T3-info-shaped:
#               {species: [{label, smiles, success}], reactions: [{label, success}]})
#             <project_dir>/output/output.yml    (arc_version, *_level, species, reactions)
#             <project_dir>/output/status.yml    (per-species convergence)
#   - a level_of_theory containing the token "mock" routes to the Mockter ESS
#     adapter (fast, deterministic, no QM) — see ARC settings levels_ess.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ARCLayout:
    """Centralized ARC file/key naming — the single source of truth.

    If ARC changes its input schema or output layout, this is the only place to
    update; all input building, tree walking, and parsing route through it.
    """

    # Subprocess invocation
    EXECUTABLE_SCRIPT: str = "ARC.py"

    # Carmel-side input filename (we name our own copy)
    INPUT_FILENAME: str = "input.yml"
    CARMEL_STDOUT_FILENAME: str = "carmel_stdout.log"
    CARMEL_STDERR_FILENAME: str = "carmel_stderr.log"

    # Output layout
    INFO_SUFFIX: str = "_info.yml"
    OUTPUT_SUBDIR: str = "output"
    OUTPUT_FILENAME: str = "output.yml"
    STATUS_FILENAME: str = "status.yml"

    # Top-level ARC input keys
    INPUT_PROJECT_KEY: str = "project"
    INPUT_LOT_KEY: str = "level_of_theory"
    INPUT_SPECIES_KEY: str = "species"
    INPUT_REACTIONS_KEY: str = "reactions"
    INPUT_JOB_TYPES_KEY: str = "job_types"
    INPUT_TS_ADAPTERS_KEY: str = "ts_adapters"
    INPUT_COMPUTE_THERMO_KEY: str = "compute_thermo"

    # <project>_info.yml / output.yml fields
    INFO_SPECIES_KEY: str = "species"
    INFO_REACTIONS_KEY: str = "reactions"
    INFO_LABEL_KEY: str = "label"
    INFO_SUCCESS_KEY: str = "success"
    OUTPUT_VERSION_KEY: str = "arc_version"


ARC_LAYOUT = ARCLayout()
ARC_TOOL_NAME = "arc"

# Default level of theory for a real (PySCF/QM) ARC job. Callers override via
# ``action.parameters["level_of_theory"]``.
DEFAULT_LEVEL_OF_THEORY = "wb97xd/def2tzvp"

# A Mockter-routing level of theory: the token "mock" routes ARC to its Mockter
# ESS adapter (fast, deterministic, no QM). Used by the Mockter wiring proof and
# the test suite so the Carmel->ARC path is validated now, decoupled from PySCF.
MOCK_LEVEL_OF_THEORY = "CCSD(T)-Mock12/cc-pVMockZ//B2PLYPDMock/Def2TZMockP"

# Conservative default ARC job-type profile: geometry optimization only. This is
# a legitimate standalone ARC job and completes deterministically under Mockter.
# Fuller profiles (freq/sp/thermo/rates) are supplied per-action via
# ``action.parameters["job_types"]`` once the real QM (PySCF) path lands (I-017).
DEFAULT_JOB_TYPES: dict[str, bool] = {
    "conf_opt": False,
    "fine": False,
    "freq": False,
    "sp": False,
    "rotors": False,
    "bde": False,
    "opt": True,
}

_log = get_logger("adapters.arc")


@dataclass(frozen=True)
class ARCGuardrails:
    """Native Carmel policy encoding the *spirit* of babysit-arc.

    Submission-rate, concurrency, and retry limits enforced by the adapter
    itself — NOT by importing or shelling out to the babysit-arc operator skill.
    A standalone ARC job is a single orchestrated submission, so the effective
    concurrency is one and there is no automatic re-submission (escalate to the
    user instead of spamming the cluster).
    """

    max_concurrent_jobs: int = 1
    max_retries: int = 0
    min_seconds_between_submissions: float = 0.0
    subprocess_grace_seconds: int = 600


ARC_GUARDRAILS = ARCGuardrails()


# ---------------------------------------------------------------------------
# ARC discovery
# ---------------------------------------------------------------------------


def is_arc_installed() -> bool:
    """Return True if the ``arc`` package is discoverable (may not be importable)."""
    return importlib.util.find_spec("arc") is not None


def is_arc_importable() -> bool:
    """Return True if ``arc`` can actually be imported in this interpreter."""
    try:
        importlib.import_module("arc")
    except Exception:  # pragma: no cover - import-time unusable
        return False
    return True


def _arc_version() -> str | None:
    """Return ARC's version string if importable, else None."""
    try:
        module = importlib.import_module("arc")
    except Exception:  # pragma: no cover - import-time unusable
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version is not None else None


def _find_arc_executable() -> list[str] | None:
    """Locate the ARC executable.

    Preference order:
    1. ``$ARC_PATH/ARC.py`` if the env var is set
    2. ``ARC.py`` next to the importable ``arc`` package (repo root)
    3. ``ARC.py`` discoverable on PATH via ``shutil.which``
    """
    env_path = os.environ.get("ARC_PATH")
    if env_path:
        candidate = Path(env_path) / ARC_LAYOUT.EXECUTABLE_SCRIPT
        if candidate.exists():
            return ["python", str(candidate)]

    spec = importlib.util.find_spec("arc")
    if spec is not None and spec.origin is not None:
        # spec.origin is .../arc/__init__.py; ARC.py lives at the repo root.
        repo_root = Path(spec.origin).parent.parent
        candidate = repo_root / ARC_LAYOUT.EXECUTABLE_SCRIPT
        if candidate.exists():
            return ["python", str(candidate)]

    which = shutil.which(ARC_LAYOUT.EXECUTABLE_SCRIPT)
    if which is not None:
        return ["python", which]

    return None


# ---------------------------------------------------------------------------
# Input building — Carmel campaign/action -> real ARC input dict
# ---------------------------------------------------------------------------


def _campaign_species_to_arc(component: Any) -> dict[str, Any]:
    """Translate a Carmel mixture component into an ARC species dict."""
    out: dict[str, Any] = {"label": component.species}
    if component.smiles:
        out["smiles"] = component.smiles
    return out


def _resolve_species(campaign: Campaign, action: PlannedAction) -> list[dict[str, Any]]:
    """Resolve the ARC species list from the action, falling back to the mixture.

    An agent-planned ``run_arc`` action carries the exact species to compute in
    ``action.parameters["species"]``. When the key is absent entirely (a
    standalone job), we derive a species list from the campaign's initial
    mixture. When the key is present, it is the exact requested target set:
    an empty list, or any malformed entry, raises rather than silently
    dropping entries or falling back to the mixture — the caller asked for
    specific targets and must get exactly that job, or an explicit error.

    Raises:
        ValueError: If ``species`` is present but empty, or contains any
            entry that is not a dict with a non-empty ``label``.
    """
    if "species" not in action.parameters:
        return [_campaign_species_to_arc(c) for c in campaign.input.initial_mixture.components]
    raw = action.parameters["species"]
    entries = list(raw or [])
    if not entries:
        raise ValueError("Action parameter 'species' is present but empty; at least one species is required")
    species: list[dict[str, Any]] = []
    malformed: list[Any] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("label"):
            item: dict[str, Any] = {"label": str(entry["label"])}
            if entry.get("smiles"):
                item["smiles"] = str(entry["smiles"])
            species.append(item)
        else:
            malformed.append(entry)
    if malformed:
        raise ValueError(
            "Action parameter 'species' contains malformed entries (each must be a "
            f"dict with a non-empty 'label'); rejected entries: {malformed!r}"
        )
    return species


def _resolve_reactions(action: PlannedAction) -> list[dict[str, Any]]:
    """Resolve the ARC reactions list from the action (optional).

    When the ``reactions`` key is absent entirely, or present but an empty
    list, there are legitimately no reactions to compute (a species-only
    thermo job is valid). When it is present and non-empty, every entry must
    be valid; any malformed entry raises instead of silently dropping it.

    Raises:
        ValueError: If ``reactions`` is present, non-empty, and contains any
            entry that is not a dict with a non-empty ``label``.
    """
    if "reactions" not in action.parameters:
        return []
    raw = action.parameters["reactions"]
    entries = list(raw or [])
    if not entries:
        return []
    reactions: list[dict[str, Any]] = []
    malformed: list[Any] = []
    for entry in entries:
        if isinstance(entry, dict) and entry.get("label"):
            reactions.append({"label": str(entry["label"])})
        else:
            malformed.append(entry)
    if malformed:
        raise ValueError(
            "Action parameter 'reactions' contains malformed entries (each must be a "
            f"dict with a non-empty 'label'); rejected entries: {malformed!r}"
        )
    return reactions


def build_arc_input(campaign: Campaign, action: PlannedAction) -> dict[str, Any]:
    """Build an ARC YAML input dictionary from a Carmel campaign and action.

    The output mirrors ARC's real top-level input schema. ``level_of_theory`` and
    the species/reactions to compute come from ``action.parameters`` (an
    agent-planned action) and fall back to the campaign's mixture for a standalone
    job.

    Args:
        campaign: The campaign being executed.
        action: The planned ``run_arc`` action.

    Returns:
        A dict suitable for serialization to ``input.yml`` for ARC.

    Raises:
        ValueError: If the project name is unsafe for path joins, or if a
            present ``species``/``reactions`` parameter yields no valid
            entries (see :func:`_resolve_species` / :func:`_resolve_reactions`).
    """
    params = action.parameters
    project = str(params.get("project") or campaign.input.workspace_name)
    if "/" in project or "\\" in project or ".." in project:
        # The project name flows into path joins (``<project>_info.yml`` under
        # the run dir); a separator or ``..`` would let the resolved info path
        # escape the run directory.
        raise ValueError(f"Invalid ARC project name {project!r}: must not contain path separators or '..'")
    level_of_theory = str(params.get("level_of_theory") or DEFAULT_LEVEL_OF_THEORY)
    job_types = params.get("job_types") or dict(DEFAULT_JOB_TYPES)
    reactions = _resolve_reactions(action)

    payload: dict[str, Any] = {
        ARC_LAYOUT.INPUT_PROJECT_KEY: project,
        ARC_LAYOUT.INPUT_LOT_KEY: level_of_theory,
        ARC_LAYOUT.INPUT_COMPUTE_THERMO_KEY: bool(params.get("compute_thermo", False)),
        ARC_LAYOUT.INPUT_JOB_TYPES_KEY: job_types,
        ARC_LAYOUT.INPUT_SPECIES_KEY: _resolve_species(campaign, action),
    }
    if reactions:
        payload[ARC_LAYOUT.INPUT_REACTIONS_KEY] = reactions
    else:
        # No reactions -> no TS search; keep the job to the requested species only.
        payload[ARC_LAYOUT.INPUT_TS_ADAPTERS_KEY] = []
    return payload


def write_arc_input_file(target_dir: Path, payload: dict[str, Any]) -> Path:
    """Atomically write an ARC ``input.yml`` into ``target_dir``.

    Args:
        target_dir: Directory to write into (created if missing).
        payload: The ARC input dict.

    Returns:
        The path to the written ``input.yml``.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / ARC_LAYOUT.INPUT_FILENAME
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Output normalization — real ARC project tree -> DiagnosticsV1
# ---------------------------------------------------------------------------


def arc_info_filename(input_dict: dict[str, Any]) -> str:
    """Resolve ARC's per-project info filename from the ARC input.

    Mirrors upstream ARC exactly (``arc/main.py``, which writes
    ``f"{self.project}_info.yml"`` into the project directory). Unlike the T3
    input, an ARC input has no ``qm`` block, so the project name comes from the
    top-level ``project`` key and nowhere else.

    Args:
        input_dict: The ARC input dict this run was built from.

    Returns:
        The expected info filename, e.g. ``"my_project_info.yml"``.

    Raises:
        ValueError: If no non-empty ``project`` key is present.
    """
    project = input_dict.get(ARC_LAYOUT.INPUT_PROJECT_KEY)
    if not project:
        raise ValueError("Cannot resolve ARC info filename: no non-empty 'project' key in the ARC input.")
    return f"{project}{ARC_LAYOUT.INFO_SUFFIX}"


def resolve_project_info_file(project_dir: Path, input_dict: dict[str, Any]) -> Path | None:
    """Resolve the ``<project>_info.yml`` path for a run, if ARC wrote it.

    Deliberately *not* a glob. A glob over the project directory can match a
    stale file from an earlier run or a foreign capture that happens to share
    the suffix, which would let Carmel report another project's results as this
    run's. Resolving the exact name ARC was told to write means a missing file
    reads as a missing file.

    Args:
        project_dir: The ARC project directory.
        input_dict: The ARC input dict this run was built from.

    Returns:
        The info file path, or None if ARC did not write it.

    Raises:
        ValueError: If the project name cannot be resolved from the input.
    """
    path = project_dir / arc_info_filename(input_dict)
    return path if path.is_file() else None


def read_arc_info_file(path: Path) -> dict[str, Any]:
    """Read an ARC ``<project>_info.yml`` file into a dict.

    Args:
        path: Path to the info file.

    Returns:
        The parsed dict (always a mapping with at least ``species`` and
        ``reactions`` keys).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a YAML mapping or is not valid YAML.
    """
    if not path.exists():
        raise FileNotFoundError(f"ARC info file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"ARC info file is not valid YAML: {path}: {err}") from err
    if not isinstance(data, dict):
        raise ValueError(f"ARC info file must be a YAML mapping: {path}")
    data.setdefault(ARC_LAYOUT.INFO_SPECIES_KEY, [])
    data.setdefault(ARC_LAYOUT.INFO_REACTIONS_KEY, [])
    return data


def read_arc_output_file(project_dir: Path) -> dict[str, Any] | None:
    """Read ``output/output.yml`` from an ARC project dir, if present.

    Raises:
        ValueError: If the file exists but is not valid YAML.
    """
    path = project_dir / ARC_LAYOUT.OUTPUT_SUBDIR / ARC_LAYOUT.OUTPUT_FILENAME
    if not path.exists():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as err:
        raise ValueError(f"ARC output file is not valid YAML: {path}: {err}") from err
    return data if isinstance(data, dict) else None


def _coerce_species_entry(entry: Any) -> SpeciesSelection | None:
    """Coerce a raw ARC info species dict into a SpeciesSelection."""
    if not isinstance(entry, dict):
        return None
    label = entry.get(ARC_LAYOUT.INFO_LABEL_KEY)
    if not label:
        return None
    success = entry.get(ARC_LAYOUT.INFO_SUCCESS_KEY)
    smiles = entry.get("smiles")
    reason = f"ARC · success={success!r}" if success is not None else "ARC"
    return SpeciesSelection(label=str(label), smiles=smiles, reason=reason)


def _coerce_reaction_entry(entry: Any) -> ReactionSelection | None:
    """Coerce a raw ARC info reaction dict into a ReactionSelection."""
    if not isinstance(entry, dict):
        return None
    label = entry.get(ARC_LAYOUT.INFO_LABEL_KEY) or entry.get("equation") or entry.get("reaction")
    if not label:
        return None
    reactants = list(entry.get("reactants") or entry.get("reactant_labels") or [])
    products = list(entry.get("products") or entry.get("product_labels") or [])
    success = entry.get(ARC_LAYOUT.INFO_SUCCESS_KEY)
    reason = f"ARC · success={success!r}" if success is not None else "ARC"
    return ReactionSelection(
        label=str(label),
        reactants=[str(r) for r in reactants],
        products=[str(p) for p in products],
        reason=reason,
    )


def extract_level_of_theory(input_dict: dict[str, Any]) -> str | None:
    """Extract the level of theory from an ARC input dict (a top-level string)."""
    lot = input_dict.get(ARC_LAYOUT.INPUT_LOT_KEY)
    return str(lot) if lot else None


def _count_converged(output_dict: dict[str, Any] | None) -> int:
    """Count converged species reported in ARC's ``output.yml``."""
    if not output_dict:
        return 0
    species = output_dict.get("species") or []
    return sum(1 for s in species if isinstance(s, dict) and s.get("converged"))


def normalize_arc_outputs(
    project_dir: Path,
    input_dict: dict[str, Any],
    campaign_id: str,
    run_id: str,
) -> DiagnosticsV1:
    """Normalize a real ARC project tree into a ``DiagnosticsV1``.

    Args:
        project_dir: The ARC project directory (where ARC wrote its output).
        input_dict: The ARC input dict (source of the level of theory).
        campaign_id: The campaign that owns this run.
        run_id: The Carmel run that produced this output.

    Returns:
        A validated ``DiagnosticsV1``.

    Raises:
        ValueError: If ARC did not write its ``<project>_info.yml``. ARC saves
            that file unconditionally at the end of a run
            (``save_project_info_file``), so its absence means the run did not
            complete — even when an ``output/output.yml`` exists. Treating
            that as success would silently report zero species/reactions.
    """
    info_path = resolve_project_info_file(project_dir, input_dict)
    output_dict = read_arc_output_file(project_dir)

    if info_path is None:
        raise ValueError(
            f"No ARC info file found under {project_dir}. ARC writes it unconditionally at the end "
            "of a run, so a missing info file means ARC did not run to completion."
        )

    species: list[SpeciesSelection] = []
    reactions: list[ReactionSelection] = []
    info = read_arc_info_file(info_path)
    for raw in info.get(ARC_LAYOUT.INFO_SPECIES_KEY, []) or []:
        sel = _coerce_species_entry(raw)
        if sel is not None:
            species.append(sel)
    for raw in info.get(ARC_LAYOUT.INFO_REACTIONS_KEY, []) or []:
        rxn = _coerce_reaction_entry(raw)
        if rxn is not None:
            reactions.append(rxn)

    arc_version = output_dict.get(ARC_LAYOUT.OUTPUT_VERSION_KEY) if output_dict else None

    return DiagnosticsV1(
        campaign_id=campaign_id,
        run_id=run_id,
        model_version=None,
        level_of_theory=extract_level_of_theory(input_dict),
        generated_at=datetime.now(UTC),
        observable_summaries=[],
        species_to_compute=species,
        reactions_to_compute=reactions,
        pdep_networks_to_compute=[],  # standalone ARC does not discover PDep networks (that is T3's job)
        pdep_sensitivity_flag=False,
        warnings=[],
        tool_metadata={
            "adapter": ARC_TOOL_NAME,
            "arc_version": arc_version,
            "species_count": len(species),
            "reaction_count": len(reactions),
            "converged_species_count": _count_converged(output_dict),
        },
    )


# ---------------------------------------------------------------------------
# Adapter — orchestrates input -> subprocess -> normalization
# ---------------------------------------------------------------------------


class ARCAdapter:
    """Deterministic adapter that runs a single ARC job and parses its output.

    Same protocol shape as :class:`carmel.adapters.t3.T3Adapter`:
    ``run(workspace_root, campaign, action) -> (RunRecord, DiagnosticsV1 | None)``.
    Phase 1 supports the ``subprocess`` submission mode only.
    """

    def __init__(
        self,
        submission_mode: SubmissionMode = SubmissionMode.SUBPROCESS,
        guardrails: ARCGuardrails = ARC_GUARDRAILS,
    ) -> None:
        self.submission_mode = submission_mode
        self.guardrails = guardrails

    def estimate_cost(self, action: PlannedAction, campaign: Campaign | None = None) -> float:
        """Estimate the CPU-hour cost of a ``run_arc`` action.

        Uses the action's declared estimate when present; otherwise falls back
        to a conservative per-species/per-reaction estimate. When ``campaign``
        is given, that fallback counts the species/reactions the ARC input
        actually resolves to (including the mixture fallback when the action
        omits ``species``) — otherwise an action with no explicit species
        would be costed (and its subprocess wall-clock bounded) as a single
        species while the job computes the whole mixture. This feeds the
        shared, adapter-agnostic execution envelope in
        :mod:`carmel.services.authorization` and the subprocess timeout.
        """
        if action.estimated_cpu_hours > 0:
            return float(action.estimated_cpu_hours)
        if campaign is not None:
            try:
                n_species = len(_resolve_species(campaign, action))
                n_reactions = len(_resolve_reactions(action))
            except KeyError, TypeError, ValueError:
                pass  # unresolvable parameters — fall back to the raw counts below
            else:
                return float(max(1, n_species) + 2 * n_reactions)
        n_species = len(action.parameters.get("species") or [])
        n_reactions = len(action.parameters.get("reactions") or [])
        return float(max(1, n_species) + 2 * n_reactions)

    def run(
        self,
        workspace_root: Path,
        campaign: Campaign,
        action: PlannedAction,
    ) -> tuple[RunRecord, DiagnosticsV1 | None]:
        """Run one ARC job end-to-end and return a typed RunRecord + diagnostics.

        On any failure a ``RunRecord`` with ``status=FAILED`` and a typed
        ``FailureCode`` is returned (diagnostics ``None``); this method never
        raises for an ARC-side failure.
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
                campaign=campaign,
                failure_code=FailureCode.SUBPROCESS_ERROR,
                error_message=f"Could not create run directory {run_dir}: {e}",
            ), None
        stdout_path = run_dir / ARC_LAYOUT.CARMEL_STDOUT_FILENAME
        stderr_path = run_dir / ARC_LAYOUT.CARMEL_STDERR_FILENAME
        input_path = run_dir / ARC_LAYOUT.INPUT_FILENAME

        # 1. Build and write the ARC input.
        try:
            payload = build_arc_input(campaign, action)
            input_path = write_arc_input_file(run_dir, payload)
        except (KeyError, TypeError, ValueError, OSError) as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                campaign=campaign,
                failure_code=FailureCode.INPUT_BUILD_ERROR,
                error_message=f"Failed to build ARC input: {e}",
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ), None

        # 2. Locate the ARC executable.
        arc_executable = _find_arc_executable()
        if arc_executable is None:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                campaign=campaign,
                failure_code=FailureCode.TOOL_NOT_FOUND,
                error_message="ARC executable not found (no ARC.py and arc not importable)",
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            ), None

        # 3. Invoke ARC. ARC derives its project_directory from the input file's
        #    location, so it writes its output tree directly under run_dir.
        command = [*arc_executable, str(input_path)]
        _log.info("Invoking ARC: %s", " ".join(command))
        timeout = int(self.estimate_cost(action, campaign) * 3600 + self.guardrails.subprocess_grace_seconds)
        try:
            with (
                open(stdout_path, "w", encoding="utf-8") as stdout_file,
                open(stderr_path, "w", encoding="utf-8") as stderr_file,
            ):
                completed = subprocess.run(  # noqa: S603 -- ARC is a trusted tool
                    command,
                    cwd=run_dir,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired as e:
            return self._failed_record(
                run_id=run_id,
                action=action,
                started=started,
                campaign=campaign,
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
                campaign=campaign,
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
                campaign=campaign,
                ended=ended,
                failure_code=FailureCode.SUBPROCESS_ERROR,
                error_message=f"ARC exited with code {completed.returncode}",
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
            ), None

        # 4. Normalize ARC's real output tree into DiagnosticsV1.
        try:
            diagnostics = normalize_arc_outputs(
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
                campaign=campaign,
                ended=ended,
                failure_code=FailureCode.INVALID_OUTPUT,
                error_message=f"Failed to normalize ARC output: {e}",
                input_path=input_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                command=command,
            ), None

        record = RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name=ARC_TOOL_NAME,
            tool_version=_arc_version(),
            status=RunStatus.SUCCEEDED,
            failure_code=FailureCode.NONE,
            started_at=started,
            ended_at=ended,
            # The same estimate the envelope and timeout were computed from, so
            # budget auditing matches what the adapter actually reserved.
            estimated_cpu_hours=self.estimate_cost(action, campaign),
            actual_cpu_hours=(ended - started).total_seconds() / 3600.0,
            submission_mode=self.submission_mode,
            command=command,
            input_path=input_path,
            output_path=run_dir / ARC_LAYOUT.OUTPUT_SUBDIR,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            level_of_theory=diagnostics.level_of_theory,
        )
        return record, diagnostics

    def _failed_record(
        self,
        run_id: str,
        action: PlannedAction,
        started: datetime,
        failure_code: FailureCode,
        error_message: str,
        campaign: Campaign | None = None,
        input_path: Path | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        command: list[str] | None = None,
        ended: datetime | None = None,
    ) -> RunRecord:
        """Build a typed failure RunRecord.

        ``campaign`` is threaded through to :meth:`estimate_cost` so failed
        run records carry the same resolved-input estimate as timeout/success
        records, rather than the blind single-species fallback used when the
        campaign (and thus the actual species/reaction count) is unknown.
        """
        return RunRecord(
            run_id=run_id,
            action_id=action.action_id,
            tool_name=ARC_TOOL_NAME,
            tool_version=_arc_version(),
            status=RunStatus.FAILED,
            failure_code=failure_code,
            started_at=started,
            ended_at=ended or datetime.now(UTC),
            estimated_cpu_hours=self.estimate_cost(action, campaign),
            submission_mode=self.submission_mode,
            command=command,
            input_path=input_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            error_message=error_message,
        )
