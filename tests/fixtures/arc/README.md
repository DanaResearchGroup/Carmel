# ARC Golden Fixture

This directory contains a small set of **real ARC artifacts** captured from a
live ARC job run through ARC's **Mockter** ESS adapter (fast, deterministic, no
QM). They are checked in so that Carmel's ARC parser/normalization tests are
deterministic and do not require running ARC (which currently cannot import on
Python 3.12 — the same distutils blocker that affects T3).

## Provenance

Files under `sample_project/` were captured by running, in the `arc_env` conda
environment on 2026-07-26:

```bash
python ARC.py input.yml   # DanaResearchGroup/ARC @ main (b0dd288c)
```

against a two-species, opt-only input whose `level_of_theory` contains the token
`mock`, which routes ARC to its Mockter adapter (see ARC's `levels_ess`
setting). The job completed cleanly (exit 0).

| Carmel path | ARC output |
|---|---|
| `sample_project/input.yml` | the ARC input given to `ARC.py` |
| `sample_project/carmel_mock_opt_info.yml` | ARC's project-root `<project>_info.yml` (species/reactions + success flags) |
| `sample_project/output/output.yml` | ARC's `output/output.yml` (versions, levels, per-species convergence) |
| `sample_project/output/status.yml` | ARC's per-species job status summary |

The artifacts are checked in verbatim with one exception: the absolute
`paths.geo` values in `status.yml` pointed at the capture host's scratch
directory, and were rewritten to repo-relative paths so the fixture carries no
environment specifics. Nothing reads `status.yml` today, so the rewrite cannot
mask a parser bug — if a later change starts consuming those paths, re-capture
the fixture rather than hand-editing it.

> **Note on scope.** The fixture is an **opt-only** ARC job. A standalone `opt`
> job is a legitimate ARC job and exercises the full Carmel→ARC path (input
> build → subprocess → project-tree harvest → `DiagnosticsV1`). It intentionally
> avoids the `freq` job type, which hit an unrelated ARC-`main` regression in the
> Mockter frequency-parsing path at capture time (the scheduler passed a bad path
> to `parser.parse_frequencies`). That is an ARC-side bug to fix separately; it
> does not affect the Carmel adapter, and the fuller freq/sp/thermo/rate profiles
> are supplied per-action via `action.parameters["job_types"]` once the real QM
> (PySCF) path lands (I-017).

## What the fixture proves

The `TestGoldenFixture` class in `tests/test_arc_adapter.py` loads this fixture
and asserts that `normalize_arc_outputs()`:

- parses `<project>_info.yml` into the right `species` labels with their
  `success` status (ARC's info file is shaped like T3's `T3_info.yml`)
- extracts the `level_of_theory` from the ARC input (ARC writes a structured
  level into `output.yml`, but Carmel keeps the input's canonical string)
- records ARC metadata (adapter, `arc_version`, species/reaction/converged
  counts) into `DiagnosticsV1.tool_metadata`
- reports **no** PDep networks (multi-job PDep discovery is T3's job, not the
  standalone ARC adapter's)
