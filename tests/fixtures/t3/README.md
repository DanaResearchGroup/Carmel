# T3 Golden Fixture

This directory contains T3/ARC-shaped artifacts used to keep Carmel's
parser/normalization tests deterministic and independent of a live T3
run (T3 currently cannot import on Python 3.12 — see
`docs/development.md`).

## Provenance — be precise about what is real vs. assembled

- `sample_project/input.yml` is copied verbatim from
  [`ReactionMechanismGenerator/T3`](https://github.com/ReactionMechanismGenerator/T3)
  `tests/data/functional_2_thermo/input.yml`. It declares
  `project: functional_2_thermo` with no `qm.project` override.
- `sample_project/iteration_1/ARC/functional_2_thermo_info.yml` and
  `sample_project/iteration_2/ARC/functional_2_thermo_info.yml` contain
  the real ``species``/``reactions`` content of two captured ARC info
  files from the upstream T3 repo's test data
  (`tests/data/restart/r6/iteration_6/ARC/T3_info.yml` and
  `tests/data/process_arc/iteration_2/ARC/T3_info.yml` respectively).
  **The filenames themselves are assembled, not captured**: real ARC
  names this file `<project>_info.yml` (see ARC's
  `save_project_info_file`, and T3's own reader in `t3/main.py`), so
  these two files were renamed from the upstream files' actual on-disk
  names to `functional_2_thermo_info.yml` — the name ARC would really
  produce for a project named `functional_2_thermo` — to match
  `input.yml`'s declared project and to exercise Carmel's real
  `<project>_info.yml` naming contract (`arc_info_filename()`).
- `sample_project/iteration_1/RMG/pdep/network1_1.py` and
  `network4_1.py` are copied verbatim from
  `tests/data/pdep_network/iteration_1/RMG/pdep/` in the upstream T3
  repo.
- `sample_project/t3.log` is a placeholder marker, not a real log (real
  `t3.log` files are large).

None of these files were assembled from a single, coherent, real T3
run — they are individually-real artifacts stitched together from
different upstream test fixtures so the aggregation-across-iterations
and multi-iteration behavior can be exercised deterministically.

## What the fixture proves — and what it does not

- Parsing a `<project>_info.yml` file yields the right `species` labels
  with their `success` status.
- Iterating across multiple `iteration_*` directories aggregates
  species across iterations.
- The `qm.level_of_theory` field is correctly extracted from the input
  YAML (T3 never writes LOT into output).
- PDep network files under `RMG/pdep/network*.py` are discovered and
  counted, and their filename stems are used as network IDs.

**This fixture does NOT validate pdep network parsing.** Carmel's
adapter (`_discover_pdep_networks` in `carmel/adapters/t3.py`) only
globs for `network*.py` filenames and uses the filename stem as the
network ID — it never opens or parses the file contents. The
`network1_1.py` / `network4_1.py` files in this fixture exist only so
that glob/discovery/counting behavior has something real to find; their
internal Python content is irrelevant to and unexercised by Carmel.

## Re-capturing

If T3's/ARC's output schema changes, re-capture by:

1. Running a tiny real T3 job (the T3 `examples/minimal/` input is the
   smallest).
2. Copying the resulting `iteration_*/ARC/<project>_info.yml`, the
   input file, and a couple of `iteration_*/RMG/pdep/network*.py` files
   into `sample_project/`, keeping ARC's real `<project>_info.yml`
   naming intact (do not rename to a fictitious filename).
3. Updating the assertions in `TestGoldenFixture` in
   `tests/test_t3_adapter.py` to match the new ground truth.
