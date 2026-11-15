# Installation

```bash
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel
make install
uv run --env-file carmel-tools.env carmel --help
```

That is the whole thing. `make install` clones the external chemistry stack,
builds the two conda environments the chemistry tools need, syncs Carmel into
its own uv environment, and records where everything ended up.

Budget around 40 minutes and 15 GB the first time — most of it compiling
RMG-Py. Re-running when nothing has changed takes under a minute: the conda
environments and the RMG-Py build are skipped, and what remains is re-syncing
Carmel from `uv.lock` and re-installing ARC and T3 in editable mode.

## Requirements

- **uv**, from [astral.sh](https://docs.astral.sh/uv/getting-started/installation/),
  for Carmel's own environment.
- **conda**, from [Miniforge](https://conda-forge.org/download/) or Anaconda,
  for the chemistry tools. Only needed to run real campaigns; `make install-dev`
  needs uv alone.
- **git**, and a C toolchain for the Cython extensions RMG-Py and ARC compile
  (`build-essential` on Debian/Ubuntu, Xcode command line tools on macOS).
- Roughly 15 GB of disk, mostly RMG-database and the conda environments.

## What gets installed

Carmel drives four external tools, and they do not fit in one environment:
their Python requirements are mutually exclusive. So there are three
environments, and they talk to each other as **separate processes**, never as
imports.

| Environment | Managed by | Holds | Python | Who launches it |
| --- | --- | --- | --- | --- |
| `.venv` (in the Carmel checkout) | **uv**, from `uv.lock` | Carmel itself and its dev tools; the `agents` extra is optional (`make install-agents-dev`) | `>=3.14` | you, with `uv run` |
| `rmg_env` | **conda** | RMG-Py, Arkane | `>=3.9,<3.12` | T3, as a subprocess |
| `t3_env` | **conda** | T3 and ARC together | `=3.14` | Carmel, as a subprocess |

Carmel itself is plain Python with wheels on PyPI, so uv installs it from a
committed lockfile: every machine and every CI lane gets the same versions, and
`uv lock --check` fails when `pyproject.toml` and `uv.lock` disagree. The
chemistry tools stay on conda because they need conda-only packages: Open Babel
with its activation hook, RMG-Py's Cython extensions built against
conda-forge's compiled libraries, and the solver stacks RMG-Py and ARC pin in
their own `environment.yml` files.

ARC shares `t3_env` because T3 imports it in-process. The repositories
themselves are cloned next to Carmel:

```
parent/
├── Carmel/
├── RMG-Py/
├── RMG-database/
├── ARC/
└── T3/
```

## Targets

Run `make help` for the list. The install ones:

| Target | Does |
| --- | --- |
| `make install` | Everything below, in order |
| `make install-stack` | Clone RMG-Py, RMG-database, ARC and T3 |
| `make install-rmg` | Create `rmg_env` and build RMG-Py |
| `make install-t3` | Create `t3_env`, install ARC and T3 into it |
| `make install-carmel` | `uv sync` Carmel into `.venv`, write `carmel-tools.env` |
| `make install-dev` | `uv sync --locked --extra dev`: Carmel and its dev dependencies into `.venv`, nothing else |
| `make install-agents-dev` | `install-dev`, plus the `agents` extra (pydantic-ai, pypdf, rdkit) |

`make install-dev` is the one to use when you are working on Carmel's own code
and have no interest in running a real campaign — it is what the lint and test
CI lanes run, and it needs only uv. Run the tools through uv with the dev extra
(`uv run --extra dev pytest`, `uv run --extra dev ruff check`), or through the
`make test`, `make lint` and `make typecheck` targets, which do the same.

After changing a dependency in `pyproject.toml`, run `uv lock` and commit the
new `uv.lock`; CI refuses a stale one.

The scripts behind these targets live in `devtools/`. `make install` is
`devtools/install_all.sh`, which calls the others in order.

## Re-running is safe

Every step checks what is actually on disk before doing anything: does this
conda environment exist, can this environment import a compiled RMG module, is
this repository already checked out. Nothing is gated on a flag handed in from
outside saying a previous step "should have" run.

So `make install` is safe to run whenever — after `git pull`, after a failed
install, on a machine where half of it is already there. It picks up exactly
where the disk says it is, and re-syncs Carmel itself (which is cheap and
changes every commit).

"Safe" here means **resumable**, not reproducible. Two things it does not
promise:

- **Upstream is not pinned.** A cold install clones each repository's default
  branch, so the same Carmel commit can install different RMG/ARC/T3 revisions
  on different days.
- **Updating does not force a rebuild.** After `--update` moves RMG-Py's HEAD,
  the previously compiled extensions still import, so the build step is
  skipped. Run `make -C <RMG-Py> clean` first if you need the new sources
  compiled.

An existing checkout is reused as-is and is **never** updated behind your back.
Pass `--update` — `bash devtools/install_all.sh --update` — to fast-forward the
upstream repositories. If a checkout has no remote pointing at the expected
repository the installer says so and uses it anyway; that is the point of being
able to redirect it, but it does mean a repurposed directory is your
responsibility.

## Pointing at what you already have

Every location is an environment variable with a sensible default. Set one and
the installer uses your copy instead of cloning or creating its own.

| Variable | Default | What it names |
| --- | --- | --- |
| `CARMEL_STACK_ROOT` | Carmel's parent directory | Where the four repositories go |
| `RMG_PATH` | `$CARMEL_STACK_ROOT/RMG-Py` | RMG-Py checkout |
| `RMG_DB_PATH` | `$CARMEL_STACK_ROOT/RMG-database` | RMG-database checkout |
| `ARC_PATH` | `$CARMEL_STACK_ROOT/ARC` | ARC checkout |
| `T3_PATH` | `$CARMEL_STACK_ROOT/T3` | T3 checkout |
| `RMG_ENV` | `rmg_env` | Name of the RMG environment |
| `T3_CONDA_ENV` | `t3_env` | Name of the T3 + ARC environment |
| `CARMEL_TOOLS_ENV_FILE` | `<Carmel>/carmel-tools.env` | Where the tool paths are recorded |

For instance, to build on top of checkouts you keep in `~/Code`:

```bash
CARMEL_STACK_ROOT="$HOME/Code" make install
```

## How Carmel finds the chemistry tools

Carmel never imports RMG, T3 or ARC. It launches them as subprocesses in their
conda environments, and it learns where they are from environment variables:

- `T3_CONDA_ENV` (the usual one): Carmel runs `conda run -n $T3_CONDA_ENV`.
- `T3_PYTHON`: names T3's interpreter directly, for a deployment with no conda.

`make install-carmel` writes `T3_CONDA_ENV`, with `T3_PATH`, `ARC_PATH`, `RMG_PATH` and
`RMG_DB_PATH`, to `carmel-tools.env` in the Carmel checkout (gitignored). Read it
in either of two ways:

```bash
uv run --env-file carmel-tools.env carmel --help
# or, for the whole shell session:
set -a; source carmel-tools.env; set +a
```

`uv run --env-file` does not override a variable that is already set in your
shell; `source` does. The file is skipped when there is no T3 checkout, so a
Carmel-only environment is not handed paths that do not exist.

`T3_CONDA_ENV` is the important one. Carmel launches T3 with
`conda run -n t3_env`, which runs *that* environment's activation hooks too.
Naming the interpreter directly is not equivalent: ARC needs Open Babel, whose
conda package exports `BABEL_LIBDIR` and `BABEL_DATADIR` from an activation
hook, and without them Open Babel registers no plugins and `import arc` — so
`import t3` — fails.

`RMG_PATH` and `RMG_DB_PATH` are read by ARC's settings module, which is where
T3 gets the database location. Without them T3 dies in its constructor before
running an iteration.

## Checking it worked

```bash
uv run --env-file carmel-tools.env python -c "from carmel.adapters.t3 import is_t3_importable; print(is_t3_importable())"
```

`True` means Carmel can reach a working T3.

If it prints `False`, ask the tools themselves what is wrong — the probe
deliberately answers only yes or no:

```bash
conda run -n t3_env --no-capture-output python -c "import t3"
```

## Troubleshooting

**`conda is not on PATH`** — install [Miniforge](https://conda-forge.org/download/)
and open a new shell.

**`uv is not on PATH`** — install [uv](https://docs.astral.sh/uv/getting-started/installation/)
and open a new shell.

**Coming from the old conda `crml_env`?** Carmel no longer uses it. Run
`make install-carmel` (or `make install-dev`) and use `uv run`. The old
environment is not deleted; remove it with `conda env remove -n crml_env` once
nothing of yours points at it.

**`import arc` fails on `arc.molecule.graph`** — ARC's Cython extensions were
not compiled. `make install-t3` rebuilds them; the editable install is what
compiles them, and it is re-run every time for exactly this reason.

**RMG-Py's build fails** — it needs a C/C++ toolchain. Install it, then re-run
`make install-rmg`; nothing already built is redone.

**A conda environment is corrupt** — delete it (`conda env remove -n t3_env`)
and re-run `make install`. The installer creates whatever is missing and leaves
the rest alone.
