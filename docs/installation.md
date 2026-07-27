# Installation

```bash
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel
make install
conda activate crml_env
carmel --help
```

That is the whole thing. `make install` clones the external chemistry stack,
builds the three conda environments Carmel needs, installs Carmel, and records
where everything ended up.

Budget around 40 minutes and 15 GB the first time — most of it compiling
RMG-Py. Re-running when nothing has changed takes under a minute: the conda
environments and the RMG-Py build are skipped, and what remains is re-installing
Carmel, ARC and T3 in editable mode.

## Requirements

- **conda**, from [Miniforge](https://conda-forge.org/download/) or Anaconda.
  `conda` has to be on your `PATH`; nothing else does.
- **git**, and a C toolchain for the Cython extensions RMG-Py and ARC compile
  (`build-essential` on Debian/Ubuntu, Xcode command line tools on macOS).
- Roughly 15 GB of disk, mostly RMG-database and the three environments.

## What gets installed

Carmel drives four external tools, and they do not fit in one environment:
their Python requirements are mutually exclusive. So there are three
environments, and they talk to each other as **separate processes**, never as
imports.

| Environment | Holds | Python | Who launches it |
| --- | --- | --- | --- |
| `rmg_env` | RMG-Py, Arkane | `>=3.9,<3.12` | T3, as a subprocess |
| `t3_env` | T3 and ARC together | `=3.14` | Carmel, as a subprocess |
| `crml_env` | Carmel itself | `>=3.14` | you |

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
| `make install-carmel` | Create `crml_env`, install Carmel, write the activation hook |
| `make install-dev` | Carmel and its dev dependencies into the **current** environment, nothing else |

`make install-dev` is the one to use when you are working on Carmel's own code
and have no interest in running a real campaign — it is what the lint and test
CI lanes run, and it takes about 20 seconds.

The scripts behind these targets live in `devtools/`. `make install` is
`devtools/install_all.sh`, which calls the others in order.

## Re-running is safe

Every step checks what is actually on disk before doing anything: does this
conda environment exist, can this environment import a compiled RMG module, is
this repository already checked out. Nothing is gated on a flag handed in from
outside saying a previous step "should have" run.

So `make install` is safe to run whenever — after `git pull`, after a failed
install, on a machine where half of it is already there. It picks up exactly
where the disk says it is, and re-installs Carmel itself (which is cheap and
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
| `CARMEL_ENV` | `crml_env` | Name of Carmel's environment |

For instance, to build on top of checkouts you keep in `~/Code`:

```bash
CARMEL_STACK_ROOT="$HOME/Code" make install
```

## The activation hook

`make install-carmel` writes
`$CONDA_PREFIX/etc/conda/activate.d/carmel.sh` into `crml_env`, exporting
`T3_CONDA_ENV`, `T3_PATH`, `RMG_PATH` and `RMG_DB_PATH`. So:

```bash
conda activate crml_env
```

is all the setup there is — no lines to add to `.bashrc`, and the settings go
away with the environment.

There is deliberately no matching `deactivate.d` hook to unset them. Carmel
launches T3 with `conda run -n t3_env`, and `conda run` **deactivates the
current environment first** — so a hook that unset these variables would remove
them from T3's environment at exactly the moment T3 reads them, and T3 dies in
its constructor on a `None` database path. Four inert path variables outliving
`conda deactivate` is the cheaper trade.

`T3_CONDA_ENV` is the important one. Carmel launches T3 with
`conda run -n t3_env`, which runs *that* environment's activation hooks too.
Naming the interpreter directly is not equivalent: ARC needs Open Babel, whose
conda package exports `BABEL_LIBDIR` and `BABEL_DATADIR` from an activation
hook, and without them Open Babel registers no plugins and `import arc` — so
`import t3` — fails.

`RMG_PATH` and `RMG_DB_PATH` are read by ARC's settings module, which is where
T3 gets the database location. Without them T3 dies in its constructor before
running an iteration.

The hook is skipped when there is no T3 checkout, so a Carmel-only environment
is not handed paths that do not exist.

## Checking it worked

```bash
conda activate crml_env
python -c "from carmel.adapters.t3 import is_t3_importable; print(is_t3_importable())"
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

**`import arc` fails on `arc.molecule.graph`** — ARC's Cython extensions were
not compiled. `make install-t3` rebuilds them; the editable install is what
compiles them, and it is re-run every time for exactly this reason.

**RMG-Py's build fails** — it needs a C/C++ toolchain. Install it, then re-run
`make install-rmg`; nothing already built is redone.

**A conda environment is corrupt** — delete it (`conda env remove -n t3_env`)
and re-run `make install`. The installer creates whatever is missing and leaves
the rest alone.
