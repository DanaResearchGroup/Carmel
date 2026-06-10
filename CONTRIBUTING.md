# Contributing to Carmel

Thanks for your interest in improving Carmel. This document explains how to
contribute and the legal terms under which contributions are accepted.

## License

Carmel is released under the **Apache License 2.0** (see [LICENSE](LICENSE)).
By contributing, you agree that your contributions will be licensed under the
same terms, and you additionally agree to the Contributor License Agreement
described below.

## Contributor License Agreement (CLA)

Before we can merge your first contribution, you must sign Carmel's
**Contributor License Agreement** (see [CLA.md](CLA.md)):

- **Individuals** sign the Individual CLA.
- **Contributing on behalf of an employer** (or where your employer may have
  rights to your work) requires the Corporate CLA, signed by someone authorized
  to bind the organization.

**Why a CLA, and not just the Apache license?** The Apache 2.0 license governs
what *downstream users* may do. The CLA governs the rights *you grant to the
project* when you contribute. It does two things the bare license does not:

1. It confirms you actually have the right to contribute the code (you wrote it,
   or you're authorized to submit it), which protects every downstream user.
2. It grants the Dana Research Group a broad copyright and patent license to your
   contribution, including the right to relicense. This keeps the project's
   long-term licensing options open — Carmel follows an open-core model in which a
   separate, commercially-licensed extension (`Carmel-Pro`) may be offered
   alongside this open-source core. The CLA is what makes that model legally clean
   without disadvantaging the open-source community: the core stays Apache 2.0 for
   everyone.

How to sign: comment on your first pull request with the statement indicated in
[CLA.md](CLA.md), or follow whatever CLA-bot flow the repository has configured.
A maintainer will confirm before merge.

### Lightweight alternative: DCO sign-off

For small fixes, maintainers may accept a [Developer Certificate of
Origin](https://developercertificate.org/) sign-off in lieu of the full CLA for
that contribution. Add a `Signed-off-by: Your Name <you@example.com>` line to
each commit (`git commit -s`). Substantial contributions still require the CLA.

## Scope: the Core / Pro boundary

Carmel is the **open-source core**. A separate private repository (`Carmel-Pro`)
holds commercial extensions that plug into the core through the typed contracts in
`carmel/contracts`. Two rules keep the boundary clean:

- **The core never depends on Pro.** Core code may import only from within the
  `carmel` package (including `carmel.contracts`); it must never import a `Pro`
  package. CI enforces this.
- **Contribute commercial-extension code to Pro, not to the core.** If a proposed
  change is specific to a commercial deployment (multi-tenant ops, customer
  adapters, proprietary data integrations), it belongs in Pro. When in doubt, open
  an issue and ask before writing code.

## Development workflow

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate crml_env

# Editable install with dev dependencies
make install
```

Before opening a pull request:

```bash
make check     # runs tests (with coverage), lint, and type checks
make format    # auto-fix formatting
```

Standards enforced in CI:

- **Tests** — `pytest` with coverage; new code needs tests (project gate is 90%).
- **Lint/format** — `ruff` (line length 120).
- **Types** — `mypy --strict` with the pydantic plugin.

## Pull request process

1. Open an issue first for anything non-trivial, so design can be discussed before
   you invest time.
2. Branch from the active development branch. Keep each PR to one logical change.
3. Ensure `make check` passes locally.
4. Sign the CLA (or DCO-sign your commits) on the PR.
5. A maintainer will review. Address feedback by pushing follow-up commits.

## Reporting bugs and requesting features

Use the GitHub issue tracker. For bugs, include: what you ran, what you expected,
what happened, and a minimal reproducible example or config where possible.

## Questions

Open a discussion or issue on
[github.com/DanaResearchGroup/Carmel](https://github.com/DanaResearchGroup/Carmel).
