# Architecture

## Design Principles

- **Deterministic code first, LLM second** — prefer algorithmic solutions over language model calls.
- **Bounded autonomy** — agents operate within predefined action sets, not free-form reasoning.
- **Typed contracts** — all data flows use validated schemas (pydantic).
- **Full provenance** — every action is recorded with inputs, outputs, versions, and costs.
- **Human-in-the-loop governance** — expensive compute and wet-lab actions require approval.

## Phase 0 — Current Architecture

Phase 0 establishes the foundational infrastructure:

```
Carmel.py                    # CLI entrypoint (repo root)
carmel/
├── __init__.py          # Package root, version export
├── version.py           # Single source of truth for version string
├── config.py            # Pydantic config models, YAML loading and validation
├── paths.py             # Path utilities, workspace directory initialization
└── logger.py            # Centralized logger configuration
```

### Config System

Configuration uses pydantic `BaseModel` subclasses with `extra="forbid"` for strict validation. The `CarmelConfig` model validates workspace settings loaded from YAML files. Validators normalize logging level casing and expand tilde in paths.

### Workspace Layout

`init_workspace()` creates a standard directory structure for campaign data:

| Directory      | Purpose                                            |
|----------------|----------------------------------------------------|
| `benchmarks/`  | Curated benchmark bundles and credence records     |
| `evidence/`    | Literature memos, extracted records, source links  |
| `models/`      | Generated mechanism versions and diffs             |
| `provenance/`  | Hashes, versions, tool settings, costs             |
| `reports/`     | Final and intermediate reports                     |
| `runs/`        | Executed tool runs and statuses                    |

### Logging

Centralized through a `carmel` namespace logger with configurable level and optional file output. Child loggers inherit configuration via `get_logger()`.

### CLI

`Carmel.py` at the repo root, following ARC/T3 convention. Argparse-based with three subcommands: `version`, `validate-config`, `init-workspace`. The `main()` function accepts an optional `argv` list for testability.

## Phase 1+ — Planned Architecture

### Agent Ensemble

| Agent                | Role                                              |
|----------------------|---------------------------------------------------|
| Planner              | Campaign state, budget management, action ranking  |
| Literature Agent     | RAG over curated corpus, typed evidence memos      |
| Data Agent           | Benchmark normalization, credence scoring          |
| Revision Router      | Discrepancy-to-action mapping (bounded action set) |
| X-Design Agent       | Deterministic experiment design generation         |
| Execution Controller | Job submission, provenance, approval enforcement   |
| Reporting Agent      | Plan summaries, approval memos, reports            |

### External Tool Integration

| Tool    | Trust Level              |
|---------|--------------------------|
| T3      | Trusted                  |
| RMG     | Trusted with caution     |
| ARC     | Trusted                  |
| Cantera | Trusted                  |
| TCKDB   | Trusted                  |

### Campaign Artifacts

Each campaign workspace will additionally contain:

- `campaign.yaml` — canonical structured input
- `campaign.md` — human-readable summary
- `preferences.md` — approval thresholds and policies
- `plan.md` — current proposed actions
- `approvals.md` — human approvals and notes
- `decision_log.jsonl` — append-only decision stream

### Hard Constraints

- One top-level planner only
- No dynamic agent spawning
- No free-form tool invocation
- Typed schemas for every tool call
- Literature never writes directly into the model
- All expensive actions gated by budget checks
- All high-stakes actions gated by HITL policy
- Append-only decision log
- Full provenance for every action
