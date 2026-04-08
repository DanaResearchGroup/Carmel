# Carmel

```
        ██████╗ █████╗ ██████╗ ███╗   ███╗███████╗██╗
        ██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝██║
        ██║     ███████║██████╔╝██╔████╔██║█████╗  ██║
        ██║     ██╔══██║██╔══██╗██║╚██╔╝██║██╔══╝  ██║
        ╚██████╗██║  ██║██║  ██║██║ ╚═╝ ██║███████╗███████╗
         ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝
             Agentic Predictive Chemical Kinetics Engine
```

**Closed-loop campaign manager for predictive chemical kinetics.**

Carmel automates the iterative cycle of building, validating, and
revising predictive chemical kinetic models. It orchestrates simulation
tools, experiment design, and model revision through a deterministic
service layer with full provenance tracking and human-in-the-loop
governance.

## Phase 1 Capabilities

- **Structured campaign intake** through a local Flask UI
- **Canonical campaign artifacts** persisted as YAML/JSON files
- **Deterministic planner** producing a baseline T3 handshake action
- **Approval policy engine** with budget-based auto-approval thresholds
- **Real T3 subprocess adapter** that normalizes output into typed `DiagnosticsV1`
- **Lifecycle state machine** with explicit transitions
- **Append-only decision log** and per-action provenance records
- **Graphical compute-selection view** rendered as deterministic SVG artifacts
- **Free-text intake** as an advisory-only review path

## Quick Start

```bash
# Clone
git clone https://github.com/DanaResearchGroup/Carmel.git
cd Carmel

# Install
conda env create -f environment.yml
conda activate crml_env
make install

# Launch the UI
carmel serve --workspaces ./workspaces
```

Then open http://127.0.0.1:5000.

## Configuration

A campaign is defined by a structured input. The minimum is:

```yaml
workspace_name: ethanol-combustion
initial_mixture:
  components:
    - species: C2H5OH
      mole_fraction: 0.05
    - species: O2
      mole_fraction: 0.20
    - species: N2
      mole_fraction: 0.75
target_observables:
  - name: ignition_delay
target_reactor_systems:
  - reactor_type: jsr
    temperature_range_K: [800, 1200]
    pressure_range_bar: [1.0, 5.0]
    residence_time_s: 1.0
budgets:
  cpu_hours: 20.0
  experiment_budget: 0.0
```

The Flask UI provides a structured form that maps to this schema. The
canonical artifact lives at `<workspace>/campaign.yaml`.

## Workspace Structure

`carmel init-workspace` (and `create_campaign`) create:

```
my-campaign/
├── campaign.yaml               # Canonical structured input
├── approval_policy.yaml        # Active approval policy
├── campaign_state.json         # Current lifecycle state
├── plan.json / plan.md         # Current plan (canonical + rendered)
├── decision_log.jsonl          # Append-only decision stream
├── diagnostics.json            # Normalized T3 output (when available)
├── benchmarks/  evidence/  models/  provenance/  reports/  runs/
```

## CLI

| Command                       | Purpose                              |
|-------------------------------|--------------------------------------|
| `carmel version`              | Print version                        |
| `carmel validate-config FILE` | Validate a config file               |
| `carmel init-workspace DIR`   | Initialize a workspace scaffold      |
| `carmel serve`                | Launch the local Flask UI            |
