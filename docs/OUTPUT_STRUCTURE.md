# Output structure

The repository keeps source code and publication assets separate from local
scientific outputs. The launcher creates new run directories; historical
folders are not moved automatically.

## Data and cache

- Downloaded, capped datasets: `data/raw/<dataset>.csv` and
  `data/raw/<dataset>_meta.json`.
- Versioned feature/preprocessing cache:
  `data/cache/<protocol>/<seed-scheme>/<dataset>/`.
- `AUTOFE_CACHE_BASE` may override only the cache base. Cache protocol and
  seed namespaces are unchanged.

The committed `config/eligibility/verified_eligibility.json` is a small,
hash-pinned prerequisite for a non-plan full run. It contains verified split
eligibility and dataset identity metadata, not raw data. The launcher checks
its SHA-256 before dispatch; use `-EligibilityPath` only for an independently
audited replacement with the same hash.

## New experiments

Full runs default to:

```text
reports/runs/full/<run-id>/
├── benchmark.db
├── manifest.jsonl
├── results.jsonl
├── logs/benchmark.log
├── logs/workers/
└── metadata/launcher_config.json
```

Pilot runs default to the equivalent `reports/runs/pilot/<run-id>/` layout,
with `pilot.db`, `pilot_manifest.jsonl`, `pilot_results.jsonl` and
`logs/pilot.log`.

Use stable identifiers:

```powershell
.\run_project.ps1 -RunId experiment_001
.\run_project.ps1 -RunId experiment_001 -Resume
.\pilot_run.ps1 -RunId pilot_001
```

The launchers use the active `python` executable by default and do not assume
the original developer's Conda environment. Use `-CondaEnv <name>` to run
through Conda, or `-PythonExe <path-or-command>` to select another Python.
Explicit `-CondaEnv` takes precedence over `-PythonExe`.

Repeating a run ID reopens the same SQLite manifest. The manifest rejects a
different protocol, seed scheme, dataset identity, grid, or task manifest
under the same ID. `-RunRoot` remains supported and is honored exactly when
provided; it is useful for a separately mounted output drive.

## Analysis and figures

Canonical post-processing remains separate from run ledgers:

- compact analysis summaries: `reports/analysis/{statistics,fsva,candidate_selection,coverage}/`;
- audit runs: `reports/audits/<audit-run-id>/`;
- figure inputs: `reports/figure_inputs/`;
- publication figures: `paper/figures/`, generated only by `reproduction/figures.py`.

The small GitHub evidence package is `report/`. It contains verified compact
derivatives, not databases, raw data, caches, or the large history corpus.

The canonical final audit remains under
`reports/revision_final/run_20261123T175648Z/`; it is local scientific
evidence, not a replacement for the compact public package.

## Historical material and cleanup

Existing folders such as `reports/benchmark_runs/`, `reports/revision_final/`,
`reports/full_benchmark_20261003/`, and preflight/provenance archives remain
in place as historical evidence. Do not move or delete them without a fresh
hash/dependency audit. `reports/` and local data/cache directories are ignored
by Git; only `report/` and approved publication figures are intended for
ordinary staging.
