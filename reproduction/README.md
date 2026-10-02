# Bounded reproduction package

These commands use the supported Conda P12 environment and explicit limits.
They do not download the benchmark collection or launch the full grid.

```powershell
conda run -n P12 python -m src.pipeline_runner `
  --dry-run-manifest --max-datasets 2 --max-seeds 1 --max-folds 1 `
  --max-conditions 2 --max-workers 1
```

For an execution smoke with two pipelines and one estimator, add local
synthetic CSVs under `data/raw/` and run:

```powershell
conda run -n P12 python -m src.pipeline_runner `
  --max-datasets 2 --max-seeds 2 --max-folds 1 --max-conditions 2 `
  --pipelines Raw,AutoFE_Baseline --models logistic_regression `
  --max-workers 1 --stop-after-tasks 16 --task-timeout-seconds 30 `
  --max-attempts 1
```

Inspect a run without changing it:

```powershell
conda run -n P12 python -m src.provenance_cli verify `
  --manifest-db reports/manifests/<run_id>.db `
  --ledger reports/tables/<ledger>.jsonl --run-id <run_id>
```

Run incomplete-run sensitivity and provenance packaging after a manifest-backed
run:

```powershell
conda run -n P12 python -m src.stats_analysis --sensitivity `
  --manifest-db reports/manifests/<run_id>.db `
  --ledger reports/tables/<ledger>.jsonl --run-id <run_id>

conda run -n P12 python -m src.provenance_cli package `
  --manifest-db reports/manifests/<run_id>.db `
  --ledger reports/tables/<ledger>.jsonl --run-id <run_id> `
  --output-dir reports/provenance/<run_id>
```

The package contains metadata, hashes, schema/status records, lineage, and
commands only. It excludes datasets, caches, benchmark outputs, credentials,
and machine-specific installation prefixes.
