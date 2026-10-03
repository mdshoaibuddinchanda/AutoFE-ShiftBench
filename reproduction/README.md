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

## Corrected-reference performance comparisons

The frozen reference is commit `913285b`; retained local copies are under ignored
`reports/references/`. Run the current harness against each source root into a
new output directory. These fixtures are synthetic and bounded; existing
measurement directories cannot be overwritten.

```powershell
conda run -n P12 python -B reproduction/benchmark_corrected.py --source-root . --output reports/performance/new_probe --production
conda run -n P12 python -B reproduction/benchmark_corrected.py --source-root . --output reports/performance/new_all_pipeline_probe --pipelines all
conda run -n P12 python -B reproduction/compare_corrected.py reports/performance/corrected_reference reports/performance/new_probe --output reports/performance/new_equivalence.json
```

`benchmark_remaining.py` measures guarded selected-input loading, exact indexed
membership and fresh scoped hashing. `benchmark_warm_models.py` runs four actual
models twice against already verified prepared inputs. `verify_prepared_predictions.py`
replays recorded estimator seeds and exact prepared dependencies to compare labels
and probabilities. `verify_thread_policy.py` retains the rejected numerical thread
rewrite evidence. See the engineering record for commands, limitations, warm-model
dependency setup, and unchanged diagnostic/resampling budgets.
