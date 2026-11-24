# Reproduction utilities

The scripts in this directory are bounded analysis, verification and
benchmarking utilities. They are not executed by the GitHub release checks.
Commands that consume historical ledgers or prepared inputs require those
artifacts to be supplied locally; no replacement data are fabricated.

## Safe inspection

Use the active Python environment, or add `conda run -n <name>` when Conda is
preferred:

```powershell
python -m src.pipeline_runner --help
python -m src.provenance_cli verify --help
python reproduction/figures.py --help
```

## Manifest-backed analysis

After a completed run, point each utility at the same run-local artifacts:

```powershell
python -m src.provenance_cli verify `
  --manifest-db reports/runs/full/<run-id>/benchmark.db `
  --ledger reports/runs/full/<run-id>/results.jsonl `
  --run-id <run-id>

python -m src.stats_analysis --sensitivity `
  --manifest-db reports/runs/full/<run-id>/benchmark.db `
  --ledger reports/runs/full/<run-id>/results.jsonl `
  --run-id <run-id>
```

## Figure generation

The canonical figure generator accepts explicit evidence and output paths:

```powershell
python reproduction/figures.py `
  --ledger reports/benchmark_runs/combined_results_20261122.jsonl `
  --output-dir paper/figures
```

The ledger and audit inputs are large local artifacts and are intentionally
not included in the reviewer release. Figure generation is therefore not
claimed to be runnable from a clean clone without external evidence.

The full benchmark has a different prerequisite: the compact verified split
eligibility contract is committed at
`config/eligibility/verified_eligibility.json`. The launcher verifies that
contract, then acquires the capped raw datasets before dispatching work.

## Bounded engineering probes

The performance and warm-model utilities require verified prepared inputs and
write only to a new output directory:

```powershell
python -B reproduction/benchmark_corrected.py `
  --source-root . `
  --output reports/performance/new_probe `
  --production
```

Inspect each script's `--help` output before running it. Some probes execute
models against prepared inputs and are not part of the non-training release
verification.

## Public evidence

The compact, sanitized evidence package is [`../report/`](../report/). It
contains coverage, statistics, FSVA summaries, candidate-selection summaries,
dataset provenance and hashes. Raw ledgers, databases, caches, histories,
manuscripts and reviewer correspondence are excluded.
