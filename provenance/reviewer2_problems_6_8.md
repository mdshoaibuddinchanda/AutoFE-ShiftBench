# Code revision record: reviewer problems 6–8

Date: 2026-10-02

Scope: dataset-level statistical analysis, complete task manifests/failure accounting, and reproducible stopping/checkpointing/recovery.

This stage continues the corrected protocol from [reviewer1_problems_1_2.md](reviewer1_problems_1_2.md):
`predictor_only_geometry_v2_baselines_operators_fsva_v1` with seed scheme
`sha256_canonical_json_u32_v1`. The full corrected benchmark remains outside
scope.

## Audit coverage

- Verified branch, remote, ancestry, clean starting tree, Conda `P12`, and the existing 33-test regression state before this stage.
- Read the active runner, checkpoint, progress, statistics, plotting/table readers, notebooks, configuration, revision record, and all focused tests. The checkout contains no dataset CSVs, result ledger, manifest, checkpoint database, or failure log to audit; generated output directories are ignored.
- The prior runner appended JSONL results and logged a completed hash to SQLite, but had no intended-task manifest, attempt history, durable-result uniqueness, failure state, timeout containment, or dataset-cluster inferential contract. Existing `stats_analysis.py` and table helpers pooled fold/replicate rows.

## Problem 6 — dataset-level statistical analysis

`src/dataset_statistics.py` defines `AnalysisConfig` (`dataset_equal_paired_v1`) and the active `run_dataset_level_analysis` reader.

- Estimand: mean `roc_auc` difference `pipeline_b - pipeline_a`, default `AutoFE_Baseline - Raw`.
- Pairing key: dataset, split policy, seed, fold, condition, severity, and estimator, with protocol and seed-scheme identities included. Pipeline identity stays on each side of the pair.
- Split policy, condition, and estimator are separate strata. Valid task differences are averaged equally within dataset, then datasets receive equal weight.
- Uncertainty uses deterministic whole-dataset bootstrap percentile intervals. The paired hypothesis procedure is a dataset-level sign-flip test: exact enumeration through 16 datasets, deterministic finite Monte Carlo above that threshold, with the symmetric-difference exchangeability assumption recorded.
- Holm correction is applied over the declared family of stratum contrasts. Raw and adjusted p-values, family ID/size, decisions, dataset counts, task counts, contributing datasets, exclusions, and task IDs are persisted.
- Outputs include `analysis_config.json`, `task_pairs.jsonl`, `dataset_contrasts.csv`, `stratum_summaries.csv`, `exclusions.jsonl`, and `multiplicity.csv` under the requested analysis directory. The input ledger SHA-256 fingerprint is recorded.
- Duplicate successful scientific tasks, incompatible protocol/seed mixes, incompatible pipeline semantics, missing partners, failed/undefined/nonfinite metrics, too few datasets, and all-zero effects have explicit outcomes.

The prior pooled Wilcoxon/Friedman helpers remain available only as legacy exploratory functions; corrected active readers route through the dataset-level contract. No empirical manuscript claims were generated.

## Problem 7 — task manifests and failure accounting

`src/task_manifest.py` defines manifest schema `task_manifest_v1`, scientific identity `scientific_task_identity_v1`, run IDs, task states (`pending`, `running`, `completed`, `failed`, `timeout`, `skipped`), append-only attempt records, durable result identity, dependency propagation, and reconciliation.

- `build_task_records` constructs precompute and model tasks before execution, including missing-dataset skips and precompute dependencies. Scientific task IDs are canonical hashes independent of enumeration order.
- `ManifestStore` persists run configuration, payloads, current state, attempts, exception details, timeout details, worker identity, result references, and checkpoint identity in SQLite. A separate JSONL manifest export is written before expensive work.
- A precompute cache or FSVA artifact is a dependency artifact; only an authoritative model result commits a model task as completed. Dependency failures explicitly skip affected model tasks.
- `check_progress.py` reports manifest task and attempt states separately from cache/artifact counts. The latter are never treated as completed model-task counts.
- `--dry-run-manifest` creates and validates the complete grid without precompute, model fitting, or dispatch.

## Problem 8 — stopping, checkpointing, and recovery

`src/pipeline_runner.py` now persists execution controls (`max_workers`, task timeout, run wall time, maximum attempts, retryable failure classes, stop-after-task limit), handles signal/declarative stopping, and supports a killable per-task timeout path. Workers claim attempts transactionally; stale workers cannot commit after a newer attempt owns the task. Result writing fsyncs JSONL before committing the manifest result and checkpoint; duplicate deliveries are idempotent and conflicting payloads raise an integrity error.

`ManifestStore.reconcile_result_ledger` handles durable-result/completion-marker crash windows, malformed/truncated JSONL records, conflicts, and completed tasks lacking durable payloads. Stale running attempts can be recovered to pending or terminal failure while preserving attempt history.

## Verification

Final command results, bounded smoke counts, injected failures/timeouts, and artifact sizes will be appended after the final P12 run. Problems 9–10 remain deferred; this stage records only the minimum run identity, coverage, compatibility, and analysis traceability dependencies needed for problems 6–8.
