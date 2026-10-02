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

`src/pipeline_runner.py` now persists execution controls (`max_workers`, task timeout, run wall time, maximum attempts, retryable failure classes, stop-after-task limit, and stale-attempt recovery threshold), handles signal/declarative stopping, and supports a killable per-task timeout path. Resume invokes stale-attempt recovery before dispatch. Workers claim attempts transactionally; stale workers cannot commit after a newer attempt owns the task. Result writing fsyncs JSONL before committing the manifest result and checkpoint; duplicate deliveries are idempotent and conflicting payloads raise an integrity error.

`ManifestStore.reconcile_result_ledger` handles durable-result/completion-marker crash windows, malformed/truncated JSONL records, conflicts, and completed tasks lacking durable payloads. Stale running attempts can be recovered to pending or terminal failure while preserving attempt history.

## Verification

Final command results, bounded smoke counts, injected failures/timeouts, and artifact sizes will be appended after the final P12 run. Problems 9–10 remain deferred; this stage records only the minimum run identity, coverage, compatibility, and analysis traceability dependencies needed for problems 6–8.

## Final verification record

All implementation and verification commands used `D:\\Conda\\P12\\python.exe`; the repository `.venv` was retained untouched.

- `python -m unittest discover -s tests -v`: **39 tests passed** in 8.5 seconds. This includes the prior problems 1–5 regression coverage, dataset-level numerical fixtures, manifest state/failure fixtures, stale-worker fencing, timeout containment, malformed-ledger reconciliation, changed-run rejection, and planned-skip accounting.
- `python -m compileall -q -f src tests`: completed with no compile errors. The two existing `invalid escape sequence \\D` warnings in `src/analysis/structural_analysis.py` remain unrelated to this stage. `git diff --check` passed.
- Manifest-only dry run with two configured (unavailable) datasets, one seed, one fold, and two conditions produced **564 intended tasks**, all explicitly `skipped: dataset_unavailable`, with no precompute, model, or dispatch work.
- Bounded production-path smoke (`smoke68verify`) used two synthetic datasets, two seeds, one fold, clean plus `gaussian_noise_0.01`, `Raw` and `AutoFE_Baseline`, and `logistic_regression`, with one worker, 30-second task timeout, one attempt, and a 16-model dispatch limit. It built **24 intended tasks** (8 precompute + 16 model), completed **24 tasks** (8 precompute + 16 model), recorded **24 attempts** and **24 durable results**, and wrote **16 authoritative model rows**. There were no failures, timeouts, skips, or pending tasks. The ledger contained both pipelines, both conditions, and both synthetic datasets. Re-running the same run ID added zero result rows and zero attempts.
- Feeding that authoritative ledger to the active corrected reader with 100 bootstrap and 100 sign-flip resamples produced two separate strata (clean and Gaussian), eight paired task records, two contributing datasets per stratum, and zero exclusions. No empirical manuscript conclusion is claimed.
- Earlier stop-limit smoke used two datasets and two conditions with the default full pipeline grid: **564 intended**, **6 completed** (2 precompute + 4 model), **6 attempts**, **6 durable results**, **4 model rows**, and **558 pending** after the declared stop limit. This confirms pending work is retained separately from completed estimator tasks.
- Controlled fault fixtures demonstrated: dependency failure propagates to explicit downstream skips; failed attempts retry without creating a second scientific observation; timeout termination returns before the child finishes; stale attempts cannot commit after takeover; duplicate delivery is idempotent; conflicting payloads raise an integrity error; truncated JSONL lines are counted and valid rows reconciled; and incompatible run configuration/task grids are rejected.

## Scope boundary

The implementation adds only the coverage, identity, compatibility, and analysis traceability dependencies required by problems 6–8. Problems 9 (formal incomplete-run sensitivity analysis) and 10 (full dataset/code/environment/cache/analysis provenance package) remain open. The checkout has no full dataset collection or historical result ledger, so the manuscript’s historical findings require a future corrected benchmark recomputation.
