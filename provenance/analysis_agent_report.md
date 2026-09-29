# Analysis-agent audit: result schemas, inference, and incomplete-run accounting

**Audit date:** 2026-09-29 (Asia/Calcutta)  
**Environment:** `D:\Conda\p12` / Python 3.12.14  
**Scope:** schema and statistical-method audit only. No full corrected benchmark was started.

This report is an analysis specification and an evidence record for the corrected campaign. It does not recalculate or replace the historical paper tables. The historical ledger and paper assets remain a separate, immutable evidence set.

## Files inspected

- [`src/stats_analysis.py`](../src/stats_analysis.py): current dataset-level Wilcoxon implementation and four Holm-adjusted Raw-versus-AutoFE contrasts.
- [`src/generate_tables.py`](../src/generate_tables.py): dataset aggregation and publication-table helpers.
- [`src/pipeline_runner.py`](../src/pipeline_runner.py): corrected task, phase, manifest, and result-ledger schemas.
- [`src/checkpoint.py`](../src/checkpoint.py): run-scoped phase checkpoint schema.
- [`src/evaluation.py`](../src/evaluation.py): metric names and undefined-metric behavior.
- [`reports/tables/results_stream.jsonl`](../reports/tables/results_stream.jsonl): historical ledger, inspected line by line.
- [`reports/tables/results_stream_backup.jsonl`](../reports/tables/results_stream_backup.jsonl): historical backup, counted and parsed but not combined with the main ledger.
- [`provenance/original_run.json`](original_run.json): historical hashes, counts, and immutability classification.
- [`provenance/corrected_smoke_manifest.json`](corrected_smoke_manifest.json): synthetic corrected-run example, not benchmark evidence.
- [`provenance/reviewer1_run_plan.md`](reviewer1_run_plan.md): frozen two-track and acceptance-gate plan.

The focused checks used the existing `p12` environment:

```text
D:\Conda\p12\python.exe -m pytest -q tests/test_leakage_controls.py \
  -k "primary_statistics_use_dataset_as_unit_and_tables_mark_missing_conditions or failure_accounting_records_phase1_and_phase2_failures"
2 passed, 17 deselected
```

The current statistics function was also run against the historical ledger only as an audit probe. It returned 23 finite datasets after filtering and yielded four contrasts. Those numbers are not corrected-run results and must not be copied into a corrected result note.

## Observed historical ledger

The configured design has 25 datasets, 5 seeds, 5 folds, 14 historical conditions, 7 pipelines, and 10 models: **612,500 intended task rows**. The main historical stream contains **560,002 JSON records**, all with `status == "success"`, from **24 datasets**. Every line parsed as JSON and the result key schema was constant at 31 fields.

| Item | Observed value | Audit meaning |
|---|---:|---|
| Intended grid | 612,500 | `25 × 5 × 5 × 14 × 7 × 10` |
| Main historical rows | 560,002 | 91.4289% of the intended grid |
| Historical datasets represented | 24 | `aps_failure` is absent |
| Historical failure rows | 0 | Failures were not represented in this ledger |
| Historical pending/skipped/timeout rows | 0 | Missing rows are right-censored, not classified |
| `covertype` rows | 10,796 | 13,704 tasks absent |
| `kddcup99` rows | 10,234 | 14,266 tasks absent |
| `wine-quality-red` rows | 24,493 | 7 tasks absent |
| `dry-bean-dataset` rows | 24,479 | 21 tasks absent |
| `aps_failure` rows | 0 | 24,500 tasks absent |

The missing-row total is 52,498, which reconciles exactly to the intended grid: 24,500 + 13,704 + 14,266 + 7 + 21. The historical backup contains 240,469 valid JSON records from 11 datasets. It is a preserved backup, not an additional replicate and must not be appended to the main stream.

The historical row schema contains dataset, seed, fold, condition, pipeline, model, status, timing, feature-count, memory, distribution-distance, and metric fields. It does **not** contain a run ID, task key, phase, source or metadata checksum, code/configuration/runtime fingerprint, split policy, group ID, or failure reason. It cannot therefore support corrected-run provenance or row/group side-by-side accounting.

### Historical metric validity

The 560,002 historical rows are marked successful, but success does not imply a finite value for every metric:

| Field | Null/nonfinite rows |
|---|---:|
| `roc_auc` | 24,859 |
| `test_auc` | 24,859 |
| `pr_auc` | 79,720 |
| `log_loss` | 150 |
| `brier_score` | 150 |
| `train_auc` | 150 |
| `wasserstein` | 123,573 |
| `ks_stat` | 123,573 |

The current [`_read_results`](../src/stats_analysis.py) drops nonfinite `roc_auc` rows and then pairs available task rows. It does not require a complete expected cell set, report per-dataset coverage, or distinguish a failed task from an undefined metric. When probed on the historical stream, it retained 23 datasets: incomplete `covertype` was included and `kddcup99` had no finite primary ROC-AUC rows. This is a useful diagnosis of the current function, not a valid historical claim or corrected estimate.

The historical conditions also use names such as `covariate_shift`, `population_shift`, `class_prior_shift`, and `feature_removal_0.2`. These are not interchangeable with the corrected primary training-corruption conditions. The corrected analysis must canonicalize only an explicitly documented alias map and must preserve experiment scope in the result metadata.

## Prespecified corrected analysis

### Estimand and unit

The primary estimand is the expected difference in finite test ROC-AUC for a new benchmark dataset. A dataset is the independent unit. Folds, seeds, models, and conditions are repeated task cells used to form one dataset-level score; they are not independent observations in the inferential test.

The primary scope is the corrected training-corruption core:

```text
clean
gaussian_noise_0.01, gaussian_noise_0.05, gaussian_noise_0.10
missing_values_0.05, missing_values_0.10, missing_values_0.20
label_noise_0.05, label_noise_0.10, label_noise_0.20
```

The row-level and group-aware split policies are separate analysis tracks. Their task keys must include `split_policy`; a row-level result must never be silently compared with a group-aware result as if it were the same task.

### Cell pairing and dataset scores

For a fixed split policy, metric, pipeline, and dataset, define the expected cell key as:

```text
(seed, fold, condition, model)
```

The expected number of primary cells per dataset and pipeline is:

```text
N_expected_cells = |seeds| × |folds| × |primary_conditions| × |models|
```

For a Raw-versus-candidate contrast, first inner-pair both pipelines on the exact cell key and retain only finite metric values from both sides. Report all of the following for every dataset and contrast:

- `n_expected_cells`;
- `n_success_cells_raw` and `n_success_cells_candidate`;
- `n_finite_pair_cells`;
- `coverage = n_finite_pair_cells / n_expected_cells`;
- `dataset_score_raw` and `dataset_score_candidate`, each the equal-weight mean over the matched cells;
- `delta = dataset_score_candidate - dataset_score_raw`;
- failure and undefined-metric counts by condition, model, seed, and fold.

The confirmatory denominator is a declared complete-case rule: a dataset enters a primary contrast only when its matched finite-cell set equals the declared expected cell set for that contrast. If a campaign adopts a less strict threshold, that threshold must be frozen before analysis and reported as a sensitivity scope. Partial datasets remain visible in coverage tables and are not silently discarded.

### Primary contrasts and Holm families

The primary ROC-AUC family is four paired Raw-versus-candidate contrasts, using one dataset-level difference per dataset:

| Contrast ID | Pipeline A | Pipeline B | Family |
|---|---|---|---|
| `roc_auc_raw_baseline` | `Raw` | `AutoFE_Baseline` | `F1_primary_roc_auc` |
| `roc_auc_raw_mi` | `Raw` | `AutoFE_MI` | `F1_primary_roc_auc` |
| `roc_auc_raw_random` | `Raw` | `AutoFE_Random` | `F1_primary_roc_auc` |
| `roc_auc_raw_nomultiply` | `Raw` | `AutoFE_NoMultiply` | `F1_primary_roc_auc` |

For each contrast, report the number of complete paired datasets, mean and median dataset-level delta, wins/ties/losses using a frozen tolerance of zero for inferential differences, a two-sided exact Wilcoxon signed-rank p-value, and a two-sided 95% paired bootstrap interval for the mean delta. Use datasets as bootstrap units, a fixed published bootstrap seed, and a fixed replicate count of 10,000. If fewer than two complete datasets are available, p-value and interval are `NA`; if all differences are exactly zero, p-value is 1 and the interval is degenerate at zero.

Apply Holm step-down adjustment across the four finite p-values in `F1_primary_roc_auc`. The adjusted value must be reported beside its family ID and family size. Do not apply a second unannounced correction over folds, seeds, models, or individual rows.

Condition-specific and metric-specific estimates are descriptive unless an additional family is registered before the run. If condition-wise inference is required, use one declared family containing all 4 pipelines × 10 primary conditions (40 dataset-level contrasts), with its own family ID and Holm size 40. Accuracy, F1, log loss, Brier score, and PR-AUC should have estimates and finite-value coverage reported, but no p-values unless included in a predeclared family.

### Confidence-interval record

The interval record for each contrast must include:

```text
ci_method: "paired_dataset_bootstrap_percentile"
ci_level: 0.95
bootstrap_replicates: 10000
bootstrap_seed: <frozen integer>
ci_estimand: "mean_dataset_delta"
ci_low: <number or null>
ci_high: <number or null>
```

The interval is over datasets, not over task rows. A fold-level or row-level standard error is not a substitute for the dataset-level interval.

## Required result-note schema

The corrected campaign should emit one machine-readable result note alongside the task ledger. The following fields are required; field names are intentionally explicit so that a row-level run and a group-aware run can be compared without merging their denominators.

```json
{
  "artifact_type": "corrected_result_note",
  "run_id": "...",
  "protocol_version": "...",
  "code_commit": "...",
  "code_fingerprint": "...",
  "configuration_fingerprint": "...",
  "runtime_fingerprint": "...",
  "created_utc": "...",
  "historical_separation": {
    "historical_run_id": "original-main-...",
    "historical_ledger_sha256": "...",
    "combined_with_corrected": false
  },
  "split_policies": [
    {
      "split_policy": "row_level|group_aware",
      "definition": "...",
      "n_splits": 5,
      "fold_feasibility": "supported|infeasible|pending",
      "zero_shared_groups_asserted": true
    }
  ],
  "task_accounting": {
    "expected_tasks": 0,
    "counts_by_status": {
      "success": 0,
      "failed": 0,
      "skipped": 0,
      "timed_out": 0,
      "pending": 0
    },
    "counts_by_phase": {},
    "finite_metric_counts": {},
    "coverage_denominator": "declared expected task grid"
  },
  "dataset_scores": [],
  "contrasts": [],
  "row_group_side_by_side": []
}
```

Each `dataset_scores` row must contain `run_id`, `split_policy`, `dataset`, `metric`, `pipeline`, `condition_scope`, `n_expected_cells`, `n_success_cells`, `n_finite_cells`, `coverage`, `score`, `score_status`, and a structured `failure_reasons` object. Each `contrasts` row must contain `family_id`, `family_size`, `contrast_id`, `split_policy`, `metric`, `pipeline_a`, `pipeline_b`, `n_complete_datasets`, `n_expected_datasets`, `mean_delta`, `median_delta`, `win_count`, `tie_count`, `loss_count`, `p_value`, `holm_adjusted_p_value`, `ci_method`, `ci_low`, `ci_high`, and `status`.

The `row_group_side_by_side` table must use the exact key:

```text
(dataset, seed, fold, condition, pipeline, model, metric)
```

and contain `row_status`, `group_status`, `row_value`, `group_value`, `row_failure_reason`, `group_failure_reason`, `pairable`, and `group_minus_row`. It must also carry the row and group run IDs and fingerprints. At the dataset-summary level, report both policies side by side with separate expected-task and complete-dataset denominators. A group-aware infeasibility result is a recorded status, not permission to substitute row-level folds.

## Incomplete-run accounting requirements

The corrected task ledger and manifest should classify every intended task exactly once at the current-attempt level as `success`, `failed`, `skipped`, `timed_out`, or `pending`. Phase detail remains separate (`phase1` feature preparation/cache and `phase2` model/evaluation). A task with phase 1 success and no terminal phase 2 record is `pending`, not successful. A retry may retain attempt history, but the result note must expose both the final task status and the number of attempts.

For each split policy and dataset, report:

```text
expected = datasets × seeds × folds × conditions × pipelines × models
terminal = success + failed + skipped + timed_out
pending = expected - terminal
success_fraction = success / expected
finite_metric_fraction = finite_metric_successes / expected
```

These counts must be available at total, dataset, condition, pipeline, model, seed, fold, phase, and split-policy levels. `success` with a null or nonfinite primary metric is not a valid primary score; it is counted as a successful task with an undefined metric and appears in the finite-metric denominator separately.

The current corrected runner already writes task keys, phase states, failure rows, fingerprints, and an expected-task count. The current manifest count implementation reports only `success` and `failed`, and the result writer has no explicit `skipped`, `timed_out`, or `pending` terminal records. An interrupted process can therefore leave intended tasks absent from both the task list and result ledger. This is acceptable for the smoke demonstration, but it is a release blocker for a confirmatory campaign until the result note derives and publishes the pending denominator.

## Historical separation and interpretation

`provenance/original_run.json` classifies the old stream as `historical_original_run` and records its hashes. The corrected campaign must keep this classification. The old 560,002-row stream is useful for reproducing the repository’s prior evidence and for identifying failure/censoring patterns; it is not a corrected estimate. In particular:

- do not append the historical stream to a corrected JSONL ledger;
- do not label historical rows with the corrected run ID or protocol;
- do not use the historical partial `covertype` or `kddcup99` blocks to satisfy corrected coverage;
- do not treat the old condition names as corrected training-corruption conditions;
- do not report pooled historical row counts as independent replication units;
- preserve the original ledger and backup byte-for-byte.

The paper’s existing 22-dataset historical scope and its descriptive intervals remain historical claims. The corrected result note must carry a separate run ID, fingerprints, split policy, task denominators, metric-validity counts, dataset-level contrasts, and Holm family identifiers before any corrected numerical claim is made.

## Audit conclusions and release gates

1. The existing corrected statistics code has the right high-level independent unit and four-contrast Holm family, and the focused regression checks passed.
2. It is not sufficient as a corrected-run result-note generator because it does not report expected-cell coverage, finite-metric denominators, confidence intervals, split policy, or failure/pending accounting.
3. The historical ledger is complete as a parseable file but incomplete as a task record: 52,498 intended rows are absent, all observed rows are marked successful, and many metric values are undefined.
4. The row-level and group-aware tracks require separate task identities and side-by-side summaries. Group-fold infeasibility must be recorded explicitly.
5. No full corrected benchmark result is available from this audit. The next gate is a bounded real-data preflight that emits the proposed status and coverage fields before the long campaign.

