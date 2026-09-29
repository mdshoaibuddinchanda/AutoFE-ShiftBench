# Corrected results note

Status: **PENDING CORRECTED RUN**. No full corrected campaign has been launched. A bounded `sonar` preflight passed under both split policies; its artifact is `provenance/reviewer1_preflight_manifest.json`.

This file is reserved for the machine-checked result record. It must be generated from a frozen run manifest and explicit result ledger, never from the historical ledger.

## Frozen record fields

- Run ID: `PENDING CORRECTED RUN`
- Code commit and source fingerprint: `PENDING CORRECTED RUN`
- Environment/package versions: `PENDING CORRECTED RUN`
- Dataset source/version, CSV checksum, sidecar checksum: `PENDING CORRECTED RUN`
- Split policies: `row_level` legacy-comparable and `group_aware` exact raw-feature groups
- Canonical grouping rule: target-excluded, unperturbed raw predictors; type/missing/numeric/categorical canonicalization and collision checks recorded in the group audit
- Conditions/pipelines/models/seeds/folds: `PENDING FROZEN MANIFEST` (bounded preflight used one clean condition, seed 42, requested fold 1, three pipelines, and logistic regression)
- Task denominator: `PENDING FROZEN MANIFEST`
- Interval method and confidence level: `PENDING FROZEN ANALYSIS`
- Multiplicity families and adjustment: dataset-level paired contrasts with prespecified Holm families

## Required side-by-side table

For every applicable dataset and overall aggregate, report:

| Dataset / aggregate | Metric | Row-level estimate | Group-aware estimate | Group minus row difference | Valid dataset count | Completed tasks / denominator | Missing/failed/skipped count | Interval | Adjusted p-value |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| all applicable datasets | Raw | PENDING CORRECTED RUN | PENDING CORRECTED RUN | PENDING CORRECTED RUN | PENDING | PENDING | PENDING | PENDING | PENDING |
| all applicable datasets | AutoFE central contrast | PENDING CORRECTED RUN | PENDING CORRECTED RUN | PENDING CORRECTED RUN | PENDING | PENDING | PENDING | PENDING | PENDING |
| all applicable datasets | matched-cap Raw contrast | PENDING CORRECTED RUN | PENDING CORRECTED RUN | PENDING CORRECTED RUN | PENDING | PENDING | PENDING | PENDING | PENDING |

Per-dataset values must be preserved below the aggregate table so reversals and dataset-specific failures remain visible.

## Duplicate and fold diagnostics

- Duplicate-group prevalence: recorded in `dataset_schema_audit.json`; all-dataset group artifact: `provenance/group_fold_audit.json`.
- Row-level cross-fold overlap: recorded in `dataset_schema_audit.md` and `group_fold_audit.md` (e.g. PhishingWebsites 65.418363%, KDDCup99 67.007%).
- Conflicting-label group counts: recorded in `dataset_schema_audit.json` and `group_fold_audit.json`.
- Group-aware shared-group overlap: zero by assertion on all feasible preflight folds; two datasets are explicitly AUC-infeasible and are not substituted.

## Mechanism/operator and coverage sections

Candidate/Jacobian summaries and operator-isolation benchmark values remain `PENDING CORRECTED RUN`. The bounded preflight recorded 950 training-only candidate-history records and estimated approximately 79.7 GiB compressed for the original 612,500-task grid at that rate. Full task coverage, failures, retries, timeouts, complete-case analysis, and missingness sensitivity remain pending.

## Historical ledger (separate)

The historical ledger has 560,002 rows and the manuscript reports a 538,972-row subset. Those values are preserved in `provenance/original_run.json` and are not corrected estimates, not group-aware estimates, and not evidence for the pending two-track comparison.
