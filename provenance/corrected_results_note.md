# Corrected results note

Status: **PENDING CORRECTED RUN**. No full corrected campaign has been launched.

This file is reserved for the machine-checked result record. It must be generated from a frozen run manifest and explicit result ledger, never from the historical ledger.

## Frozen record fields

- Run ID: `PENDING CORRECTED RUN`
- Code commit and source fingerprint: `PENDING CORRECTED RUN`
- Environment/package versions: `PENDING CORRECTED RUN`
- Dataset source/version, CSV checksum, sidecar checksum: `PENDING CORRECTED RUN`
- Split policies: `row_level` legacy-comparable and `group_aware` exact raw-feature groups
- Canonical grouping rule: target-excluded, unperturbed raw predictors; type/missing/numeric/categorical canonicalization and collision checks recorded in the group audit
- Conditions/pipelines/models/seeds/folds: `PENDING FROZEN MANIFEST`
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

- Duplicate-group prevalence: recorded in `dataset_schema_audit.json`; corrected run artifact: `PENDING`.
- Row-level cross-fold overlap: recorded in `dataset_schema_audit.md`; corrected run artifact: `PENDING`.
- Conflicting-label group counts: recorded in `dataset_schema_audit.json`; corrected run artifact: `PENDING`.
- Group-aware shared-group overlap: must be exactly zero per valid fold; corrected value: `PENDING CORRECTED RUN`.

## Mechanism/operator and coverage sections

Candidate/Jacobian summaries, operator-isolation results, feature counts/cost, task coverage, failures, retries, timeouts, complete-case analysis, and missingness sensitivity are all `PENDING CORRECTED RUN`.

## Historical ledger (separate)

The historical ledger has 560,002 rows and the manuscript reports a 538,972-row subset. Those values are preserved in `provenance/original_run.json` and are not corrected estimates, not group-aware estimates, and not evidence for the pending two-track comparison.
