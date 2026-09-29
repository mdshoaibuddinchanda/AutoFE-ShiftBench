# Reviewer #1 correction run plan

Status: **preflight / implementation in progress** (2026-09-29, Asia/Calcutta).

This plan freezes the work required before any approximately 14-day corrected campaign. The campaign is not launched by this plan. Existing Conda `p12` at `D:\Conda\p12` is the only execution environment.

## Acceptance gates

1. All current tracked and untracked work is inventoried; no historical result file is overwritten.
2. The final protocol has two explicitly labeled tracks: row-level legacy-comparable folds and group-aware folds that assert zero shared exact raw-feature groups.
3. Every configured dataset has verified source identity, CSV/sidecar hashes, canonical duplicate-group metadata, fold-feasibility status, and metric-support status.
4. Training-only preprocessing, feature synthesis, selection, caching, and analysis paths have negative-control tests; transductive partition experiments remain segregated.
5. FSVA measurement, candidate histories, baseline-cap, and operator-isolation scopes are prespecified with finite-value and undefined-case handling.
6. Result accounting records every intended task as success, failed, skipped with a reason, timed out, or pending, and resume identity is immutable.
7. The complete regression suite, compile/import check, schema/hash audit, group-fold feasibility audit, and bounded real-data preflight pass in `p12`.
8. A descriptive milestone commit is made after each integrated milestone. The expensive run starts only after a clean gate review and frozen task manifest.

## Owners and bounded assignments

| Assignment | Owner | Files allowed | Evidence required | Status |
|---|---|---|---|---|
| Source identity, canonical raw-feature groups, group-aware folds, fold feasibility | lead + data/folds agent | `src/group_splits.py`, `tests/test_group_splits.py`, dataset audit additions | source IDs, canonicalization definition, group sizes/conflicting labels, zero-overlap assertions, class support and AUC feasibility | in progress |
| FSVA mechanism and operator ablations | mechanism agent | new mechanism modules/tests only unless lead integrates a focused patch | Jacobian definition/norm, analytic-vs-numeric checks, candidate history schema/storage estimate, operator budget and undefined cases | in progress |
| Dataset-level analysis and result accounting | analysis agent | new analysis/provenance files only unless lead integrates a focused patch | primary contrasts, denominators, paired dataset inference, multiplicity, coverage, row/group comparison schema | in progress |
| Independent information-flow audit | independent audit agent | report only; no source edits | paths for target/held-out data/statistics, stale cache/run identity, unsupported claims, blockers | in progress |

## Frozen two-track core (before corrected results)

The common core is Raw versus `AutoFE_Baseline` under clean training data plus Gaussian-noise, missing-value, and label-noise primary training-corruption conditions, using the same configured datasets, seeds, folds, model set, and task accounting under both split policies. The existing original pipelines and conditions remain in the inventory and are reported separately when not in the core. Matched-cap Raw, full-dimensional Raw, and operator/isolation analyses are prespecified secondary scopes. No corrected numerical estimate is available yet.

Row-level folds are retained for legacy comparability. Group-aware folds use deterministic group IDs from unperturbed, target-excluded raw predictors and must assert zero shared groups per fold. A dataset is not silently dropped or changed to a different fold count; infeasibility is recorded and handled only by a prespecified documented rule.

Primary inference will summarize each dataset first, use datasets as the independent unit, report paired contrasts, confidence intervals, exact valid dataset/task denominators, and Holm-adjusted p-values within prespecified families. Fold/seed/model/condition repetitions are not independent datasets, and winner selection on the same folds is exploratory unless nested selection is added.

## Expected size and resource gate

The original corrected grid is `25 datasets × 5 seeds × 5 folds × 14 conditions × 7 pipelines × 10 models = 612,500 task records` before failures. A two-track common core at the same seed/fold/model settings approximately doubles the split-policy workload; mechanism and operator scopes add further tasks. Exact runtime/storage estimates must be measured by the bounded real-data preflight, including the slowest large datasets, cache matrices, candidate histories, and compressed result/checkpoint artifacts. The 14-day estimate is not an authorization to launch without margin.

## Required durable outputs

- `reviewer1_change_catalog.md`: one entry per Reviewer #1 concern, with implementation/evidence/status and dated decision log.
- `reviewer1_response_draft.md`: point-by-point journal response; every unavailable corrected value is `PENDING CORRECTED RUN`.
- `corrected_results_note.md`: machine-checked result record with run ID, commit, hashes, split policy, denominators, estimates, intervals, adjusted p-values, coverage, failures, and row/group differences. Historical ledger is a separate labeled section.
- `dataset_schema_audit.{json,md}`: current source/schema/hash/duplicate-overlap evidence.

## Command gate checklist

- `D:\Conda\p12\python.exe -m pytest -q -rs tests`
- `D:\Conda\p12\python.exe -m compileall -q src tests main.py`
- `D:\Conda\p12\python.exe -m provenance.audit_dataset_schemas`
- `D:\Conda\p12\python.exe -m provenance.audit_group_folds`
- bounded real-data preflight command recorded after implementation
- frozen manifest and clean Git status before the long campaign

Baseline evidence: `D:\\Conda\\p12\\python.exe -m pytest -q -rs tests` -> `19 passed, 2 warnings in 27.12s`; `D:\\Conda\\p12\\python.exe -m compileall -q src tests main.py` -> pass. Warnings are Loky physical-core detection and Woodwork `pkg_resources` deprecation. The full suite was run after the Dry Bean downloader change and before the new Reviewer #1 modules are integrated.

Current blocker: repeated exact raw feature vectors cross row-level folds; group-aware fold feasibility and the final common-core runtime are not yet verified. No long benchmark is running.
