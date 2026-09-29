# Reviewer #1 correction run plan

Status: **bounded preflight complete; full campaign intentionally held** (2026-09-29, Asia/Calcutta).

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
| Source identity, canonical raw-feature groups, group-aware folds, fold feasibility | lead + data/folds agent | `src/group_splits.py`, `tests/test_group_splits.py`, dataset audit additions | source IDs, canonicalization definition, group sizes/conflicting labels, zero-overlap assertions, class support and AUC feasibility | implemented; 25/25 structural, 23/25 AUC-supported |
| FSVA mechanism and operator ablations | mechanism agent | new mechanism modules/tests only unless lead integrates a focused patch | Jacobian definition/norm, analytic-vs-numeric checks, candidate history schema/storage estimate, operator budget and undefined cases | implemented; bounded history preflight complete |
| Dataset-level analysis and result accounting | analysis agent | new analysis/provenance files only unless lead integrates a focused patch | primary contrasts, denominators, paired dataset inference, multiplicity, coverage, row/group comparison schema | implemented; corrected ledger pending |
| Independent information-flow audit | independent audit agent | report only; no source edits | paths for target/held-out data/statistics, stale cache/run identity, unsupported claims, blockers | in progress |

## Frozen two-track core (before corrected results)

The common core is Raw versus `AutoFE_Baseline` under clean training data plus Gaussian-noise, missing-value, and label-noise primary training-corruption conditions, using the same configured datasets, seeds, folds, model set, and task accounting under both split policies. The existing original pipelines and conditions remain in the inventory and are reported separately when not in the core. Matched-cap Raw, full-dimensional Raw, and operator/isolation analyses are prespecified secondary scopes. No corrected numerical estimate is available yet.

Row-level folds are retained for legacy comparability. Group-aware folds use deterministic group IDs from unperturbed, target-excluded raw predictors and must assert zero shared groups per fold. A dataset is not silently dropped or changed to a different fold count; infeasibility is recorded and handled only by a prespecified documented rule.

Primary inference will summarize each dataset first, use datasets as the independent unit, report paired contrasts, confidence intervals, exact valid dataset/task denominators, and Holm-adjusted p-values within prespecified families. Fold/seed/model/condition repetitions are not independent datasets, and winner selection on the same folds is exploratory unless nested selection is added.

## Frozen AUC policy, scope, and resource gate

The group-aware primary ROC-AUC denominator contains all 25 configured datasets. `wine-quality-red` is recorded as skipped because one or more test folds miss a target class. `kddcup99` is recorded as skipped because a target class has fewer rows/groups than five splits and at least one test fold misses a class. These two datasets remain visible with their reasons; no row-level folds are substituted. The row-level track retains all 25 datasets. The resulting all-pipeline counts are 875,000 intended tasks per policy, 1,750,000 across both policies, with 70,000 group-policy tasks explicitly skipped for AUC infeasibility and 1,680,000 group/row task cells AUC-eligible. The core Raw/AutoFE-Baseline scope is 125,000 tasks per policy; its group track has 115,000 eligible and 10,000 explicitly skipped cells.

The large-dataset scale preflight on 100,000-row `airlines` took 57.45 seconds for three row-level tasks and 101.89 seconds for three group-aware tasks. Linear planning extrapolation gives approximately 27.7 days for the two-pipeline row core with ten models, 49.1 days for the group core, and 537.9 days for all 14 pipelines under both policies. These are resource projections, not corrected performance results.

The `D:` volume had 437.14 GiB free. The 79.66 GiB compressed candidate-history estimate alone leaves 357.48 GiB, and projected results/manifests/checkpoints add about 20.87 GiB. However, the measured `airlines` feature-cache rate projects to about 47,378 GiB for all 14 pipelines and both policies, or about 6,768 GiB for the core alone, under the current retention policy. Storage therefore fails with the current cache design. The long run remains blocked until caches are streamed/deleted or a larger storage target is provisioned. Exact values are frozen in [`reviewer1_scope_manifest.json`](reviewer1_scope_manifest.json) and [`reviewer1_scale_preflight_manifest.json`](reviewer1_scale_preflight_manifest.json).

The frozen source identity is commit `8d0c90093232d4b4afd86a133b94f37052c1928b` with source fingerprint `7f1a85210f1c3dd43c59381cad891e291242d874886f283b1e2b95d02abc84a7`; dataset-list, schema-audit, and group-audit hashes are recorded in the scope manifest. This is a protocol freeze only, not authorization to launch.

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
- `D:\Conda\p12\python.exe -m provenance.reviewer1_scale_preflight`
- frozen manifest and clean Git status before the long campaign

Verification evidence for the pre-optimization audit milestone: `D:\\Conda\\p12\\python.exe -m pytest -q -rs tests` -> `42 passed, 7 warnings in 10.40s`; `D:\\Conda\\p12\\python.exe -m compileall -q src tests main.py` -> pass; `D:\\Conda\\p12\\python.exe -m provenance.audit_dataset_schemas` -> `audited=25/25; verified=25/25`; `D:\\Conda\\p12\\python.exe -m provenance.audit_group_folds` -> `25` structural, `23` class/AUC-supported. After optimization integration, the same p12 suite passes `61 tests`; compileall and diff checks pass. Warnings are the expected Loky physical-core detection, Woodwork deprecation, and intentional undersupported-class group tests.

Current decision: the bounded preflight passed for `sonar` and the large-dataset scale preflight passed for `airlines` under both policies. The full campaign is **DO NOT LAUNCH**: two datasets are explicitly AUC-infeasible under group folds, the retained-cache projection exceeds available space by orders of magnitude, and the ten-model runtime projection is far beyond 14 days. The runner records group infeasibility as `blocked_group_split_infeasible` and never falls back to row-level folds. No long benchmark is running.

Preflight artifacts: [`reviewer1_preflight_manifest.json`](reviewer1_preflight_manifest.json) and [`reviewer1_scale_preflight_manifest.json`](reviewer1_scale_preflight_manifest.json). They record zero failures and explicit cache/result byte counts. These timings are gate measurements used only for the transparent planning projections above.

## Optimization milestone after the original freeze

The original retain-all cache gate remains rejected, and its 47,377.68 GiB projection is retained as the historical resource evidence. Versioned optimized scope and recovery controls are recorded in [`reviewer1_optimization_plan_v1.md`](reviewer1_optimization_plan_v1.md), [`reviewer1_optimization_scope_v1.json`](reviewer1_optimization_scope_v1.json), and [`performance_recovery_report.md`](performance_recovery_report.md). The runner now supports bounded regenerable feature caches with leases, ready markers, checksums, high-water accounting, and cleanup after the last compatible model. It also supports a durable task scheduler with leases, heartbeats, retry classification, atomic result envelopes, and reconciliation. These controls preserve the data hashes, AUC denominator, split policies, task counts, and analysis definitions; they do not produce corrected benchmark results.

The bounded profile measured a complete 80-task Sonar mix at 1/2/4 workers in 91.1865/46.4736/30.6148 seconds, with zero one-versus-four prediction-hash mismatches. Applying the observed 2.979x speedup to the prior linear projections gives planning scenarios of 9.30 days (core row), 16.49 days (core group), 65.10 days (all-14 row), 115.46 days (all-14 group), and 180.56 days (both policies); cache-reuse benefit is not assumed. These are not a 14-day claim. The friend-PC host probe, representative large-task interruption/resume, and final storage-margin checks remain launch blockers.
