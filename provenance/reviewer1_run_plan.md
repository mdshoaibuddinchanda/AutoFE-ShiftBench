# Reviewer #1 correction run plan

## 2026-10-01 launch-preparation update

The original resource and code-fingerprint statements below are historical planning evidence. The current runner has a compact SQLite task-status ledger for grids above 10,000 cells, so `manifest.json` no longer grows with every completed full-grid cell. Metric-only group-fold AUC infeasibility is recorded as per-cell `skipped` results on the actual group folds, with no row-level substitution. An all-configured-seed rule excludes a group-aware dataset from AUC when any of the five configured seeds lacks all-fold support. The all-seed audit and versioned seven-run manifest are `group_seed_grid_audit_v1.json` and `reviewer1_launch_scope_v2.json`.

The seven prespecified run identities cover two primary policies (1,750,000 intended cells), row-level transductive domain partitions (175,000), and two-policy feature-availability and majority-label-relabeling sensitivities (175,000 each): **2,275,000 intended cells** in total. With the planned two group-AUC-ineligible datasets, 84,000 cells are explicit group-policy skips before any condition-specific transductive skips. The primary all-fold group-AUC comparison remains at most 23 datasets. The all-seed audit must confirm this exact denominator before freeze.

The user has allowed a run longer than ten days, so the prior 7,000-valid-cells/hour threshold is no longer a hard acceptance gate. Representative both-policy large-dataset timing, memory, bounded-cache high water, storage margin, and interruption/resume on the **intended host** are still needed for an honest runtime and recovery plan. The 2026-10-01 local probe measured 434.94 GiB free on `D:` with the existing `p12`; it says nothing about a different host. Candidate histories are not enabled by the seven runner commands. Their prior 79.66 GiB estimate assumed 612,500 history tasks without an implemented full-run writer and must not be reported as actual allocated output. Until the host and mechanism-history decisions are resolved, the long campaign remains **HOLD — NOT LAUNCHED**.

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

## Condition-accounting prelaunch checklist

| Gate | Status | Evidence |
|---|---|---|
| All 14 original conditions accounted for | **PASS** | [`reviewer1_condition_crosswalk.md`](reviewer1_condition_crosswalk.md) |
| Every condition classified by estimand | **PASS** | Crosswalk; `README.md`; `src/pipeline_runner.py` scope enforcement |
| No target or held-out leakage in corrected primary protocol | **PASS** | [`audit_report.md`](audit_report.md); `tests/test_leakage_controls.py`; `tests/test_condition_protocol.py` |
| Zero duplicate-feature group overlap in group-aware folds | **PASS** | [`group_fold_audit.md`](group_fold_audit.md); `src/group_splits.py` |
| 23/25 group-aware ROC-AUC eligibility and two explicit skips | **PASS** | [`reviewer1_scope_manifest.json`](reviewer1_scope_manifest.json); [`group_fold_audit.json`](group_fold_audit.json) |
| Operator truth table and result identities | **PASS** | [`reviewer1_operator_ablation_manifest_v1.json`](reviewer1_operator_ablation_manifest_v1.json); `tests/test_operator_ablations.py` |
| Primary contrasts, dataset statistical unit, and Holm correction | **PASS** | `src/reviewer1_analysis.py`; [`corrected_results_note.md`](corrected_results_note.md) |
| Dataset hashes, source identities, seeds, code, task manifest, split checksums | **PASS** | [`reviewer1_scope_manifest.json`](reviewer1_scope_manifest.json); [`reviewer1_optimization_scope_v1.json`](reviewer1_optimization_scope_v1.json) |
| Full tests, bounded preflight, forced interruption, new-process resume | **PASS** | [`performance_recovery_report.md`](performance_recovery_report.md); latest `p12` test gate below |
| Friend-PC hardware, free disk, peak RAM/storage, and throughput | **PENDING** | [`host_measurement.json`](host_measurement.json) is local evidence; friend-PC measurement is not recorded |
| Runtime and storage fit the launch budget | **PENDING** | Current optimized estimate is 180.56 planning days and the retained-cache projection is 47,377.68 GiB; both fail a 14-day/local-storage gate pending friend-PC measurement and redesign |

The last two resource gates remain pending until measurements are taken on the intended friend PC. They do not authorize launch. The condition crosswalk records the four separate sensitivity tracks and their exact planned cell counts without changing the frozen 1,750,000-cell primary manifest.

## Frozen AUC policy, scope, and resource gate

The group-aware primary ROC-AUC denominator contains all 25 configured datasets. `wine-quality-red` is recorded as skipped because one or more test folds miss a target class. `kddcup99` is recorded as skipped because a target class has fewer rows/groups than five splits and at least one test fold misses a class. These two datasets remain visible with their reasons; no row-level folds are substituted. The row-level track retains all 25 datasets. The resulting all-pipeline counts are 875,000 intended tasks per policy, 1,750,000 across both policies, with 70,000 group-policy tasks explicitly skipped for AUC infeasibility and 1,680,000 group/row task cells AUC-eligible. The core Raw/AutoFE-Baseline scope is 125,000 tasks per policy; its group track has 115,000 eligible and 10,000 explicitly skipped cells.

The large-dataset scale preflight on 100,000-row `airlines` took 57.45 seconds for three row-level tasks and 101.89 seconds for three group-aware tasks. Linear planning extrapolation gives approximately 27.7 days for the two-pipeline row core with ten models, 49.1 days for the group core, and 537.9 days for all 14 pipelines under both policies. These are resource projections, not corrected performance results.

The `D:` volume had 437.14 GiB free. The 79.66 GiB compressed candidate-history estimate alone leaves 357.48 GiB, and projected results/manifests/checkpoints add about 20.87 GiB. However, the measured `airlines` feature-cache rate projects to about 47,378 GiB for all 14 pipelines and both policies, or about 6,768 GiB for the core alone, under the current retention policy. Storage therefore fails with the current cache design. The long run remains blocked until caches are streamed/deleted or a larger storage target is provisioned. Exact values are frozen in [`reviewer1_scope_manifest.json`](reviewer1_scope_manifest.json) and [`reviewer1_scale_preflight_manifest.json`](reviewer1_scale_preflight_manifest.json).

The frozen source identity is commit `8d0c90093232d4b4afd86a133b94f37052c1928b` with source fingerprint `7f1a85210f1c3dd43c59381cad891e291242d874886f283b1e2b95d02abc84a7`; dataset-list, schema-audit, and group-audit hashes are recorded in the scope manifest. This is a protocol freeze only, not authorization to launch.

## Required durable outputs

- `reviewer1_change_catalog.md`: one entry per Reviewer #1 concern, with implementation/evidence/status and dated decision log.
- `reviewer1_condition_crosswalk.md` and `reviewer1_condition_scope_manifest_v1.json`: complete 14-condition accounting and separate sensitivity task/resource plan.
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

Verification evidence for the pre-optimization audit milestone: `D:\\Conda\\p12\\python.exe -m pytest -q -rs tests` -> `42 passed, 7 warnings in 10.40s`; `D:\\Conda\\p12\\python.exe -m compileall -q src tests main.py` -> pass; `D:\\Conda\\p12\\python.exe -m provenance.audit_dataset_schemas` -> `audited=25/25; verified=25/25`; `D:\\Conda\\p12\\python.exe -m provenance.audit_group_folds` -> `25` structural, `23` class/AUC-supported. After condition accounting, the same p12 suite passes `75 tests`; compileall, manifest/hash checks, and diff checks pass. Warnings are the expected Loky physical-core detection, Woodwork deprecation, and intentional undersupported-class group tests.

Current decision: the bounded preflight passed for `sonar` and the large-dataset scale preflight passed for `airlines` under both policies. The full campaign is **DO NOT LAUNCH**: two datasets are explicitly AUC-infeasible under group folds, the retained-cache projection exceeds available space by orders of magnitude, and the ten-model runtime projection is far beyond 14 days. The runner records group infeasibility as `blocked_group_split_infeasible` and never falls back to row-level folds. No long benchmark is running.

Preflight artifacts: [`reviewer1_preflight_manifest.json`](reviewer1_preflight_manifest.json) and [`reviewer1_scale_preflight_manifest.json`](reviewer1_scale_preflight_manifest.json). They record zero failures and explicit cache/result byte counts. These timings are gate measurements used only for the transparent planning projections above.

## Optimization milestone after the original freeze

The original retain-all cache gate remains rejected, and its 47,377.68 GiB projection is retained as the historical resource evidence. Versioned optimized scope and recovery controls are recorded in [`reviewer1_optimization_plan_v1.md`](reviewer1_optimization_plan_v1.md), [`reviewer1_optimization_scope_v1.json`](reviewer1_optimization_scope_v1.json), and [`performance_recovery_report.md`](performance_recovery_report.md). The runner now supports bounded regenerable feature caches with leases, ready markers, checksums, high-water accounting, and cleanup after the last compatible model. It also supports a durable task scheduler with leases, heartbeats, retry classification, atomic result envelopes, and reconciliation. These controls preserve the data hashes, AUC denominator, split policies, task counts, and analysis definitions; they do not produce corrected benchmark results.

The integrated runner also accepts `--workers N` and executes model fits in a
bounded Windows spawn pool while the coordinator owns scheduler leases, cache
publication, checkpoints, and result rows. A focused smoke test records
overlapping child-process intervals and unique task IDs. The bounded
four-dataset pilot covers 11,200 intended cells across both policies with one
seed and one fold; its timing and coverage are recorded in
[`four_dataset_performance_pilot.md`](four_dataset_performance_pilot.md), and
all corrected performance effects remain **PENDING CORRECTED RUN**.

The bounded profile measured a complete 80-task Sonar mix at 1/2/4 workers in 91.1865/46.4736/30.6148 seconds, with zero one-versus-four prediction-hash mismatches. Applying the observed 2.979x speedup to the prior linear projections gives planning scenarios of 9.30 days (core row), 16.49 days (core group), 65.10 days (all-14 row), 115.46 days (all-14 group), and 180.56 days (both policies); cache-reuse benefit is not assumed. These are not a 14-day claim. The friend-PC host probe, representative large-task interruption/resume, and final storage-margin checks remain launch blockers.

The operator truth table and candidate-validity policy are frozen in [`reviewer1_operator_ablation_manifest_v1.json`](reviewer1_operator_ablation_manifest_v1.json). It confirms that the existing 14-pipeline scope is unchanged: `AutoFE_Baseline` is the full arithmetic reference, `AutoFE_NoMultiply` remains the historical joint multiplication-and-division removal, and multiplication/division isolate and leave-one-out variants are separately executable. The bounded smoke and full p12 suite verify code behavior only; operator performance effects remain **PENDING CORRECTED RUN**.

## Friend-PC launch audit (2026-09-30)

The bounded cache now has an opt-in fan-out audit and the durable scheduler has single-host replacement recovery and checkpoint repair. Ten-consumer and mid-group interruption smokes pass in `p12`; they are code verification, not benchmark results. `provenance/friend_pc_launch_audit.md` records the exact host-probe, scheduling, GPU, preflight, stop/resume, and final-verification commands. The earlier `host_measurement.json` remains local-host evidence only. The friend-PC host manifest, representative both-policy throughput, cache peak, storage margin, and thermal/GPU measurements are **PENDING**. The complete 1,680,000-cell executable scope therefore has an **UNKNOWN / DO NOT LAUNCH** verdict until the friend-PC evidence is collected and the 7,000 valid-cells/hour gate is met with margin.

The condition crosswalk is frozen in [`reviewer1_condition_crosswalk.md`](reviewer1_condition_crosswalk.md). It accounts for the four historical conditions outside the primary ten: target-exposed historical PCA/K-means partitions are invalidated for corrected inference and reimplemented as target-free transductive row-level sensitivities; feature removal is reimplemented as a training feature-availability sensitivity; and class-prior shift is reimplemented as majority-label relabeling. A group-aware transductive request fails closed until a group-constrained assignment rule exists. Domain fold metadata records global target-free fit scope, all-categorical fallback, class support, and AUC status; unsupported domain AUC cells are explicitly skipped rather than written as successful NaNs. Corrected condition effects remain **PENDING CORRECTED RUN**.

## Four-dataset pilot diagnosis and full-run gate (2026-09-30)

The bounded four-dataset grid completed at 5,600/5,600 authoritative successes per policy and 11,200 total, with the same fourteen pipeline definitions and ten classifiers. Row-level's twelve failed JSONL attempt rows and the schedulers' 23 intentional replacement attempts remain retry history; no committed cell was refit. The group-aware run finished with zero terminal failures, zero lease expiries, full result/checkpoint/mirror parity, and zero feature-cache payloads after final cleanup. A 15-minute sample added 225 valid cells, or 895/hour, with fresh 600-second leases. The full group target spanned 8.277 hours; the 4,695-cell post-migration continuation took 7.939 hours. Group-aware split construction cost only 0.223–0.595 seconds once per dataset/seed in a bounded profile. Repeated whole-scheduler-result and whole-cache scans during claim/audit were measured coordinator bottlenecks. See [`four_dataset_pilot_diagnosis.md`](four_dataset_pilot_diagnosis.md) for exact attempt IDs, per-dataset timing, stage measurements, and final accounting.

After the pilot finished, a first 560-cell performance profile exposed a transient Windows cache-manifest `PermissionError` and ended with 559 successes; it is excluded from timing comparisons. Atomic cache replacement retries were committed in `90a8eea`, and a fresh complete baseline took 415.5 seconds. Sequential, same-cell profiles on all four datasets under group-aware clean/seed 42/fold 1 measured task-local claims at 193.3 seconds, artifact-local reader audit alone at 397.5 seconds, and both at 186.0 seconds. A repeated task-local/both pair took 195.0/186.7 seconds. The committed main branch completed the same cells in 183.6 seconds, with final source fingerprint `6afa679eba50c26fac66f3d2f5b9c34691ca14c3a4329516998f8b476e5996f3`. All complete modes have 560/560 successes, exact matrix and prediction hashes and metric parity, and 56 builds/504 hits/56 deletions. Peak process-tree RAM was 1,053.9–1,058.7 MiB; system VRAM readings of 1,391–1,404 MiB include other processes and are not pilot GPU usage. Both scan changes are retained in `4686e32`; the integrated full `p12` suite passed 94 tests, with `compileall` and `git diff --check` clean. These are bounded code-verification timings, not corrected ROC-AUC or 25-dataset throughput results.

This pilot does not change the versioned 1,750,000 intended / 1,680,000 AUC-eligible full-run task accounting. The full 25-dataset campaign remains **DO NOT LAUNCH**: the friend-PC and representative large-dataset measurements must demonstrate at least **7,000 valid cells/hour** sustained for a ten-day target, plus disk and RAM margin and the remaining fault-matrix gate. The implementation changed after the historical full-scope freeze, so refresh the code fingerprint and versioned manifest before any launch. Throughput from four small datasets cannot establish the full-run requirement. Corrected ROC-AUC, operator, and mechanism effects remain **PENDING CORRECTED RUN**.
