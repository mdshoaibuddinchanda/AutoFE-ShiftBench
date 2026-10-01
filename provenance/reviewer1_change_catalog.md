# Reviewer #1 change catalog

Status legend: `open` = not yet implemented; `implemented-not-run` = code exists but the required evidence is pending; `verified` = the stated checks passed; `blocked` = a documented protocol or resource blocker remains. This catalog is an engineering/evidence ledger, not the author response.

## R1. FSVA validation

- Exact concern: Jacobian norms and candidate-level selection histories were not directly measured.
- Original problem / affected claim: FSVA was presented as a mechanism without direct derivative, candidate-history, or dataset-level association evidence.
- Relevant files: `src/feature_engineering.py`, `src/pipeline_runner.py`, `src/mechanism_audit.py`, `tests/test_mechanism_audit.py`.
- Decision and rationale: define a feature-map Jacobian only for supported arithmetic primitives; record undefined/non-finite cases; treat associations as evidence consistent with a mechanism, not proof of causality.
- Implementation: arithmetic Jacobian/finite-difference checks, scaled local norms, candidate-history schema, storage estimator, and optional Featuretools candidate logging are implemented. The runner does not enable full-grid history by default because the bounded estimate is large.
- Tests and commands: `D:\Conda\p12\python.exe -m pytest -q tests/test_mechanism_audit.py` -> 14 passed; bounded real-data preflight wrote 950 training-only candidate records with all four operators and finite scores.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: preflight `provenance/reviewer1_preflight_manifest.json`; full corrected values remain `PENDING CORRECTED RUN`.
- Limitations: Featuretools internals may not expose every generated expression or derivative boundary.
- Status: `implemented-not-run`.

## R2. Statistical analysis

- Exact concern: pooled row/task-level tests treated folds, seeds, models, or conditions as independent datasets.
- Original problem / affected claim: uncertainty and significance could be overstated.
- Relevant files: `src/stats_analysis.py`, `src/generate_tables.py`, `provenance/corrected_results_note.md`.
- Decision and rationale: dataset is the primary independent unit; prespecified paired contrasts, intervals, exact denominators, and Holm correction are required. Winner selection on shared folds remains exploratory unless nested selection is added.
- Implementation: `src/reviewer1_analysis.py` now emits expected-cell coverage, finite-metric denominators, paired dataset bootstrap intervals (10,000 replicates, seed 20260929), Holm family metadata, and row/group side-by-side schema. No corrected campaign ledger has been analyzed.
- Tests and commands: focused post-integration suite passed 42 tests; the final full `p12` suite after condition accounting passed 75 tests.
- Commit hash: `d4a897a` integration milestone; integrated source fingerprint `7f1a85210f1c3dd43c59381cad891e291242d874886f283b1e2b95d02abc84a7` is recorded in the schema audit.
- Benchmark artifact IDs / observed values: historical values remain separate; corrected values are `PENDING CORRECTED RUN`.
- Limitations: No corrected benchmark estimates exist yet.
- Status: `implemented-not-run`.

## R3. Leakage and duplicate-overlap evaluation

- Exact concern: target/held-out information paths and exact duplicate feature vectors crossing folds were not fully addressed.
- Original problem / affected claim: row-level folds can put identical raw feature vectors in train and test, allowing memorization; target-free partition boundaries must remain explicit.
- Relevant files: `src/splitters.py`, `src/group_splits.py`, `provenance/dataset_schema_audit.{json,md}`, `provenance/group_fold_audit.{json,md}`, `tests/test_group_splits.py`.
- Decision and rationale: retain row-level folds as legacy-comparable sensitivity; add group-aware folds using deterministic canonical target-excluded raw predictors and assert zero shared groups. Never silently fall back or discard infeasible datasets.
- Implementation: deterministic typed canonicalization, SHA-256 group IDs with collision checks, `StratifiedGroupKFold`, zero-shared-group assertions, runner split-policy identity, and explicit infeasibility manifests are implemented.
- Tests and commands: `python -m provenance.audit_group_folds` -> 25/25 structural group splits, 23/25 all-fold class/AUC support; `wine-quality-red` and `kddcup99` are explicitly infeasible for all-fold AUC; targeted group tests pass 7 tests. Row-level overlap examples include PhishingWebsites 65.42% and KDDCup99 67.01%.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: `provenance/group_fold_audit.json` and `.md`; corrected performance remains `PENDING CORRECTED RUN`.
- Limitations: Grouping removes duplicate-vector overlap, not semantic target proxies or deployment shift.
- Status: `implemented-not-run`.

## R4. Terminology and condition handling

- Exact concern: conditions were not always distinguished as training corruption, transductive partitioning, or deployment shift.
- Decision and rationale: preserve condition names, add explicit scope metadata, and reserve deployment-shift language for a protocol that changes held-out deployment distribution.
- Relevant files: `src/pipeline_runner.py`, `src/shift_generator.py`, `README.md`, `provenance/reviewer1_run_plan.md`.
- Implementation: primary/transductive/availability/relabeling scopes already fail closed when mixed; runner now persists `experiment_scope` and `split_policy` in manifests and task rows. Manuscript phrase checklist remains pending.
- Tests and commands: `tests/test_leakage_controls.py` condition semantics test passed in baseline suite.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: none; corrected values `PENDING CORRECTED RUN`.
- Limitations: Manuscript source still contains historical OpenML and shift-language statements.
- Status: `implemented-not-run`.

## R5. Condition accounting and terminology

- Exact concern: the original 14-condition design was reduced to a 10-condition corrected primary grid without a complete accounting of the four excluded conditions, and partition experiments were easy to misdescribe as deployment shift.
- Decision and rationale: reconstruct all 14 original condition IDs, preserve the historical ledger, keep the 10-condition training-corruption grid frozen, and place the four outside conditions in explicitly labeled sensitivity scopes. Do not silently add their cells to the primary run.
- Relevant files: [`provenance/reviewer1_condition_crosswalk.md`](reviewer1_condition_crosswalk.md), `src/pipeline_runner.py`, `src/splitters.py`, `src/shift_generator.py`, `tests/test_condition_protocol.py`, `tests/test_leakage_controls.py`.
- Findings: historical `covariate_shift` and `population_shift` passed the full dataframe into PCA/K-means, exposing the numeric target and held-out predictors. Historical `class_prior_shift` actually relabeled training labels, and `feature_removal_0.2` randomly removed training columns despite its documentation. Their historical rows remain immutable but cannot support the corrected estimands.
- Correction: current partitioners receive target-excluded `X`, record global target-free fit scope, distinguish PCA/K-means from all-categorical KFold fallback, check fold class support/AUC feasibility, and skip unsupported domain metric cells explicitly. Group-aware transductive requests fail closed rather than bypassing zero shared groups. Training feature availability and majority-label relabeling use training rows/labels only and remain separate sensitivity tracks.
- Counts: primary remains 875,000 intended cells per policy and 1,750,000 across both policies. Separate planning is 175,000 row-level cells for both domain conditions, 175,000 both-policy cells for feature availability, and 175,000 both-policy cells for label relabeling. No sensitivity cells were added to the frozen manifest.
- Tests and commands: condition protocol suite passed 7 tests; targeted leakage/condition tests passed 11 tests; corrected effects remain **PENDING CORRECTED RUN**.
- Status: `verified-code-and-bounded-smoke`; no long run launched.

## R6. Baseline fairness

- Exact concern: Raw and AutoFE dimensionality/selection caps were not matched.
- Decision and rationale: retain the original Raw baseline, add a cap-matched Raw comparator and full-dimensional Raw comparator, and report pre/post-synthesis caps, selected counts, and cost.
- Relevant files: `src/pipeline_runner.py`, `src/feature_engineering.py`, result schema (planned).
- Implementation: `Raw_CapMatched` and full-dimensional `Raw` are explicit; isolate-one and leave-one-out operator configs share depth/base/output/variance budgets. Preflight ran Raw, cap-matched Raw, and AutoFE baseline under both policies.
- Tests and commands: bounded preflight completed in 1.52–1.80 seconds per three-task policy run on `sonar`; result/cache sizes are recorded in `provenance/reviewer1_preflight_manifest.json`.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: `PENDING CORRECTED RUN`.
- Limitations: Matched-cap results cannot be inferred from historical results.
- Status: `implemented-not-run`.

## R7. Individual operator effects

- Exact concern: `AutoFE_NoMultiply` excludes multiplication and division, and does not isolate individual operator effects.
- Decision and rationale: preserve its historical meaning; add isolate-one-operator and justified leave-one-out configurations with shared budgets and finite-value rules.
- Relevant files: `src/feature_engineering.py`, `src/pipeline_runner.py`, `src/shift_generator.py` (planned integration).
- Implementation: the existing `AutoFE_Baseline` is the full arithmetic reference; no duplicate all-operator runner ID was added. `AutoFE_NoMultiply` is preserved as the historical joint removal of multiplication and division. The isolate-one and leave-one-out variants already existed in the 14-pipeline scope and are now verified end to end. Invalid arithmetic columns are rejected before selection, and result rows record generated/rejected/eligible/selected/duplicate counts by operator.

| Actual runner pipeline | Add | Subtract | Multiply | Divide | Status |
|---|:---:|:---:|:---:|:---:|---|
| `Raw` | No | No | No | No | existing core; no synthesis |
| `Raw_CapMatched` | No | No | No | No | existing frozen added baseline; no synthesis |
| `AutoFE_MI` | Yes | Yes | Yes | Yes | existing frozen added; all arithmetic |
| `AutoFE_Random` | Yes | Yes | Yes | Yes | existing frozen added; all arithmetic |
| `AutoFE_Baseline` | Yes | Yes | Yes | Yes | existing full reference |
| `AutoFE_NoMultiply` | Yes | Yes | No | No | existing historical joint removal |
| `AutoFE_Isolate_Add` | Yes | No | No | No | existing added; verified |
| `AutoFE_Isolate_Subtract` | No | Yes | No | No | existing added; verified |
| `AutoFE_Isolate_Multiply` | No | No | Yes | No | existing added; separately verified |
| `AutoFE_Isolate_Divide` | No | No | No | Yes | existing added; separately verified |
| `AutoFE_LeaveOut_Add` | No | Yes | Yes | Yes | existing added; verified |
| `AutoFE_LeaveOut_Subtract` | Yes | No | Yes | Yes | existing added; verified |
| `AutoFE_LeaveOut_Multiply` | Yes | Yes | No | Yes | existing added; division retained |
| `AutoFE_LeaveOut_Divide` | Yes | Yes | Yes | No | existing added; multiplication retained |

The code registry also contains `Raw_Variance` and `Raw_MI` (both no synthesis); these utility configurations are outside the frozen 14-pipeline Reviewer #1 manifest and are not added to the task grid.

- Tests and commands: `tests/test_operator_ablations.py` covers the truth table, operand ordering, zero/near-zero/nonfinite division, duplicate/rejected candidates, identity separation, repeated metadata stability, all ten bounded operator configurations, and result-note operator auditing. Focused operator tests pass; corrected performance effects remain pending.
- Commit hash: `0f272f4f52998f003a9ead8c17e702e5c406d09d` (final operator implementation, validity boundary, and raw-baseline metadata; earlier truth-table milestone `db29aec`).
- Configuration artifact: `provenance/reviewer1_operator_ablation_manifest_v1.json`.
- Benchmark artifact IDs / observed values: bounded synthetic smoke only; ROC-AUC effects are **PENDING CORRECTED RUN**.
- Limitations: Featuretools can produce different candidate counts for noncommutative versus commutative operators; counts are reported, not padded. No scientific performance conclusion is made.
- Status: `verified-code-and-bounded-smoke`.

## R8. Incomplete runs and stopping

- Exact concern: missing tasks and stopped blocks could be mistaken for zeroes or complete evidence.
- Decision and rationale: every task must have success, failed, skipped-with-reason, timed-out, or pending status with resumable identity and coverage by all design dimensions.
- Relevant files: `src/checkpoint.py`, `src/pipeline_runner.py`, `src/check_progress.py`, `provenance/corrected_results_note.md`.
- Implementation: checkpoint schema accepts success/failed/skipped/timed_out/pending; manifests publish expected, terminal, pending, phase, and status counts while retaining legacy `counts` compatibility.
- Tests and commands: failure/resume tests and result-note accounting tests pass; a long-run ledger is not available.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: no corrected run; historical ledger remains separately labeled.
- Limitations: unknown outcomes cannot support a causal or confirmatory claim.
- Status: `implemented-not-run`.

## R9. Reproducibility

- Exact concern: stable seeds, source identity, cache/run identity, and final artifacts must be independently reproducible.
- Decision and rationale: use canonical SHA-256 identities, source/sidecar checksums, runtime/package fingerprints, immutable task manifests, and explicit resume commands.
- Relevant files: `src/provenance.py`, `src/data_loader.py`, `src/pipeline_runner.py`, `requirements.txt`, `provenance/dataset_schema_audit.*`.
- Implementation: baseline reproducibility controls and UCI Dry Bean correction are committed in `a6a8e3b`; split policy, group digests, task fingerprints, and preflight artifacts are now included.
- Tests and commands: focused post-integration suite passed 42 tests; the condition-accounting suite passed 75 tests; the final launch-audit full `p12` suite passed 83 tests; compileall, schema/hash audit, and group audit also passed.
- Commit hash: `a6a8e3b` baseline provenance milestone.
- Benchmark artifact IDs / observed values: integrated source fingerprint `7f1a85210f1c3dd43c59381cad891e291242d874886f283b1e2b95d02abc84a7`; corrected run `PENDING CORRECTED RUN`.
- Limitations: historical Dry Bean input is unavailable and historical scores cannot be reconstructed.
- Status: `implemented-not-run`.

## R10. Confirmatory analysis now

- Exact concern: major fixes must be made and protocol frozen before the expensive run.
- Decision and rationale: do not launch until both split policies, common core, ablations, resource estimates, and result schemas are frozen and preflighted.
- Relevant files: `provenance/reviewer1_run_plan.md`, `provenance/corrected_results_note.md`, frozen manifest (planned).
- Implementation: plan, change catalog, response draft, corrected-result schema, small and large bounded preflights, exact task-count manifest, and storage/runtime gate are present. The long campaign is deliberately not launched: two configured datasets cannot support all-fold group-aware AUC, retained feature caches project to about 47,378 GiB for the full two-track ablation scope, and ten-model runtime projects far beyond 14 days.
- Tests and commands: bounded row/group preflight completed on `sonar`; large `airlines` scale preflight completed under both policies; the post-optimization full-suite gate passed 68 tests; only the pre-existing notebook execution-count edit remains outside the audit commits.
- Commit hash: `e3ed283` integrated audit milestone.
- Benchmark artifact IDs / observed values: none; no long run is running.
- Limitations: corrected numerical results are unavailable.
- Status: `blocked`.

## Dated decision log

- 2026-09-29: Existing branch/worktree inspected; all prior edits found uncommitted. No historical result file was rewritten.
- 2026-09-29: Baseline suite rerun in existing `p12`: 19 passed, 2 warnings in 27.12s; compileall passed.
- 2026-09-29: 25 datasets verified; Dry Bean source corrected to UCI ID 602. Commit `a6a8e3b` records baseline provenance and plan.
- 2026-09-29: Reviewer #1 run plan frozen provisionally; group-aware feasibility, mechanism measurement, operator isolation, and final resource gate remain open.
- 2026-09-29: Group audit completed in `p12`: all 25 group partitions construct, 23 support all-fold class/AUC metrics; `wine-quality-red` and `kddcup99` are recorded as infeasible rather than substituted.
- 2026-09-29: Bounded `sonar` preflight completed under both policies with Raw, cap-matched Raw, and AutoFE baseline; no full campaign was started. Candidate-history preflight wrote 950 training-only records and estimated approximately 79.7 GiB compressed for the full grid.
- 2026-09-29: Final post-integration gate in `p12`: 42 tests passed, compileall passed, schema audit 25/25, and compact group audit 25/25 structural with 23/25 AUC-supported.
- 2026-09-29: Large-dataset `airlines` scale gate measured 57.45 s for three row-level tasks, 101.89 s for three group-aware tasks, and about 0.87 GB of feature caches per three-task run. Frozen scope manifest records 1,750,000 intended two-policy tasks and a storage/runtime `DO_NOT_LAUNCH` decision under current retention.
- 2026-09-30: Bounded profiling measured 80 Sonar pipeline/model tasks at 1/2/4 workers with 0/0/0 failures and zero one-versus-four prediction-hash mismatches. Lease-aware bounded cache and durable scheduler integration committed in `906e4cd` and `ef98cc8`; full p12 suite passed 68 tests, including retained-versus-bounded parity and interrupted new-process runner resume. The optimized versioned scope retains the original AUC denominator, data hashes, analysis definitions, and 1,750,000 intended task cells. Friend-PC host probe remains required before launch.
- 2026-09-30: Operator audit completed in `c6e1b32` after truth-table milestone `db29aec`, with final raw-baseline metadata correction in `0f272f4`: all ten arithmetic ablation variants plus the four other frozen-scope names were smoke-tested. Multiplication-only, division-only, without-multiplication, and without-division paths were separately verified; historical `AutoFE_NoMultiply` remains the joint multiplication-and-division removal. Candidate validity accounting and result-note operator metadata were added; no pipeline-count change.
- 2026-09-30: Condition-accounting audit completed in `8a19bfe`, with final result-metadata identity pinned in `e5e06c4`: all 14 historical conditions are crosswalked; target-exposed historical partitions are excluded from corrected inference; transductive group-aware requests fail closed; domain AUC support and all-categorical fallbacks are explicit. The launch-audit additions then passed 82 p12 tests; no primary task count changed.

## R11. Performance, storage, and crash recovery before launch

- Exact concern: retained feature caches, unbounded worker behavior, and process loss could make a long run exceed storage or produce ambiguous task outcomes.
- Decision and rationale: use bounded regenerable feature caches keyed by immutable feature-task identity; limit worker count to the measured 1–4 range with one BLAS/OpenMP thread per worker; use durable leases, heartbeats, retry classification, atomic result envelopes, and reconciliation.
- Relevant files: `src/cache_manager.py`, `src/task_scheduler.py`, `src/pipeline_runner.py`, `provenance/profile_runner.py`, `provenance/reviewer1_optimization_plan_v1.md`, `provenance/performance_recovery_report.md`.
- Evidence: cache manager build-once/admission tests; scheduler lease/recovery tests; bounded ten-consumer fan-out, publication-boundary, parity, and resume tests; complete `p12` suite 82 passed; Sonar 80-task worker profile all successful; `airlines` stage profile and both-policy scale preflight recorded. Friend-PC throughput remains unmeasured.
- Status: **implemented-not-run** for the corrected campaign; launch remains **blocked** pending the intended host probe, representative large-task resume, and final storage margin.

## R12. Friend-PC launch audit and cache/recovery gates

- Exact concern: the local Sonar profile and earlier local disk measurement were not evidence for the friend PC, and cache/scheduler behavior lacked auditable fan-out and restart evidence.
- Decision and rationale: keep the frozen 1,680,000-cell executable scope; require a host-specific probe and sustained representative throughput before launch. Use opt-in cache audit evidence so normal manifests remain compact.
- Relevant files: `src/cache_manager.py`, `src/task_scheduler.py`, `src/pipeline_runner.py`, `provenance/measure_host.py`, `provenance/profile_runner.py`, `provenance/friend_pc_launch_audit.md`, `tests/test_bounded_runner_cache.py`, `tests/test_cache_manager.py`, `tests/test_task_scheduler.py`, `tests/test_model_factory.py`.
- Implementation: build-once cache admission is locked and double-checked; reader leases, terminal-consumer checks, retry-safe deletion deferral, per-run result-key indexing, foreign-lease reclamation, scheduler-artifact checkpoint repair, explicit XGBoost/CatBoost GPU routing, and host disk/topology probing were added. The existing notebook execution-count change is untouched.
- Evidence: bounded ten-model fan-out reports one build/nine hits; mid-group forced termination and new-process resume report no refit and no duplicate logical task. These are code-verification smokes only. The friend-PC probe and 10-day throughput evidence are not available in this session.
- Status: **UNKNOWN / DO NOT LAUNCH** pending friend-PC host measurement, representative both-policy throughput, storage margin, and restart preflight.

## R13. Integrated runner concurrency and bounded pilot

- Exact concern: the main runner was still serial; the separate profiler did not prove concurrent scheduler/cache execution.
- Decision and rationale: add a bounded Windows spawn pool for model fits while keeping scheduler leases, cache publication, checkpoints, and result writes in one coordinator. Run a four-dataset, one-seed/one-fold pilot through this real entry point.
- Relevant files: `src/pipeline_runner.py`, `tests/test_bounded_runner_cache.py`, `provenance/four_dataset_performance_pilot.py`, `provenance/four_dataset_performance_pilot.md`, `provenance/performance_optimization_change_log.md`.
- Scientific scope: four prespecified group-AUC-eligible small datasets; 10 conditions, 14 frozen pipelines, 10 classifiers, one seed, one fold, both split policies; 11,200 intended cells. No model hyperparameters or metric definitions changed.
- Evidence: focused integrated-run test proves overlapping worker PIDs and unique durable task rows. Pilot timing and completion counts are recorded separately; corrected effects remain **PENDING CORRECTED RUN**.
- Status: **pilot evidence**; friend-PC launch verdict remains **UNKNOWN / DO NOT LAUNCH**.

\r\n

## R14. Pilot checkpoint recovery

- Exact concern: the first four-dataset pilot lost coordinator leases after a transient manifest replacement error and must resume from individual durable model cells.
- Implementation: preserved the aborted run, backed up scheduler/checkpoint databases with SQLite backup API, validated 5,422 overlapping successful cells by exact hashes/metrics, imported only compatible successes, rebuilt scheduler artifacts, and resumed the same row-level run identity.
- Evidence: row-level pilot terminal manifest has 5,600/5,600 successful cells, 5,600 unique scheduler results, and 12 failed attempt rows retained only as retry history. No previously committed model fit was refit. Group-aware pilot resumed under a 600-second lease after an explicit operational-identity migration; the first 905 cells were retained; no corrected performance claim is made.
- Relevant artifacts: `provenance/pilot_checkpoint_recovery.md`, `provenance/pilot_checkpoint_recovery_initial.json`, `provenance/pilot_checkpoint_recovery_merge.json`, and `provenance/reconcile_pilot_progress.py`.
- Limitation: the row-level historical run identity used a 3,600-second lease; a fresh implementation identity is required to retrofit ten-minute leases to any future continuation of that run.
- Status: **row-level pilot recovered and complete; group-aware pilot in progress; corrected effects PENDING CORRECTED RUN**.

## R15. Four-dataset pilot recovery and runtime diagnosis

- Exact concern: distinguish historical failed attempts from terminal results, prove group-aware progress and lease safety, identify measured runtime costs, and finish the bounded pilot before any full-run decision.
- Evidence: [`four_dataset_pilot_diagnosis.md`](four_dataset_pilot_diagnosis.md) records the 15-minute, 60-second-interval live sample (225 group successes, 895 valid cells/hour), all twelve failed mirror rows and their successful retries, 23 intentional replacement attempts, zero final scheduler/checkpoint/JSONL/artifact disagreements across 11,200 successes, cache fan-out and final deletion, matched stage profiles, and the ranked scheduler/cache scan costs. Both policies ended at 5,600/5,600 successes and zero terminal failures.
- Repairs: transient Windows cache-manifest replacement retries with preserved atomic publication (`90a8eea`); an OS-backed single-coordinator fence, task-local claim reconciliation, artifact-local cache reader counts, and read-only/profile audit tools (`4686e32`). New-process crash/restart and ten-model fan-out tests pass. A post-retry 560-cell baseline took 415.5 s; task-local claims took 193.3 s, cache-local audit alone 397.5 s, and both 186.0 s. A repeated claim/both pair took 195.0/186.7 s. The committed main worktree then completed the same 560 cells in 183.6 s (2.26× baseline), with source fingerprint `6afa679eba50c26fac66f3d2f5b9c34691ca14c3a4329516998f8b476e5996f3`. All paired cells, prediction/matrix hashes, metrics, and cache build/hit/deletion counts matched. The integrated full `p12` suite passes 94 tests. No scientific configuration or 25-dataset task count changes.
- Status: **bounded pilot and verified repairs complete; corrected scientific effects PENDING CORRECTED RUN**. The full benchmark remains **DO NOT LAUNCH** pending representative large-dataset throughput, storage, friend-PC, remaining fault-matrix gates, and a refreshed code/manifest freeze.

## R16. Seven-run launch scope, all-seed AUC policy, and compact accounting (2026-10-01)

- Exact concern: the old group-aware runner stopped at a metric-infeasible dataset, and its full-grid manifest would scan and rewrite every earlier task after every new task. This contradicted the planned 70,000 explicit skips and made a 875,000-cell policy run impractical.
- Implementation: group folds are planned for every configured seed before task execution. Structural split failure still fails closed; metric-only AUC infeasibility excludes that dataset from the all-configured-seed AUC comparison and writes a per-cell skip with its actual group-fold hashes and reason. Full grids use a durable compact SQLite current-status ledger, periodic small manifest summaries, and live `check_progress` counts. A separate `--scope` selects each of the three sensitivity tracks. The seven-run freeze tool records input hashes, commands, intended/skipped counts, analysis hashes, host probe, and outstanding gates.
- Verification: focused group skip, mixed-seed exclusion, compact-ledger restart, and corrected Table 9/Figure 10 tests; the full `p12` suite passed 99 tests after the numerical-thread guard. The completed all-five-seed group audit confirms 23 complete-seed AUC-supported datasets. The earlier seed-42-only audit was insufficient to establish the all-seed denominator: `wine-quality-red` supports AUC at seed 123 but not all configured seeds.
- Artifacts: `group_seed_grid_audit_v1.json`, `reviewer1_launch_scope_v2.json`, `launch_host_measurement_v2.json`, `provenance/README.md`, `tests/test_launch_accounting.py`.
- Status: **code verified on bounded paths; corrected scientific effects PENDING CORRECTED RUN**. Representative large-dataset host calibration and a separately specified mechanism-history run remain pending.

## R17. Parallel numerical-thread identity (2026-10-01)

- Exact concern: the first large-dataset calibration started four model-fit workers without the one-thread numerical-library settings used in the earlier pilot profile. That could distort throughput and memory estimates and was absent from immutable run identity.
- Implementation: interrupted `calibration-row_level-001` after one terminal success; retained its partial ledger as diagnostic evidence. The runner now records `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` in configuration identity and rejects a parallel compact run unless each equals `1` before Python starts. The replacement calibration sets them before importing the runner and uses new `-002` IDs; the seven-run command sheet sets the same values.
- Status: **bounded code verification complete; representative `-002` calibration pending**. No full campaign started and no corrected performance effect is claimed.

## R18. Separate candidate-history and Jacobian measurement scope (2026-10-01)

- Exact concern: the primary performance runner records per-operator candidate counts but does not write candidate-level histories or Jacobians. The historical 79.66 GiB history estimate assumed 612,500 tasks without an executable full-run writer.
- Implementation: `provenance/run_mechanism_history.py` defines two separate clean-condition `AutoFE_Baseline` runs on all five seeds/folds and both split policies. It writes compressed full candidate selection histories and per-candidate scaled arithmetic Jacobian summaries using at most 256 deterministically sampled training rows and training-fold parent scales. Each feature task has a hashed identity, source and split hashes, matrix hashes, artifact checksums, a durable status file, and a coordinator lock. `provenance/analyze_mechanism_history.py` prespecifies complete-case dataset-level clean AUC gain pairing and exploratory Spearman association, with permutation/bootstrap uncertainty and a two-policy Holm correction.
- Scope and storage: 1,250 intended feature tasks, 50 prespecified group skips, 1,200 executable histories. The bounded 950-candidate `sonar` files total 76,096 compressed bytes per feature task; simple extrapolation is 0.085 GiB, and 1 GiB is reserved to cover variation and filesystem overhead. This separate scope does not alter the seven performance runs or their 2,275,000-cell denominator.
- Verification: bounded row/group `haberman` and `sonar` tasks completed; both policies' feature-matrix hashes matched the existing runner pilot on both datasets. Focused tests cover deterministic history bytes, candidate/Jacobian row alignment, four operators, feature-matrix parity, terminal-success AUC pairing, and deterministic association calculations.
- Status: **mechanism protocol ready; full histories and all scientific associations PENDING CORRECTED RUN**. The clean-only design cannot establish condition-specific or causal mechanism effects.

## R19. Corrected table and figure propagation (2026-10-01)

- Concern: new operator counts, AUC coverage, and Jacobian measurements existed in result records but did not reach dedicated paper assets.
- Implementation: `provenance/generate_corrected_assets.py` streams both complete primary ledgers into CSV tables for all 14 pipeline AUC/resource summaries, condition AUC, dataset coverage and skip reasons, and generated/rejected/duplicate/eligible/selected candidate counts deduplicated across classifiers. It renders paired AUC, selected-candidate, full-arithmetic candidate-stage, and coverage PNG/PDF figures. After the separate mechanism association is complete, it adds dataset Jacobian/AUC and association CSVs plus a PNG/PDF scatter figure. After all five sensitivity runs complete, it adds separate sensitivity AUC/count tables and a sensitivity PNG/PDF figure without pooling their estimands into the primary. It refuses incomplete or identity-mismatched runs and hashes each input in `asset_manifest.json`.
- Verification: `tests/test_corrected_paper_assets.py` injects known synthetic primary AUC, sensitivity AUC, operator counts, and Jacobian values and checks their CSV entries and rendered image files. A separate end-to-end schema smoke streamed the two completed four-dataset pilot ledgers, 5,600 terminal cells per policy, into five CSVs and four PNG/PDF figure pairs under ignored `corrected_runs/reporting_pilot_smoke/`. The custom scope, asset manifest, and figure titles visibly label that output as pilot diagnostic. Neither the synthetic values nor the four-dataset pilot are corrected scientific results; all effects remain **PENDING CORRECTED RUN**.

