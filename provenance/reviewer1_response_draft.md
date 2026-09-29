# Draft response to Reviewer #1

This draft is intentionally conservative. Any corrected benchmark value or empirical conclusion that is not yet supported is marked **PENDING CORRECTED RUN**.

## 1. FSVA validation

Reviewer point: Jacobian norms and candidate-level selection histories were not directly measured.

Response: We agree that the original evidence did not directly measure the proposed mechanism. We added a prespecified training-fold measurement contract that defines the feature map and Jacobian norm, checks analytic arithmetic derivatives against finite differences on known cases, records undefined/non-finite cases, and streams candidate identity, parent features, operator, admissibility, selection score, selection decision, and seed. The bounded real-data preflight produced 950 training-only candidate records across all four arithmetic operators with finite selection scores. The revised manuscript will distinguish association between measured amplification/selection behavior and performance from a causal proof. Corrected benchmark values: **PENDING CORRECTED RUN**. Remaining limitation: Featuretools-derived expressions for which a mathematically stable derivative is unavailable remain explicitly undefined.

## 2. Statistical analysis

Reviewer point: pooled tests treated folds, seeds, models, or conditions as independent datasets.

Response: We agree. The confirmatory unit is the dataset. `src/reviewer1_analysis.py` now summarizes within dataset before paired Raw-versus-AutoFE contrasts, reports exact expected/finite denominators, 10,000-replicate paired dataset bootstrap intervals with a frozen seed, and Holm-adjusted p-values. Shared-fold winner selection remains exploratory unless a nested selection analysis is added. Corrected values: **PENDING CORRECTED RUN**.

## 3. Leakage and duplicate overlap

Reviewer point: target/held-out information paths and exact duplicate groups require a complete audit.

Response: We audited the split, preprocessing, feature-selection, synthesis, caching, and analysis boundaries with target-permutation and held-out-label negative controls. The corrected protocol retains row-level folds only as a legacy-comparable sensitivity track and adds group-aware folds based on deterministic target-excluded raw-feature groups, with a zero-overlap assertion per fold. The all-dataset group audit found 25/25 structurally constructible group plans and 23/25 with all-fold class/AUC support; `wine-quality-red` and `kddcup99` are explicitly infeasible and are never silently replaced by row-level folds. The schema screen found no exact target copies or deterministic one-feature target mappings. Row-level overlap values and group diagnostics are in `dataset_schema_audit.md` and `group_fold_audit.md`; corrected performance values: **PENDING CORRECTED RUN**. These controls do not rule out semantic proxies or deployment shift.

## 4. Terminology and condition handling

Reviewer point: training corruption, transductive partitions, and deployment shifts were not consistently separated.

Response: We reconstructed all 14 original conditions in [`reviewer1_condition_crosswalk.md`](reviewer1_condition_crosswalk.md). The primary ten remain clean, Gaussian-noise, missing-value, and label-noise training corruptions. The four outside conditions are preserved as separate sensitivity designs: target-free but transductive PCA covariate partitions, target-free but transductive K-means population partitions, a training feature-availability ablation, and majority-label relabeling. The historical partition implementations passed the full dataframe, including the numeric target, and are therefore invalid for the corrected conclusion; the corrected runner passes target-excluded predictors and records held-out-feature use explicitly. The historical `class_prior_shift` implementation actually relabeled training labels, and `feature_removal` randomly removed columns despite its documentation, so their historical rows cannot support the original estimands. Group-aware requests for transductive partitions now fail closed rather than bypassing zero shared groups. We reserve “deployment-time distribution shift” for a protocol that changes the held-out population. Corrected values: **PENDING CORRECTED RUN**.

## 5. Baseline fairness

Reviewer point: Raw and AutoFE comparisons did not necessarily use matched dimensionality and selection budgets.

Response: We retain the original Raw baseline for comparability and added `Raw_CapMatched` plus full-dimensional `Raw` comparators. The result schema records source-feature counts, post-synthesis counts, retained counts, runtime, memory, operator counts, and split policy under identical folds, training data, models, and conditions. The bounded preflight exercised all three configurations; corrected benchmark values: **PENDING CORRECTED RUN**.

## 6. Individual operator effects

Reviewer point: `AutoFE_NoMultiply` excludes multiplication and division and does not isolate individual operators.

Response: We agree and preserve `AutoFE_NoMultiply` as the historical joint removal of multiplication and division. The actual runner truth table now records `AutoFE_Baseline` as the full arithmetic reference; separate isolate-one variants for addition, subtraction, multiplication, and division; and leave-one-out variants where `AutoFE_LeaveOut_Multiply` retains division and `AutoFE_LeaveOut_Divide` retains multiplication. The bounded smoke verified that disabled operators generate zero candidates and enabled operators generate candidates. Subtraction/division operand order, division-by-zero, near-zero denominators, nonfinite values, duplicate candidates, rejected candidates, cache keys, task IDs, result metadata, and repeated-run stability are covered by focused tests. Invalid arithmetic candidates are removed before selection and counts are recorded by operator. Corrected operator performance values: **PENDING CORRECTED RUN**.

## 7. Incomplete runs and stopping

Reviewer point: incomplete blocks and missing tasks must not be converted into zeroes or silently omitted.

Response: The corrected run manifest and checkpoint database record task identity, phase, attempt, timing, status, exception type, and error summary. The status taxonomy now includes success, failed, skipped, timed_out, and pending; the result-note builder reports expected, terminal, pending, finite-metric, and complete-case denominators by split policy. Historical counts remain labeled historical. Corrected values: **PENDING CORRECTED RUN**.

## 8. Reproducibility

Reviewer point: source, package, seed, cache, resume, and artifact identity must be reproducible.

Response: The baseline correction uses canonical SHA-256 seeds, source CSV and sidecar checksums, dataset identity, package/runtime fingerprints, immutable run/configuration identities, split-policy task keys, and run-scoped cache manifests. The Dry Bean downloader now uses UCI ID 602 rather than the erroneous OpenML mapping. The bounded preflight passed in the existing `p12` environment for both tracks; corrected benchmark values remain **PENDING CORRECTED RUN**.

## 9. Confirmatory analysis now

Reviewer point: major fixes should be resolved before the expensive corrected campaign.

Response: We froze the AUC denominator and resource gate before launch. The row-level track retains all 25 datasets. The group-aware track retains all 25 in the intended denominator, records `wine-quality-red` and `kddcup99` as explicitly skipped for their class-support/AUC reasons, and never substitutes row-level folds. The all-pipeline two-policy scope contains 1,750,000 intended task cells; 70,000 group-policy cells are explicitly infeasible. The 100,000-row `airlines` scale preflight measured 57.45 seconds for three row-level tasks and 101.89 seconds for three group-aware tasks. Although 437.14 GiB was free and the 79.66 GiB history estimate alone fits, retained feature caches project to approximately 47,378 GiB for the full scope under the current policy, so the decision is **DO NOT LAUNCH** until cache retention/storage and scope are redesigned. No corrected benchmark has been launched. Run ID and final numerical values: **PENDING CORRECTED RUN**.

Condition accounting: the 10-condition primary grid remains 875,000 intended cells per policy and 1,750,000 across both policies. Separate sensitivity planning is 175,000 row-level cells for the two transductive domain conditions, 175,000 both-policy cells for feature availability, and 175,000 both-policy cells for majority-label relabeling. These are not appended to the primary manifest. The transductive track has no group-aware executable count until a group-constrained assignment rule is prespecified; feature-availability and relabeling each have 80,500 group AUC-eligible cells and 7,000 explicit AUC skips. Historical condition rows remain visible but cannot be merged into corrected estimates. Corrected condition effects: **PENDING CORRECTED RUN**.

## 10. Performance and crash recovery gate

Reviewer point: the long-run plan needs measured bottlenecks, bounded storage, controlled parallelism, and crash-safe resume before launch.

Response: We profiled small and large local workloads in the existing `p12` environment. Sonar's complete 8-pipeline × 10-model mix took 91.1865, 46.4736, and 30.6148 seconds at 1, 2, and 4 workers, with all 240 bounded tasks successful and zero one-versus-four prediction-hash mismatches. Stage measurements include `airlines` Raw (8.6129 seconds, approximately 1.31 GiB RSS after fitting) and `AutoFE_Baseline` (29.7011 seconds, 970 candidates and 100 retained features). BLAS/OpenMP pools were fixed to one thread per worker; the local RTX 3050 was inventoried but not used. These values are local planning evidence, not friend-PC measurements or corrected results.

The retained-cache design is replaced by a lease-aware bounded regenerable manager with ready markers, checksums, locks, high-water monitoring, and cleanup after the last compatible model. The optional durable runner mode uses a WAL/FULL SQLite scheduler with immutable task identity, 3600-second leases, heartbeats, three-attempt retry classification, fsynced atomic result envelopes, and reconciliation. Focused cache and scheduler tests pass, including child-process faults before and after result rename; the complete p12 suite now passes 83 tests, including retained-versus-bounded parity, condition-scope checks, GPU routing, and interrupted new-process runner resume. The optimized versioned scope preserves the original data hashes, split policies, AUC denominator, analysis definitions, and exact primary task counts. Corrected performance, Jacobian associations, operator effects, condition effects, and statistical conclusions remain **PENDING CORRECTED RUN** until the intended host probe, storage-margin check, and full ledger are complete.

The launch audit now records an opt-in ten-consumer cache proof: one build, nine hits, active-reader history, terminal consumer IDs, and deletion after the final durable consumer. The same bounded smoke kills the process mid-group and resumes in a new process without refitting completed classifiers or double-counting a task. Scheduler publication/checkpoint recovery repairs a missing checkpoint from a verified result artifact, and startup reclaims leases owned by a replaced single-host process. XGBoost and CatBoost GPU routing is explicit, while CPU-only models retain CPU backends. The friend PC was not available for measurement, so host identity, P/E scheduling, GPU parity, sustained valid-task rate, storage margin, and the ten-day gate are **UNKNOWN**. The full frozen benchmark is **DO NOT LAUNCH** until those measurements pass; no corrected performance claim is made.

## Historical evidence boundary

The historical ledger and manuscript values are preserved as historical evidence. They are not merged into corrected estimates and cannot be used to infer the missing row-level/group-aware comparison.

\r\n
