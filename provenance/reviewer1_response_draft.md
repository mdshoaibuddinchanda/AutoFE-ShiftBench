# Draft response to Reviewer #1

This draft is intentionally conservative. Any corrected benchmark value or empirical conclusion that is not yet supported is marked **PENDING CORRECTED RUN**.

## 1. FSVA validation

Reviewer point: Jacobian norms and candidate-level selection histories were not directly measured.

Response: We agree that the original evidence did not directly measure the proposed mechanism. We added a prespecified training-fold measurement contract that defines the arithmetic feature-map Jacobian and scaled norm, checks analytic derivatives against finite differences, and records undefined cases. A separate frozen clean-condition mechanism workflow writes every candidate's identity, parents, operator, admissibility, training-only selection score and decision, plus a Jacobian summary from at most 256 sampled training rows with training-fold parent scales. It covers 625 feature tasks per split policy, with 50 explicit group-policy skips, and has a 1 GiB storage reservation. Bounded row/group smokes on `haberman` and `sonar` reproduced the runner's exact feature-matrix hashes; `sonar` generated 950 training-only records per policy across all four arithmetic operators. After corrected primary results exist, we will compare dataset-level selected-candidate sensitivity with paired clean AutoFE-minus-Raw ROC-AUC, reporting separate row/group Spearman associations and uncertainty. This is exploratory association, not causal proof or evidence for other corruption conditions. Corrected benchmark values: **PENDING CORRECTED RUN**. Featuretools expressions without a stable derivative remain explicitly undefined.

## 2. Statistical analysis

Reviewer point: pooled tests treated folds, seeds, models, or conditions as independent datasets.

Response: We agree. The confirmatory unit is the dataset. The authoritative streaming `provenance/generate_corrected_assets.py` now summarizes within dataset before paired Raw-versus-AutoFE contrasts, reports exact expected/finite denominators, 10,000-replicate paired dataset bootstrap intervals with a frozen seed, and Holm-adjusted p-values. Shared-fold winner selection remains exploratory unless a nested selection analysis is added. Corrected values: **PENDING CORRECTED RUN**.

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

Response: The corrected run manifest and checkpoint database record task identity, phase, attempt, timing, status, exception type, and error summary. The status taxonomy includes success, failed, skipped, timed_out, and pending. The streaming exporter reports expected, terminal, finite-metric and complete-dataset denominators, common-complete-dataset row/group comparisons, and primary eligible-dataset AUC [0,1] identification bounds for missing outcomes. Structural group-AUC infeasibility is reported explicitly without artificial bounds. Running/partial ledgers are refused; terminal failures remain visible and scientifically incomplete. Separate sensitivity runs have descriptive coverage/effects, without these primary bounds. Historical counts remain labeled historical. Corrected values: **PENDING CORRECTED RUN**.

## 8. Reproducibility

Reviewer point: source, package, seed, cache, resume, and artifact identity must be reproducible.

Response: The baseline correction uses canonical SHA-256 seeds, source CSV and sidecar checksums, dataset identity, package/runtime fingerprints, immutable run/configuration identities, split-policy task keys, and run-scoped cache manifests. The Dry Bean downloader now uses UCI ID 602 rather than the erroneous OpenML mapping. The bounded preflight passed in the existing `p12` environment for both tracks; corrected benchmark values remain **PENDING CORRECTED RUN**.

## 9. Confirmatory campaign

Reviewer point: major fixes should be resolved before the expensive corrected campaign.

Response: The all-five-seed audit fixes the AUC denominator: row-level retains 25datasets; group-aware retains all 25 in intended coverage but explicitly skips wine-quality-red and kddcup99 under the all-seed/all-fold support rule, leaving at most 23 datasets for primary group AUC. Neither receives row-level fallback. The v3 scientific scope retains14 pipelines and ten classifiers. Primary row/group total1,750,000 intended cells with70,000 group skips. Five separate sensitivity runs add525,000 intended cells and 14,000group skips. Total: 2,275,000 intended;84,000 planned group-AUC skips; at most 2,191,000 eligible before domain-condition skips. Transductive partitions have a row-only design; availability and relabeling have both policies. Separate clean mechanism runs add1,250 feature tasks,50 planned skips,1,200 histories and a 1 GiB reserve. Definitions/data/code/analysis are frozen before launch. Corrected numerical results and manuscript revision remain **PENDING CORRECTED RUN**.

## 10. Runtime, cache and recovery verification

Engineering follow-up: preserve the complete protocol while improving runtime, bounding storage and verifying recovery.

Response: The final audit found a parallel cache-lifecycle error missed by the previous bounded readiness check: artifacts could accumulate until run end and exhaust the cap. The updated runner reclaims a group after every planned classifier consumer reaches a durable terminal state, drains active work under temporary capacity pressure, memoizes group preparation/arrays/digests, and publishes any completed worker without waiting for a slower earlier submission. Four workers, bounded queue size, one numerical-library thread, CPU classifiers, data/folds/seeds/caps/candidate policies and stopping rules remain fixed. The cache records true staged/publication peaks and includes metadata in final admission bytes. Ready cache allowance8 GiB plus 8 GiB temporary publication headroom replaces the unsupported retained-cache requirement. Final results/checkpoints projection28.13 GiB and mechanism reserve 1 GiB require 73.26 GiB free with the doubled-results margin.

New tests cover cap pressure across groups, crash after durable publication before cache removal, input mutation isolation and completion-order dispatch. Exact scientific parity uses both split policies/all14 pipelines/all ten classifiers on a bounded four-dataset workload, plus fresh Covertype and Airlines checks and 600-second-lease forced restarts. Existing120-cell large calibrations are preserved as historical-source timing evidence. Updated source gets v3 run identities and must pass a fresh verifier; the old v2 green report cannot authorize new-source commands. The full campaign has not started. Current test/resource/timing evidence is in the final optimization and v3 readiness reports.

The exporter now writes14 primary CSVs,7PNG/PDF figure pairs and a machine-checked result record, including the prespecified ROC-AUC statistical family F1, bootstrap/Holm, common-set row/group comparisons, descriptive matched-cap/operator contrasts, candidate/resource values and primary missingness bounds. F1 is the family identifier; classifier F1 remains a separate descriptive metric. Optional Jacobian and sensitivity assets preserve their estimands. Synthetic/real-schema diagnostic exports are visibly labeled and cannot establish corrected performance effects.

The user accepts a run longer than ten days. Bounded speed measurements do not establish full-grid runtime; the older 229.69-day all-seven-run planning scenario is highly uncertain. One heavy coordinator runs at a time. Historical concurrency-induced allocation failures and interrupted attempts are preserved as diagnostic evidence. The complete [Reviewer #1 checklist](reviewer1_completion_checklist.md) and [manuscript phrase checklist](reviewer1_manuscript_phrase_checklist.md) show engineering completion separately from empirical and submission work. All corrected performance, Jacobian, operator, condition and statistical conclusions remain **PENDING CORRECTED RUN**.

## Historical evidence boundary

The historical ledger and manuscript values are preserved as historical evidence. They are not merged into corrected estimates and cannot be used to infer the missing row-level/group-aware comparison.
