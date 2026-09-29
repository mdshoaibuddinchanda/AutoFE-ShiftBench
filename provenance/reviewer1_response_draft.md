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

Response: We now encode primary training-data corruption, transductive PCA/K-means partitions, feature-availability ablation, and majority-label relabeling as separate scopes that cannot be mixed. The primary claim will use “robustness to training-data corruption.” Deployment-time distribution-shift language will be reserved for a protocol that actually changes held-out deployment data. Corrected values: **PENDING CORRECTED RUN**.

## 5. Baseline fairness

Reviewer point: Raw and AutoFE comparisons did not necessarily use matched dimensionality and selection budgets.

Response: We retain the original Raw baseline for comparability and added `Raw_CapMatched` plus full-dimensional `Raw` comparators. The result schema records source-feature counts, post-synthesis counts, retained counts, runtime, memory, operator counts, and split policy under identical folds, training data, models, and conditions. The bounded preflight exercised all three configurations; corrected benchmark values: **PENDING CORRECTED RUN**.

## 6. Individual operator effects

Reviewer point: `AutoFE_NoMultiply` excludes multiplication and division and does not isolate individual operators.

Response: We agree and label the historical configuration accurately as addition/subtraction-only. We added isolate-one-operator and leave-one-out configurations for addition, subtraction, multiplication, and division with matched depth/base/output/variance budgets and explicit finite-value handling. Corrected operator performance values: **PENDING CORRECTED RUN**.

## 7. Incomplete runs and stopping

Reviewer point: incomplete blocks and missing tasks must not be converted into zeroes or silently omitted.

Response: The corrected run manifest and checkpoint database record task identity, phase, attempt, timing, status, exception type, and error summary. The status taxonomy now includes success, failed, skipped, timed_out, and pending; the result-note builder reports expected, terminal, pending, finite-metric, and complete-case denominators by split policy. Historical counts remain labeled historical. Corrected values: **PENDING CORRECTED RUN**.

## 8. Reproducibility

Reviewer point: source, package, seed, cache, resume, and artifact identity must be reproducible.

Response: The baseline correction uses canonical SHA-256 seeds, source CSV and sidecar checksums, dataset identity, package/runtime fingerprints, immutable run/configuration identities, split-policy task keys, and run-scoped cache manifests. The Dry Bean downloader now uses UCI ID 602 rather than the erroneous OpenML mapping. The bounded preflight passed in the existing `p12` environment for both tracks; corrected benchmark values remain **PENDING CORRECTED RUN**.

## 9. Confirmatory analysis now

Reviewer point: major fixes should be resolved before the expensive corrected campaign.

Response: We froze a preflight plan and did not launch the long campaign. Both split tracks pass on the bounded `sonar` smoke; the complete group-aware design is blocked for two datasets by explicit AUC infeasibility, and candidate-history storage is estimated at approximately 79.7 GiB compressed at the observed preflight rate for the original grid. No corrected benchmark has been launched. Run ID and final numerical values: **PENDING CORRECTED RUN**.

## Historical evidence boundary

The historical ledger and manuscript values are preserved as historical evidence. They are not merged into corrected estimates and cannot be used to infer the missing row-level/group-aware comparison.
