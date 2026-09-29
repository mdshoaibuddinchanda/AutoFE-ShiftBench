# Draft response to Reviewer #1

This draft is intentionally conservative. Any corrected benchmark value or empirical conclusion that is not yet supported is marked **PENDING CORRECTED RUN**.

## 1. FSVA validation

Reviewer point: Jacobian norms and candidate-level selection histories were not directly measured.

Response: We agree that the original evidence did not directly measure the proposed mechanism. We are adding a prespecified training-fold measurement that defines the feature map and Jacobian norm, checks analytic arithmetic derivatives against finite differences on known cases, records undefined/non-finite cases, and streams candidate identity, parent features, operator, admissibility, selection score, selection decision, and seed. The revised manuscript will distinguish association between measured amplification/selection behavior and performance from a causal proof. Corrected numerical values: **PENDING CORRECTED RUN**. Remaining limitation: Featuretools-derived expressions for which a mathematically stable derivative is unavailable will remain explicitly undefined.

## 2. Statistical analysis

Reviewer point: pooled tests treated folds, seeds, models, or conditions as independent datasets.

Response: We agree. The confirmatory unit is the dataset. Scores are summarized within dataset before paired Raw-versus-AutoFE contrasts; uncertainty intervals, exact denominators, effect sizes, and Holm-adjusted p-values will be reported. Shared-fold winner selection remains exploratory unless a nested selection analysis is added. Corrected values: **PENDING CORRECTED RUN**.

## 3. Leakage and duplicate overlap

Reviewer point: target/held-out information paths and exact duplicate groups require a complete audit.

Response: We audited the split, preprocessing, feature-selection, synthesis, caching, and analysis boundaries with target-permutation and held-out-label negative controls. The corrected protocol retains row-level folds only as a legacy-comparable sensitivity track and adds group-aware folds based on deterministic target-excluded raw-feature groups, with a zero-overlap assertion per fold. The schema screen found no exact target copies or deterministic one-feature target mappings. Row-level overlap values are recorded in `dataset_schema_audit.md`; group-aware corrected values: **PENDING CORRECTED RUN**. These controls do not rule out semantic proxies or deployment shift.

## 4. Terminology and condition handling

Reviewer point: training corruption, transductive partitions, and deployment shifts were not consistently separated.

Response: We now encode primary training-data corruption, transductive PCA/K-means partitions, feature-availability ablation, and majority-label relabeling as separate scopes that cannot be mixed. The primary claim will use “robustness to training-data corruption.” Deployment-time distribution-shift language will be reserved for a protocol that actually changes held-out deployment data. Corrected values: **PENDING CORRECTED RUN**.

## 5. Baseline fairness

Reviewer point: Raw and AutoFE comparisons did not necessarily use matched dimensionality and selection budgets.

Response: We will retain the original Raw baseline for comparability and add prespecified cap-matched Raw and full-dimensional Raw comparators. The result schema will record source-feature counts, post-synthesis counts, retained counts, runtime, and memory under identical folds, training data, models, and conditions. Corrected values: **PENDING CORRECTED RUN**.

## 6. Individual operator effects

Reviewer point: `AutoFE_NoMultiply` excludes multiplication and division and does not isolate individual operators.

Response: We agree and will label the historical configuration accurately as addition/subtraction-only. We are adding isolate-one-operator and justified leave-one-out configurations for addition, subtraction, multiplication, and division with matched budgets and explicit division-by-zero and finite-value handling. Corrected values: **PENDING CORRECTED RUN**.

## 7. Incomplete runs and stopping

Reviewer point: incomplete blocks and missing tasks must not be converted into zeroes or silently omitted.

Response: The corrected run manifest and checkpoint database record task identity, phase, attempt, timing, status, exception type, and error summary. The final result note will report coverage by dataset, split policy, condition, pipeline, model, and operator configuration, with complete-case and missingness sensitivity analyses. Historical counts remain labeled historical. Corrected values: **PENDING CORRECTED RUN**.

## 8. Reproducibility

Reviewer point: source, package, seed, cache, resume, and artifact identity must be reproducible.

Response: The baseline correction uses canonical SHA-256 seeds, source CSV and sidecar checksums, dataset identity, package/runtime fingerprints, immutable run/configuration identities, and run-scoped cache manifests. The Dry Bean downloader now uses UCI ID 602 rather than the erroneous OpenML mapping. The full suite, compile check, schema audit, group-fold audit, and bounded preflight will be rerun before freezing the campaign. Corrected values: **PENDING CORRECTED RUN**.

## 9. Confirmatory analysis now

Reviewer point: major fixes should be resolved before the expensive corrected campaign.

Response: We have frozen a preflight plan and will not launch the long campaign until both split tracks, the common core, ablations, task accounting, source hashes, and resource margins pass their gates. No corrected benchmark has been launched. Run ID and final numerical values: **PENDING CORRECTED RUN**.

## Historical evidence boundary

The historical ledger and manuscript values are preserved as historical evidence. They are not merged into corrected estimates and cannot be used to infer the missing row-level/group-aware comparison.
