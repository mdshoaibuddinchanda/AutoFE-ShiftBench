# Group-aware fold feasibility audit

This audit uses the existing `p12` environment, five folds, seed 42, and the canonical target-excluded raw-feature grouping in `src/group_splits.py`.

Datasets: 25; group split construction feasible: 25; all-fold class support: 23; all-fold ROC-AUC support: 23.

| Dataset | Rows | Groups | Conflicting-label groups | Row-level test rows matching train X | Group split | Class support | AUC support | Reasons |
|---|---:|---:|---:|---:|---|---|---|---|
| `haberman` | 306 | 283 | 6 | 11.111111% | True | supported | supported | — |
| `sonar` | 208 | 208 | 0 | 0.0% | True | supported | supported | — |
| `ionosphere` | 351 | 350 | 0 | 0.569801% | True | supported | supported | — |
| `heart-disease` | 303 | 302 | 0 | 0.660066% | True | supported | supported | — |
| `breast-cancer-wisconsin` | 569 | 569 | 0 | 0.0% | True | supported | supported | — |
| `blood-transfusion-service-center` | 748 | 502 | 31 | 39.171123% | True | supported | supported | — |
| `diabetes` | 768 | 768 | 0 | 0.0% | True | supported | supported | — |
| `titanic` | 1,309 | 1,309 | 0 | 0.0% | True | supported | supported | — |
| `credit-g` | 1,000 | 1,000 | 0 | 0.0% | True | supported | supported | — |
| `wine-quality-red` | 1,599 | 1,359 | 0 | 22.764228% | True | infeasible | infeasible | one_or_more_test_folds_missing_a_target_class |
| `kr-vs-kp` | 3,196 | 3,196 | 0 | 0.0% | True | supported | supported | — |
| `mushroom` | 8,124 | 8,124 | 0 | 0.0% | True | supported | supported | — |
| `spambase` | 4,601 | 4,207 | 3 | 11.454032% | True | supported | supported | — |
| `jm1` | 10,885 | 8,824 | 88 | 23.647221% | True | supported | supported | — |
| `PhishingWebsites` | 11,055 | 5,785 | 64 | 65.418363% | True | supported | supported | — |
| `default-of-credit-card-clients` | 30,000 | 29,944 | 21 | 0.293333% | True | supported | supported | — |
| `magic-telescope` | 19,020 | 18,905 | 0 | 0.935857% | True | supported | supported | — |
| `dry-bean-dataset` | 13,611 | 13,543 | 0 | 0.837558% | True | supported | supported | — |
| `adult` | 48,842 | 48,630 | 25 | 0.694075% | True | supported | supported | — |
| `bank-marketing` | 45,211 | 45,211 | 0 | 0.0% | True | supported | supported | — |
| `electricity` | 45,312 | 45,312 | 0 | 0.0% | True | supported | supported | — |
| `aps_failure` | 76,000 | 76,000 | 0 | 0.0% | True | supported | supported | — |
| `covertype` | 100,000 | 84,760 | 1,405 | 23.789% | True | supported | supported | — |
| `airlines` | 100,000 | 84,280 | 6,116 | 24.606% | True | supported | supported | — |
| `kddcup99` | 100,000 | 35,758 | 0 | 67.007% | True | infeasible | infeasible | a_target_class_has_fewer_rows_than_splits, a_target_class_occurs_in_fewer_groups_than_splits, one_or_more_test_folds_missing_a_target_class |

Group-aware values are feasibility diagnostics, not corrected performance estimates. A dataset is not silently substituted with row-level folds when group-aware AUC support is unavailable.
