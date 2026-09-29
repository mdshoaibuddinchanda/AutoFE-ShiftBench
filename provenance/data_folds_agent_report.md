# Group-aware raw-feature fold audit

Date: 2026-09-29  
Environment: `D:\Conda\p12`  
Scope: 25 saved `data/raw/*.csv` files; no benchmark training was run.

## Method

`src/group_splits.py` defines the separate group-aware evaluation track. It
requires an explicit target-excluded raw predictor frame. Each row receives a
canonical key containing column names, logical dtype families, and typed cell
values. Missing scalar values use one sentinel; numeric values use normalized
decimal text (`1` and `1.0` are equal within numeric columns); text/category
values remain text/category typed, so text `"1"` is distinct from numeric `1`.
The SHA-256 group ID is checked against the complete canonical key for every
digest, so a digest collision cannot silently merge different rows.

Folds use sklearn `StratifiedGroupKFold` with `shuffle=True` and the supplied
seed. There is no row-level fallback. The integrity check asserts row bounds,
held-out-once coverage, train/test row disjointness, and zero shared exact
feature groups. Class support and ROC-AUC support are reported separately from
structural split feasibility. ROC-AUC is marked supported only when every
train and test fold contains every global target class.

## Targeted tests

Command:

```text
D:\Conda\p12\python.exe -m pytest -q tests/test_group_splits.py
```

Result: **7 passed, 5 warnings**. The warnings are sklearn notices in tests
that intentionally construct undersupported classes to verify explicit status
handling. No new environment was created.

## Saved-data results

All 25 configured CSVs produced a deterministic five-fold group plan with
`split_feasible=True`; the audit asserted zero shared feature groups for the
folds it constructed. `conflicts` counts exact-feature groups containing more
than one target label.

| Dataset | Rows | Exact groups | Conflicts | Class support | AUC support | Notes |
|---|---:|---:|---:|---|---|---|
| adult | 48,842 | 48,630 | 25 | supported | supported | |
| airlines | 100,000 | 84,280 | 6,116 | supported | supported | |
| aps_failure | 76,000 | 76,000 | 0 | supported | supported | |
| bank-marketing | 45,211 | 45,211 | 0 | supported | supported | |
| blood-transfusion-service-center | 748 | 502 | 31 | supported | supported | |
| breast-cancer-wisconsin | 569 | 569 | 0 | supported | supported | |
| covertype | 100,000 | 84,760 | 1,405 | supported | supported | |
| credit-g | 1,000 | 1,000 | 0 | supported | supported | |
| default-of-credit-card-clients | 30,000 | 29,944 | 21 | supported | supported | |
| diabetes | 768 | 768 | 0 | supported | supported | |
| dry-bean-dataset | 13,611 | 13,543 | 0 | supported | supported | |
| electricity | 45,312 | 45,312 | 0 | supported | supported | |
| haberman | 306 | 283 | 6 | supported | supported | |
| heart-disease | 303 | 302 | 0 | supported | supported | |
| ionosphere | 351 | 350 | 0 | supported | supported | |
| jm1 | 10,885 | 8,824 | 88 | supported | supported | |
| kddcup99 | 100,000 | 35,758 | 0 | infeasible | infeasible | A class has fewer than five rows/groups; at least one test fold misses a class |
| kr-vs-kp | 3,196 | 3,196 | 0 | supported | supported | |
| magic-telescope | 19,020 | 18,905 | 0 | supported | supported | |
| mushroom | 8,124 | 8,124 | 0 | supported | supported | |
| PhishingWebsites | 11,055 | 5,785 | 64 | supported | supported | |
| sonar | 208 | 208 | 0 | supported | supported | |
| spambase | 4,601 | 4,207 | 3 | supported | supported | |
| titanic | 1,309 | 1,309 | 0 | supported | supported | |
| wine-quality-red | 1,599 | 1,359 | 0 | infeasible | infeasible | At least one test fold misses a target class |

The two metric-support limitations are explicit audit outcomes. The fold
construction itself remains structurally feasible, so accuracy-like metrics
can be considered under a documented policy; ROC-AUC tasks must be recorded as
unsupported/skipped unless the fold count or metric protocol is changed in a
future frozen run plan. The implementation does not silently change the fold
count or substitute row-level folds.

## Files

- `src/group_splits.py`: canonical keys, collision checks, group summaries,
  feasibility status, sklearn group folds, and integrity assertions.
- `tests/test_group_splits.py`: deterministic canonicalization, collision,
  conflict, infeasibility, target exclusion, and zero-overlap tests.

