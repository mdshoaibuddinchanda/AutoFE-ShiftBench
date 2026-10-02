# Code revision record: reviewer problems 1–2

Date: 2026-10-02

Scope: target leakage and evaluation boundaries; reproducible task seeds.

Evaluation protocol: `predictor_only_geometry_v2_baselines_operators_fsva_v1`.

Seed scheme: `sha256_canonical_json_u32_v1`.

## Repository audit

- The audited source tree was based on `85747a4672040591d8b35c53c3aa06b1b728e10c` (`Improve data preprocessing`). Work is on `revision/leakage-seed-stability`; the separate remote branch `revision/fix-leakage-provenance` was left untouched.
- No applicable `AGENTS.md` was present in the repository or its parent directories.
- The tracked project contains 22 Python source files, two notebooks, one dataset-list YAML, root launch/setup scripts, requirements, README files, manuscript and bibliography text, and ten PDF figures. The notebooks’ markdown and code cells were read. Notebook outputs were not re-executed. The PDF files were inventoried by name and size; their binary contents were not extracted.
- There are no dataset CSVs, result JSONL ledgers, SQLite checkpoints, task manifests, or failure logs in this checkout. Generated data and report directories are Git-ignored. The manuscript and figure PDFs therefore describe historical results, but the underlying row-level records could not be audited here.
- The canonical benchmark path is `main.py` or `setup_and_run.bat` into `src.pipeline_runner.main`, which precomputes fold/pipeline matrices, dispatches training workers, then writes JSONL results and SQLite checkpoints. `src.preprocessing.py` is a separate train/test utility. The active summary readers are `check_progress.py`, `stats_analysis.py`, and `plotting_q1.py`; legacy table/plot/analysis readers were also routed to the versioned ledger without changing their analysis methods. `shap_explainer.py` is imported by the runner but is not called from its training task.
- End-to-end task trace: CSV load → target separation → policy-specific split or cache → fold integrity check → training-only corruption on copies → label encoding fitted on training labels → preprocessing fitted on corrupted training inputs → train-derived DFS definitions and selection applied to held-out inputs → model fit on training rows → held-out metrics → versioned result writer and checkpoint. The smoke test exercised precompute plus one spawned `Raw`/logistic-regression worker and then resumed the completed task.

## Problem 1 — target leakage and evaluation boundaries

**Reviewer concern.** Covariate and population split geometry could include target information. The reviewer also requires fold-local learned transforms, training-only corruption, intact held-out labels, row/group separation checks, and explicit failure instead of a silent alternate split.

**Before.** `src/pipeline_runner.py` passed the full dataframe, including its target column, to `get_covariate_splits` and `get_population_splits`. The old splitters selected all numeric columns, so a numeric target entered PCA or K-means. Rare classes silently switched stratified CV to random K-fold; no-numeric and empty-cluster cases also substituted random folds. The runner already fit preprocessing on training data and transformed held-out rows, but the boundary was implicit and lacked focused coverage.

**Implemented.**

- `src/pipeline_runner.py`: `split_predictors_and_target` separates `X` and `y` before splitting. Covariate and population policies receive only `X`; stratification receives `y`. `apply_training_condition` corrupts copies of training inputs and returns unchanged held-out inputs and labels. Held-out classes absent from training raise `SplitInfeasibleError`.
- `src/splitters.py`: covariate folds retain the existing scaled numeric PCA/binning policy and population folds retain scaled numeric K-means, with the target excluded. These policies use all predictor rows to define an unsupervised partition; this remains a transductive stress test and is not a deployment estimate. Fold integrity checks enforce a complete train/test partition, disjoint rows, and one-time test coverage; optional group identifiers are checked for train/test overlap. Rare-class stratification, no-numeric geometry, empty clusters, and invalid folds now fail explicitly rather than fall back.
- The existing preprocessing fit/transform boundary in `src/pipeline_runner.py` and train-derived feature definitions in `src/feature_engineering.py` were retained. Focused tests verify train-only fit state and that changing held-out values does not change the synthesized training matrix.
- `src/protocol.py`, `src/checkpoint.py`, caches, result readers, `README.md`, and `setup_and_run.bat` give corrected outputs a new protocol identity. Old cache paths, result ledgers, and checkpoint hashes are not treated as current outputs.
- `tests/test_leakage_boundaries.py` covers target separation and geometry invariance, label-based stratification, infeasible splits, row/group integrity, train-only preprocessing/synthesis, training corruption copies, and legacy artifact rejection.

**Status.** Implemented and verified on controlled fixtures. The audited baseline did not contain a group-aware splitter or group identifier policy; the generic integrity check enforces group isolation only when a caller supplies groups. No group-aware benchmark policy is claimed as implemented.

## Problem 2 — stable seeds and paired randomness

**Reviewer concern.** Python’s salted `hash()` made corruption draws vary across processes and restarts. Random streams also need task semantics, purpose separation, pipeline pairing, and versioned run/cache identity.

**Before.** The production runner derived perturbation seeds with `hash((seed, fold, condition))`; those values depended on Python’s hash salt and omitted dataset identity and randomness purpose. Model and feature-selection seeds reused the replication seed. Metadata feature sampling called NumPy’s global RNG without a seed.

**Implemented.**

- `src/seeding.py`: canonical UTF-8 JSON (sorted keys, compact separators, preserved Unicode) is hashed with SHA-256; the first four bytes, unsigned big-endian, yield a valid uint32. The scheme is versioned as `sha256_canonical_json_u32_v1`; seed ranges and accepted identity types are documented.
- `src/pipeline_runner.py`: separate streams derive split, training corruption, feature-selection, estimator, and distribution-sampling seeds from dataset, split policy, repetition seed, fold/condition where relevant, and purpose. Split and corruption seeds are shared across pipelines; estimator seeds are shared across pipelines for paired model comparisons; selection seeds include the pipeline. The result row records the policy, scheme, and derived seeds.
- `src/data_loader.py`: metadata sampling, mutual-information estimation, and PCA use stable seeds. `src/evaluation.py` accepts the task-derived seed for large held-out distance samples.
- `src/checkpoint.py` includes protocol, seed-scheme, split-policy, and task identity in checkpoint hashes. `src/protocol.py` places caches under `data/cache/predictor_only_geometry_v2_baselines_operators_fsva_v1/sha256_canonical_json_u32_v1/` and writes `reports/tables/results_stream_predictor_only_geometry_v2_baselines_operators_fsva_v1_sha256_canonical_json_u32_v1.jsonl`. Legacy output readers and the setup message point to the active ledger.
- `tests/test_seed_scheme.py` verifies canonical ordering, task-field/purpose separation, pipeline-paired streams, process restart under different `PYTHONHASHSEED` values, scheduling-order independence, and rejection of old seed identities. `tests/test_production_task_resume.py` compares reconstructed cached inputs/seeds and executes the worker/writer/checkpoint/resume path.

**Limits.** Stable random inputs do not establish bitwise numerical determinism across libraries, CPU/GPU hardware, or GPU kernels. Such cross-hardware agreement was not measured. The active runner’s seed inputs are explicit; separate utilities retain fixed default random states and are not used by the canonical benchmark worker.

## Verification

Environment: existing Conda `P12`, Python 3.12.14. The earlier-created `.venv` directory was left intact; final verification used P12 only.

- `conda run -n P12 python -m unittest discover -s tests -v` — final result is recorded below after the problems 3–5 additions; the focused suite covers problems 1–5.
- `conda run -n P12 python -m compileall -q -f src tests` — exited successfully. It reported two pre-existing invalid-escape `SyntaxWarning`s for `\Delta` strings in `src/analysis/structural_analysis.py`; that unrelated math-label formatting was not changed.
- `git diff --check` — passed.
- The separate-process test compared derived seeds, split indices, and corruption output under `PYTHONHASHSEED=1` and `PYTHONHASHSEED=9127` — passed.
- Production smoke task limits: one synthetic dataset, one configured repetition seed, one of five folds, one clean condition and one Gaussian-noise training condition; precompute created all 14 configured pipeline caches for each condition, then one spawned `Raw` + logistic-regression training task wrote one result. A second worker invocation skipped via checkpoint. No full benchmark was run.
- The repository had no pre-existing automated test suite or declared linter/type-check configuration; the focused suite added for this work is the verification suite.

## Historical results and compatibility

- All old result rows must remain identified as legacy. Covariate/population rows may change because their original partition geometry included the target. All policies may change because split and other random streams now derive from the versioned semantic seed scheme. Old rare-class runs may also have used the now-rejected random K-fold fallback. Do not append corrected rows to the old ledger or relabel old rows.
- The existing manuscript text and PDF figures remain historical and were not edited. In particular, the Methods description of the rare-class K-fold fallback and the limitations discussion of target leakage must be aligned with corrected code after recomputation. No source ledger or datasets were available to quantify affected historical rows in this checkout.
- A complete corrected benchmark and replacement manuscript analyses remain necessary before updated numerical claims can be made. This task implemented code and passed focused tests; it did **not** complete the corrected benchmark.

## Remaining roadmap

## Problems 3–5 extension

The follow-up audit re-read the active runner, feature-generation and selection code, notebooks, README, manuscript FSVA definitions, and the prior tests. It confirmed that the previous `Raw` configuration selected at most 20 numeric base columns before modeling, while AutoFE generated candidates and applied a 100-column final cap. The previous arithmetic path delegated operator selection to Featuretools primitive names and did not persist candidate-level events or direct FSVA measurements.

### Problem 3 — fair raw-feature baselines

`src/pipeline_runner.py` now exposes distinct executable identities:

- `Raw` — historical 20-column training-variance pre-cap, retained for legacy comparability and explicitly labeled historical.
- `Raw_Full` — every eligible numeric feature after the common training-fitted preprocessing, no synthesis and no cap.
- `Raw_Capped` — every eligible base feature followed by the same post-candidate top-k cap stage (`post_candidate_topk_v1`) and variance selector used by the declared AutoFE cap.
- `Raw_Variance` and `Raw_MI` — historical capped raw variants retained with their original pre-cap and selector semantics.

`src/feature_engineering.py` records requested caps, base-feature eligibility, candidate counts, raw/generated retained counts, actual estimator dimension, selector identity, selected feature IDs, and train-only selection histories. A cap below the available dimension selects deterministically; a cap above it retains all available features without padding. All controls use the same fold, corruption, preprocessing, estimator, and seed inputs as AutoFE.

### Problem 4 — arithmetic operator ablations

`src/operator_registry.py` is the authoritative executable registry (`arithmetic_operator_registry_v1`) with safe arithmetic semantics version `safe_division_1e-12_clip_1e12_v1`. The required sets are:

| Identity | Operators |
| --- | --- |
| `full_arithmetic_v1` | add, sub, mul, div |
| `add_sub_v1` | add, sub |
| `add_sub_div_v1` | add, sub, div |
| `add_sub_mul_v1` | add, sub, mul |
| `multiply_only_v1` | mul |
| `divide_only_v1` | div |

`AutoFE_NoMultiply` remains a compatibility name for the historical `{add, sub}` result and is labeled as such; `AutoFE_AddSub` is the explicit display identity. Candidate construction uses nested expression trees, stable candidate IDs, deterministic tie ordering, and the declared safe-division policy (`1e-12` denominator threshold, finite clipping at `1e12`, explicit validity counts). Operator-set IDs and registry versions are part of pipeline/cache/checkpoint/result identity.

### Problem 5 — direct FSVA measurements and selection histories

`src/fsva.py` measures the frozen mapping from preprocessed, corrupted training inputs to the selected executed arithmetic features. It records exact analytic Jacobians with the chain rule, protected-division status, Frobenius and dimension-adjusted norms, per-output norms, deterministic finite-difference validation, and empirical amplification ratios at predeclared perturbation magnitudes (`1e-3`, `1e-2`, `5e-2`). Raw identity features are measured as a same-coordinate control. Diagnostics are bounded by a configurable row budget and use an independent stable diagnostic seed. The schema is `fsva_diagnostics_v1`.

When `--enable-fsva-diagnostics` is supplied, the production precompute/worker path writes one JSONL candidate history and one FSVA JSON artifact per pipeline/task. Histories capture candidate identity, expression, parents, operators, depth, scorer/score, eligibility, evaluated/selected decisions, rejection reason, and numerical-validity counts. Cached features without the required history and FSVA artifacts are not accepted as diagnostic-complete. `selection_stability` reports descriptive pairwise Jaccard overlap, candidate availability, and selection frequencies with an explicit empty-set convention.

The diagnostics are descriptive and frozen-map measurements. They do not refit preprocessing, synthesis, selectors, or models and do not change predictions or benchmark metrics. GPU/library bitwise determinism and causal FSVA claims remain untested.

## Verification for problems 3–5

- `conda run -n P12 python -m unittest discover -s tests -v` — **23 tests passed in 14.017 seconds** in the final run; no rerun was needed.
- `conda run -n P12 python -m compileall -q -f src tests` — successful with the two pre-existing `\Delta` invalid-escape warnings.
- `git diff --check` — passed after the reader/documentation updates.
- Baseline/operator fixtures verify full versus cap-matched dimensions, mixed-type handling, training-only selected identities, exact operator membership, forbidden nested operators, safe division, deterministic candidate IDs and tie ordering, analytic derivatives versus finite differences, raw controls, amplification budgets, and selection stability.
- The production smoke task is synthetic and bounded: one dataset, one seed, one fold, two conditions (clean and Gaussian-noise training corruption), all 14 configured baseline/operator pipelines precomputed for both conditions (28 pipeline-condition units), one spawned `Raw`/logistic-regression worker, candidate histories and FSVA artifacts generated, one JSONL result written, and a resumed task skipped by checkpoint. No full benchmark was run.
- Production precompute cost was measured in P12 on that same 60-row smoke fixture: 28 pipeline-condition units completed in **0.6215 seconds**, producing 28 history JSONL files (253,393 bytes total), 28 FSVA JSON files (134,606 bytes total), and 113 cache/meta files (507,492 bytes total). This is a small-fixture engineering measurement, not a full-benchmark runtime estimate.

Problems 6–10 remain deferred: dataset-level/hierarchical inference; complete task manifests and failure accounting; broader stopping/recovery; incomplete-run sensitivity analysis; and full dataset/code/environment/cache/analysis provenance. The diagnostic schema and pipeline identities are minimal dependencies for this stage, not a claim that problem 10 is complete.
