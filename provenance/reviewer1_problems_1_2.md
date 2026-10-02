# Code revision record: reviewer problems 1–2

Date: 2026-10-02

Scope: target leakage and evaluation boundaries; reproducible task seeds.

Evaluation protocol: `predictor_only_geometry_v2`.

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
- `src/checkpoint.py` includes protocol, seed-scheme, split-policy, and task identity in checkpoint hashes. `src/protocol.py` places caches under `data/cache/predictor_only_geometry_v2/sha256_canonical_json_u32_v1/` and writes `reports/tables/results_stream_predictor_only_geometry_v2_sha256_canonical_json_u32_v1.jsonl`. Legacy output readers and the setup message point to the active ledger.
- `tests/test_seed_scheme.py` verifies canonical ordering, task-field/purpose separation, pipeline-paired streams, process restart under different `PYTHONHASHSEED` values, scheduling-order independence, and rejection of old seed identities. `tests/test_production_task_resume.py` compares reconstructed cached inputs/seeds and executes the worker/writer/checkpoint/resume path.

**Limits.** Stable random inputs do not establish bitwise numerical determinism across libraries, CPU/GPU hardware, or GPU kernels. Such cross-hardware agreement was not measured. The active runner’s seed inputs are explicit; separate utilities retain fixed default random states and are not used by the canonical benchmark worker.

## Verification

Environment: existing Conda `P12`, Python 3.12.14. The earlier-created `.venv` directory was left intact; final verification used P12 only.

- `conda run -n P12 python -m unittest discover -s tests -v` — passed, 16 tests.
- `conda run -n P12 python -m compileall -q -f src tests` — exited successfully. It reported two pre-existing invalid-escape `SyntaxWarning`s for `\Delta` strings in `src/analysis/structural_analysis.py`; that unrelated math-label formatting was not changed.
- `git diff --check` — passed.
- The separate-process test compared derived seeds, split indices, and corruption output under `PYTHONHASHSEED=1` and `PYTHONHASHSEED=9127` — passed.
- Production smoke task limits: one synthetic dataset, one configured repetition seed, one of five folds, one Gaussian-noise condition; precompute created the seven pipeline caches for that fold/condition, then one spawned `Raw` + logistic-regression training task wrote one result. A second worker invocation skipped via checkpoint. No full benchmark was run.
- The repository had no pre-existing automated test suite or declared linter/type-check configuration; the 16 discovered tests are the focused suite added for this work.

## Historical results and compatibility

- All old result rows must remain identified as legacy. Covariate/population rows may change because their original partition geometry included the target. All policies may change because split and other random streams now derive from the versioned semantic seed scheme. Old rare-class runs may also have used the now-rejected random K-fold fallback. Do not append corrected rows to the old ledger or relabel old rows.
- The existing manuscript text and PDF figures remain historical and were not edited. In particular, the Methods description of the rare-class K-fold fallback and the limitations discussion of target leakage must be aligned with corrected code after recomputation. No source ledger or datasets were available to quantify affected historical rows in this checkout.
- A complete corrected benchmark and replacement manuscript analyses remain necessary before updated numerical claims can be made. This task implemented code and passed focused tests; it did **not** complete the corrected benchmark.

## Remaining roadmap

Problems 3–10 were not implemented: fair raw-feature baselines; separate arithmetic ablations; direct FSVA measurements and selection histories; dataset-level/hierarchical inference; complete task manifests and failure accounting; broader reproducible stopping/recovery; incomplete-run sensitivity analysis; and full dataset/code/environment/cache/analysis provenance. Split infeasibility is explicit in the splitter API, while the existing worker still logs caught exceptions rather than producing the structured failure ledger deferred to problem 7.
