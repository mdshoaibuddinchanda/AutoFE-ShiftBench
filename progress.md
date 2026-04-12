# Progress Log

## Date

- 2026-04-12

## Goal

- Create the requested AutoFE-ShiftBench project structure.
- Add starter code and config files.
- Track all created files and any fixes.

## Directories Created

- config/
- data/raw/
- data/processed/
- data/shifted/
- src/
- experiments/logs/
- experiments/metrics/
- experiments/results/
- reports/figures/
- reports/tables/
- reports/paper_draft/
- notebooks/

## Files Created and What Is Inside

- config/config.yaml
  - Default runtime settings: dataset path, target column, task type, split seed, output directory.
- config/dataset_list.yaml
  - Dataset registry scaffold with one sample dataset entry.
- src/data_loader.py
  - CSV loader and feature/target split helpers with validation checks.
- src/preprocessing.py
  - Basic preprocessing: deduplication, numeric median imputation, categorical mode imputation.
- src/feature_engineering.py
  - Interaction-term and ratio-feature generation for numeric columns.
- src/feature_selection.py
  - Variance-based numeric feature selection using scikit-learn.
- src/shift_generator.py
  - Synthetic shift generation functions (gaussian noise and scaling) plus variant builder.
- src/model.py
  - Baseline model builder for classification/regression and train/predict helpers.
- src/evaluation.py
  - Task-specific metric evaluation (classification and regression).
- src/statistics.py
  - Mean/std drift comparison summary between baseline and shifted data.
- src/pipeline_runner.py
  - End-to-end orchestration with train/test split, training, evaluation, and metric export.
- notebooks/sanity_checks.ipynb
  - Notebook for quick dataset shape/head/missing-value checks.
- notebooks/visualization.ipynb
  - Notebook for basic numeric feature histogram visualization.
- main.py
  - CLI entry point to load YAML config and run pipeline.
- requirements.txt
  - Core dependencies: numpy, pandas, scikit-learn, pyyaml, matplotlib.
- README.md
  - Project overview, structure summary, and quick-start run instructions.
- LICENSE
  - MIT license text.

## File Updates After Initial Creation

- src/shift_generator.py
  - Updated gaussian noise generation from deprecated pd.np access to numpy.random API.
- LICENSE
  - Added top-level heading to satisfy markdown lint rule in this workspace.

## Errors Found and Fixed

- Error found:
  - MD041/first-line-heading/first-line-h1 on LICENSE first line.
- Fix applied:
  - Changed first line from "MIT License" to "# MIT License".
- Validation result after fix:
  - No remaining errors reported by workspace diagnostics.

## Final Status

- Requested folder structure created.
- Requested files created.
- progress.md created and updated with all actions and fixes.

## Update: OpenML Dataset Benchmark Setup (2026-04-12)

### Request Coverage (Selection and Shift)

- Replaced dataset selection with the validated 12-dataset list.
- Replaced global configuration with the requested experiment settings.
- Implemented OpenML-based downloader logic in the data loader.

### File-Level Changes (Selection and Shift)

- config/config.yaml
  - Replaced previous single-dataset pipeline config with:
    - `seeds: 20`
    - `folds: 5`
    - `shift.types: [random, important]`
    - `shift.severity: [0.2, 0.4, 0.6, 0.8, 1.0]`
    - `feature_engineering.dfs_depth: 2`
    - `feature_engineering.max_features: 100`
    - `model.type: xgboost`
    - `model.params.n_estimators: 100`
    - `model.params.max_depth: 6`

- config/dataset_list.yaml
  - Replaced old placeholder entry with final dataset list:
    - `credit-g`
    - `phoneme`
    - `diabetes`
    - `bank-marketing`
    - `blood-transfusion`
    - `ilpd`
    - `kc1`
    - `pc1`
    - `adult`
    - `jungle_chess`
    - `churn`
    - `credit-approval`

- src/data_loader.py
  - Added loader to read dataset names from `config/dataset_list.yaml`.
  - Added OpenML download logic using `sklearn.datasets.fetch_openml`.
  - Added name-fallback mapping for known OpenML naming variants.
  - Implemented download output to `data/raw/{dataset_name}.csv`.
  - Implemented explicit separation of `X` (features) and `y` (target), then recombination with a standardized target column for CSV export.
  - Enforced row cap (`<= 15000`) with deterministic sampling (`random_state`) for oversized datasets.
  - Kept existing CSV load and feature-target split utilities for compatibility.
  - Added runnable module entry point to trigger full dataset downloads.

- requirements.txt
  - Added `xgboost>=2.1` to align dependencies with `model.type: xgboost` in global config.

### Issues and Fixes

- No new diagnostics errors were reported after these updates.
- Post-validation fix applied:
  - Replaced tab indentation with spaces in `config/config.yaml` to ensure YAML parser compatibility.

### Validation Status

- `config/config.yaml`: no errors
- `config/dataset_list.yaml`: no errors
- `src/data_loader.py`: no errors

## Update: OpenML Warning and Stability Fix (2026-04-12)

### Problem Observed

- Running `src/data_loader.py` emitted repeated OpenML warnings about multiple active versions for several datasets when using name-based lookup with `version="active"`.
- A prior run also showed a user-interrupted import path (`KeyboardInterrupt`) while waiting on heavy sklearn/scipy import and dataset retrieval.

### Fix Applied

- Updated `src/data_loader.py` to prefer explicit versioned OpenML fetches (`version=1`) per benchmark dataset.
- Added fallback to `version="active"` only if versioned fetch fails.
- Moved `from sklearn.datasets import fetch_openml` inside the fetch function (lazy import) to avoid importing the full sklearn stack at module import time when not needed.
- Improved error message context to include candidate names and attempted versions.

### Verification

- Single-dataset check in `py312`:
  - `download_openml_dataset("credit-g", "data/raw", ...)` completed successfully and printed output path.
- Full script check in `py312`:
  - `python src/data_loader.py` completed and emitted all 12 dataset outputs successfully.
- Result:
  - No multi-active-version OpenML warnings during the validated run after this fix.

## Update: Preprocessing and DFS Feature Engineering (2026-04-12)

### User Request Implemented

- Implemented preprocessing pipeline from `data/raw/*.csv` to `data/processed/{dataset}.pkl`.
- Enforced split-before-preprocessing to prevent data leakage.
- Implemented Featuretools DFS-based feature engineering from processed artifacts.

### Files Updated (Figure Polish)

- src/preprocessing.py
  - Added `PreprocessingConfig` and batch preprocessing CLI.
  - Added train/test split as the first operation.
  - Added numeric imputation (`mean`) and categorical imputation (`most_frequent`).
  - Added configurable encoding (`onehot` or `label`).
  - Added optional numeric scaling.
  - Added `.pkl` artifact persistence containing split data + fitted preprocessor.
  - Added loader for processed artifacts.

- src/feature_engineering.py
  - Added `DFSConfig` and batch DFS CLI.
  - Added Featuretools DFS expansion pipeline over processed train/test sets.
  - Added feature explosion controls:
    - cap base columns by variance (`max_base_features`)
    - cap final engineered matrix width (`max_features`)
  - Added RAM usage monitoring (via `psutil`, when available).
  - Added metadata capture for requested/effective DFS depth, feature counts, selected columns, and RAM usage.
  - Kept legacy interaction/ratio helper functions for backward compatibility.

- requirements.txt
  - Added `featuretools>=1.31`
  - Added `psutil>=5.9`
  - Added `setuptools<81` for `woodwork/featuretools` runtime compatibility (`pkg_resources`).

### Runtime Validation (Selection and Shift)

- Preprocessing batch run:
  - `python src/preprocessing.py --raw-dir data/raw --output-dir data/processed --encoding onehot`
  - Generated processed artifacts for all 12 datasets.
- DFS smoke test:
  - `engineer_processed_dataset('data/processed/credit-g.pkl', ...)`
  - Generated `data/processed/credit-g_features.pkl` successfully.

### Errors Found / Fixed

- Static typing issue in preprocessing conversion:
  - Potential sparse matrix to DataFrame mismatch.
  - Fix: explicit dense conversion helper returning `numpy.ndarray`.
- Missing runtime dependency:
  - `ModuleNotFoundError: featuretools`.
  - Fix: installed and pinned featuretools dependency.
- Featuretools/woodwork compatibility issue in environment:
  - `ModuleNotFoundError: pkg_resources` with newer setuptools.
  - Fix: pinned `setuptools<81` and validated runtime.

## Update: Feature Selection, Shifting, and Model Pipelines (2026-04-12)

### Request Coverage

- Implemented feature selection from engineered features to top 100 features.
- Implemented shift generation on test data only with random and important strategies.
- Implemented XGBoost model training with two pipelines:
  - Pipeline A: raw processed features
  - Pipeline B: AutoFE + selection features

### File-Level Changes

- src/feature_selection.py
  - Replaced variance-only selector with combined L1 + mutual information selection.
  - Added top-100 feature capping (`max_features`).
  - Added engineered artifact reader/writer:
    - input: `*_features.pkl`
    - output: `*_selected.pkl`
  - Added metadata storage (selected features and method scores).
  - Added CLI for batch selection.

- src/shift_generator.py
  - Replaced generic shifts with benchmark-specific generator.
  - Enforced test-only shifting (`x_test_selected`/`x_test_fe`/`x_test` source).
  - Added shift types:
    - random feature selection
    - important feature selection (from selection metadata, with variance fallback)
  - Added severity levels: 20%, 40%, 60%, 80%, 100%.
  - Added corruption rules:
    - numeric: Gaussian noise
    - categorical: random value replacement and missing injection
  - Added per-variation artifact output:
    - `data/shifted/{dataset}/{variation}.pkl`
  - Added CLI for batch generation.

- src/model.py
  - Replaced RandomForest baseline with XGBoost model builder.
  - Added two-pipeline training/comparison flow:
    - Pipeline A (`x_train`/`x_test`)
    - Pipeline B (`x_train_selected`/`x_test_selected` preferred, then AutoFE fallback)
  - Added train/test metrics and overfitting gap computation.
  - Added persisted model artifacts and JSON summary output.
  - Added feature-name sanitization and value sanitization for XGBoost compatibility.
  - Kept backward-compatible wrappers (`build_default_model`, `train_model`, `predict_model`).

### Runtime Validation

- Feature selection smoke test:
  - Generated `data/processed/credit-g_selected.pkl`.
- Shift generation smoke test:
  - Generated 10 variations in `data/shifted/credit-g/` for both shift types and all severities.
- Model smoke test:
  - Generated:
    - `experiments/results/models/credit-g/pipeline_a_raw_model.pkl`
    - `experiments/results/models/credit-g/pipeline_b_autofe_selection_model.pkl`
    - `experiments/results/models/credit-g/pipeline_comparison.json`

### Issues and Fixes (Selection and Shift)

- Feature selection error:
  - `ValueError: Input X contains infinity` during L1 fit.
  - Fix: sanitize `inf/-inf`, fill missing values, and clip extreme values before scoring.
- Shift generator warning:
  - Runtime warning in variance ranking caused by invalid numeric values.
  - Fix: sanitize `inf/-inf` and missing values before variance ranking.
- Model training error:
  - `ValueError: feature_names must be string, and may not contain [, ] or <` in XGBoost.
  - Fix: sanitize/uniquify feature names and sanitize feature matrices before fitting.

## Update: Controlled-to-Full Scaling Run (2026-04-12)

### Controlled Validation Gate (3 Datasets)

- Completed full stage validation on:
  - `credit-g`
  - `diabetes`
  - `bank-marketing`
- Verified for each of the three datasets:
  - feature engineering output exists
  - feature selection output exists
  - shift artifacts created
  - model comparison results saved
- Spot checks:
  - selected feature count for all three = 100 train/test features
  - shift files present
  - model summary JSON present

### Batch Prerequisite Gate (All 12)

- Ran full feature engineering batch for base processed artifacts.
- Ran full feature selection batch for base engineered artifacts.
- Enforced prerequisite check before full shifting:
  - every dataset in `config/dataset_list.yaml` has both:
    - `data/processed/{dataset}_features.pkl`
    - `data/processed/{dataset}_selected.pkl`
- Result:
  - `missing_prerequisites []`

### Full Shift Generation Outcome (All 12)

- Ran full batch shift generation after prerequisites passed.
- Output root confirmed:
  - `data/shifted/{dataset}/...`
- Shift files per dataset after run:
  - `credit-g`: 20
  - `phoneme`: 10
  - `diabetes`: 20
  - `bank-marketing`: 20
  - `blood-transfusion`: 10
  - `ilpd`: 10
  - `kc1`: 10
  - `pc1`: 10
  - `adult`: 10
  - `jungle_chess`: 10
  - `churn`: 10
  - `credit-approval`: 10
- Note:
  - Datasets with previous runs (`credit-g`, `diabetes`, `bank-marketing`) now contain additional `__run001` files due no-overwrite policy.

### Code Fixes During Scaling

- Updated `src/shift_generator.py`:
  - added non-overwriting save path logic (`__runNNN` suffix when variation file exists)
  - ensures reruns never overwrite prior shift files.
- Updated `src/feature_engineering.py` batch loader:
  - exclude `*_selected.pkl` from engineering inputs.
- Updated `src/feature_selection.py` batch loader:
  - exclude selected-derived engineered files from selection inputs.
- Cleanup applied:
  - removed accidental `*_selected_features.pkl` files created before batch-filter patch.

### Runtime Warning Notes

- `featuretools/woodwork` emits a `pkg_resources` deprecation warning.
- `sklearn` emitted `Liblinear failed to converge` warnings during selection on some datasets.
- Neither warning blocked artifact generation in this run.

## Update: Unified Results and Paper Outputs (2026-04-12)

### Final-Phase Implementation Scope

- Implemented a unified experiment runner to generate final experiment outputs in one workflow.
- Added aggregation logic for dataset/severity performance summaries.
- Added per-dataset Wilcoxon statistical testing and paper table generation.
- Upgraded visualization notebook to generate publication-ready figures.

### Files Updated (Results Phase)

- src/pipeline_runner.py
  - Rewritten to execute unified loops:
    - dataset loop
    - seed loop
    - shift variation loop
  - Trains Pipeline A and Pipeline B per dataset/seed.
  - Evaluates both pipelines on shifted test distributions.
  - Writes mandatory output:
    - `experiments/results/final_results.csv`
  - Preserves required columns:
    - `dataset`, `seed`, `shift_type`, `severity`, `pipeline`, `roc_auc`
  - Triggers post-processing outputs:
    - `experiments/metrics/aggregated_results.csv`
    - `experiments/metrics/statistical_results.csv`
    - `reports/tables/main_results.csv`

- src/evaluation.py
  - Added ROC-AUC computation helper for binary/multiclass settings.
  - Added aggregation function producing:
    - mean ROC-AUC per dataset/pipeline/severity
    - std deviation
    - degradation from the lowest severity baseline (20%)
  - Writes:
    - `experiments/metrics/aggregated_results.csv`

- src/statistics.py
  - Added Wilcoxon signed-rank analysis across seeds per dataset.
  - Added significance flag (`p < 0.05`).
  - Writes:
    - `experiments/metrics/statistical_results.csv`
  - Added final paper table builder with winner column.
  - Writes:
    - `reports/tables/main_results.csv`

- src/model.py
  - Added reusable helpers for final runner:
    - training one XGBoost pipeline bundle
    - inference feature alignment/sanitization to trained schema
  - Ensures shifted matrices with noisy names/values are safely evaluable.

- src/shift_generator.py
  - Added reusable helper for generating a shifted test frame from shift settings.
  - Used by final runner when aligning shift evaluation across feature spaces.

- notebooks/visualization.ipynb
  - Replaced exploratory notebook content with final-results plotting workflow.
  - Added `metadata.id` and `metadata.language` on existing cells for notebook JSON format compliance.
  - Generates and saves:
    - `reports/figures/degradation_curve.png`
    - `reports/figures/average_performance.png`

### Validation (Runner Phase)

- Smoke execution run completed via module entry:
  - `python -m src.pipeline_runner --max-datasets 3 --max-seeds 2 ...`
- Generated files confirmed:
  - `experiments/results/final_results.csv`
  - `experiments/metrics/aggregated_results.csv`
  - `experiments/metrics/statistical_results.csv`
  - `reports/tables/main_results.csv`
- Verified final results schema:
  - columns = `dataset, seed, shift_type, severity, pipeline, roc_auc`
- Smoke-run row count:
  - `120` rows (3 datasets × 2 seeds × 10 shifts × 2 pipelines)

### Notes

- The smoke run validates correctness and format.
- Full-scale execution (12 datasets × 20 seeds) can be launched with the same runner by removing `--max-datasets` and `--max-seeds` limits.

## Update: Full-Dataset Breadth Validation (2026-04-12)

### Execution Profile

- Ran unified runner across all 12 datasets with 1 seed:
  - `python -m src.pipeline_runner --max-seeds 1 ...`

### Coverage Outcome

- `experiments/results/final_results.csv`
  - rows: `240`
  - datasets covered: `12`
  - seeds covered: `1`
- `experiments/metrics/aggregated_results.csv`
  - rows: `120`
- `experiments/metrics/statistical_results.csv`
  - rows: `12`
- `reports/tables/main_results.csv`
  - rows: `12`

### Interpretation Note

- Breadth validation confirms full dataset traversal and output integrity.
- For paper-grade significance testing, run the same pipeline with full 20 seeds.

## Update: Figure Generation Execution (2026-04-12)

### Action Performed (README)

- Executed visualization workflow using `experiments/results/final_results.csv`.
- Generated mandatory paper figures in `reports/figures/`.

### Figure Outputs

- `reports/figures/degradation_curve.png`
  - size: 107105 bytes
- `reports/figures/average_performance.png`
  - size: 38773 bytes

### Status

- Required figure artifacts now exist and are ready for paper inclusion.

## Update: README Completion Pass (2026-04-12)

### Action Performed

- Rewrote `README.md` into a complete execution guide using current project state.

### Added Coverage

- Full setup instructions (conda environment + dependency install).
- End-to-end stage-by-stage commands from data download to final experiment outputs.
- Unified runner usage with fast smoke mode and full-run mode.
- Explicit per-run output list (CSV metrics, statistics, table, figures).
- Required `final_results.csv` schema documented.
- Notebook regeneration path for figures documented.
- Troubleshooting notes for common runtime warnings and figure toggling.

### Result

- New users can run the project from scratch without missing steps or hidden assumptions.

## Update: Per-Run Full Output Automation (2026-04-12)

### User Requirement

- Ensure each pipeline run emits outputs for all final artifacts (not only partial files).

### Implementation

- Updated `src/pipeline_runner.py` to auto-generate all outputs on every run:
  - `experiments/results/final_results.csv`
  - `experiments/metrics/aggregated_results.csv`
  - `experiments/metrics/statistical_results.csv`
  - `reports/tables/main_results.csv`
  - `reports/figures/degradation_curve.png`
  - `reports/figures/average_performance.png`
- Added direct script execution compatibility (`python src/pipeline_runner.py`) by handling package-path resolution.
- Added CLI options:
  - `--figure-dir` for output figure folder
  - `--skip-figures` when figure generation is intentionally disabled
- Added explicit console output lines listing every saved artifact path per run.

### Validation (Figure Polish)

- Executed:
  - `python src/pipeline_runner.py --max-datasets 1 --max-seeds 1 ...`
- Observed console output listing all six output paths.
- Confirmed artifacts exist in:
  - `reports/figures/`
  - `experiments/metrics/`
  - `reports/tables/`

## Update: Paper Figure Label/Caption Polish (2026-04-12)

### User-Facing Figure Wording Improvements

- Updated plotting labels to replace generic pipeline names with paper-ready labels:
  - `A` -> `Baseline (XGBoost)`
  - `B` -> `AutoFE Pipeline`
- Updated figure titles to publication wording:
  - degradation: `Performance Degradation Under Increasing Feature Corruption`
  - average: `Mean ROC-AUC Comparison of Baseline and AutoFE Pipelines Under Feature Shift`

### Files Updated

- `src/pipeline_runner.py`
  - Added pipeline label mapping in figure generation.
  - Updated degradation and average chart titles.
  - Added automatic caption writer for figures:
    - `reports/figures/figure_captions.md`
  - Added saved-path print for caption file in run summary.

- `notebooks/visualization.ipynb`
  - Updated degradation curve labels/titles to baseline vs AutoFE wording.
  - Updated average performance plot labels/titles accordingly.
  - Added caption markdown file write in notebook workflow.

- `reports/figures/figure_captions.md`
  - Added publication-ready captions matching both generated figures.

### Validation

- Diagnostics check:
  - `src/pipeline_runner.py`: no errors
  - `notebooks/visualization.ipynb`: no errors

## Update: Full-Run Stability and Progress Checkpointing (2026-04-13)

### Problem Observed

- Full benchmark command was being manually interrupted (`KeyboardInterrupt`) during long runs because no intermediate progress was visible and the runner repeated expensive per-seed/per-shift preparation.

### Fix Implemented

- Updated `src/pipeline_runner.py` to reduce repeated heavy operations:
  - Preloads and reuses shifted variation payloads once per dataset (instead of reloading each variation for every seed).
  - Caches pre-aligned inference matrices per pipeline feature schema.
  - Uses numpy matrix inference path for repeated `predict_proba` calls.
- Added explicit run progress logging:
  - dataset progress (`[dataset i/n]`)
  - seed progress (`[seed j/m]`)
  - elapsed minutes and generated row count.
- Added partial checkpoint output during long runs:
  - default `experiments/results/final_results.partial.csv`
  - refreshed after each dataset.
- Added new CLI flags:
  - `--checkpoint-path`
  - `--progress-every`

### Validation

- Smoke validation:
  - `python src/pipeline_runner.py --max-datasets 1 --max-seeds 1 ...`
  - Completed successfully with progress logs and all final artifacts.
- Medium validation:
  - `python src/pipeline_runner.py --max-datasets 2 --max-seeds 3 ...`
  - Completed successfully.
- Full validation (all datasets, all seeds):
  - `python src/pipeline_runner.py --task classification --n-estimators 100 --max-depth 6`
  - Completed successfully in ~3.3 minutes.
  - Final row count: `4800`.
  - Partial checkpoint removed automatically on success.
