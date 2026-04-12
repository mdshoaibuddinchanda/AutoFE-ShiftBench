# AutoFE-ShiftBench

AutoFE-ShiftBench is a reproducible benchmark for evaluating the trade-off between
predictive performance and robustness under controlled feature corruption.

It compares:

- Pipeline A: raw features + model grid
- Pipeline B: AutoFE (DFS + selection) + model grid

on the same datasets, seeds, shifts, and metrics.

## Why This Project

Most AutoML and feature-engineering evaluations optimize only for clean-test
accuracy. This project adds a robustness lens by applying synthetic feature shifts
at multiple severities and measuring how quickly performance degrades.

## At a Glance

| Item | Value |
| --- | --- |
| Datasets | 12 OpenML tabular datasets |
| Task | Classification |
| Seeds | 20 |
| Shift Types | random, important, missing |
| Shift Severities | 0.2, 0.4, 0.6, 0.8, 1.0 |
| Models | XGBoost, Random Forest |
| AutoFE Feature Counts | 100, 200 |
| Statistical Test | Wilcoxon signed-rank |

## Empirical Snapshot (Prior XGBoost-Only Configuration)

- Baseline (A) wins on 7/12 datasets.
- AutoFE (B) wins on 5/12 datasets.
- All per-dataset A-vs-B differences are significant.
- In aggregate, A has higher mean ROC-AUC and lower variance.
- Both pipelines degrade under corruption; B degrades slightly faster on average.

These summary numbers are from the earlier XGBoost-only setup. Re-run the full benchmark
commands above to refresh tables for the expanded multi-model, multi-feature-count design.

Run verification any time with:

```bash
python src/verify_claims.py
```

## Quick Start

Run commands from the repository root.

### 1) Create environment

```bash
conda create -n py312 python=3.12 -y
conda activate py312
```

### 2) Install dependencies

Preferred:

```bash
uv pip install --system -r requirements.txt
```

Alternative:

```bash
pip install -r requirements.txt
```

## End-to-End Reproduction

### Step 1: Download datasets

```bash
python src/data_loader.py
```

### Step 2: Preprocess (split before transform)

```bash
python src/preprocessing.py --raw-dir data/raw --output-dir data/processed --encoding onehot
```

### Step 3: Feature engineering (DFS)

```bash
python src/feature_engineering.py --processed-dir data/processed --output-dir data/processed --depth 2 --max-features 200 --max-base-features 40
```

### Step 4: Feature selection

```bash
python src/feature_selection.py --processed-dir data/processed --output-dir data/processed --max-features 200 --l1-strength 0.1 --random-state 42
```

### Step 5: Shift generation

```bash
python src/shift_generator.py --input-dir data/processed --output-root data/shifted --shift-types "random,important,missing" --severities "0.2,0.4,0.6,0.8,1.0" --random-state 42
```

### Step 6: Full benchmark run

```bash
python src/pipeline_runner.py --task classification --n-estimators 100 --max-depth 6 --model-types "xgboost,random_forest" --feature-counts "100,200"
```

### Optional: Fast smoke run

```bash
python src/pipeline_runner.py --max-datasets 1 --max-seeds 1 --task classification --n-estimators 100 --max-depth 6 --model-types "xgboost,random_forest" --feature-counts "100,200" --skip-figures
```

## Outputs

Each full run writes:

- reports/tables/final_results.csv
- reports/tables/aggregated_results.csv
- reports/tables/statistical_results.csv
- reports/tables/main_results.csv
- reports/figures/degradation_curve.png
- reports/figures/degradation_curve.tiff
- reports/figures/degradation_curve.pdf
- reports/figures/average_performance.png
- reports/figures/average_performance.tiff
- reports/figures/average_performance.pdf

Column schema for final_results.csv:

- dataset
- seed
- model_type
- feature_count
- feature_count_used
- shift_type
- severity
- pipeline
- roc_auc

## Figure Export Standard

Default figure generation is publication-oriented:

- PNG and TIFF at 600 DPI
- Vector PDF export
- colorblind-safe palette
- consistent font and line styling

Useful flags:

```bash
python src/pipeline_runner.py --figure-dpi 300
python src/pipeline_runner.py --no-pdf-figures
python src/pipeline_runner.py --no-tiff-figures
python src/pipeline_runner.py --skip-figures
```

## Optional Notebook Workflow

Notebook:

- notebooks/visualization.ipynb

Run all cells to regenerate figures and captions.

## Repository Layout

- config/: runtime config files
- src/: pipeline implementation
- notebooks/: exploratory and plotting notebooks
- data/: generated raw/processed/shifted artifacts (gitignored)
- reports/: generated tables and figures (gitignored)

## Version-Control Policy

This repository intentionally does not track generated benchmark artifacts.
Generated files are reproducible from the commands above.

Local-only files excluded from git include:

- progress.md
- .vscode/

## Troubleshooting

- If featuretools import fails, reinstall dependencies from requirements.txt.
- Liblinear convergence warnings may appear during feature selection and do not
  necessarily invalidate outputs.
- Re-run claim checks after reruns:

```bash
python src/verify_claims.py
```

## License

See LICENSE.
