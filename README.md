# AutoFE-ShiftBench

AutoFE-ShiftBench is a reproducible benchmark pipeline for measuring how
feature engineering affects robustness under synthetic feature shifts.

This repository now supports a complete flow:

1. Download benchmark datasets from OpenML.
2. Preprocess with split-before-transform safeguards.
3. Engineer features with controlled DFS expansion.
4. Select top features with L1 + mutual information.
5. Generate shifted test sets across shift types and severities.
6. Train and evaluate Pipeline A (raw) vs Pipeline B (AutoFE + selection).
7. Export final results, aggregated metrics, significance tests, tables, and figures.

## Getting Started (No Bottlenecks)

Run everything from the repository root.

### 1. Create environment

```bash
conda create -n py312 python=3.12 -y
conda activate py312
```

### 2. Install dependencies

Preferred:

```bash
uv pip install --system -r requirements.txt
```

Alternative:

```bash
pip install -r requirements.txt
```

## Default Benchmark Configuration

Configured in `config/config.yaml`:

- seeds: 20
- folds: 5
- shift types: random, important
- shift severities: 0.2, 0.4, 0.6, 0.8, 1.0
- DFS depth: 2
- max features: 100
- model: xgboost (n_estimators=100, max_depth=6)

Datasets are listed in `config/dataset_list.yaml` (12 datasets).

## Full End-to-End Run

### Step 1: Download datasets

```bash
python src/data_loader.py
```

Writes CSV files to `data/raw/`.

### Step 2: Preprocess (split before preprocessing)

```bash
python src/preprocessing.py --raw-dir data/raw --output-dir data/processed --encoding onehot
```

Writes `data/processed/{dataset}.pkl`.

### Step 3: Feature engineering (DFS)

```bash
python src/feature_engineering.py --processed-dir data/processed --output-dir data/processed --depth 2 --max-features 100 --max-base-features 40
```

Writes `data/processed/{dataset}_features.pkl`.

### Step 4: Feature selection

```bash
python src/feature_selection.py --processed-dir data/processed --output-dir data/processed --max-features 100 --l1-strength 0.1 --random-state 42
```

Writes `data/processed/{dataset}_selected.pkl`.

### Step 5: Shift generation

```bash
python src/shift_generator.py --input-dir data/processed --output-root data/shifted --shift-types random,important --severities 0.2,0.4,0.6,0.8,1.0 --random-state 42
```

Writes `data/shifted/{dataset}/{variation}.pkl`.

Note: reruns never overwrite existing shift files. New files receive `__runNNN` suffixes.

### Step 6: Unified final experiment run

```bash
python src/pipeline_runner.py --task classification --n-estimators 100 --max-depth 6
```

This single command now produces all final outputs and prints their paths.

## One-Command Validation Run (Fast)

Use this for smoke checks before full 20-seed execution:

```bash
python src/pipeline_runner.py --max-datasets 1 --max-seeds 1 --task classification --n-estimators 100 --max-depth 6
```

## Final Output Files

Every unified runner execution writes:

- `reports/tables/final_results.csv`
- `reports/tables/aggregated_results.csv`
- `reports/tables/statistical_results.csv`
- `reports/tables/main_results.csv`
- `reports/figures/degradation_curve.png`
- `reports/figures/average_performance.png`
- `reports/figures/degradation_curve.tiff`
- `reports/figures/average_performance.tiff`
- `reports/figures/degradation_curve.pdf`
- `reports/figures/average_performance.pdf`

Figure generation defaults are publication-oriented:

- PNG and TIFF at 600 DPI
- Vector PDF export enabled
- Colorblind-safe palette and consistent font/line styling

`final_results.csv` columns:

- dataset
- seed
- shift_type
- severity
- pipeline
- roc_auc

## Optional Notebook Workflow

Notebook file: `notebooks/visualization.ipynb`

Run all cells to regenerate:

- `reports/figures/degradation_curve.png`
- `reports/figures/average_performance.png`
- `reports/figures/degradation_curve.tiff`
- `reports/figures/average_performance.tiff`
- `reports/figures/degradation_curve.pdf`
- `reports/figures/average_performance.pdf`

## Claim Verification (One Command)

Run this after the benchmark to verify key claims directly from generated outputs:

```bash
python src/verify_claims.py
```

This checks:

- row completeness/integrity
- baseline vs AutoFE win counts
- significance summary
- aggregate mean/variance comparison
- degradation trend slopes

## Troubleshooting

- If `featuretools` or `pkg_resources` import issues appear, reinstall dependencies from `requirements.txt`.
- `Liblinear failed to converge` warnings during feature selection may appear on some datasets; this does not block artifact generation.
- If you only want CSV/table outputs from the final runner, skip figures:

```bash
python src/pipeline_runner.py --skip-figures
```

- If a venue requires a different raster resolution:

```bash
python src/pipeline_runner.py --figure-dpi 300
```

- If you only want PNG figures and no PDFs:

```bash
python src/pipeline_runner.py --no-pdf-figures
```

- If you want to disable TIFF exports:

```bash
python src/pipeline_runner.py --no-tiff-figures
```

## Project Structure

- `config/`: runtime configuration files
- `data/raw/`: downloaded dataset CSV files
- `data/processed/`: preprocessing, engineered, and selected artifacts
- `data/shifted/`: shifted test-set variations
- `src/`: core pipeline modules
- `reports/figures/`: paper figures
- `reports/tables/`: unified CSV outputs and paper tables
- `notebooks/`: exploratory and visualization notebooks
