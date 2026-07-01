# AutoFE-ShiftBench

AutoFE-ShiftBench is a reproducible, large-scale benchmark for evaluating the trade-off between predictive performance and robustness under realistic feature corruption. It compares standard, raw-feature models against Automated Feature Engineering (AutoFE) enhanced pipelines across 20 diverse datasets and 10 models.

## Why This Project?

Most AutoML and Feature Engineering evaluations optimize only for clean-test accuracy. This project adds a strict **robustness lens** by injecting synthetic perturbations (Gaussian noise, missing values, label noise) into the data. Crucially, the benchmark implements a **Nested Cross-Validation** approach where AutoFE is strictly fit *inside* the fold, ensuring zero data leakage.

---

## Experimental Protocol & Configurations

| Configuration | Details |
| :--- | :--- |
| **Datasets** | 20 OpenML tabular datasets (capped at 100,000 rows max) |
| **Task** | Classification (Binary & Multiclass) |
| **Validation Strategy** | Stratified 5-Fold Cross-Validation |
| **Replications** | 5 Random Seeds |
| **Models (10)** | Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, SVM, KNN, Gaussian Naive Bayes, MLP Neural Network |
| **Perturbations (10)** | `clean`, `gaussian_0.01`, `gaussian_0.05`, `gaussian_0.10`, `missing_0.05`, `missing_0.10`, `missing_0.20`, `label_0.05`, `label_0.10`, `label_0.20` |
| **Metrics** | ROC-AUC, PR-AUC, F1 (Macro), MCC, Balanced Accuracy, Accuracy, Log Loss, Brier Score, Precision, Recall |
| **Effect Sizes** | Cliff's Delta, Wilcoxon Signed-Rank, Friedman, Nemenyi |

---

## Datasets

Below are the 20 OpenML datasets included in this benchmark. The pipeline automatically caps datasets exceeding 100K rows (via random downsampling) to keep runtimes feasible on standard hardware.

| # | Dataset | Domain / Topic | Link |
| :--- | :--- | :--- | :--- |
| 1 | **Adult** | Income Prediction | [OpenML Search](https://www.openml.org/search?type=data&q=adult) |
| 2 | **Bank Marketing** | Marketing | [OpenML Search](https://www.openml.org/search?type=data&q=bank-marketing) |
| 3 | **APS Failure** | Industrial / Sensor | [OpenML Search](https://www.openml.org/search?type=data&q=aps_failure) |
| 4 | **Electricity** | Energy | [OpenML Search](https://www.openml.org/search?type=data&q=electricity) |
| 5 | **Covertype** | Forest Cover | [OpenML Search](https://www.openml.org/search?type=data&q=covertype) |
| 6 | **Dry Bean** | Agriculture | [OpenML Search](https://www.openml.org/search?type=data&q=dry-bean-dataset) |
| 7 | **Crop Recommendation** | Agriculture | [OpenML Search](https://www.openml.org/search?type=data&q=crop-recommendation) |
| 8 | **Breast Cancer Wisconsin** | Medical | [OpenML Search](https://www.openml.org/search?type=data&q=breast-cancer-wisconsin) |
| 9 | **Heart Disease** | Medical | [OpenML Search](https://www.openml.org/search?type=data&q=heart-disease) |
| 10 | **Diabetes** | Medical | [OpenML Search](https://www.openml.org/search?type=data&q=diabetes) |
| 11 | **Haberman Survival** | Medical | [OpenML Search](https://www.openml.org/search?type=data&q=haberman) |
| 12 | **Ionosphere** | Physics / Radar | [OpenML Search](https://www.openml.org/search?type=data&q=ionosphere) |
| 13 | **Sonar** | Physics / Sonar | [OpenML Search](https://www.openml.org/search?type=data&q=sonar) |
| 14 | **Statlog German Credit**| Finance | [OpenML Search](https://www.openml.org/search?type=data&q=credit-g) |
| 15 | **Credit Default** | Finance | [OpenML Search](https://www.openml.org/search?type=data&q=default-of-credit-card-clients) |
| 16 | **Mushroom** | Biology | [OpenML Search](https://www.openml.org/search?type=data&q=mushroom) |
| 17 | **Magic Telescope** | Astronomy | [OpenML Search](https://www.openml.org/search?type=data&q=magic-telescope) |
| 18 | **Spambase** | NLP / Email | [OpenML Search](https://www.openml.org/search?type=data&q=spambase) |
| 19 | **Wine Quality (Red)** | Chemistry | [OpenML Search](https://www.openml.org/search?type=data&q=wine-quality-red) |
| 20 | **Rice (Cammeo/Osmancik)**| Agriculture | [OpenML Search](https://www.openml.org/search?type=data&q=rice-cammeo-and-osmancik) |

---

## Quick Start & Reproduction

### 1) Environment Setup

We recommend using `uv` or `pip` to install dependencies in a virtual environment.

```bash
# Create environment
uv venv
# Activate environment (Windows)
.venv\Scripts\activate
# Install dependencies
uv pip install -r requirements.txt
```

### 2) Download & Prepare Datasets

This command will download all 20 datasets from OpenML, cap them at 100K rows, compute structural meta-features, and save them in `data/raw`.

```bash
python src/data_loader.py
```

### 3) Run the Benchmark

The benchmark uses multiprocessing to parallelize tasks while strictly avoiding thread thrashing (by forcing model algorithms to execute synchronously within the worker process). The `--n-workers` flag dictates how many datasets/folds to process simultaneously.

```bash
# Run the full benchmark suite
# (Adjust n-workers based on your CPU. It is recommended to leave 1 core free for IO)
python -m src.pipeline_runner --n-workers 4
```

Results are streamed sequentially to a JSON Lines file (`reports/tables/results_stream.jsonl`) to ensure nothing is lost during long-running benchmarks.

### Optional: Smoke Testing

If you want to quickly test the pipeline end-to-end on a single dataset, fold, and condition, run:

```bash
python -m src.pipeline_runner --max-datasets 1 --max-seeds 1 --max-folds 1 --max-conditions 1 --n-workers 1
```

---

## Project Structure

```text
AutoFE-ShiftBench/
├── config/
│   └── dataset_list.yaml       # Defines the 20 benchmark datasets to download and run
├── data/                       # (Git-ignored) Where artifacts are cached
│   └── raw/                    # Downloaded CSV datasets and JSON meta-features
├── reports/                    # (Git-ignored) Where outputs are saved
│   ├── figures/                # Generated publication plots (PDF, PNG)
│   ├── tables/
│   │   ├── results_stream.jsonl  # Streaming pipeline results (1 row per model fit)
│   │   └── statistical_results.csv # Final effect sizes and significance tests
│   └── terminal.log            # Running log of the pipeline executions
├── src/
│   ├── data_loader.py          # Downloads OpenML datasets and computes meta-features
│   ├── pipeline_runner.py      # Multiprocessing orchestrator and Nested CV loop
│   ├── model.py                # Instantiates the 10 core ML algorithms
│   ├── shift_generator.py      # Injects realistic perturbations (noise, missing, label)
│   ├── preprocessing.py        # Handles standard scaling and encoding
│   ├── feature_engineering.py  # Wrapper for automated feature generation (Featuretools)
│   ├── feature_selection.py    # Filters the generated AutoFE explosion
│   ├── evaluation.py           # Calculates the 10 core classification metrics
│   ├── statistics.py           # Computes Cliff's Delta, Wilcoxon, Friedman, Nemenyi
│   ├── shap_explainer.py       # Computes fast clustered SHAP importances
│   ├── plotting.py             # Generates the 30-figure Seaborn visual suite
│   └── verify_claims.py        # (Deprecated) Old claim verifier, superseded by statistics.py
├── task.md                     # Development tracking checklist
├── walkthrough.md              # Detailed implementation notes
├── requirements.txt            # Project dependencies
└── README.md                   # This documentation
```

---

## License

See `LICENSE` file for details.
