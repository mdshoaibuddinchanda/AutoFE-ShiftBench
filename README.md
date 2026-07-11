# AutoFE-ShiftBench

AutoFE-ShiftBench is a reproducible, large-scale benchmark for evaluating the trade-off between predictive performance and robustness under realistic feature corruption. It compares standard, raw-feature models against Automated Feature Engineering (AutoFE) enhanced pipelines across 25 diverse datasets and 10 models.

## Why This Project?

Most AutoML and Feature Engineering evaluations optimize only for clean-test accuracy. This project adds a strict **robustness lens** by injecting synthetic perturbations (Gaussian noise, missing values, label noise) into the data. Crucially, the benchmark implements a **Nested Cross-Validation** approach where AutoFE is strictly fit *inside* the fold, ensuring zero data leakage.

---

## Experimental Protocol & Configurations

| Configuration | Details |
| :--- | :--- |
| **Datasets** | 25 OpenML tabular datasets (capped at 100,000 rows max) |
| **Task** | Classification (Binary & Multiclass) |
| **Validation Strategy** | Stratified 5-Fold Cross-Validation |
| **Replications** | 5 Random Seeds |
| **Models (10)** | Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, SVM, KNN, Gaussian Naive Bayes, MLP Neural Network |
| **Perturbations** | 8 Distinct Shift Families (Clean, Gaussian Noise, Missing Values, Label Noise, Covariate Shift, Feature Removal, Population Shift, Class Prior Shift) |
| **Metrics** | ROC-AUC, PR-AUC, F1 (Macro), MCC, Balanced Accuracy, Accuracy, Log Loss, Brier Score, Precision, Recall |
| **Effect Sizes** | Cliff's Delta, Wilcoxon Signed-Rank, Friedman, Nemenyi |

---

## Hardware Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **CPU** | 8 cores | 16+ cores (Intel Ultra / Ryzen 9) |
| **RAM** | 16 GB | 32 GB |
| **GPU** | None (CPU-only mode) | NVIDIA RTX 5060+ (8 GB VRAM) |
| **Disk** | 50 GB free | 100 GB free |

The benchmark automatically detects hardware and scales parallelization:
- **CPU tasks** use `cpu_count - 1` workers (leaves 1 core for OS/IO)
- **GPU tasks** (XGBoost + CatBoost) run on a single GPU worker

---

## Datasets

| # | Dataset | Domain / Topic |
| :--- | :--- | :--- |
| 1 | **Haberman** | Medical |
| 2 | **Sonar** | Physics / Sonar |
| 3 | **Ionosphere** | Physics / Radar |
| 4 | **Heart Disease** | Medical |
| 5 | **Breast Cancer Wisconsin** | Medical |
| 6 | **Blood Transfusion** | Medical |
| 7 | **Diabetes** | Medical |
| 8 | **Titanic** | Survival |
| 9 | **Statlog German Credit** | Finance |
| 10 | **Wine Quality (Red)** | Chemistry |
| 11 | **KR-VS-KP** | Game / Chess |
| 12 | **Mushroom** | Biology |
| 13 | **Spambase** | NLP / Email |
| 14 | **JM1** | Software Defect |
| 15 | **Phishing Websites** | Cyber Security |
| 16 | **Credit Default** | Finance |
| 17 | **Magic Telescope** | Astronomy |
| 18 | **Dry Bean** | Agriculture |
| 19 | **Adult** | Income Prediction |
| 20 | **Bank Marketing** | Marketing |
| 21 | **Electricity** | Energy |
| 22 | **APS Failure** | Industrial / Sensor |
| 23 | **Covertype** | Forest Cover |
| 24 | **Airlines** | Logistics |
| 25 | **KDDCup99** | Cyber Security |

---

## The 8 Shift Families

| Shift Family             | Real-world motivation                              | Severity Levels |
| ------------------------ | -------------------------------------------------- | --------------- |
| **1. Clean**             | Baseline, no perturbation                          | N/A |
| **2. Gaussian Noise**    | Sensor measurement noise                           | 0.01, 0.05, 0.10 |
| **3. Missing Values**    | Data collection failures / dropped packets         | 5%, 10%, 20% |
| **4. Label Noise**       | Annotation errors / misclicks                      | 5%, 10%, 20% |
| **5. Covariate Shift**   | Population distribution changes (PCA-based split)  | N/A |
| **6. Feature Removal**   | Broken sensors, suddenly unavailable variables     | 20% |
| **7. Population Shift**  | Deployment to a different user/customer population | N/A |
| **8. Class Prior Shift** | Different prevalence of classes in deployment      | N/A |

---

## Quick Start & Reproduction

### Option A: One-Command Setup (Windows)

```batch
setup_and_run.bat
```

This will install dependencies, download all 25 datasets, and start the benchmark.

### Option B: Step-by-Step

#### 1) Install Dependencies

```bash
pip install -r requirements.txt
```

#### 2) Download & Prepare Datasets

```bash
python -c "from src.data_loader import download_datasets_from_list; download_datasets_from_list()"
```

#### 3) Run the Benchmark

```bash
python -m src.pipeline_runner
```

The benchmark automatically detects your CPU cores and GPU, then parallelizes accordingly.

#### 4) Smoke Test (Quick Validation)

```bash
python -m src.pipeline_runner --max-datasets 1 --max-seeds 1 --max-folds 1 --max-conditions 1
```

---

## Tracking Progress

Phase 1 (cache generation) is the longest phase and can take **1-3 days** depending on hardware. Caches are saved with **human-readable names** so you can track progress at any time.

### Cache Layout

```
data/cache/
├── adult/
│   ├── Raw_s42_f1_clean_train.pkl
│   ├── Raw_s42_f1_clean_test.pkl
│   ├── Raw_s42_f1_clean_meta.json
│   ├── AutoFE_MI_s42_f1_gaussian_noise_0.05_train.pkl
│   ├── ...
│   └── splits_s42_covariate_shift.pkl
├── diabetes/
│   ├── ...
└── ...
```

Each dataset produces **2,450 cache sets** (7 pipelines × 5 seeds × 5 folds × 14 conditions).

### Check Progress

Run the built-in progress tracker:

```bash
python -m src.check_progress
```

This shows a per-dataset progress bar:

```
======================================================================
  AutoFE-ShiftBench — Progress Report
======================================================================
  Expected caches per dataset: 2,450
  Total datasets: 25
  Total expected: 61,250
----------------------------------------------------------------------
  Dataset                             Cached   Expected   Progress
----------------------------------------------------------------------
  haberman                              2450 / 2450     ████████████████████ ✓ DONE
  sonar                                 1200 / 2450     █████████░░░░░░░░░░░  49.0%
  adult                                    0 / 2450     ░░░░░░░░░░░░░░░░░░░░   0.0%
  ...
----------------------------------------------------------------------
  TOTAL                                 3650 / 61250                         6.0%
======================================================================
```

You can also manually check by counting files:

```bash
# Count completed caches for a specific dataset (Windows)
dir /b data\cache\adult\*_train.pkl | find /c /v ""

# Count completed caches for a specific dataset (Linux/Mac)
ls data/cache/adult/*_train.pkl | wc -l
```

---

## Project Structure

```text
AutoFE-ShiftBench/
├── config/
│   └── dataset_list.yaml          # Defines the 25 benchmark datasets
├── data/                          # (Git-ignored) Generated artifacts
│   ├── raw/                       # Downloaded CSV datasets + JSON meta-features
│   └── cache/                     # Human-readable pipeline caches (per dataset)
├── reports/                       # (Git-ignored) Outputs
│   ├── figures/                   # Generated publication plots (PDF, PNG)
│   ├── tables/
│   │   └── results_stream.jsonl   # Streaming results (1 row per model fit)
│   ├── worker_logs/               # Error logs for debugging
│   └── terminal.log               # Running log
├── src/
│   ├── check_progress.py          # Progress tracking utility
│   ├── checkpoint.py              # SQLite checkpoint DB for resume support
│   ├── data_loader.py             # Downloads OpenML datasets + meta-features
│   ├── evaluation.py              # 10 classification metrics + distribution distances
│   ├── feature_engineering.py     # Featuretools DFS wrapper + ablations
│   ├── feature_selection.py       # Variance / MI / Random feature filtering
│   ├── model.py                   # 10 model factory (CPU + GPU)
│   ├── pipeline_runner.py         # Main orchestrator (parallelized)
│   ├── plotting.py                # Seaborn visual suite
│   ├── preprocessing.py           # Standard scaling + encoding
│   ├── shap_explainer.py          # SHAP feature importance
│   ├── shift_generator.py         # 8 perturbation families
│   ├── splitters.py               # Stratified / Covariate / Population splits
│   └── stats_analysis.py          # Cliff's Delta, Wilcoxon, Friedman, Nemenyi
├── notebooks/
│   └── visualization.ipynb        # Interactive exploration notebook
├── main.py                        # Single entry point: download + run
├── setup_and_run.bat              # Windows one-command setup
├── requirements.txt               # Python dependencies
├── LICENSE
└── README.md
```

---

## License

See `LICENSE` file for details.
