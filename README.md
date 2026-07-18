<div align="center">

# AutoFE-ShiftBench 🚀

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Datasets](https://img.shields.io/badge/Datasets-25-orange.svg)]()
[![Models](https://img.shields.io/badge/Models-10-purple.svg)]()

*A reproducible, large-scale benchmark for evaluating the trade-off between predictive performance and robustness under realistic feature corruption.*

</div>

---

## 🎯 The Hook: Why This Project?

Most AutoML and Automated Feature Engineering (AutoFE) frameworks optimize for a single objective: **clean-test accuracy**. They assume that the deployment environment perfectly mirrors the training environment. 

In the real world, this is rarely true. Sensors degrade, data collection pipelines drop packets, populations shift, and annotators make mistakes. When AutoFE blindly expands the feature space (generating hundreds of interacting features), it often increases the model's surface area to these corruptions. 

**AutoFE-ShiftBench** adds a strict **robustness lens** to the evaluation of feature engineering. It systematically answers the question: *When we let AutoFE maximize accuracy on clean data, how brittle does the resulting model become when faced with realistic data shifts?*

---

## 💡 The Solution

AutoFE-ShiftBench is an end-to-end benchmarking suite that strictly separates **perturbation types** from **perturbation severity**. It compares standard, raw-feature models against AutoFE-enhanced pipelines (using Deep Feature Synthesis) across 25 diverse datasets and 10 models.

Crucially, the benchmark implements a **Nested Cross-Validation** approach where all feature engineering, imputation, and scaling are fit strictly *inside* the training fold, ensuring zero data leakage.

### Key Capabilities:
- **8 Distinct Shift Families**: Clean, Gaussian Noise, Missing Values, Label Noise, Covariate Shift, Feature Removal, Population Shift, Class Prior Shift.
- **10 Core ML Algorithms**: Including CPU-bound (Random Forest, SVM, KNN) and GPU-accelerated (XGBoost, CatBoost).
- **Automated Parallelization**: Automatically scales to utilize `cpu_count - 1` cores and available GPUs to maximize hardware efficiency.
- **Human-Readable Caching**: Intelligent resuming and cache tracking (e.g., `data/cache/adult/AutoFE_MI_s42_f1_gaussian_noise_0.05_train.pkl`).

---

## 🚀 Quick Start & Commands

The benchmark is designed to run with maximum hardware utilization out of the box (requires ~32GB RAM and an 8GB+ GPU for optimal speed). 

### Option A: One-Command Setup (Windows)
We provide a unified batch script that installs dependencies, downloads all datasets, and initiates the benchmark.
```batch
setup_and_run.bat
```

### Option B: Step-by-Step (Linux/Mac/Windows)

**1. Install Dependencies**
```bash
pip install -r requirements.txt
```

**2. Download & Prepare Datasets**
Downloads all 25 OpenML datasets, caps them at 100K rows, and generates metadata.
```bash
python -c "from src.data_loader import download_datasets_from_list; download_datasets_from_list()"
```

**3. Run the Benchmark**
Automatically detects hardware, allocates CPU cores, and assigns GPU tasks.
```bash
python -m src.pipeline_runner
```

**4. Track Progress**
Because Phase 1 (cache generation) can take 1-3 days, a built-in tracker is provided:
```bash
python -m src.check_progress
```

---

## 📊 Datasets

The benchmark relies on 25 carefully curated OpenML datasets spanning diverse domains (medical, finance, cyber-security, etc.). Datasets exceeding 100K rows are automatically downsampled to keep execution times feasible.

> Note: Small datasets run first to provide immediate results.

| # | Dataset | OpenML Link | Instances | Features | Classes | Task | Missing (%) | Numerical (%) | Categorical (%) |
|:---|:---|:---|---:|---:|---:|:---|---:|---:|---:|
| 1 | **haberman** | [Search](https://www.openml.org/search?q=haberman&type=data) | 306 | 3 | 2 | Binary | 0.0% | 66.7% | 33.3% |
| 2 | **sonar** | [Search](https://www.openml.org/search?q=sonar&type=data) | 208 | 60 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 3 | **ionosphere** | [Search](https://www.openml.org/search?q=ionosphere&type=data) | 351 | 34 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 4 | **heart-disease** | [ID: 43398](https://www.openml.org/d/43398) | 303 | 13 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 5 | **breast-cancer-wisconsin** | [Search](https://www.openml.org/search?q=breast-cancer-wisconsin&type=data) | 569 | 30 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 6 | **blood-transfusion** | [ID: 1464](https://www.openml.org/d/1464) | 748 | 4 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 7 | **diabetes** | [Search](https://www.openml.org/search?q=diabetes&type=data) | 768 | 8 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 8 | **titanic** | [ID: 40945](https://www.openml.org/d/40945) | 1309 | 13 | 2 | Binary | 22.7% | 46.2% | 53.8% |
| 9 | **credit-g** | [Search](https://www.openml.org/search?q=credit-g&type=data) | 1000 | 20 | 2 | Binary | 0.0% | 35.0% | 65.0% |
| 10 | **wine-quality-red** | [Search](https://www.openml.org/search?q=wine-quality-red&type=data) | 1599 | 11 | 6 | Multiclass | 0.0% | 100.0% | 0.0% |
| 11 | **kr-vs-kp** | [ID: 3](https://www.openml.org/d/3) | 3196 | 36 | 2 | Binary | 0.0% | 0.0% | 100.0% |
| 12 | **mushroom** | [Search](https://www.openml.org/search?q=mushroom&type=data) | 8124 | 22 | 2 | Binary | 1.4% | 0.0% | 100.0% |
| 13 | **spambase** | [Search](https://www.openml.org/search?q=spambase&type=data) | 4601 | 57 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 14 | **jm1** | [Search](https://www.openml.org/search?q=jm1&type=data) | 10885 | 21 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 15 | **PhishingWebsites** | [Search](https://www.openml.org/search?q=PhishingWebsites&type=data) | 11055 | 30 | 2 | Binary | 0.0% | 0.0% | 100.0% |
| 16 | **default-credit-card** | [Search](https://www.openml.org/search?q=default-of-credit-card-clients&type=data) | 30000 | 23 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 17 | **magic-telescope** | [Search](https://www.openml.org/search?q=magic-telescope&type=data) | 19020 | 10 | 2 | Binary | 0.0% | 100.0% | 0.0% |
| 18 | **dry-bean-dataset** | [ID: 42585](https://www.openml.org/d/42585) | 344 | 6 | 3 | Multiclass | 0.9% | 66.7% | 33.3% |
| 19 | **adult** | [Search](https://www.openml.org/search?q=adult&type=data) | 48842 | 14 | 2 | Binary | 0.9% | 14.3% | 85.7% |
| 20 | **bank-marketing** | [Search](https://www.openml.org/search?q=bank-marketing&type=data) | 45211 | 16 | 2 | Binary | 0.0% | 43.8% | 56.2% |
| 21 | **electricity** | [Search](https://www.openml.org/search?q=electricity&type=data) | 45312 | 8 | 2 | Binary | 0.0% | 87.5% | 12.5% |
| 22 | **aps_failure** | [ID: 41138](https://www.openml.org/d/41138) | 76000 | 170 | 2 | Binary | 8.3% | 100.0% | 0.0% |
| 23 | **covertype** | [Search](https://www.openml.org/search?q=covertype&type=data) | 100000 | 54 | 7 | Multiclass | 0.0% | 25.9% | 74.1% |
| 24 | **airlines** | [ID: 1169](https://www.openml.org/d/1169) | 100000 | 7 | 2 | Binary | 0.0% | 42.9% | 57.1% |
| 25 | **kddcup99** | [ID: 1113](https://www.openml.org/d/1113) | 100000 | 41 | 23 | Multiclass | 0.0% | 82.9% | 17.1% |

---

## 📂 Project Structure & Expectations

The framework is highly modular, split by pipeline responsibilities. Below is what you can expect from the directory tree:

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
│   ├── check_progress.py          # Progress tracking utility (Run this during Phase 1)
│   ├── checkpoint.py              # SQLite checkpoint DB for crash/resume support
│   ├── data_loader.py             # Downloads OpenML datasets + calculates meta-features
│   ├── evaluation.py              # Calculates 10 classification metrics + Wasserstein dist
│   ├── feature_engineering.py     # Featuretools DFS wrapper + ablation strategies
│   ├── feature_selection.py       # Variance / MI / Random feature filtering
│   ├── model.py                   # Model factory (CPU models: n_jobs=1, GPU models: CUDA)
│   ├── pipeline_runner.py         # Main orchestrator (handles multiprocessing and scheduling)
│   ├── preprocessing.py           # Standard scaling + One-Hot encoding (pipeline safe)
│   ├── shap_explainer.py          # Fast clustered SHAP importance calculation
│   ├── shift_generator.py         # Injects 8 perturbation families (Noise, Missing, Covariate, etc.)
│   ├── splitters.py               # Handles Stratified, Covariate, and Population CV splits
│   └── stats_analysis.py          # Computes Cliff's Delta, Wilcoxon, Friedman, Nemenyi
├── notebooks/
│   └── visualization.ipynb        # Interactive results exploration
├── main.py                        # Single entry point: download + run
├── setup_and_run.bat              # Windows one-command setup
└── requirements.txt               # Strict versioned Python dependencies
```

---

## 🛠️ Hardware Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **CPU** | 8 cores | 16+ cores (Intel Ultra / Ryzen 9) |
| **RAM** | 16 GB | 32 GB |
| **GPU** | None (CPU-only mode) | NVIDIA RTX 3060+ (8 GB+ VRAM) |
| **Disk** | 50 GB free | 100 GB+ NVMe SSD |

*The framework natively mitigates thread thrashing by setting `n_jobs=1` on individual model algorithms and elevating parallelism to the cross-validation task level.*

## 📜 License
See the [LICENSE](LICENSE) file for more details.
