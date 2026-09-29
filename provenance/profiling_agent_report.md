# Bounded performance profile

**Run date:** 2026-09-29 20:02 UTC<br>
**Environment:** existing Conda `p12` at `D:\Conda\p12`<br>
**Profile commit:** `6107c9ac6e04322a41f4b2bf9b4b39d42eab12fb`<br>
**Code fingerprint:** `df9c4f1500a441909dd04b38fdacda27b84098aaf31330a39788eb28a7618cbd`

This was a bounded profiling run. It did not call `run_experiment`, create a benchmark run directory, modify the runner, or launch the full dataset × seed × fold × condition × pipeline × model campaign. The machine-readable output is [`profiling_profile.json`](profiling_profile.json), and the repeatable harness is [`profile_runner.py`](profile_runner.py).

## Workload definition

All stage and task measurements use seed 42, fold 1 of a five-fold stratified split, and the `clean` condition. Stage timing covers `Raw` and `AutoFE_Baseline` with logistic regression on the full local CSV for each named dataset.

The worker benchmark uses Sonar and the complete bounded primary mix of 8 pipelines × 10 models = **80 tasks**:

```text
Raw, Raw_Variance, Raw_MI, Raw_CapMatched,
AutoFE_Baseline, AutoFE_MI, AutoFE_Random, AutoFE_NoMultiply
```

The ten models are logistic regression, random forest, extra trees, linear SVM, k-nearest neighbors, Gaussian NB, MLP, LightGBM, XGBoost, and CatBoost. Every task completed successfully at worker counts 1, 2, and 4.

## Host and execution settings

| Item | Observed value |
|---|---|
| OS | Windows 11 `10.0.26200-SP0` |
| CPU | Intel64 Family 6 Model 141, AMD64, 8 logical CPUs |
| Python | 3.12.14 |
| GPU probe | NVIDIA GeForce RTX 3050 Laptop GPU; driver 616.64; 4096 MiB total; 1452 MiB used; 6% utilization at probe time |
| BLAS/OpenMP environment | `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1` |
| Thread pools | NumPy OpenBLAS 1 thread, SciPy OpenBLAS 1 thread, scikit-learn OpenMP 1 thread |
| Free disk at probe | 467,628,756,992 bytes of 512,092,008,448 total |
| Profile package versions | NumPy 1.26.4; pandas 2.2.3; scikit-learn 1.5.2; SciPy 1.13.1; Featuretools 1.31.0; LightGBM 4.3.0; XGBoost 2.0.3; CatBoost 1.2.5; psutil 5.9.8 |

The GPU was inventoried but not used: the profiling harness passes `use_gpu=False` to the model factory so that the worker comparison measures CPU process scaling consistently. No friend-PC or remote equivalent was available in this session; these values are from the local Windows host only.

## Stage timing and memory

The timing columns are seconds. RSS is the process resident set immediately before and after the task; the delta is not a guaranteed native peak.

| Dataset / pipeline | Rows × columns | Load + split | Preparation | Fit | Predict | Total | RSS before → after (MiB) | Features out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sonar / Raw | 208 × 61 | 0.0087 | 0.0115 | 0.0042 | 0.0003 | 0.0239 | 152.26 → 153.59 (+1.33) | 60 |
| Sonar / AutoFE_Baseline | 208 × 61 | 0.0087 | 1.2782 | 0.0882 | 0.0003 | 1.3812 | 153.70 → 175.85 (+22.15) | 100 retained / 970 generated |
| Airlines / Raw | 100,000 × 8 | 0.1231 | 1.7667 | 6.2752 | 0.1037 | 8.6129 | 180.65 → 1,337.91 (+1,157.26) | 605 one-hot columns |
| Airlines / AutoFE_Baseline | 100,000 × 8 | 0.1231 | 10.5193 | 19.0140 | 0.0222 | 29.7011 | 180.68 → 471.41 (+290.73) | 100 retained / 970 generated |

The Airlines raw task retained 605 one-hot encoded columns and reached approximately 1.31 GiB RSS after model fitting. The AutoFE task generated 970 candidate features, retained 100, and took 10.52 seconds for preparation plus 19.01 seconds for logistic-regression fitting in this single-fold probe.

The load/split value is measured once per dataset context and is repeated in the stage rows for reference. The two rows for a dataset run in one process, so RSS is affected by allocator reuse and garbage collection; compare the recorded before/after values rather than summing row deltas as a peak.

## Worker scaling

Each worker count ran the same 80 Sonar tasks in fresh spawned processes. Timings include process startup and task serialization. The task function independently prepares features for each pipeline/model cell, so this is a bounded task-level CPU profile, not a claim about a cache-aware full-run scheduler.

| Workers | Elapsed (s) | Throughput (tasks/s) | Median task (s) | P95 task (s) | Successful | Failed | Max observed child RSS (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 91.1865 | 0.8773 | 0.4876 | 3.6219 | 80 | 0 | 228.11 |
| 2 | 46.4736 | 1.7214 | 0.5413 | 3.1928 | 80 | 0 | 228.95 |
| 4 | 30.6148 | 2.6131 | 0.8516 | 3.5834 | 80 | 0 | 227.45 |

Relative to one worker, the observed speedups were 1.962× at two workers and 2.979× at four workers. This corresponds to 98.1% two-worker efficiency and 74.5% four-worker efficiency for this small task mix. The maximum reported RSS is per child after a task; aggregate memory can be several times larger when workers overlap, and the parent process is not included in that column.

All 80 prediction hashes matched between the one-worker and four-worker runs. There were no missing tasks or failures in either run.

## Cache and disk measurement

The temporary Sonar `AutoFE_Baseline` cache was measured using the project’s existing cache manifest and verification code:

| Measurement | Value |
|---|---:|
| First cache creation | 1.1730 s |
| Cache-hit read | 0.0335 s |
| Measured speedup | 35.04× |
| Cache files | 2 |
| Cache bytes after first call | 196,934 |
| Cache bytes after hit | 196,934 |
| Temporary cache removed | true |

The cache profile used a temporary directory that was deleted when the probe completed. The profiler created no new `corrected_runs/` output.

## Numerical parity

For Sonar Raw/logistic regression, the direct task was executed twice and once in a spawned one-worker process. The train-matrix, test-matrix, and prediction SHA-256 digests were identical in all three executions:

```text
train_matrix_sha256 = 99d4fa4676a675d819313da141979f6a7d4d63efb2092e9710e867d985560b3a
test_matrix_sha256  = b1c2fa11fbe20324e7f7ab173a4b2db48efda46656f67f2539944546a9ed8c02
prediction_sha256   = 5733222bdec8d49faf947f5f8c623ba02ab9bef8700e99451ae4a62db3d24b6d
```

The direct repeat and direct-versus-worker checks both returned `true`; ROC-AUC and F1 deltas were zero. The 1-versus-4-worker comparison also reported zero hash mismatches for all 80 task keys.

## Limitations and interpretation

- This is one local Windows host, one environment, one seed, one fold, and one clean condition. It is not a runtime estimate for the full 25-dataset corrected campaign.
- Airlines was profiled at the locally saved 100,000-row cap. It was not profiled across all five folds or all conditions.
- The worker benchmark bypasses `run_experiment` and intentionally rebuilds each pipeline/model cell independently. It includes process startup and serialization and does not model reuse of a feature cache across the ten models for one pipeline.
- RSS is sampled before and after each task. Native allocator peaks, aggregate parent-plus-child RSS, and disk I/O queue depth were not captured.
- GPU availability was recorded, but no GPU model execution was measured. The model factory was explicitly called with `use_gpu=False`.
- Several expected warnings occurred: Woodwork reports the `pkg_resources` deprecation, and logistic regression / linear SVM report convergence warnings on some task cells. They did not cause task failures; all 80 mix tasks returned successfully.
- No friend-PC-equivalent machine was connected or inspected, so cross-host variance remains unmeasured.

These measurements support a bounded planning conclusion: feature preparation dominates small Sonar AutoFE tasks, Airlines preprocessing and model fitting are materially more expensive than Raw, and four worker processes nearly triple throughput under one-thread BLAS settings while increasing task-tail latency. A full-run wall-time or memory budget still requires a larger preflight covering representative large datasets, cache reuse, both split policies, and the intended task scheduler.
