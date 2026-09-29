# Performance, storage, and crash-recovery report

Date: 2026-09-30  
Environment: existing Conda `p12` (`D:\Conda\p12`)  
Campaign status: **not launched; corrected results pending**

## Evidence artifacts

- [`profiling_profile.json`](profiling_profile.json) and [`profiling_agent_report.md`](profiling_agent_report.md): bounded stage and worker measurements.
- [`reviewer1_scale_preflight_manifest.json`](reviewer1_scale_preflight_manifest.json): 100,000-row `airlines` scale gate.
- [`storage_agent_report.md`](storage_agent_report.md): lease-aware cache implementation and nine focused tests.
- [`scheduler_agent_report.md`](scheduler_agent_report.md): durable scheduler implementation and seven process-boundary tests.
- [`reviewer1_optimization_scope_v1.json`](reviewer1_optimization_scope_v1.json): frozen counts, hashes, and launch settings.

## Machine and execution settings

The local Windows 11 host is an Intel i5-11260H with 4 physical and 8 logical cores, 31.73 GiB RAM, NVMe storage, and an RTX 3050 Laptop GPU.  The profile fixed `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`; NumPy, SciPy, and scikit-learn thread pools each reported one thread.  The GPU was probed but all model calls used `use_gpu=False`.  The local `D:` volume had 435.51 GiB free when the host probe was recorded.  These measurements are not the friend's reported Ultra7/RTX5060 machine; the probe must be run there before launch.

## Bottleneck measurements

The bounded stage profile used seed 42, fold 1 of 5, clean data:

| Dataset/pipeline | Preparation (s) | Fit (s) | Total (s) | RSS after (MiB) | Feature result |
|---|---:|---:|---:|---:|---|
| Sonar / Raw | 0.0115 | 0.0042 | 0.0239 | 153.59 | 60 |
| Sonar / AutoFE_Baseline | 1.2782 | 0.0882 | 1.3812 | 175.85 | 100 retained / 970 generated |
| Airlines / Raw | 1.7667 | 6.2752 | 8.6129 | 1,337.91 | 605 one-hot |
| Airlines / AutoFE_Baseline | 10.5193 | 19.0140 | 29.7011 | 471.41 | 100 retained / 970 generated |

The profile's complete 8-pipeline × 10-model Sonar mix had 80 successful tasks at each worker count:

| Workers | Elapsed (s) | Throughput (tasks/s) | Speedup vs 1 | P95 task (s) | Max child RSS (MiB) |
|---:|---:|---:|---:|---:|---:|
| 1 | 91.1865 | 0.8773 | 1.000x | 3.6219 | 228.11 |
| 2 | 46.4736 | 1.7214 | 1.962x | 3.1928 | 228.95 |
| 4 | 30.6148 | 2.6131 | 2.979x | 3.5834 | 227.45 |

One-to-four-worker prediction hashes had zero mismatches.  A temporary Sonar AutoFE cache took 1.1730 seconds to create and 0.0335 seconds to read, a 35.04x hit speedup at 196,934 bytes.  The airlines scale gate measured 57.45 seconds for three row-level tasks and 101.89 seconds for three group-aware tasks, with approximately 0.87 GB of retained feature-cache files under the old policy.  These are planning measurements, not corrected model contrasts.

## Storage and dependency controls

The old retention policy projected 47,377.68 GiB of feature cache for all 14 pipelines and both policies, so it fails the local 435.51 GiB volume even though the 79.66 GiB candidate-history estimate plus 20.87 GiB ancillary allowance would leave roughly 334.98 GiB.  The integrated bounded manager writes payload, manifest, and ready marker atomically; validates size and SHA-256; leases reads; refuses replacement or deletion while active or malformed leases exist; reconciles incomplete/corrupt/temp state; records high-water bytes; and performs age/LRU cleanup.  The runner deletes a feature artifact after the last compatible model consumer and performs final reconciliation.  Cache identities include source, split, condition, pipeline, configuration, code, runtime, and indices, so feature matrices are never reused across incompatible tasks.

## Crash recovery and fault injection

The scheduler uses SQLite WAL with `synchronous=FULL`, immutable task specifications, a 3-attempt retry budget, 3600-second leases, heartbeats before preparation and fitting, and explicit transient versus deterministic failure classification.  Result envelopes are fsynced to a temporary file, atomically renamed, then committed to SQLite.  Reconciliation validates envelope identity and digest, adopts a result renamed before a process crash, expires dead leases, and removes stale temporary debris.  Child-process tests cover crashes both after temporary fsync and after final rename; all seven scheduler tests pass.  The runner's optional durable mode is exercised by the bounded-runner test, which also confirms two scheduler result artifacts and cache cleanup.

## Interpretation and remaining gates

The measured 4-worker speedup supports a planning scenario, not a guarantee for every dataset or the full grid.  Aggregate memory can approach four times the per-child RSS, and the friend's hardware remains unmeasured.  The optimized scope retains the original 1,750,000 intended task cells, AUC denominator, split policies, data hashes, and analysis definitions.  Runtime scenarios and storage margin must be rechecked on the launch host, followed by a new-process interrupted preflight/resume.  Until those gates pass, corrected performance, Jacobian associations, operator effects, and statistical conclusions remain **PENDING CORRECTED RUN**.
