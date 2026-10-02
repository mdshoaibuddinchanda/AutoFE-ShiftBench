# Final runtime work reuse: bounded timing evidence

Sequential bounded observations on the same host, without randomized repetitions or control of host load/thermal state. Small-run wall time increased despite lower preprocessing work; no universal speedup or campaign ETA is asserted. GPU observations include forced restart and transport-policy changes.

| Workload | Policy | Prior v4 seconds | Current v5 seconds |
|---|---|---:|---:|
| small | row_level | 126.61 | 156.53 |
| small | group_aware | 144.58 | 169.59 |
| covertype | row_level | 157.70 | 145.23 |
| covertype | group_aware | 223.53 | 229.17 |
| airlines | row_level | 104.03 | 118.05 |
| airlines | group_aware | 213.09 | 237.81 |

Airlines values include a deliberate stop/resume; they are not uninterrupted speed comparisons. All six workloads preserve their diagnostic cells.

| Covertype KNN worker | Prior seconds | Current seconds |
|---|---:|---:|
| row_level | 108.02 | 88.36 |
| group_aware | 110.08 | 83.62 |

Training probability inference remains for training ROC-AUC. Removal of unused training hard predictions reduces redundant KNN work; these observations are not a universal speed estimate.

| Unique-feature preprocessing sum | Prior seconds | Current seconds |
|---|---:|---:|
| row_level | 0.5090 | 0.0426 |
| group_aware | 0.5041 | 0.0451 |

Detailed metadata and source/report hashes are in [runtime_work_reuse_verification_v1.json](runtime_work_reuse_verification_v1.json). The table/figure diagnostic output is under `corrected_runs/runtime_reporting_smoke_v1/`. Corrected scientific effects remain **PENDING CORRECTED RUN**.
