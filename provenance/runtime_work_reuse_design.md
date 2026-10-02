# Runtime work reuse: final pre-launch optimization

The author's approval covered the remaining optimizations identified in the read-only audit. These changes preserve datasets, folds, seeds, conditions, models, feature/candidate caps, selection and stopping rules. No full benchmark is launched by this work.

## Implemented

1. Remove training hard predictions and eight unused training metrics. Keep training probability inference and the exact training ROC-AUC definition, including its historical joint ROC/PR failure boundary. All test metrics remain.
2. Reuse training corruption, one-hot encoding and numeric scaling once per identical feature input across pipeline variants. Each variant receives private DataFrames and encoded labels. The first generated variant carries the shared preprocessing cost; later variants record `preprocessing_reused=true` and zero additional preprocessing cost. This is an execution cost attribution, not a claim that preprocessing is unnecessary scientifically.
3. Publish large numeric arrays once per feature group for spawned workers. NumPy copy-on-write mappings preserve dtype/layout and prevent cross-fit mutations. Workers close mappings before their Future completes and copy predictions before closing. Smaller arrays or budget pressure fall back to pickle without changing fitted inputs.
4. Skip GPU inventory subprocess calls while every verified device already has an admitted fit. A fresh inventory is still required before admitting the next fit.

## Dynamic policy and identity

The mapping budget is the host-derived feature-cache budget: minimum of 40% total RAM and 2.5% output-volume capacity unless an explicit cache override exists. The auto threshold is that budget divided by the worker ceiling and configured classifier count. This uses the existing CPU/RAM/VRAM resource policy; it creates no environment and installs no dependencies. Both reuse and transport settings enter serialized configuration and resume identity. Full v5 commands use reuse and `--array-transport auto`.

Temporary transport files live under `corrected_runs/<run-id>/array_transport/<coordinator-epoch>/`. They are bounded by a separate disk reservation, reclaimed after all feature consumers are terminal, and regenerated on restart. Stale cleanup verifies paths and dead-owner process creation time. Durable checkpoint/scheduler/result files remain research records; historical diagnostic runs are preserved. Resume restores cumulative transport counters and peak disk use.

New result fields are `train_infer_time_s`, `scoring_time_s`, `input_open_time_s`, `array_transport` and `preprocessing_reused`. Existing `worker_elapsed_s` remains the total worker measurement. The primary resource CSV/figure displays transport counts and phase-time sums. Pipeline summaries retain means of these timing fields. Worker-time sums can overlap and are not campaign wall time.

## Deferred optimization

CPU look-ahead while a GPU task waits is not implemented. It would change producer/admission order, retain additional prepared groups and complicate lease/recovery behavior. Existing bounded evidence does not quantify enough avoidable CPU idle time to justify that change before launch. It remains a profiling opportunity, not a known launch blocker. No claim is made that every possible optimization is exhausted.

## Evidence and interpretation

Focused tests, the full suite, current-source both-policy CPU/GPU checks, and forced mapped-input recovery are bound in the v5 readiness snapshot. Real-data comparisons use exact matrices, predictions, metrics and candidate counts on CPU. GPU comparisons assert feature/candidate identity and actual supported backend, not CPU/GPU metric equality or deterministic GPU fits. Bounded timing comparisons do not establish a full-campaign ETA. Corrected operator effects remain **PENDING CORRECTED RUN**.

Timing scope: `input_open_time_s` measures opening mapped inputs inside the worker. Pickle serialization/deserialization occurs outside this interval. These phase timings are not a complete transport-cost profile; comparisons that include a forced restart are not uninterrupted speed comparisons. Host load/thermal state can change model fit times even when outputs match exactly.

Transport telemetry resumes from its last durable manifest snapshot. A hard kill can lose volatile increments since that snapshot; the scheduler attempts and committed result ledgers remain the authoritative recovery record.

Final evidence: **179 passing tests**; **1,156 bounded successful cells** (1,144 historical exact CPU and 12 GPU feature/candidate comparisons); four additional mapped-versus-pickle CPU pairs; four forced-restart proofs; four exact mechanism tasks. New diagnostic export has 15 CSVs/eight PNG-PDF pairs and a targeted mapped-GPU resource check. [Bounded timing observations](runtime_work_reuse_timing.md) and [v5 readiness](reviewer1_launch_readiness_v5.json) retain the limits above.
