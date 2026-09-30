# Performance and recovery optimization change log

This log records bounded execution changes. Scientific model settings,
feature caps, candidate budgets, split identities, and metric definitions were
not changed.

| Change | Files | Reason | Before | After | Verification |
|---|---|---|---|---|---|
| Integrated bounded process workers | `src/pipeline_runner.py`, `tests/test_bounded_runner_cache.py` | The main runner task loop was serial even though the profiling harness used workers | `workers` absent/inert; one coordinator PID | `workers=2/4` submits model fits to a bounded Windows spawn pool; coordinator owns scheduler, cache, checkpoints, and JSONL | Focused integrated-run test proves overlapping timestamps, multiple worker PIDs, unique task rows, and complete status counts |
| Worker result provenance | `src/pipeline_runner.py` | Make actual overlap inspectable | Result rows had no worker timing identity | Rows include worker PID, start/end Unix timestamps, elapsed fit time, and backend | Pilot manifests and result rows; no AUC interpretation |
| Bounded queue and durable publication | `src/pipeline_runner.py` | Prevent queued matrices from exhausting RAM and avoid concurrent writers | Synchronous phase-2 fit/write | At most `2 × workers` serialized futures; parent publishes result, scheduler artifact, checkpoint, and manifest | Existing recovery and cache tests plus integrated concurrency smoke |
| Four-dataset pilot driver | `provenance/four_dataset_performance_pilot.py` | Exercise both split policies through the real runner | No integrated pilot artifact | Prespecified four-dataset, one-seed/one-fold, 10-condition, 14-pipeline, 10-model pilot | `four_dataset_performance_pilot.md/json` |

The worker pool is an execution optimization only. Any numerical parity check
is limited to matrix/prediction hashes and bounded smoke evidence; corrected
benchmark effects remain **PENDING CORRECTED RUN**.
