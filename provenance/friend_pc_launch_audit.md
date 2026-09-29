# Friend PC launch audit

Date: 2026-09-30 (Asia/Calcutta)  
Repository: `D:\DR2\AutoFE_Submission`  
Environment used for bounded verification: existing `p12` (`D:\Conda\p12\python.exe`)  
Launch-audit code commit: `75c4420008d29fab5171cf9013f3b74ccdc1fc6e`  
Full benchmark: **not launched**

## Launch verdict

**UNKNOWN — do not launch the frozen benchmark yet.** The friend PC was not available in this session, so its CPU topology, RAM headroom, GPU backend support, drive throughput, thermal behavior, and sustained benchmark rate are unmeasured. The complete frozen scope requires 1,680,000 valid executable cells and at least 7,000 valid completed cells/hour for ten days, with additional margin. The local Sonar profile and the older local `D:` free-space measurement are not evidence for that host.

The current runner is serial at the task-loop level. The profiling harness measures bounded worker pools, but those workers are not yet the full campaign executor. Therefore the prior worker speedup is a planning observation, not a launch-rate measurement.

## Frozen scope checks

| Gate | Evidence | Status |
|---|---|---|
| 14 pipeline operator identity | `reviewer1_operator_ablation_manifest_v1.json`, `tests/test_operator_ablations.py` | PASS |
| 10-condition primary grid and 14-condition crosswalk | `reviewer1_condition_crosswalk.md`, `reviewer1_condition_scope_manifest_v1.json` | PASS |
| Row/group AUC accounting | 25 row-level; 23 group-aware eligible; `wine-quality-red` and `kddcup99` explicit skips | PASS |
| Cache build-once fan-out | 10-consumer bounded runner audit test | PASS (bounded synthetic evidence) |
| New-process cache resume | Mid-group kill/resume audit test | PASS (bounded synthetic evidence) |
| Friend host measurement | `host_measurement.json` is from the earlier local host, not the friend PC | UNKNOWN |
| GPU fit and CPU/GPU parity | GPU routing is explicit for XGBoost/CatBoost, but no friend-PC GPU fit was run | UNKNOWN |
| 10-day throughput gate | No representative friend-PC measurement covering both policies, large tails, histories, checkpointing, and cleanup | UNKNOWN |

## Cache lifecycle evidence

The bounded runner cache key contains dataset/split/seed/fold/condition/pipeline/configuration/code/runtime identities and the training/test index hashes. `CacheManager.get_or_create_bytes` holds the per-artifact lock through the double-check and build, so concurrent consumers cannot publish duplicate builds. Payload, manifest, and ready marker are checksum-verified and atomically published. Reader leases are visible in the optional audit manifest.

Run a bounded fan-out audit with `cache_audit=True` (the focused test does this for ten models). Acceptance is:

* `build_count = 1`;
* `hit_count = 9` for ten compatible classifier consumers;
* all ten consumer task IDs have durable terminal records;
* active-reader history shows a live lease while bytes are read;
* `regeneration_count = 0` without an interruption;
* deletion is observed only after the final consumer is terminal and no lease remains.

The mid-group process-boundary test kills the runner after five completed consumers, starts a new process with the same run identity, and verifies one build, nine hits, no completed-task refit, no double-counted result, and deletion after the remaining consumers finish. A retryable scheduler task keeps the cache until its consumer is terminal; final cleanup is deferred while pending/running durable consumers exist.

## Recovery evidence

The scheduler uses SQLite WAL/FULL, immutable task specs, leases, heartbeats, atomic result envelopes, digest validation, and reconciliation. A replacement process reclaims foreign leases after reconciling any artifact already published by the old process. If scheduler publication completed before the checkpoint write, resume repairs both phase checkpoint rows from the verified scheduler artifact instead of refitting. The result ledger has a run-scoped task-key index rebuilt at startup so a crash after JSONL append does not create a second logical result row.

These controls were verified with the focused scheduler, cache, and bounded runner tests. They are recovery controls, not corrected benchmark results.

## Friend-PC measurement command

Run on the friend PC from the repository root, using the existing environment:

```bat
conda activate p12
D:\Conda\p12\python.exe provenance\measure_host.py --output provenance\friend_pc_host_manifest.json --probe-root D:\DR2\AutoFE_Submission --probe-mib 64
```

The probe records executable/prefix, package versions, CPU and logical counts, best-effort hybrid topology, active power plan, GPU/driver/CUDA visibility, free space, and bounded sequential read/write throughput. It does not create an environment. Do not replace this artifact with the earlier local `host_measurement.json`.

## Bounded scheduling and GPU commands

With the friend probe saved, run the same representative mix under controlled workers and one-thread numerical libraries:

```bat
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
D:\Conda\p12\python.exe -m provenance.profile_runner --stage-datasets sonar airlines --mix-dataset sonar --workers 1 2 3 4
D:\Conda\p12\python.exe -m provenance.profile_runner --stage-datasets sonar airlines --mix-dataset sonar --workers 1 2 3 4 --use-gpu
```

The profile labels Windows default and all-available worker measurements. P-core preference and OS-headroom modes remain pending until the probe exposes the friend PC topology and available RAM; no logical-processor count is treated as a safe worker count. GPU mode routes only XGBoost and CatBoost; LightGBM and the sklearn models remain CPU. Compare total valid tasks/hour, peak RSS, disk throughput, fit/transfer/predict time, metric parity, and error rate before selecting a launch mode.

## Preflight, launch, status, stop, resume, and final verification

```bat
conda activate p12
D:\Conda\p12\python.exe -m provenance.reviewer1_preflight
D:\Conda\p12\python.exe -m provenance.reviewer1_scale_preflight

REM bounded corrected run only; use a new ID for every code/configuration identity
D:\Conda\p12\python.exe -m src.pipeline_runner --run-id friend-bounded-001 --split-policy row_level --cache-policy bounded --cache-max-gib 8 --durable-scheduler --cache-audit --scheduler-lease-seconds 3600 --pipelines Raw Raw_CapMatched AutoFE_Baseline --models logistic_regression random_forest extra_trees linear_svm knn gaussian_nb mlp lightgbm xgboost catboost --max-datasets 1 --max-seeds 1 --max-folds 1 --max-conditions 1

D:\Conda\p12\python.exe -m src.check_progress --run-dir corrected_runs\friend-bounded-001

REM graceful stop: allow the current task to finish, then terminate the process
REM resume: rerun the exact same command and run ID after a restart

D:\Conda\p12\python.exe -m src.pipeline_runner --run-id friend-bounded-001 --split-policy row_level --cache-policy bounded --cache-max-gib 8 --durable-scheduler --cache-audit --scheduler-lease-seconds 3600 --pipelines Raw Raw_CapMatched AutoFE_Baseline --models logistic_regression random_forest extra_trees linear_svm knn gaussian_nb mlp lightgbm xgboost catboost --max-datasets 1 --max-seeds 1 --max-folds 1 --max-conditions 1

D:\Conda\p12\python.exe -m provenance.audit_dataset_schemas
D:\Conda\p12\python.exe -m provenance.audit_group_folds
D:\Conda\p12\python.exe -m pytest -q -rs tests
D:\Conda\p12\python.exe -m compileall -q src tests main.py provenance
git diff --check
```

The long command in `README.md` remains held until the friend-PC gates pass. A full launch must use the frozen manifest and both split policies as separate run identities; no row-level fallback is permitted for the two group-AUC skips.

## Remaining launch blockers

1. Obtain the friend-PC host manifest and verify the `p12` executable/package identity.
2. Measure sustained throughput on representative small, medium, and slow large datasets under both split policies with candidate-history writes, bounded cache, checkpointing, and cleanup enabled.
3. Run a forced termination/new-process resume on that host and inspect cache audit, scheduler reconciliation, and unique task coverage.
4. Confirm compressed candidate history plus results/checkpoints and the 8 GiB live-cache cap fit the actual free volume with margin.
5. Compute the measured valid tasks/hour and uncertainty range. `PASS` requires full 1,680,000-cell scope support above 7,000 valid cells/hour with margin; otherwise the verdict is `FAIL` or `UNKNOWN`.

All corrected performance, operator, condition, Jacobian, and statistical effects remain **PENDING CORRECTED RUN**.
