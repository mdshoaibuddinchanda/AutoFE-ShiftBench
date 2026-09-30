# Four-dataset pilot recovery and runtime diagnosis

Status: both four-dataset pilot policies finished with 5,600 authoritative successes each. Corrected performance effects remain **PENDING CORRECTED RUN**. The 25-dataset benchmark was not launched.

## Scope, identity, and preservation

The pilot has four datasets (`sonar`, `heart-disease`, `haberman`, `ionosphere`), one seed (42), one fold (1), ten primary training-corruption conditions, fourteen frozen pipelines, and ten classifiers. Each split policy has 5,600 intended logical cells. The row-level and group-aware runs have separate run IDs, scheduler databases, result artifacts, manifests, and 3,600/600-second lease settings respectively. The active coordinator was PID 35152 with child fit workers 6320 and 3376. All pilot result rows inspected in the sample reported `model_backend=cpu`; `nvidia-smi` listed no pilot PID among compute apps.

The diagnosis began from repository HEAD `b4ac3828a73e08817a1c488c3037ccacfcd50d15`. The row-level manifest records code fingerprint `488a02a2f7950fce9ef70077b2f0ca8f1892914cb8ccb21756fbc386ee47773e` and configuration fingerprint `fad2f2fe0f2bf40657ed8633fe34e339c52aa4dfaff34d7dd18b1f991adc4996`; the group-aware manifest records the same source fingerprint and configuration fingerprint `882aa5c71a60ad46f4e32b258170373c8f5b87d5ff5e933194a29b4a255f053a`. The differing historical `code_commit` fields reflect the commit at each launch, while the source fingerprints match. Dataset CSV hashes are recorded in both manifests and were not changed. Logical task IDs hash dataset, seed, fold, condition, pipeline, classifier, split policy, and run ID. Each scheduler result has a unique task key and checksum-validated artifact.

| Dataset | CSV SHA-256 | Sidecar SHA-256 |
|---|---|---|
| `sonar` | `3c68a7327e052bc027b5ff59702d959439759023b21baf5fcaee12f081741baf` | `1994f3820717612ae61652a459efa014706c8cf3e1916ce4c61917667debbdb0` |
| `heart-disease` | `d0eb82e29f70cbbdd73789209674ab609151e96ce5ffd5ef0666ca0113798c03` | `ae828681fc7e0bc01b1db5d459874fbabee91cbc8e7bb48b30c6c5e69d6a3aab` |
| `haberman` | `74984cac06e6026c4d330428fef101893e2349d8d1aa15b61fcb47691db0b7af` | `188a736332cb97f17a5ef826433f485b39f8812867a0551518647893dd213fb5` |
| `ionosphere` | `9cb05353326e0803d8d8651d7731a20ae44fac9df42582a86270e8c2723b3bc6` | `85dd35641bd296672c7a9bc40eb3cf38d709b1f14ac1964086ed87cea16c641c` |

The active group scheduler and checkpoint databases were copied with SQLite's backup API at `corrected_runs/four_dataset_pilot/diagnosis_snapshot_20260930T105404Z`; the manifest and JSONL mirror were preserved in that snapshot. No `.log`, `.out`, or `.err` file existed under the pilot run directory at inspection; the scheduler event table and attempt rows provide the available execution history. At that point the group ledger had 1,722 authoritative successes, seven running leases, one registered pending task, and no scheduler/checkpoint disagreement. The pre-existing notebook execution-count edit in `notebooks/visualization.ipynb` was left untouched.

## Fifteen-minute live sample

Sampling ran every approximately 60 seconds from 10:55:29 to 11:10:34 UTC on 2026-09-30. The scheduler result count and success task count agreed at every observation. The `diagnosis_progress_sample.jsonl` artifact is retained under the ignored pilot output directory. Build/hit counts come from the manifest audit, which the guarded live launcher publishes every 50 outcomes, so they advance in steps.

The versioned read-only `audit_four_dataset_pilot.py` validates each authoritative artifact digest and task ID; compares scheduler, phase-2 checkpoint, and successful JSONL key sets; and compares matrix/prediction hashes and all classification metrics between the artifact and JSONL. The checkpoint stores completion state and task identity, not metric values. At a live 2,999-cell group checkpoint the auditor found zero task-ID or result-value disagreements and exactly 2,999 records in each of those three ledgers. The manifest still showed its prior 2,950-cell publication because the guarded launcher batches it; this is expected progress-view lag, not a recreated pending task. Historical cache/task fingerprint fields differ across the documented lease migration and are excluded from this science-value parity check. The final manifest is flushed when the run finishes.

| Minute | UTC | Success | New valid cells | Running leases | Oldest heartbeat (s) | Cache builds | Cache hits |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | 10:55:29 | 1,753 | — | 7 | 4.5 | 177 | 1,582 |
| 1 | 10:56:30 | 1,769 | 16 | 7 | 3.2 | 177 | 1,582 |
| 2 | 10:57:30 | 1,785 | 16 | 7 | 0.8 | 177 | 1,582 |
| 3 | 10:58:30 | 1,801 | 16 | 7 | 2.1 | 182 | 1,627 |
| 4 | 10:59:30 | 1,817 | 16 | 7 | 0.4 | 182 | 1,627 |
| 5 | 11:00:31 | 1,832 | 15 | 7 | 1.4 | 182 | 1,627 |
| 6 | 11:01:31 | 1,845 | 13 | 8 | 3.4 | 182 | 1,627 |
| 7 | 11:02:31 | 1,860 | 15 | 7 | 1.3 | 187 | 1,672 |
| 8 | 11:03:32 | 1,874 | 14 | 7 | 3.7 | 187 | 1,672 |
| 9 | 11:04:32 | 1,889 | 15 | 7 | 2.8 | 187 | 1,672 |
| 10 | 11:05:32 | 1,904 | 15 | 7 | 1.4 | 192 | 1,717 |
| 11 | 11:06:33 | 1,919 | 15 | 7 | 2.8 | 192 | 1,717 |
| 12 | 11:07:33 | 1,934 | 15 | 7 | 2.4 | 192 | 1,717 |
| 13 | 11:08:33 | 1,948 | 14 | 7 | 0.8 | 192 | 1,717 |
| 14 | 11:09:33 | 1,963 | 15 | 8 | 4.3 | 197 | 1,762 |
| 15 | 11:10:34 | 1,978 | 15 | 7 | 2.1 | 197 | 1,762 |

The sample completed 225 cells in 905.2 seconds, or 895 valid cells/hour. The shortest remaining 600-second lease was 595.7 seconds; no heartbeat exceeded 4.4 seconds, no lease expiry was observed, and no result or database error appeared. The coordinator used 756.8 CPU seconds during the interval, versus 59.5 and 56.2 CPU seconds for the two child fit workers. The workers spent 778.7 seconds with no fit executing, 124.3 seconds with one fit, and only 2.2 seconds with two overlapping fits. Four workers were configured; only two spawned because the coordinator supplied work slowly. Pending futures were heartbeated by the coordinator while it waited for fits, independently of fit completion; the sampled heartbeats remained fresh.

Coordinator RSS stayed between 178.5 and 186.4 MiB, child RSS between 121.0 and 168.8 MiB / 129.1 and 150.0 MiB, and system available RAM between 9.0 and 9.6 GiB. Swap use rose by under 4 MiB. The coordinator's logical reads rose by 21.25 GiB, about 96.7 MiB per valid cell, and writes by 0.21 GiB; system-wide physical disk reads/writes rose by 0.20/4.60 GiB and include other processes. The coordinator had 19 threads and each child 30–31; no BLAS/OpenMP thread-limit variables were set. GPU utilization was 3–15% and VRAM use 1.46–1.75 GiB, but those readings are not attributable to pilot fits. A 100-query read-only SQLite probe showed 0.282 ms median, 0.586 ms p95, and 7.704 ms maximum latency; no `database is locked` failure was recorded. These observations do not measure write-lock wait directly.

## Failed attempts and recovery

The row-level JSONL mirror has twelve failed-attempt rows, all for `ionosphere`, `label_noise_0.10`, and all eventually followed by a successful authoritative result for the same logical task. One is a `LeaseLost` phase-1 row after an intentional coordinator replacement; two are `RuntimeError` Windows spawn bootstrap failures whose stored message explicitly names the missing `if __name__ == '__main__'` idiom; nine are `BrokenProcessPool` consequences of that bootstrap failure. The unguarded temporary Windows launcher was corrected to use a `__main__` guard. This was an operational launcher defect, not evidence of model, GPU, dataset, or cache failure. No `CacheBusyError`, out-of-memory, GPU, serialization/pickling, or SQLite lock exception appears in these attempts. The twelve attempts held leases for a combined 154.0 task-seconds; overlapping durations must not be read as wall-clock time lost.

| Task key | Pipeline | Classifier | Mirror exception | Failed attempt | Lease-held seconds | Retry outcome |
|---|---|---|---|---:|---:|---|
| `bf4227b7ed99e2f72a0cde83955a18127913f5e469caf67a803e02b0c9318a63` | `AutoFE_LeaveOut_Add` | `knn` | `LeaseLost` | 1 | 28.0 | success on 2 |
| `d2dd711b5a32d7dccdd69db2e949c393bdd4f46c6f01a720def05ed1412387f0` | `AutoFE_LeaveOut_Add` | `extra_trees` | `RuntimeError` | 2 | 15.1 | success on 3 |
| `478de917cd1394aec9e6967250d100732f13eae440259f11e99eee0d5328798d` | `AutoFE_LeaveOut_Add` | `linear_svm` | `RuntimeError` | 2 | 20.5 | success on 3 |
| `1b1c61ba30c1ba1dddc4060a5d139085d0f2d756fa541f5372af9593e43c6e5e` | `AutoFE_LeaveOut_Subtract` | `logistic_regression` | `BrokenProcessPool` | 1 | 13.7 | success on 2 |
| `48f5a8e75686fac993a1e8585d38a445026066d0b90d37c58b2ebcdc5cd27c8b` | `AutoFE_LeaveOut_Subtract` | `random_forest` | `BrokenProcessPool` | 1 | 9.0 | success on 2 |
| `e36228193d806d1b1c806e44cae7b470e22143465a6724a6885d9722d2eed2b1` | `AutoFE_LeaveOut_Subtract` | `extra_trees` | `BrokenProcessPool` | 1 | 8.6 | success on 2 |
| `600c8ca0b0dd58d0dbd4b401585ecdde9ae84f1b17dbf894181199ba5343418a` | `AutoFE_LeaveOut_Subtract` | `linear_svm` | `BrokenProcessPool` | 1 | 11.9 | success on 2 |
| `8a1683f20bef8b6467b62397323eedbf83da8cce89ebf29da9b7fdfde71e9707` | `AutoFE_LeaveOut_Subtract` | `knn` | `BrokenProcessPool` | 1 | 11.6 | success on 2 |
| `19811d92bb8a03f7ae4322bc67c722ba18919e47c6686fd1f3680468cbc30a65` | `AutoFE_LeaveOut_Subtract` | `gaussian_nb` | `BrokenProcessPool` | 1 | 9.2 | success on 2 |
| `42bbbee5597903cdfa3808b43323258ce4d3406b61fb221d22711c0753182c1f` | `AutoFE_LeaveOut_Subtract` | `mlp` | `BrokenProcessPool` | 1 | 9.5 | success on 2 |
| `1a7e34df7cf808d401905843b2d3a80f1178a478daba48d1ea60bcaea2a17ca5` | `AutoFE_LeaveOut_Subtract` | `lightgbm` | `BrokenProcessPool` | 1 | 8.5 | success on 2 |
| `5a05eedcd30679629e9a5401416023eecbc603f7dbac873f3c1dad133fcaddba` | `AutoFE_LeaveOut_Subtract` | `xgboost` | `BrokenProcessPool` | 1 | 8.5 | success on 2 |

The row scheduler also has 16 `WorkerReplaced` attempts (3,457.7 overlapping task-lease seconds) from intentional stops/restarts, including the first attempt of three keys above. The group scheduler has seven `WorkerReplaced` attempts (391.6 overlapping task-lease seconds) at the planned 905-cell lease migration, all `sonar` / `missing_values_0.20` / `AutoFE_Isolate_Add` or `AutoFE_Isolate_Subtract`; all seven later succeeded. Neither scheduler recorded `LeaseExpired`. The earlier superseded row-level attempt with 597 failed rows remains quarantined; its 5,422 successful cells were reconciled by exact task, matrix, prediction, and metric identity as documented in `pilot_checkpoint_recovery.md` and `pilot_checkpoint_recovery_merge.json`.

The exact replacement-attempt identities are below. Every listed task subsequently has one authoritative success; the durations overlap and are lease-held task-seconds, not additive elapsed time.

| Policy | Task key | Dataset | Condition | Pipeline | Classifier | Attempt | Seconds |
|---|---|---|---|---|---|---:|---:|
| `row_level` | `355aa27d5550d52d69c6edd020f5819c5881b1d0631b0213a3f758e5825af07d` | `sonar` | `label_noise_0.20` | `Raw` | `xgboost` | 1 | 379.6 |
| `row_level` | `92fbd3c2a7b8b8d2a96a7324f6bdab59ded4ae171d9be38e424485a97a995314` | `sonar` | `label_noise_0.20` | `Raw` | `catboost` | 1 | 377.4 |
| `row_level` | `e7449625afe761d7e3bc4d3fc2a11b44dcb3ae110d1ba6724846503c6f488e06` | `sonar` | `label_noise_0.20` | `Raw_CapMatched` | `logistic_regression` | 1 | 375.6 |
| `row_level` | `91501443265d11b0e3d23339205cea00cbc9da92a7b1be8deea5752108163c39` | `sonar` | `label_noise_0.20` | `Raw_CapMatched` | `random_forest` | 1 | 373.3 |
| `row_level` | `7ce6a62bbeba3341219403de6e43c79fb531cbab5b9dfe1d602a0ced11ad8cb0` | `sonar` | `label_noise_0.20` | `Raw_CapMatched` | `extra_trees` | 1 | 371.2 |
| `row_level` | `2a29f1c6fcac94b60e8ef24b20437fdc15babb81ed87f12911572d338262c36a` | `sonar` | `label_noise_0.20` | `Raw_CapMatched` | `linear_svm` | 1 | 369.1 |
| `row_level` | `9ef5aae7b472f01f876bcd151ebf3f78e9ac39820619f7bf56117d09793e96ac` | `sonar` | `label_noise_0.20` | `Raw_CapMatched` | `knn` | 1 | 367.2 |
| `row_level` | `424f61c6ad51fa2b3c8486e4c82bdff3038f7f13f496eaf4b5fb8c397dd97a36` | `sonar` | `label_noise_0.20` | `Raw_CapMatched` | `gaussian_nb` | 1 | 365.2 |
| `row_level` | `d2dd711b5a32d7dccdd69db2e949c393bdd4f46c6f01a720def05ed1412387f0` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `extra_trees` | 1 | 35.9 |
| `row_level` | `478de917cd1394aec9e6967250d100732f13eae440259f11e99eee0d5328798d` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `linear_svm` | 1 | 26.8 |
| `row_level` | `bf4227b7ed99e2f72a0cde83955a18127913f5e469caf67a803e02b0c9318a63` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `knn` | 1 | 28.0 |
| `row_level` | `3d52ed42207038924000efc9edd61ba288faa34c66921b6dafa7a61170edd047` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `gaussian_nb` | 1 | 40.0 |
| `row_level` | `6702ddadf2b8908012c259b68d9fd77c16e62f4bc09d1602fdd9eaf00c290abc` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `mlp` | 1 | 21.8 |
| `row_level` | `442b36763a34afc54f4d38a491ed84ad0e85ad2715e2cfae7606d1ed7a507eaa` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `lightgbm` | 1 | 18.8 |
| `row_level` | `6d204dda6c50eb09852ca08a74ffd19876ea7d201c372c7f3b3bf6f8e26c7db7` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `xgboost` | 1 | 160.8 |
| `row_level` | `3284e927a23adcf5950c62503098bb340a6ef80ae6c380feb80a21d820cdcc05` | `ionosphere` | `label_noise_0.10` | `AutoFE_LeaveOut_Add` | `catboost` | 1 | 147.1 |
| `group_aware` | `d1c0deb797445f0b86388bc3380d2f254f0a08de875dd75be11b5ea5ec97e4c7` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Add` | `gaussian_nb` | 1 | 61.1 |
| `group_aware` | `9f775a2b9582d752bfac35c9302e728aec889d86577f1dcaf2e77d5434a9c06e` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Add` | `mlp` | 1 | 59.4 |
| `group_aware` | `c4914a92b109f6e15ba298280b4181c06e42a0f3dc8e46a920aae8fae4a25bfc` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Add` | `lightgbm` | 1 | 57.6 |
| `group_aware` | `8ffc0c19962a147cd31e2c33843251f2bc7f883dbaf21ebdaa1709b2724e7200` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Add` | `xgboost` | 1 | 56.1 |
| `group_aware` | `8ba5d688415c57b48f95e2781a1c97ad8c3fa03272bdd343313ce9cdd72762f9` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Add` | `catboost` | 1 | 54.6 |
| `group_aware` | `dca88bc0dd9f231bef25fdc981385cc47c95e4af1262b7d38fc1311f9d90bdce` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Subtract` | `logistic_regression` | 1 | 52.5 |
| `group_aware` | `037ece5fefb57f918ced97ae82724c5c2df9c43d75cd3c61a016287973275586` | `sonar` | `missing_values_0.20` | `AutoFE_Isolate_Subtract` | `random_forest` | 1 | 50.3 |

The authoritative scheduler stores one unique result per task key and commits its result row and success transition in one SQLite transaction after atomic artifact publication. JSONL is a retry-history mirror. At inspection, every completed scheduler task had one result artifact and a phase-2 success checkpoint; there were no duplicate successful attempts or duplicate successful mirror keys. The 5,422 compatible old row-level successes and the first 905 group-aware successes remained committed through recovery. An in-flight fit can repeat after interruption; no previously committed fit was found to have repeated. There is no evidence that a successful task was recreated as pending.

## Stage and bottleneck measurements

Direct bounded profiling on the same four CSVs, seed 42, fold 1, clean condition, and both split policies measured split construction once per dataset/seed: row-level 0.001–0.003 seconds, group-aware 0.223–0.595 seconds. The additional group split cost amortizes to below 0.5 ms per 1,400-cell dataset and does not explain the hours-long track. `Raw` preparation took 0.011–0.019 seconds. `AutoFE_Baseline` preparation took 0.053–1.549 seconds, of which feature generation took 0.041–1.539 seconds and preprocessing about 0.009–0.018 seconds. Sonar group-aware preparation across all fourteen pipelines took about 11.9 seconds for 140 eventual model consumers. Candidate-history writing was disabled in the pilot (`records_written=0`) and therefore contributed zero. These bounded profiles ran during the active pilot and are indicative stage measurements, not corrected model outcomes.

| Dataset | Policy | Pipeline | Split once (s) | Preprocessing (s) | Feature generation (s) | Preparation total (s) |
|---|---|---|---:|---:|---:|---:|
| `sonar` | `row_level` | `Raw` | 0.001 | 0.012 | 0.001 | 0.014 |
| `sonar` | `row_level` | `AutoFE_Baseline` | 0.001 | 0.009 | 1.539 | 1.549 |
| `sonar` | `group_aware` | `Raw` | 0.379 | 0.012 | 0.002 | 0.013 |
| `sonar` | `group_aware` | `AutoFE_Baseline` | 0.379 | 0.018 | 1.423 | 1.441 |
| `heart-disease` | `row_level` | `Raw` | 0.003 | 0.016 | 0.002 | 0.018 |
| `heart-disease` | `row_level` | `AutoFE_Baseline` | 0.003 | 0.014 | 0.628 | 0.642 |
| `heart-disease` | `group_aware` | `Raw` | 0.344 | 0.014 | 0.001 | 0.015 |
| `heart-disease` | `group_aware` | `AutoFE_Baseline` | 0.344 | 0.013 | 0.768 | 0.781 |
| `haberman` | `row_level` | `Raw` | 0.002 | 0.016 | 0.002 | 0.019 |
| `haberman` | `row_level` | `AutoFE_Baseline` | 0.002 | 0.015 | 0.057 | 0.072 |
| `haberman` | `group_aware` | `Raw` | 0.223 | 0.009 | 0.002 | 0.012 |
| `haberman` | `group_aware` | `AutoFE_Baseline` | 0.223 | 0.011 | 0.041 | 0.053 |
| `ionosphere` | `row_level` | `Raw` | 0.002 | 0.009 | 0.002 | 0.011 |
| `ionosphere` | `row_level` | `AutoFE_Baseline` | 0.002 | 0.011 | 0.991 | 1.002 |
| `ionosphere` | `group_aware` | `Raw` | 0.595 | 0.011 | 0.002 | 0.013 |
| `ionosphere` | `group_aware` | `AutoFE_Baseline` | 0.595 | 0.012 | 1.055 | 1.066 |

For the first 1,847 group successes, classifier fit times summed to 732.2 seconds and full worker execution to 802.8 seconds; CatBoost accounted for 584.8 fit seconds, median 3.309 seconds per cell. On matched `sonar` cells, group-aware CatBoost median fit time was 3.277 seconds versus 3.449 row-level; on the first 49 matched `heart-disease` CatBoost cells, 4.698 versus 4.091 seconds. Those differences depend on fold composition and are not a performance-effect analysis. They show that group-aware model fits alone do not explain the observed coordinator-bound throughput.

A separate ten-classifier `sonar` / group-aware / clean / `AutoFE_Baseline` profile timed the two calls to `compute_classification_metrics` (test and train) separately from fitting and inference. These are representative one-cell stage times under concurrent pilot load; the final result ledger will supply full per-model summaries.

| Classifier | Fit (s) | Inference (s) | Evaluation (s) | Worker total (s) |
|---|---:|---:|---:|---:|
| `logistic_regression` | 0.245 | 0.000 | 0.023 | 0.270 |
| `random_forest` | 0.221 | 0.010 | 0.022 | 0.265 |
| `extra_trees` | 0.126 | 0.011 | 0.021 | 0.171 |
| `linear_svm` | 0.209 | 0.003 | 0.016 | 0.234 |
| `knn` | 0.000 | 0.111 | 0.017 | 0.151 |
| `gaussian_nb` | 0.001 | 0.000 | 0.019 | 0.021 |
| `mlp` | 0.021 | 0.000 | 0.033 | 0.057 |
| `lightgbm` | 0.052 | 0.002 | 0.022 | 0.246 |
| `xgboost` | 0.183 | 0.002 | 0.025 | 0.246 |
| `catboost` | 3.634 | 0.004 | 0.023 | 3.758 |

An isolated result publication with a representative 4,746-byte payload took 0.00075 seconds for the JSONL mirror, 0.00723 seconds for atomic scheduler publication, and 0.00617 seconds for the phase-2 checkpoint. A 2,450-outcome manifest write took 0.47 seconds, amortized across approximately 50 live outcomes by the guarded launcher. A dry-run cache cleanup of 253 copied artifacts took 5.88 seconds; the active process defers that scan until consumers are terminal. These are isolated operation times, because the live run did not record commit and cleanup spans individually.

| Rank | Bottleneck or stage | Measured evidence | Interpretation |
|---:|---|---|---|
| 1 | Global scheduler reconciliation on each claim | Every claim scans all published result files and opens a SQLite connection for each. A read-only scan of 2,324 immutable artifacts took 0.58–0.77 s; an isolated same-volume copied-ledger claim took 16.65 s with the full scan versus 0.0075 s with task-local recovery. | Confirmed growing bookkeeping cost; the copied-ledger timing includes cold-file effects and is not a live cells/hour estimate. |
| 2 | Whole-cache scan for one reader audit | `manager.reconcile()` took 0.288 s median at 221 artifacts; a key-specific count took 0.000129 s median. The live coordinator read 21.25 GiB in 15 minutes. | Confirmed unnecessary repeated hashing and metadata reads; sampled per-call saving does not equal full-run speedup. |
| 3 | Serial coordinator dispatch and publication | Only 126.5 of 905.2 sampled seconds had any fit running; two-fit overlap was 2.2 s. Coordinator used 756.8 CPU seconds. | Integrated four-worker setting is limited by coordinator preparation/recovery/bookkeeping. |
| 4 | Feature generation and classifier fits | Sonar fourteen-pipeline clean preparation about 11.9 s; group CatBoost median 3.309 s among first 184 successes. | Real scientific work, but less than the observed wall time at this small-dataset mix. |
| 5 | Cache cleanup and manifest publication | Isolated dry-run cleanup across 253 copied artifacts took 5.88 s. A 2,450-outcome manifest write took 0.47 s; live launcher writes approximately every 50 outcomes. | Cleanup is deferred until terminal consumers; neither explains each live cell. |

Successes 1–100 were published over 40.6 seconds, while successes 2,601–2,700 took 508.6 seconds. These windows contain different datasets/pipelines and so are not a paired speed comparison, but the sustained slowdown as both scheduler results and cache artifacts accumulated matches the two growing scans above. The 15-minute sample was already down to 895 cells/hour. A paired same-cell run after completion is needed to quantify how much of that trend the staged changes remove.

The independent minute ledger monitor continued after that sample. Its later windows show a steady decline without a new failed attempt or stale heartbeat. The windows include changes in task mix, so they support the observed bottleneck trend but do not by themselves measure either proposed optimization's effect.

| Group successes at window bounds | UTC window | New valid cells | Minutes | Valid cells/hour |
|---|---|---:|---:|---:|
| 3,006 to 3,500 | 12:33–13:22 | 494 | 49.0 | 605 |
| 3,500 to 4,000 | 13:22–14:19 | 500 | 57.0 | 526 |
| 4,000 to 4,504 | 14:19–15:23 | 504 | 64.0 | 472 |
| 4,504 to 5,005 | 15:23–16:35 | 501 | 72.0 | 417 |
| 5,005 to 5,254 | 16:35–17:14 | 249 | 39.0 | 383 |

The pilot's feature-cache audit showed representative completed ten-model groups with exactly one build and nine hits. At an intermediate snapshot there were 266 logical feature groups and 357 cache-audit identities. Ninety-one groups had both pre- and post-lease-migration cache fingerprints; 90 of those still had only one build across both identities. One interrupted `sonar` / `missing_values_0.20` / `AutoFE_Isolate_Add` group (`feature_task_key=be290924804ba48ab62d`) had two builds: the old cache served seven attempted consumers (one build, six hits) and the new fingerprint served the remaining work (one build, four hits). Its ten model tasks still each have one authoritative success. This was one unnecessary feature rebuild under the conservative cache fingerprint change, not a repeated committed model fit. Direct reconstruction of the same feature task took 0.205–0.533 seconds in three repetitions, measuring the scale of that extra preprocessing rather than the exact historical duration. The row-level audit covers the target continuation only; its 5,422 imported successes are proven by scheduler/result identity rather than retrospective cache hit counts. Cache deletion was zero during the group sample because the bounded runner retained artifacts while futures could still be in flight; final cleanup must be checked after completion. The sample's cache occupied roughly 34–39 MiB. Result commit was not instrumented in the active process; `worker_finished_unix` to scheduler publication includes FIFO queue waiting, so it is an upper bound rather than a commit duration. Per-task stage attribution for the active process is limited by that instrumentation gap.

## Staged repairs and verification

The completed pilot used the frozen source. After it exited, the first 560-cell paired baseline exposed one terminal `PermissionError` on a Windows atomic cache-manifest replacement (`haberman` / `Raw_CapMatched` / `linear_svm`, task key `470e4ca1e5f923bd7030f1173d3134d4c7b18f3bfc4ada1214b0a26006167ffe`, attempt 1, 0.561 lease seconds). The profile ended with 559 successes and one failed cell and is preserved as failure evidence, not used as a paired timing baseline. Cache-manager atomic replacements now retry a transient destination lock up to eight times while preserving the temporary-file, checksum, and ready-marker boundaries; exhaustion raises and cleans the temp file. Focused injected-error tests cover both paths. This correctness repair was separately committed on the main revision branch as `90a8eeab773716095bb3dd3839ec7d30e2231d90`, with 86 passing `p12` tests. A fresh complete baseline from that commit is required for timing.

In a detached worktree, the scheduler claim path was narrowed to reconciliation of the requested task; a full orphan/debris scan remains at startup and for unqualified claims. A one-run OS file lock fences a live coordinator before any scheduler reconciliation, preventing a second process from converting its leases to `WorkerReplaced`; the lock is released by the OS on process exit. Cache reader audit now counts only leases for the artifact being read; the old `active_readers` value was a global cache count, so the audit-field meaning changes to the relevant feature key. Dataset, split, condition, pipeline, classifier, feature cap, metrics, and AUC rules are unchanged.

Focused tests show that task-local reconciliation adopts a result published immediately before a crash, skips unrelated completed artifacts, and preserves one attempt/result; a second process cannot reclaim a live coordinator's lease. A new-process crash/restart inside a group-aware ten-model cache group preserves five committed consumers, reuses the cache for the rest, produces one build and nine hits, and ends with ten unique authoritative results. The read-only ledger auditor and paired-output comparator also have focused tests. The staged full `p12` suite passed **94 tests**; compileall and diff checks will be rerun after integration. A four-mode paired representative-cell benchmark (post-retry baseline, each scan removal alone, both together) is required before retaining the performance changes and will be recorded below.

The read-only accounting command is `D:\Conda\p12\python.exe provenance\audit_four_dataset_pilot.py D:\DR2\AutoFE_Submission\corrected_runs\four_dataset_pilot --output D:\DR2\AutoFE_Submission\corrected_runs\four_dataset_pilot\diagnosis_ledger_audit.json`. It writes only the requested diagnostic JSON and reads the original SQLite files in `mode=ro`; it does not reconcile or reclaim live leases.

## Final accounting and verdict

The pilot coordinator exited normally at 18:16 UTC on 2026-09-30. A final read-only audit validated all 11,200 success artifacts and found zero discrepancies among the scheduler, phase-2 checkpoint, JSONL success mirror, and final manifest. Each policy has 5,600 registered tasks, 5,600 authoritative results, zero pending/running/skipped/terminally failed tasks, and no duplicate successful attempt or mirror key. The row mirror retains twelve historical failed-attempt rows; its scheduler retains eleven failed attempts (nine `BrokenProcessPool`, two spawn `RuntimeError`) and sixteen intentional `WorkerReplaced` attempts. The one `LeaseLost` mirror failure corresponds to an intentional replacement rather than a separate scheduler failure status. The group scheduler retains seven intentional `WorkerReplaced` attempts. Both have zero `LeaseExpired` attempts. All such tasks ultimately succeeded. No previously committed cell was refit.

| Policy | Intended | Final scheduler successes | Phase-2 successes | JSONL unique successes | Manifest successes | Terminal failures | Remaining |
|---|---:|---:|---:|---:|---:|---:|---:|
| `row_level` | 5,600 | 5,600 | 5,600 | 5,600 | 5,600 | 0 | 0 |
| `group_aware` | 5,600 | 5,600 | 5,600 | 5,600 | 5,600 | 0 | 0 |

The row-level superseded attempt started at 23:54 UTC on September 29 and its last attempt ended at 06:40 UTC on September 30. Its compatible 5,422 successful cells were reconciled into the target ledger; the target run spans 06:45–09:59 UTC (3.228 hours), including imported cells and interruption time. After the 09:10:20 UTC recovery merge, the remaining 178 successful cells took 49.0 wall minutes, including the temporary unguarded-launcher failure and retry. Thus 5,600 divided by the target run span is **not** a valid fresh-fit throughput. The group-aware target run spans 09:59–18:16 UTC (8.277 hours, 677 valid cells/hour overall). Its first 905 cells preceded the lease migration; the 4,695 later cells span 10:19–18:16 UTC (7.939 hours, 591 valid cells/hour). Across the old row attempt and both target runs, first attempt to final group result was 18.357 wall hours. These spans include recovery and coordinator overhead; sums of worker fit times below are concurrent CPU-work totals and should not be mistaken for wall time.

| Dataset | Row scheduler span (min) | Group scheduler span (min) | Row fit / worker time (min) | Group fit / worker time (min) |
|---|---:|---:|---:|---:|
| `sonar` | 40.5 | 38.2 | 9.5 / 10.4 | 9.0 / 9.9 |
| `heart-disease` | 31.4 | 97.9 | 6.4 / 7.0 | 7.9 / 8.7 |
| `haberman` | 51.2 | 149.2 | 1.5 / 2.0 | 1.9 / 2.6 |
| `ionosphere` | 70.7 | 213.1 | 14.2 / 15.1 | 12.5 / 13.4 |

Each dataset has 1,400 final successes per policy. The dataset scheduler span runs from its first claim to its last authoritative publication, including intervening gaps and retries. The row spans are affected by imported historical successes; comparisons of those spans with group times are descriptive execution accounting, not controlled policy effects. CatBoost was the slowest fit family: 1,498.9 row-level and 1,450.8 group-aware fit seconds over 560 cells per policy; median fits were 2.311 and 2.637 seconds. Random Forest was next at 106.6 and 121.0 total fit seconds. CatBoost's cost is real, but it does not account for the multi-hour group coordinator spans on these small datasets.

The final group cache audit has 560 logical feature groups, 561 builds, 5,041 hits, and one extra build across the lease-identity migration documented above. It marked 561 built identities deleted. Final cache cleanup removed 562 physical artifact digests and reclaimed 101,056,731 bytes (96.4 MiB), leaving zero payload files and no busy or leased skips. The extra physical digest lacks a corresponding completed build entry in the carried-forward audit history; physical deletion and zero remaining payloads are the final storage authority. The target row cache audit covers only its continuation: 144 recorded builds, 1,309 hits, 146 deletion flags, and zero final payload files. Those partial audit counts cannot be interpreted as the full 5,600-cell row fan-out. A late coordinator scan caused a transient roughly 63-second maximum heartbeat age near 18:10 UTC; publication resumed, all leases retained over 530 seconds at the pause, and no task was reclaimed or retried.

## Paired performance verification

After the cache replacement retry was committed, seven sequential complete profiles used identical four CSV hashes, the `group_aware` policy, seed 42, fold 1, clean condition, all fourteen frozen pipelines, all ten classifiers, four CPU workers, an 8 GiB bounded feature-cache cap, durable scheduling, and cache audit. Each profile comprised 560 unique cells. The first pre-retry profile with one failed cell is excluded. The repaired baseline (`paired-retry-baseline`) retained both original global scans; `paired-local-claim` used only task-local claim reconciliation; `paired-local-cache` used only artifact-local cache reader accounting; `paired-both` used both. Two rows repeat the incremental cache comparison, and the final row reruns both changes from the committed main branch at `21311bcacf55f8f872e4e23746982cd5aba856ea`. Source code fingerprints were `57cbf2b9f0fa9424f68c2a813a8383fc19dc466907e0f4e93a89e0a4063ab56d` for the repaired baseline, `c4155990528ba2f59291773d07eff04e255109b42a9742266a7aa302b28b092b` for the detached staging worktree, and `6afa679eba50c26fac66f3d2f5b9c34691ca14c3a4329516998f8b476e5996f3` for the final main worktree. The staging and main worktrees have different line endings in untouched source files, which changes the byte-level fingerprint. The optimized logic was committed as `4686e329c0c9997b19fffee2dc94cc4e5c65daf3` and validated again from main.

| Mode | Valid / intended | Wall (s) | Valid cells/hour | Peak process-tree RAM (MiB) | Peak system VRAM (MiB) | Cache builds / hits / deletions |
|---|---:|---:|---:|---:|---:|---:|
| Post-retry baseline | 560 / 560 | 415.5 | 4,852 | 1,058.7 | 1,404 | 56 / 504 / 56 |
| Task-local claim only | 560 / 560 | 193.3 | 10,428 | 1,057.0 | 1,391 | 56 / 504 / 56 |
| Artifact-local cache audit only | 560 / 560 | 397.5 | 5,071 | 1,057.1 | 1,392 | 56 / 504 / 56 |
| Both changes | 560 / 560 | 186.0 | 10,837 | 1,057.2 | 1,391 | 56 / 504 / 56 |
| Task-local claim repeat | 560 / 560 | 195.0 | 10,341 | 1,056.5 | 1,402 | 56 / 504 / 56 |
| Both changes repeat | 560 / 560 | 186.7 | 10,799 | 1,053.9 | 1,397 | 56 / 504 / 56 |
| Both changes, committed main branch | 560 / 560 | 183.6 | 10,982 | 1,056.3 | 1,394 | 56 / 504 / 56 |

The task-local claim change saved 222.2 seconds (2.15×) against the same-cell baseline. The artifact-local cache audit alone saved 18.0 seconds (4.5%); when added to task-local claims it saved 7.3 seconds (3.9%), and the repeat saved 8.3 seconds (4.4%). The staged combined change saved 229.5 seconds (2.23×) against baseline. The committed main branch saved 231.9 seconds (2.26×) against baseline. Every paired comparison, including staging to final main, matched all 560 logical cells and reported zero train/test matrix, prediction, or metric mismatches; the comparator rejected configuration or dataset-hash differences and verified the same `p12` executable. The cache fan-out counts were identical in every complete run. Peak process-tree RAM did not materially increase. The VRAM column is a system reading that includes other processes; all pilot model results used CPU and the VRAM values are not attributable to AutoFE. These are bounded clean-condition measurements on four small datasets, not an estimate for the full campaign.

The integrated revision branch passed 36 focused recovery/cache/runner/comparator tests and the full **94-test** `p12` suite after the optimization copy. `compileall` and `git diff --check` passed. The separate cache retry commit passed an 86-test full suite before this integration. The unrelated `notebooks/visualization.ipynb` execution-count edit remains uncommitted and untouched. Raw CSVs, cache files, scheduler databases, and generated profile results remain outside the commits.

| Finding | Verdict | Evidence or remaining limit |
|---|---|---|
| Twelve historical failed mirror attempts | **fixed** | All twelve task keys have one authoritative success; the Windows temporary launcher now has a main guard. |
| Transient Windows cache-manifest replacement error | **fixed** | A separate bounded profile exposed one terminal error; the atomic retry and exhaustion tests pass, and the repaired 560-cell baseline completed. |
| Second-coordinator live lease reclamation risk | **fixed** | OS-backed run fence and new-process rejection test; intentional historical replacement attempts remain audit evidence. |
| Growing global scheduler and cache scans | **fixed** | Same-cell profiles and repeated incremental cache pair show faster completion with exact output and cache-count parity. |
| Group split construction and CatBoost fit time | **measured but expected** | Group split is 0.223–0.595 s once per dataset; CatBoost is the slowest fit family but far smaller than the accumulated coordinator span. |
| Full 25-dataset ten-day and host/storage feasibility | **unresolved** | Requires representative large-dataset sustained throughput of at least 7,000 valid cells/hour with host and disk margin, a refreshed code/manifest freeze, and the remaining fault-matrix gate. |

The full 25-dataset run remains **DO NOT LAUNCH**. The versioned scientific task accounting remains 1,750,000 intended and 1,680,000 AUC-eligible cells, but the new code commit requires a refreshed frozen implementation identity before any long-run launch. Four small datasets cannot establish the ten-day rate or storage margin. Corrected ROC-AUC, operator, Jacobian, and statistical effects remain **PENDING CORRECTED RUN**.
