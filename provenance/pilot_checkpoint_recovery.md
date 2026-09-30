# Pilot checkpoint recovery

Status: four-dataset pilot recovery and both policy runs complete; corrected performance effects remain **PENDING CORRECTED RUN**.

## Scope and identity

The bounded pilot is the prespecified four-dataset scope (`sonar`, `heart-disease`, `haberman`, `ionosphere`), one seed (42), one fold (1), ten primary conditions, fourteen frozen pipelines, ten classifiers, and separate `row_level` and `group_aware` split-policy runs. The intended denominator is 5,600 cells per policy and 11,200 across both policies. No 25-dataset benchmark was launched.

The interrupted row-level attempt is preserved at `corrected_runs/four_dataset_pilot/pilot-row_level-001-preheartbeat-aborted`. The repaired continuation reuses run identity `pilot-row_level-001`; its implementation metadata is recorded in the manifest and recovery JSON files.

## Original failure and preservation

Before repair, the old coordinator and workers were checked and stopped. The first attempt had 4,826 manifest successes and 597 failures (mostly `LeaseLost`, caused by a transient Windows `PermissionError` while replacing `manifest.json`; the resulting failure handling cascaded). It had 5,422 valid successful result rows and no evidence that feature-cache files represented completed model fits. The scheduler and checkpoint databases were backed up with SQLite's `.backup()` API before reconciliation:

- `scheduler.backup.sqlite`
- `checkpoints.backup.sqlite`

The old result, manifest, cache, and logs were retained. Failed rows and cache-only artifacts were not imported as completed cells.

## Compatibility proof and reconciliation

Old implementation commit: `82efe0049be8beafca7f1a9d627cb9234d9e40e7`.
Repaired implementation commit at reconciliation: `530a2efbd009b7a4bff201bfaf4eb66959dda1a2` (current provenance commit is recorded in the target manifest).

The migration `provenance/reconcile_pilot_progress.py` compared every overlapping successful task. The overlap count and matching count were both 5,422; prediction, train/test matrix hashes, and recorded metrics matched. The source and target configuration fingerprints and dataset identities matched, and the source change was limited to runner/coordinator orchestration. No failed task or cache artifact was treated as a completed model evaluation. Scheduler task specifications and task fingerprints were normalized after recovery without changing logical task keys.

## Durable boundaries

- SQLite checkpoint and scheduler databases use WAL and `synchronous=FULL` with bounded busy timeouts.
- The scheduler keeps immutable task keys and attempt history, with atomic result artifacts and unique `scheduler_results.task_key` rows.
- Pending futures are heartbeated while the coordinator drains results; stale foreign leases are reclaimed only after restart/reconciliation.
- JSONL is a reporting mirror; scheduler/checkpoint state and checksummed result artifacts are the recovery authority.
- Feature artifacts are checksum and identity validated, leased across compatible model consumers, and never used as proof that a classifier finished.

The recovered row-level run retained its pre-existing 3,600-second lease identity. The group-aware continuation was explicitly migrated to a 600-second lease without changing logical task keys; this operational migration is recorded below.

## Fault and restart evidence

Existing focused tests cover cache interruption/reuse, scheduler publication crashes before and after artifact rename, durable publication, stale lease handling, and integrated bounded worker overlap. The pilot recovery itself exercised process termination, SQLite backup, compatibility checks, scheduler reconstruction, and a new-process resume. The first throttled launcher attempt also exposed and corrected a Windows spawn main-guard issue without accepting any of its failed cells as success. A complete new-process fault matrix (feature creation, post-publication, fit, pre/post completion transaction, and pre-cache deletion) remains to be added and run before long-run authorization.

## Counts and timing

The target ledger had 5,422 valid cells before repair. The old attempt's 597 failed rows remain quarantined. The row continuation completed the remaining 178 cells. The group-aware run retained 905 committed cells at its lease migration and completed the remaining 4,695. Both final ledgers have 5,600 authoritative successes and zero terminal failures; detailed elapsed, cache, and per-dataset accounting is in [`four_dataset_pilot_diagnosis.md`](four_dataset_pilot_diagnosis.md).

## Resume command

Use existing Conda environment `p12`. The guarded coordinator command used for the continuation was an in-memory throttled-manifest wrapper around `provenance.four_dataset_performance_pilot.main()`; it did not change source identity or scientific configuration. A normal rerun must use a `__main__` guard on Windows spawn and the same run IDs.

## Remaining limitations

This artifact is execution and recovery evidence, not a corrected performance result. ROC-AUC effects, operator effects, Jacobian associations, and statistical conclusions remain **PENDING CORRECTED RUN**. The 25-dataset benchmark remains blocked by the existing storage/host gates and was not launched.

## Historical checkpoint snapshot during execution (2026-09-30)

| Split policy | Intended | Valid recovered before continuation | Newly executed after recovery | Terminal successes | Retry/failure attempts retained | Remaining |
|---|---:|---:|---:|---:|---:|---:|
| `row_level` | 5,600 | 5,422 | 178 successful cells; 12 failed attempts were retried | 5,600 | 12 failed attempt rows, 0 terminal failures | 0 |
| `group_aware` | 5,600 | 905 retained before lease migration | in progress (scheduler snapshot: 1,302 successes) | 1,302 at snapshot | 0 terminal failures at snapshot | 4,298 at snapshot |

At this snapshot, the row-level JSONL contained 5,600 unique successful logical task keys and 12 historical failed-attempt rows; the scheduler and checkpoint ledgers contained exactly 5,600 successful task records. No previously committed model result was refit. The 12 retries corresponded to worker/coordinator failure attempts with no committed model result; they were retained as attempt evidence. Group-aware execution was still active; its first 905 cells were produced under the prior 3,600-second setting; the coordinator was then stopped, task identities were migrated without changing logical task keys, and continuation resumed with a 600-second lease. No performance effect is inferred from these execution values.

## Final reconciled outcome

The group-aware coordinator exited normally at 18:16 UTC on 2026-09-30. Final scheduler result artifacts, phase-2 checkpoint successes, unique successful JSONL rows, and manifest successes each number 5,600 under each policy, with zero disagreements and zero duplicate successful attempts. The group final cache cleanup removed 562 physical artifact digests, reclaimed 101,056,731 bytes, and left zero payload files. The row cache also has zero remaining payload files. Historical retry attempts remain visible: row-level 12 failed JSONL rows plus 16 intentional `WorkerReplaced` scheduler attempts, and group-aware seven intentional replacement attempts. Neither run had a `LeaseExpired` attempt or refit a previously committed cell. The prior snapshot table above is retained as a dated execution record, not a current count.

