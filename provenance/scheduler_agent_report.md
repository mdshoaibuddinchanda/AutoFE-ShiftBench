# Scheduler and crash-recovery audit

Date: 2026-09-30
Scope: bounded scheduler assignment; no full benchmark run and no edits to
`src/pipeline_runner.py` or `src/checkpoint.py`.

## Implementation

`src/task_scheduler.py` adds a standalone durable lifecycle around a task
specification. It uses one SQLite database with `journal_mode=WAL`,
`synchronous=FULL`, foreign keys, a busy timeout, and `BEGIN IMMEDIATE` for
state-changing operations. The schema contains:

* `scheduler_tasks`: immutable task identity/specification hash, retry budget,
  current status, availability time, and final artifact identity;
* `scheduler_attempts`: worker ID, unique lease token, heartbeat and expiry,
  attempt status, error classification, and finish time;
* `scheduler_results`: one committed result per task; and
* `scheduler_events`: append-only lifecycle evidence for registration, claim,
  heartbeat, expiry, failure, publication, and recovery.

`TaskSpec` registration is idempotent for the same run/task payload and rejects
changed payloads or retry budgets with `TaskIdentityMismatch`. A task is claimed
by one worker inside an immediate transaction. The lease token and worker ID
are checked for every heartbeat, failure, and publication. A heartbeat extends
the lease only while it is still live; a stale worker receives `LeaseLost`.

`classify_failure` uses an explicit retry allowlist. Timeouts, transient I/O,
SQLite lock/busy errors, and worker-loss errors are retryable. Common
deterministic input/programming errors and unknown errors are non-retryable by
default. `record_failure` records the original error and returns the task to
`pending` only when the classification is retryable and the attempt budget
remains. The optional retry backoff is deterministic and exponential by attempt
number.

## Atomic result publication

`publish_result` serializes the result with canonical JSON, computes its SHA-256
digest, writes a self-describing envelope to a same-directory temporary file,
flushes and `fsync`s the file, and atomically renames it to the deterministic
task artifact path. It then commits the result row, successful attempt, and
successful task in one SQLite transaction. The envelope includes schema
version, run ID, task key, attempt number, lease token, payload digest, and
payload. Repeated publication of the identical result is idempotent; a
different payload raises `ResultConflict`.

The `fault_stage` argument is test-only. A crash after the temporary file but
before rename leaves no final result and can be retried. A crash after rename
but before the database transaction leaves a verifiable final artifact.
`reconcile` validates the envelope identity and digest, adopts that artifact,
marks the attempt and task successful, and records `result_recovered=1`.
Expired leases are marked `timed_out` and are returned to `pending` until the
attempt budget is exhausted. Invalid or orphan artifacts are reported rather
than silently adopted. Stale temporary files are removed only after the
configured age threshold.

## Process-boundary evidence

The focused tests in `tests/test_task_scheduler.py` execute child Python
processes with the existing `p12` interpreter:

1. The child claims a task and faults after artifact rename. The parent
   reconciliation at a later timestamp records one expired attempt and adopts
   one result; the task is `success` and the result row is marked recovered.
2. The child faults after temporary-file `fsync` and before rename. The parent
   reconciliation records one expiry, adopts zero results, removes the stale
   temporary file, and the replacement worker claims attempt 2 and publishes a
   successful result.

The same suite covers WAL/full-synchronous settings, immutable registration,
exclusive claiming, heartbeat extension, lease expiry, retry classification,
attempt limits, normal atomic publication, idempotence, conflicting results,
and invalid artifact reporting.

## Exact checks

Command:

`D:\Conda\p12\python.exe -m pytest -q tests/test_task_scheduler.py`

Result: **7 passed in 1.46s**.

Command:

`D:\Conda\p12\python.exe -m compileall -q src\task_scheduler.py tests\test_task_scheduler.py`

Result: passed with no output. `git diff --check` passed for both new source
and test files; Git emitted only its normal LF-to-CRLF working-copy notices.

## Integration boundary and limitations

The module is `implemented-not-integrated`. Existing runner/checkpoint code was
left untouched so the scheduler can be reviewed independently. Before a long
campaign, the lead should map the existing task manifest fields to `TaskSpec`,
call `reconcile` before claiming work, publish result artifacts through the
lease, and include scheduler DB/artifact checksums in the run manifest.

The scheduler cannot infer whether a worker process was killed after an
external side effect. Result payloads must therefore be written only after the
task's side effects are complete, and task code should be idempotent or use its
own transactional resource boundaries. WAL protects SQLite state transitions;
it does not make arbitrary external filesystems transactional. The durable
artifact plus reconciliation protocol covers the result publication boundary
implemented here.
