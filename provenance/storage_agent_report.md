# Storage and feature-cache lifecycle audit

Date: 2026-09-30  
Environment: `D:\Conda\p12`  
Scope: isolated storage module and targeted tests; no full benchmark was run.  
Milestone commit: `5826839`

## Existing cache lifecycle

The current `src/pipeline_runner.py` cache path is run-scoped under
`<run_dir>/cache/`. `_cache_paths()` uses a feature-task key and pipeline name
to produce a pickle plus JSON manifest. `_load_or_create_feature_cache()` reads
the pair after manifest verification, deletes both on a stale/malformed cache,
computes a payload, writes a process-specific temporary pickle, atomically
renames the pickle, and then writes the manifest atomically.

The audit found no ready marker, consumer lease/refcount, cache-age policy,
byte budget, stale-temp reconciliation, or high-water monitor in the current
runner path. A process can therefore leave a final pickle without its manifest
or leave temporary files after interruption; there is also no lifecycle signal
that makes deletion safe while another consumer is reading a cache. The
existing path remains unchanged by this milestone.

## Isolated implementation

`src/cache_manager.py` provides a future integration seam with these rules:

- `put_bytes()` writes payload, manifest, and a final ready marker through
  process-specific temporary files and atomic renames. Readers accept an item
  only when all three files exist, the manifest says `state=ready`, the marker
  matches the manifest checksum, and the payload size/checksum match.
- `acquire()` creates a lease file with an expiry and returns a context-managed
  consumer lease. Multiple active lease files are the consumer refcount.
  Heartbeats extend the expiry; release removes the lease. Malformed lease
  files are treated as active, so cleanup fails closed.
- A per-digest lock is shared by lease acquisition, writes, and cleanup. A
  writer refuses replacement while an active or malformed lease exists.
- `reconcile()` classifies ready, incomplete, corrupt, temporary, and leased
  state without deleting anything.
- `cleanup()` can remove stale ready artifacts, over-budget least-recently
  accessed artifacts, old incomplete/corrupt debris, old temp files, and
  expired lease files. Every artifact candidate is rechecked under its lock;
  leased or malformed-lease artifacts are skipped. `dry_run=True` reports
  candidates without mutation.
- `high_water()` records current and peak managed bytes in
  `storage.highwater.json` and reports a configured byte limit. Lease and lock
  bookkeeping are excluded from cache-byte accounting.

The manager stores bytes and leaves pickle/object serialization to its caller.
The reviewed integration in `src/pipeline_runner.py` serializes feature payloads
with pickle, wraps reads in leases, keys artifacts by the immutable feature-task
identity, and reclaims each artifact after the final compatible model consumer.
The runner records reconciliation and high-water evidence in its run manifest.

## Verification

Command:

```text
D:\Conda\p12\python.exe -m pytest -q tests/test_cache_manager.py
```

Result: **9 passed in 0.97s**.

Additional checks:

```text
D:\Conda\p12\python.exe -m compileall -q src/cache_manager.py tests/test_cache_manager.py
git diff --check
```

The tests cover atomic ready publication, unreadable partial state, multiple
consumer refcounts, deletion blocking while leased, write replacement blocking,
heartbeat/release, conservative malformed-lease handling, stale debris cleanup,
high-water/LRU cleanup, and dry-run behavior.

## Limits and follow-up

This milestone does not alter the current runner or migrate existing `.pkl`
cache files. Cleanup is an explicit operation and is not a background quota
enforcer. Expiry uses the local wall clock; malformed lease files intentionally
require manual inspection/removal. The manager protects operations that use its
lease API; direct filesystem deletion outside the manager remains outside its
control. A future integration must also decide retention by run/condition and
whether an artifact replacement is allowed after a completed task ledger row.
