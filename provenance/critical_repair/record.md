# Critical repairs and performance verification

This is an engineering record, not new manuscript evidence. The user authorizes
repairs, regression tests, bounded synthetic execution, local commits and a
non-force push. Full benchmark execution, collection download and manuscript
rewriting are excluded.

## Baseline

- Branch: `revision/leakage-seed-stability`; source: `55cf47307c86ad68e827326d862469ec89193a7d`.
- Clean worktree before work; no repository/ancestor `AGENTS.md` found.
- P12: Python 3.12.14; Windows 11; 8 logical CPUs; RTX 3050 Laptop, 4096 MiB,
  NVIDIA driver 616.64. Existing P12 and `.venv` are retained.
- Command: `D:\Conda\P12\python.exe -B -m pytest -q -p no:cacheprovider` with
  `PYTHONDONTWRITEBYTECODE=1`: 46 passed, 7 subtests, 13 Matplotlib/Pyparsing
  dependency warnings, 14.36 s. A passing baseline does not invalidate audit defects.
- Raw dataset collection and current real-run manifests/ledgers are absent.
  Historical manuscript counts concern the old 7-pipeline experiment; the current
  14-pipeline grid has 8,750 precompute units and 1,225,000 model tasks.
- Frozen scientific configuration: `baseline_contract.json`. Unknown exact
  source versions and absent bytes are explicitly unavailable, not inferred.
- The manuscript describes **class-prior relabeling**, so preserve historical
  draws/rows under an accurate condition semantics identity. No new prior-sampling
  condition is inserted into the existing grid.

## Verification policy

Each repair receives an original-failure regression and affected-suite results.
Scientific corrections receive new semantics/artifact identities; optimizations
are compared to the corrected reference, not the erroneous baseline. Root seeds
and purpose derivation remain unchanged. Reduction tolerances are declared in
the frozen contract before measurements and cannot conceal discrete decisions.

The issue register records B01–B18, C01–C19, E01–E24 and reporting/maintenance
items individually. Final implementation, tests, smoke, measured performance,
real-data evidence and full-run completion are separate statuses.

## B01 / C12: metric coordinates and explicit F1 identities

Original P12 probe with y=[0,1,2], diagonal probabilities 0.8 and off-diagonal
0.1, original string classes a/b/c: log-loss=0, class-mean Brier=0.22,
PR-AUC=undefined, ROC-AUC=1. Correct encoded coordinates give -log(0.8)
=0.2231435513142097, Brier=(0.04+0.01+0.01)/3=0.02, PR-AUC=1.

`tests/test_critical_metrics.py` originally: 9 failed, 1 passed. The repaired
evaluation validates coordinates/probabilities, uses actual estimator column
classes, and records undefined states. Binary Brier is positive-class MSE;
multiclass Brier is the mean of one-vs-rest MSEs across declared classes.
The historical binary `f1` stays binary; the new `f1_macro` is calculated
separately rather than renaming its meaning.

Metric semantics: `encoded_class_probability_metrics_v2`; incompatible outputs
use `predictor_only_geometry_v3_integrity_metrics_fsva_v2`. Seed derivation is
unchanged. Historical calibration metrics must be recomputed; historical AUC
requires class/validity revalidation rather than a blanket assurance.

Affected command: P12 pytest critical metrics, seed scheme, production resume:
17 passed, 5 subtests, 15 warnings, 11.07 s. Two additional warnings concern the
deliberate single-class fixture. The existing production worker executes the
corrected coordinate mapping; all-pipeline precompute is not evidence of
all-pipeline model fits.

API references verified against installed sklearn 1.5.2:
- https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.log_loss.html
- https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.brier_score_loss.html
- https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.roc_auc_score.html

## B02: data and cache compatibility

Original regression: 3 failed, 1 passed. Same-path changed CSVs retained task
IDs; changed matrices reused features; corrupt train matrices were accepted.
Dataset identity now hashes exact CSV bytes plus parsed schema/target/rows.
Run and task identities include it; root-seed inputs do not include it and are
unchanged. Cache dependencies include exact matrices, labels, pipeline spec,
protocol and preprocessing semantics. Publication records file hashes, frame
identities and estimator dimensions; readers reject incomplete or corrupt
artifacts. Different dependencies use separate directories, preserving older
valid caches. Split caches are hashed and atomically published under OS locks.
Full precompute/diagnostic ownership remains B16; this change alone does not
claim concurrent history publication is resolved.

Affected B02 suite: artifact integrity, task manifest, leakage boundaries,
production resume: 24 passed, 2 subtests, 13 warnings, 13.10 s.

## B03: retryable dependencies

Original first precompute failure permanently skipped its model. The new
regression demonstrated that state directly. Failure propagation now occurs
only after a terminal outcome and the store refuses propagation from pending
or running dependencies. A successful retry leaves models executable; failed
attempts remain recorded. Terminal propagation is idempotent. Affected recovery
and manifest tests: 12 passed, 13 warnings, 2.76 s. No scientific inputs change.

## B18 / C03 foundation: SQLite ownership

Original resource checks failed: a context-exited connection remained usable,
and no read-only store mode existed. Connections now close in a `finally` after
transaction exit, on both success and exception. A read-only store uses SQLite
URI `mode=ro`, verifies file presence first, and never initializes schema.
Sensitivity uses this mode; provenance integration follows in B05.
Windows tests reopen/rename/delete after closure and reject writes through a
read-only connection. Resource, manifest and sensitivity suite: 15 passed,
13 warnings, 4.15 s. This is not evidence that the historical native access
violation's cause has been established.

References: Python 3.12 sqlite3 connection-context behavior and
https://www.sqlite.org/uri.html (read-only mode).

## B04: transactional authority and idempotent export

Original durable-publication checks: 3 failed, 1 passed. An injected fsync
failure left a running task without authoritative completion. Current runs
declare `sqlite_result_outbox_v2`: result, task and attempt completion commit
in one SQLite transaction **before** JSONL export. The ledger is an export,
not a second transactional authority. Startup reconstructs missing/truncated
exports from committed results; conflicting complete rows are rejected.
Repair retains original bytes in a content-addressed recovery backup.

Historical ledger-first stores may reconcile an active owning attempt before
stale fencing. New SQLite-first stores cannot import uncommitted ledger-only
rows. Late fenced attempts remain rejected, including duplicate-shaped writes.

Affected publication/recovery/manifest/production suite: 17 passed, 13 warnings,
10.01 s. Additional transaction rollback, pre-fence legacy reconciliation and
late-takeover controls: all 7 publication tests passed, 2.86 s. These tests cover
commit/export boundaries; coordinator/writer process-failure supervision and
bounded queue integration remain B15/B17 work.

## B05 / C03 / C04 / M04 / M05: provenance authority

Original checks reproduced valid certification for a changed metric and empty
ledger when no package was supplied. Required components could disappear or an
inventory could be emptied while remaining valid. Original targeted lineage,
untracked-source and external-path controls also failed (5/5).

Verification now compares every exported model payload with its authoritative
result, reports missing rows and invalid durable hashes/states, checks executed
dataset bytes, and uses read-only SQLite. Package components and their digests
are anchored in the authoritative manifest, so editing both a package component
and its self-declared checksum does not certify it. Missing required components
are invalid. Statuses are valid/incomplete/unverified/invalid, with package
integrity and benchmark readiness separate. CLI exit codes: invalid=2;
incomplete/unverified=3. Empty or incompletely linked evidence is never ready.

Lineage contains real task/result/data IDs and recorded artifact dependencies,
with unavailable input links marked incomplete. Production artifact completion
follows in B16/C04; merely constructing a graph is not proof of full lineage.
Code identity hashes tracked and untracked contents. External paths have unique
portable identifiers and require an explicit location mapping to verify.

The prior test's expectation that a package with missing data and unrecorded
input lineage was globally valid was corrected: its package integrity remains
valid while readiness is incomplete. This is an intentional stricter contract,
not suppression of a regression. Current affected tests: 23 passed (provenance
integrity, prior provenance, task manifest), 24.91 s. Follow-up provenance checks
after dataset/durable-state controls: results recorded below.

Follow-up B05 provenance suite: 13 passed, 12.03 s.

## B07 / B08 / C01 / C02 / C15: analysis inputs and bounds

Original controls: 6 failed (right-only failed row IndexError, AUC 1.5 accepted,
disjoint runs pooled, operator semantics ignored, durable change retaining
snapshot ID, and unbounded eligible task silently dropped). The corrected
reader selects one declared run, checks every compatibility field and shared
data/code/environment identity, excludes range-invalid metrics, chooses an
available source safely and supports explicit empty outcomes. Snapshot IDs
hash authoritative task/attempt/result contents. Reader-added source-line
metadata no longer creates false payload conflicts; attempt sequence determines
latest attempt. Bounds are unsupported if any eligible task lacks a bound.

Analysis version: `dataset_equal_paired_run_v2`. Declared analysis root states
are retained; new configuration/input semantics make historical inference
outputs incompatible. Existing synthetic record helpers now supply the required
declared run and semantics rather than bypassing the integrity checks.
Affected critical/dataset/sensitivity tests: 17 passed, 2.69 s; added empty-side
control is included in the next affected check. Historical prefix availability
and unknown eligibility are addressed separately in B06/C14.

Integrated affected analysis plus production-resume checks: 19 passed,
13 warnings, 12.51 s. The B06 prefix controls still fail as expected before
that repair: final bounds 0.125 instead of [-0.4625,0.5375], and changing only
future metrics changes the prefix interval to 0.5275. These failures are retained.
