# Code revision record: reviewer problems 9–10

Date: 2026-10-03

Scope: incomplete-run sensitivity analysis and full provenance/reproducibility
metadata. The full benchmark, external dataset download, and manuscript remain
outside scope.

## Problem 9 — incomplete-run sensitivity

`src/sensitivity_analysis.py` freezes a single SQLite manifest snapshot and a
SHA-256 ledger fingerprint before constructing the analysis universe. Only
manifest model tasks are included; precompute rows, retries, cache files, and
diagnostic artifacts are not observations. Durable manifest results provide the
metric values, while ledger-only or malformed rows are explicit exclusions.

The predeclared `SensitivityConfig` emits the historical `Raw` contrast and
`Raw_Capped` and `Raw_Full` fair controls against `AutoFE_Baseline`. It writes
task coverage/status tables, matched task blocks, regime membership,
dataset-level contrasts, Holm-adjusted summaries, ROC-AUC identification
bounds, leave-one-dataset-out rows, and stop-prefix/cutoff summaries. Regimes
are `primary_observed`, `matched_task_blocks`, `common_eligible_task_set`,
`complete_datasets`, fixed coverage thresholds, and timestamp stop prefixes.
The active reader is available through `src.stats_analysis` and its CLI.

## Problem 10 — provenance and reproducibility

`src/provenance.py` and `src.provenance_cli` provide a read-only registry,
compatibility checker, and metadata-only package. The package records local
dataset status and hashes when bytes exist, explicit unavailable/unverified
states, code commit/branch/worktree identity, portable Conda P12 environment
identity, manifest/ledger/analysis artifact hashes, lineage edges, and
reproduction commands. It never invents missing source metadata or copies
datasets, caches, credentials, or machine-specific prefixes. Protocol and seed
scheme compatibility are checked at run and task level.

`reproduction/README.md` contains bounded dry-run, smoke, sensitivity, and
provenance commands. The existing `.venv` and Conda `P12` environment are
retained; no environment was deleted or recreated.

## Verification record

The final P12 test, compile, smoke, interrupted-run/resume, provenance, and
remote verification results are recorded below as they are completed. The
checkout currently has no full raw dataset collection or historical full-run
ledger, so no manuscript-scale empirical conclusion is claimed.
