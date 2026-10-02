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

Final record:

- `D:\Conda\P12\python.exe -m pytest -q`: **46 passed**, 13 existing dependency warnings, and 7 subtests passed in 17.54 seconds.
- `D:\Conda\P12\python.exe -m pytest -q tests/test_production_task_resume.py::ProductionResumeTests::test_worker_execution_and_resume_reconstruct_identical_task_inputs`: **1 passed** in 9.88 seconds. The isolated writer/resume path exited cleanly; the earlier Windows writer access violation was not reproducible in this run.
- `D:\Conda\P12\python.exe -m compileall -q -f src tests`: completed with no compile errors. The two pre-existing `invalid escape sequence \\D` warnings in `src/analysis/structural_analysis.py` remain unrelated. `git diff --check` passed.
- Bounded production smoke ran in a disposable source copy with two synthetic configured datasets, two seeds, one fold, clean plus Gaussian noise, `Raw`, `Raw_Capped`, `Raw_Full`, and `AutoFE_Baseline`, logistic regression, one worker, 30-second task timeout, one attempt, and FSVA diagnostics capped at 16 rows. The manifest declared **40 intended tasks** (8 precompute + 32 model); **32 completed** (8 precompute + 24 model rows in the ledger) and **8 model tasks remained pending** after the declared stop limit was reached at a dataset boundary. There were no failures, timeouts, or skips, with **32 attempts**, **32 durable results**, and **24 diagnostic-bearing ledger rows**. The ledger contained all four declared pipelines. The smoke source copy was removed after inspection.
- The active sensitivity CLI read that smoke snapshot and wrote **60 summary rows** plus coverage, pair, regime, bound, leave-one-dataset-out, cutoff, and exclusion artifacts. The provenance package and read-only verifier returned `overall_status: valid`; protocol and seed-scheme checks were valid, synthetic dataset bytes were hashed, and the remaining configured datasets were explicitly unavailable.
- The current repository contains no raw benchmark CSVs or full result ledger. No full benchmark, manuscript, or historical claim was generated. The existing `.venv` was not removed; all commands used Conda `P12`.

Commits pushed to `origin/revision/leakage-seed-stability`:

- `4dcb8ef` — Add incomplete-run sensitivity analysis.
- `b4b7b1a` — Add provenance packaging and verification.

Remote verification matched local `HEAD` at
`b4b7b1a1e9960583104f99b893ce81ddd0d42941`; the working tree is clean.
