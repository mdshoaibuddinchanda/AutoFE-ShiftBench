# Provenance and diagnostic file policy

This directory is the versioned audit record for the Reviewer #1 revision. The files here are **permanent research records**, including preliminary diagnostics and historical plans. They are not corrected benchmark results. New decisions supersede older plans by version; the older files stay intact so the decision trail remains reviewable.

## What to keep

| Material | Location | Retention and use |
|---|---|---|
| Source, fold, condition, operator, and scope audits | `dataset_schema_audit.*`, `group_fold_audit.*`, `group_seed_grid_audit_v1.json`, `reviewer1_*manifest*.json`, `reviewer1_launch_scope_v2.json`, `reviewer1_condition_crosswalk.md`, `reviewer1_change_catalog.md` | Versioned methods evidence. Cite the final frozen versions, retain earlier versions as history. |
| Pilot, recovery, host, and calibration diagnostics | `*pilot*`, `*preflight*`, `*profile*`, `performance_*`, `host_measurement.json`, `launch_host_measurement_v2.json`, `friend_pc_launch_audit.md` | Permanent launch-gate evidence. Their timings and scores are **not** corrected benchmark effects. Host measurements apply only to the named computer. |
| Internal independent checks | `*_agent_report.md`, `audit_report.md` | Permanent internal audit trail. These reports are not manuscript tables or performance evidence. |
| Historical result identity | `original_run.json`, `corrected_smoke_manifest.json` | Keep historical and synthetic results separately labeled. |
| Pilot visualization | `figures/four_dataset_pilot_status.svg` | Permanent explanatory graphic. It shows task accounting, not ROC-AUC conclusions. |
| Mechanism scope and bounded evidence | `mechanism_history_scope_v1.json`, `run_mechanism_history.py`, `analyze_mechanism_history.py`; local `../corrected_runs/mechanism-smoke-*/` | Versioned clean-condition candidate/Jacobian protocol and local code-verification evidence. The bounded samples are not corrected performance effects. |
| Dataset inputs | `../data/raw/*.csv` and matching `*_meta.json` | Local, ignored source snapshots. Keep exact hashes and sidecars for a reproducible run. |
| Corrected run ledgers | `../corrected_runs/<run-id>/` | Local, ignored **primary research evidence**. Preserve `manifest.json`, `results.jsonl`, SQLite files, scheduler result artifacts, and run IDs after a run. Back them up before any cleanup. |
| Corrected paper assets | `../corrected_runs/paper_assets/` | Local, ignored publication drafts generated from completed corrected runs; CSV tables, PNG/PDF figures, and a source-hash manifest. Keep and back up with the source ledgers. Synthetic fixture figures in pytest's temporary directory are only code-verification evidence. |
| Reporting pilot smoke | `../corrected_runs/reporting_pilot_smoke/` | Local, ignored diagnostic generated from both completed four-dataset pilots. Its custom scope and asset manifest state `pilot code verification`; every figure carries a visible pilot diagnostic label. These values must not enter corrected paper conclusions. |
| Large dataset forced restart | `large_dataset_recovery_v1.py` and, after successful execution, `large_dataset_recovery_v1.json`; local `../corrected_runs/large_recovery/` | Versioned 600-second lease recovery proof on `airlines`, one bounded 20-cell run per split policy. The local ledgers and coordinator logs are diagnostic evidence and are not paper performance results. |
| Historical outputs | `../reports/tables/`, `../paper/figures/` | Historical, not corrected. Never overwrite with the corrected campaign. |

## Regenerable and transient files

`__pycache__/`, `*.pyc`, `.pytest_cache/`, zero-byte SQLite WAL files, and coordinator `.lock` files are execution auxiliaries. A `.lock` may still protect an active run, so never delete it while a coordinator is running. `corrected_runs/<run-id>/cache_bounded/` contains regenerable feature payloads managed by the runner; its metadata and high-water report remain part of recovery evidence. A `scheduler_results/` JSON file is a durable result artifact and must **not** be treated as disposable cache. There were no `.tmp`, `.part`, or `.bak` files under `corrected_runs/` at the 2026-10-01 inventory.

At that inventory, `corrected_runs/` held 22,290 files (2.13 GiB), mostly completed pilot and scale-preflight evidence. `data/raw/` held 25 CSVs and 25 sidecars (0.12 GiB). `reports/tables/` held four historical files (0.59 GiB). The `D:` volume had 434.94 GiB free. These counts describe this checkout only and may change during bounded verification.

`corrected_runs/large_calibration/calibration-row_level-001/` is an **interrupted diagnostic**, retained after its thread-pool settings were found to be unfrozen. It is not a corrected result and is not a resume target for the revised code identity. The replacement calibration uses a new `-002` run ID with numerical-library threads set to one.

`corrected_runs/large_recovery/recovery-airlines-row-001/` and `corrected_runs/large_calibration/calibration-group_aware-002/` are **failed concurrent-memory diagnostics**. Running both four-worker jobs at once caused allocation failures on this host. Their logs and ledgers are retained for the resource audit and are never resumed or used as corrected performance results. The completed `calibration-row_level-002` (120/120 successes) remains valid; clean group calibration uses `calibration-group_aware-003` alone. The 600-second recovery check will use a new diagnostic version after group calibration finishes.

The pre-existing execution-count edit in `../notebooks/visualization.ipynb` is outside the Reviewer #1 run freeze and must not be included in its code commit. The corrected runner and analysis do not use that notebook.
