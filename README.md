# AutoFE-ShiftBench

A benchmark of fold-local feature engineering and classifier robustness under training corruption and predictor-wide stress partitions. The latter are transductive stress tests, not deployment estimates.

## Current scientific scope

The full configuration declares 25 datasets, five root seeds (42, 123, 456, 789, 2025), five folds, 14 conditions, 14 pipelines, and ten estimators: **8,750 precompute units and 1,225,000 model tasks**. These are intended counts, not completed results. Historical manuscript counts describe a separate seven-pipeline experiment.

Preprocessing and feature selection use training data. Held-out data do not choose features. Historical `class_prior_shift` performs majority-half relabeling; its accurate versioned meaning is training label corruption. Binary F1 and macro F1 are distinct metrics. Active inference pairs tasks within strata, averages differences within datasets, weights datasets equally, and uses dataset bootstrap intervals and Holm-adjusted sign-flip tests. Legacy pooled analyses are explicitly exploratory or retired.

The current protocol is `predictor_only_geometry_v3_integrity_metrics_fsva_v2`; seed derivation remains `sha256_canonical_json_u32_v1`. Corrected metrics, MI provenance, candidate/history schemas and distribution distances require compatible regenerated evidence. Historical artifacts remain preserved.

## Full-project Windows launcher in Conda P12

Use `run_project.ps1` as the single Windows launcher. It acquires every declared dataset with the hard 100,000-row cap, validates the pinned eligibility bundle, and then runs or resumes the complete durable benchmark using the active `python` by default. Conda remains supported explicitly with `-CondaEnv P12`. It chooses conservative worker, memory and disk limits from the current machine. No model task is silently replaced when GPU support is unavailable.

```powershell
.\run_project.ps1 -PlanOnly
.\run_project.ps1
```

New runs use stable, isolated destinations under `reports/runs/full/<run-id>/`
and pilots under `reports/runs/pilot/<run-id>/`. For example:

```powershell
.\run_project.ps1 -RunId experiment_001
.\run_project.ps1 -RunId experiment_001 -Resume
.\pilot_run.ps1 -RunId pilot_001
# Optional Conda selection:
.\run_project.ps1 -CondaEnv P12 -RunId experiment_001
```

Each run keeps its SQLite manifest, JSONL ledgers, coordinator log, worker
logs and launcher metadata together. An explicit `-RunRoot` remains supported
and is never silently redirected. See [docs/OUTPUT_STRUCTURE.md](docs/OUTPUT_STRUCTURE.md)
for the complete routing contract.

Use `-RunId` to resume a named run, `-SkipDownload` only when all raw CSV/metadata pairs already exist, and `-CpuOnly` only when a reduced run without XGBoost/CatBoost is scientifically acceptable. Use `pilot_run.ps1` for a bounded one-dataset smoke run; it is not the full-project launcher.

Benchmark CSVs have a hard **100,000-row maximum**. Larger source populations use the existing uniform pandas sample without replacement with seed 42; smaller sources keep all rows. Oversized manually supplied CSVs are rejected without editing them. Provider downloads may contain the original population before the local sample is selected:

The full launcher performs this acquisition step itself. For an explicitly selected acquisition, the equivalent command is `conda run -n P12 python -m src.data_loader --datasets haberman --max-rows 100000`.

It requests an exact source version, preserves existing bytes, rejects ambiguous metadata, and fails unresolved requests. Do not acquire the full collection implicitly.

## Execution, recovery and evidence

The SQLite manifest owns task membership, transactional claims, attempts and results. JSONL is an idempotent export. Dependency-ready tasks run in supervised, killable spawned processes; the worker limit covers precompute and model work, with at most one GPU model concurrently. The stop limit counts model-attempt launches, including retries. Deadlines are hard during work; bounded cleanup/export follows. Resource-infeasible exact work fails explicitly without changing rows, categories, precision or candidates.

Healthy CPU workers handle up to eight tasks and retire after 64 MiB of retained RSS growth; both limits are configurable. GPU workers retire after each task. CPU models use verified read-only selected numeric inputs. Native BLAS/OpenMP limits remain those of the corrected reference: reducing them changed Logistic Regression probabilities in the production comparison. Estimator `n_jobs=1` alone does not limit all native threads.

Split, feature, history and diagnostic publication uses locks, hashes and atomic replacement. Model workers load their requested prepared pipeline and never regenerate diagnostics. A missing required artifact reopens its precompute dependency for one bounded exact repair; prior result evidence remains archived. File counts alone are not completion proof.

### Rolling cache for a 500 GB drive

The production CLI defaults to `--cache-policy rolling`. It prepares one dataset/seed/fold/condition unit, executes all its declared pipeline/model tasks with the configured worker count, and closes those worker processes before admitting the next unit. Task deadlines, stop requests, failed models, missing results or failed checks retain the unfinished unit's inputs.

Eviction requires completed task and attempt ownership, successful defined metrics (or explicitly undefined metrics), matching scientific identities, authoritative SQLite result hashes, fsynced JSONL export acknowledgements, a read-back comparison of every exported result, verified preparation/input hashes, retained diagnostic evidence, and confirmed worker/descendant exit. Windows exclusively opens every disposable file with no sharing, verifies its bytes through the handle, and sets deletion disposition only after all checks pass. A sharing violation or mapped view blocks cleanup; partial deletion intent is recorded for recovery. No directory tree is recursively removed.

Only selected train/test DataFrame pickles and their derived float32 numeric arrays are removed. Raw datasets, fitted preprocessing, splits, descriptors, metadata, selection histories, diagnostics, results, progress, tables and figures are retained. Receipts under the manifest directory's `cache_lifecycle/` folder distinguish verified historical input hashes from currently available input files. Completed resume does not regenerate intentionally evicted matrices; a new run can rebuild them only with exact scientific hash agreement. The current runner does not save fitted classifier model binaries.

```powershell
.\run_project.ps1 -MaxWorkers 2 -CacheMaxGiB 100 -MinFreeGiB 50
```

This command runs the configured benchmark; it is not a cache-only command. `--cache-max-gib` bounds admission of new units using the disposable matrix footprint; retained evidence is governed by the free-space reserve. Disk checks stop dispatch and preserve partial artifacts, rather than reducing data or silently evicting unfinished work. A unit in progress and atomic temporary writes can exceed the admission threshold. `--cache-policy retain` preserves every input. Concurrent production coordinators sharing a cache namespace are rejected by an OS lease. Worker PID and creation-time registrations are persisted before commands are dispatched; surviving workers from a crashed coordinator block another run until they exit.

The storage probe's stopped cache-only run and its files remain preserved. The 100,000-row KDDCup99 sample still has a one-row class and cannot satisfy stratified five-fold CV; rolling storage does not alter that scientific limitation.

```powershell
conda run -n P12 python -m src.check_progress
conda run -n P12 python -m src.stats_analysis --help
conda run -n P12 python -m src.provenance_cli verify --help
```

Verification is read-only and distinguishes package integrity, missing evidence and benchmark readiness. Analysis selects one declared run and verifies scientific compatibility. Active tables, figures and the notebook use the same dataset-level contract; required raster quality and vector output defaults are retained.

## Files and their roles

| Location | Responsibility |
| --- | --- |
| `config/dataset_list.yaml` | Declared dataset requests |
| `src/pipeline_runner.py`, `src/coordinator.py` | Production preparation, dispatch, deadlines and writer supervision |
| `src/task_manifest.py`, `src/historical_snapshot.py` | Authoritative accounting, fencing, recovery and cutoff states |
| `src/data_loader.py`, `src/splitters.py`, `src/shift_generator.py` | Acquisition, splits and training corruption |
| `src/preprocessing.py`, `src/feature_engineering.py`, `src/operator_registry.py` | Fold-local mapping and expression universe |
| `src/feature_selection.py`, `src/fsva.py` | Selection and complete declared diagnostics |
| `src/artifact_integrity.py`, `src/prepared_inputs.py`, `src/resource_limits.py` | Verified prepared artifacts and allocation guards |
| `src/model.py`, `src/model_prediction.py`, `src/device_policy.py`, `src/evaluation.py` | Estimators, proven prediction reuse, device validation and defined metrics |
| `src/dataset_statistics.py`, `src/sensitivity_analysis.py`, `src/reporting.py` | Paired dataset inference, incomplete-run regimes and reporting inputs |
| `reproduction/figures.py`, `src/generate_tables.py` | Sole manuscript-figure generator and table generator |
| `notebooks/visualization.ipynb` | Read-only viewer for existing manuscript figures |
| `src/provenance.py`, `src/provenance_evidence.py` | Anchored evidence and actual artifact lineage |
| `tests/`, `reproduction/` | Regression controls and bounded reproduction tools |
| `provenance/critical_repair/` | Scientific contracts, complete issue register and engineering record |
| `paper/` | Historical manuscript and publication assets; preserved |
| `data/`, `reports/` | Ignored local datasets, caches, manifests and outputs |

## Verification record

Implementation and safe regression verification are documented in
`docs/OUTPUT_STRUCTURE.md`, the external release audit, and the compact
`report/` evidence package. Historical scientific contracts and large local
evidence are intentionally not part of the reviewer-release checkout.

## License

See `LICENSE`.
