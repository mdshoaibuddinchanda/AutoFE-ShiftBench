# AutoFE-ShiftBench

A benchmark of fold-local feature engineering and classifier robustness under training corruption and predictor-wide stress partitions. The latter are transductive stress tests, not deployment estimates.

## Current scientific scope

The full configuration declares 25 datasets, five root seeds (42, 123, 456, 789, 2025), five folds, 14 conditions, 14 pipelines, and ten estimators: **8,750 precompute units and 1,225,000 model tasks**. These are intended counts, not completed results. Historical manuscript counts describe a separate seven-pipeline experiment.

Preprocessing and feature selection use training data. Held-out data do not choose features. Historical `class_prior_shift` performs majority-half relabeling; its accurate versioned meaning is training label corruption. Binary F1 and macro F1 are distinct metrics. Active inference pairs tasks within strata, averages differences within datasets, weights datasets equally, and uses dataset bootstrap intervals and Holm-adjusted sign-flip tests. Legacy pooled analyses are explicitly exploratory or retired.

The current protocol is `predictor_only_geometry_v3_integrity_metrics_fsva_v2`; seed derivation remains `sha256_canonical_json_u32_v1`. Corrected metrics, MI provenance, candidate/history schemas and distribution distances require compatible regenerated evidence. Historical artifacts remain preserved.

## Bounded execution in Conda P12

The Windows launcher forwards runner arguments through P12. It does not install dependencies or acquire data.

```powershell
.\setup_and_run.bat --dry-run-manifest --max-datasets 1 --max-seeds 1 --max-folds 1 --max-conditions 1 --pipelines Raw --models logistic_regression --max-workers 1
```

For an explicitly selected local-data pilot, remove `--dry-run-manifest` and set `--stop-after-tasks`, `--task-timeout-seconds` and `--run-wall-time-seconds`. Inspect `python -m src.pipeline_runner --help` in P12 for supported controls. `python main.py` delegates to this same runner without acquisition. No full benchmark is claimed by this repair record.

Acquisition is a separate explicit selection:

```powershell
conda run -n P12 python -m src.data_loader --datasets haberman
```

It requests an exact source version, preserves existing bytes, rejects ambiguous metadata, and fails unresolved requests. Do not acquire the full collection implicitly.

## Execution, recovery and evidence

The SQLite manifest owns task membership, transactional claims, attempts and results. JSONL is an idempotent export. Dependency-ready tasks run in supervised, killable spawned processes; the worker limit covers precompute and model work, with at most one GPU model concurrently. The stop limit counts model-attempt launches, including retries. Deadlines are hard during work; bounded cleanup/export follows. Resource-infeasible exact work fails explicitly without changing rows, categories, precision or candidates.

Split, feature, history and diagnostic publication uses locks, hashes and atomic replacement. Model workers load their requested prepared pipeline and never regenerate diagnostics. A missing required artifact reopens its precompute dependency for one bounded exact repair; prior result evidence remains archived. Caches needed for resume and evidence referenced by committed results are retained. File counts alone are not completion proof.

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
| `src/model.py`, `src/device_policy.py`, `src/evaluation.py` | Estimators, device validation and defined metrics |
| `src/dataset_statistics.py`, `src/sensitivity_analysis.py`, `src/reporting.py` | Paired dataset inference, incomplete-run regimes and reporting inputs |
| `src/plotting_q1.py`, `src/generate_tables.py`, `notebooks/visualization.ipynb` | Active publication output paths |
| `src/provenance.py`, `src/provenance_evidence.py` | Anchored evidence and actual artifact lineage |
| `tests/`, `reproduction/` | Regression controls and bounded reproduction tools |
| `provenance/critical_repair/` | Scientific contracts, complete issue register and engineering record |
| `paper/` | Historical manuscript and publication assets; preserved |
| `data/`, `reports/` | Ignored local datasets, caches, manifests and outputs |

## Verification record

See [the engineering record](provenance/critical_repair/record.md) and [the issue register](provenance/critical_repair/issue_register.json). Implementation, regression verification, production smoke, synthetic timing, real-data verification and full-run completion are separate states. No unsupported multi-day completion estimate is supplied.

## License

See `LICENSE`.
