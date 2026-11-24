# AutoFE-ShiftBench

Research code for fold-local feature engineering and classifier robustness
under training corruption and predictor-wide stress partitions. Predictor-wide
partitions are transductive stress tests, not deployment-shift estimates.

## Quick navigation

| Need | Location |
| --- | --- |
| Install and run | `run_project.ps1`, `pilot_run.ps1` |
| Output locations | [`docs/OUTPUT_STRUCTURE.md`](docs/OUTPUT_STRUCTURE.md) |
| Public scientific evidence | [`report/`](report/) |
| Safe verification tests | `tests/` |
| Reproduction utilities | `reproduction/` |

## Scientific scope

The declared corrected grid contains 25 datasets, five root seeds, five folds,
14 conditions, 14 pipelines and ten estimators. These are configured counts,
not claims that every task completed. Dataset-level inference uses paired
contrasts, dataset bootstrap intervals and Holm-adjusted tests. Preprocessing
and feature selection are fold-local; held-out data do not select features.

Protocol identity:

```text
predictor_only_geometry_v3_integrity_metrics_fsva_v2
```

Seed derivation:

```text
sha256_canonical_json_u32_v1
```

## Installation

Create any supported Python or Conda environment, then install the pinned
dependencies:

```powershell
python -m pip install -r requirements.txt
```

The launchers use the active `python` by default. To select a Conda
environment explicitly, pass `-CondaEnv <name>`; to select another interpreter,
pass `-PythonExe <command-or-path>`.

## Running experiments

Use a stable identifier so repeated execution resumes the same experiment:

```powershell
.\run_project.ps1 -PlanOnly
.\run_project.ps1 -RunId experiment_001
.\run_project.ps1 -RunId experiment_001 -Resume
.\pilot_run.ps1 -RunId pilot_001
```

Full runs are written to `reports/runs/full/<run-id>/`; pilots use
`reports/runs/pilot/<run-id>/`. Each run keeps its database, manifests,
results, coordinator log, worker logs and launcher metadata together.

Override the output root when required:

```powershell
.\run_project.ps1 -RunId experiment_001 -RunRoot "E:\Experiments\experiment_001"
```

Datasets are downloaded to `data/raw/` with the 100,000-row cap. The versioned
cache remains under `data/cache/<protocol>/<seed-scheme>/`; set
`AUTOFE_CACHE_BASE` to place that cache elsewhere.

Before a non-plan full run, the launcher verifies the committed eligibility
contract at `config/eligibility/verified_eligibility.json` against the pinned
SHA-256 in `config/eligibility/manifest.json`. This compact contract records
which configured dataset/seed/fold/condition units are eligible or
scientifically infeasible; raw datasets are downloaded separately. An audited
external contract can be supplied with `-EligibilityPath <path>` only when it
has the same pinned hash.

## Recovery and cache safety

The SQLite manifest is authoritative for task membership, claims, attempts and
results. JSONL files are durable exports. The default rolling cache policy
evicts only verified disposable matrices after completed units; raw data,
metadata, diagnostics, histories, results and receipts are retained. A
sharing violation, active worker or failed integrity check blocks eviction.

Use `--cache-policy retain` when every prepared input must remain on disk. The
runner does not save fitted classifier binaries. KDD Cup 99 feasibility and
Covertype coverage limitations are scientific constraints, not cache behavior.

## Safe verification

These commands do not download data or fit models:

```powershell
python -m pytest tests/test_output_routing.py tests/test_sanity_checks.py tests/test_reporting_contract.py -q
python -m src.pipeline_runner --help
```

The full benchmark and reproduction utilities that consume large historical
artifacts are intentionally not run by the release checks.

## Repository map

| Area | Responsibility |
| --- | --- |
| `config/` | Dataset declarations and configuration |
| `src/` | Production loading, preparation, scheduling, evaluation and reporting |
| `tests/` | Regression and non-training verification |
| `reproduction/` | Bounded audits, diagnostics, comparisons and figure generation |
| `docs/` | Operational output and portability documentation |
| `report/` | Compact, sanitized GitHub evidence |
| `data/` | Local datasets and versioned caches; not published |
| `reports/` | Large local scientific outputs; not published |

## Public-release boundaries

The reviewer release intentionally excludes raw datasets, caches, databases,
large ledgers, history corpora, manuscripts, bibliographies, reviewer
correspondence and journal-submission files. Historical evidence remains in
the local research checkout or an external archive. The public summaries in
`report/` include their source-hash manifest and state their limitations.

## Platform support

Windows is the validated launcher platform. Linux/macOS path handling uses
`pathlib` and the active Python interpreter, but full benchmark execution has
not been tested on those platforms. GPU support depends on the installed
driver and optional estimator packages.

## License

See [`LICENSE`](LICENSE).
