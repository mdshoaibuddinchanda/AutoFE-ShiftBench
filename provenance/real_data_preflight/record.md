# Real-data preflight and bounded calibration

Scope: audit the current local collection, freeze the intended grid without
dispatch, run the declared isolated production pilot, verify recovery/storage,
and commit/push reviewable changes. Existing bulk caches, stopped jobs, raw
versions, P12, `.venv`, and manuscript assets are preserved.

## Starting source and environment

Branch `revision/leakage-seed-stability`, starting HEAD
`33c34571060761f698c818769ed86f2f847acb53`, initially clean. No applicable
AGENTS.md. Actual Windows host: i5-11260H, 4 physical/8 logical CPUs,
31.73 GiB RAM, RTX 3050 Laptop 4096 MiB, driver 616.64. P12 Python 3.12.14;
NumPy 1.26.4, pandas 2.2.3, SciPy 1.13.1, sklearn 1.5.2, XGBoost 2.0.3,
CatBoost 1.2.5, LightGBM 4.3.0. Native OpenBLAS/OpenMP pools retain their
inherited eight-thread limits. The rejected one-thread rewrite stays disabled.
Detailed machine/root/package/thread/starting footprints: local `environment.json`.

## Dataset verification and eligibility

All 25 configured working CSVs reproduce byte-for-byte from retained provider
responses and recorded selected positions. OpenML verification parses a preserved
copy of the existing sklearn download cache with network access forbidden,
compares provider ID/version/name/target/checksum, verifies decompressed raw MD5,
recomputes the original parsed source-frame fingerprint, applies exact pandas
sample-without-replacement seed 42, and reproduces the CSV SHA-256. UCI 602 uses
the retained original archive/ARFF and its SHA-256. No new download occurred.

All existing capped populations remain at most 100,000 rows. Airlines:
539,383→100,000; declared Covertype source 180: 110,393→100,000;
KDDCup99: 494,020→100,000. Smaller sources retain all rows. Dry Bean is
UCI 602 version 1, 13,611 rows/16 predictors/seven classes; the wrong OpenML
Penguins version is preserved separately and excluded. `datasets.md` lists every
source and working-byte hash; local `datasets.json` retains full schemas, counts,
missingness, source evidence, row-policy and source-position fingerprints.

Five roots × five folds × all conditions were checked under actual split rules,
without classifier fits. 8,750 intended units: 8,335 split-eligible and 415
scientifically infeasible. The 415 units imply 58,100 planned model skips,
leaving 1,166,900 split-eligible intended model tasks; estimator/device/resource
eligibility is an additional requirement.

KDD has 21 observed classes; some observed classes have only one row, and phf
and spy are absent from the preserved sample. Its 300 stratified units (12
conditions × five roots × five folds) reject the exact five-fold requirement.
Five covariate and ten population units additionally leave held-out classes
absent from training. The other 35 geometry units are split-eligible, with
possible explicitly undefined AUC/PR-AUC. Mushroom and kr-vs-kp have no numeric
predictors under the declared geometry rule: 50 units each are planned skips.
No sampling, labels, folds, geometry, or condition semantics were substituted.

Duplicate predictors and cross-fold predictor-hash overlaps are recorded as
evaluation limitations. They establish neither group independence nor a new
group-aware protocol. Geometry uses full predictor populations transductively;
learned model transformations remain training-only.

## Demonstrated execution repairs

- `graceful_stop` previously terminated owned work. A soft stop now latches,
  drains active attempts under existing task/run deadlines, preserves unfinished
  unit inputs, and resumes remaining declared membership. Hard deadlines remain.
- Planned split infeasibility now propagates to preparation and all dependent
  model records without claims; rolling advancement and final status distinguish
  declared scientific skips from unexpected failure. Sensitivity recognizes the
  explicit scientific-infeasibility reason family.
- Opt-in cache-base isolation keeps protocol, seed derivation, and scientific
  content unchanged. The bounded driver selects exact named units and freezes
  a separate complete intended nonexecuting manifest.
- Payload publication/numeric writes/result commits/JSONL writes and streamed
  export check the free-space reserve under a shared project write lease.
  This is not a filesystem quota: external applications and SQLite bookkeeping
  can consume space. Admission remains a disposable-matrix threshold, not an
  absolute peak guarantee.
- Resource/device outcome names were being normalized to worker exceptions;
  their distinct nonretryable classes are now retained. Exact allocation budget
  is explicit and configurable in ExecutionConfig; pilot ceiling 4 GiB is also
  limited by currently available memory/concurrency, without changing precision.
- The full suite exposed an existing ambiguous-metadata error-order mismatch:
  absent source identity was classified as a provider conflict. The established
  incomplete-identity rejection now runs first. Existing bytes stay untouched.
- Optional per-process operational timing sidecars record stages and complete
  resource sampling; immutable scientific metadata remains separate.

Controls: `tests/test_pilot_controls.py` initially 8 passed in 10.04 s. Full
P12 suite first 260 passed/one failed (metadata rejection order), seven subtests,
15 warnings, 156.30 s. After repair: **261 passed, seven subtests, 15 warnings,
143.38 s**. Historical 80 was a focused subset, 227 the pre-rolling full suite;
current collection includes the later rolling and eight new controls.
Exact commands/XML and counted factory-fit verification hooks are retained under
the ignored evidence root, not presented as real-data timings.

## Pilot declared before dispatch

Local evidence root: `reports/real_data_preflight_20261003/`.
`reproduction.real_data_preflight` phases audit, freeze, pilot, report use exact
named selections and the production coordinator/worker/writer paths. Pilot:
Haberman clean root42/fold1, Dry Bean clean root42/fold1, APS clean root42/fold1,
Dry Bean Gaussian 0.05 root42/fold1, Dry Bean clean root123/fold1. All 14 true
pipeline identities and ten original estimators: five preparations/700 intended
models. Metadata selects small numeric, medium multiclass, and large wide missing
data; accuracy has not been used for selection.

Hard envelope: 840 model-attempt/factory-fit launches across pilot/verification,
four hours cumulative execution, two supervised workers, one GPU concurrently,
30-minute task ceiling bounded by remaining run time, 100 GiB disposable admission,
50 GiB free reserve. FSVA 128 rows, FD32, all default magnitudes/full histories;
existing row populations/categories/features/candidates/precision/hyperparameters
unchanged. A predeclared mid-unit graceful stop and same-run resume exercise
recovery. Any unexpected unit failure retains cache and stops advancement.

Commands (bounded; do not substitute the full benchmark launcher):

```powershell
conda run -n P12 python -B -m reproduction.real_data_preflight audit
conda run -n P12 python -B -m reproduction.real_data_preflight freeze
conda run -n P12 python -B -m reproduction.real_data_preflight pilot --interrupt
conda run -n P12 python -B -m reproduction.real_data_preflight pilot
conda run -n P12 python -B -m reproduction.real_data_preflight report
```

The freeze and audit phases preserve existing outputs. Inspect existing budgets
before any pilot continuation. Final outcomes/projections are in
[results.md](results.md). The frozen pilot stopped with 166 completed models,
11 failed and 523 pending, plus two completed/three pending preparations.
No full-run completion or 500 GB capacity promise is made. The later probability
reduction repair changes executed Python content: the archived pilot must not
be resumed under that changed source.

## Remaining issue-register classifications

Historical corrected B/C entries retain regression evidence; real-data source,
split eligibility, and rolling calibration now have separate local evidence.
Scientific blockers are the preserved infeasible populations/policies above.
Resource/device blockers remain unmeasured large-estimator workspace, finite
VRAM, full retained-evidence growth and unprofiled runtime strata. Native
oversubscription remains because its tested reduction changed numerical outputs.
Optional deferred work: normalized sensitivity storage, complete regime inference
reuse, Monte Carlo/bootstrap batching, RNG-preserving perturbation vectorization,
extra import splitting, rendering/worker-recycling calibration. No unsupported
optimization from those entries is enabled. Full-scale analysis materialization
also needs a memory profile; passing pilot readers cannot certify million-row use.

## References checked

Installed-source checks and Windows fault execution are primary evidence.
[Microsoft file-sharing semantics](https://learn.microsoft.com/en-us/windows/desktop/FileIO/creating-and-opening-files)
support the exclusive-handle gate.
[sklearn 1.5 native parallelism](https://scikit-learn.org/1.5/computing/parallelism.html)
distinguishes estimator n_jobs from native BLAS/OpenMP pools. Neither documentation
is substituted for actual device fits, mapped-view tests or measured capacity.

## Final verification

After the demonstrated probability-sum repair, the complete P12 suite passed
**263 tests and seven subtests**, 15 warnings, in 151.60 seconds. Focused metrics
passed 12 tests. In-memory compilation passed 86 Python files/four notebook code
cells; `git diff --check` passed. The original pilot remains stopped with its
failed unit/cache intact. Same-run interruption, completed-unit resume, Windows
faults and incomplete provenance are reported with their exact scope in
[results.md](results.md) and [metric_validation_repair.md](metric_validation_repair.md).
