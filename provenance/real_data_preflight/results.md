# Bounded real-data preflight results — 2026-10-03

**Readiness: blocked for a full benchmark.** The authorized bounded pilot stopped
at a genuine probability-validation conflict. Its failed unit, completed results
and diagnostics are preserved. A demonstrated float32 summation defect was
repaired without increasing the tolerance. A reproduced MLP case still exceeds
the declared tolerance. No full benchmark, bulk-cache restart, environment
deletion, manuscript change or new broad download occurred.

## 1. Source, data and full intended accounting

Branch: `revision/leakage-seed-stability`; starting HEAD `33c34571060761f698c818769ed86f2f847acb53`.
The frozen original pilot executed clean commit
`05d6c753248d969cc574fdeb4c9a2955e364a2d0`; subsequent metric repair `b2bfa35`
is a separate change, not relabeled historical execution. Protocol
`predictor_only_geometry_v3_integrity_metrics_fsva_v2` and seed derivation
`sha256_canonical_json_u32_v1` remain unchanged.

All 25 configured sources were verified against retained provider responses,
metadata/checksums, source-frame fingerprints and exact selected source positions.
Working CSV reconstruction was byte-for-byte. [datasets.md](datasets.md) gives
all source IDs/versions, rows/features and CSV SHA-256. Local `datasets.json`
includes target definitions, parsing/dtypes, class counts, nonfinite/missing
values, duplicate overlaps, source fingerprints and selected-position hashes.
Final SHA verification confirms all 25 working CSVs remain unchanged.

Uniform pandas sample seed42/cap100,000 applies only to Airlines 539,383,
declared Covertype OpenML180 population 110,393, and KDDCup99 494,020.
The remaining populations retain every row. Dry Bean is UCI602/v1,
13,611 rows/16 predictors/seven classes; the incompatible earlier Penguins
download remains preserved and excluded. Target is never an input predictor.
Geometry splits use full predictor populations transductively; learned
preprocessing and corruption remain training-only. Exact predictor duplication
and 64-bit cross-fold overlap checks do not establish group independence.

The complete, nonexecuting full-grid gzip manifest is 84,357,002 bytes;
SHA-256 `b61622cf95740fa904090795d2b5d50d4d32327ad0bf17e68b4e493396cf0a8e`.
It contains 8,750 preparations and 1,225,000 model tasks, with **zero dispatched**.
An independent streamed verification found 1,233,750 unique scientific task IDs,
no duplicates, exactly 140 model members for every preparation and matching
planned-skip counts/SHA-256. It performed no task dispatch.
Five roots × five folds × 14 conditions × 25 datasets are accounted for:

| Population/policy issue | Infeasible units | Planned model skips |
| --- | ---: | ---: |
| KDD: 12 stratified conditions, five roots/folds | 300 | 42,000 |
| KDD: held-out classes absent from training in geometry | 15 | 2,100 |
| Mushroom: no numeric geometry predictors | 50 | 7,000 |
| kr-vs-kp: no numeric geometry predictors | 50 | 7,000 |
| Total | 415 | 58,100 |

Thus **8,335 units/1,166,900 model tasks are split-eligible**, before estimator,
metric/device/resource feasibility. KDD has 21 observed classes; ftp_write,
multihop, perl and rootkit each have one row; imap/land have three,
loadmodule two and warezmaster four. phf/spy are absent. Its 15 extra infeasible
geometry units are five covariate and ten population units. The other 35
geometry units have valid split geometry but may have undefined held-out-class
AUC/PR-AUC and additional rare-class estimator/calibration failures.

Rows, labels, folds and seeds were not changed to make KDD feasible. A different
population or rare-class evaluation protocol would require a new scientific
declaration and would change the evaluated population/estimand; even uncapped
source classes with fewer than five observations cannot satisfy this five-fold
rule. That alternative was not implemented.

## 2. Remaining blockers and deferred work

- **Probability contract:** original Dry Bean failures are seven MLP/four
  XGBoost. Accurate float64 accumulation fixes the demonstrated XGBoost false
  rejection while preserving 1e-7. An exact-input MLP reference still has
  represented probability mass residual 2.095644059396662e-7 and is rejected.
  See [metric_validation_repair.md](metric_validation_repair.md). No tolerance
  relaxation or probability renormalization was introduced. Not all failed
  combinations were rerun under the correction.
- **Scientific eligibility:** the 415 units above remain planned infeasible.
  Undefined metrics on otherwise valid tasks are different from these units.
  Default general launcher does not automatically load the preflight eligibility
  mapping: it would still encounter these populations as runtime preparation
  failures. The frozen nonexecuting manifest has the correct planned reasons;
  a future full dispatcher must consume that accounting explicitly.
- **Resource/device capacity:** actual small/medium fits confirm XGBoost CUDA0
  and CatBoost GPU, but do not establish wider-case VRAM/workspace support.
  APS execution, worst unit/temp peaks and accumulated retained evidence are
  unmeasured. The reserve lease guards known payload/array/commit/export writes;
  external writers, SQLite bookkeeping, profiling sidecars and recovery-copy
  overhead are not a filesystem quota. No reserve crossing occurred here.
- **Analysis scale:** pilot reader integrity passed; million-task sensitivity
  membership/snapshot materialization still needs a memory/time profile.
- **Optional optimizations:** expanded sensitivity-schema normalization,
  complete regime-inference reuse, Monte Carlo/bootstrap batching,
  RNG-preserving perturbation vectorization, extra import splitting, rendering
  and recycling calibration remain deferred. Native-thread reduction remains
  rejected because historical Logistic Regression equivalence failed.

The historical issue-by-issue statuses and current dispositions are in
[issue_dispositions.md](issue_dispositions.md). Ancillary files marked indexed
in [coverage.json](coverage.json) are not claimed manually reviewed.

## 3. Declared and actual pilot coverage

Run `run_a9c1f66889c309ba669dedb1`: five preparations/700 models. Limits before
dispatch: 840 total launches/four hours, two supervised workers/one GPU,
30-minute task ceiling limited by remaining run time, 100 GiB disposable
admission, 50 GiB free reserve, dynamic dense-allocation guard with 4 GiB ceiling.
Rows, categories, candidates, precision, model parameters and native threads
remain unchanged. FSVA: 128 sampled rows, FD32, magnitudes .001/.01/.05 and
full candidate histories. Both prepared units have all 14 complete diagnostics.

| Dataset/unit | Preparations complete/pending | Models complete/failed/pending | Model attempts |
| --- | --- | --- | ---: |
| Haberman, clean, root42/fold1 | 1 / 0 | 140 / 0 / 0 | 140 |
| Dry Bean, clean, root42/fold1 | 1 / 0 | 26 / 11 / 103 | 37 |
| APS, clean, root42/fold1 | 0 / 1 | 0 / 0 / 140 | 0 |
| Dry Bean, Gaussian .05, root42/fold1 | 0 / 1 | 0 / 0 / 140 | 0 |
| Dry Bean, clean, root123/fold1 | 0 / 1 | 0 / 0 / 140 | 0 |
| Total | 2 / 3 | 166 / 11 / 523 | 177 |

179 production attempts include two preparations. No pilot retry, timeout,
unexpected skip or running task remains. Of 705 total tasks: 168 completed,
11 failed and 526 pending. All 14 pipeline identities and all ten estimators
completed on Haberman. Dry Bean attempted each pipeline/model name, but only
37 of their 140 combinations. Unexecuted coverage is censored, not fast work.

The 143 counted verification/reference factory fits include all three full-suite
invocations (47 each) and two exact-input reproductions. Global total **320**.
Reallocation of three unused launch slots for required repair verification was
declared after the pilot halted; the 840/four-hour envelope was not expanded.
Budget charged 680.942 seconds (11.35 min), including conservative startup and
reader charges documented in `budget_ledger.json`. Actual coordinator pilot
invocations total **176.079 seconds**, separate from test/reference execution.

## 4. Measured stages, resources and storage

Actual host: i5-11260H (4 physical/8 logical CPUs), 31.73 GiB RAM,
RTX3050 Laptop 4,096 MiB/driver616.64; P12 Python3.12.14. No synthetic timing
was substituted. Cold preparation attempts: Haberman **3.341 s**, Dry Bean
**38.480 s**; all model tasks reuse verified prepared inputs. Haberman's active
coordinator unit cost including stop/resume segments and cleanup was **78.688 s**.

Shared stage totals below are instrumented function spans, not costs per model.
Inclusive parents contain child stages and must not be added to those children.
Diagnostic memoization executes 10 kernels on Haberman and nine on Dry Bean
while publishing all 14 pipeline-specific diagnostic records.

| Shared preparation stage, seconds | Haberman | Dry Bean |
| --- | ---: | ---: |
| Dataset load | .0010 | .0312 |
| Split | .0018 | .0104 |
| Training corruption (clean) | .0002 | .0010 |
| Preprocessor fit/transform | .0038 | .0098 |
| Candidate/scoring/selection, inclusive | .1172 | 28.5889 |
| Candidate generation, contained above | .0037 | .0793 |
| MI scoring, contained above | .0642 | 27.8005 |
| FSVA Jacobian | .0290 | .0885 |
| FSVA amplification | .0756 | .2590 |
| FSVA finite differences | .2732 | 1.0421 |
| Distribution distances | .2518 | 4.3772 |
| Numeric preparation, inclusive | .2818 | .5580 |
| Descriptor/numeric publication, inclusive | .4125 | .7069 |

Model factory-fit medians/ranges below include failed tasks' completed fits;
fit completion does not establish successful metrics. Each Haberman row has
14 observations; Dry Bean counts are incomplete, as shown.

| Estimator | Haberman median [min,max] s | Dry Bean n; median [min,max] s |
| --- | --- | --- |
| Logistic Regression | .0036 [.0023,.0249] | 1; .1478 [.1478,.1478] |
| Random Forest | .1326 [.1025,.3006] | 3; 9.4957 [3.7077,10.3457] |
| Extra Trees | .0837 [.0684,.1367] | 2; 1.3924 [1.1502,1.6345] |
| Calibrated Linear SVM | .0100 [.0086,.0256] | 2; 1.3751 [.2102,2.5400] |
| KNN | .0014 [.0010,.0019] | 2; .0035 [.0022,.0047] |
| Gaussian NB | .0016 [.0012,.0023] | 8; .0092 [.0034,.0174] |
| MLP | .0332 [.0179,.0668] | 7 failed; .8595 [.3692,2.6628] |
| LightGBM | .0187 [.0119,.0496] | 4; 6.9976 [1.2050,7.7948] |
| XGBoost | .3110 [.2622,.5330] | 4 failed; 3.1064 [1.5178,3.2381] |
| CatBoost | 1.0811 [.6958,1.8754] | 4; 1.3771 [1.0668,2.4536] |

Aggregate Haberman fit/prediction/metrics: 25.272/1.084/3.068 s; Dry Bean
77.241/6.920/1.842 s (11 metric exceptions). Prediction includes both training
and held-out arrays. Writer authoritative commits: 1.458 s/166 results;
fsynced-export acknowledgement updates: .862 s. Recovery export: .311 s.
Haberman cleanup **1.106 s**, including export readback .037 s and nested
receipt/hash gates. Across invocations receipt verification .159 s/prepared
integrity .327 s. These overlapping spans cannot be summed as wall time.

53 worker processes spawned/51 recycled. Spawn-call total .189 s excludes
child imports/startup; attempt durations include that overhead. The explicit
boundary release had zero idle workers because earlier processes already
exited; .000 s recorded by the Windows clock is not proof of free worker
startup/release. The actual receipt verifies 40 exited owned processes for
Haberman. Standalone fsync-write duration is not isolated; commit/ack/readback
and authoritative ownership prove persistence, with remaining overhead included
in attempt/coordinator times.

One-second resource samples (cache/GPU every two ticks):

- Largest sampled worker/writer RSS: **.462 GiB**; combined children **.958 GiB**.
  Machine peak used **19.377 GiB**, minimum available **12.356 GiB**. CPU median
  46.4%, maximum 100%; a worker reached 234.9% and 39 threads, consistent with
  inherited native pools (each reported eight threads). Coordinator RSS is not
  isolated; machine values include other applications.
- Whole-device GPU memory **371–3,835 MiB**, utilization maximum 82%. This
  includes external activity; actual device use is established separately by
  fitted estimator configuration. Model-specific VRAM peaks are not certified.
- Cache sampled peak **170,966,114 bytes (.159 GiB)**, disposable
  **160,143,550 (.149 GiB)**, retained **10,822,564**. Dry Bean retains all
  56 disposable artifacts because its unit failed. Haberman's 56 matrices were
  evicted, leaving 46 declared retained artifacts totaling **1,376,082 bytes**.
  Dry Bean's 46 declared retained artifacts total **9,436,828 bytes**.
- Minimum sampled disk free **407.595 GiB**, above the 50 GiB reserve.
  No temp file was seen at sampling instants; largest instrumented atomic byte
  payload **1,014,950 bytes**. Numeric temp writes are separately staged and
  reserve checked, but subsecond temp peaks can be missed. Peaks are lower
  bounds rather than absolute maxima.
- Task logical I/O: **1,347,265,392 bytes read/177,032,007 written**. Windows
  counters do not establish physical SSD traffic; nested spans are excluded
  from these task totals.

Prior preserved bulk cache: 22,754,117,472 bytes, including 11,245,683,109
disposable and 11,508,434,363 retained. Final footprint differs by one 1-byte
`.coordinator.lock` created by regression tests; original disposable byte
count is unchanged. The lock was preserved. No recursive cleanup or removal
of old bulk artifacts occurred. Model weights/checkpoints are not emitted by
this runner: fitted estimators are discarded after metrics; preprocessing,
splits/descriptors, metadata, histories, diagnostics, SQLite, JSONL, analyses
and receipts remain.

## 5. Recovery, faults and scientific integrity

The predeclared graceful interruption completed four models/one preparation in
9.141 s, drained active workers, preserved unfinished inputs and resumed under
the same run/configuration. The second invocation completed Haberman and stopped
on Dry Bean's unexpected unit failures. A third compatible resume before the
code repair took 4.063 s, dispatched **zero attempts**, verified Haberman's
receipt and refused the failed unit. No model scientific-task ID is duplicated:
166 JSONL rows equal 166 authoritative model results, with valid payload hashes.

Haberman receipt: 141 declared results, 56 deleted allowlisted files absent,
46 retained artifacts with matching hashes, 40 exited process registrations;
receipt SHA-256
`0b8ef7f0d263d30b56a89288eb1bd79ea038be8f1dcb4456c69589b0b57e41b7`.
Dry Bean has no eviction receipt and all its prepared inputs verify. Recomputed
purpose seeds, train/test row counts, selected identities/dimensions, diagnostic
settings and native thread policy match all 166 completed observations.
Histories retain all candidates: e.g. Dry Bean baseline/MI/random each 800 rows,
Haberman each 33. Model inputs remain historical float32/NaN preserved/inf1e10;
arithmetic FSVA coordinates use the separately declared float64 calculations.

Final Windows fixture suite independently exercised live mappings, corrupt
matrix/export/receipt, missing histories/input ownership, live registered workers,
writer failure, worker crash/retry, hard deadlines, and recovery after deletion
intent. Gates retain inputs or restore incomplete intent as declared. Complete
whole-run resume with zero attempts is verified by synthetic production fixtures;
the real pilot is incomplete, so only completed-unit/blocked-run resume is claimed.

Corrected paired reader produced ten stratum summaries and explicitly excluded
one missing partner. Sensitivity read exactly 700 model tasks and produced
480 summary rows. Provenance returned **461 valid checks/one incomplete check**
(`eligible_task_completion`), no conflict or invalid checks. Incompleteness is
the actual failed/pending coverage, not missing evicted-input evidence.
These analyses are pilot integration evidence, not manuscript inference.

## 6. Conditional projections and graded readiness

Only Haberman clean root42/fold1 has a complete measured unit. Repeating exactly
those observed costs across its 25 clean root/fold units gives a **32.8–44.5 min
scenario**: measured scheduling versus serial attempt/preparation/cleanup sum.
This assumes fixed costs across unseen roots/folds; it is neither a confidence
interval nor a reliable completion bound. Dry Bean's 37 attempts cannot forecast
its remaining combinations, and no timing for APS/non-clean conditions/second
root or other datasets is substituted. **Full-grid runtime upper bound unknown.**

For storage, linear repetition of the declared retained-artifact sizes over
25 clean units gives approximately 34.4 MB (Haberman) and 235.9 MB (Dry Bean),
before shared-artifact deduplication and result/export/analysis growth. These are
two selected-population scenarios, not estimates for all 8,335 eligible units.
Required drive space is preserved raw/bulk/reports + worst active disposable/temp
unit + accumulated retained evidence/results + the 50 GiB free reserve.
Several terms remain unmeasured. **A full rolling benchmark fitting a 500 GB
drive is not established.** The old 2,719,557,455,300-byte (~2.72 TB decimal)
retain-all estimate already used capped populations and a representative cache
probe; this .159 GiB pilot peak has different scope and is not its reduction
factor. No synthetic 9x factor or ideal linear worker scaling is used.

| Readiness area | Assessment |
| --- | --- |
| Dataset provenance | Verified for all 25 actual sources/working bytes |
| Split/metric eligibility | 415 planned units infeasible; MLP contract blocker remains |
| Model/device support | All10 completed on small control; medium/wide grid incomplete |
| Execution/recovery | Real unit/blocked resume and Windows faults verified; full pilot incomplete |
| Rolling capacity | Small/medium unit measured; full retained/temp/wide demand unknown |
| Runtime estimate | Measured cell profiles/scenarios only; full bound unknown |
| Full run | Blocked; no authorization or dispatch |

## 7. Checks, records and delivery

P12 commands (every new base resolves within the evidence root):

```powershell
D:\Conda\P12\python.exe -B -m pytest -q --basetemp=reports/real_data_preflight_20261003/full_suite_03 --junitxml=reports/real_data_preflight_20261003/full_suite_03.xml
D:\Conda\P12\python.exe -B -m pytest -q tests/test_critical_metrics.py --basetemp=reports/real_data_preflight_20261003/metric_fix_tests --junitxml=reports/real_data_preflight_20261003/metric_fix_tests.xml
git diff --check
```

An opt-in PYTHONPATH/sitecustomize factory counter was used in every full suite;
exact command/environment/fit-reserve declarations are retained locally. Counts
reconcile as follows: historical 80 focused; historical 227 full; initial control 8;
first current full 260 passed/one metadata-order failure; second261 plus seven
subtests; final after two metric regressions **263 plus seven subtests**, zero
failures/skips and 15 warnings in 151.60 s. XML reports 270 including subtests.
Final in-memory compilation passed 86 Python files/four notebook cells without
executing notebooks or writing bytecode.

Durable local evidence root: `reports/real_data_preflight_20261003/` contains
environment/data/eligibility reports; frozen full/pilot manifests/configurations;
budget/fit counter; three invocation receipts; pilot SQLite/JSONL and lifecycle
receipts; all preserved input/diagnostic/history evidence; per-process/resource
traces; `stage_timings.csv`, `per_model_attempts.csv`, `model_timings.csv`,
`measurements.json`; exact probability reproductions; paired/sensitivity/provenance
packages; XML/compilation/preservation results. Operational helper scripts remain
locally inspectable and excluded from Git with the bulky evidence.

Tracked engineering records: this file, [record.md](record.md),
[datasets.md](datasets.md), [coverage.json](coverage.json),
[issue_dispositions.md](issue_dispositions.md), and the separate metric repair.
Only task-owned source/tests and concise engineering records are committed;
raw data, caches, probability matrices, large ledgers and environments are not.
Verified commit/push details are retained locally as `reports/real_data_preflight_20261003/delivery.json` after delivery.
