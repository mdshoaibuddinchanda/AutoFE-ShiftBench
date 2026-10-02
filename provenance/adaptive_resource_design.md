> Current execution follow-up: [runtime_work_reuse_design.md](runtime_work_reuse_design.md) and [v5 readiness](reviewer1_launch_readiness_v5.json) supersede the v4 launch identity/storage/evidence described below. CPU/RAM/VRAM rules remain; v5 adds a separate transport disk allowance and verifies mapped CPU/GPU recovery. The v4 records below are retained as engineering history.

# Adaptive host resource policy

Implemented for the 2026-10-01 request to replace fixed execution limits with machine-aware CPU/RAM/VRAM admission. The corrected campaign has not started. Use the existing Conda p12; no packages or environments were added.

## Host and defaults

| Resource | Measured capacity | Execution policy |
|---|---:|---|
| CPU | 8 logical slots, 4 physical cores exposed | Ceiling 6 model workers; reserve 2 logical slots. Workers share the six-slot affinity mask. Numerical libraries use one thread per fit. |
| RAM | 31.73 GiB | Soft budget 25.39 GiB, reserve 6.35 GiB. Actual system availability and estimated/observed worker RSS control admission. |
| GPU | RTX 3050 Laptop, 4 GiB VRAM | XGBoost/CatBoost passed actual GPU fits. Admission budget 3.2 GiB, reserve 0.8 GiB; one active fit per visible, verified device. |
| Disk feature cache | 476.92 GiB volume capacity | Bounded ready cache derives from min(40% total RAM, 2.5% volume capacity): about 11.92 GiB. Separate staging allowance is the same size. Retain mode remains uncapped. |

CPU reserve counts logical slots, not physical cores. Hosts with fewer slots fall back to one worker and mark that the full requested reserve cannot be satisfied. Other applications can use spare CPU time. Small useful workloads will consume less than 20 GiB RAM; no dummy allocation is used to hit a minimum. These memory limits are soft admission estimates, not operating-system/native allocator enforcement. Running native fits are not killed when another application consumes RAM.

## Scientific comparison and GPU capacity

All datasets, split policies, seeds, folds, conditions, 14 pipelines, ten classifiers, hyperparameters, candidate selection/stopping rules and scientific 20-parent/100-output caps remain in the study. Resource controls change scheduling and supported model backends.

The backend capacity bound is shared across all configured pipelines for a dataset/split/seed/fold/condition/model comparison. It uses training feature cardinalities, the largest configured output width, total train/test row count, float64 matrices, label vectors and a fixed small overhead. GPU admission estimates 256 MiB plus twice the shared float64 host matrix bound (four float32 working-matrix equivalents). This heuristic is verified on bounded datasets; it is not an exact prediction of library VRAM allocation. In auto mode, an estimate above every verified device budget selects CPU for the entire comparison and records the reason. Temporary live VRAM pressure pauses admission without changing backend. Require mode reports unsupported/oversized GPU work as an error. Actual backend and shared bound are recorded in each success; the exporter refuses mixed backends within operator comparisons.

Visible devices respect CUDA_VISIBLE_DEVICES and map physical IDs to logical ordinals. Each model/device pair is probed. CatBoost receives an explicit device and a live gpu_ram_part derived from headroom. XGBoost fitted configuration is checked to reject silent CPU substitution. LightGBM and the remaining current classifiers retain their supported CPU implementation. GPU training is a backend change; CPU/GPU metrics and GPU repeatability are not asserted identical. Resolved backend defaults are recorded rather than hidden. See [CatBoost GPU training](https://catboost.ai/docs/en/features/training-on-gpu), [CatBoost resource parameters](https://catboost.ai/docs/en/references/training-parameters/performance), and [XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/).

## Identity, pressure and recovery

The frozen v4 commands pass --resource-profile provenance/reviewer1_launch_scope_v4.json. Source, hardware, driver, visibility, policy settings and optional overrides must match; the profile hash enters configuration/task/cache identities. New frozen runs reuse the verified capability plan. Resume reuses the initial plan and rejects changed host/settings/profile/source/runtime. Fluctuating free RAM/VRAM stays in telemetry, outside immutable identities.

Admission drains active work and renews scheduler leases while waiting. Learned sampled worker RSS improves later estimates. Prepared feature groups and bounded queued fits limit serialized matrix memory. Each verified GPU uses one persistent device worker, avoiding repeated CUDA contexts across CPU workers; active CPU and GPU fits together obey the worker ceiling. A worker exits when its coordinator disappears, preventing orphan pools from holding RAM after a crash. Telemetry survives resume. Sustained external RAM/VRAM pressure can leave admission waiting until resources become available.

Each result records actual backend/reason/device, CatBoost allocation fraction, shared comparison bytes and worker RSS sampled every 100 ms. model_parameters/<hash>.json stores deduplicated resolved classifier parameters; result rows link its path/fingerprint. Export validates catalog hashes, distinguishes missing historical resource metadata from zero, and adds primary_resource_policy_and_use.csv plus primary_resource_budget_and_use.png/.pdf. Primary reporting now produces 15 CSVs and eight PNG/PDF pairs, with optional separate mechanism/sensitivity outputs.

## Freeze, scope and verification

The v3 scope and green reports remain historical fixed-policy evidence. The preliminary v4 profiles and interrupted -001/-002 diagnostics are retained under reviewer1_launch_scope_v4_preliminary*.json and corrected_runs/adaptive_verification/; they do not authorize launch. The final v4 freeze binds the comparison-wide capacity policy and resource profile.

The study remains seven performance runs: 2,275,000 intended cells, 84,000 planned group-AUC skips, at most 2,191,000 eligible before condition-specific skips. Group all-seed AUC covers at most 23 datasets; wine-quality-red/kddcup99 keep explicit skip reasons. Separate mechanism work adds 1,250 feature tasks / 50 skips.

Storage planning uses 31.13 GiB results/checkpoints/catalogs (historical 27.13 plus 1 GiB timing and 3 GiB adaptive metadata allowances), twice that projection for margin, two 11.92 GiB cache allowances and 1 GiB mechanism history: about 87.11 GiB free. This is an estimate, not an output-size guarantee. The older 229.69-day runtime scenario remains historical; bounded v4 timings are not a full-campaign ETA.

Verification: focused host/pressure/GPU-slot/capacity/backend/profile/resume tests, known-number table/figure tests and full p12 suite; bounded four-dataset all-14/all-ten CPU parity, both-policy Covertype Raw checks, and both-policy Airlines GPU forced restarts. The Airlines checks use Raw/AutoFE_Baseline and XGBoost/CatBoost (four cells per policy), directly exercising both supported GPU paths without repeating the unchanged, costly CPU estimators. Prior v3 twenty-cell Airlines restart proofs remain historical; the current bounded CPU matrix covers all classifiers. Exact CPU fields are compared; GPU comparisons require exact feature/candidate identities and record backend differences. Full corrected scientific effects remain PENDING CORRECTED RUN.

Status: **READY IN THE DATED V4 SNAPSHOT; FULL CAMPAIGN NOT STARTED.** All seven gates passed; re-run the verifier immediately before each long command.


### Bounded admission correction

The first comparison-wide guard used four times the float64 host bound. With the desktop's existing VRAM use, it paused the 605-column Airlines Raw matrix before fitting, although the actual GPU workload fits. This wait is preserved in the preliminary2 scope and interrupted -002 diagnostics. The corrected float32 working estimate and persistent device worker completed four additional Airlines GPU cells under the default 20% headroom. The final frozen verification completed all six bounded runs on current source. A heuristic is not an allocation guarantee; actual GPU failures remain visible and are not silently changed to CPU.


## Final bounded evidence

Source commit: `bf64794b976424677323518a6f0d50e91f444fbd`. The full p12 suite passed **160 tests, 27 warnings, 109.18 seconds**. The saved adaptive report is independently revalidated by every launch gate.

| Current-source diagnostic | Cells | Elapsed seconds, including forced restart when used | Compared evidence |
|---|---:|---:|---|
| Four small datasets, row-level | 560 | 126.609 | All 24 CPU comparison fields exactly equal |
| Four small datasets, group-aware | 560 | 144.578 | All 24 CPU comparison fields exactly equal |
| Covertype Raw, row-level | 10 | 157.703 | 8 CPU exact; 2 GPU feature/candidate identities exact |
| Covertype Raw, group-aware | 10 | 223.531 | 8 CPU exact; 2 GPU feature/candidate identities exact |
| Airlines Raw/Baseline, row-level | 4 | 104.031 | GPU feature/candidate identities exact; committed cell retained without refit after restart |
| Airlines Raw/Baseline, group-aware | 4 | 213.093 | GPU feature/candidate identities exact; committed cell retained without refit after restart |

Total: **1,148 successes, 1,136 exact CPU comparisons and 12 GPU feature/candidate comparisons**. GPU metrics are deliberately excluded from CPU-equivalence assertions. Four more GPU admission smoke cells completed in a single persistent device process. Four mechanism smoke tasks match the runner's matrices exactly. These bounded timings are not a speed guarantee or full-campaign ETA.

Real CPU-schema diagnostic export produced 15 CSVs and eight PNG/PDF pairs under `corrected_runs/adaptive_reporting_smoke/assets/`. A targeted GPU resource export under `gpu_resource_assets/` shows four actual GPU successes per policy, sampled RAM, device budgets, three slot waits per policy and resolved parameter records. The full paper exporter correctly refuses the two-pipeline GPU subset as a full paper scope; only its validated collection/resource-plot functions are used for that additional metadata check. Both asset manifests explicitly mark `scientific_complete=false`. The resource diagnostic banner has tight title spacing; values remain legible. Full corrected figures have no pilot banner.

The final [readiness snapshot](reviewer1_launch_readiness_v4.json) records all seven passing gates, matching 21 exact requirement pins, data/source/analysis/profile identities, the disk margin and diagnostic artifact hashes. Independent read-only review is recorded in `adaptive_resource_agent_review_v1.md`. Full corrected effects remain **PENDING CORRECTED RUN**.

Dependency constraint supplement: the launch verifier checks the 21 exact pins. An additional read-only metadata check confirmed installed setuptools 80.10.2 satisfies the remaining setuptools<81 requirement. Existing p12 was not modified.
