# Reviewer #1 seven-run command sheet

This v5 sheet is the executable plan for the corrected primary and separate sensitivity tracks. The [dated readiness report](reviewer1_launch_readiness_v5.json) records the fresh gate for all seven IDs on this host; **run the verifier again immediately before each run** because disk, dependencies, source identity, and concurrent work can change. The all-seed audit and both scope manifests are already frozen; do not regenerate them just to launch. Use the existing `p12`; no new environment is created. The two primary runs are the paper's corrected benchmark. The five sensitivity runs have their own result identities and are never pooled into the primary estimand. Run one heavy coordinator at a time.

The scope's `unresolved_gates` and calibration `PENDING` fields describe its freeze-time planning state. Completed bounded evidence and the dated readiness snapshot record their resolution. Preserve the scope bytes; its hash is the verified execution-profile identity.

From `D:\DR2\AutoFE_Submission` in PowerShell:

```powershell
$py = 'D:\Conda\p12\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$common = @(
  '--cache-policy','bounded',
  '--durable-scheduler','--manifest-policy','compact',
  '--scheduler-lease-seconds','600','--resource-policy','adaptive',
  '--reserve-cpus','2','--ram-target-fraction','0.8',
  '--vram-target-fraction','0.8','--gpu-policy','auto',
  '--array-transport','auto',
  '--resource-profile','provenance/reviewer1_launch_scope_v5.json',
  '--pipelines','Raw','AutoFE_Baseline','Raw_CapMatched','AutoFE_MI','AutoFE_Random',
  'AutoFE_NoMultiply','AutoFE_Isolate_Add','AutoFE_Isolate_Subtract',
  'AutoFE_Isolate_Multiply','AutoFE_Isolate_Divide','AutoFE_LeaveOut_Add',
  'AutoFE_LeaveOut_Subtract','AutoFE_LeaveOut_Multiply','AutoFE_LeaveOut_Divide',
  '--models','logistic_regression','random_forest','extra_trees','linear_svm','knn',
  'gaussian_nb','mlp','lightgbm','xgboost','catboost'
)

function Invoke-FrozenRun {
  param([string]$RunId, [string]$Scope, [string]$Policy)
  & $py -m provenance.verify_reviewer1_launch_v2 --scope provenance/reviewer1_launch_scope_v5.json --run-id $RunId
  if ($LASTEXITCODE -ne 0) { throw "Launch gate failed for $RunId" }
  & $py -m src.pipeline_runner @common --run-id $RunId --scope $Scope --split-policy $Policy
  if ($LASTEXITCODE -ne 0) { throw "Runner failed for $RunId; inspect and resume the identical command" }
  $manifestPath = Join-Path (Join-Path 'corrected_runs' $RunId) 'manifest.json'
  $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
  $counts = $manifest.counts_by_status
  if ($manifest.status -ne 'complete' -or $counts.pending -ne 0 -or
      $counts.failed -ne 0 -or $counts.timed_out -ne 0 -or
      ($counts.success + $counts.skipped) -ne $manifest.expected_tasks) {
    throw "Incomplete or failed run: $RunId; inspect $manifestPath"
  }
  & $py -m src.check_progress --run-dir (Join-Path 'corrected_runs' $RunId)
}
```

Invoke the following **one line at a time**, reviewing the preceding run's terminal counts and skip reasons before proceeding. The function stops on a failed gate, nonzero runner exit, or incomplete terminal accounting.

```powershell
Invoke-FrozenRun r1-v5-primary-row primary row_level
Invoke-FrozenRun r1-v5-primary-group primary group_aware
Invoke-FrozenRun r1-v5-domain-row transductive_domain_partition row_level
Invoke-FrozenRun r1-v5-availability-row feature_availability_ablation row_level
Invoke-FrozenRun r1-v5-availability-group feature_availability_ablation group_aware
Invoke-FrozenRun r1-v5-relabel-row majority_label_relabeling row_level
Invoke-FrozenRun r1-v5-relabel-group majority_label_relabeling group_aware
```

**Counts:** primary row 875,000 intended; primary group 875,000 intended with 70,000 prespecified AUC skips; domain row 175,000 intended, with any condition-specific AUC skips recorded at execution; each availability/relabeling policy 87,500 intended, and each group policy has 7,000 prespecified AUC skips. Total: **2,275,000 intended cells**, **84,000 prespecified group-AUC skips**, at most **2,191,000 eligible cells** before domain-condition skips. This count includes all 14 pipelines and ten classifiers in every run. Group-aware transductive conditions are deliberately absent because no group-constrained partition rule is prespecified.

The frozen adaptive policy leaves two logical CPU slots, uses a soft 80% RAM budget, and probes XGBoost/CatBoost on visible NVIDIA GPUs. This host resolves to six workers, 25.39 GiB RAM budget and about 11.92 GiB ready cache plus 11.92 GiB staging. GPU admission reserves 20% VRAM, permits one active fit per device and records comparison-wide static capacity CPU fallback in `auto` mode. These are admission estimates; native memory allocations can exceed estimates. Every model and feature parameter remains in the study, and the actual resolved model parameter catalog documents backend defaults. CPU/GPU metric equivalence is not asserted.

The storage plan reserves 31.13 GiB results/checkpoints/catalog metadata plus 1 GiB mechanism history; the doubled-results margin, both cache allowances and the separate 11.92 GiB array-transport allowance require about **99.03 GiB free**. Recheck the gate before each command. No candidate history is enabled in performance runs. Do not add `--cache-audit` to the compact full grid. Omit fixed worker/cache overrides when using this frozen command. Other hosts need a new host freeze and diagnostic evidence.

## Separate clean-condition mechanism runs

The [frozen mechanism scope](mechanism_history_scope_v4.json) adds two distinct run IDs. It writes full candidate-selection records and sampled training-fold arithmetic Jacobians for `AutoFE_Baseline` under clean conditions. This is **1,250 intended feature tasks**: 625 row-level and 625 group-aware, with 50 group-aware skips for the two all-seed AUC-ineligible datasets. Its measured Sonar extrapolation is 0.085 GiB compressed for 1,200 executable tasks; reserve 1 GiB because candidate counts and filesystem overhead vary. This is separate from the seven performance runs and does not produce classifier scores.

```powershell
& $py -m provenance.run_mechanism_history --run-id r1-v5-mechanism-row --split-policy row_level --scope provenance/reviewer1_launch_scope_v5.json
& $py -m provenance.run_mechanism_history --run-id r1-v5-mechanism-group --split-policy group_aware --scope provenance/reviewer1_launch_scope_v5.json

# After both primary and both mechanism runs are complete:
& $py -m provenance.analyze_mechanism_history `
  --row-results corrected_runs/r1-v5-primary-row/results.jsonl `
  --group-results corrected_runs/r1-v5-primary-group/results.jsonl `
  --row-mechanism corrected_runs/r1-v5-mechanism-row `
  --group-mechanism corrected_runs/r1-v5-mechanism-group `
  --output corrected_runs/analysis/mechanism_association.json
```

The association uses datasets as the unit, requires complete clean-condition paired AUC cells and mechanism folds, checks feature-matrix hashes across run types, and reports separate row/group Spearman associations with permutation and bootstrap uncertainty. It is exploratory and cannot establish causality or condition-specific mechanism effects. The full performance runs continue to carry operator candidate counts by cell.

## Corrected paper tables and figures

After both complete primary runs, generate the streaming paper assets. Once the separate mechanism association above exists, add it to the same export. No command accepts a running/partial primary manifest. Terminal failures produce visibly incomplete complete-case/bounded diagnostics with scientific_complete=false; the campaign helper still stops for inspection instead of proceeding past failure. They do not overwrite historical `reports/` or `paper/figures/` material.

```powershell
& $py -m provenance.generate_corrected_assets
& $py -m provenance.generate_corrected_assets --mechanism-association corrected_runs/analysis/mechanism_association.json
& $py -m provenance.generate_corrected_assets --sensitivity-root corrected_runs --mechanism-association corrected_runs/analysis/mechanism_association.json
```

The outputs are under `corrected_runs/paper_assets/`: CSV tables for dataset AUC coverage/reasons, all 14 pipeline AUC and resource summaries, condition-specific AUC, and per-operator generated/rejected/duplicate/eligible/selected candidate counts. Fifteen primary CSVs and eight PNG/PDF pairs include prespecified dataset ROC-AUC contrasts with bootstrap/Holm, descriptive matched-cap/operator contrasts, common-set row/group differences, eligible-primary missing-outcome bounds, selected candidate rates, candidate stages, dataset coverage, and the adaptive resource policy, observed use, array transport counts and worker phase timings. F1 in table names is the statistical-family identifier, distinct from classifier F1. A machine-checked corrected_results_note.md is written alongside the asset manifest. Supplying the mechanism JSON adds two CSVs and one PNG/PDF scatter pair for Jacobian exposure and clean AUC gain. Supplying the sensitivity root after all five separate runs complete adds four sensitivity CSVs and one PNG/PDF pair, preserving each run/condition/policy identity. `asset_manifest.json` records source hashes and denominators. Back up this directory with the ledgers.

To monitor a run without loading its whole JSONL ledger:

```powershell
& $py -m src.check_progress --run-dir corrected_runs/r1-v5-primary-row
```

After a safe stop or process crash, rerun the **identical** command with the same run ID. The scheduler and result index reconcile committed cells. A changed code, package, dataset, scope, worker count, or other configuration requires a **new** run ID. The outputs for each run remain in `corrected_runs/<run-id>/`; they are ignored by Git and must be backed up as research data.

## Execution reuse and temporary transport

Training-only preprocessing is reused across the 14 pipeline variants for identical dataset/split/seed/fold/condition inputs. Large numeric inputs are published once per feature group as NumPy files; workers open private copy-on-write mappings, close them before completion, and terminal groups are reclaimed. The auto threshold is derived from the host budget, worker ceiling and classifier count. `--no-preprocessing-reuse` and `--array-transport pickle` are diagnostic overrides and require new run identities; they are not in the frozen campaign commands. Temporary mappings are regenerable and must not be confused with durable results/checkpoints. The v5 disk gate reserves an additional full mapping budget alongside live cache and cache publication space. All seven scientific task counts are unchanged.
