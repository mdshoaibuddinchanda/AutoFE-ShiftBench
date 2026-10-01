# Reviewer #1 seven-run command sheet

This sheet is the executable plan for the corrected primary and separate sensitivity tracks. **It does not authorize a launch yet.** Run the all-seed audit, freeze `reviewer1_launch_scope_v2.json`, inspect the intended host calibration and disk margin, then run one policy at a time. Use the existing `p12`; no new environment is created. The two primary runs are the paper's corrected benchmark. The five sensitivity runs have their own result identities and are never pooled into the primary estimand.

From `D:\DR2\AutoFE_Submission` in PowerShell:

```powershell
$py = 'D:\Conda\p12\python.exe'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
& $py -m provenance.audit_group_seed_grid
& $py -m provenance.freeze_reviewer1_launch_v2
& $py -m provenance.freeze_mechanism_scope_v1
# One-time gate, only after clean group calibration -003 is complete:
# & $py -m provenance.large_dataset_recovery_v1 --version 002

$common = @(
  '--cache-policy','bounded','--cache-max-gib','8',
  '--durable-scheduler','--manifest-policy','compact',
  '--scheduler-lease-seconds','600','--workers','4',
  '--pipelines','Raw','AutoFE_Baseline','Raw_CapMatched','AutoFE_MI','AutoFE_Random',
  'AutoFE_NoMultiply','AutoFE_Isolate_Add','AutoFE_Isolate_Subtract',
  'AutoFE_Isolate_Multiply','AutoFE_Isolate_Divide','AutoFE_LeaveOut_Add',
  'AutoFE_LeaveOut_Subtract','AutoFE_LeaveOut_Multiply','AutoFE_LeaveOut_Divide',
  '--models','logistic_regression','random_forest','extra_trees','linear_svm','knn',
  'gaussian_nb','mlp','lightgbm','xgboost','catboost'
)

# Run the next line only after auditing the preceding run's terminal counts.
# First verify its exact ID: & $py -m provenance.verify_reviewer1_launch_v2 --run-id <ID>
# Stop if verification exits nonzero.
& $py -m src.pipeline_runner @common --run-id r1-v2-primary-row --scope primary --split-policy row_level
& $py -m src.pipeline_runner @common --run-id r1-v2-primary-group --scope primary --split-policy group_aware
& $py -m src.pipeline_runner @common --run-id r1-v2-domain-row --scope transductive_domain_partition --split-policy row_level
& $py -m src.pipeline_runner @common --run-id r1-v2-availability-row --scope feature_availability_ablation --split-policy row_level
& $py -m src.pipeline_runner @common --run-id r1-v2-availability-group --scope feature_availability_ablation --split-policy group_aware
& $py -m src.pipeline_runner @common --run-id r1-v2-relabel-row --scope majority_label_relabeling --split-policy row_level
& $py -m src.pipeline_runner @common --run-id r1-v2-relabel-group --scope majority_label_relabeling --split-policy group_aware
```

**Counts:** primary row 875,000 intended; primary group 875,000 intended with 70,000 prespecified AUC skips; domain row 175,000 intended, with any condition-specific AUC skips recorded at execution; each availability/relabeling policy 87,500 intended, and each group policy has 7,000 prespecified AUC skips. Total: **2,275,000 intended cells**, **84,000 prespecified group-AUC skips**, at most **2,191,000 eligible cells** before domain-condition skips. This count includes all 14 pipelines and ten classifiers in every run. Group-aware transductive conditions are deliberately absent because no group-constrained partition rule is prespecified.

`--workers 4`, the 8 GiB bounded cache cap, CPU mode, and 600-second lease are the measured local pilot settings, not a universal configuration for another host. Any different setting changes run identity and requires a new freeze and run IDs. Do not enable `--cache-audit` for the compact full grid; its per-feature lists are intended for bounded verification. The full runner does not write candidate histories; the separate frozen mechanism runs below provide those records.

## Separate clean-condition mechanism runs

The [frozen mechanism scope](mechanism_history_scope_v1.json) adds two distinct run IDs. It writes full candidate-selection records and sampled training-fold arithmetic Jacobians for `AutoFE_Baseline` under clean conditions. This is **1,250 intended feature tasks**: 625 row-level and 625 group-aware, with 50 group-aware skips for the two all-seed AUC-ineligible datasets. Its measured Sonar extrapolation is 0.085 GiB compressed for 1,200 executable tasks; reserve 1 GiB because candidate counts and filesystem overhead vary. This is separate from the seven performance runs and does not produce classifier scores.

```powershell
& $py -m provenance.run_mechanism_history --run-id r1-v2-mechanism-row --split-policy row_level
& $py -m provenance.run_mechanism_history --run-id r1-v2-mechanism-group --split-policy group_aware

# After both primary and both mechanism runs are complete:
& $py -m provenance.analyze_mechanism_history `
  --row-results corrected_runs/r1-v2-primary-row/results.jsonl `
  --group-results corrected_runs/r1-v2-primary-group/results.jsonl `
  --row-mechanism corrected_runs/r1-v2-mechanism-row `
  --group-mechanism corrected_runs/r1-v2-mechanism-group `
  --output corrected_runs/analysis/mechanism_association.json
```

The association uses datasets as the unit, requires complete clean-condition paired AUC cells and mechanism folds, checks feature-matrix hashes across run types, and reports separate row/group Spearman associations with permutation and bootstrap uncertainty. It is exploratory and cannot establish causality or condition-specific mechanism effects. The full performance runs continue to carry operator candidate counts by cell.

## Corrected paper tables and figures

After both complete primary runs, generate the streaming paper assets. Once the separate mechanism association above exists, add it to the same export. Neither command accepts a running primary manifest. They do not overwrite historical `reports/` or `paper/figures/` material.

```powershell
& $py -m provenance.generate_corrected_assets
& $py -m provenance.generate_corrected_assets --mechanism-association corrected_runs/analysis/mechanism_association.json
& $py -m provenance.generate_corrected_assets --sensitivity-root corrected_runs --mechanism-association corrected_runs/analysis/mechanism_association.json
```

The outputs are under `corrected_runs/paper_assets/`: CSV tables for dataset AUC coverage/reasons, all 14 pipeline AUC and resource summaries, condition-specific AUC, and per-operator generated/rejected/duplicate/eligible/selected candidate counts. Four PNG/PDF pairs show paired AUC differences, selected candidate rates, full-arithmetic candidate stages, and dataset coverage. Supplying the mechanism JSON adds two CSVs and one PNG/PDF scatter pair for Jacobian exposure and clean AUC gain. Supplying the sensitivity root after all five separate runs complete adds four sensitivity CSVs and one PNG/PDF pair, preserving each run/condition/policy identity. `asset_manifest.json` records source hashes and denominators. Back up this directory with the ledgers.

To monitor a run without loading its whole JSONL ledger:

```powershell
& $py -m src.check_progress --run-dir corrected_runs/r1-v2-primary-row
```

After a safe stop or process crash, rerun the **identical** command with the same run ID. The scheduler and result index reconcile committed cells. A changed code, package, dataset, scope, worker count, or other configuration requires a **new** run ID. The outputs for each run remain in `corrected_runs/<run-id>/`; they are ignored by Git and must be backed up as research data.
