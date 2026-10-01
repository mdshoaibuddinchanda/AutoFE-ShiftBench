# Corrected results note template

Status: **PENDING CORRECTED RUN**. The historical ledger/manuscript is separate. Bounded pilots, optimizer parity, synthetic tables/figures and forced-restart diagnostics verify code paths and supply resource evidence; none is a corrected 25-dataset result.

The current seven-run v3 scope contains2,275,000 intended performance cells and84,000 prespecified group-AUC skips, leaving at most 2,191,000 eligible before condition-specific domain skips. Its primary component is1,750,000 intended cells with70,000 group skips. Primary complete-seed group AUC covers at most 23 datasets; wine-quality-red/kddcup99 remain visible without substitute folds. Separate clean mechanism scope:1,250 feature tasks,50 skips,1,200 executable histories.

## Machine-checked record after the run

`python -m provenance.generate_corrected_assets` writes the actual record to `corrected_runs/paper_assets/corrected_results_note.md`, alongside14 primary CSVs,7PNG/PDF figure pairs and `asset_manifest.json`. It includes source/scope/data/run/config/runtime identities and hashes, expected/success/skipped/failed/timed-out counts, finite and complete-dataset denominators, prespecified dataset ROC-AUC contrasts with uncertainty/Holm, per-dataset and common-set row/group comparisons, missing-outcome identification bounds, exact table/figure paths, and limitations. Optional mechanism/sensitivity exports add separate evidence. The statistical family named F1 is not the classifier F1 metric.

Running/partial ledgers are rejected. Terminal failures receive explicit unknown-outcome labels and `scientific_complete=false`; bounds do not impute AUC, assign undefined structural group AUC, or permit a claim that the full campaign succeeded. History and manuscript numeric revision remain pending until suitable completed evidence exists.

## Required final claims

- All reported values come from corrected ledgers with frozen identities, not historical560,002records or the manuscript538,972subset.
- Dataset is the independent unit. Row/group folds and predictions are never treated as paired held-out samples.
- Candidate generated/rejected/duplicate/eligible/selected counts use unique feature tasks, not ten classifier copies.
- Measured feature-generation/preprocessing/preparation costs are shared across classifiers; sum at unique feature-task level when reporting total preparation cost. Missing old timing fields remain null.
- Mechanism evidence is clean-condition exploratory association, not causal proof.
- Storage uses bounded/regenerable cache and separate limited mechanism history. Historical79.66GiB history and47,378 GiB retained-cache projections are superseded planning evidence.
