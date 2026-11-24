# AutoFE-ShiftBench Public Evidence Report

This directory contains a compact, GitHub-safe evidence package for the corrected AutoFE-ShiftBench analysis. It contains summaries and audit outputs, not the raw task ledger, databases, caches, model artifacts or the large candidate-history collection.

## Canonical experiment

The canonical post-processing audit is `reports/revision_final/run_20261123T175648Z/`. The public files were copied from that run and from the corresponding measured FSVA summaries. Historical seven-pipeline results remain separate from the corrected multi-pipeline evidence.

The declared corrected configuration contains 25 datasets, five root seeds, five folds, 14 corruption/stress conditions, 14 pipelines and ten estimators. Coverage files distinguish configured, eligible, observed and missing tasks; they do not turn incomplete configurations into completed results.

## Contents

- `experimental_coverage/`: eligible-task coverage, dataset/seed coverage and missingness/special-dataset sensitivity.
- `statistics/`: dataset-level paired contrasts, Holm adjustments, statistical methods and sensitivity results.
- `fsva/`: measured Jacobian summaries, numerical verification, operator stratification, missingness and descriptive Jacobian-performance association.
- `candidate_selection/`: candidate operator counts, selection summaries, stability and history-schema validation.
- `reproducibility/`: verified dataset-source notes and compact FSVA summary artifacts.
- `manifest/`: checksums and source-to-public artifact provenance.

## Interpretation limits

The statistical analysis is dataset-level and uses paired contrasts with multiplicity adjustment as described in `statistics/statistical_methods.md`. Transductive covariate and population partitions are stress tests, not deployment-time distribution-shift estimates.

Jacobian-performance results are descriptive matched associations, not causal evidence that sensitivity degrades generalization. The complete covariance-weighted FSVA index was not measured. Candidate histories are summarized from the available history artifacts rather than the full history corpus.

Covertype does not provide complete five-seed evidence. Affected KDD Cup 99 ROC-AUC records are reported as unavailable/invalid where applicable; no values were imputed.

## Omitted large artifacts

The full ledgers, SQLite manifests, caches, model inputs, model outputs and 19.61 GB history collection remain local under `reports/` or require an external archive. They are intentionally excluded from this public package.

## Reproduction and provenance

Figure regeneration is handled by `reproduction/figures.py`; audit and summary utilities are in `reproduction/`. The canonical source evidence paths and SHA-256 values for every public artifact are recorded in `manifest/public_artifact_manifest.csv`, `manifest/public_artifact_manifest.json` and `manifest/source_artifact_checksums.sha256`.

Public files are sanitized derivatives where necessary. Historical originals were not rewritten.

Reviewer correspondence, manuscript text, submission metadata and reviewer-response
derivatives are intentionally excluded from this public package.
