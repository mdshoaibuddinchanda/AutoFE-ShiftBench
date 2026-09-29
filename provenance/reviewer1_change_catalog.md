# Reviewer #1 change catalog

Status legend: `open` = not yet implemented; `implemented-not-run` = code exists but the required evidence is pending; `verified` = the stated checks passed; `blocked` = a documented protocol or resource blocker remains. This catalog is an engineering/evidence ledger, not the author response.

## R1. FSVA validation

- Exact concern: Jacobian norms and candidate-level selection histories were not directly measured.
- Original problem / affected claim: FSVA was presented as a mechanism without direct derivative, candidate-history, or dataset-level association evidence.
- Relevant files: `src/feature_engineering.py`, `src/pipeline_runner.py`, `src/mechanism_audit.py` (planned), `tests/test_mechanism_audit.py` (planned).
- Decision and rationale: define a feature-map Jacobian only for supported arithmetic primitives; record undefined/non-finite cases; treat associations as evidence consistent with a mechanism, not proof of causality.
- Implementation: pending mechanism agent integration.
- Tests and commands: pending targeted derivative/history tests and bounded preflight.
- Commit hash: pending.
- Benchmark artifact IDs / observed values: `PENDING CORRECTED RUN`.
- Limitations: Featuretools internals may not expose every generated expression or derivative boundary.
- Status: `open`.

## R2. Statistical analysis

- Exact concern: pooled row/task-level tests treated folds, seeds, models, or conditions as independent datasets.
- Original problem / affected claim: uncertainty and significance could be overstated.
- Relevant files: `src/stats_analysis.py`, `src/generate_tables.py`, `provenance/corrected_results_note.md`.
- Decision and rationale: dataset is the primary independent unit; prespecified paired contrasts, intervals, exact denominators, and Holm correction are required. Winner selection on shared folds remains exploratory unless nested selection is added.
- Implementation: existing dataset-level Wilcoxon path is baseline evidence; expanded result-note and interval checks are pending.
- Tests and commands: baseline suite passed 19 tests; post-integration checks pending.
- Commit hash: `a6a8e3b` baseline provenance milestone.
- Benchmark artifact IDs / observed values: historical values remain separate; corrected values are `PENDING CORRECTED RUN`.
- Limitations: No corrected benchmark estimates exist yet.
- Status: `implemented-not-run`.

## R3. Leakage and duplicate-overlap evaluation

- Exact concern: target/held-out information paths and exact duplicate feature vectors crossing folds were not fully addressed.
- Original problem / affected claim: row-level folds can put identical raw feature vectors in train and test, allowing memorization; target-free partition boundaries must remain explicit.
- Relevant files: `src/splitters.py`, `src/group_splits.py` (planned), `provenance/dataset_schema_audit.{json,md}`, `tests/test_group_splits.py` (planned).
- Decision and rationale: retain row-level folds as legacy-comparable sensitivity; add group-aware folds using deterministic canonical target-excluded raw predictors and assert zero shared groups. Never silently fall back or discard infeasible datasets.
- Implementation: group-aware module and feasibility audit pending agent integration.
- Tests and commands: current target-free negative controls pass; group tests pending.
- Commit hash: pending integration.
- Benchmark artifact IDs / observed values: schema audit verified 25/25 CSV/sidecars; row-level overlap examples include PhishingWebsites 65.42% and KDDCup99 67.01%; group-track values are `PENDING CORRECTED RUN`.
- Limitations: Grouping removes duplicate-vector overlap, not semantic target proxies or deployment shift.
- Status: `implemented-not-run`.

## R4. Terminology and condition handling

- Exact concern: conditions were not always distinguished as training corruption, transductive partitioning, or deployment shift.
- Decision and rationale: preserve condition names, add explicit scope metadata, and reserve deployment-shift language for a protocol that changes held-out deployment distribution.
- Relevant files: `src/pipeline_runner.py`, `src/shift_generator.py`, `README.md`, `provenance/reviewer1_run_plan.md`.
- Implementation: primary/transductive/availability/relabeling scopes already fail closed when mixed; manuscript phrase checklist remains pending.
- Tests and commands: `tests/test_leakage_controls.py` condition semantics test passed in baseline suite.
- Commit hash: `a6a8e3b` baseline provenance milestone.
- Benchmark artifact IDs / observed values: none; corrected values `PENDING CORRECTED RUN`.
- Limitations: Manuscript source still contains historical OpenML and shift-language statements.
- Status: `implemented-not-run`.

## R5. Baseline fairness

- Exact concern: Raw and AutoFE dimensionality/selection caps were not matched.
- Decision and rationale: retain the original Raw baseline, add a cap-matched Raw comparator and full-dimensional Raw comparator, and report pre/post-synthesis caps, selected counts, and cost.
- Relevant files: `src/pipeline_runner.py`, `src/feature_engineering.py`, result schema (planned).
- Implementation: pending protocol integration.
- Tests and commands: pending configuration and smoke checks.
- Commit hash: pending.
- Benchmark artifact IDs / observed values: `PENDING CORRECTED RUN`.
- Limitations: Matched-cap results cannot be inferred from historical results.
- Status: `open`.

## R6. Individual operator effects

- Exact concern: `AutoFE_NoMultiply` excludes multiplication and division, and does not isolate individual operator effects.
- Decision and rationale: preserve its historical meaning; add isolate-one-operator and justified leave-one-out configurations with shared budgets and finite-value rules.
- Relevant files: `src/feature_engineering.py`, `src/pipeline_runner.py`, `src/shift_generator.py` (planned integration).
- Implementation: pending operator-agent evidence and lead integration.
- Tests and commands: pending.
- Commit hash: pending.
- Benchmark artifact IDs / observed values: `PENDING CORRECTED RUN`.
- Limitations: Division-by-zero behavior and Featuretools expression naming require explicit validation.
- Status: `open`.

## R7. Incomplete runs and stopping

- Exact concern: missing tasks and stopped blocks could be mistaken for zeroes or complete evidence.
- Decision and rationale: every task must have success, failed, skipped-with-reason, timed-out, or pending status with resumable identity and coverage by all design dimensions.
- Relevant files: `src/checkpoint.py`, `src/pipeline_runner.py`, `src/check_progress.py`, `provenance/corrected_results_note.md`.
- Implementation: phase-level success/failure accounting exists; skipped/timeout taxonomy and complete coverage summaries are pending.
- Tests and commands: baseline failure/resume tests passed; post-integration checks pending.
- Commit hash: `a6a8e3b` baseline provenance milestone.
- Benchmark artifact IDs / observed values: no corrected run; historical ledger remains separately labeled.
- Limitations: unknown outcomes cannot support a causal or confirmatory claim.
- Status: `implemented-not-run`.

## R8. Reproducibility

- Exact concern: stable seeds, source identity, cache/run identity, and final artifacts must be independently reproducible.
- Decision and rationale: use canonical SHA-256 identities, source/sidecar checksums, runtime/package fingerprints, immutable task manifests, and explicit resume commands.
- Relevant files: `src/provenance.py`, `src/data_loader.py`, `src/pipeline_runner.py`, `requirements.txt`, `provenance/dataset_schema_audit.*`.
- Implementation: baseline reproducibility controls and UCI Dry Bean correction are committed in `a6a8e3b`; final corrected-run freeze is pending.
- Tests and commands: 19 tests passed, compileall passed, 25/25 schema/hash checks passed; full post-agent gate pending.
- Commit hash: `a6a8e3b` baseline provenance milestone.
- Benchmark artifact IDs / observed values: source fingerprint `d46c872632bcfe45a8e15b415f0ac4cc2ebbf403f25610a6f277fc057d77cbab` before new modules; corrected run `PENDING CORRECTED RUN`.
- Limitations: historical Dry Bean input is unavailable and historical scores cannot be reconstructed.
- Status: `implemented-not-run`.

## R9. Confirmatory analysis now

- Exact concern: major fixes must be made and protocol frozen before the expensive run.
- Decision and rationale: do not launch until both split policies, common core, ablations, resource estimates, and result schemas are frozen and preflighted.
- Relevant files: `provenance/reviewer1_run_plan.md`, `provenance/corrected_results_note.md`, frozen manifest (planned).
- Implementation: plan exists; group-aware, mechanism, operator, baseline, and runtime gates remain in progress.
- Tests and commands: baseline suite/compile/schema audit passed; remaining gates pending.
- Commit hash: `a6a8e3b` plan and baseline milestone.
- Benchmark artifact IDs / observed values: none; no long run is running.
- Limitations: corrected numerical results are unavailable.
- Status: `open`.

## Dated decision log

- 2026-09-29: Existing branch/worktree inspected; all prior edits found uncommitted. No historical result file was rewritten.
- 2026-09-29: Baseline suite rerun in existing `p12`: 19 passed, 2 warnings in 27.12s; compileall passed.
- 2026-09-29: 25 datasets verified; Dry Bean source corrected to UCI ID 602. Commit `a6a8e3b` records baseline provenance and plan.
- 2026-09-29: Reviewer #1 run plan frozen provisionally; group-aware feasibility, mechanism measurement, operator isolation, and final resource gate remain open.
