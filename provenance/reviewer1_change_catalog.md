# Reviewer #1 change catalog

Status legend: `open` = not yet implemented; `implemented-not-run` = code exists but the required evidence is pending; `verified` = the stated checks passed; `blocked` = a documented protocol or resource blocker remains. This catalog is an engineering/evidence ledger, not the author response.

## R1. FSVA validation

- Exact concern: Jacobian norms and candidate-level selection histories were not directly measured.
- Original problem / affected claim: FSVA was presented as a mechanism without direct derivative, candidate-history, or dataset-level association evidence.
- Relevant files: `src/feature_engineering.py`, `src/pipeline_runner.py`, `src/mechanism_audit.py`, `tests/test_mechanism_audit.py`.
- Decision and rationale: define a feature-map Jacobian only for supported arithmetic primitives; record undefined/non-finite cases; treat associations as evidence consistent with a mechanism, not proof of causality.
- Implementation: arithmetic Jacobian/finite-difference checks, scaled local norms, candidate-history schema, storage estimator, and optional Featuretools candidate logging are implemented. The runner does not enable full-grid history by default because the bounded estimate is large.
- Tests and commands: `D:\Conda\p12\python.exe -m pytest -q tests/test_mechanism_audit.py` -> 14 passed; bounded real-data preflight wrote 950 training-only candidate records with all four operators and finite scores.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: preflight `provenance/reviewer1_preflight_manifest.json`; full corrected values remain `PENDING CORRECTED RUN`.
- Limitations: Featuretools internals may not expose every generated expression or derivative boundary.
- Status: `implemented-not-run`.

## R2. Statistical analysis

- Exact concern: pooled row/task-level tests treated folds, seeds, models, or conditions as independent datasets.
- Original problem / affected claim: uncertainty and significance could be overstated.
- Relevant files: `src/stats_analysis.py`, `src/generate_tables.py`, `provenance/corrected_results_note.md`.
- Decision and rationale: dataset is the primary independent unit; prespecified paired contrasts, intervals, exact denominators, and Holm correction are required. Winner selection on shared folds remains exploratory unless nested selection is added.
- Implementation: `src/reviewer1_analysis.py` now emits expected-cell coverage, finite-metric denominators, paired dataset bootstrap intervals (10,000 replicates, seed 20260929), Holm family metadata, and row/group side-by-side schema. No corrected campaign ledger has been analyzed.
- Tests and commands: focused post-integration suite passed 42 tests; full suite gate remains to be rerun after final documentation commit.
- Commit hash: `d4a897a` integration milestone; integrated source fingerprint `7f1a85210f1c3dd43c59381cad891e291242d874886f283b1e2b95d02abc84a7` is recorded in the schema audit.
- Benchmark artifact IDs / observed values: historical values remain separate; corrected values are `PENDING CORRECTED RUN`.
- Limitations: No corrected benchmark estimates exist yet.
- Status: `implemented-not-run`.

## R3. Leakage and duplicate-overlap evaluation

- Exact concern: target/held-out information paths and exact duplicate feature vectors crossing folds were not fully addressed.
- Original problem / affected claim: row-level folds can put identical raw feature vectors in train and test, allowing memorization; target-free partition boundaries must remain explicit.
- Relevant files: `src/splitters.py`, `src/group_splits.py`, `provenance/dataset_schema_audit.{json,md}`, `provenance/group_fold_audit.{json,md}`, `tests/test_group_splits.py`.
- Decision and rationale: retain row-level folds as legacy-comparable sensitivity; add group-aware folds using deterministic canonical target-excluded raw predictors and assert zero shared groups. Never silently fall back or discard infeasible datasets.
- Implementation: deterministic typed canonicalization, SHA-256 group IDs with collision checks, `StratifiedGroupKFold`, zero-shared-group assertions, runner split-policy identity, and explicit infeasibility manifests are implemented.
- Tests and commands: `python -m provenance.audit_group_folds` -> 25/25 structural group splits, 23/25 all-fold class/AUC support; `wine-quality-red` and `kddcup99` are explicitly infeasible for all-fold AUC; targeted group tests pass 7 tests. Row-level overlap examples include PhishingWebsites 65.42% and KDDCup99 67.01%.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: `provenance/group_fold_audit.json` and `.md`; corrected performance remains `PENDING CORRECTED RUN`.
- Limitations: Grouping removes duplicate-vector overlap, not semantic target proxies or deployment shift.
- Status: `implemented-not-run`.

## R4. Terminology and condition handling

- Exact concern: conditions were not always distinguished as training corruption, transductive partitioning, or deployment shift.
- Decision and rationale: preserve condition names, add explicit scope metadata, and reserve deployment-shift language for a protocol that changes held-out deployment distribution.
- Relevant files: `src/pipeline_runner.py`, `src/shift_generator.py`, `README.md`, `provenance/reviewer1_run_plan.md`.
- Implementation: primary/transductive/availability/relabeling scopes already fail closed when mixed; runner now persists `experiment_scope` and `split_policy` in manifests and task rows. Manuscript phrase checklist remains pending.
- Tests and commands: `tests/test_leakage_controls.py` condition semantics test passed in baseline suite.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: none; corrected values `PENDING CORRECTED RUN`.
- Limitations: Manuscript source still contains historical OpenML and shift-language statements.
- Status: `implemented-not-run`.

## R5. Baseline fairness

- Exact concern: Raw and AutoFE dimensionality/selection caps were not matched.
- Decision and rationale: retain the original Raw baseline, add a cap-matched Raw comparator and full-dimensional Raw comparator, and report pre/post-synthesis caps, selected counts, and cost.
- Relevant files: `src/pipeline_runner.py`, `src/feature_engineering.py`, result schema (planned).
- Implementation: `Raw_CapMatched` and full-dimensional `Raw` are explicit; isolate-one and leave-one-out operator configs share depth/base/output/variance budgets. Preflight ran Raw, cap-matched Raw, and AutoFE baseline under both policies.
- Tests and commands: bounded preflight completed in 1.52–1.80 seconds per three-task policy run on `sonar`; result/cache sizes are recorded in `provenance/reviewer1_preflight_manifest.json`.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: `PENDING CORRECTED RUN`.
- Limitations: Matched-cap results cannot be inferred from historical results.
- Status: `implemented-not-run`.

## R6. Individual operator effects

- Exact concern: `AutoFE_NoMultiply` excludes multiplication and division, and does not isolate individual operator effects.
- Decision and rationale: preserve its historical meaning; add isolate-one-operator and justified leave-one-out configurations with shared budgets and finite-value rules.
- Relevant files: `src/feature_engineering.py`, `src/pipeline_runner.py`, `src/shift_generator.py` (planned integration).
- Implementation: `AutoFE_Isolate_{Add,Subtract,Multiply,Divide}` and `AutoFE_LeaveOut_{Add,Subtract,Multiply,Divide}` configs are wired with matched budgets; historical `AutoFE_NoMultiply` remains explicitly addition/subtraction-only.
- Tests and commands: mechanism tests pass; operator-generation benchmark values remain pending.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: `PENDING CORRECTED RUN`.
- Limitations: Division-by-zero behavior and Featuretools expression naming require explicit validation.
- Status: `implemented-not-run`.

## R7. Incomplete runs and stopping

- Exact concern: missing tasks and stopped blocks could be mistaken for zeroes or complete evidence.
- Decision and rationale: every task must have success, failed, skipped-with-reason, timed-out, or pending status with resumable identity and coverage by all design dimensions.
- Relevant files: `src/checkpoint.py`, `src/pipeline_runner.py`, `src/check_progress.py`, `provenance/corrected_results_note.md`.
- Implementation: checkpoint schema accepts success/failed/skipped/timed_out/pending; manifests publish expected, terminal, pending, phase, and status counts while retaining legacy `counts` compatibility.
- Tests and commands: failure/resume tests and result-note accounting tests pass; a long-run ledger is not available.
- Commit hash: `d4a897a` integration milestone.
- Benchmark artifact IDs / observed values: no corrected run; historical ledger remains separately labeled.
- Limitations: unknown outcomes cannot support a causal or confirmatory claim.
- Status: `implemented-not-run`.

## R8. Reproducibility

- Exact concern: stable seeds, source identity, cache/run identity, and final artifacts must be independently reproducible.
- Decision and rationale: use canonical SHA-256 identities, source/sidecar checksums, runtime/package fingerprints, immutable task manifests, and explicit resume commands.
- Relevant files: `src/provenance.py`, `src/data_loader.py`, `src/pipeline_runner.py`, `requirements.txt`, `provenance/dataset_schema_audit.*`.
- Implementation: baseline reproducibility controls and UCI Dry Bean correction are committed in `a6a8e3b`; split policy, group digests, task fingerprints, and preflight artifacts are now included.
- Tests and commands: focused post-integration suite passed 42 tests; compileall, full suite, schema/hash audit, and group audit are final gates.
- Commit hash: `a6a8e3b` baseline provenance milestone.
- Benchmark artifact IDs / observed values: integrated source fingerprint `7f1a85210f1c3dd43c59381cad891e291242d874886f283b1e2b95d02abc84a7`; corrected run `PENDING CORRECTED RUN`.
- Limitations: historical Dry Bean input is unavailable and historical scores cannot be reconstructed.
- Status: `implemented-not-run`.

## R9. Confirmatory analysis now

- Exact concern: major fixes must be made and protocol frozen before the expensive run.
- Decision and rationale: do not launch until both split policies, common core, ablations, resource estimates, and result schemas are frozen and preflighted.
- Relevant files: `provenance/reviewer1_run_plan.md`, `provenance/corrected_results_note.md`, frozen manifest (planned).
- Implementation: plan, change catalog, response draft, corrected-result schema, group audit, and bounded real-data preflight are present. The long campaign is deliberately not launched: two configured datasets cannot support all-fold group-aware AUC, and candidate-history storage is estimated at about 79.7 GiB compressed for the original 612,500-task grid at the observed preflight rate.
- Tests and commands: bounded row/group preflight completed; final full-suite gate passed; only the pre-existing notebook execution-count edit remains outside the audit commits.
- Commit hash: `e3ed283` integrated audit milestone.
- Benchmark artifact IDs / observed values: none; no long run is running.
- Limitations: corrected numerical results are unavailable.
- Status: `blocked`.

## Dated decision log

- 2026-09-29: Existing branch/worktree inspected; all prior edits found uncommitted. No historical result file was rewritten.
- 2026-09-29: Baseline suite rerun in existing `p12`: 19 passed, 2 warnings in 27.12s; compileall passed.
- 2026-09-29: 25 datasets verified; Dry Bean source corrected to UCI ID 602. Commit `a6a8e3b` records baseline provenance and plan.
- 2026-09-29: Reviewer #1 run plan frozen provisionally; group-aware feasibility, mechanism measurement, operator isolation, and final resource gate remain open.
- 2026-09-29: Group audit completed in `p12`: all 25 group partitions construct, 23 support all-fold class/AUC metrics; `wine-quality-red` and `kddcup99` are recorded as infeasible rather than substituted.
- 2026-09-29: Bounded `sonar` preflight completed under both policies with Raw, cap-matched Raw, and AutoFE baseline; no full campaign was started. Candidate-history preflight wrote 950 training-only records and estimated approximately 79.7 GiB compressed for the full grid.
- 2026-09-29: Final post-integration gate in `p12`: 42 tests passed, compileall passed, schema audit 25/25, and compact group audit 25/25 structural with 23/25 AUC-supported.
