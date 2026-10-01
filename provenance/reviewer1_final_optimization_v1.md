# Final optimization and Reviewer #1 audit (2026-10-01)

No full benchmark was launched. Existing Conda p12 was used throughout. The independent audit found and the lead fixed parallel cache accumulation and FIFO dispatch stalls. Source fingerprint: `35502d6cbe0d2e83183e3aee51825f9d42cc05d3230afba71fd9af5bd76ae98b`. Prior v2 freeze/readiness is historical; current scientific task counts are unchanged.

## Measured same-design comparison

Four small datasets, one seed/fold, clean condition, all 14 pipelines/all ten classifiers, four workers and numerical threads 1; 560 complete cells per policy/version. Baseline commit `d874fd43694d0d2f49b589e26d67056d6b47ef40` was checked out separately. No new environment was created.

| Policy | Baseline seconds | Updated seconds | Time reduction | Exact scientific parity |
|---|---:|---:|---:|---|
| row_level | 182.71 | 107.83 | 41.0% | 560/560; zero differences in all 24 checked fields |
| group_aware | 200.35 | 133.13 | 33.6% | 560/560; zero differences in all 24 checked fields |

Checked fields include exact train/test matrix and prediction hashes; ROC-AUC, train AUC, PR-AUC, log loss, Brier score, accuracy, balanced accuracy, F1, precision, recall, MCC, KS/Wasserstein; operator configurations/counts; original/retained/generated/train/test counts; and classifier backend. Timing/resource fields are measurements and may differ. Baseline ran first; OS cache/host variation can affect timing. These are not full-grid speedups, corrected performance conclusions or a campaign ETA. Numerical settings and every scientific task remain unchanged.

## Large-data and recovery evidence

Fresh Covertype Raw runs completed 10/10 each policy, with zero differences against corresponding old-source calibration cells on all 24 scientific fields. Measured wall times were 143.31 seconds (row) and 204.26 seconds (group); process-tree RSS peaks about 2.33 GiB, summing resident memory and possibly double counting shared pages. Historical 120-cell row/group calibrations remain old-source timing evidence.

The first Covertype Raw/AutoFE diagnostic was deliberately stopped at 19/20: the remaining AutoFE linear-SVM fit historically takes about 50 minutes per policy. Its partial results/interruption record are preserved; they are not counted as a complete run or accepted launch evidence. Fresh Covertype verification is Raw only. The full corrected campaign retains this expensive model/pipeline pair. Large AutoFE ten-model validation is the separate Airlines restart proof.

Fresh source-bound Airlines 600-second-lease forced stop/resume is `large_dataset_recovery_v3.json`, 20 cells per policy (Raw and AutoFE_Baseline, all ten classifiers). The verifier requires unchanged pre-stop artifacts/attempts and exact historical calibration output parity. The fresh v3 readiness report records completion, timings and host gate; no older green flag authorizes changed code.

## Runtime/storage changes and limits

- Reclaim one cache group after every planned durable consumer is terminal; pending retries remain protected. Capacity pressure drains completed workers and heartbeats the current admission task; preparation is memoized across retries.
- Publish completed futures in completion order; preserve four-worker/eight-pending bounds. Prepare/conversion/hash work occurs once per feature group; serial estimator mutations cannot reach later inputs.
- Measure generation, preprocessing and preparation time rather than fabricated generation zero. These group costs are shared across classifiers and require deduplication for totals. Cache high water includes staged publication; final admission includes payload/manifest/ready bytes.
- Retain the complete 2,275,000 intended performance scope, 84,000 planned group-AUC skips, at most 2,191,000 eligible before domain skips; separate 1,250 mechanism tasks with 50 skips. No model reduction, approximate fit, changed feature budget, changed stopping rule or GPU routing was introduced.
- Reserve 28.13 GiB results/checkpoints (historical 27.13 GiB plus 1 GiB for timing metadata), 8 GiB ready cache, 8 GiB publication temporary allowance and 1 GiB mechanism histories. With doubled-results margin the gate needs 73.26 GiB free. This is a planning reserve, not a guaranteed full-grid size; check live disk and back up outputs.
- Retryable-failed groups and externally held/malformed leases intentionally retain cache; extensive failures can require stop/recovery. Bounded samples cannot prove every full-grid failure mode or give a reliable all-dataset ETA. The older 229.69-day scenario is a historical uncertain reference. The user accepts longer runtime.

## Reviewer #1 delivery and validation

The [completion checklist](reviewer1_completion_checklist.md) maps all nine scientific concerns and the engineering follow-up. The [response draft](reviewer1_response_draft.md), [change catalog](reviewer1_change_catalog.md) and [manuscript phrase checklist](reviewer1_manuscript_phrase_checklist.md) are updated. Implemented scientific methods are bounded-verified; empirical conclusions and later manuscript/rebuttal numerical revision remain **PENDING CORRECTED RUN**.

Paper propagation now uses the streaming exporter: 14 primary CSVs, 7 PNG/PDF pairs, asset manifest and machine-checked result record. Includes prespecified ROC-AUC statistical-family F1 bootstrap/Holm, common-dataset row/group comparisons, descriptive matched-cap/operator contrasts, primary missing-outcome bounds, candidate stages and measured resource fields. F1 is the family identifier, not the classifier F1 metric. Mechanism grid/source/artifact identity is checked before association. Separate sensitivity export remains descriptive; primary bounds are not silently applied there.

Validation details: 34 focused cache/runner tests passed; exporter 9 focused tests passed; strengthened mechanism 14 focused tests passed. The final integrated p12 suite passed **139 tests**, 26 warnings, in 94.31 seconds after all version-wiring checks. Fresh readiness is recorded in `reviewer1_launch_readiness_v3.json`. A preliminary focused recovery test was invalidated by source edits between subprocess start/resume and then passed in the stable rerun; source identity correctly rejected mixed runs.

`notebooks/visualization.ipynb` has an existing execution-count-only edit, untouched and excluded from revision commits. Four pre-existing Markdown table-spacing edits were inspected and included as formatting only. No push or submission was performed.
