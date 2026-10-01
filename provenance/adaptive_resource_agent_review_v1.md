# Independent adaptive execution review

Recorded 2026-10-01 UTC. The lead integrated the read-only findings returned by `/root/final_runtime_audit`, as requested by the author. The independent agent performed no edits, tests, fits or benchmark launches during this final check.

## Conclusion

**No remaining launch blocker was found on the recorded host.** All seven decisions in `reviewer1_launch_readiness_v4.json` are green. Re-run the verifier immediately before each long command; this is a dated snapshot, not a live guarantee. The full corrected campaign has not started.

The earlier code-review stage reported HOLD while bounded evidence, mechanism scope and readiness were missing. The final evidence review below resolves that engineering hold.

## Independently checked evidence

- Source fingerprint `68d54329e107d42751eaa546a5048c270191e4c89f644ca6e039a2fb20dac4d4` matches frozen source commit `bf64794b976424677323518a6f0d50e91f444fbd`.
- All **161 checked evidence, scope, analysis and reporting hashes** match actual files. Seven distinct run IDs match the frozen scope; all seven readiness decisions are green.
- All **1,148 successful diagnostic rows** were compared independently: **1,136 exact CPU comparisons** across all 24 recorded comparison fields and **12 GPU feature/candidate comparisons**, with no discrepancies. GPU metric equality with CPU is not asserted.
- Disabled operators have zero candidate counts. All 14 configured pipeline meanings remain correct; the historical joint-removal identity is preserved.
- Mechanism smoke **005** has four complete tasks, matching source/script identities, history/Jacobian hashes and adaptive runner matrices.
- Reporting contains **15 primary CSVs and eight PNG/PDF pairs**, plus the separate targeted GPU resource table/figure pair. Diagnostic labels are visible. The resource diagnostic banner/title spacing is a cosmetic limitation; values remain legible.
- One persistent process per verified GPU reuses its device context. Active CPU/GPU fits together respect the worker ceiling. The common comparison capacity bound keeps backend selection consistent across pipelines; live VRAM pressure waits rather than changing backend.
- Frozen profile hash, source, host, settings and overrides are bound to configuration/task/cache identities. Changed identities are rejected.

The independent agent checked the saved full-suite evidence (**160 passed, 27 warnings, 109.18 seconds in existing p12**) and did not rerun tests.

## Exact configured operator matrix

| Pipeline | Addition | Subtraction | Multiplication | Division |
|---|:---:|:---:|:---:|:---:|
| Raw | No | No | No | No |
| Raw_CapMatched | No | No | No | No |
| AutoFE_Baseline | Yes | Yes | Yes | Yes |
| AutoFE_MI | Yes | Yes | Yes | Yes |
| AutoFE_Random | Yes | Yes | Yes | Yes |
| AutoFE_NoMultiply | Yes | Yes | No | No |
| AutoFE_Isolate_Add | Yes | No | No | No |
| AutoFE_Isolate_Subtract | No | Yes | No | No |
| AutoFE_Isolate_Multiply | No | No | Yes | No |
| AutoFE_Isolate_Divide | No | No | No | Yes |
| AutoFE_LeaveOut_Add | No | Yes | Yes | Yes |
| AutoFE_LeaveOut_Subtract | Yes | No | Yes | Yes |
| AutoFE_LeaveOut_Multiply | Yes | Yes | No | Yes |
| AutoFE_LeaveOut_Divide | Yes | Yes | Yes | No |

No pipeline was added or redefined by the adaptive follow-up. Seven performance runs retain **2,275,000 intended cells**, **84,000 planned group-AUC skips**, and at most **2,191,000 eligible cells** before condition-specific skips. The separate mechanism scope has **1,250 tasks / 50 skips / 1,200 histories**.

## Limits and permanent records

RAM/VRAM budgets are soft estimates. Sustained external memory pressure can pause admission indefinitely. Useful small workloads use less than 20 GiB. Bounded timings provide no validated full-campaign ETA. GPU numerical equality/repeatability, operator effects, Jacobian associations and corrected statistical conclusions remain **PENDING CORRECTED RUN**. The targeted GPU proof covers clean Raw/Baseline paths, not every operator/condition performance effect.

The immutable scope retains freeze-time `unresolved_gates` and calibration `PENDING` fields. Completed evidence and the dated readiness snapshot record their resolution. Preserve scope bytes and historical diagnostics.

Lead dependency supplement: the verifier checks **21 exact package pins**; a read-only metadata check also confirmed `setuptools 80.10.2` satisfies the remaining `setuptools<81` constraint. Existing p12 was not modified. Dated disk space is **431.85 GiB free**, versus **87.11 GiB** required by the planning margin.

The pre-existing `notebooks/visualization.ipynb` execution-count change remains untouched, outside the frozen source and revision commits.
