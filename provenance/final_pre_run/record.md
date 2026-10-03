# Final required pre-run repair and launch record

Starting branch `revision/leakage-seed-stability`, HEAD
`9ff798113cc4b117b61d6abd4c11c4973c34c892`, clean; existing evidence reused.
No applicable AGENTS.md. Conda P12 retained; `.venv`, source populations,
old failed attempts/caches and unrelated work preserved.

## Numerical contract, derived before acceptance evaluation

Installed sklearn 1.5.2 MLP `inplace_softmax` subtracts a row maximum, writes
exponentials in the input dtype, sums those represented nonnegative numerators,
and divides in that dtype. Exponential rounding affects the numerators but
does not change the mass identity of their exact normalization. For k output
columns, round-to-nearest unit roundoff u=eps(dtype)/2, denominator relative
error is bounded by g=(k-1)u/(1-(k-1)u). Each division contributes at most u.
Thus the sum of represented normalized entries has mass error at most
(g+u)/(1-g). Subnormal absolute rounding contributes k*smallest_subnormal/2;
the validator's float64 sum gets its own gamma_(k-1) accumulation bound.

The acceptance contract is separately versioned as
`probability_mass_normalization_roundoff_v1`: **1e-7 baseline + the derived
normalization/subnormal/accumulation allowance**, independent of an observed
failing residual. `probability_mass_float64_strict_1e7_v1` remains available as
the strict legacy comparison, and every accepted matrix records whether that
comparison passes. This explicitly changes acceptance semantics; the estimator,
original probability entries, logits, dtype, seeds and metric formulas remain
unchanged. No normalization, clipping or recomputation is performed.

Only the supported float32/float64 probability formats receive this allowance.
Range, finiteness, row/class shape and coordinate checks remain strict. Implicit
missing columns are rejected; an explicit estimator-class subset retains the
existing zero-alignment/undefined-held-out-class contract. A class dimension
without a small supported bound is rejected. The error model assumes supported
normalization operations, nonnegative numerators and no overflow; malformed
outputs are not rescued by the allowance.

The [primary softmax error analysis](https://academic.oup.com/imajna/article/41/4/2311/5893596)
and installed producer source informed the arithmetic inspection. The explicit
mass bound above is derived for the actual reduction/division normalization
path, rather than tuned to MLP or generic allclose defaults. Local declared
limits and derivation precede any new factory fit.

## Bounded verification declaration

At most 200 **additional** factory fits and three hours additional model/profile
execution; prior stage count320 remains separate. Two workers/one GPU maximum,
100 GiB rolling admission/50 GiB reserve, inherited native threads. The counter
starts before any fit. APS: original76,000 rows, clean root42/fold1, preparation
of all14 pipelines; only Raw_Full/AutoFE_Baseline with all10 models are probed.
Begin with one worker, 30-minute per-task ceiling limited by remaining budget,
one attempt, and a 4 GiB exact allocation ceiling further restricted by currently
available memory. This subset does not change the canonical full grid.

Working evidence: `reports/final_pre_run_20261003/`. Final outcomes, checks,
freeze/remote identity and launch or specific blocker are appended after checks.

## Required outcomes

**All three requirements passed. Full production launch is authorized.**

Numerical verification reused the two saved full probability matrices. All four
affected XGBoost tasks pass the accurate float64 strict legacy comparison;
only the three without saved probabilities were refitted with exact old inputs.
The saved MLP matrix passes the new contract and fails the strict legacy rule.
For seven float32 columns, the derived additional allowance is
4.172327912057336e-7, giving a total acceptance limit5.172327912057337e-7.
Saved XGBoost/MLP probability file hashes remain
`2dc69b1f2e213ac53c422c6c5081fc536bb32901f0c0f71249167accc56e38af` /
`f00706fb93994d428d3a9444826b98a132149b5e884e85be20fbf4c8537f7b2e`.
All checked metric values equal unchanged sklearn calculations. Model seeds
remain3951082829 /3763576478. The entire seed source and all scientific task
coordinates remain unchanged; the full model-purpose-seed sequence was checked.
Validation version changes model task identity and run compatibility, with
commit, production-claim, cleanup and provenance compatibility checks. Legacy
archived evidence remains verifiable without relabeling historical results.
Metric validation errors remain failures and retain inputs.

The ordinary runner accepts paired `--eligibility` /`--eligibility-sha256`,
and requires verified eligibility for the complete benchmark. The bundle pins
dataset identity, metadata, split/condition source, dependency versions,
protocol, seeds, fold configuration and each preparation-unit identity.
Pinned bytes, stale source/data/environment, missing/duplicate identities and
unsupported failure reasons are rejected before preparation. Full membership
was regenerated in nonexecuting mode and compared with the previous grid:

| Membership | Intended | Planned scientific skips | Split eligible |
|---|---:|---:|---:|
| Preparation units |8,750|415|8,335|
| Model tasks |1,225,000|58,100|1,166,900|

All1,233,750 task identities are unique, dependencies and initial states match,
and no models were dispatched by this grid check. Infeasible units: KDD315;
Mushroom50; kr-vs-kp50. Exact reasons/identities are preserved. Current verified
bundle SHA-256:
`6417a17150e250e21e5ab356a7eeeb522405c28956222604c9fe6aee89edec46`.
The bounded production fixture verifies eligible→planned-infeasible→eligible
advancement, zero skipped attempts, no fabricated eviction receipts, failure
retention/blocking, and unchanged behavior on resume. Ordinary CLI dry-run
integration is separately tested. Scientific split skips, undefined metrics,
device/resource failures and unexpected validation/worker failures retain their
separate accounting; no model failure was converted into a planned skip.

APS run `run_4fd5eb2af7e45b6af104778b`: exact76,000×170 population, all14
preparation pipelines,20 successful model results, one worker, inherited native
threads, one GPU at a time, full diagnostics/history. Preparation291.00s;
total691.47s; summed top-level model fits352.82s (mean17.64s). Full completion
and cleanup were verified; coordinator status `stopped` reflects the declared
20-model probe launch limit after all21 tasks completed, not a failed gate.

| APS observed measurement | Value |
|---|---:|
| Peak child RSS / summed children RSS |4.27 /4.57 GiB|
| Minimum available machine RAM |9.12 GiB|
| Peak machine CPU / process CPU |96.9% /688%|
| Peak GPU total used / utilization |3,849 MiB /68%|
| Peak disposable matrices |1,121,852,190 bytes (1.04 GiB)|
| Observed child write bytes, summed by PID |1,145,684,413|
| Minimum disk free |436,364,877,824 bytes (406.39 GiB)|
| Retained cache evidence after cleanup |16,653,185 bytes|

XGBoost reports cuda:0; CatBoost reports actual GPU execution. GPU totals include
other allocations. Temporary files were sampled at0 bytes, which does not
exclude subsecond atomic writes. All21 authoritative result hashes, worker
release, ledger export, retained histories/diagnostics and the eviction receipt
were verified.56 disposable files were deleted; no disposable matrices remain
in this isolated APS namespace. Receipt SHA-256
`e5baa8adc97ce466f6132956289a1c599a550d75b335bcf08962f7b4ab83d880`.
Unchanged estimator settings produced SVM/logistic convergence warnings; no
scientific settings were changed in response.

## Final checks and verification envelope

Focused numerical/eligibility/rolling/resume checks:81 passed,4 warnings.
Final added CLI/environment/stale-result/cleanup checks:4 passed.
P12 full suite ran **once after final source/test changes**:
**305 passed +7 subtests passed**,17 warnings,168.88s.
Compilation:91 Python files +4 notebook code cells. `git diff --check` passed.
All25 working CSV identities and metadata hashes match the prior evidence;
factory/seeding sources and production environment identity match. The initial
environment comparison omitted the runner's existing LOKY_MAX_CPU_COUNT=7
startup setting; importing the actual production entry point confirms an exact
environment match. No package/environment change was required.
Historical model ledger hash remains
`52399c7574fa20ae1999bef3adf91bc75d3571ef9d8c676bc226de4ee24b70d3`.
Historical bulk caches, sources, `.venv` and P12 were preserved.

Additional fits103 of200; previous stage320; cumulative423. Budgeted execution
999.72s of10,800s, including failed invocations. Initial focused invocation had
three test-cache collisions (78 passed/3 failed); isolated per-test cache
namespaces resolved them. A reporting-only DataFrame dtype error consumed no
fits and was corrected before successful probability verification. All attempts
remain accounted. No more pilot/optional optimization work is scheduled.

## Code freeze and production handoff

The frozen commit is the commit containing this record; its exact SHA and remote
verification are recorded locally in `reports/final_pre_run_20261003/freeze.json`.
The persistent launcher records that same SHA and refuses a dirty working tree
or a duplicate live coordinator. Source/configuration/environment must remain
frozen during the full experiment.

Production identity: `full_corrected_20261003_v1`; namespace
`reports/full_benchmark_20261003/`. The supported P12 module invocation is
equivalent to `setup_and_run.bat`, with these actual CLI arguments:

```text
--cache-policy rolling --cache-max-gib 100 --min-free-gib 50
--max-workers 2 --dense-budget-gib 4 --enable-fsva-diagnostics
--eligibility reports/final_pre_run_20261003/eligibility.json
--eligibility-sha256 6417a17150e250e21e5ab356a7eeeb522405c28956222604c9fe6aee89edec46
--run-id full_corrected_20261003_v1
--manifest-db reports/full_benchmark_20261003/manifest.db
--manifest-path reports/full_benchmark_20261003/manifest.jsonl
--results-path reports/full_benchmark_20261003/results.jsonl
--log-file reports/full_benchmark_20261003/terminal.log
```

There are no population/grid selectors, model launch cap, verification fit hook
or run deadline. Full production preserves its defaults: one attempt; no
per-task/run timeout. The30-minute ceiling was explicitly for new pre-run
checks. Dense allocations retain the dynamic available-memory guard. All25
populations, five roots/folds,14 conditions/pipelines, ten models, diagnostics
and histories remain in the manifest. No historical pilot run is reused.

Persistent Windows launch/resume helper:
`reports/final_pre_run_20261003/launch_full.ps1`, run using PowerShell with
`-NoProfile -ExecutionPolicy Bypass -File`; resume uses this same helper **only
after the recorded coordinator has exited**, under the unchanged frozen code
and configuration. Each invocation has separate stdout/stderr/terminal receipts.
`launch.json` records actual process/start/command identity; `progress.json`
and `progress_history.jsonl` report authoritative counts every60s. Read-only
progress command: `D:\Conda\P12\python.exe -B
reports/final_pre_run_20261003/full_progress.py`, which uses the supported
`ManifestStore` API. SQLite manifest/attempts/results are authoritative;
`results.jsonl` is the fsynced export. Initial live or terminal evidence is saved
after launch in `reports/final_pre_run_20261003/handoff.json`.

**Limitations:**100 GiB is disposable-cache admission, not a peak quota. The APS
check demonstrates immediate feasibility on this i5-11260H/32GB/RTX3050 4GB
host, with one worker; it does not prove two-worker scaling, every future cell,
or full retained evidence fits500GB. RAM/disk/device and rolling failure gates
remain active. Applying only the measured291s preparation cost to all350 APS
units would be28.3h if their costs matched; the other conditions/roots/folds and
other24 datasets are unmeasured. The model-probe mean17.64s does not represent
all14 pipelines. Full duration is uncertain, slow tails and export/prediction
costs matter, and October16 completion cannot be promised. No experiment
membership or scientific content is cut to fit the deadline.
