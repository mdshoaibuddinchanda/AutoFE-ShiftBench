# Real-data probability validation repair

The frozen production pilot executed commit
`05d6c753248d969cc574fdeb4c9a2955e364a2d0`. Its Dry Bean clean unit stopped with
11 `MetricInputError: probability rows must sum to one` failures (seven MLP,
four XGBoost). The authoritative ledger, failed attempts, input cache and frozen
configuration remain unchanged. No next preparation unit was admitted.

## Demonstrated defect and bounded correction

One predeclared exact-input XGBoost reproduction used the failed
`AutoFE_NoDivision` task, root42/fold1, its original 10,888 training/2,723 test
rows, 100 selected coordinates, model seed 3951082829, historical float32 input
conversion and unchanged GPU factory. The fitted booster reported `cuda:0`.
Its probability matrix is finite, in [0,1], float32 and seven columns. Native
float32 summation has maximum residual 1.1920928955078125e-7; 279 rows fail the
old 1e-7 check. Summing the **same represented probabilities** in float64 gives
maximum residual 8.731115030968795e-8, inside the original tolerance. The
unchanged installed sklearn 1.5.2 metrics accept this matrix.

`src/evaluation.py` now accumulates only the mass validation sum in float64.
The tolerance stays `rtol=0, atol=1e-7`. Probability arrays, estimator inputs,
parameters, seeds, native threads, class alignment and metric calculations
remain unchanged. This fixes an inaccurate reduction rather than normalizing
probabilities or relaxing acceptance. The stored reproduced probability matrix
passes the repaired guard with ROC-AUC and log loss exactly equal to the original
sklearn calls; this recheck launches no fit and does not write a result row.

Two new metric regressions use the observed seven-column row and verify exact
sklearn metrics, unchanged float32 bytes, and continued rejection of represented
mass genuinely outside 1e-7. Focused metrics: **12 passed**, two warnings, 1.36 s.
Final complete suite: **263 passed, seven subtests, 15 warnings, 151.60 s**.
Compilation: 86 Python files/four notebook code cells; no notebook execution or
bytecode generation. `git diff --check` passed.

## Remaining blocker

One separately predeclared exact-input CPU MLP reproduction (same pipeline and
rows; model seed 3763576478) still fails the unchanged guard after repair.
Its maximum float64 mass residual is 2.095644059396662e-7 (native float32 residual
2.384185791015625e-7). This is finite float32 softmax output which sklearn's
existing metric implementation accepts, but it violates this project's stricter
declared mass tolerance. No normalization, tolerance increase or dtype change
was applied. Resolving that conflict needs a separately declared validation
contract and equivalence review. The remaining original failed combinations
were not all rerun; the correction is not a claim that every XGBoost/MLP cell is
now executable.

The old frozen pilot cannot resume under changed Python content. Its successful
same-run stop/resume checks preceded this repair. No new pilot identity was used
to bypass the failed unit's cleanup gate. APS, Gaussian corruption, the second
root and the remaining Dry Bean models stay explicitly unexecuted.

## Execution accounting and evidence

The user envelope remains 840 launches/four hours/two workers/one GPU. The first
two full suites consumed 94 top-level factory fits; the original XGBoost
reproduction consumed one. After the pilot stopped at 177 model attempts,
`repair_verification_plan.json` predeclared one further full suite (47 factory
fits) and one MLP reference fit. Three unused launch slots were reassigned from
the original verification reserve, without increasing the global envelope.
Final counts: **177 pilot attempts + 143 verification/reference factory fits =
320**. Inner calibration fits belong to their containing factory/task attempt.
No retries or scientific observations were hidden in the reference outputs.

Exact commands, fit counter SQLite, probability bytes/hashes, input verification,
reproduction plans and results, XML, final source compilation, timing and budget
receipts remain in `reports/real_data_preflight_20261003/` (ignored locally).
The reference watchdog was 120 seconds; measured fit durations were 3.187 s
(XGBoost) and 0.859 s (MLP). These are verification fits, separate from pilot
performance and outcomes.

The [NumPy floating-point reference](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
defines epsilon; installed NumPy 1.26.4 reports float32 epsilon above the existing
1e-7 tolerance. The [sklearn 1.5 ROC-AUC contract](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.roc_auc_score.html)
requires class probability mass and coordinate alignment. Actual saved matrices
and installed-source calls provide the defect evidence; neither page authorizes
changing this project's tolerance.
