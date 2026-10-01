# FSVA, Jacobian, candidate history, and operator audit

Date: 2026-09-29
Scope: bounded mechanism assignment; synthetic checks only; no corrected benchmark run.

## Finding

Before integration, the pipeline did not measure FSVA or candidate-level
selection histories. The integrated path now exposes retained feature metadata,
operator counts, and an opt-in candidate-history writer in
`src/feature_engineering.py`; the bounded preflight exercised it on real
`sonar` training rows. Jacobian norms remain a reviewed mechanism contract
(`src/mechanism_audit.py`) rather than a claim that the full campaign has
measured every Featuretools expression. `src/pipeline_runner.py` retains
`AutoFE_NoMultiply` as add/subtract-only and now defines isolate-one and
leave-one-out operator configurations.

The new [mechanism contract](../src/mechanism_audit.py) is independent of the
runner so that its definitions can be reviewed before instrumentation is added
to an expensive campaign. It does not claim that the current benchmark has
measured FSVA.

## Jacobian definition

For a binary candidate with parent vector `u=(a,b)`, the candidate map is the
Featuretools arithmetic map `f(u)` and its Jacobian is the one-row gradient
`J_f(u) = [df/da, df/db]`. The supported analytic maps are:

| Operator | Map | Jacobian |
| --- | --- | --- |
| `add_numeric` | `a + b` | `[1, 1]` |
| `subtract_numeric` | `a - b` | `[1, -1]` |
| `multiply_numeric` | `a * b` | `[b, a]` |
| `divide_numeric` | `a / b` | `[1/b, -a/b²]`, only when `b != 0` |

The Jacobian is evaluated row by row on the **training-fold parent values**.
The proposed scale is the training-fold population standard deviation for each
parent. A non-finite or zero scale is undefined and is kept as `NaN`; it is not
replaced by one. For a finite scale vector `s`, the dimensionless local gain is
the norm of `J_f(u) diag(s)`. The implementation provides row-wise `l1`, `l2`,
and `linf` norms, with `l2` as the default. The planned primary summary is the
finite-only mean and median across training rows, together with the exact
finite and undefined counts and undefined reason counts.

The resulting value is a local sensitivity measure. It is not a causal proof
that an operator caused a performance change. A later dataset-level analysis
must treat the measured norm and selection behavior as explanatory covariates
and retain the performance contrast as the outcome.

## Analytic and numerical validation

`analytic_jacobian` implements the four formulas above. `finite_difference_jacobian`
uses a central difference with per-parent step
`1e-6 * max(1, abs(parent))`. A row is valid only when both perturbed outputs
and the resulting derivative are finite. `check_derivative` compares analytic
and numerical values with explicit absolute and relative tolerances.

Division by zero, non-finite parents, overflow, non-finite outputs, and
non-finite derivatives are recorded as undefined. The implementation never
clips, imputes, or substitutes a convenient derivative at such a point. A
derivative check with no valid rows returns status `undefined`; otherwise it
returns `pass` or `fail` and reports the number of valid rows and entries.

## Candidate-history contract

`CandidateHistoryRecord` requires these fields for every generated candidate:

`dataset`, `split_policy`, `seed`, `fold`, `condition`, `iteration`,
`candidate_id`, `parent_features`, `operator`, `admissible`,
`rejection_reason`, `selection_score`, `selection_score_status`,
`selection_decision`, and `candidate_seed`.

The score scope is fixed to `train`. A selected candidate must be admissible and
have a finite training-side score. An inadmissible candidate must have an
explicit rejection reason. Undefined or unavailable scores remain `None` with
an explicit status; they cannot be selected. Held-out labels are not part of
this schema and must remain in the separate evaluation ledger.

`CandidateHistoryWriter` appends and flushes one JSONL record at a time. A
`.gz` suffix enables gzip compression. It reports records written and
uncompressed bytes written; no record-drop path exists. The writer is intended
to be used during candidate generation, including rejected candidates, so the
history remains sufficient to reconstruct admissibility and selection.

### Storage estimate

The helper uses a default planning estimate of 420 UTF-8 bytes per JSONL
record and a configurable compressed ratio of 0.35. These are planning values,
not observed benchmark measurements; a short preflight must measure actual
records before a long run. For the 612,500-task historical grid:

| Candidate records per task | Records | Uncompressed | Gzip planning estimate |
| ---: | ---: | ---: | ---: |
| 100 | 61,250,000 | 24.0 GiB | 8.4 GiB |
| 1,000 | 612,500,000 | 239.6 GiB | 83.9 GiB |
| 2,000 | 1,225,000,000 | 479.2 GiB | 167.7 GiB |

These estimates exclude filesystem overhead and other caches. Detailed history
should therefore be streamed to compressed JSONL, and the preflight should
record the observed bytes per candidate before enabling it for the full grid.

## Operator-isolation recommendations

`recommended_operator_isolation_configs()` returns scopes with the same depth,
base-feature cap, output-feature cap, and variance-selection rule:

* `AutoFE_Baseline_AllOperators`: add, subtract, multiply, divide.
* `AutoFE_NoMultiply_Historical`: add and subtract only; the historical alias
  `AutoFE_NoMultiply` must retain the explicit note that it also excludes
  division.
* `AutoFE_Isolate_Add`, `AutoFE_Isolate_Subtract`,
  `AutoFE_Isolate_Multiply`, and `AutoFE_Isolate_Divide`: one enabled operator
  each.
* `AutoFE_LeaveOut_Add`, `AutoFE_LeaveOut_Subtract`,
  `AutoFE_LeaveOut_Multiply`, and `AutoFE_LeaveOut_Divide`: all operators
  except the named one.

The module validates that scopes use identical generation and selection
budgets. Division-by-zero and non-finite candidate filtering must use the same
policy as the Jacobian measurement. A real pipeline integration must also log
operator counts, admissible and selected counts, feature counts, runtime, and
undefined derivative counts per task.

## Evidence and limitations

Targeted synthetic tests cover all four analytic formulas, central-difference
checks, division-by-zero handling, finite-only norm aggregation, constant
training-scale handling, history validation and gzip streaming, storage
estimation, the historical no-multiply semantics, and isolate/leave-one-out
scope names. They do not exercise Featuretools candidate enumeration or any
real dataset and do not establish a performance mechanism.

Current pipeline inspection found:

* `src/feature_engineering.py:19-34`: `DFSConfig` defaults to the four
  arithmetic primitives.
* `src/feature_engineering.py:59-89`: feature selection is variance, MI, or
  random; MI receives `y_train`, but candidate histories are not emitted.
* `src/feature_engineering.py:108-189`: DFS is calculated from training
  definitions and applied to test data, while metadata retains only selected
  feature descriptions.
* `src/pipeline_runner.py:78-81`: `AutoFE_NoMultiply` enables add/subtract and
  therefore excludes multiply/divide.

The mechanism contract is `implemented-not-run` for the full campaign. The
bounded real-data integration wrote 950 candidate records (all four operators,
training-only finite scores) and measured approximately 53 KiB compressed for
the local history. At that observed rate, enabling detailed history for the
original 612,500-task grid would require approximately 79.7 GiB compressed;
the full history is therefore not enabled by default.
