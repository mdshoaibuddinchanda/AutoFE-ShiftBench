# Statistical methods

## Input and pairing

The corrected ledger was read as JSONL without fitting any model. The exact
pairing key was:

```text
(dataset, seed, fold, condition, model,
 source_code_fingerprint, environment_fingerprint)
```

Pipeline pairs were matched only when both metrics were present in the same
context. Repeated contexts were retained once for deterministic pairing;
duplicate task IDs and metric conflicts were audited separately.

## Dataset-level inference

- Differences were averaged within dataset, making dataset the independent
  unit.
- Confidence intervals are 10,000-draw percentile bootstrap intervals over
  dataset effects.
- Two-sided Wilcoxon signed-rank tests were applied to dataset effects.
- Holm correction was applied over all 91 pipeline pairs within each metric and
  scope.

Scopes include all available rows, the principal analysis excluding incomplete
Covertype, and corresponding analyses excluding `covariate_shift` and
`population_shift`. These are post-hoc analyses of saved metrics, not new
model experiments.
