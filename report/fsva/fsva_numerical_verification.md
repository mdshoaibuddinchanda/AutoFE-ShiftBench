# FSVA numerical verification

This is a read-only synthesis of saved FSVA revalidation artifacts. No estimator, preprocessor, split, or feature-search fit was executed.

## Scope and reproduction

- FSVA artifacts: 31,450; flagged artifacts: 8,877; reconstructed contexts: 1,970.
- Original saved metrics reproduced: 8,877; mismatches: 0.
- Measured norm: dimension-adjusted Frobenius norm (`||J||_F / sqrt(number of selected outputs)`); ratio: selected mapping norm divided by paired raw-control mapping norm.
- Sidecars cover 14 configured pipelines across 10 datasets.

## Finite-difference behavior

- At relative coordinate step 1e-7, 51,658,816 values were compared; 49,108,668 were classified smooth; 3,261 smooth-point checks failed the stated scale-aware tolerance.
- The targeted residual probe contains 3,261 comparisons across 37 contexts and 153 artifacts. All failures were denominator derivatives with absolute denominators in [2.33677e-08, 3.07617e-05].
- Denominator-relative central differences (`h = epsilon * |denominator|`) resolved all 3,261 targeted residual comparisons at epsilons 1e-3, 1e-4, 1e-5, and 1e-6. This is targeted evidence, not a full-artifact rerun under that rule.
- Large analytic derivatives and ratios remain genuine local sensitivity observations; the numerical caveat is finite-difference conditioning near zero denominators.

## Measured quantities and boundaries

Measured: sampled local feature-map Jacobian norms, paired normalized ratios, finite-difference residual behavior, and operator-stratified summaries.
Not measured: estimator-prediction Jacobians, a complete covariance-weighted FSVA index, search-multiplicity terms, variance/bias terms, or a causal mechanism linking local mapping sensitivity to generalization.

## Decision

The saved Jacobian formulas and reported magnitudes are numerically defensible as local feature-map diagnostics with explicit near-zero-division conditioning limits. They do not justify a causal generalization claim without the matched, clustered association analysis and careful qualification.
