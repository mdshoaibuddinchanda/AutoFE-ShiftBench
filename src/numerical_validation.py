"""Versioned acceptance of represented probability mass; never changes entries.

For nonnegative represented numerators a_i normalized in their output dtype,
the denominator reduction has relative error <= gamma_(k-1). Each division
has relative error <= u. Thus the represented mass differs from one by at
most (gamma_(k-1) + u)/(1 - gamma_(k-1)). The original absolute criterion is
kept explicit, with this additional arithmetic allowance separately recorded.
"""
import numpy as np

NUMERICAL_VALIDATION_VERSION = 'probability_mass_normalization_roundoff_v1'
LEGACY_VALIDATION_VERSION = 'probability_mass_float64_strict_1e7_v1'
BASELINE_MASS_ATOL = 1e-7


def mass_error_budget(dtype, columns):
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')) or columns < 1:
        raise ValueError('probability validation supports float32/float64 and positive class dimension')
    u = float(np.finfo(dtype).eps) / 2
    m = columns - 1
    if m * u >= .5:
        raise ValueError('class dimension has no supported small rounding allowance')
    gamma = m * u / (1 - m * u)
    normalization = (gamma + u) / (1 - gamma)
    # Absolute rounding of subnormal division outputs, without a relative model.
    subnormal = columns * float(np.finfo(dtype).smallest_subnormal) / 2
    u64 = float(np.finfo(np.float64).eps) / 2
    gamma64 = m * u64 / (1 - m * u64)
    accumulation = gamma64 * (1 + normalization + subnormal)
    allowance = normalization + subnormal + accumulation
    return {'baseline_mass_atol': BASELINE_MASS_ATOL, 'unit_roundoff': u,
            'class_columns': columns, 'normalization_rounding_bound': normalization,
            'subnormal_absolute_bound': subnormal, 'float64_accumulation_bound': accumulation,
            'additional_rounding_allowance': allowance,
            'acceptance_limit': BASELINE_MASS_ATOL + allowance}


def validate_probability_mass(probabilities, version=NUMERICAL_VALIDATION_VERSION):
    if version not in (NUMERICAL_VALIDATION_VERSION, LEGACY_VALIDATION_VERSION):
        raise ValueError('unknown numerical validation version')
    values = np.asarray(probabilities)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError('probability dimensions must be a nonempty class matrix')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError('probabilities must be finite and within [0,1]')
    budget = mass_error_budget(values.dtype, values.shape[1])
    residual = np.abs(values.sum(axis=1, dtype=np.float64) - 1.)
    maximum = float(residual.max()) if len(residual) else 0.
    legacy = maximum <= BASELINE_MASS_ATOL
    accepted_limit = budget['acceptance_limit'] if version == NUMERICAL_VALIDATION_VERSION else BASELINE_MASS_ATOL
    if maximum > accepted_limit:
        raise ValueError('probability rows must sum to one within the declared numerical validation budget')
    return {'version': version, 'probability_dtype': str(values.dtype), **budget,
            'applied_acceptance_limit': accepted_limit, 'maximum_mass_residual': maximum,
            'strict_legacy_pass': legacy, 'probability_entries_changed': False}
