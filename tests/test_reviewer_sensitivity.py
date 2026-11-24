import numpy as np
import pandas as pd

from src.analysis.reviewer_sensitivity import (
    association_summary,
    dataset_associations,
    lodo_contrasts,
    spearman_safe,
    extreme_sensitivity,
)


def test_constant_spearman_is_undefined():
    rho, status = spearman_safe([1, 1, 1], [1, 2, 3])
    assert np.isnan(rho)
    assert status == "constant_predictor_or_outcome"


def test_dataset_association_keeps_dataset_as_cluster():
    frame = pd.DataFrame({
        "dataset": ["a"] * 4 + ["b"] * 4,
        "division_enabled": [True] * 8,
        "log10_jacobian_ratio": [1, 2, 3, 4, 4, 3, 2, 1],
        "mean_delta_auc_raw_minus_autofe": [1, 2, 3, 4, 1, 2, 3, 4],
    })
    result = dataset_associations(frame)
    assert list(result.dataset) == ["a", "b"]
    assert result.valid_mapping_units.tolist() == [4, 4]
    summary = association_summary(result, reps=100, )
    assert summary.iloc[0].datasets_estimable == 2


def test_lodo_uses_saved_dataset_effects():
    contrasts = pd.DataFrame([{
        "metric": "roc_auc", "analysis_scope": "principal_nontransductive",
        "left": "Raw_Capped", "right": "AutoFE_Baseline",
        "dataset_equal_mean_difference": 2.0,
        "dataset_effects": '{"a": 1.0, "b": 2.0, "c": 3.0}',
    }])
    result = lodo_contrasts(contrasts)
    row = result[(result.pipeline_contrast == "Raw_Capped vs AutoFE_Baseline") & (result.excluded_dataset == "a")].iloc[0]
    assert row.lodo_effect == 2.5
    assert row.status == "estimable"


def test_extreme_sensitivity_is_zero_against_its_own_original_estimand():
    frame = pd.DataFrame({
        "dataset": ["a"] * 4 + ["b"] * 4,
        "division_enabled": [True] * 8,
        "log10_jacobian_ratio": [1, 2, 3, 4, 4, 3, 2, 1],
        "mean_delta_auc_raw_minus_autofe": [1, 2, 3, 4, 1, 2, 3, 4],
    })
    result = extreme_sensitivity(frame, reps=20)
    original = result[result.sensitivity_treatment == "original"].iloc[0]
    assert original.difference_from_original == 0
