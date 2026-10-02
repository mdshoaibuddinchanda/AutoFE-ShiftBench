from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.feature_engineering import DFSConfig, expand_features_with_dfs
from src.fsva import compute_empirical_amplification, compute_jacobian_diagnostic, selection_stability, validate_jacobian_finite_difference
from src.operator_registry import (
    OPERATOR_SET_REGISTRY,
    SAFE_DIVISION_EPSILON,
    candidate_id,
    expression_to_string,
    op_expression,
    raw_expression,
    operator_set_manifest,
)
from src.pipeline_runner import PIPELINE_CONFIGS, pipeline_identity_token


class BaselineOperatorFsvaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.train = pd.DataFrame(
            {
                "a": np.linspace(1.0, 8.0, 12),
                "b": np.linspace(2.0, 13.0, 12),
                "c": np.linspace(-3.0, 2.0, 12),
                "category": ["x", "y"] * 6,
            }
        )
        self.test = pd.DataFrame(
            {
                "a": np.linspace(1.5, 8.5, 4),
                "b": np.linspace(2.5, 13.5, 4),
                "c": np.linspace(-2.5, 1.5, 4),
                "category": ["x", "new", "y", "new"],
            }
        )
        self.y = np.tile([0, 1], 6)

    def test_required_pipeline_specs_are_distinct_and_exact(self) -> None:
        expected = {
            "AutoFE_Baseline": "full_arithmetic_v1",
            "AutoFE_NoMultiply": "add_sub_v1",
            "AutoFE_AddSub": "add_sub_v1",
            "AutoFE_AddSubDiv": "add_sub_div_v1",
            "AutoFE_NoDivision": "add_sub_mul_v1",
            "AutoFE_MultiplyOnly": "multiply_only_v1",
            "AutoFE_DivideOnly": "divide_only_v1",
        }
        for name, operator_set in expected.items():
            self.assertEqual(PIPELINE_CONFIGS[name].operator_set_id, operator_set)
            self.assertEqual(tuple(PIPELINE_CONFIGS[name].trans_primitives), OPERATOR_SET_REGISTRY[operator_set])
        self.assertNotEqual(pipeline_identity_token("Raw_Full"), pipeline_identity_token("Raw_Capped"))
        self.assertIn("legacy", PIPELINE_CONFIGS["AutoFE_NoMultiply"].display_identity)
        self.assertEqual(
            operator_set_manifest()["add_sub_div_v1"]["operators"],
            ["add_numeric", "subtract_numeric", "divide_numeric"],
        )

    def test_full_raw_and_cap_matched_raw_have_declared_dimensions(self) -> None:
        full, _, full_meta = expand_features_with_dfs(
            self.train, self.test,
            config=DFSConfig(enable_dfs=False, max_features=None, max_base_features=None,
                              operator_set_id="none_v1", selection_method="none", monitor_ram=False),
        )
        capped, _, capped_meta = expand_features_with_dfs(
            self.train, self.test,
            config=DFSConfig(enable_dfs=False, max_features=2, max_base_features=None,
                              operator_set_id="none_v1", selection_method="variance", monitor_ram=False),
        )
        self.assertEqual(full.shape[1], 3)
        self.assertEqual(full_meta["baseline_kind"], "raw")
        self.assertEqual(full_meta["retained_generated_count"], 0)
        self.assertEqual(capped.shape[1], 2)
        self.assertEqual(capped_meta["requested_cap"], 2)
        self.assertEqual(capped_meta["actual_estimator_input_dimension"], 2)
        self.assertEqual(capped_meta["selector_identity"], "variance")

    def test_raw_cap_handles_fewer_features_and_mixed_type_encoding(self) -> None:
        selected, _, meta = expand_features_with_dfs(
            self.train, self.test,
            config=DFSConfig(enable_dfs=False, max_features=100, max_base_features=None,
                              operator_set_id="none_v1", selection_method="variance", monitor_ram=False),
        )
        self.assertEqual(selected.shape[1], 3)
        self.assertEqual(meta["eligible_base_feature_count"], 3)
        self.assertEqual(meta["requested_cap"], 100)
        self.assertEqual(meta["actual_estimator_input_dimension"], 3)
        self.assertNotIn("category", selected.columns)

    def test_selected_identities_are_training_only(self) -> None:
        altered_test = self.test.copy()
        altered_test.loc[:, "a"] = 1e9
        config = DFSConfig(enable_dfs=True, depth=1, max_features=5, max_base_features=None,
                           operator_set_id="add_sub_v1", selection_method="variance", monitor_ram=False)
        first, _, first_meta = expand_features_with_dfs(self.train, self.test, self.y, config)
        second, _, second_meta = expand_features_with_dfs(self.train, altered_test, self.y, config)
        assert_frame_equal(first, second)
        self.assertEqual(first_meta["selected_feature_identities"], second_meta["selected_feature_identities"])

    def test_operator_sets_control_actual_candidates_and_safe_division(self) -> None:
        for operator_set, forbidden in (
            ("add_sub_v1", {"multiply_numeric", "divide_numeric"}),
            ("add_sub_div_v1", {"multiply_numeric"}),
            ("add_sub_mul_v1", {"divide_numeric"}),
            ("multiply_only_v1", {"add_numeric", "subtract_numeric", "divide_numeric"}),
            ("divide_only_v1", {"add_numeric", "subtract_numeric", "multiply_numeric"}),
        ):
            _, _, meta = expand_features_with_dfs(
                self.train, self.test, self.y,
                DFSConfig(enable_dfs=True, depth=1, max_features=None, max_base_features=None,
                          operator_set_id=operator_set, selection_method="none", monitor_ram=False),
            )
            operators = {op for item in meta["selection_history"] for op in item["operators"]}
            self.assertTrue(operators.issubset(set(OPERATOR_SET_REGISTRY[operator_set])))
            self.assertTrue(operators.isdisjoint(forbidden))
        div_expr = op_expression("divide_numeric", raw_expression("a"), raw_expression("b"))
        near_zero = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, SAFE_DIVISION_EPSILON / 2]})
        _, _, div_meta = expand_features_with_dfs(
            near_zero, near_zero,
            config=DFSConfig(enable_dfs=True, depth=1, max_features=None, max_base_features=None,
                              operator_set_id="divide_only_v1", selection_method="none", monitor_ram=False),
        )
        matching = [row for row in div_meta["selection_history"] if row["expression_text"] == expression_to_string(div_expr)]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["validity"]["protected_count"], 2)

    def test_jacobian_and_amplification_match_known_mapping(self) -> None:
        frame = pd.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        add = op_expression("add_numeric", raw_expression("a"), raw_expression("b"))
        mul = op_expression("multiply_numeric", raw_expression("a"), raw_expression("b"))
        div = op_expression("divide_numeric", raw_expression("a"), raw_expression("b"))
        result = compute_jacobian_diagnostic(frame, [add, mul, div], raw_control_expressions=[raw_expression("a"), raw_expression("b")], max_rows=8, random_state=3)
        self.assertEqual(result["diagnostic_status"], "diagnostic_complete")
        self.assertAlmostEqual(result["per_output"][0]["mean_l2"], np.sqrt(2.0), places=8)
        self.assertGreater(result["selected"]["dimension_adjusted_frobenius"], 0.0)
        validation = validate_jacobian_finite_difference(frame, [add, mul, div], epsilon=1e-6)
        self.assertEqual(validation["status"], "validated")
        self.assertLess(validation["max_absolute_error"], 1e-4)
        amp = compute_empirical_amplification(frame, [mul], raw_control_expressions=[raw_expression("a"), raw_expression("b")], magnitudes=[1e-3, 1e-2], max_rows=8, random_state=3)
        self.assertEqual(len(amp["magnitudes"]), 2)
        self.assertTrue(all(row["status"] == "diagnostic_complete" for row in amp["magnitudes"]))

    def test_selection_stability_handles_empty_and_candidate_availability(self) -> None:
        value = selection_stability([
            {"selected_feature_identities": ["a", "b"], "candidate_history": [{"candidate_id": "a", "eligible": True, "selected": True}]},
            {"selected_feature_identities": ["b", "c"], "candidate_history": [{"candidate_id": "b", "eligible": True, "selected": True}]},
        ])
        self.assertAlmostEqual(value["pairwise_jaccard_mean"], 1 / 3)
        self.assertEqual(selection_stability([{"selected_feature_identities": [], "candidate_history": []}])["empty_set_jaccard"], 1.0)
        self.assertEqual(candidate_id(raw_expression("a")), candidate_id(raw_expression("a")))


if __name__ == "__main__":
    unittest.main()
