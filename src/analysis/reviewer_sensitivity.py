"""Reviewer sensitivity analyses over saved Jacobian and contrast evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ANALYSIS_VERSION = "reviewer_sensitivity_v1"


def spearman_safe(x, y):
    frame = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3:
        return np.nan, "insufficient_valid_units"
    if frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return np.nan, "constant_predictor_or_outcome"
    return float(spearmanr(frame.x, frame.y).statistic), "estimable"


def dataset_associations(frame):
    required = {"dataset", "division_enabled", "log10_jacobian_ratio", "mean_delta_auc_raw_minus_autofe"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing Jacobian association columns: {sorted(missing)}")
    rows = []
    for (division, dataset), group in frame.groupby(["division_enabled", "dataset"], sort=True):
        rho, status = spearman_safe(group.log10_jacobian_ratio, group.mean_delta_auc_raw_minus_autofe)
        valid = group[["log10_jacobian_ratio", "mean_delta_auc_raw_minus_autofe"]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"dataset": dataset, "operator_family": "division_enabled" if bool(division) else "division_free",
                     "valid_mapping_units": len(valid), "spearman_rho": rho,
                     "association_direction": "positive" if np.isfinite(rho) and rho > 0 else "negative" if np.isfinite(rho) and rho < 0 else "undefined",
                     "estimator_status": status})
    return pd.DataFrame(rows)


def cluster_bootstrap(values, reps=10_000, seed=20261124):
    values = np.asarray([float(v) for v in values if np.isfinite(v)])
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(reps)])
    return tuple(np.quantile(means, [0.025, 0.975]))


def association_summary(table, reps=10_000):
    rows = []
    for family, group in table.groupby("operator_family", sort=True):
        values = group.loc[np.isfinite(group.spearman_rho), "spearman_rho"].to_numpy()
        low, high = cluster_bootstrap(values, reps)
        rows.append({"operator_family": family, "datasets_estimable": len(values),
                     "total_mapping_units": int(group.valid_mapping_units.sum()),
                     "dataset_median_rho": float(np.median(values)) if len(values) else np.nan,
                     "dataset_equal_mean_rho": float(np.mean(values)) if len(values) else np.nan,
                     "cluster_bootstrap_ci95_low": low, "cluster_bootstrap_ci95_high": high,
                     "positive_datasets": int((values > 0).sum()), "negative_datasets": int((values < 0).sum()),
                     "undefined_datasets": int(len(group) - len(values))})
    return pd.DataFrame(rows)


def _dataset_equal_association(group, ratio):
    """Return the mean of dataset-level associations for one operator family."""
    work = group.assign(_analysis_ratio=ratio)
    estimates = []
    for _, dataset_group in work.groupby("dataset", sort=True):
        rho, status = spearman_safe(dataset_group._analysis_ratio,
                                    dataset_group.mean_delta_auc_raw_minus_autofe)
        if status == "estimable":
            estimates.append(rho)
    return float(np.mean(estimates)) if estimates else np.nan


def extreme_sensitivity(frame, reps=10_000):
    ratio = frame.log10_jacobian_ratio.replace([np.inf, -np.inf], np.nan)
    q95, q99 = ratio.quantile([0.95, 0.99])
    treatments = {"original": ratio, "remove_upper_1pct": ratio.where(ratio <= q99),
                  "winsorize_99pct": ratio.clip(upper=q99), "remove_above_95pct": ratio.where(ratio <= q95)}
    rows = []
    for treatment, transformed in treatments.items():
        work = frame.assign(analysis_ratio=transformed)
        for division, group in work.groupby("division_enabled", sort=True):
            retained = group.analysis_ratio.notna() & group.mean_delta_auc_raw_minus_autofe.notna()
            rhos = []
            for _, dataset_group in group.loc[retained].groupby("dataset"):
                rho, status = spearman_safe(dataset_group.analysis_ratio, dataset_group.mean_delta_auc_raw_minus_autofe)
                if status == "estimable":
                    rhos.append(rho)
            low, high = cluster_bootstrap(rhos, reps)
            estimate = float(np.mean(rhos)) if rhos else np.nan
            # Compare each treatment with the same dataset-equal estimand under
            # the unmodified ratios.  A treatment named ``original`` therefore
            # has an exact zero difference from its own baseline.
            original = _dataset_equal_association(
                group, group.log10_jacobian_ratio
            )
            rows.append({"operator_family": "division_enabled" if bool(division) else "division_free",
                         "sensitivity_treatment": treatment,
                         "threshold_log10_ratio": np.nan if treatment == "original" else float(q95 if "95" in treatment else q99),
                         "observations_retained": int(retained.sum()), "datasets_represented": int(group.loc[retained, "dataset"].nunique()),
                         "dataset_equal_association": estimate, "ci95_low": low, "ci95_high": high,
                         "difference_from_original": estimate - original if np.isfinite(estimate) and np.isfinite(original) else np.nan,
                         "interpretation": "direction_consistent" if np.isfinite(estimate) and np.isfinite(original) and np.sign(estimate) == np.sign(original) else "unstable_or_undefined"})
    return pd.DataFrame(rows)


def lodo_contrasts(contrasts, scope="principal_nontransductive"):
    requested = [("Raw_Capped", "AutoFE_Baseline"), ("Raw_Full", "AutoFE_Baseline"),
                 ("Raw_Capped", "AutoFE_AddSub"), ("AutoFE_AddSub", "AutoFE_AddSubDiv"),
                 ("AutoFE_AddSub", "AutoFE_NoDivision")]
    rows = []
    for left, right in requested:
        hit = contrasts[(contrasts.metric == "roc_auc") & (contrasts.analysis_scope == scope) &
                        (contrasts.left == left) & (contrasts.right == right)]
        if hit.empty:
            rows.append({"pipeline_contrast": f"{left} vs {right}", "status": "not_computable_from_saved_scope"})
            continue
        record = hit.iloc[0]
        effects = json.loads(record.dataset_effects)
        original = float(record.dataset_equal_mean_difference)
        exclusions = {name: float(np.mean([value for key, value in effects.items() if key != name])) for name in effects}
        changes = np.abs(np.asarray(list(exclusions.values())) - original)
        for dataset, effect in exclusions.items():
            rows.append({"pipeline_contrast": f"{left} vs {right}", "excluded_dataset": dataset,
                         "datasets_remaining": len(effects) - 1, "original_paired_effect": original,
                         "lodo_effect": effect, "absolute_change": abs(effect - original),
                         "sign_reversal": bool(np.sign(effect) != np.sign(original) and effect != 0 and original != 0),
                         "influential": bool(abs(effect - original) == changes.max()), "status": "estimable"})
    return pd.DataFrame(rows)


def write_latex(frame, path, caption):
    path.write_text("% Generated by reviewer_sensitivity.py; saved-evidence table.\n" +
                    frame.to_latex(index=False, escape=True, caption=caption), encoding="utf-8")


def run(jacobian_path, contrasts_path, output_dir, reps=10_000):
    jacobian_path, contrasts_path, output_dir = map(Path, (jacobian_path, contrasts_path, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    jacobian, contrasts = pd.read_csv(jacobian_path), pd.read_csv(contrasts_path)
    assoc = dataset_associations(jacobian)
    summary = association_summary(assoc, reps)
    extreme = extreme_sensitivity(jacobian, reps)
    lodo = lodo_contrasts(contrasts)
    provenance = pd.DataFrame([{"status": "NOT_COMPUTABLE_FROM_PUBLISHED_AGGREGATES",
                                "reason": "No balanced provenance-stratified dataset-level matched contrast table is saved."}])
    assoc.to_csv(output_dir / "jacobian_dataset_associations.csv", index=False)
    summary.to_csv(output_dir / "jacobian_association_summary.csv", index=False)
    extreme.to_csv(output_dir / "jacobian_extreme_value_sensitivity.csv", index=False)
    lodo.to_csv(output_dir / "primary_contrast_lodo.csv", index=False)
    provenance.to_csv(output_dir / "provenance_sensitivity.csv", index=False)
    write_latex(assoc, output_dir / "table_S1_jacobian_association.tex", "Dataset-level Jacobian sensitivity and performance association")
    write_latex(extreme, output_dir / "table_S2_jacobian_robustness.tex", "Jacobian extreme-value robustness")
    write_latex(lodo, output_dir / "table_S3_lodo_robustness.tex", "Leave-one-dataset-out ROC-AUC stability")
    manifest = {"analysis_version": ANALYSIS_VERSION, "bootstrap_replicates": reps, "bootstrap_seed": 20261124,
                "jacobian_input": jacobian_path.name, "contrasts_input": contrasts_path.name,
                "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (jacobian_path, contrasts_path)},
                "jacobian_rows": len(jacobian), "jacobian_datasets": sorted(jacobian.dataset.unique().tolist()),
                "additional_figures_required": "NO", "provenance_status": "NOT_COMPUTABLE_FROM_PUBLISHED_AGGREGATES"}
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary_text = ["# Reviewer sensitivity analysis", "", "Read-only descriptive analysis of saved evidence.",
                    f"Jacobian rows: {len(jacobian)}; represented datasets: {jacobian.dataset.nunique()}.",
                    "Dataset is the independent cluster for associations and bootstrap intervals.",
                    "Denominator/protection-level statistics are unavailable from the saved matched table.",
                    "", "ADDITIONAL FIGURES REQUIRED: NO", "",
                    "Provenance sensitivity: NOT COMPUTABLE FROM PUBLISHED AGGREGATES."]
    (output_dir / "reviewer_sensitivity_summary.md").write_text("\n".join(summary_text) + "\n", encoding="utf-8")
    return {"association_rows": len(assoc), "extreme_rows": len(extreme), "lodo_rows": len(lodo)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacobian", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(run(args.jacobian, args.contrasts, args.output_dir, args.bootstrap_reps), indent=2))


if __name__ == "__main__":
    main()
