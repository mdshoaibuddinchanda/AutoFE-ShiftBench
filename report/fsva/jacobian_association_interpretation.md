# Jacobian–performance association interpretation

The join uses only locally available sidecars matched to saved performance rows by diagnostic identity and the full dataset/seed/fold/condition/model/provenance context. Multiple estimator rows referencing the same feature mapping are collapsed to one mapping unit by averaging their matched AUC differences; they are not treated as independent Jacobian observations.

The outcome is `Raw AUC - AutoFE AUC`, so positive values indicate lower AutoFE AUC. Jacobian ratios are analysed on a log10 scale because division-enabled mappings contain extreme, scientifically retained values caused by near-zero denominators. Spearman associations are descriptive and do not establish causality. The feature-map Jacobian is not an estimator-prediction Jacobian and does not measure the complete covariance-weighted FSVA index. Dataset-level association is not estimable when a dataset/pipeline has insufficient variation.
