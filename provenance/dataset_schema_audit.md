# Downloaded dataset schema and integrity audit

Audit date: 2026-09-29 (Asia/Calcutta). Environment: existing Conda `p12` at `D:\Conda\p12`.

Downloaded and inspected **25 of 25** configured benchmark datasets. The downloader caps each saved dataset at 100,000 rows. SHA-256 and sidecar schema checks passed for **25 of 25** dataset/sidecar pairs.

This is a structural screen. Exact target matches, target-like names, perfect one-column target mappings, and identifier-like fields are candidates for human review and are never removed automatically. Column names and values alone cannot establish semantic leakage.

The audit also replayed the current row-level stratified five-fold splitter with seed 42 and checked whether each test row had an identical feature vector in training. This measures split overlap in the saved data; it does not estimate score inflation. These repeated rows remain in the data because their frequencies are part of the source distribution.

| Dataset | Rows | Features | Target (classes) | Missing feature cells | Repeated X rows | Conflicting X groups | Test rows with train X match, 5-fold seed 42 | Exact copies | Target-like names | Perfect mappings | Identifier-name candidates | High-cardinality text fields |
|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| `haberman` | 306 | 3 | `target` (2) | 0 | 45 | 6 | 11.1111% | — | — | — | — | — |
| `sonar` | 208 | 60 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `ionosphere` | 351 | 34 | `target` (2) | 0 | 2 | 0 | 0.5698% | — | — | — | — | — |
| `heart-disease` | 303 | 13 | `target` (2) | 0 | 2 | 0 | 0.6601% | — | — | — | — | — |
| `breast-cancer-wisconsin` | 569 | 30 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `blood-transfusion-service-center` | 748 | 4 | `target` (2) | 0 | 315 | 31 | 39.1711% | — | — | — | — | — |
| `diabetes` | 768 | 8 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `titanic` | 1,309 | 13 | `target` (2) | 3,855 | 0 | 0 | 0.0% | — | — | — | — | `name` |
| `credit-g` | 1,000 | 20 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `wine-quality-red` | 1,599 | 11 | `target` (6) | 0 | 460 | 0 | 22.7642% | — | — | — | — | — |
| `kr-vs-kp` | 3,196 | 36 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `mushroom` | 8,124 | 22 | `target` (2) | 2,480 | 0 | 0 | 0.0% | — | — | — | — | — |
| `spambase` | 4,601 | 57 | `target` (2) | 0 | 577 | 3 | 11.454% | — | — | — | — | — |
| `jm1` | 10,885 | 21 | `target` (2) | 25 | 2,730 | 88 | 23.6472% | — | — | — | — | — |
| `PhishingWebsites` | 11,055 | 30 | `target` (2) | 0 | 7,884 | 64 | 65.4184% | — | — | — | `Google_Index` | — |
| `default-of-credit-card-clients` | 30,000 | 23 | `target` (2) | 0 | 108 | 21 | 0.2933% | — | — | — | — | — |
| `magic-telescope` | 19,020 | 10 | `target` (2) | 0 | 230 | 0 | 0.9359% | — | — | — | — | — |
| `dry-bean-dataset` | 13,611 | 16 | `target` (7) | 0 | 136 | 0 | 0.8376% | — | — | — | — | — |
| `adult` | 48,842 | 14 | `target` (2) | 6,465 | 419 | 25 | 0.6941% | — | — | — | — | — |
| `bank-marketing` | 45,211 | 16 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `electricity` | 45,312 | 8 | `target` (2) | 0 | 0 | 0 | 0.0% | — | — | — | — | — |
| `aps_failure` | 76,000 | 170 | `target` (2) | 1,078,695 | 0 | 0 | 0.0% | — | — | — | — | — |
| `covertype` | 100,000 | 54 | `target` (7) | 0 | 28,738 | 1,405 | 23.789% | — | — | — | — | — |
| `airlines` | 100,000 | 7 | `target` (2) | 0 | 29,937 | 6,116 | 24.606% | — | — | — | — | — |
| `kddcup99` | 100,000 | 41 | `target` (21) | 0 | 68,032 | 0 | 67.007% | — | — | — | — | — |

## Per-dataset schema

### haberman

Source: OpenML, id `43`; target `target` from `Survival_status`; 306 saved rows, 3 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`Age_of_patient_at_time_of_operation`: int64, 0.0%, 49 unique; `Patients_year_of_operation`: int64, 0.0%, 12 unique; `Number_of_positive_axillary_nodes_detected`: int64, 0.0%, 31 unique

Target frequency sample: `1`=225, `2`=81.

Repeated rows: 17 including target and 23 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### sonar

Source: OpenML, id `40`; target `target` from `Class`; 208 saved rows, 60 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`attribute_1`: float64, 0.0%, 177 unique; `attribute_2`: float64, 0.0%, 182 unique; `attribute_3`: float64, 0.0%, 190 unique; `attribute_4`: float64, 0.0%, 181 unique; `attribute_5`: float64, 0.0%, 193 unique; `attribute_6`: float64, 0.0%, 196 unique; `attribute_7`: float64, 0.0%, 195 unique; `attribute_8`: float64, 0.0%, 201 unique; `attribute_9`: float64, 0.0%, 205 unique; `attribute_10`: float64, 0.0%, 207 unique; `attribute_11`: float64, 0.0%, 203 unique; `attribute_12`: float64, 0.0%, 206 unique; `attribute_13`: float64, 0.0%, 198 unique; `attribute_14`: float64, 0.0%, 202 unique; `attribute_15`: float64, 0.0%, 203 unique; `attribute_16`: float64, 0.0%, 203 unique; `attribute_17`: float64, 0.0%, 202 unique; `attribute_18`: float64, 0.0%, 204 unique; `attribute_19`: float64, 0.0%, 206 unique; `attribute_20`: float64, 0.0%, 203 unique; `attribute_21`: float64, 0.0%, 200 unique; `attribute_22`: float64, 0.0%, 203 unique; `attribute_23`: float64, 0.0%, 199 unique; `attribute_24`: float64, 0.0%, 201 unique; `attribute_25`: float64, 0.0%, 198 unique; `attribute_26`: float64, 0.0%, 194 unique; `attribute_27`: float64, 0.0%, 190 unique; `attribute_28`: float64, 0.0%, 194 unique; `attribute_29`: float64, 0.0%, 197 unique; `attribute_30`: float64, 0.0%, 202 unique; `attribute_31`: float64, 0.0%, 207 unique; `attribute_32`: float64, 0.0%, 205 unique; `attribute_33`: float64, 0.0%, 205 unique; `attribute_34`: float64, 0.0%, 206 unique; `attribute_35`: float64, 0.0%, 205 unique; `attribute_36`: float64, 0.0%, 205 unique; `attribute_37`: float64, 0.0%, 206 unique; `attribute_38`: float64, 0.0%, 206 unique; `attribute_39`: float64, 0.0%, 204 unique; `attribute_40`: float64, 0.0%, 206 unique; `attribute_41`: float64, 0.0%, 204 unique; `attribute_42`: float64, 0.0%, 208 unique; `attribute_43`: float64, 0.0%, 205 unique; `attribute_44`: float64, 0.0%, 196 unique; `attribute_45`: float64, 0.0%, 205 unique; `attribute_46`: float64, 0.0%, 199 unique; `attribute_47`: float64, 0.0%, 202 unique; `attribute_48`: float64, 0.0%, 204 unique; `attribute_49`: float64, 0.0%, 193 unique; `attribute_50`: float64, 0.0%, 154 unique; `attribute_51`: float64, 0.0%, 160 unique; `attribute_52`: float64, 0.0%, 144 unique; `attribute_53`: float64, 0.0%, 134 unique; `attribute_54`: float64, 0.0%, 134 unique; `attribute_55`: float64, 0.0%, 129 unique; `attribute_56`: float64, 0.0%, 122 unique; `attribute_57`: float64, 0.0%, 121 unique; `attribute_58`: float64, 0.0%, 124 unique; `attribute_59`: float64, 0.0%, 119 unique; `attribute_60`: float64, 0.0%, 109 unique

Target frequency sample: `Mine`=111, `Rock`=97.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### ionosphere

Source: OpenML, id `59`; target `target` from `class`; 351 saved rows, 34 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`a01`: int64, 0.0%, 2 unique; `a02`: int64, 0.0%, 1 unique; `a03`: float64, 0.0%, 219 unique; `a04`: float64, 0.0%, 269 unique; `a05`: float64, 0.0%, 204 unique; `a06`: float64, 0.0%, 259 unique; `a07`: float64, 0.0%, 231 unique; `a08`: float64, 0.0%, 260 unique; `a09`: float64, 0.0%, 244 unique; `a10`: float64, 0.0%, 267 unique; `a11`: float64, 0.0%, 246 unique; `a12`: float64, 0.0%, 269 unique; `a13`: float64, 0.0%, 238 unique; `a14`: float64, 0.0%, 266 unique; `a15`: float64, 0.0%, 234 unique; `a16`: float64, 0.0%, 270 unique; `a17`: float64, 0.0%, 254 unique; `a18`: float64, 0.0%, 280 unique; `a19`: float64, 0.0%, 254 unique; `a20`: float64, 0.0%, 266 unique; `a21`: float64, 0.0%, 248 unique; `a22`: float64, 0.0%, 265 unique; `a23`: float64, 0.0%, 248 unique; `a24`: float64, 0.0%, 264 unique; `a25`: float64, 0.0%, 256 unique; `a26`: float64, 0.0%, 273 unique; `a27`: float64, 0.0%, 256 unique; `a28`: float64, 0.0%, 281 unique; `a29`: float64, 0.0%, 244 unique; `a30`: float64, 0.0%, 266 unique; `a31`: float64, 0.0%, 243 unique; `a32`: float64, 0.0%, 263 unique; `a33`: float64, 0.0%, 245 unique; `a34`: float64, 0.0%, 263 unique

Target frequency sample: `g`=225, `b`=126.

Repeated rows: 1 including target and 1 by features alone; constant features: `a02`; duplicate feature groups: —; all-missing features: —.

### heart-disease

Source: OpenML, id `43398`; target `target` from `target`; 303 saved rows, 13 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`age`: float64, 0.0%, 41 unique; `sex`: float64, 0.0%, 2 unique; `cp`: float64, 0.0%, 4 unique; `trestbps`: float64, 0.0%, 49 unique; `chol`: float64, 0.0%, 152 unique; `fbs`: float64, 0.0%, 2 unique; `restecg`: float64, 0.0%, 3 unique; `thalach`: float64, 0.0%, 91 unique; `exang`: float64, 0.0%, 2 unique; `oldpeak`: float64, 0.0%, 40 unique; `slope`: float64, 0.0%, 3 unique; `ca`: float64, 0.0%, 5 unique; `thal`: float64, 0.0%, 4 unique

Target frequency sample: `1.0`=165, `0.0`=138.

Repeated rows: 1 including target and 1 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### breast-cancer-wisconsin

Source: OpenML, id `1510`; target `target` from `Class`; 569 saved rows, 30 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`V1`: float64, 0.0%, 456 unique; `V2`: float64, 0.0%, 479 unique; `V3`: float64, 0.0%, 522 unique; `V4`: float64, 0.0%, 539 unique; `V5`: float64, 0.0%, 474 unique; `V6`: float64, 0.0%, 537 unique; `V7`: float64, 0.0%, 537 unique; `V8`: float64, 0.0%, 542 unique; `V9`: float64, 0.0%, 432 unique; `V10`: float64, 0.0%, 499 unique; `V11`: float64, 0.0%, 540 unique; `V12`: float64, 0.0%, 519 unique; `V13`: float64, 0.0%, 533 unique; `V14`: float64, 0.0%, 528 unique; `V15`: float64, 0.0%, 547 unique; `V16`: float64, 0.0%, 541 unique; `V17`: float64, 0.0%, 533 unique; `V18`: float64, 0.0%, 507 unique; `V19`: float64, 0.0%, 498 unique; `V20`: float64, 0.0%, 545 unique; `V21`: float64, 0.0%, 457 unique; `V22`: float64, 0.0%, 511 unique; `V23`: float64, 0.0%, 514 unique; `V24`: float64, 0.0%, 544 unique; `V25`: float64, 0.0%, 411 unique; `V26`: float64, 0.0%, 529 unique; `V27`: float64, 0.0%, 539 unique; `V28`: float64, 0.0%, 492 unique; `V29`: float64, 0.0%, 500 unique; `V30`: float64, 0.0%, 535 unique

Target frequency sample: `1`=357, `2`=212.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### blood-transfusion-service-center

Source: OpenML, id `1464`; target `target` from `Class`; 748 saved rows, 4 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`V1`: int64, 0.0%, 31 unique; `V2`: int64, 0.0%, 33 unique; `V3`: int64, 0.0%, 33 unique; `V4`: int64, 0.0%, 78 unique

Target frequency sample: `1`=570, `2`=178.

Repeated rows: 215 including target and 246 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### diabetes

Source: OpenML, id `37`; target `target` from `class`; 768 saved rows, 8 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`preg`: int64, 0.0%, 17 unique; `plas`: int64, 0.0%, 136 unique; `pres`: int64, 0.0%, 47 unique; `skin`: int64, 0.0%, 51 unique; `insu`: int64, 0.0%, 186 unique; `mass`: float64, 0.0%, 248 unique; `pedi`: float64, 0.0%, 517 unique; `age`: int64, 0.0%, 52 unique

Target frequency sample: `tested_negative`=500, `tested_positive`=268.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### titanic

Source: OpenML, id `40945`; target `target` from `survived`; 1,309 saved rows, 13 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`pclass`: int64, 0.0%, 3 unique; `name`: object, 0.0%, 1307 unique; `sex`: object, 0.0%, 2 unique; `age`: float64, 20.091673%, 99 unique; `sibsp`: int64, 0.0%, 7 unique; `parch`: int64, 0.0%, 8 unique; `ticket`: object, 0.0%, 929 unique; `fare`: float64, 0.076394%, 282 unique; `cabin`: object, 77.463713%, 187 unique; `embarked`: object, 0.152788%, 4 unique; `boat`: object, 62.872422%, 28 unique; `body`: float64, 90.756303%, 122 unique; `home.dest`: object, 43.086325%, 370 unique

Missing-value fields: `age` (263, 20.0917%), `fare` (1, 0.0764%), `cabin` (1014, 77.4637%), `embarked` (2, 0.1528%), `boat` (823, 62.8724%), `body` (1188, 90.7563%), `home.dest` (564, 43.0863%).

Target frequency sample: `0`=809, `1`=500.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### credit-g

Source: OpenML, id `31`; target `target` from `class`; 1,000 saved rows, 20 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`checking_status`: object, 0.0%, 4 unique; `duration`: int64, 0.0%, 33 unique; `credit_history`: object, 0.0%, 5 unique; `purpose`: object, 0.0%, 10 unique; `credit_amount`: int64, 0.0%, 921 unique; `savings_status`: object, 0.0%, 5 unique; `employment`: object, 0.0%, 5 unique; `installment_commitment`: int64, 0.0%, 4 unique; `personal_status`: object, 0.0%, 4 unique; `other_parties`: object, 0.0%, 3 unique; `residence_since`: int64, 0.0%, 4 unique; `property_magnitude`: object, 0.0%, 4 unique; `age`: int64, 0.0%, 53 unique; `other_payment_plans`: object, 0.0%, 3 unique; `housing`: object, 0.0%, 3 unique; `existing_credits`: int64, 0.0%, 4 unique; `job`: object, 0.0%, 4 unique; `num_dependents`: int64, 0.0%, 2 unique; `own_telephone`: object, 0.0%, 2 unique; `foreign_worker`: object, 0.0%, 2 unique

Target frequency sample: `good`=700, `bad`=300.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### wine-quality-red

Source: OpenML, id `40691`; target `target` from `class`; 1,599 saved rows, 11 features, 6 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`fixed_acidity`: float64, 0.0%, 96 unique; `volatile_acidity`: float64, 0.0%, 143 unique; `citric_acid`: float64, 0.0%, 80 unique; `residual_sugar`: float64, 0.0%, 91 unique; `chlorides`: float64, 0.0%, 153 unique; `free_sulfur_dioxide`: float64, 0.0%, 60 unique; `total_sulfur_dioxide`: float64, 0.0%, 144 unique; `density`: float64, 0.0%, 436 unique; `pH`: float64, 0.0%, 89 unique; `sulphates`: float64, 0.0%, 96 unique; `alcohol`: float64, 0.0%, 65 unique

Target frequency sample: `5`=681, `6`=638, `7`=199, `4`=53, `8`=18, `3`=10.

Repeated rows: 240 including target and 240 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### kr-vs-kp

Source: OpenML, id `3`; target `target` from `class`; 3,196 saved rows, 36 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`bkblk`: object, 0.0%, 2 unique; `bknwy`: object, 0.0%, 2 unique; `bkon8`: object, 0.0%, 2 unique; `bkona`: object, 0.0%, 2 unique; `bkspr`: object, 0.0%, 2 unique; `bkxbq`: object, 0.0%, 2 unique; `bkxcr`: object, 0.0%, 2 unique; `bkxwp`: object, 0.0%, 2 unique; `blxwp`: object, 0.0%, 2 unique; `bxqsq`: object, 0.0%, 2 unique; `cntxt`: object, 0.0%, 2 unique; `dsopp`: object, 0.0%, 2 unique; `dwipd`: object, 0.0%, 2 unique; `hdchk`: object, 0.0%, 2 unique; `katri`: object, 0.0%, 3 unique; `mulch`: object, 0.0%, 2 unique; `qxmsq`: object, 0.0%, 2 unique; `r2ar8`: object, 0.0%, 2 unique; `reskd`: object, 0.0%, 2 unique; `reskr`: object, 0.0%, 2 unique; `rimmx`: object, 0.0%, 2 unique; `rkxwp`: object, 0.0%, 2 unique; `rxmsq`: object, 0.0%, 2 unique; `simpl`: object, 0.0%, 2 unique; `skach`: object, 0.0%, 2 unique; `skewr`: object, 0.0%, 2 unique; `skrxp`: object, 0.0%, 2 unique; `spcop`: object, 0.0%, 2 unique; `stlmt`: object, 0.0%, 2 unique; `thrsk`: object, 0.0%, 2 unique; `wkcti`: object, 0.0%, 2 unique; `wkna8`: object, 0.0%, 2 unique; `wknck`: object, 0.0%, 2 unique; `wkovl`: object, 0.0%, 2 unique; `wkpos`: object, 0.0%, 2 unique; `wtoeg`: object, 0.0%, 2 unique

Target frequency sample: `won`=1669, `nowin`=1527.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### mushroom

Source: OpenML, id `24`; target `target` from `class`; 8,124 saved rows, 22 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`cap-shape`: object, 0.0%, 6 unique; `cap-surface`: object, 0.0%, 4 unique; `cap-color`: object, 0.0%, 10 unique; `bruises%3F`: object, 0.0%, 2 unique; `odor`: object, 0.0%, 9 unique; `gill-attachment`: object, 0.0%, 2 unique; `gill-spacing`: object, 0.0%, 2 unique; `gill-size`: object, 0.0%, 2 unique; `gill-color`: object, 0.0%, 12 unique; `stalk-shape`: object, 0.0%, 2 unique; `stalk-root`: object, 30.526834%, 5 unique; `stalk-surface-above-ring`: object, 0.0%, 4 unique; `stalk-surface-below-ring`: object, 0.0%, 4 unique; `stalk-color-above-ring`: object, 0.0%, 9 unique; `stalk-color-below-ring`: object, 0.0%, 9 unique; `veil-type`: object, 0.0%, 1 unique; `veil-color`: object, 0.0%, 4 unique; `ring-number`: object, 0.0%, 3 unique; `ring-type`: object, 0.0%, 5 unique; `spore-print-color`: object, 0.0%, 9 unique; `population`: object, 0.0%, 6 unique; `habitat`: object, 0.0%, 7 unique

Missing-value fields: `stalk-root` (2480, 30.5268%).

Target frequency sample: `e`=4208, `p`=3916.

Repeated rows: 0 including target and 0 by features alone; constant features: `veil-type`; duplicate feature groups: —; all-missing features: —.

### spambase

Source: OpenML, id `44`; target `target` from `class`; 4,601 saved rows, 57 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`word_freq_make`: float64, 0.0%, 142 unique; `word_freq_address`: float64, 0.0%, 171 unique; `word_freq_all`: float64, 0.0%, 214 unique; `word_freq_3d`: float64, 0.0%, 43 unique; `word_freq_our`: float64, 0.0%, 255 unique; `word_freq_over`: float64, 0.0%, 141 unique; `word_freq_remove`: float64, 0.0%, 173 unique; `word_freq_internet`: float64, 0.0%, 170 unique; `word_freq_order`: float64, 0.0%, 144 unique; `word_freq_mail`: float64, 0.0%, 245 unique; `word_freq_receive`: float64, 0.0%, 113 unique; `word_freq_will`: float64, 0.0%, 316 unique; `word_freq_people`: float64, 0.0%, 158 unique; `word_freq_report`: float64, 0.0%, 133 unique; `word_freq_addresses`: float64, 0.0%, 118 unique; `word_freq_free`: float64, 0.0%, 253 unique; `word_freq_business`: float64, 0.0%, 197 unique; `word_freq_email`: float64, 0.0%, 229 unique; `word_freq_you`: float64, 0.0%, 575 unique; `word_freq_credit`: float64, 0.0%, 148 unique; `word_freq_your`: float64, 0.0%, 401 unique; `word_freq_font`: float64, 0.0%, 99 unique; `word_freq_000`: float64, 0.0%, 164 unique; `word_freq_money`: float64, 0.0%, 143 unique; `word_freq_hp`: float64, 0.0%, 395 unique; `word_freq_hpl`: float64, 0.0%, 281 unique; `word_freq_george`: float64, 0.0%, 240 unique; `word_freq_650`: float64, 0.0%, 200 unique; `word_freq_lab`: float64, 0.0%, 156 unique; `word_freq_labs`: float64, 0.0%, 179 unique; `word_freq_telnet`: float64, 0.0%, 128 unique; `word_freq_857`: float64, 0.0%, 106 unique; `word_freq_data`: float64, 0.0%, 184 unique; `word_freq_415`: float64, 0.0%, 110 unique; `word_freq_85`: float64, 0.0%, 177 unique; `word_freq_technology`: float64, 0.0%, 159 unique; `word_freq_1999`: float64, 0.0%, 188 unique; `word_freq_parts`: float64, 0.0%, 53 unique; `word_freq_pm`: float64, 0.0%, 163 unique; `word_freq_direct`: float64, 0.0%, 125 unique; `word_freq_cs`: float64, 0.0%, 108 unique; `word_freq_meeting`: float64, 0.0%, 186 unique; `word_freq_original`: float64, 0.0%, 136 unique; `word_freq_project`: float64, 0.0%, 160 unique; `word_freq_re`: float64, 0.0%, 230 unique; `word_freq_edu`: float64, 0.0%, 227 unique; `word_freq_table`: float64, 0.0%, 38 unique; `word_freq_conference`: float64, 0.0%, 106 unique; `char_freq_%3B`: float64, 0.0%, 313 unique; `char_freq_%28`: float64, 0.0%, 641 unique; `char_freq_%5B`: float64, 0.0%, 225 unique; `char_freq_%21`: float64, 0.0%, 964 unique; `char_freq_%24`: float64, 0.0%, 504 unique; `char_freq_%23`: float64, 0.0%, 316 unique; `capital_run_length_average`: float64, 0.0%, 2161 unique; `capital_run_length_longest`: int64, 0.0%, 271 unique; `capital_run_length_total`: int64, 0.0%, 919 unique

Target frequency sample: `0`=2788, `1`=1813.

Repeated rows: 391 including target and 394 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### jm1

Source: OpenML, id `1053`; target `target` from `defects`; 10,885 saved rows, 21 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`loc`: float64, 0.0%, 365 unique; `v(g)`: float64, 0.0%, 108 unique; `ev(g)`: float64, 0.0%, 74 unique; `iv(g)`: float64, 0.0%, 82 unique; `n`: float64, 0.0%, 806 unique; `v`: float64, 0.0%, 3991 unique; `l`: float64, 0.0%, 55 unique; `d`: float64, 0.0%, 2695 unique; `i`: float64, 0.0%, 4268 unique; `e`: float64, 0.0%, 6978 unique; `b`: float64, 0.0%, 310 unique; `t`: float64, 0.0%, 6761 unique; `lOCode`: int64, 0.0%, 291 unique; `lOComment`: int64, 0.0%, 88 unique; `lOBlank`: int64, 0.0%, 95 unique; `locCodeAndComment`: int64, 0.0%, 30 unique; `uniq_Op`: float64, 0.045935%, 69 unique; `uniq_Opnd`: float64, 0.045935%, 172 unique; `total_Op`: float64, 0.045935%, 582 unique; `total_Opnd`: float64, 0.045935%, 469 unique; `branchCount`: float64, 0.045935%, 147 unique

Missing-value fields: `uniq_Op` (5, 0.0459%), `uniq_Opnd` (5, 0.0459%), `total_Op` (5, 0.0459%), `total_Opnd` (5, 0.0459%), `branchCount` (5, 0.0459%).

Target frequency sample: `False`=8779, `True`=2106.

Repeated rows: 1,973 including target and 2,061 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### PhishingWebsites

Source: OpenML, id `4534`; target `target` from `Result`; 11,055 saved rows, 30 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`having_IP_Address`: int64, 0.0%, 2 unique; `URL_Length`: int64, 0.0%, 3 unique; `Shortining_Service`: int64, 0.0%, 2 unique; `having_At_Symbol`: int64, 0.0%, 2 unique; `double_slash_redirecting`: int64, 0.0%, 2 unique; `Prefix_Suffix`: int64, 0.0%, 2 unique; `having_Sub_Domain`: int64, 0.0%, 3 unique; `SSLfinal_State`: int64, 0.0%, 3 unique; `Domain_registeration_length`: int64, 0.0%, 2 unique; `Favicon`: int64, 0.0%, 2 unique; `port`: int64, 0.0%, 2 unique; `HTTPS_token`: int64, 0.0%, 2 unique; `Request_URL`: int64, 0.0%, 2 unique; `URL_of_Anchor`: int64, 0.0%, 3 unique; `Links_in_tags`: int64, 0.0%, 3 unique; `SFH`: int64, 0.0%, 3 unique; `Submitting_to_email`: int64, 0.0%, 2 unique; `Abnormal_URL`: int64, 0.0%, 2 unique; `Redirect`: int64, 0.0%, 2 unique; `on_mouseover`: int64, 0.0%, 2 unique; `RightClick`: int64, 0.0%, 2 unique; `popUpWidnow`: int64, 0.0%, 2 unique; `Iframe`: int64, 0.0%, 2 unique; `age_of_domain`: int64, 0.0%, 2 unique; `DNSRecord`: int64, 0.0%, 2 unique; `web_traffic`: int64, 0.0%, 3 unique; `Page_Rank`: int64, 0.0%, 2 unique; `Google_Index`: int64, 0.0%, 2 unique; `Links_pointing_to_page`: int64, 0.0%, 3 unique; `Statistical_report`: int64, 0.0%, 2 unique

Target frequency sample: `1`=6157, `-1`=4898.

Repeated rows: 5,206 including target and 5,270 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### default-of-credit-card-clients

Source: OpenML, id `42477`; target `target` from `y`; 30,000 saved rows, 23 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`x1`: int64, 0.0%, 81 unique; `x2`: int64, 0.0%, 2 unique; `x3`: int64, 0.0%, 7 unique; `x4`: int64, 0.0%, 4 unique; `x5`: int64, 0.0%, 56 unique; `x6`: int64, 0.0%, 11 unique; `x7`: int64, 0.0%, 11 unique; `x8`: int64, 0.0%, 11 unique; `x9`: int64, 0.0%, 11 unique; `x10`: int64, 0.0%, 10 unique; `x11`: int64, 0.0%, 10 unique; `x12`: int64, 0.0%, 22723 unique; `x13`: int64, 0.0%, 22346 unique; `x14`: int64, 0.0%, 22026 unique; `x15`: int64, 0.0%, 21548 unique; `x16`: int64, 0.0%, 21010 unique; `x17`: int64, 0.0%, 20604 unique; `x18`: int64, 0.0%, 7943 unique; `x19`: int64, 0.0%, 7899 unique; `x20`: int64, 0.0%, 7518 unique; `x21`: int64, 0.0%, 6937 unique; `x22`: int64, 0.0%, 6897 unique; `x23`: int64, 0.0%, 6939 unique

Target frequency sample: `0`=23364, `1`=6636.

Repeated rows: 35 including target and 56 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### magic-telescope

Source: OpenML, id `1120`; target `target` from `class:`; 19,020 saved rows, 10 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`fLength:`: float64, 0.0%, 18643 unique; `fWidth:`: float64, 0.0%, 18200 unique; `fSize:`: float64, 0.0%, 7228 unique; `fConc:`: float64, 0.0%, 6410 unique; `fConc1:`: float64, 0.0%, 4421 unique; `fAsym:`: float64, 0.0%, 18704 unique; `fM3Long:`: float64, 0.0%, 18693 unique; `fM3Trans:`: float64, 0.0%, 18390 unique; `fAlpha:`: float64, 0.0%, 17981 unique; `fDist:`: float64, 0.0%, 18437 unique

Target frequency sample: `g`=12332, `h`=6688.

Repeated rows: 115 including target and 115 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### dry-bean-dataset

Source: UCI, id `602`; target `target` from `Class`; 13,611 saved rows, 16 features, 7 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`Area`: int64, 0.0%, 12011 unique; `Perimeter`: float64, 0.0%, 13351 unique; `MajorAxisLength`: float64, 0.0%, 13543 unique; `MinorAxisLength`: float64, 0.0%, 13543 unique; `AspectRatio`: float64, 0.0%, 13543 unique; `Eccentricity`: float64, 0.0%, 13543 unique; `ConvexArea`: int64, 0.0%, 12066 unique; `EquivDiameter`: float64, 0.0%, 12011 unique; `Extent`: float64, 0.0%, 13535 unique; `Solidity`: float64, 0.0%, 13522 unique; `Roundness`: float64, 0.0%, 13540 unique; `Compactness`: float64, 0.0%, 13543 unique; `ShapeFactor1`: float64, 0.0%, 13521 unique; `ShapeFactor2`: float64, 0.0%, 13506 unique; `ShapeFactor3`: float64, 0.0%, 13543 unique; `ShapeFactor4`: float64, 0.0%, 13532 unique

Target frequency sample: `DERMASON`=3546, `SIRA`=2636, `SEKER`=2027, `HOROZ`=1928, `CALI`=1630, `BARBUNYA`=1322, `BOMBAY`=522.

Repeated rows: 68 including target and 68 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### adult

Source: OpenML, id `179`; target `target` from `class`; 48,842 saved rows, 14 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`age`: int64, 0.0%, 5 unique; `workclass`: object, 5.730724%, 9 unique; `fnlwgt`: int64, 0.0%, 28523 unique; `education`: object, 0.0%, 16 unique; `education-num`: int64, 0.0%, 16 unique; `marital-status`: object, 0.0%, 7 unique; `occupation`: object, 5.751198%, 15 unique; `relationship`: object, 0.0%, 6 unique; `race`: object, 0.0%, 5 unique; `sex`: object, 0.0%, 2 unique; `capitalgain`: int64, 0.0%, 5 unique; `capitalloss`: int64, 0.0%, 5 unique; `hoursperweek`: int64, 0.0%, 5 unique; `native-country`: object, 1.754637%, 42 unique

Missing-value fields: `workclass` (2799, 5.7307%), `occupation` (2809, 5.7512%), `native-country` (857, 1.7546%).

Target frequency sample: `<=50K`=37155, `>50K`=11687.

Repeated rows: 187 including target and 212 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### bank-marketing

Source: OpenML, id `1461`; target `target` from `Class`; 45,211 saved rows, 16 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`V1`: int64, 0.0%, 77 unique; `V2`: object, 0.0%, 12 unique; `V3`: object, 0.0%, 3 unique; `V4`: object, 0.0%, 4 unique; `V5`: object, 0.0%, 2 unique; `V6`: int64, 0.0%, 7168 unique; `V7`: object, 0.0%, 2 unique; `V8`: object, 0.0%, 2 unique; `V9`: object, 0.0%, 3 unique; `V10`: int64, 0.0%, 31 unique; `V11`: object, 0.0%, 12 unique; `V12`: int64, 0.0%, 1573 unique; `V13`: int64, 0.0%, 48 unique; `V14`: int64, 0.0%, 559 unique; `V15`: int64, 0.0%, 41 unique; `V16`: object, 0.0%, 4 unique

Target frequency sample: `1`=39922, `2`=5289.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### electricity

Source: OpenML, id `151`; target `target` from `class`; 45,312 saved rows, 8 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`date`: float64, 0.0%, 933 unique; `day`: int64, 0.0%, 7 unique; `period`: float64, 0.0%, 48 unique; `nswprice`: float64, 0.0%, 4089 unique; `nswdemand`: float64, 0.0%, 5266 unique; `vicprice`: float64, 0.0%, 3798 unique; `vicdemand`: float64, 0.0%, 2846 unique; `transfer`: float64, 0.0%, 1878 unique

Target frequency sample: `DOWN`=26075, `UP`=19237.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### aps_failure

Source: OpenML, id `41138`; target `target` from `class`; 76,000 saved rows, 170 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`aa_000`: int64, 0.0%, 25211 unique; `ab_000`: float64, 77.226316%, 31 unique; `ac_000`: float64, 5.606579%, 2230 unique; `ad_000`: float64, 24.792105%, 2028 unique; `ae_000`: float64, 4.197368%, 356 unique; `af_000`: float64, 4.197368%, 459 unique; `ag_000`: float64, 1.131579%, 199 unique; `ag_001`: float64, 1.131579%, 796 unique; `ag_002`: float64, 1.131579%, 2953 unique; `ag_003`: float64, 1.131579%, 9384 unique; `ag_004`: float64, 1.131579%, 26855 unique; `ag_005`: float64, 1.131579%, 48328 unique; `ag_006`: float64, 1.131579%, 48481 unique; `ag_007`: float64, 1.131579%, 38290 unique; `ag_008`: float64, 1.131579%, 21564 unique; `ag_009`: float64, 1.131579%, 5924 unique; `ah_000`: float64, 1.078947%, 52009 unique; `ai_000`: float64, 1.042105%, 4805 unique; `aj_000`: float64, 1.042105%, 1032 unique; `ak_000`: float64, 7.365789%, 195 unique; `al_000`: float64, 1.067105%, 10457 unique; `am_0`: float64, 1.042105%, 12215 unique; `an_000`: float64, 1.067105%, 57988 unique; `ao_000`: float64, 0.988158%, 57387 unique; `ap_000`: float64, 1.067105%, 53132 unique; `aq_000`: float64, 0.988158%, 41141 unique; `ar_000`: float64, 4.588158%, 73 unique; `as_000`: float64, 1.042105%, 31 unique; `at_000`: float64, 1.042105%, 4026 unique; `au_000`: float64, 1.042105%, 72 unique; `av_000`: float64, 4.194737%, 4221 unique; `ax_000`: float64, 4.196053%, 2416 unique; `ay_000`: float64, 1.135526%, 576 unique; `ay_001`: float64, 1.135526%, 1111 unique; `ay_002`: float64, 1.135526%, 1195 unique; `ay_003`: float64, 1.135526%, 1250 unique; `ay_004`: float64, 1.135526%, 2127 unique; `ay_005`: float64, 1.135526%, 22881 unique; `ay_006`: float64, 1.135526%, 41539 unique; `ay_007`: float64, 1.135526%, 46023 unique; `ay_008`: float64, 1.135526%, 44764 unique; `ay_009`: float64, 1.135526%, 569 unique; `az_000`: float64, 1.135526%, 10291 unique; `az_001`: float64, 1.135526%, 8288 unique; `az_002`: float64, 1.135526%, 10237 unique; `az_003`: float64, 1.135526%, 24772 unique; `az_004`: float64, 1.135526%, 40464 unique; `az_005`: float64, 1.135526%, 53356 unique; `az_006`: float64, 1.135526%, 14148 unique; `az_007`: float64, 1.135526%, 4564 unique; `az_008`: float64, 1.135526%, 1454 unique; `az_009`: float64, 1.135526%, 387 unique; `ba_000`: float64, 1.159211%, 53547 unique; `ba_001`: float64, 1.159211%, 47872 unique; `ba_002`: float64, 1.159211%, 42463 unique; `ba_003`: float64, 1.159211%, 38701 unique; `ba_004`: float64, 1.159211%, 36077 unique; `ba_005`: float64, 1.159211%, 34938 unique; `ba_006`: float64, 1.159211%, 34659 unique; `ba_007`: float64, 1.159211%, 29877 unique; `ba_008`: float64, 1.159211%, 13926 unique; `ba_009`: float64, 1.159211%, 8089 unique; `bb_000`: float64, 1.078947%, 59706 unique; `bc_000`: float64, 4.590789%, 3123 unique; `bd_000`: float64, 4.593421%, 3945 unique; `be_000`: float64, 4.201316%, 4368 unique; `bf_000`: float64, 4.196053%, 1220 unique; `bg_000`: float64, 1.067105%, 52013 unique; `bh_000`: float64, 1.067105%, 29069 unique; `bi_000`: float64, 0.988158%, 49751 unique; `bj_000`: float64, 0.988158%, 45122 unique; `bk_000`: float64, 38.326316%, 14030 unique; `bl_000`: float64, 45.398684%, 13086 unique; `bm_000`: float64, 65.914474%, 10255 unique; `bn_000`: float64, 73.318421%, 8217 unique; `bo_000`: float64, 77.248684%, 6829 unique; `bp_000`: float64, 79.553947%, 5864 unique; `bq_000`: float64, 81.188158%, 5117 unique; `br_000`: float64, 82.096053%, 4608 unique; `bs_000`: float64, 1.221053%, 13715 unique; `bt_000`: float64, 0.256579%, 54524 unique; `bu_000`: float64, 1.159211%, 59654 unique; `bv_000`: float64, 1.159211%, 59650 unique; `bx_000`: float64, 5.425%, 66035 unique; `by_000`: float64, 0.763158%, 25857 unique; `bz_000`: float64, 4.586842%, 19261 unique; `ca_000`: float64, 7.318421%, 31827 unique; `cb_000`: float64, 1.221053%, 33990 unique; `cc_000`: float64, 5.421053%, 52486 unique; `cd_000`: float64, 1.132895%, 2 unique; `ce_000`: float64, 4.197368%, 25506 unique; `cf_000`: float64, 24.792105%, 584 unique; `cg_000`: float64, 24.792105%, 717 unique; `ch_000`: float64, 24.792105%, 3 unique; `ci_000`: float64, 0.557895%, 55004 unique; `cj_000`: float64, 0.557895%, 9092 unique; `ck_000`: float64, 0.557895%, 53628 unique; `cl_000`: float64, 15.805263%, 1090 unique; `cm_000`: float64, 16.388158%, 2347 unique; `cn_000`: float64, 1.159211%, 1874 unique; `cn_001`: float64, 1.159211%, 6457 unique; `cn_002`: float64, 1.159211%, 17135 unique; `cn_003`: float64, 1.159211%, 39884 unique; `cn_004`: float64, 1.159211%, 50097 unique; `cn_005`: float64, 1.159211%, 46038 unique; `cn_006`: float64, 1.159211%, 38609 unique; `cn_007`: float64, 1.159211%, 25026 unique; `cn_008`: float64, 1.159211%, 11140 unique; `cn_009`: float64, 1.159211%, 3431 unique; `co_000`: float64, 24.792105%, 2046 unique; `cp_000`: float64, 4.588158%, 2570 unique; `cq_000`: float64, 1.159211%, 59652 unique; `cr_000`: float64, 77.226316%, 87 unique; `cs_000`: float64, 1.128947%, 10173 unique; `cs_001`: float64, 1.128947%, 3715 unique; `cs_002`: float64, 1.128947%, 33135 unique; `cs_003`: float64, 1.128947%, 41632 unique; `cs_004`: float64, 1.128947%, 40804 unique; `cs_005`: float64, 1.128947%, 50809 unique; `cs_006`: float64, 1.128947%, 48558 unique; `cs_007`: float64, 1.128947%, 18940 unique; `cs_008`: float64, 1.128947%, 821 unique; `cs_009`: float64, 1.128947%, 65 unique; `ct_000`: float64, 23.060526%, 2825 unique; `cu_000`: float64, 23.060526%, 3819 unique; `cv_000`: float64, 23.060526%, 38942 unique; `cx_000`: float64, 23.060526%, 29453 unique; `cy_000`: float64, 23.060526%, 837 unique; `cz_000`: float64, 23.060526%, 12164 unique; `da_000`: float64, 23.060526%, 291 unique; `db_000`: float64, 23.060526%, 150 unique; `dc_000`: float64, 23.060526%, 39279 unique; `dd_000`: float64, 4.198684%, 7227 unique; `de_000`: float64, 4.589474%, 2088 unique; `df_000`: float64, 6.713158%, 467 unique; `dg_000`: float64, 6.713158%, 1621 unique; `dh_000`: float64, 6.713158%, 1196 unique; `di_000`: float64, 6.710526%, 6671 unique; `dj_000`: float64, 6.711842%, 81 unique; `dk_000`: float64, 6.711842%, 314 unique; `dl_000`: float64, 6.713158%, 218 unique; `dm_000`: float64, 6.714474%, 279 unique; `dn_000`: float64, 1.159211%, 23936 unique; `do_000`: float64, 4.589474%, 23341 unique; `dp_000`: float64, 4.592105%, 12600 unique; `dq_000`: float64, 4.592105%, 9588 unique; `dr_000`: float64, 4.592105%, 7879 unique; `ds_000`: float64, 4.593421%, 30564 unique; `dt_000`: float64, 4.593421%, 17597 unique; `du_000`: float64, 4.592105%, 33451 unique; `dv_000`: float64, 4.592105%, 35506 unique; `dx_000`: float64, 4.588158%, 18080 unique; `dy_000`: float64, 4.589474%, 7343 unique; `dz_000`: float64, 4.585526%, 52 unique; `ea_000`: float64, 4.585526%, 142 unique; `eb_000`: float64, 6.711842%, 33326 unique; `ec_00`: float64, 16.951316%, 36264 unique; `ed_000`: float64, 15.805263%, 4315 unique; `ee_000`: float64, 1.135526%, 49557 unique; `ee_001`: float64, 1.135526%, 45033 unique; `ee_002`: float64, 1.135526%, 40697 unique; `ee_003`: float64, 1.135526%, 37292 unique; `ee_004`: float64, 1.135526%, 41658 unique; `ee_005`: float64, 1.135526%, 43052 unique; `ee_006`: float64, 1.135526%, 37561 unique; `ee_007`: float64, 1.135526%, 35860 unique; `ee_008`: float64, 1.135526%, 28413 unique; `ee_009`: float64, 1.135526%, 10948 unique; `ef_000`: float64, 4.586842%, 31 unique; `eg_000`: float64, 4.585526%, 56 unique

Missing-value fields: `ab_000` (58692, 77.2263%), `ac_000` (4261, 5.6066%), `ad_000` (18842, 24.7921%), `ae_000` (3190, 4.1974%), `af_000` (3190, 4.1974%), `ag_000` (860, 1.1316%), `ag_001` (860, 1.1316%), `ag_002` (860, 1.1316%), `ag_003` (860, 1.1316%), `ag_004` (860, 1.1316%), `ag_005` (860, 1.1316%), `ag_006` (860, 1.1316%), `ag_007` (860, 1.1316%), `ag_008` (860, 1.1316%), `ag_009` (860, 1.1316%), `ah_000` (820, 1.0789%), `ai_000` (792, 1.0421%), `aj_000` (792, 1.0421%), `ak_000` (5598, 7.3658%), `al_000` (811, 1.0671%), `am_0` (792, 1.0421%), `an_000` (811, 1.0671%), `ao_000` (751, 0.9882%), `ap_000` (811, 1.0671%), `aq_000` (751, 0.9882%), `ar_000` (3487, 4.5882%), `as_000` (792, 1.0421%), `at_000` (792, 1.0421%), `au_000` (792, 1.0421%), `av_000` (3188, 4.1947%), `ax_000` (3189, 4.1961%), `ay_000` (863, 1.1355%), `ay_001` (863, 1.1355%), `ay_002` (863, 1.1355%), `ay_003` (863, 1.1355%), `ay_004` (863, 1.1355%), `ay_005` (863, 1.1355%), `ay_006` (863, 1.1355%), `ay_007` (863, 1.1355%), `ay_008` (863, 1.1355%), `ay_009` (863, 1.1355%), `az_000` (863, 1.1355%), `az_001` (863, 1.1355%), `az_002` (863, 1.1355%), `az_003` (863, 1.1355%), `az_004` (863, 1.1355%), `az_005` (863, 1.1355%), `az_006` (863, 1.1355%), `az_007` (863, 1.1355%), `az_008` (863, 1.1355%), `az_009` (863, 1.1355%), `ba_000` (881, 1.1592%), `ba_001` (881, 1.1592%), `ba_002` (881, 1.1592%), `ba_003` (881, 1.1592%), `ba_004` (881, 1.1592%), `ba_005` (881, 1.1592%), `ba_006` (881, 1.1592%), `ba_007` (881, 1.1592%), `ba_008` (881, 1.1592%), `ba_009` (881, 1.1592%), `bb_000` (820, 1.0789%), `bc_000` (3489, 4.5908%), `bd_000` (3491, 4.5934%), `be_000` (3193, 4.2013%), `bf_000` (3189, 4.1961%), `bg_000` (811, 1.0671%), `bh_000` (811, 1.0671%), `bi_000` (751, 0.9882%), `bj_000` (751, 0.9882%), `bk_000` (29128, 38.3263%), `bl_000` (34503, 45.3987%), `bm_000` (50095, 65.9145%), `bn_000` (55722, 73.3184%), `bo_000` (58709, 77.2487%), `bp_000` (60461, 79.5539%), `bq_000` (61703, 81.1882%), `br_000` (62393, 82.0961%), `bs_000` (928, 1.2211%), `bt_000` (195, 0.2566%), `bu_000` (881, 1.1592%), `bv_000` (881, 1.1592%), `bx_000` (4123, 5.425%), `by_000` (580, 0.7632%), `bz_000` (3486, 4.5868%), `ca_000` (5562, 7.3184%), `cb_000` (928, 1.2211%), `cc_000` (4120, 5.4211%), `cd_000` (861, 1.1329%), `ce_000` (3190, 4.1974%), `cf_000` (18842, 24.7921%), `cg_000` (18842, 24.7921%), `ch_000` (18842, 24.7921%), `ci_000` (424, 0.5579%), `cj_000` (424, 0.5579%), `ck_000` (424, 0.5579%), `cl_000` (12012, 15.8053%), `cm_000` (12455, 16.3882%), `cn_000` (881, 1.1592%), `cn_001` (881, 1.1592%), `cn_002` (881, 1.1592%), `cn_003` (881, 1.1592%), `cn_004` (881, 1.1592%), `cn_005` (881, 1.1592%), `cn_006` (881, 1.1592%), `cn_007` (881, 1.1592%), `cn_008` (881, 1.1592%), `cn_009` (881, 1.1592%), `co_000` (18842, 24.7921%), `cp_000` (3487, 4.5882%), `cq_000` (881, 1.1592%), `cr_000` (58692, 77.2263%), `cs_000` (858, 1.1289%), `cs_001` (858, 1.1289%), `cs_002` (858, 1.1289%), `cs_003` (858, 1.1289%), `cs_004` (858, 1.1289%), `cs_005` (858, 1.1289%), `cs_006` (858, 1.1289%), `cs_007` (858, 1.1289%), `cs_008` (858, 1.1289%), `cs_009` (858, 1.1289%), `ct_000` (17526, 23.0605%), `cu_000` (17526, 23.0605%), `cv_000` (17526, 23.0605%), `cx_000` (17526, 23.0605%), `cy_000` (17526, 23.0605%), `cz_000` (17526, 23.0605%), `da_000` (17526, 23.0605%), `db_000` (17526, 23.0605%), `dc_000` (17526, 23.0605%), `dd_000` (3191, 4.1987%), `de_000` (3488, 4.5895%), `df_000` (5102, 6.7132%), `dg_000` (5102, 6.7132%), `dh_000` (5102, 6.7132%), `di_000` (5100, 6.7105%), `dj_000` (5101, 6.7118%), `dk_000` (5101, 6.7118%), `dl_000` (5102, 6.7132%), `dm_000` (5103, 6.7145%), `dn_000` (881, 1.1592%), `do_000` (3488, 4.5895%), `dp_000` (3490, 4.5921%), `dq_000` (3490, 4.5921%), `dr_000` (3490, 4.5921%), `ds_000` (3491, 4.5934%), `dt_000` (3491, 4.5934%), `du_000` (3490, 4.5921%), `dv_000` (3490, 4.5921%), `dx_000` (3487, 4.5882%), `dy_000` (3488, 4.5895%), `dz_000` (3485, 4.5855%), `ea_000` (3485, 4.5855%), `eb_000` (5101, 6.7118%), `ec_00` (12883, 16.9513%), `ed_000` (12012, 15.8053%), `ee_000` (863, 1.1355%), `ee_001` (863, 1.1355%), `ee_002` (863, 1.1355%), `ee_003` (863, 1.1355%), `ee_004` (863, 1.1355%), `ee_005` (863, 1.1355%), `ee_006` (863, 1.1355%), `ee_007` (863, 1.1355%), `ee_008` (863, 1.1355%), `ee_009` (863, 1.1355%), `ef_000` (3486, 4.5868%), `eg_000` (3485, 4.5855%).

Target frequency sample: `neg`=74625, `pos`=1375.

Repeated rows: 0 including target and 0 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### covertype

Source: OpenML, id `180`; target `target` from `class`; 100,000 saved rows, 54 features, 7 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`elevation`: int64, 0.0%, 1757 unique; `aspect`: int64, 0.0%, 361 unique; `slope`: int64, 0.0%, 57 unique; `horizontal_distance_to_hydrology`: int64, 0.0%, 482 unique; `Vertical_Distance_To_Hydrology`: int64, 0.0%, 557 unique; `Horizontal_Distance_To_Roadways`: int64, 0.0%, 5316 unique; `Hillshade_9am`: int64, 0.0%, 194 unique; `Hillshade_Noon`: int64, 0.0%, 155 unique; `Hillshade_3pm`: int64, 0.0%, 253 unique; `Horizontal_Distance_To_Fire_Points`: int64, 0.0%, 5211 unique; `wilderness_area1`: int64, 0.0%, 2 unique; `wilderness_area2`: int64, 0.0%, 2 unique; `wilderness_area3`: int64, 0.0%, 2 unique; `wilderness_area4`: int64, 0.0%, 2 unique; `soil_type_1`: int64, 0.0%, 2 unique; `soil_type_2`: int64, 0.0%, 2 unique; `soil_type_3`: int64, 0.0%, 2 unique; `soil_type_4`: int64, 0.0%, 2 unique; `soil_type_5`: int64, 0.0%, 2 unique; `soil_type_6`: int64, 0.0%, 2 unique; `soil_type_7`: int64, 0.0%, 2 unique; `soil_type_8`: int64, 0.0%, 2 unique; `soil_type_9`: int64, 0.0%, 2 unique; `soil_type_10`: int64, 0.0%, 2 unique; `soil_type_11`: int64, 0.0%, 2 unique; `soil_type_12`: int64, 0.0%, 2 unique; `soil_type_13`: int64, 0.0%, 2 unique; `soil_type_14`: int64, 0.0%, 2 unique; `soil_type_15`: int64, 0.0%, 2 unique; `soil_type_16`: int64, 0.0%, 2 unique; `soil_type_17`: int64, 0.0%, 2 unique; `soil_type_18`: int64, 0.0%, 2 unique; `soil_type_19`: int64, 0.0%, 2 unique; `soil_type_20`: int64, 0.0%, 2 unique; `soil_type_21`: int64, 0.0%, 2 unique; `soil_type_22`: int64, 0.0%, 2 unique; `soil_type_23`: int64, 0.0%, 2 unique; `soil_type_24`: int64, 0.0%, 2 unique; `soil_type_25`: int64, 0.0%, 2 unique; `soil_type_26`: int64, 0.0%, 2 unique; `soil_type_27`: int64, 0.0%, 2 unique; `soil_type_28`: int64, 0.0%, 2 unique; `soil_type_29`: int64, 0.0%, 2 unique; `soil_type_30`: int64, 0.0%, 2 unique; `soil_type_31`: int64, 0.0%, 2 unique; `soil_type_32`: int64, 0.0%, 2 unique; `soil_type_33`: int64, 0.0%, 2 unique; `soil_type_34`: int64, 0.0%, 2 unique; `soil_type_35`: int64, 0.0%, 2 unique; `soil_type_36`: int64, 0.0%, 2 unique; `soil_type_37`: int64, 0.0%, 2 unique; `soil_type_38`: int64, 0.0%, 2 unique; `soil_type_39`: int64, 0.0%, 2 unique; `soil_type_40`: int64, 0.0%, 2 unique

Target frequency sample: `Lodgepole_Pine`=46772, `Spruce_Fir`=35284, `Ponderosa_Pine`=6681, `Krummholz`=4088, `Douglas_fir`=3605, `Aspen`=2371, `Cottonwood_Willow`=1199.

Repeated rows: 13,825 including target and 15,240 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### airlines

Source: OpenML, id `1169`; target `target` from `Delay`; 100,000 saved rows, 7 features, 2 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`Airline`: object, 0.0%, 18 unique; `Flight`: int64, 0.0%, 6519 unique; `AirportFrom`: object, 0.0%, 292 unique; `AirportTo`: object, 0.0%, 292 unique; `DayOfWeek`: int64, 0.0%, 7 unique; `Time`: int64, 0.0%, 1115 unique; `Length`: int64, 0.0%, 418 unique

Target frequency sample: `0`=55506, `1`=44494.

Repeated rows: 9,604 including target and 15,720 by features alone; constant features: —; duplicate feature groups: —; all-missing features: —.

### kddcup99

Source: OpenML, id `1113`; target `target` from `label`; 100,000 saved rows, 41 features, 21 target classes; CSV checksum verified: True; sidecar schema verified: True.

Columns (`name`: dtype, missing %, unique values):

`duration`: int64, 0.0%, 732 unique; `protocol_type`: object, 0.0%, 3 unique; `service`: object, 0.0%, 64 unique; `flag`: object, 0.0%, 10 unique; `src_bytes`: int64, 0.0%, 1751 unique; `dst_bytes`: int64, 0.0%, 4922 unique; `land`: int64, 0.0%, 2 unique; `wrong_fragment`: int64, 0.0%, 3 unique; `urgent`: int64, 0.0%, 1 unique; `hot`: int64, 0.0%, 16 unique; `num_failed_logins`: int64, 0.0%, 2 unique; `logged_in`: int64, 0.0%, 2 unique; `lnum_compromised`: int64, 0.0%, 8 unique; `lroot_shell`: int64, 0.0%, 2 unique; `lsu_attempted`: int64, 0.0%, 3 unique; `lnum_root`: int64, 0.0%, 10 unique; `lnum_file_creations`: int64, 0.0%, 7 unique; `lnum_shells`: int64, 0.0%, 3 unique; `lnum_access_files`: int64, 0.0%, 4 unique; `lnum_outbound_cmds`: int64, 0.0%, 1 unique; `is_host_login`: int64, 0.0%, 1 unique; `is_guest_login`: int64, 0.0%, 2 unique; `count`: int64, 0.0%, 416 unique; `srv_count`: int64, 0.0%, 346 unique; `serror_rate`: float64, 0.0%, 65 unique; `srv_serror_rate`: float64, 0.0%, 32 unique; `rerror_rate`: float64, 0.0%, 61 unique; `srv_rerror_rate`: float64, 0.0%, 34 unique; `same_srv_rate`: float64, 0.0%, 86 unique; `diff_srv_rate`: float64, 0.0%, 61 unique; `srv_diff_host_rate`: float64, 0.0%, 56 unique; `dst_host_count`: int64, 0.0%, 255 unique; `dst_host_srv_count`: int64, 0.0%, 255 unique; `dst_host_same_srv_rate`: float64, 0.0%, 101 unique; `dst_host_diff_srv_rate`: float64, 0.0%, 101 unique; `dst_host_same_src_port_rate`: float64, 0.0%, 101 unique; `dst_host_srv_diff_host_rate`: float64, 0.0%, 54 unique; `dst_host_serror_rate`: float64, 0.0%, 75 unique; `dst_host_srv_serror_rate`: float64, 0.0%, 39 unique; `dst_host_rerror_rate`: float64, 0.0%, 101 unique; `dst_host_srv_rerror_rate`: float64, 0.0%, 96 unique

Target frequency sample: `smurf`=56970, `neptune`=21680, `normal`=19588, `back`=437, `satan`=311, `ipsweep`=271, `portsweep`=226, `warezclient`=220, `teardrop`=175, `nmap`=46, `pod`=41, `buffer_overflow`=11.

Repeated rows: 64,242 including target and 64,242 by features alone; constant features: `urgent`, `lnum_outbound_cmds`, `is_host_login`; duplicate feature groups: [['urgent', 'lnum_outbound_cmds', 'is_host_login']]; all-missing features: —.
