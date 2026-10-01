# Manuscript update checklist for Reviewer #1

These are locations in the **historical** submission to revise after the corrected campaign. Line references are from the present working tree. This checklist does not overwrite historical numerical claims or submission figures before replacement evidence exists.

| File / location | Existing wording or claim to revisit | Required correction / evidence |
|---|---|---|
| `paper/manuscript.tex:59` (abstract) | Historical headline differences, dataset counts and FSVA interpretation | Replace with corrected scope, dataset denominators, uncertainty and measured mechanism limitations. All values pending. |
| `paper/manuscript.tex:122` (analysis scope) | 25 OpenML datasets; 22-dataset primary scope; 538,972 evaluations | State 24 OpenML plus UCI Dry Bean, distinguish historical ledger from the corrected 25-dataset intended scope and at-most-23 group-AUC comparison. Use machine-checked terminal/finite denominators. |
| `paper/manuscript.tex:146` (task design/splitters) | Fallback ordinary folds and globally computed partition geometry | Describe actual row/group split policies with no group fallback, all-seed feasibility rule, target-free transductive geometry, and separate primary/sensitivity estimands. |
| `paper/manuscript.tex:149` (pipelines) | Seven pipelines and capped Raw baselines | Replace with actual frozen 14-pipeline truth table, full Raw/cap-matched Raw, preserved joint removal and individual isolates/leave-outs. Report actual candidate budgets. |
| `paper/manuscript.tex:166` (analysis) | Available-record pooled summaries and promised future inference | Use complete-case dataset contrasts, prespecified F1/Holm families, bootstrap definition, common-dataset row/group comparison and explicit primary missing-outcome bounds. Winner selection remains exploratory. |
| `paper/manuscript.tex:331` (mechanism/results) | FSVA explanation without measured corrected association | Insert clean training-fold measured Jacobian/history evidence and dataset association with uncertainty. Do not describe association as causal proof. |
| `paper/manuscript.tex:562` and limitations/conclusion | Historical leakage, significance, operator and generalization assertions | Match corrected code/protocol and ledger evidence; keep exact-vector grouping limitations, no nested selection, training-corruption semantics and incomplete-outcome accounting explicit. |
| `paper/manuscript.tex:583` and data/code availability | All data described as OpenML; generated artifacts excluded from Git | Correct Dry Bean source and provide durable dataset/ledger/artifact archive identifiers/hashes when available. Git ignore status does not make research evidence temporary. |
| `paper/highlights.txt:1` | Historical scope and empirical superiority/harm summaries | Replace after corrected results; do not imply bounded diagnostic timings are scientific performance findings. |
| `paper/cover_letter.txt:7` | Historical benchmark totals/findings | Align with final corrected scope and completed evidence; submission is a later user action. |

Search the complete manuscript/package for `nested`, `leakage-free`, `zero leakage`, `OpenML`, `22`, `538,972`, `FSVA`, `Jacobian`, `NoMultiply`, and deployment/shift claims before submission. A zero-overlap assertion applies to exact raw-feature groups only. Training-corruption robustness is not deployment-time distribution-shift evidence. Historical 79.66 GiB candidate-history and 47,378 GiB retained-cache projections are planning records, not current required storage.

Status: **PENDING CORRECTED RUN AND MANUSCRIPT REVISION**. The engineering checklist and response draft are ready to receive verified results; the scientific reviewer response cannot yet be called complete.
