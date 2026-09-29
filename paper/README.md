# Manuscript package

## Provenance status

The manuscript source, its numerical claims, and the PDFs in figures/ are preserved as historical original-run materials. They have not been updated from the corrected runner. The locally available original ledger is recorded by checksum in ../provenance/original_run.json; it must not be described as corrected evidence. A full corrected benchmark and dataset-level analyses are required before revising manuscript methods, results, or figure captions. The corrected code excludes transductive domain partitions from primary comparisons and writes new ledgers under corrected_runs/<run-id>/.

`manuscript.tex` is the formula-driven manuscript prepared for Knowledge-Based Systems. Its title introduces Feature-Synthesis Variance Amplification (FSVA) as the paper’s mechanistic contribution, and the theory section gives FSVA a compact defining equation. It uses the 22-dataset primary scope, contains a sub-250-word abstract, defines task-level and data-set-balanced estimands, derives error-propagation and bias--variance conditions, reports condition-wise results, and links only to the self-contained PDF figures in `figures/`. The workflow artwork is an internal evaluation-protocol diagram; coverage exceptions remain documented in the manuscript rather than being mixed into the method graphic.

Files:

- `manuscript.tex` — editable Elsevier `elsarticle` source.
- `references.bib` — bibliography for the methods and software cited in the manuscript.
- `highlights.txt` — five short submission highlights.
- `cover_letter.txt` — submission-ready cover letter naming Md. Shoaib Uddin Chanda as corresponding author.
- `figures/` — self-contained PDF copies of the historical workflow and Figures 2–10. They are preserved with the original manuscript and are not corrected-run outputs. The manuscript uses only these PDF assets and does not depend on paths outside `paper/`.

The source code and manuscript package are maintained at <https://github.com/mdshoaibuddinchanda/AutoFE-ShiftBench>. The local environment used for this audit does not contain `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`, so the source has not been compiled here. Compile from this directory with an Elsevier-compatible LaTeX installation, for example:

```text
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Generated result and report artifacts are intentionally excluded from version control; the local result and table archives are preserved separately from this paper package.
