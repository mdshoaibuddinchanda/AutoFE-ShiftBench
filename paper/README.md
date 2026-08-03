# Manuscript package

`manuscript.tex` is an expanded, formula-driven draft prepared for a Knowledge-Based Systems submission. Its title introduces Feature-Synthesis Variance Amplification (FSVA) as the paper’s mechanistic contribution, and the theory section gives FSVA a compact defining equation. It uses the 22-dataset primary scope, contains a sub-250-word abstract, defines task-level and data-set-balanced estimands, derives error-propagation and bias--variance conditions, reports condition-wise results, and links only to the self-contained PDF figures in `figures/`. The workflow artwork is an internal evaluation-protocol diagram; coverage exceptions remain documented in the manuscript rather than being mixed into the method graphic.

Files:

- `manuscript.tex` — editable Elsevier `elsarticle` source.
- `references.bib` — bibliography for the methods and software cited in the draft.
- `highlights.txt` — five short submission highlights.
- `figures/` — self-contained PDF copies of the workflow plus regenerated Figures 2–10. The manuscript uses only these PDF assets and does not depend on paths outside `paper/`.

Before submission, complete the author affiliation/e-mail, funding statement, persistent data/code archive, environment lockfile, and any required institutional declarations. The local environment used for this audit does not contain `pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`, so the source has not been compiled here. Compile from this directory with an Elsevier-compatible LaTeX installation, for example:

```text
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

No existing result or report file was deleted or overwritten while preparing this package.
